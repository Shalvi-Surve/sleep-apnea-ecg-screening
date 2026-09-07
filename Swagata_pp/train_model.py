"""
Apnea-ECG model training & evaluation
======================================

Consumes the feature tables produced by `preprocess_apnea_ecg.py`
(train_features.csv / test_features.csv, one row per 1-minute epoch) and
trains/evaluates classifiers for per-minute apnea detection.

Design choices (and why)
-------------------------
- **Group-aware cross-validation**: epochs from the same recording are
  correlated (autocorrelated apnea episodes, shared patient physiology).
  Row-level k-fold CV would leak information across folds and inflate
  scores, so this uses GroupKFold on the `record` column for all model
  selection.
- **Class imbalance handling**: apnea is typically the minority class.
  Every model here uses `class_weight="balanced"` (or scale_pos_weight
  for XGBoost) rather than naive oversampling, so imbalance handling stays
  inside each CV fold instead of leaking synthetic points across folds.
- **Model comparison, not a single fixed choice**: Logistic Regression
  (baseline / interpretable), Random Forest, Gradient Boosting, and
  XGBoost (if installed) are all evaluated with the same CV protocol, and
  the best by mean CV ROC-AUC is refit on the full training set.
- **Two levels of evaluation**:
    1. Standard per-epoch classification metrics (accuracy, precision,
       recall, F1, ROC-AUC, confusion matrix) - the ML-standard view.
    2. Per-recording apnea-minute correlation - this mirrors how the
       original CinC 2000 challenge actually scored entries: an apnea
       diagnosis in the sleep lab cares about total apnea burden per
       night (correlates with the Apnea-Hypopnea Index) more than
       whether any single minute was labeled exactly right.

Usage
-----
    python train_model.py --train-csv ./processed/train_features.csv \\
                           --test-csv ./processed/test_features.csv \\
                           --out-dir ./model_output

    # Using the preprocessing script's self-test output
    python train_model.py --train-csv ./processed_selftest/train_features_selftest.csv \\
                           --test-csv ./processed_selftest/test_features_selftest.csv \\
                           --out-dir ./model_output_selftest
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupKFold, cross_val_predict

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s",
                     datefmt="%H:%M:%S")
log = logging.getLogger("apnea_ecg_model")

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    log.warning("xgboost not installed - skipping XGBoost from the model comparison")

NON_FEATURE_COLS = {"record", "epoch_idx", "label"}


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

def load_features(path: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    df = pd.read_csv(path)
    df = df.dropna(subset=["label"]).reset_index(drop=True)
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    X = df[feature_cols].to_numpy()
    y = df["label"].to_numpy().astype(int)
    groups = df["record"].to_numpy()
    log.info(f"Loaded {path}: {X.shape[0]} epochs x {X.shape[1]} features, "
             f"{groups.__len__() and len(set(groups))} unique records, "
             f"class balance {np.bincount(y) / len(y)}")
    return df, X, y, feature_cols


# --------------------------------------------------------------------------- #
# Model zoo
# --------------------------------------------------------------------------- #

def build_model_zoo(pos_weight: float) -> dict:
    zoo = {
        "logistic_regression": LogisticRegression(
            class_weight="balanced", max_iter=2000, C=1.0
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=5,
            class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42
        ),
    }
    if HAS_XGB:
        zoo["xgboost"] = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            scale_pos_weight=pos_weight, eval_metric="logloss",
            random_state=42, n_jobs=-1
        )
    return zoo


# --------------------------------------------------------------------------- #
# Group-aware cross-validation for model selection
# --------------------------------------------------------------------------- #

def cross_validate_group_aware(models: dict, X: np.ndarray, y: np.ndarray,
                                 groups: np.ndarray, n_splits: int = 4) -> pd.DataFrame:
    n_splits = min(n_splits, len(set(groups)))
    if n_splits < 2:
        raise ValueError(
            f"Need at least 2 distinct recordings for group-aware CV, got {len(set(groups))}."
        )
    gkf = GroupKFold(n_splits=n_splits)

    rows = []
    for model_name, model in models.items():
        fold_metrics = []
        for fold_i, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            if len(set(y_val)) < 2:
                continue  # skip degenerate folds with only one class in validation

            m = _clone_model(model)
            m.fit(X_tr, y_tr)
            y_pred = m.predict(X_val)
            y_proba = _predict_proba_safe(m, X_val)

            fold_metrics.append({
                "fold": fold_i,
                "accuracy": accuracy_score(y_val, y_pred),
                "precision": precision_score(y_val, y_pred, zero_division=0),
                "recall": recall_score(y_val, y_pred, zero_division=0),
                "f1": f1_score(y_val, y_pred, zero_division=0),
                "roc_auc": roc_auc_score(y_val, y_proba) if y_proba is not None else np.nan,
            })

        if not fold_metrics:
            log.warning(f"{model_name}: no valid folds (check group/class balance)")
            continue

        fold_df = pd.DataFrame(fold_metrics)
        summary = fold_df.mean(numeric_only=True).to_dict()
        summary["model"] = model_name
        summary["n_folds_used"] = len(fold_df)
        rows.append(summary)
        log.info(f"{model_name}: CV ROC-AUC={summary['roc_auc']:.3f}  "
                 f"F1={summary['f1']:.3f}  recall={summary['recall']:.3f}  "
                 f"precision={summary['precision']:.3f}  ({len(fold_df)} folds)")

    return pd.DataFrame(rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)


def _clone_model(model):
    from sklearn.base import clone
    return clone(model)


def _predict_proba_safe(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return None


# --------------------------------------------------------------------------- #
# Final fit + held-out test evaluation
# --------------------------------------------------------------------------- #

def find_optimal_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Youden's J statistic (maximizes sensitivity + specificity - 1). The
    default 0.5 cutoff is arbitrary under class imbalance; this picks the
    point on the ROC curve that best trades off the two error types instead
    of assuming the two classes are equally likely."""
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    j_scores = tpr - fpr
    return float(thresholds[np.argmax(j_scores)])


def evaluate_on_test(model, X_test: np.ndarray, y_test: np.ndarray, out_dir: Path,
                      model_name: str, threshold: float = 0.5) -> dict:
    y_proba = _predict_proba_safe(model, X_test)
    y_pred = (y_proba >= threshold).astype(int) if y_proba is not None else model.predict(X_test)

    metrics = {
        "threshold_used": threshold,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba) if y_proba is not None else None,
    }
    log.info(f"TEST SET ({model_name}, threshold={threshold:.3f}): " +
             ", ".join(f"{k}={v:.3f}" for k, v in metrics.items() if v is not None))
    log.info("\n" + classification_report(y_test, y_pred, target_names=["normal", "apnea"],
                                           zero_division=0))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred),
                            display_labels=["normal", "apnea"]).plot(ax=ax[0], colorbar=False)
    ax[0].set_title("Confusion matrix (test set)")

    if y_proba is not None:
        RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax[1])
        ax[1].set_title("ROC curve (test set)")
    else:
        ax[1].axis("off")

    fig.tight_layout()
    fig.savefig(out_dir / "test_evaluation.png", dpi=150)
    plt.close(fig)

    return metrics


def per_recording_apnea_burden(df_test: pd.DataFrame, y_pred: np.ndarray, out_dir: Path) -> dict:
    """Aggregates per-minute predictions into per-recording apnea-minute
    counts and correlates with ground truth - this is closer to how apnea
    severity (AHI-like burden) is actually used clinically than per-minute
    accuracy is."""
    df = df_test.copy()
    df["pred"] = y_pred
    agg = df.groupby("record").agg(true_apnea_minutes=("label", "sum"),
                                    pred_apnea_minutes=("pred", "sum"),
                                    total_minutes=("label", "size")).reset_index()
    if len(agg) < 2:
        log.info("Only one test recording present - skipping per-recording burden correlation.")
        return {}

    corr = float(np.corrcoef(agg["true_apnea_minutes"], agg["pred_apnea_minutes"])[0, 1])
    mae = float(np.mean(np.abs(agg["true_apnea_minutes"] - agg["pred_apnea_minutes"])))
    log.info(f"Per-recording apnea-minute correlation: r={corr:.3f}, MAE={mae:.2f} minutes")

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(agg["true_apnea_minutes"], agg["pred_apnea_minutes"])
    lims = [0, max(agg["true_apnea_minutes"].max(), agg["pred_apnea_minutes"].max()) + 1]
    ax.plot(lims, lims, "k--", alpha=0.5, label="perfect agreement")
    ax.set_xlabel("True apnea minutes per recording")
    ax.set_ylabel("Predicted apnea minutes per recording")
    ax.set_title(f"Per-recording apnea burden (r={corr:.2f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "per_recording_apnea_burden.png", dpi=150)
    plt.close(fig)

    agg.to_csv(out_dir / "per_recording_apnea_burden.csv", index=False)
    return {"correlation": corr, "mae_minutes": mae}


def plot_feature_importance(model, feature_cols: list[str], out_dir: Path, model_name: str):
    importances = None
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_).ravel()

    if importances is None:
        return

    order = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh([feature_cols[i] for i in order][::-1], importances[order][::-1])
    ax.set_title(f"Feature importance ({model_name})")
    fig.tight_layout()
    fig.savefig(out_dir / "feature_importance.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--test-csv", type=Path, default=None,
                         help="Optional. If omitted, only CV results are reported.")
    parser.add_argument("--out-dir", type=Path, default=Path("./model_output"))
    parser.add_argument("--cv-folds", type=int, default=5)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_df, X_train, y_train, feature_cols = load_features(args.train_csv)
    groups = train_df["record"].to_numpy()
    pos_weight = float(np.sum(y_train == 0)) / max(np.sum(y_train == 1), 1)

    log.info(f"Comparing models with {args.cv_folds}-fold GROUP-aware CV "
             f"(grouped by recording, not by row)...")
    models = build_model_zoo(pos_weight)
    cv_results = cross_validate_group_aware(models, X_train, y_train, groups,
                                             n_splits=args.cv_folds)
    cv_results.to_csv(args.out_dir / "cv_model_comparison.csv", index=False)

    if cv_results.empty:
        raise RuntimeError("No model produced valid CV folds - check your data/groups.")

    best_name = cv_results.iloc[0]["model"]
    log.info(f"Best model by mean CV ROC-AUC: {best_name}")

    # Group-aware CV splits, reused for both calibration and threshold tuning
    # so neither step sees a fold it wasn't held out for.
    n_splits = min(args.cv_folds, len(set(groups)))
    cv_splits = list(GroupKFold(n_splits=n_splits).split(X_train, y_train, groups))

    base_model = build_model_zoo(pos_weight)[best_name]
    log.info("Calibrating probabilities via cross-validated isotonic regression...")
    calibrated_model = CalibratedClassifierCV(base_model, method="isotonic", cv=cv_splits)
    calibrated_model.fit(X_train, y_train)

    log.info("Computing out-of-fold predictions to pick an optimal decision threshold "
             "(Youden's J) instead of assuming 0.5...")
    oof_proba = cross_val_predict(build_model_zoo(pos_weight)[best_name], X_train, y_train,
                                   cv=cv_splits, method="predict_proba")[:, 1]
    optimal_threshold = find_optimal_threshold(y_train, oof_proba)
    log.info(f"Optimal threshold from out-of-fold CV predictions: {optimal_threshold:.3f} "
             f"(vs naive default of 0.5)")

    best_model = calibrated_model
    joblib.dump({"model": best_model, "feature_cols": feature_cols, "model_name": best_name,
                 "decision_threshold": optimal_threshold},
                args.out_dir / "best_model.pkl")
    plot_feature_importance(base_model.fit(X_train, y_train), feature_cols, args.out_dir, best_name)

    results = {"cv_summary": cv_results.to_dict(orient="records"), "best_model": best_name,
               "decision_threshold": optimal_threshold}

    if args.test_csv is not None:
        test_df, X_test, y_test, _ = load_features(args.test_csv)
        test_metrics = evaluate_on_test(best_model, X_test, y_test, args.out_dir, best_name,
                                         threshold=optimal_threshold)
        y_proba_test = _predict_proba_safe(best_model, X_test)
        y_pred_test = (y_proba_test >= optimal_threshold).astype(int)
        burden_metrics = per_recording_apnea_burden(test_df, y_pred_test, args.out_dir)
        results["test_metrics"] = test_metrics
        results["per_recording_burden"] = burden_metrics
    else:
        log.info("No --test-csv provided; skipping held-out evaluation.")

    with open(args.out_dir / "results_summary.json", "w") as f:
        json.dump(results, f, indent=2, default=float)

    log.info(f"All outputs written to {args.out_dir}/")


if __name__ == "__main__":
    main()
