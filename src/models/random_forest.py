from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline


N_SPLITS = 5
SEED = 42

BASE_DIR = Path(__file__).resolve().parents[2]

X_PATH = (
    BASE_DIR
    / "results"
    / "experiments"
    / "temporal_X.npy"
)

Y_PATH = (
    BASE_DIR
    / "results"
    / "experiments"
    / "temporal_y.npy"
)

GROUPS_PATH = (
    BASE_DIR
    / "results"
    / "experiments"
    / "temporal_groups.npy"
)

RESULTS_DIR = BASE_DIR / "results"
PRED_DIR = RESULTS_DIR / "predictions"
METRICS_DIR = RESULTS_DIR / "metrics"
MODEL_DIR = BASE_DIR / "models" / "trained"

for directory in [
    PRED_DIR,
    METRICS_DIR,
    MODEL_DIR,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


def calculate_metrics(
    y_true,
    y_pred,
    y_prob,
):

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    return {
        "accuracy": accuracy_score(
            y_true,
            y_pred,
        ),
        "balanced_accuracy": balanced_accuracy_score(
            y_true,
            y_pred,
        ),
        "precision": precision_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "sensitivity": recall_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "specificity": specificity,
        "f1": f1_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(
            y_true,
            y_prob,
        ),
        "pr_auc": average_precision_score(
            y_true,
            y_prob,
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def run_random_forest(
    n_splits: int = N_SPLITS,
):

    print("=" * 70)
    print("RANDOM FOREST — SUBJECT-WISE 5-FOLD EVALUATION")
    print("=" * 70)

    print("\nLoading temporal dataset...")

    X_temporal = np.load(X_PATH)
    y = np.load(Y_PATH)
    groups = np.load(
        GROUPS_PATH,
        allow_pickle=True,
    )

    print(f"Temporal X shape : {X_temporal.shape}")
    print(f"Labels shape     : {y.shape}")
    print(f"Groups shape     : {groups.shape}")

    # --------------------------------------------------------
    # Flatten the temporal representation for Random Forest.
    #
    # CNN / Transformer:
    #     (samples, 5 minutes, 11 features)
    #
    # Random Forest:
    #     (samples, 55 features)
    #
    # Same samples, labels and subject groups.
    # --------------------------------------------------------

    X = X_temporal.reshape(
        X_temporal.shape[0],
        -1,
    )

    print(f"RF input shape    : {X.shape}")

    cv = GroupKFold(
        n_splits=n_splits
    )

    fold_results = []
    all_predictions = []

    for fold, (
        train_idx,
        test_idx,
    ) in enumerate(
        cv.split(
            X,
            y,
            groups,
        ),
        start=1,
    ):

        print(
            f"\nFOLD {fold}/{n_splits}"
        )

        X_train = X[train_idx]
        X_test = X[test_idx]

        y_train = y[train_idx]
        y_test = y[test_idx]

        pipeline = Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    ),
                ),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=150,
                        max_depth=12,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        random_state=SEED,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

        pipeline.fit(
            X_train,
            y_train,
        )

        probabilities = pipeline.predict_proba(
            X_test
        )[:, 1]

        predictions = (
            probabilities >= 0.5
        ).astype(int)

        metrics = calculate_metrics(
            y_test,
            predictions,
            probabilities,
        )

        metrics["fold"] = fold

        fold_results.append(
            metrics
        )

        fold_predictions = pd.DataFrame(
            {
                "fold": fold,
                "sample_index": test_idx,
                "group": groups[test_idx],
                "y_true": y_test,
                "y_pred": predictions,
                "y_prob": probabilities,
            }
        )

        all_predictions.append(
            fold_predictions
        )

        print(
            f"F1={metrics['f1']:.4f} | "
            f"Sensitivity={metrics['sensitivity']:.4f} | "
            f"Specificity={metrics['specificity']:.4f} | "
            f"ROC-AUC={metrics['roc_auc']:.4f}"
        )

    # --------------------------------------------------------
    # Aggregate results
    # --------------------------------------------------------

    metrics_names = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "sensitivity",
        "specificity",
        "f1",
        "roc_auc",
        "pr_auc",
    ]

    summary = {}

    for metric in metrics_names:

        values = [
            result[metric]
            for result in fold_results
        ]

        summary[metric] = {
            "mean": float(
                np.mean(values)
            ),
            "std": float(
                np.std(values)
            ),
        }

    # --------------------------------------------------------
    # Save fold metrics
    # --------------------------------------------------------

    fold_df = pd.DataFrame(
        fold_results
    )

    fold_df.to_csv(
        METRICS_DIR
        / "random_forest_folds.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    predictions_df = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    predictions_df.to_csv(
        PRED_DIR
        / "random_forest_predictions.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    summary["model"] = (
        "Random Forest"
    )

    summary["n_splits"] = n_splits
    summary["n_samples"] = int(
        len(y)
    )

    summary["n_subjects"] = int(
        len(np.unique(groups))
    )

    summary["input_shape"] = list(
        X.shape
    )

    with open(
        METRICS_DIR
        / "random_forest_summary.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Final console summary
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("RANDOM FOREST FINAL RESULTS")
    print("=" * 70)

    for metric in metrics_names:

        mean = summary[
            metric
        ]["mean"]

        std = summary[
            metric
        ]["std"]

        print(
            f"{metric:20s}: "
            f"{mean:.4f} ± {std:.4f}"
        )

    print("\nSaved:")
    print(
        METRICS_DIR
        / "random_forest_summary.json"
    )

    print(
        PRED_DIR
        / "random_forest_predictions.csv"
    )

    print("=" * 70)


if __name__ == "__main__":

    run_random_forest()