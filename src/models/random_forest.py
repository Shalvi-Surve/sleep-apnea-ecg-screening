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
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = [
    "n_beats",
    "mean_rr",
    "sdnn",
    "rmssd",
    "pnn50",
    "cvrr",
    "mean_hr",
    "min_rr",
    "max_rr",
    "rr_range",
    "edr_amp_std",
]


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
            y_true, y_pred
        ),
        "balanced_accuracy":
            balanced_accuracy_score(
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
    data_path="data/processed/ml_ready_dataset.csv",
    n_splits=5,
):

    df = pd.read_csv(data_path)

    X = df[
        FEATURE_COLUMNS
    ].to_numpy(dtype=np.float32)

    y = df["label"].astype(int).to_numpy()
    groups = df["record"].to_numpy()

    cv = GroupKFold(
        n_splits=n_splits
    )

    fold_results = []
    all_predictions = []

    for fold, (train_idx, test_idx) in enumerate(
        cv.split(
            X,
            y,
            groups,
        ),
        start=1,
    ):

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
                        random_state=42,
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

        fold_results.append(metrics)

        fold_predictions = pd.DataFrame(
            {
                "fold": fold,
                "record": groups[test_idx],
                "y_true": y_test,
                "y_prob": probabilities,
                "y_pred": predictions,
            }
        )

        all_predictions.append(
            fold_predictions
        )

        print(
            f"Fold {fold}: "
            f"F1={metrics['f1']:.4f} | "
            f"Sensitivity={metrics['sensitivity']:.4f} | "
            f"Specificity={metrics['specificity']:.4f}"
        )

    results_df = pd.DataFrame(
        fold_results
    )

    prediction_df = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    summary = {
        metric: {
            "mean": float(
                results_df[metric].mean()
            ),
            "std": float(
                results_df[metric].std()
            ),
        }
        for metric in [
            "accuracy",
            "balanced_accuracy",
            "precision",
            "sensitivity",
            "specificity",
            "f1",
            "roc_auc",
            "pr_auc",
        ]
    }

    output_dir = Path(
        "results"
    )

    (output_dir / "metrics").mkdir(
        parents=True,
        exist_ok=True,
    )

    (output_dir / "predictions").mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        output_dir
        / "metrics"
        / "random_forest_folds.csv",
        index=False,
    )

    prediction_df.to_csv(
        output_dir
        / "predictions"
        / "random_forest_predictions.csv",
        index=False,
    )

    with open(
        output_dir
        / "metrics"
        / "random_forest_summary.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    print("\nRANDOM FOREST")
    print("=" * 50)

    for metric, values in summary.items():

        print(
            f"{metric:20s}: "
            f"{values['mean']:.4f} ± "
            f"{values['std']:.4f}"
        )

    return results_df, prediction_df


if __name__ == "__main__":
    run_random_forest()