from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

METRICS_DIR = BASE_DIR / "results" / "metrics"
COMPLEXITY_DIR = BASE_DIR / "results" / "complexity"
OUTPUT_DIR = BASE_DIR / "results" / "model_comparison"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def load_metrics(name):

    path = (
        METRICS_DIR
        / f"{name}_metrics.json"
    )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


def main():

    rows = []

    # --------------------------------------------------------
    # RANDOM FOREST
    # --------------------------------------------------------

    rf_path = (
        METRICS_DIR
        / "random_forest_metrics.json"
    )

    if rf_path.exists():

        with open(
            rf_path,
            "r",
            encoding="utf-8",
        ) as f:

            rf = json.load(f)

        rows.append(
            {
                "model": "Random Forest",
                "accuracy": rf["accuracy"],
                "accuracy_std": rf.get(
                    "accuracy_std",
                    0,
                ),
                "balanced_accuracy": rf[
                    "balanced_accuracy"
                ],
                "balanced_accuracy_std": rf.get(
                    "balanced_accuracy_std",
                    0,
                ),
                "precision": rf["precision"],
                "precision_std": rf.get(
                    "precision_std",
                    0,
                ),
                "sensitivity": rf["sensitivity"],
                "sensitivity_std": rf.get(
                    "sensitivity_std",
                    0,
                ),
                "specificity": rf["specificity"],
                "specificity_std": rf.get(
                    "specificity_std",
                    0,
                ),
                "f1": rf["f1"],
                "f1_std": rf.get(
                    "f1_std",
                    0,
                ),
                "roc_auc": rf["roc_auc"],
                "roc_auc_std": rf.get(
                    "roc_auc_std",
                    0,
                ),
                "pr_auc": rf["pr_auc"],
                "pr_auc_std": rf.get(
                    "pr_auc_std",
                    0,
                ),
            }
        )

    # --------------------------------------------------------
    # CNN
    # --------------------------------------------------------

    cnn = load_metrics(
        "cnn"
    )

    rows.append(
        {
            "model": "1D CNN",
            "accuracy": cnn["accuracy"],
            "accuracy_std": cnn[
                "accuracy_std"
            ],
            "balanced_accuracy": cnn[
                "balanced_accuracy"
            ],
            "balanced_accuracy_std": cnn[
                "balanced_accuracy_std"
            ],
            "precision": cnn[
                "precision"
            ],
            "precision_std": cnn[
                "precision_std"
            ],
            "sensitivity": cnn[
                "sensitivity"
            ],
            "sensitivity_std": cnn[
                "sensitivity_std"
            ],
            "specificity": cnn[
                "specificity"
            ],
            "specificity_std": cnn[
                "specificity_std"
            ],
            "f1": cnn["f1"],
            "f1_std": cnn[
                "f1_std"
            ],
            "roc_auc": cnn[
                "roc_auc"
            ],
            "roc_auc_std": cnn[
                "roc_auc_std"
            ],
            "pr_auc": cnn[
                "pr_auc"
            ],
            "pr_auc_std": cnn[
                "pr_auc_std"
            ],
        }
    )

    # --------------------------------------------------------
    # TRANSFORMER
    # --------------------------------------------------------

    transformer = load_metrics(
        "transformer"
    )

    rows.append(
        {
            "model": "Lightweight Transformer",
            "accuracy": transformer[
                "accuracy"
            ],
            "accuracy_std": transformer[
                "accuracy_std"
            ],
            "balanced_accuracy": transformer[
                "balanced_accuracy"
            ],
            "balanced_accuracy_std": transformer[
                "balanced_accuracy_std"
            ],
            "precision": transformer[
                "precision"
            ],
            "precision_std": transformer[
                "precision_std"
            ],
            "sensitivity": transformer[
                "sensitivity"
            ],
            "sensitivity_std": transformer[
                "sensitivity_std"
            ],
            "specificity": transformer[
                "specificity"
            ],
            "specificity_std": transformer[
                "specificity_std"
            ],
            "f1": transformer[
                "f1"
            ],
            "f1_std": transformer[
                "f1_std"
            ],
            "roc_auc": transformer[
                "roc_auc"
            ],
            "roc_auc_std": transformer[
                "roc_auc_std"
            ],
            "pr_auc": transformer[
                "pr_auc"
            ],
            "pr_auc_std": transformer[
                "pr_auc_std"
            ],
        }
    )

    df = pd.DataFrame(
        rows
    )

    output_path = (
        OUTPUT_DIR
        / "model_comparison.csv"
    )

    df.to_csv(
        output_path,
        index=False,
    )

    print()
    print(
        "=" * 70
    )

    print(
        "MODEL COMPARISON"
    )

    print(
        "=" * 70
    )

    display_cols = [
        "model",
        "accuracy",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "f1",
        "roc_auc",
        "pr_auc",
    ]

    print(
        df[display_cols].to_string(
            index=False
        )
    )

    print()
    print(
        f"Saved: {output_path}"
    )


if __name__ == "__main__":
    main()