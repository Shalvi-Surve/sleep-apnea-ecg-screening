from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

METRICS_DIR = ROOT / "results" / "metrics"
OUT_DIR = ROOT / "results" / "model_comparison"

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


MODELS = {
    "Random Forest":
        METRICS_DIR / "random_forest_summary.json",

    "1D CNN":
        METRICS_DIR / "cnn_metrics.json",

    "Transformer V1":
        METRICS_DIR / "transformer_metrics.json",

    "Proposed Transformer V2":
        METRICS_DIR / "proposed_transformer_v2_summary.json",
}


METRICS = [
    "accuracy",
    "balanced_accuracy",
    "precision",
    "sensitivity",
    "specificity",
    "f1",
    "roc_auc",
    "pr_auc",
]


def load(path):

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


def extract(data, metric):

    value = data.get(metric)

    if isinstance(value, dict):

        return (
            value.get("mean"),
            value.get("std"),
        )

    mean = data.get(
        f"{metric}_mean"
    )

    std = data.get(
        f"{metric}_std"
    )

    if mean is not None:

        return mean, std

    if value is not None:

        return value, None

    return None, None


def main():

    rows = []

    for model, path in MODELS.items():

        if not path.exists():

            print(
                f"Missing: {path}"
            )

            continue

        data = load(path)

        row = {
            "model": model
        }

        for metric in METRICS:

            mean, std = extract(
                data,
                metric,
            )

            row[metric] = mean

            row[
                f"{metric}_std"
            ] = std

        rows.append(row)

    df = pd.DataFrame(rows)

    df.to_csv(
        OUT_DIR
        / "complete_model_comparison.csv",
        index=False,
    )

    print()
    print("=" * 100)
    print(
        "FINAL MODEL COMPARISON"
    )
    print("=" * 100)

    print(
        df[
            [
                "model",
                "accuracy",
                "balanced_accuracy",
                "precision",
                "sensitivity",
                "specificity",
                "f1",
                "roc_auc",
                "pr_auc",
            ]
        ].to_string(
            index=False,
            float_format=lambda x:
            f"{x:.4f}",
        )
    )

    print()
    print(
        "BEST MODEL UNDER EACH METRIC"
    )
    print("-" * 100)

    winners = []

    for metric in METRICS:

        valid = df.dropna(
            subset=[metric]
        )

        if valid.empty:
            continue

        idx = valid[metric].idxmax()

        winner = valid.loc[
            idx,
            "model",
        ]

        score = valid.loc[
            idx,
            metric,
        ]

        winners.append(
            {
                "metric": metric,
                "best_model": winner,
                "score": score,
            }
        )

        print(
            f"{metric:<22} "
            f"{winner:<28} "
            f"{score:.4f}"
        )

    winners_df = pd.DataFrame(
        winners
    )

    winners_df.to_csv(
        OUT_DIR
        / "best_model_by_metric.csv",
        index=False,
    )

    print()
    print(
        "PROPOSED MODEL ADVANTAGE"
    )
    print("-" * 100)

    if "Proposed Transformer V2" in df["model"].values:

        v2 = df[
            df["model"]
            == "Proposed Transformer V2"
        ].iloc[0]

        for metric in METRICS:

            valid = df[
                df["model"]
                != "Proposed Transformer V2"
            ][metric].dropna()

            if valid.empty:
                continue

            best_baseline = valid.max()

            difference = (
                v2[metric]
                - best_baseline
            )

            status = (
                "BETTER"
                if difference > 0
                else "NOT BEST"
            )

            print(
                f"{metric:<22} "
                f"V2={v2[metric]:.4f} | "
                f"Best baseline={best_baseline:.4f} | "
                f"Δ={difference:+.4f} | "
                f"{status}"
            )

    print()
    print(
        "Saved:"
    )

    print(
        OUT_DIR
        / "complete_model_comparison.csv"
    )

    print(
        OUT_DIR
        / "best_model_by_metric.csv"
    )

    print("=" * 100)


if __name__ == "__main__":
    main()