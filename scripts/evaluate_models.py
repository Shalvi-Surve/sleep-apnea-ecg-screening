from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch

from src.models.cnn_1d import (
    LightweightCNN1D,
)
from src.models.lightweight_transformer import (
    LightweightTransformer,
)


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


def load_summary(model):

    path = Path(
        "results/metrics"
    ) / f"{model}_summary.json"

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


def count_parameters(model):

    return sum(
        p.numel()
        for p in model.parameters()
    )


def model_size_kb(model):

    parameters = count_parameters(
        model
    )

    return (
        parameters * 4
    ) / 1024


def main():

    rows = []

    rf = load_summary(
        "random_forest"
    )

    cnn = load_summary(
        "cnn"
    )

    transformer = load_summary(
        "transformer"
    )

    for model_name, summary in [
        ("Random Forest", rf),
        ("1D CNN", cnn),
        ("Lightweight Transformer", transformer),
    ]:

        row = {
            "model": model_name
        }

        for metric in METRICS:

            row[
                f"{metric}_mean"
            ] = summary[
                metric
            ]["mean"]

            row[
                f"{metric}_std"
            ] = summary[
                metric
            ]["std"]

        rows.append(row)

    comparison = pd.DataFrame(
        rows
    )

    cnn_model = LightweightCNN1D()

    transformer_model = (
        LightweightTransformer()
    )

    complexity = pd.DataFrame(
        [
            {
                "model":
                    "1D CNN",
                "parameters":
                    count_parameters(
                        cnn_model
                    ),
                "model_size_kb":
                    model_size_kb(
                        cnn_model
                    ),
            },
            {
                "model":
                    "Lightweight Transformer",
                "parameters":
                    count_parameters(
                        transformer_model
                    ),
                "model_size_kb":
                    model_size_kb(
                        transformer_model
                    ),
            },
        ]
    )

    Path(
        "results/model_comparison"
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    Path(
        "results/complexity"
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison.to_csv(
        "results/model_comparison/"
        "model_comparison.csv",
        index=False,
    )

    complexity.to_csv(
        "results/complexity/"
        "deep_model_complexity.csv",
        index=False,
    )

    print("\nMODEL COMPARISON")
    print("=" * 80)

    print(
        comparison[
            [
                "model",
                "accuracy_mean",
                "sensitivity_mean",
                "specificity_mean",
                "f1_mean",
                "roc_auc_mean",
                "pr_auc_mean",
            ]
        ].to_string(
            index=False
        )
    )

    print("\nMODEL COMPLEXITY")
    print("=" * 60)

    print(
        complexity.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()