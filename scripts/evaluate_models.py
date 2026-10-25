from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

METRICS_DIR = ROOT / "results" / "metrics"
COMPARISON_DIR = ROOT / "results" / "model_comparison"


def load_json(path: Path):
    if not path.exists():
        return None

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():

    print("=" * 70)
    print("FINAL MODEL EVALUATION")
    print("=" * 70)

    comparison_file = (
        COMPARISON_DIR /
        "complete_model_comparison.csv"
    )

    if not comparison_file.exists():

        print("\nERROR:")
        print(
            "complete_model_comparison.csv "
            "was not found."
        )

        print(
            "\nRun first:"
        )

        print(
            "python scripts/build_model_comparison.py"
        )

        return

    comparison = pd.read_csv(
        comparison_file
    )

    print("\nFINAL MODEL PERFORMANCE")
    print("-" * 70)

    print(
        comparison.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    final_file = (
        COMPARISON_DIR /
        "final_evaluation_summary.csv"
    )

    comparison.to_csv(
        final_file,
        index=False
    )

    print("\nSaved:")
    print(final_file)

    print("\n" + "=" * 70)
    print("AVAILABLE MODEL ARTIFACTS")
    print("=" * 70)

    artifacts = [
        (
            "Random Forest",
            METRICS_DIR /
            "random_forest_summary.json"
        ),
        (
            "1D CNN",
            METRICS_DIR /
            "cnn_metrics.json"
        ),
        (
            "Transformer V1",
            METRICS_DIR /
            "transformer_metrics.json"
        ),
        (
            "Random Forest predictions",
            ROOT /
            "results/predictions/"
            "random_forest_predictions.csv"
        ),
        (
            "CNN predictions",
            ROOT /
            "results/predictions/"
            "cnn_predictions.csv"
        ),
        (
            "Transformer predictions",
            ROOT /
            "results/predictions/"
            "transformer_predictions.csv"
        ),
    ]

    for name, path in artifacts:

        if path.exists():
            print(f"✓ {name}")

        else:
            print(f"✗ MISSING — {name}")

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()