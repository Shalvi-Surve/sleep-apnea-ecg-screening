from __future__ import annotations

import subprocess
import sys
from pathlib import Path
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_step(title: str, command: list[str]) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    start = time.time()

    result = subprocess.run(
        [sys.executable, *command],
        cwd=PROJECT_ROOT,
    )

    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"\nFAILED: {title}")
        print(f"Time: {elapsed:.1f} seconds")
        sys.exit(result.returncode)

    print(f"\nCOMPLETED: {title}")
    print(f"Time: {elapsed:.1f} seconds")


def main():
    print("=" * 70)
    print("PHASE 2 — MODEL DEVELOPMENT AND EVALUATION")
    print("=" * 70)

    print(
        """
This phase evaluates ECG-based sleep apnea screening
using classical machine learning and lightweight
deep-learning models.

Baseline models:
  • Random Forest
  • 1D CNN

Proposed screening model:
  • Lightweight Transformer

Evaluation:
  • Subject-wise 5-fold validation
  • Accuracy
  • Balanced Accuracy
  • Sensitivity
  • Specificity
  • Precision
  • F1-score
  • ROC-AUC
  • PR-AUC
"""
    )

    run_step(
        "STEP 1 — Build temporal dataset",
        ["-m", "src.evaluation.temporal_dataset"],
    )

    run_step(
        "STEP 2 — Train Random Forest",
        ["-m", "src.models.random_forest"],
    )

    run_step(
        "STEP 3 — Train 1D CNN",
        ["scripts/train_cnn.py"],
    )

    run_step(
        "STEP 4 — Train Lightweight Transformer",
        ["scripts/train_transformer.py"],
    )

    run_step(
        "STEP 5 — Generate model comparison",
        ["scripts/build_model_comparison.py"],
    )

    run_step(
        "STEP 6 — Generate evaluation plots",
        ["scripts/save_evaluation_plots.py"],
    )

    print("\n" + "=" * 70)
    print("PHASE 2 COMPLETE")
    print("=" * 70)

    print(
        """
Generated outputs can be found in:

results/
models/trained/

The next stage is the Streamlit screening demonstration.
"""
    )


if __name__ == "__main__":
    main()