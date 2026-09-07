from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    RocCurveDisplay,
)


MODELS = [
    "random_forest",
    "cnn",
    "transformer",
]


def main():

    figure_dir = Path(
        "results/figures"
    )

    figure_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for model in MODELS:

        path = (
            Path("results/predictions")
            / f"{model}_predictions.csv"
        )

        df = pd.read_csv(path)

        y_true = df["y_true"]
        y_pred = df["y_pred"]
        y_prob = df["y_prob"]

        # Confusion matrix

        cm = confusion_matrix(
            y_true,
            y_pred,
        )

        fig, ax = plt.subplots(
            figsize=(5, 5)
        )

        ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=[
                "Non-apnea",
                "Apnea",
            ],
        ).plot(
            ax=ax,
            values_format="d",
        )

        ax.set_title(
            f"{model.replace('_', ' ').title()} "
            "Confusion Matrix"
        )

        fig.tight_layout()

        fig.savefig(
            figure_dir
            / f"{model}_confusion_matrix.png",
            dpi=200,
        )

        plt.close(fig)

        # ROC

        fig, ax = plt.subplots(
            figsize=(6, 5)
        )

        RocCurveDisplay.from_predictions(
            y_true,
            y_prob,
            ax=ax,
        )

        ax.set_title(
            f"{model.replace('_', ' ').title()} "
            "ROC Curve"
        )

        fig.tight_layout()

        fig.savefig(
            figure_dir
            / f"{model}_roc_curve.png",
            dpi=200,
        )

        plt.close(fig)

        print(
            f"Saved evaluation plots for {model}"
        )


if __name__ == "__main__":
    main()