from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    auc,
)


ROOT = Path(__file__).resolve().parents[1]

PREDICTIONS_DIR = ROOT / "results" / "predictions"
FIGURES_DIR = ROOT / "results" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


FILES = {
    "random_forest":
        PREDICTIONS_DIR / "random_forest_predictions.csv",

    "cnn":
        PREDICTIONS_DIR / "cnn_predictions.csv",

    "transformer_v1":
        PREDICTIONS_DIR / "transformer_predictions.csv",

    "proposed_transformer_v2":
        PREDICTIONS_DIR
        / "proposed_transformer_v2_predictions.csv",
}


def find_column(df: pd.DataFrame, candidates: list[str]) -> str:
    for col in candidates:
        if col in df.columns:
            return col

    raise KeyError(
        f"Could not find any of {candidates}. "
        f"Available columns: {list(df.columns)}"
    )


def load_predictions(path: Path):

    df = pd.read_csv(path)

    true_col = find_column(
        df,
        [
            "y_true",
            "true_label",
            "label",
            "target",
            "actual",
        ],
    )

    pred_col = None

    for candidate in [
        "y_pred",
        "prediction",
        "pred",
        "predicted",
        "predicted_label",
    ]:
        if candidate in df.columns:
            pred_col = candidate
            break

    prob_col = None

    for candidate in [
        "y_prob",
        "probability",
        "prob",
        "apnea_probability",
        "predicted_probability",
        "score",
    ]:
        if candidate in df.columns:
            prob_col = candidate
            break

    y_true = df[true_col].astype(int).to_numpy()

    if pred_col is not None:
        y_pred = df[pred_col].astype(int).to_numpy()

    elif prob_col is not None:
        y_pred = (
            df[prob_col].to_numpy() >= 0.5
        ).astype(int)

    else:
        raise KeyError(
            f"No prediction column found in {path}. "
            f"Columns: {list(df.columns)}"
        )

    y_prob = None

    if prob_col is not None:
        y_prob = df[prob_col].astype(float).to_numpy()

    return y_true, y_pred, y_prob


def save_confusion_matrix(
    model_name,
    y_true,
    y_pred,
):

    cm = confusion_matrix(
        y_true,
        y_pred,
    )

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "Non-apnea",
            "Apnea",
        ],
    )

    disp.plot(
        ax=ax,
        values_format="d",
    )

    ax.set_title(
        f"{model_name} — Confusion Matrix"
    )

    fig.tight_layout()

    output = (
        FIGURES_DIR
        / f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png"
    )

    fig.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    return output


def save_roc_curve(
    model_name,
    y_true,
    y_prob,
):

    if y_prob is None:
        return None

    fpr, tpr, _ = roc_curve(
        y_true,
        y_prob,
    )

    score = auc(
        fpr,
        tpr,
    )

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    ax.plot(
        fpr,
        tpr,
        label=f"AUC = {score:.3f}",
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
    )

    ax.set_xlabel(
        "False Positive Rate"
    )

    ax.set_ylabel(
        "True Positive Rate"
    )

    ax.set_title(
        f"{model_name} — ROC Curve"
    )

    ax.legend()

    fig.tight_layout()

    output = (
        FIGURES_DIR
        / f"{model_name.lower().replace(' ', '_')}_roc_curve.png"
    )

    fig.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    return output


def save_pr_curve(
    model_name,
    y_true,
    y_prob,
):

    if y_prob is None:
        return None

    precision, recall, _ = precision_recall_curve(
        y_true,
        y_prob,
    )

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    ax.plot(
        recall,
        precision,
    )

    ax.set_xlabel(
        "Recall / Sensitivity"
    )

    ax.set_ylabel(
        "Precision"
    )

    ax.set_title(
        f"{model_name} — Precision-Recall Curve"
    )

    fig.tight_layout()

    output = (
        FIGURES_DIR
        / f"{model_name.lower().replace(' ', '_')}_pr_curve.png"
    )

    fig.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    return output


def main():

    print("=" * 70)
    print("GENERATING MODEL EVALUATION VISUALIZATIONS")
    print("=" * 70)

    for model_name, path in FILES.items():

        if not path.exists():

            print(
                f"SKIPPED: {model_name} "
                f"(prediction file missing)"
            )

            continue

        try:

            y_true, y_pred, y_prob = load_predictions(
                path
            )

            cm_path = save_confusion_matrix(
                model_name,
                y_true,
                y_pred,
            )

            roc_path = save_roc_curve(
                model_name,
                y_true,
                y_prob,
            )

            pr_path = save_pr_curve(
                model_name,
                y_true,
                y_prob,
            )

            print()
            print(
                f"{model_name.upper()}"
            )
            print(
                f"  Samples       : {len(y_true)}"
            )
            print(
                f"  Confusion     : {cm_path.name}"
            )

            if roc_path:
                print(
                    f"  ROC curve     : {roc_path.name}"
                )

            if pr_path:
                print(
                    f"  PR curve      : {pr_path.name}"
                )

        except Exception as exc:

            print(
                f"ERROR: {model_name}: {exc}"
            )

    print()
    print(
        f"All available figures saved to:"
    )
    print(
        FIGURES_DIR
    )

    print("=" * 70)


if __name__ == "__main__":
    main()