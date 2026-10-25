from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "epoch_features.csv"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures"
METRIC_DIR = PROJECT_ROOT / "results" / "metrics"


FIGURE_DIR.mkdir(parents=True, exist_ok=True)
METRIC_DIR.mkdir(parents=True, exist_ok=True)


CORE_FEATURES = [
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


def load_dataset() -> pd.DataFrame:
    """Load the Phase-1 processed epoch feature dataset."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    return df


def create_dataset_summary(
    df: pd.DataFrame,
) -> dict:
    """Create a reproducible dataset summary."""

    labeled = df[df["label"].notna()].copy()

    summary = {
        "dataset": {
            "source": "PhysioNet Apnea-ECG",
            "processed_file": str(DATA_PATH),
            "total_epochs": int(len(df)),
            "labeled_epochs": int(len(labeled)),
            "unlabeled_epochs": int(df["label"].isna().sum()),
            "records": int(df["record"].nunique()),
            "features": int(len(CORE_FEATURES)),
        },
        "labels": {
            "non_apnea_epochs": int(
                (labeled["label"] == 0).sum()
            ),
            "apnea_epochs": int(
                (labeled["label"] == 1).sum()
            ),
            "non_apnea_percentage": float(
                (labeled["label"] == 0).mean() * 100
            ),
            "apnea_percentage": float(
                (labeled["label"] == 1).mean() * 100
            ),
        },
        "r_peak_sources": {
            str(key): int(value)
            for key, value in df["peak_source"]
            .value_counts(dropna=False)
            .items()
        },
        "missing_values": {
            str(key): int(value)
            for key, value in df[CORE_FEATURES]
            .isna()
            .sum()
            .items()
        },
        "feature_statistics": {},
    }

    for feature in CORE_FEATURES:
        if feature not in df.columns:
            continue

        values = df[feature].dropna()

        if len(values) == 0:
            continue

        summary["feature_statistics"][feature] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "median": float(values.median()),
            "min": float(values.min()),
            "max": float(values.max()),
        }

    output_path = METRIC_DIR / "dataset_summary.json"

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    print(
        f"Saved dataset summary: {output_path}"
    )

    return summary


def plot_class_distribution(
    df: pd.DataFrame,
) -> None:
    """Plot apnea vs non-apnea epoch distribution."""

    labeled = df[df["label"].notna()].copy()

    counts = (
        labeled["label"]
        .map({
            0.0: "Non-apnea",
            1.0: "Apnea",
        })
        .value_counts()
        .reindex([
            "Non-apnea",
            "Apnea",
        ])
    )

    plt.figure(figsize=(7, 5))

    bars = plt.bar(
        counts.index,
        counts.values,
    )

    plt.title(
        "Class Distribution of Labeled ECG Epochs"
    )

    plt.ylabel("Number of 1-minute epochs")

    plt.xlabel("Class")

    for bar, value in zip(
        bars,
        counts.values,
    ):
        plt.text(
            bar.get_x()
            + bar.get_width() / 2,
            bar.get_height(),
            f"{int(value):,}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()

    output_path = (
        FIGURE_DIR
        / "class_distribution.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved: {output_path}"
    )


def plot_per_recording_apnea_burden(
    df: pd.DataFrame,
) -> None:
    """
    Plot percentage of labeled epochs marked as apnea
    for every recording.
    """

    labeled = df[df["label"].notna()].copy()

    burden = (
        labeled.groupby("record")["label"]
        .mean()
        .mul(100)
        .sort_values(
            ascending=False
        )
    )

    plt.figure(
        figsize=(12, 6)
    )

    bars = plt.bar(
        burden.index,
        burden.values,
    )

    plt.title(
        "Per-Recording Apnea Epoch Burden"
    )

    plt.xlabel("Recording")

    plt.ylabel(
        "Apnea epochs (%)"
    )

    plt.xticks(
        rotation=45,
        ha="right",
    )

    for bar, value in zip(
        bars,
        burden.values,
    ):
        plt.text(
            bar.get_x()
            + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    plt.tight_layout()

    output_path = (
        FIGURE_DIR
        / "per_recording_apnea_burden.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved: {output_path}"
    )


def plot_feature_distributions(
    df: pd.DataFrame,
) -> None:
    """
    Plot distributions of core physiological features
    separately for apnea and non-apnea epochs.

    Each feature gets its own figure.
    """

    labeled = df[df["label"].notna()].copy()

    for feature in CORE_FEATURES:

        if feature not in labeled.columns:
            continue

        non_apnea = labeled.loc[
            labeled["label"] == 0,
            feature,
        ].dropna()

        apnea = labeled.loc[
            labeled["label"] == 1,
            feature,
        ].dropna()

        if (
            len(non_apnea) == 0
            or len(apnea) == 0
        ):
            continue

        plt.figure(
            figsize=(7, 5)
        )

        plt.hist(
            non_apnea,
            bins=40,
            alpha=0.6,
            label="Non-apnea",
            density=True,
        )

        plt.hist(
            apnea,
            bins=40,
            alpha=0.6,
            label="Apnea",
            density=True,
        )

        plt.title(
            f"{feature}: Apnea vs Non-apnea"
        )

        plt.xlabel(feature)

        plt.ylabel("Density")

        plt.legend()

        plt.tight_layout()

        output_path = (
            FIGURE_DIR
            / f"{feature}_distribution.png"
        )

        plt.savefig(
            output_path,
            dpi=200,
            bbox_inches="tight",
        )

        plt.close()

    print(
        "Saved feature distribution plots."
    )


def plot_feature_correlation(
    df: pd.DataFrame,
) -> None:
    """Plot correlation matrix of core features."""

    available = [
        feature
        for feature in CORE_FEATURES
        if feature in df.columns
    ]

    correlation = df[
        available
    ].corr()

    plt.figure(
        figsize=(10, 8)
    )

    image = plt.imshow(
        correlation,
        aspect="auto",
    )

    plt.colorbar(
        image,
        label="Correlation",
    )

    plt.xticks(
        range(len(available)),
        available,
        rotation=45,
        ha="right",
    )

    plt.yticks(
        range(len(available)),
        available,
    )

    plt.title(
        "Correlation Matrix of ECG-Derived Features"
    )

    plt.tight_layout()

    output_path = (
        FIGURE_DIR
        / "feature_correlation.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved: {output_path}"
    )


def plot_hrv_comparison(
    df: pd.DataFrame,
) -> None:
    """
    Compare major HRV features between apnea
    and non-apnea epochs.
    """

    labeled = df[df["label"].notna()].copy()

    features = [
        "mean_hr",
        "sdnn",
        "rmssd",
        "pnn50",
    ]

    available = [
        feature
        for feature in features
        if feature in labeled.columns
    ]

    if not available:
        return

    means_non_apnea = []
    means_apnea = []

    for feature in available:

        non_apnea_mean = labeled.loc[
            labeled["label"] == 0,
            feature,
        ].mean()

        apnea_mean = labeled.loc[
            labeled["label"] == 1,
            feature,
        ].mean()

        means_non_apnea.append(
            non_apnea_mean
        )

        means_apnea.append(
            apnea_mean
        )

    x = np.arange(
        len(available)
    )

    width = 0.35

    plt.figure(
        figsize=(10, 6)
    )

    plt.bar(
        x - width / 2,
        means_non_apnea,
        width,
        label="Non-apnea",
    )

    plt.bar(
        x + width / 2,
        means_apnea,
        width,
        label="Apnea",
    )

    plt.xticks(
        x,
        available,
        rotation=30,
        ha="right",
    )

    plt.ylabel(
        "Mean feature value"
    )

    plt.title(
        "Mean HRV Feature Comparison"
    )

    plt.legend()

    plt.tight_layout()

    output_path = (
        FIGURE_DIR
        / "sample_hrv_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved: {output_path}"
    )


def main() -> None:

    print(
        "\nLoading Phase-1 dataset..."
    )

    df = load_dataset()

    print(
        f"Dataset shape: {df.shape}"
    )

    print(
        f"Records: {df['record'].nunique()}"
    )

    print(
        f"Labeled epochs: "
        f"{df['label'].notna().sum()}"
    )

    print(
        f"Unlabeled epochs: "
        f"{df['label'].isna().sum()}"
    )

    print(
        "\nGenerating EDA artifacts..."
    )

    create_dataset_summary(df)

    plot_class_distribution(df)

    plot_per_recording_apnea_burden(df)

    plot_feature_distributions(df)

    plot_feature_correlation(df)

    plot_hrv_comparison(df)

    print(
        "\nPhase-1 EDA completed successfully."
    )

    print(
        f"Figures saved to: {FIGURE_DIR}"
    )

    print(
        f"Summary saved to: {METRIC_DIR}"
    )


if __name__ == "__main__":
    main()