from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer


ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
XAI_DIR = RESULTS_DIR / "explainability"
FIG_DIR = RESULTS_DIR / "figures"

XAI_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


FEATURES = [
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


def random_forest_xai():

    print("=" * 70)
    print("XAI — RANDOM FOREST FEATURE IMPORTANCE")
    print("=" * 70)

    X = np.load(RESULTS_DIR / "experiments" / "temporal_X.npy")
    y = np.load(RESULTS_DIR / "experiments" / "temporal_y.npy")

    # Flatten 5 temporal steps × 11 features.
    X_flat = X.reshape(X.shape[0], -1)

    # Impute missing values.
    imputer = SimpleImputer(strategy="median")
    X_flat = imputer.fit_transform(X_flat)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_flat, y)

    importance = model.feature_importances_

    # Aggregate importance across the five temporal positions.
    aggregated = np.zeros(len(FEATURES))

    for t in range(X.shape[1]):
        start = t * len(FEATURES)
        end = start + len(FEATURES)
        aggregated += importance[start:end]

    aggregated /= X.shape[1]

    df = pd.DataFrame({
        "feature": FEATURES,
        "importance": aggregated,
    })

    df = df.sort_values(
        "importance",
        ascending=False,
    )

    df.to_csv(
        XAI_DIR / "random_forest_feature_importance.csv",
        index=False,
    )

    plt.figure(figsize=(9, 6))

    plt.barh(
        df["feature"],
        df["importance"],
    )

    plt.xlabel("Mean Feature Importance")
    plt.ylabel("ECG-derived Feature")
    plt.title("Random Forest Feature Importance")

    plt.gca().invert_yaxis()

    plt.tight_layout()

    plt.savefig(
        FIG_DIR / "10_rf_feature_importance.png",
        dpi=200,
    )

    plt.close()

    print("\nFeature importance:")
    print(df.to_string(index=False))

    print(
        "\nSaved:",
        XAI_DIR / "random_forest_feature_importance.csv",
    )

    print(
        "Saved:",
        FIG_DIR / "10_rf_feature_importance.png",
    )


def transformer_temporal_xai():

    print("\n" + "=" * 70)
    print("XAI — TEMPORAL IMPORTANCE")
    print("=" * 70)

    X = np.load(
        RESULTS_DIR / "experiments" / "temporal_X.npy"
    )

    y = np.load(
        RESULTS_DIR / "experiments" / "temporal_y.npy"
    )

    # A model-independent temporal saliency proxy:
    # calculate the magnitude of each feature at each temporal
    # position relative to the dataset distribution.
    mean = np.nanmean(X, axis=0)

    std = np.nanstd(X, axis=0) + 1e-8

    standardized = np.abs(
        (X - mean) / std
    )

    temporal_importance = np.nanmean(
        standardized,
        axis=(0, 2),
    )

    temporal_df = pd.DataFrame({
        "temporal_position": [
            "t-2 min",
            "t-1 min",
            "t",
            "t+1 min",
            "t+2 min",
        ],
        "relative_temporal_importance":
            temporal_importance,
    })

    temporal_df.to_csv(
        XAI_DIR / "temporal_feature_saliency.csv",
        index=False,
    )

    plt.figure(figsize=(8, 5))

    plt.bar(
        temporal_df["temporal_position"],
        temporal_df[
            "relative_temporal_importance"
        ],
    )

    plt.xlabel("Temporal Position")
    plt.ylabel("Relative Saliency")
    plt.title(
        "Temporal Importance Across 5-Minute Context Window"
    )

    plt.tight_layout()

    plt.savefig(
        FIG_DIR / "11_temporal_importance.png",
        dpi=200,
    )

    plt.close()

    print(
        temporal_df.to_string(index=False)
    )

    print(
        "\nSaved:",
        XAI_DIR / "temporal_feature_saliency.csv",
    )

    print(
        "Saved:",
        FIG_DIR / "11_temporal_importance.png",
    )


def main():

    random_forest_xai()
    transformer_temporal_xai()

    print("\n" + "=" * 70)
    print("XAI COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()