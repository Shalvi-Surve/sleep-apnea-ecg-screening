from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.dataset_utils import get_learning_records
from src.data.load_ecg import load_record
from src.features.rpeak_detection import detect_r_peaks
from src.preprocessing.signal_cleaning import clean_ecg


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
FIGURES_DIR = Path("results/figures")

EPOCH_FEATURES_PATH = PROCESSED_DIR / "epoch_features.csv"

FS = 100
SAMPLE_RECORD = "a01"
SAMPLE_SECONDS = 60


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

log = logging.getLogger(__name__)


# ============================================================
# DIRECTORY SETUP
# ============================================================

def setup_directories() -> None:
    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# LOAD PROCESSED DATA
# ============================================================

def load_processed_dataset() -> pd.DataFrame:
    if not EPOCH_FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {EPOCH_FEATURES_PATH}"
        )

    df = pd.read_csv(
        EPOCH_FEATURES_PATH
    )

    log.info(
        "Loaded processed dataset: %s",
        df.shape,
    )

    return df


# ============================================================
# RAW + CLEANED RECORD STATISTICS
# ============================================================

def generate_record_statistics(
    records: list[str],
) -> pd.DataFrame:

    rows = []

    log.info(
        "Generating raw vs cleaned signal statistics..."
    )

    for name in records:

        try:
            record = load_record(
                name,
                DATA_DIR,
            )

            raw = np.asarray(
                record.ecg,
                dtype=float,
            )

            cleaned = clean_ecg(
                raw,
                record.fs,
            )

            raw_duration_sec = (
                len(raw) / record.fs
            )

            cleaned_duration_sec = (
                len(cleaned) / record.fs
            )

            rows.append(
                {
                    "record": name,
                    "sampling_rate_hz": record.fs,
                    "raw_samples": len(raw),
                    "cleaned_samples": len(cleaned),
                    "raw_duration_min": (
                        raw_duration_sec / 60
                    ),
                    "cleaned_duration_min": (
                        cleaned_duration_sec / 60
                    ),
                    "raw_min": float(
                        np.min(raw)
                    ),
                    "raw_max": float(
                        np.max(raw)
                    ),
                    "raw_mean": float(
                        np.mean(raw)
                    ),
                    "raw_std": float(
                        np.std(raw)
                    ),
                    "cleaned_min": float(
                        np.min(cleaned)
                    ),
                    "cleaned_max": float(
                        np.max(cleaned)
                    ),
                    "cleaned_mean": float(
                        np.mean(cleaned)
                    ),
                    "cleaned_std": float(
                        np.std(cleaned)
                    ),
                    "raw_nan_count": int(
                        np.isnan(raw).sum()
                    ),
                    "cleaned_nan_count": int(
                        np.isnan(cleaned).sum()
                    ),
                    "apn_labels": (
                        len(record.apn_labels)
                        if record.apn_labels is not None
                        else 0
                    ),
                }
            )

            log.info(
                "%s: %.2f min | raw std %.4f | "
                "cleaned std %.4f",
                name,
                raw_duration_sec / 60,
                np.std(raw),
                np.std(cleaned),
            )

        except Exception as exc:

            log.exception(
                "Failed statistics for %s: %s",
                name,
                exc,
            )

    result = pd.DataFrame(
        rows
    )

    output_path = (
        PROCESSED_DIR
        / "record_statistics.csv"
    )

    result.to_csv(
        output_path,
        index=False,
    )

    log.info(
        "Saved record statistics: %s",
        output_path,
    )

    return result


# ============================================================
# ML-READY DATASET
# ============================================================

def create_ml_ready_dataset(
    df: pd.DataFrame,
) -> pd.DataFrame:

    if "signal_quality" in result.columns:
        result = result[
            result["signal_quality"].eq("usable")
        ].copy()

    # Supervised learning requires valid labels.
    result = result[
        result["label"].notna()
    ].copy()

    # Keep only valid binary labels.
    result = result[
        result["label"].isin([0.0, 1.0])
    ].copy()

    result["label"] = (
        result["label"]
        .astype(int)
    )

    # Metadata columns are retained for traceability
    # but will NOT be used as model features.
    metadata_columns = [
        "record",
        "epoch_idx",
        "peak_source",
        "signal_quality",
        "quality_reason",
    ]

    feature_columns = [
        column
        for column in result.columns
        if column not in metadata_columns + ["label"]
    ]

    # Numeric feature table.
    feature_table = result[
        metadata_columns
        + feature_columns
        + ["label"]
    ].copy()

    output_path = (
        PROCESSED_DIR
        / "ml_ready_dataset.csv"
    )

    feature_table.to_csv(
        output_path,
        index=False,
    )

    log.info(
        "Saved ML-ready dataset: %s",
        output_path,
    )

    log.info(
        "ML-ready shape: %s",
        feature_table.shape,
    )

    log.info(
        "ML features: %d",
        len(feature_columns),
    )

    return feature_table


# ============================================================
# FEATURE STATISTICS
# ============================================================

def generate_feature_statistics(
    ml_df: pd.DataFrame,
) -> pd.DataFrame:

    excluded = {
        "record",
        "epoch_idx",
        "peak_source",
        "label",
    }

    feature_columns = [
        c
        for c in ml_df.columns
        if c not in excluded
    ]

    rows = []

    for feature in feature_columns:

        series = pd.to_numeric(
            ml_df[feature],
            errors="coerce",
        )

        rows.append(
            {
                "feature": feature,
                "dtype": str(
                    ml_df[feature].dtype
                ),
                "count": int(
                    series.notna().sum()
                ),
                "missing": int(
                    series.isna().sum()
                ),
                "missing_percent": float(
                    series.isna().mean() * 100
                ),
                "mean": float(
                    series.mean()
                ),
                "std": float(
                    series.std()
                ),
                "min": float(
                    series.min()
                ),
                "median": float(
                    series.median()
                ),
                "max": float(
                    series.max()
                ),
            }
        )

    result = pd.DataFrame(
        rows
    )

    output_path = (
        PROCESSED_DIR
        / "feature_statistics.csv"
    )

    result.to_csv(
        output_path,
        index=False,
    )

    log.info(
        "Saved feature statistics: %s",
        output_path,
    )

    return result


# ============================================================
# FIGURE 1 — RAW VS CLEANED ECG
# ============================================================

def plot_raw_vs_cleaned_ecg(
    record_name: str = SAMPLE_RECORD,
) -> None:

    record = load_record(
        record_name,
        DATA_DIR,
    )

    raw = np.asarray(
        record.ecg,
        dtype=float,
    )

    cleaned = clean_ecg(
        raw,
        record.fs,
    )

    n_samples = min(
        len(raw),
        SAMPLE_SECONDS * record.fs,
    )

    time = (
        np.arange(n_samples)
        / record.fs
    )

    fig = plt.figure(
        figsize=(12, 6)
    )

    plt.plot(
        time,
        raw[:n_samples],
        linewidth=0.8,
        label="Raw ECG",
        alpha=0.7,
    )

    plt.plot(
        time,
        cleaned[:n_samples],
        linewidth=0.9,
        label="Cleaned ECG",
    )

    plt.xlabel(
        "Time (seconds)"
    )

    plt.ylabel(
        "ECG amplitude"
    )

    plt.title(
        f"Raw vs Cleaned ECG — {record_name}"
    )

    plt.legend()

    plt.tight_layout()

    path = (
        FIGURES_DIR
        / "01_raw_vs_cleaned_ecg.png"
    )

    plt.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    log.info(
        "Saved %s",
        path,
    )


# ============================================================
# FIGURE 2 — ECG + R-PEAKS
# ============================================================

def plot_ecg_rpeaks(
    record_name: str = SAMPLE_RECORD,
) -> None:

    record = load_record(
        record_name,
        DATA_DIR,
    )

    cleaned = clean_ecg(
        record.ecg,
        record.fs,
    )

    n_samples = min(
        len(cleaned),
        SAMPLE_SECONDS * record.fs,
    )

    signal_segment = cleaned[
        :n_samples
    ]

    try:
        peaks = detect_r_peaks(
            signal_segment,
            record.fs,
            None,
        )
    except Exception:
        peaks = np.array(
            [],
            dtype=int,
        )

    peaks = np.asarray(
        peaks,
        dtype=int,
    )

    peaks = peaks[
        (peaks >= 0)
        & (peaks < n_samples)
    ]

    time = (
        np.arange(n_samples)
        / record.fs
    )

    fig = plt.figure(
        figsize=(12, 5)
    )

    plt.plot(
        time,
        signal_segment,
        linewidth=0.8,
        label="Cleaned ECG",
    )

    if len(peaks) > 0:

        plt.scatter(
            peaks / record.fs,
            signal_segment[peaks],
            s=18,
            label="Detected R-peaks",
            zorder=3,
        )

    plt.xlabel(
        "Time (seconds)"
    )

    plt.ylabel(
        "ECG amplitude"
    )

    plt.title(
        f"R-Peak Detection — {record_name}"
    )

    plt.legend()

    plt.tight_layout()

    path = (
        FIGURES_DIR
        / "02_ecg_rpeaks.png"
    )

    plt.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    log.info(
        "Saved %s",
        path,
    )


# ============================================================
# FIGURE 3 — RR TACHOGRAM
# ============================================================

def plot_rr_tachogram(
    record_name: str = SAMPLE_RECORD,
) -> None:

    record = load_record(
        record_name,
        DATA_DIR,
    )

    cleaned = clean_ecg(
        record.ecg,
        record.fs,
    )

    n_samples = min(
        len(cleaned),
        SAMPLE_SECONDS * record.fs,
    )

    segment = cleaned[
        :n_samples
    ]

    try:
        peaks = detect_r_peaks(
            segment,
            record.fs,
            None,
        )
    except Exception:
        peaks = np.array(
            [],
            dtype=int,
        )

    peaks = np.asarray(
        peaks,
        dtype=int,
    )

    if len(peaks) < 2:

        log.warning(
            "Not enough R-peaks for RR tachogram."
        )

        return

    rr = np.diff(
        peaks
    ) / record.fs

    rr_time = (
        peaks[1:]
        / record.fs
    )

    valid = (
        (rr >= 0.3)
        & (rr <= 2.0)
    )

    rr = rr[valid]
    rr_time = rr_time[valid]

    fig = plt.figure(
        figsize=(12, 5)
    )

    plt.plot(
        rr_time,
        rr,
        marker=".",
        markersize=4,
        linewidth=0.8,
    )

    plt.xlabel(
        "Time (seconds)"
    )

    plt.ylabel(
        "RR interval (seconds)"
    )

    plt.title(
        f"RR Interval Tachogram — {record_name}"
    )

    plt.tight_layout()

    path = (
        FIGURES_DIR
        / "03_rr_tachogram.png"
    )

    plt.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    log.info(
        "Saved %s",
        path,
    )


# ============================================================
# FIGURE 4 — CLASS DISTRIBUTION
# ============================================================

def plot_class_distribution(
    ml_df: pd.DataFrame,
) -> None:

    counts = (
        ml_df["label"]
        .value_counts()
        .sort_index()
    )

    labels = [
        "Non-apnea",
        "Apnea",
    ]

    values = [
        counts.get(0, 0),
        counts.get(1, 0),
    ]

    fig = plt.figure(
        figsize=(7, 5)
    )

    plt.bar(
        labels,
        values,
    )

    plt.ylabel(
        "Number of epochs"
    )

    plt.title(
        "ML-Ready Dataset Class Distribution"
    )

    for index, value in enumerate(values):

        plt.text(
            index,
            value,
            f"{value:,}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()

    path = (
        FIGURES_DIR
        / "04_class_distribution.png"
    )

    plt.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    log.info(
        "Saved %s",
        path,
    )


# ============================================================
# FIGURE 5 — PER-RECORDING APNEA BURDEN
# ============================================================

def plot_apnea_burden(
    ml_df: pd.DataFrame,
) -> None:

    burden = (
        ml_df
        .groupby("record")["label"]
        .mean()
        .sort_values(
            ascending=False
        )
        * 100
    )

    fig = plt.figure(
        figsize=(12, 6)
    )

    plt.bar(
        burden.index,
        burden.values,
    )

    plt.ylabel(
        "Apnea-labelled epochs (%)"
    )

    plt.xlabel(
        "Recording"
    )

    plt.title(
        "Per-Recording Epoch-Level Apnea Burden"
    )

    plt.xticks(
        rotation=60
    )

    plt.tight_layout()

    path = (
        FIGURES_DIR
        / "05_per_recording_apnea_burden.png"
    )

    plt.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    log.info(
        "Saved %s",
        path,
    )


# ============================================================
# FIGURE 6 — FEATURE DISTRIBUTIONS
# ============================================================

def plot_feature_distributions(
    ml_df: pd.DataFrame,
) -> None:

    features = [
        "mean_rr",
        "sdnn",
        "rmssd",
        "pnn50",
        "cvrr",
        "mean_hr",
        "rr_range",
        "edr_amp_std",
    ]

    available = [
        feature
        for feature in features
        if feature in ml_df.columns
    ]

    if not available:
        return

    n_cols = 2
    n_rows = int(
        np.ceil(
            len(available) / n_cols
        )
    )

    fig = plt.figure(
        figsize=(12, 4 * n_rows)
    )

    for i, feature in enumerate(
        available,
        start=1,
    ):

        ax = fig.add_subplot(
            n_rows,
            n_cols,
            i,
        )

        for label, group in ml_df.groupby(
            "label"
        ):

            values = pd.to_numeric(
                group[feature],
                errors="coerce",
            ).dropna()

            if len(values) > 0:

                ax.hist(
                    values,
                    bins=40,
                    alpha=0.5,
                    label=(
                        "Apnea"
                        if label == 1
                        else "Non-apnea"
                    ),
                )

        ax.set_title(
            feature
        )

        ax.set_xlabel(
            feature
        )

        ax.set_ylabel(
            "Frequency"
        )

        ax.legend()

    plt.tight_layout()

    path = (
        FIGURES_DIR
        / "06_feature_distributions.png"
    )

    plt.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    log.info(
        "Saved %s",
        path,
    )


# ============================================================
# FIGURE 7 — FEATURE MISSINGNESS
# ============================================================

def plot_feature_missingness(
    feature_stats: pd.DataFrame,
) -> None:

    data = (
        feature_stats
        .sort_values(
            "missing_percent",
            ascending=False,
        )
    )

    fig = plt.figure(
        figsize=(10, 5)
    )

    plt.bar(
        data["feature"],
        data["missing_percent"],
    )

    plt.ylabel(
        "Missing values (%)"
    )

    plt.xlabel(
        "Feature"
    )

    plt.title(
        "Feature Missingness in Processed Dataset"
    )

    plt.xticks(
        rotation=60
    )

    plt.tight_layout()

    path = (
        FIGURES_DIR
        / "07_feature_missingness.png"
    )

    plt.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    log.info(
        "Saved %s",
        path,
    )


# ============================================================
# FIGURE 8 — EPOCHS PER RECORD
# ============================================================

def plot_epochs_per_record(
    ml_df: pd.DataFrame,
) -> None:

    counts = (
        ml_df["record"]
        .value_counts()
        .sort_index()
    )

    fig = plt.figure(
        figsize=(12, 5)
    )

    plt.bar(
        counts.index,
        counts.values,
    )

    plt.xlabel(
        "Recording"
    )

    plt.ylabel(
        "Valid labelled epochs"
    )

    plt.title(
        "ML-Ready Epochs per Recording"
    )

    plt.xticks(
        rotation=60
    )

    plt.tight_layout()

    path = (
        FIGURES_DIR
        / "08_dataset_records.png"
    )

    plt.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    log.info(
        "Saved %s",
        path,
    )


# ============================================================
# FIGURE 9 — PREPROCESSING SUMMARY
# ============================================================

def plot_preprocessing_summary(
    record_stats: pd.DataFrame,
    processed_df: pd.DataFrame,
    ml_df: pd.DataFrame,
) -> None:

    raw_samples = (
        record_stats["raw_samples"]
        .sum()
    )

    cleaned_samples = (
        record_stats["cleaned_samples"]
        .sum()
    )

    raw_hours = (
        record_stats["raw_duration_min"]
        .sum()
        / 60
    )

    cleaned_hours = (
        record_stats["cleaned_duration_min"]
        .sum()
        / 60
    )

    processed_epochs = len(
        processed_df
    )

    ml_epochs = len(
        ml_df
    )

    dropped_epochs = (
        processed_epochs
        - ml_epochs
    )

    labels = [
        "Raw ECG\nsamples",
        "Cleaned ECG\nsamples",
        "Epochs\nbefore label filter",
        "ML-ready\nlabelled epochs",
    ]

    values = [
        raw_samples,
        cleaned_samples,
        processed_epochs,
        ml_epochs,
    ]

    fig = plt.figure(
        figsize=(10, 6)
    )

    plt.bar(
        labels,
        values,
    )

    plt.ylabel(
        "Count"
    )

    plt.title(
        "Phase-1 Data Processing Summary"
    )

    for i, value in enumerate(values):

        if value >= 1_000_000:
            text = f"{value / 1_000_000:.2f}M"
        elif value >= 1_000:
            text = f"{value / 1_000:.1f}K"
        else:
            text = f"{value:,}"

        plt.text(
            i,
            value,
            text,
            ha="center",
            va="bottom",
        )

    plt.tight_layout()

    path = (
        FIGURES_DIR
        / "09_preprocessing_summary.png"
    )

    plt.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    log.info(
        "Saved %s",
        path,
    )

    log.info(
        "Total raw duration: %.2f hours",
        raw_hours,
    )

    log.info(
        "Total cleaned duration: %.2f hours",
        cleaned_hours,
    )

    log.info(
        "Epochs removed by label filtering: %d",
        dropped_epochs,
    )


# ============================================================
# DATASET SUMMARY JSON
# ============================================================

def generate_dataset_summary(
    processed_df: pd.DataFrame,
    ml_df: pd.DataFrame,
    record_stats: pd.DataFrame,
    feature_stats: pd.DataFrame,
) -> None:

    label_counts = (
        ml_df["label"]
        .value_counts()
        .sort_index()
    )

    summary = {
        "phase": "Phase 1 - ECG preprocessing and ML-ready dataset",
        "dataset": {
            "records_processed": int(
                ml_df["record"].nunique()
            ),
            "processed_epochs": int(
                len(processed_df)
            ),
            "ml_ready_epochs": int(
                len(ml_df)
            ),
            "labelled_epochs_removed": int(
                len(processed_df)
                - len(ml_df)
            ),
            "features_available": int(
                len(
                    [
                        c
                        for c in ml_df.columns
                        if c
                        not in {
                            "record",
                            "epoch_idx",
                            "peak_source",
                            "label",
                        }
                    ]
                )
            ),
            "sampling_rate_hz": FS,
            "epoch_length_seconds": 60,
        },
        "labels": {
            "non_apnea_epochs": int(
                label_counts.get(0, 0)
            ),
            "apnea_epochs": int(
                label_counts.get(1, 0)
            ),
            "non_apnea_percent": float(
                (
                    label_counts.get(0, 0)
                    / max(len(ml_df), 1)
                )
                * 100
            ),
            "apnea_percent": float(
                (
                    label_counts.get(1, 0)
                    / max(len(ml_df), 1)
                )
                * 100
            ),
        },
        "r_peak_sources": {
            str(k): float(v)
            for k, v in (
                processed_df[
                    "peak_source"
                ]
                .value_counts(
                    normalize=True
                )
                .items()
            )
        },
        "signal": {
            "total_raw_samples": int(
                record_stats[
                    "raw_samples"
                ].sum()
            ),
            "total_cleaned_samples": int(
                record_stats[
                    "cleaned_samples"
                ].sum()
            ),
            "total_raw_duration_hours": float(
                record_stats[
                    "raw_duration_min"
                ].sum()
                / 60
            ),
            "total_cleaned_duration_hours": float(
                record_stats[
                    "cleaned_duration_min"
                ].sum()
                / 60
            ),
            "raw_nan_samples": int(
                record_stats[
                    "raw_nan_count"
                ].sum()
            ),
            "cleaned_nan_samples": int(
                record_stats[
                    "cleaned_nan_count"
                ].sum()
            ),
        },
        "features": {
            "feature_names": feature_stats[
                "feature"
            ].tolist(),
            "features_with_missing_values": int(
                (
                    feature_stats[
                        "missing"
                    ]
                    > 0
                ).sum()
            ),
        },
        "methodology_notes": [
            "34 learning-set records were processed.",
            "c05 was excluded because it duplicates c06.",
            "R-peaks were primarily detected independently using NeuroKit2.",
            "PhysioNet QRS annotations were used only as a fallback when necessary.",
            "Epochs with fewer than 20 beats were removed.",
            "Epochs without valid apnea labels are excluded from supervised ML.",
            "record and epoch_idx are metadata and must not be used as predictive features.",
            "peak_source is provenance metadata and must not be used as a predictive feature.",
            "Scaling and imputation must be fitted within each GroupKFold training fold during model development.",
        ],
    }

    output_path = (
        PROCESSED_DIR
        / "dataset_summary.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
        )

    log.info(
        "Saved dataset summary: %s",
        output_path,
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    setup_directories()

    log.info(
        "========== PHASE 1 ANALYSIS =========="
    )

    processed_df = (
        load_processed_dataset()
    )

    records = get_learning_records(
        keep_duplicate=False
    )

    log.info(
        "Learning records: %d",
        len(records),
    )

    record_stats = (
        generate_record_statistics(
            records
        )
    )

    ml_df = create_ml_ready_dataset(
        processed_df
    )

    feature_stats = (
        generate_feature_statistics(
            ml_df
        )
    )

    log.info(
        "Generating visualizations..."
    )

    plot_raw_vs_cleaned_ecg(
        SAMPLE_RECORD
    )

    plot_ecg_rpeaks(
        SAMPLE_RECORD
    )

    plot_rr_tachogram(
        SAMPLE_RECORD
    )

    plot_class_distribution(
        ml_df
    )

    plot_apnea_burden(
        ml_df
    )

    plot_feature_distributions(
        ml_df
    )

    plot_feature_missingness(
        feature_stats
    )

    plot_epochs_per_record(
        ml_df
    )

    plot_preprocessing_summary(
        record_stats,
        processed_df,
        ml_df,
    )

    generate_dataset_summary(
        processed_df,
        ml_df,
        record_stats,
        feature_stats,
    )

    # --------------------------------------------------------
    # FINAL CONSOLE SUMMARY
    # --------------------------------------------------------

    print()
    print("=" * 65)
    print("PHASE 1 COMPLETE")
    print("=" * 65)

    print(
        f"Records processed       : "
        f"{ml_df['record'].nunique()}"
    )

    print(
        f"Processed epochs        : "
        f"{len(processed_df):,}"
    )

    print(
        f"ML-ready epochs         : "
        f"{len(ml_df):,}"
    )

    print(
        f"Removed invalid labels  : "
        f"{len(processed_df) - len(ml_df):,}"
    )

    print(
        f"Non-apnea epochs        : "
        f"{(ml_df['label'] == 0).sum():,}"
    )

    print(
        f"Apnea epochs            : "
        f"{(ml_df['label'] == 1).sum():,}"
    )

    print(
        f"Feature count           : "
        f"{len(feature_stats)}"
    )

    print()
    print(
        "ML-ready dataset:"
    )
    print(
        f"  {PROCESSED_DIR / 'ml_ready_dataset.csv'}"
    )

    print()
    print(
        "Dataset summary:"
    )
    print(
        f"  {PROCESSED_DIR / 'dataset_summary.json'}"
    )

    print()
    print(
        "Figures:"
    )
    print(
        f"  {FIGURES_DIR}"
    )

    print("=" * 65)


if __name__ == "__main__":
    main()