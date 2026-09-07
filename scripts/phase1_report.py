from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.dataset_utils import get_learning_records
from src.data.load_ecg import load_record
from src.preprocessing.signal_cleaning import clean_ecg
from src.features.rpeak_detection import detect_r_peaks
from src.features.rr_intervals import rr_intervals_from_peaks


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

RESULTS_DIR = Path("results")
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"
EXPERIMENTS_DIR = RESULTS_DIR / "experiments"

EPOCH_FILE = PROCESSED_DIR / "epoch_features.csv"

ML_DATASET_FILE = PROCESSED_DIR / "ml_ready_dataset.csv"
X_FILE = PROCESSED_DIR / "X.npy"
Y_FILE = PROCESSED_DIR / "y.npy"
GROUPS_FILE = PROCESSED_DIR / "groups.npy"

# Additional Phase-1 deliverables
RPEAK_SUMMARY_FILE = PROCESSED_DIR / "rpeak_summary.csv"
RR_HRV_SUMMARY_FILE = PROCESSED_DIR / "rr_hrv_summary.csv"

RECORD_STATS_FILE = PROCESSED_DIR / "record_statistics.csv"
FEATURE_STATS_FILE = PROCESSED_DIR / "feature_statistics.csv"

SUMMARY_JSON_FILE = PROCESSED_DIR / "dataset_summary.json"
PHASE1_JSON_FILE = METRICS_DIR / "phase1_summary.json"
PHASE1_TXT_FILE = EXPERIMENTS_DIR / "phase1_summary.txt"


# Final figure names
RAW_CLEANED_FIGURE = FIGURES_DIR / "01_raw_vs_cleaned_ecg.png"
RPEAK_FIGURE = FIGURES_DIR / "02_ecg_rpeaks.png"
RR_TACHOGRAM_FIGURE = FIGURES_DIR / "03_rr_tachogram.png"
CLASS_FIGURE = FIGURES_DIR / "04_class_distribution.png"
APNEA_BURDEN_FIGURE = FIGURES_DIR / "05_per_recording_apnea_burden.png"
FEATURE_DISTRIBUTION_FIGURE = FIGURES_DIR / "06_feature_distributions.png"
DATASET_QUALITY_FIGURE = FIGURES_DIR / "07_dataset_quality.png"
DATASET_RECORDS_FIGURE = FIGURES_DIR / "08_dataset_records.png"
PREPROCESSING_SUMMARY_FIGURE = FIGURES_DIR / "09_preprocessing_summary.png"


# Obsolete / duplicate figures to remove
OBSOLETE_FIGURES = [
    FIGURES_DIR / "raw_vs_cleaned_ecg.png",
    FIGURES_DIR / "rpeak_detection.png",
    FIGURES_DIR / "feature_distributions.png",
    FIGURES_DIR / "apnea_burden_by_record.png",
    FIGURES_DIR / "feature_correlation.png",
    FIGURES_DIR / "07_feature_missingness.png",
]


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

log = logging.getLogger(__name__)


# ============================================================
# DIRECTORY SETUP
# ============================================================

def create_directories() -> None:
    """Create all required Phase-1 output directories."""

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    EXPERIMENTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# CLEAN OLD / DUPLICATE FIGURES
# ============================================================

def remove_obsolete_figures() -> None:
    """
    Remove old duplicate or superseded Phase-1 figures.

    This prevents the results/figures directory from containing
    multiple versions of the same visualization.
    """

    for path in OBSOLETE_FIGURES:

        if path.exists():

            try:
                path.unlink()

                log.info(
                    "Removed obsolete figure: %s",
                    path.name,
                )

            except Exception as exc:

                log.warning(
                    "Could not remove %s: %s",
                    path,
                    exc,
                )


# ============================================================
# LOAD PROCESSED DATA
# ============================================================

def load_processed_dataset() -> pd.DataFrame:
    """
    Load the already-generated epoch-level Phase-1 dataset.

    IMPORTANT:
    This function does not rerun ECG preprocessing.
    """

    if not EPOCH_FILE.exists():

        raise FileNotFoundError(
            f"Processed dataset not found: {EPOCH_FILE}"
        )

    df = pd.read_csv(
        EPOCH_FILE
    )

    log.info(
        "Loaded processed dataset: %s",
        df.shape,
    )

    required_columns = {
        "record",
        "epoch_idx",
        "peak_source",
        "label",
        "n_beats",
    }

    missing = required_columns.difference(
        df.columns
    )

    if missing:

        raise ValueError(
            "epoch_features.csv is missing required columns: "
            + ", ".join(sorted(missing))
        )

    return df


# ============================================================
# RAW DATA STATISTICS
# ============================================================

def calculate_raw_statistics(
    records: list[str],
) -> dict:
    """
    Calculate statistics directly from the raw PhysioNet
    recordings.
    """

    total_samples = 0
    total_duration_sec = 0.0
    raw_epochs = 0
    successful_records = 0

    record_statistics = []

    for name in records:

        try:

            record = load_record(
                name,
                DATA_DIR,
            )

            n_samples = len(
                record.ecg
            )

            duration_sec = (
                n_samples / record.fs
            )

            n_epochs = (
                n_samples
                // (record.fs * 60)
            )

            total_samples += n_samples
            total_duration_sec += duration_sec
            raw_epochs += n_epochs
            successful_records += 1

            record_statistics.append(
                {
                    "record": name,
                    "samples": int(
                        n_samples
                    ),
                    "sampling_rate_hz": int(
                        record.fs
                    ),
                    "duration_minutes": round(
                        duration_sec / 60,
                        2,
                    ),
                    "raw_epochs": int(
                        n_epochs
                    ),
                }
            )

        except Exception as exc:

            log.warning(
                "Could not load raw record %s: %s",
                name,
                exc,
            )

    record_df = pd.DataFrame(
        record_statistics
    )

    if not record_df.empty:

        record_df.to_csv(
            RECORD_STATS_FILE,
            index=False,
        )

    return {
        "records_requested": len(records),
        "records_successfully_loaded": successful_records,
        "total_samples": total_samples,
        "total_duration_minutes": round(
            total_duration_sec / 60,
            2,
        ),
        "raw_one_minute_epochs": raw_epochs,
        "record_statistics": record_statistics,
    }


# ============================================================
# R-PEAK SUMMARY
# ============================================================

def create_rpeak_summary(
    processed_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create one R-peak quality summary row per recording.

    This summary is derived from the already-generated
    epoch_features.csv and therefore does not rerun
    R-peak detection.
    """

    summary_rows = []

    for record_name, group in processed_df.groupby(
        "record",
        sort=True,
    ):

        total_epochs = len(
            group
        )

        usable_epochs = int(
            (
                group["n_beats"]
                >= 20
            ).sum()
        )

        unusable_epochs = int(
            (
                group["n_beats"]
                < 20
            ).sum()
        )

        source_counts = (
            group["peak_source"]
            .fillna("unknown")
            .value_counts()
        )

        total_beats = int(
            pd.to_numeric(
                group["n_beats"],
                errors="coerce",
            )
            .fillna(0)
            .sum()
        )

        mean_beats = (
            total_beats / max(
                total_epochs,
                1,
            )
        )

        summary_rows.append(
            {
                "record": record_name,
                "total_epochs": total_epochs,
                "usable_epochs": usable_epochs,
                "unusable_epochs": unusable_epochs,
                "neurokit_epochs": int(
                    source_counts.get(
                        "neurokit",
                        0,
                    )
                ),
                "qrs_fallback_epochs": int(
                    source_counts.get(
                        "qrs_fallback",
                        0,
                    )
                ),
                "total_beats": total_beats,
                "mean_beats_per_epoch": round(
                    mean_beats,
                    2,
                ),
                "min_beats_per_epoch": int(
                    pd.to_numeric(
                        group["n_beats"],
                        errors="coerce",
                    )
                    .min()
                ),
                "max_beats_per_epoch": int(
                    pd.to_numeric(
                        group["n_beats"],
                        errors="coerce",
                    )
                    .max()
                ),
            }
        )

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_df.to_csv(
        RPEAK_SUMMARY_FILE,
        index=False,
    )

    log.info(
        "Saved R-peak summary: %s",
        RPEAK_SUMMARY_FILE,
    )

    return summary_df


# ============================================================
# RR / HRV SUMMARY
# ============================================================

def create_rr_hrv_summary(
    processed_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create a per-recording summary of RR/HRV features.

    The values are calculated from the epoch-level physiological
    features already generated during Phase 1.
    """

    hrv_features = [
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

    available_features = [
        feature
        for feature in hrv_features
        if feature in processed_df.columns
    ]

    rows = []

    for record_name, group in processed_df.groupby(
        "record",
        sort=True,
    ):

        row = {
            "record": record_name,
            "epochs": len(group),
        }

        for feature in available_features:

            values = pd.to_numeric(
                group[feature],
                errors="coerce",
            )

            row[
                f"{feature}_mean"
            ] = round(
                values.mean(),
                6,
            )

            row[
                f"{feature}_median"
            ] = round(
                values.median(),
                6,
            )

            row[
                f"{feature}_std"
            ] = round(
                values.std(),
                6,
            )

            row[
                f"{feature}_missing"
            ] = int(
                values.isna().sum()
            )

        rows.append(
            row
        )

    summary_df = pd.DataFrame(
        rows
    )

    summary_df.to_csv(
        RR_HRV_SUMMARY_FILE,
        index=False,
    )

    log.info(
        "Saved RR/HRV summary: %s",
        RR_HRV_SUMMARY_FILE,
    )

    return summary_df


# ============================================================
# ML-READY DATASET
# ============================================================

def create_ml_ready_dataset(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """
    Create the final supervised ML dataset.

    Only valid labels and finite numerical feature rows
    are retained.

    IMPORTANT:
    No scaling or imputation is performed here.

    Scaling/imputation must be performed inside each
    GroupKFold training split during model evaluation.
    """

    original_rows = len(
        df
    )

    # --------------------------------------------------------
    # Remove rows without valid labels
    # --------------------------------------------------------

    valid_label_mask = df[
        "label"
    ].isin(
        [0.0, 1.0]
    )

    invalid_label_rows = int(
        (~valid_label_mask).sum()
    )

    df_valid = df.loc[
        valid_label_mask
    ].copy()

    # --------------------------------------------------------
    # Identify modelling features
    # --------------------------------------------------------

    metadata_columns = {
        "record",
        "epoch_idx",
        "peak_source",
        "label",
    }

    feature_columns = [
        column
        for column in df_valid.columns
        if column not in metadata_columns
    ]

    # --------------------------------------------------------
    # Convert features to numeric
    # --------------------------------------------------------

    for column in feature_columns:

        df_valid[column] = pd.to_numeric(
            df_valid[column],
            errors="coerce",
        )

    # --------------------------------------------------------
    # Remove non-finite feature rows
    # --------------------------------------------------------

    feature_array = (
        df_valid[
            feature_columns
        ]
        .to_numpy(
            dtype=float
        )
    )

    feature_valid_mask = np.isfinite(
        feature_array
    ).all(
        axis=1
    )

    invalid_feature_rows = int(
        (~feature_valid_mask).sum()
    )

    df_ml = df_valid.loc[
        feature_valid_mask
    ].copy()

    # --------------------------------------------------------
    # Consistent ordering
    # --------------------------------------------------------

    df_ml = df_ml.sort_values(
        [
            "record",
            "epoch_idx",
        ]
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # Create X / y / groups
    # --------------------------------------------------------

    X = df_ml[
        feature_columns
    ].to_numpy(
        dtype=np.float32
    )

    y = df_ml[
        "label"
    ].to_numpy(
        dtype=np.int64
    )

    groups = (
        df_ml[
            "record"
        ]
        .astype(str)
        .to_numpy()
    )

    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------

    df_ml.to_csv(
        ML_DATASET_FILE,
        index=False,
    )

    np.save(
        X_FILE,
        X,
    )

    np.save(
        Y_FILE,
        y,
    )

    np.save(
        GROUPS_FILE,
        groups,
    )

    summary = {
        "original_processed_rows": original_rows,
        "invalid_label_rows_removed": invalid_label_rows,
        "invalid_feature_rows_removed": invalid_feature_rows,
        "ml_ready_rows": len(df_ml),
        "feature_count": len(
            feature_columns
        ),
        "feature_columns": feature_columns,
        "non_apnea_epochs": int(
            (y == 0).sum()
        ),
        "apnea_epochs": int(
            (y == 1).sum()
        ),
        "subject_group_count": int(
            df_ml[
                "record"
            ].nunique()
        ),
        "X_shape": list(
            X.shape
        ),
        "y_shape": list(
            y.shape
        ),
        "groups_shape": list(
            groups.shape
        ),
    }

    return (
        df_ml,
        summary,
    )


# ============================================================
# FEATURE STATISTICS
# ============================================================

def create_feature_statistics(
    ml_df: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """
    Save descriptive statistics for all ML features.
    """

    statistics = (
        ml_df[
            feature_columns
        ]
        .describe()
        .T
        .reset_index()
        .rename(
            columns={
                "index": "feature"
            }
        )
    )

    statistics.to_csv(
        FEATURE_STATS_FILE,
        index=False,
    )

    return statistics


# ============================================================
# DATASET SUMMARY
# ============================================================

def create_dataset_summary(
    processed_df: pd.DataFrame,
    ml_df: pd.DataFrame,
    raw_stats: dict,
    ml_stats: dict,
    rpeak_summary: pd.DataFrame,
    rr_hrv_summary: pd.DataFrame,
) -> dict:

    raw_epochs = int(
        raw_stats[
            "raw_one_minute_epochs"
        ]
    )

    processed_epochs = int(
        len(processed_df)
    )

    low_quality_removed = max(
        raw_epochs
        - processed_epochs,
        0,
    )

    invalid_labels = int(
        ml_stats[
            "invalid_label_rows_removed"
        ]
    )

    invalid_features = int(
        ml_stats[
            "invalid_feature_rows_removed"
        ]
    )

    summary = {

        "project":
            "Lightweight ECG-Based Screening System for Early Sleep Apnea Detection",

        "phase":
            "Phase 1 - ECG and Data Preparation Pipeline",

        "dataset":
            "PhysioNet Apnea-ECG",

        "sampling_frequency_hz":
            100,

        "epoch_duration_seconds":
            60,

        "duplicate_record_note":
            "c05 is excluded because it is treated as a duplicate of c06.",

        "raw_data": {
            "records":
                raw_stats[
                    "records_successfully_loaded"
                ],

            "total_samples":
                raw_stats[
                    "total_samples"
                ],

            "total_duration_minutes":
                raw_stats[
                    "total_duration_minutes"
                ],

            "raw_one_minute_epochs":
                raw_epochs,
        },

        "processed_data": {

            "rows":
                processed_epochs,

            "columns":
                len(
                    processed_df.columns
                ),

            "records":
                int(
                    processed_df[
                        "record"
                    ].nunique()
                ),

            "usable_epochs":
                processed_epochs,
        },

        "quality_control": {

            "low_quality_epochs_removed":
                low_quality_removed,

            "invalid_label_rows_removed":
                invalid_labels,

            "invalid_feature_rows_removed":
                invalid_features,

            "peak_sources":
                {
                    str(key): int(value)
                    for key, value in
                    processed_df[
                        "peak_source"
                    ]
                    .fillna("unknown")
                    .value_counts()
                    .items()
                },
        },

        "ml_ready_data":
            ml_stats,

        "rpeak_summary": {

            "records":
                len(
                    rpeak_summary
                ),

            "total_epochs":
                int(
                    rpeak_summary[
                        "total_epochs"
                    ].sum()
                ),

            "usable_epochs":
                int(
                    rpeak_summary[
                        "usable_epochs"
                    ].sum()
                ),

            "unusable_epochs":
                int(
                    rpeak_summary[
                        "unusable_epochs"
                    ].sum()
                ),
        },

        "output_files": {

            "epoch_features":
                str(EPOCH_FILE),

            "ml_ready_dataset":
                str(ML_DATASET_FILE),

            "X":
                str(X_FILE),

            "y":
                str(Y_FILE),

            "groups":
                str(GROUPS_FILE),

            "rpeak_summary":
                str(RPEAK_SUMMARY_FILE),

            "rr_hrv_summary":
                str(RR_HRV_SUMMARY_FILE),

            "record_statistics":
                str(RECORD_STATS_FILE),

            "feature_statistics":
                str(FEATURE_STATS_FILE),

            "dataset_summary":
                str(SUMMARY_JSON_FILE),

            "phase1_summary":
                str(PHASE1_JSON_FILE),

            "text_report":
                str(PHASE1_TXT_FILE),
        },

        "feature_statistics":
            ml_df[
                ml_stats[
                    "feature_columns"
                ]
            ]
            .describe()
            .round(4)
            .to_dict(),
    }

    return summary


# ============================================================
# VISUALIZATION 01
# RAW VS CLEANED ECG
# ============================================================

def plot_raw_vs_cleaned_ecg(
    record_name: str = "a01",
    seconds: int = 10,
) -> None:

    try:

        record = load_record(
            record_name,
            DATA_DIR,
        )

        n_samples = min(
            seconds * record.fs,
            len(record.ecg),
        )

        raw = record.ecg[
            :n_samples
        ]

        cleaned_full = clean_ecg(
            record.ecg,
            record.fs,
        )

        cleaned = cleaned_full[
            :n_samples
        ]

        time = (
            np.arange(
                n_samples
            )
            / record.fs
        )

        fig, ax = plt.subplots(
            figsize=(12, 6)
        )

        ax.plot(
            time,
            raw,
            label="Raw ECG",
            alpha=0.65,
        )

        ax.plot(
            time,
            cleaned,
            label="Cleaned ECG",
        )

        ax.set_xlabel(
            "Time (seconds)"
        )

        ax.set_ylabel(
            "Amplitude"
        )

        ax.set_title(
            f"Raw vs Cleaned ECG — {record_name}"
        )

        ax.legend()

        fig.tight_layout()

        fig.savefig(
            RAW_CLEANED_FIGURE,
            dpi=200,
            bbox_inches="tight",
        )

        plt.close(fig)

        log.info(
            "Saved %s",
            RAW_CLEANED_FIGURE.name,
        )

    except Exception as exc:

        log.warning(
            "Could not generate raw-vs-cleaned ECG plot: %s",
            exc,
        )


# ============================================================
# VISUALIZATION 02
# R-PEAK DETECTION
# ============================================================

def plot_rpeak_detection(
    record_name: str = "a01",
    seconds: int = 10,
) -> None:

    try:

        record = load_record(
            record_name,
            DATA_DIR,
        )

        cleaned = clean_ecg(
            record.ecg,
            record.fs,
        )

        n_samples = min(
            seconds * record.fs,
            len(cleaned),
        )

        segment = cleaned[
            :n_samples
        ]

        peaks = detect_r_peaks(
            segment,
            record.fs,
            None,
        )

        peaks = np.asarray(
            peaks,
            dtype=int,
        )

        valid_peaks = peaks[
            (peaks >= 0)
            & (peaks < n_samples)
        ]

        time = (
            np.arange(
                n_samples
            )
            / record.fs
        )

        fig, ax = plt.subplots(
            figsize=(12, 5)
        )

        ax.plot(
            time,
            segment,
            label="Cleaned ECG",
        )

        if len(valid_peaks) > 0:

            peak_times = (
                valid_peaks
                / record.fs
            )

            ax.scatter(
                peak_times,
                segment[
                    valid_peaks
                ],
                marker="o",
                label="Detected R-peaks",
                zorder=3,
            )

        ax.set_xlabel(
            "Time (seconds)"
        )

        ax.set_ylabel(
            "Amplitude"
        )

        ax.set_title(
            f"R-Peak Detection — {record_name}"
        )

        ax.legend()

        fig.tight_layout()

        fig.savefig(
            RPEAK_FIGURE,
            dpi=200,
            bbox_inches="tight",
        )

        plt.close(fig)

        log.info(
            "Saved %s",
            RPEAK_FIGURE.name,
        )

    except Exception as exc:

        log.warning(
            "Could not generate R-peak plot: %s",
            exc,
        )


# ============================================================
# VISUALIZATION 03
# RR TACHOGRAM
# ============================================================

def plot_rr_tachogram(
    record_name: str = "a01",
    epoch_index: int = 0,
) -> None:
    """
    Generate an RR tachogram for one representative
    one-minute epoch.
    """

    try:

        record = load_record(
            record_name,
            DATA_DIR,
        )

        cleaned = clean_ecg(
            record.ecg,
            record.fs,
        )

        epoch_len = (
            record.fs * 60
        )

        start = (
            epoch_index
            * epoch_len
        )

        end = min(
            start + epoch_len,
            len(cleaned),
        )

        if end <= start:

            raise ValueError(
                "Requested epoch is outside recording."
            )

        segment = cleaned[
            start:end
        ]

        peaks = detect_r_peaks(
            segment,
            record.fs,
            None,
        )

        peaks = np.asarray(
            peaks,
            dtype=int,
        )

        peaks = peaks[
            (peaks >= 0)
            & (peaks < len(segment))
        ]

        if len(peaks) < 3:

            raise ValueError(
                "Not enough R-peaks detected for RR tachogram."
            )

        rr, rr_times = (
            rr_intervals_from_peaks(
                peaks,
                record.fs,
            )
        )

        rr = np.asarray(
            rr,
            dtype=float,
        )

        rr_times = np.asarray(
            rr_times,
            dtype=float,
        )

        valid = (
            np.isfinite(rr)
            & np.isfinite(rr_times)
        )

        rr = rr[
            valid
        ]

        rr_times = rr_times[
            valid
        ]

        if len(rr) == 0:

            raise ValueError(
                "No valid RR intervals available."
            )

        fig, ax = plt.subplots(
            figsize=(12, 5)
        )

        ax.plot(
            rr_times,
            rr,
            marker="o",
            markersize=3,
            linewidth=1,
        )

        ax.set_xlabel(
            "Time (seconds)"
        )

        ax.set_ylabel(
            "RR interval (seconds)"
        )

        ax.set_title(
            f"RR Tachogram — {record_name}, "
            f"Epoch {epoch_index}"
        )

        ax.grid(
            alpha=0.25
        )

        fig.tight_layout()

        fig.savefig(
            RR_TACHOGRAM_FIGURE,
            dpi=200,
            bbox_inches="tight",
        )

        plt.close(fig)

        log.info(
            "Saved %s",
            RR_TACHOGRAM_FIGURE.name,
        )

    except Exception as exc:

        log.warning(
            "Could not generate RR tachogram: %s",
            exc,
        )


# ============================================================
# VISUALIZATION 04
# CLASS DISTRIBUTION
# ============================================================

def plot_class_distribution(
    df: pd.DataFrame,
) -> None:

    counts = (
        df["label"]
        .dropna()
        .value_counts()
        .sort_index()
    )

    labels = [
        "Non-apnea (0)",
        "Apnea (1)",
    ]

    values = [
        int(
            counts.get(
                0.0,
                0,
            )
        ),
        int(
            counts.get(
                1.0,
                0,
            )
        ),
    ]

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    bars = ax.bar(
        labels,
        values,
    )

    ax.set_ylabel(
        "Number of epochs"
    )

    ax.set_title(
        "Apnea vs Non-Apnea Epoch Distribution"
    )

    for bar, value in zip(
        bars,
        values,
    ):

        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            bar.get_height(),
            f"{value:,}",
            ha="center",
            va="bottom",
        )

    fig.tight_layout()

    fig.savefig(
        CLASS_FIGURE,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# VISUALIZATION 05
# APNEA BURDEN BY RECORD
# ============================================================

def plot_apnea_burden_by_record(
    df: pd.DataFrame,
) -> None:

    record_stats = (
        df.dropna(
            subset=["label"]
        )
        .groupby("record")
        .agg(
            total_epochs=(
                "label",
                "size",
            ),
            apnea_epochs=(
                "label",
                "sum",
            ),
        )
    )

    record_stats[
        "apnea_percentage"
    ] = (
        record_stats[
            "apnea_epochs"
        ]
        / record_stats[
            "total_epochs"
        ]
        * 100
    )

    record_stats = (
        record_stats
        .sort_values(
            "apnea_percentage"
        )
    )

    fig, ax = plt.subplots(
        figsize=(12, 7)
    )

    ax.bar(
        record_stats.index,
        record_stats[
            "apnea_percentage"
        ],
    )

    ax.set_xlabel(
        "Recording"
    )

    ax.set_ylabel(
        "Apnea epochs (%)"
    )

    ax.set_title(
        "Per-Recording Apnea Burden"
    )

    ax.tick_params(
        axis="x",
        rotation=90,
    )

    fig.tight_layout()

    fig.savefig(
        APNEA_BURDEN_FIGURE,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# VISUALIZATION 06
# FEATURE DISTRIBUTIONS — APNEA VS NON-APNEA
# ============================================================

def plot_feature_distributions(
    df: pd.DataFrame,
) -> None:
    """
    Plot physiological feature distributions separately
    for apnea and non-apnea epochs.

    This comparison is intentionally retained because it
    shows whether physiological features differ between
    the two target classes.
    """

    metadata = {
        "record",
        "epoch_idx",
        "peak_source",
        "label",
    }

    features = [
        column
        for column in df.columns
        if column not in metadata
    ]

    n_features = len(
        features
    )

    columns = 2

    rows = int(
        np.ceil(
            n_features / columns
        )
    )

    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(14, 4 * rows),
    )

    axes = np.atleast_1d(
        axes
    ).flatten()

    non_apnea = df[
        df["label"] == 0
    ]

    apnea = df[
        df["label"] == 1
    ]

    for index, feature in enumerate(
        features
    ):

        ax = axes[
            index
        ]

        non_apnea_values = pd.to_numeric(
            non_apnea[feature],
            errors="coerce",
        ).dropna()

        apnea_values = pd.to_numeric(
            apnea[feature],
            errors="coerce",
        ).dropna()

        if len(non_apnea_values) > 0:

            ax.hist(
                non_apnea_values,
                bins=30,
                alpha=0.65,
                label="Non-apnea",
            )

        if len(apnea_values) > 0:

            ax.hist(
                apnea_values,
                bins=30,
                alpha=0.65,
                label="Apnea",
            )

        ax.set_title(
            feature
        )

        ax.set_xlabel(
            "Value"
        )

        ax.set_ylabel(
            "Frequency"
        )

        ax.legend(
            fontsize=8
        )

    # Hide unused subplot
    for index in range(
        n_features,
        len(axes),
    ):

        axes[
            index
        ].axis(
            "off"
        )

    fig.suptitle(
        "Physiological Feature Distributions: "
        "Apnea vs Non-Apnea",
        fontsize=16,
    )

    fig.tight_layout(
        rect=[
            0,
            0,
            1,
            0.97,
        ]
    )

    fig.savefig(
        FEATURE_DISTRIBUTION_FIGURE,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

# ============================================================
# VISUALIZATION 7
# FEATURE MISSINGNESS
# ============================================================

def plot_feature_missingness(
    df: pd.DataFrame,
) -> None:
    """
    Visualize missing values in the final ML-ready feature set.

    If no feature values are missing, generate an explicit
    no-missing-values figure instead of an empty chart.
    """

    metadata = {
        "record",
        "epoch_idx",
        "peak_source",
        "label",
    }

    features = [
        column
        for column in df.columns
        if column not in metadata
    ]

    missing_counts = (
        df[features]
        .apply(pd.to_numeric, errors="coerce")
        .isna()
        .sum()
    )

    total_missing = int(
        missing_counts.sum()
    )

    fig = plt.figure(
        figsize=(11, 6)
    )

    if total_missing == 0:

        plt.text(
            0.5,
            0.55,
            "No missing feature values",
            ha="center",
            va="center",
            fontsize=22,
            fontweight="bold",
            transform=plt.gca().transAxes,
        )

        plt.text(
            0.5,
            0.40,
            f"{len(df):,} ML-ready epochs × "
            f"{len(features)} features",
            ha="center",
            va="center",
            fontsize=14,
            transform=plt.gca().transAxes,
        )

        plt.text(
            0.5,
            0.25,
            "All feature values are finite and available "
            "for model training.",
            ha="center",
            va="center",
            fontsize=11,
            transform=plt.gca().transAxes,
        )

        plt.axis("off")

    else:

        missing_counts = (
            missing_counts[
                missing_counts > 0
            ]
            .sort_values(
                ascending=False
            )
        )

        bars = plt.bar(
            missing_counts.index,
            missing_counts.values,
        )

        plt.ylabel(
            "Missing values"
        )

        plt.xlabel(
            "Feature"
        )

        plt.title(
            "Feature Missingness — ML-Ready Dataset"
        )

        plt.xticks(
            rotation=45,
            ha="right",
        )

        for bar, value in zip(
            bars,
            missing_counts.values,
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
        FIGURES_DIR
        / "07_feature_missingness.png"
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    log.info(
        "Saved 07_feature_missingness.png"
    )


# ============================================================
# VISUALIZATION 08
# DATASET RECORD COVERAGE
# ============================================================

def plot_dataset_records(
    processed_df: pd.DataFrame,
) -> None:
    """
    Show the number of processed epochs contributed by
    each recording.
    """

    counts = (
        processed_df[
            "record"
        ]
        .value_counts()
        .sort_index()
    )

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    ax.bar(
        counts.index,
        counts.values,
    )

    ax.set_xlabel(
        "Recording"
    )

    ax.set_ylabel(
        "Processed epochs"
    )

    ax.set_title(
        "Processed Epoch Coverage Across Learning Records"
    )

    ax.tick_params(
        axis="x",
        rotation=90,
    )

    fig.tight_layout()

    fig.savefig(
        DATASET_RECORDS_FIGURE,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# VISUALIZATION 9
# PREPROCESSING SUMMARY
# ============================================================

def plot_preprocessing_summary(
    raw_stats: dict,
    processed_df: pd.DataFrame,
    ml_df: pd.DataFrame,
    ml_stats: dict,
) -> None:
    """
    Show the complete Phase 1 data reduction pipeline.

    The figure deliberately separates:
        raw candidate epochs
        low-quality epochs removed
        usable processed epochs
        invalid labels removed
        final ML-ready epochs

    This avoids implying that the already-filtered processed
    dataset is filtered a second time.
    """

    raw_epochs = int(
        raw_stats["raw_one_minute_epochs"]
    )

    processed_epochs = int(
        len(processed_df)
    )

    low_quality_removed = (
        raw_epochs
        - processed_epochs
    )

    invalid_labels_removed = int(
        ml_stats["invalid_label_rows_removed"]
    )

    invalid_feature_rows_removed = int(
        ml_stats["invalid_feature_rows_removed"]
    )

    ml_ready_epochs = int(
        len(ml_df)
    )

    # --------------------------------------------------------
    # Create figure
    # --------------------------------------------------------

    fig = plt.figure(
        figsize=(13, 7)
    )

    ax = plt.gca()

    ax.axis("off")

    plt.title(
        "Phase 1 ECG-to-ML Data Preparation",
        fontsize=20,
        pad=25,
    )

    # --------------------------------------------------------
    # Pipeline stages
    # --------------------------------------------------------

    stages = [
        (
            "RAW 1-MINUTE EPOCHS",
            f"{raw_epochs:,}",
            "PhysioNet learning recordings",
        ),
        (
            "USABLE PROCESSED EPOCHS",
            f"{processed_epochs:,}",
            "After ECG quality filtering",
        ),
        (
            "ML-READY LABELLED EPOCHS",
            f"{ml_ready_epochs:,}",
            "Valid labels + valid features",
        ),
    ]

    x_positions = [
        0.18,
        0.50,
        0.82,
    ]

    for x, (title, value, subtitle) in zip(
        x_positions,
        stages,
    ):

        plt.text(
            x,
            0.58,
            value,
            ha="center",
            va="center",
            fontsize=27,
            fontweight="bold",
            transform=ax.transAxes,
        )

        plt.text(
            x,
            0.45,
            title,
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            transform=ax.transAxes,
        )

        plt.text(
            x,
            0.38,
            subtitle,
            ha="center",
            va="center",
            fontsize=10,
            transform=ax.transAxes,
        )

    # --------------------------------------------------------
    # Arrows
    # --------------------------------------------------------

    ax.annotate(
        "",
        xy=(0.40, 0.58),
        xytext=(0.27, 0.58),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops=dict(
            arrowstyle="->",
            lw=2,
        ),
    )

    ax.annotate(
        "",
        xy=(0.72, 0.58),
        xytext=(0.59, 0.58),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops=dict(
            arrowstyle="->",
            lw=2,
        ),
    )

    # --------------------------------------------------------
    # Removal annotations
    # --------------------------------------------------------

    plt.text(
        0.335,
        0.70,
        f"− {low_quality_removed:,} low-quality epochs",
        ha="center",
        va="center",
        fontsize=11,
        transform=ax.transAxes,
    )

    removal_text = (
        f"− {invalid_labels_removed:,} invalid-label rows"
    )

    if invalid_feature_rows_removed > 0:
        removal_text += (
            f"\n− {invalid_feature_rows_removed:,} "
            "invalid-feature rows"
        )

    plt.text(
        0.665,
        0.70,
        removal_text,
        ha="center",
        va="center",
        fontsize=11,
        transform=ax.transAxes,
    )

    # --------------------------------------------------------
    # Dataset details
    # --------------------------------------------------------

    feature_count = int(
        ml_stats["feature_count"]
    )

    subject_count = int(
        ml_stats["subject_group_count"]
    )

    plt.text(
        0.50,
        0.18,
        (
            f"{feature_count} physiological features  •  "
            f"{subject_count} subject groups  •  "
            f"100 Hz ECG  •  60-second epochs"
        ),
        ha="center",
        va="center",
        fontsize=11,
        transform=ax.transAxes,
    )

    plt.text(
        0.50,
        0.10,
        (
            "Final dataset preserves record-level subject IDs "
            "for grouped subject-wise validation."
        ),
        ha="center",
        va="center",
        fontsize=10,
        transform=ax.transAxes,
    )

    plt.tight_layout()

    output_path = (
        FIGURES_DIR
        / "09_preprocessing_summary.png"
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    log.info(
        "Saved 09_preprocessing_summary.png"
    )


    # --------------------------------------------------------
    # Stages
    # --------------------------------------------------------

    stage_labels = [
        "Raw one-minute\nepochs",
        "Processed\nepochs",
        "After quality\nfilter",
        "ML-ready labelled\nepochs",
    ]

    stage_values = [
        raw_epochs,
        processed_epochs,
        processed_epochs,
        ml_ready_epochs,
    ]

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    bars = ax.bar(
        stage_labels,
        stage_values,
    )

    ax.set_ylabel(
        "Epoch count"
    )

    ax.set_title(
        "Phase 1 ECG-to-ML Data Preparation"
    )

    for bar, value in zip(
        bars,
        stage_values,
    ):

        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            bar.get_height(),
            f"{value:,}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # --------------------------------------------------------
    # Add transition information
    # --------------------------------------------------------

    subtitle = (
        f"Low-quality epochs removed: "
        f"{low_quality_removed:,}    |    "

        '''
        f"Invalid labels removed: "
        f"{invalid_label_rows:,}    |    "
        f"Invalid feature rows removed: "
        f"{invalid_feature_rows:,}"
        '''
    )

    fig.text(
        0.5,
        0.02,
        subtitle,
        ha="center",
        fontsize=9,
    )

    fig.tight_layout(
        rect=[
            0,
            0.06,
            1,
            1,
        ]
    )

    fig.savefig(
        PREPROCESSING_SUMMARY_FIGURE,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# WRITE TEXT REPORT
# ============================================================

def write_text_report(
    raw_stats: dict,
    processed_df: pd.DataFrame,
    ml_df: pd.DataFrame,
    ml_stats: dict,
    rpeak_summary: pd.DataFrame,
    rr_hrv_summary: pd.DataFrame,
) -> None:

    non_apnea = int(
        (
            ml_df["label"]
            == 0
        ).sum()
    )

    apnea = int(
        (
            ml_df["label"]
            == 1
        ).sum()
    )

    raw_epochs = int(
        raw_stats[
            "raw_one_minute_epochs"
        ]
    )

    processed_epochs = int(
        len(processed_df)
    )

    low_quality_removed = max(
        raw_epochs
        - processed_epochs,
        0,
    )

    invalid_labels = int(
        ml_stats[
            "invalid_label_rows_removed"
        ]
    )

    invalid_features = int(
        ml_stats[
            "invalid_feature_rows_removed"
        ]
    )

    report = f"""
============================================================
                 PHASE 1 COMPLETE
============================================================

PROJECT
------------------------------------------------------------
Lightweight ECG-Based Screening System for Early Sleep
Apnea Detection

Dataset:
PhysioNet Apnea-ECG

============================================================
DATASET SCOPE
============================================================

Learning records considered       : {raw_stats["records_successfully_loaded"]}
Duplicate handling                : c05 excluded as duplicate of c06

Sampling frequency                : 100 Hz
Epoch duration                    : 60 seconds

============================================================
RAW PHYSIONET DATA
============================================================

Total ECG samples                 : {raw_stats["total_samples"]:,}
Total recording duration          : {raw_stats["total_duration_minutes"]:,.2f} minutes
Raw one-minute epochs             : {raw_epochs:,}

============================================================
PROCESSED DATA
============================================================

Processed epoch dataset shape     : {processed_df.shape}
Processed records                 : {processed_df["record"].nunique()}
Processed epochs                  : {processed_epochs:,}

Low-quality epochs removed        : {low_quality_removed:,}

============================================================
R-PEAK PROCESSING
============================================================

R-peak summary file:
{RPEAK_SUMMARY_FILE}

Total processed epochs            : {int(rpeak_summary["total_epochs"].sum()):,}
Usable epochs                     : {int(rpeak_summary["usable_epochs"].sum()):,}
Unusable epochs                   : {int(rpeak_summary["unusable_epochs"].sum()):,}

Peak source distribution:
{processed_df["peak_source"].fillna("unknown").value_counts().to_string()}

============================================================
RR / HRV PROCESSING
============================================================

RR/HRV summary file:
{RR_HRV_SUMMARY_FILE}

Physiological features generated:
- n_beats
- mean_rr
- sdnn
- rmssd
- pnn50
- cvrr
- mean_hr
- min_rr
- max_rr
- rr_range
- edr_amp_std

============================================================
LABEL QUALITY
============================================================

Invalid / missing labels removed : {invalid_labels:,}
Invalid feature rows removed      : {invalid_features:,}

============================================================
ML-READY DATASET
============================================================

ML-ready dataset shape            : {ml_df.shape}
Feature count                     : {ml_stats["feature_count"]}
Subject groups                    : {ml_stats["subject_group_count"]}

Non-apnea epochs                  : {non_apnea:,}
Apnea epochs                      : {apnea:,}

Non-apnea percentage              : {non_apnea / max(len(ml_df), 1) * 100:.2f}%
Apnea percentage                  : {apnea / max(len(ml_df), 1) * 100:.2f}%

X shape                           : {tuple(ml_stats["X_shape"])}
y shape                           : {tuple(ml_stats["y_shape"])}
groups shape                      : {tuple(ml_stats["groups_shape"])}

============================================================
FEATURES
============================================================

{chr(10).join("• " + feature for feature in ml_stats["feature_columns"])}

============================================================
DATASET PREVIEW
============================================================

{ml_df.head(10).to_string(index=False)}

============================================================
OUTPUT FILES
============================================================

Epoch features:
{EPOCH_FILE}

ML-ready dataset:
{ML_DATASET_FILE}

Feature matrix:
{X_FILE}

Labels:
{Y_FILE}

Subject groups:
{GROUPS_FILE}

R-peak summary:
{RPEAK_SUMMARY_FILE}

RR/HRV summary:
{RR_HRV_SUMMARY_FILE}

Record statistics:
{RECORD_STATS_FILE}

Feature statistics:
{FEATURE_STATS_FILE}

Dataset summary:
{SUMMARY_JSON_FILE}

Phase 1 JSON summary:
{PHASE1_JSON_FILE}

Phase 1 text report:
{PHASE1_TXT_FILE}

Figures:
{FIGURES_DIR}

============================================================
PHASE 1 STATUS
============================================================

✓ PhysioNet records loaded
✓ ECG preprocessing completed
✓ R-peak information summarized
✓ RR interval / HRV features available
✓ Minute-level labels assigned
✓ Low-quality epochs accounted for
✓ Invalid labels removed
✓ ML-ready dataset generated
✓ X / y / subject groups generated
✓ R-peak summary generated
✓ RR/HRV summary generated
✓ Dataset statistics generated
✓ Phase-1 visualizations generated

NOTE
------------------------------------------------------------
The ML-ready dataset contains physiological features only.
Scaling and imputation are intentionally NOT performed here.
They must be fitted inside each subject-wise GroupKFold
training split during model evaluation to avoid data leakage.

============================================================
          PHASE 1 — DATA PIPELINE COMPLETE ✓
============================================================
"""

    PHASE1_TXT_FILE.write_text(
        report.strip(),
        encoding="utf-8",
    )


# ============================================================
# FINAL TERMINAL SUMMARY
# ============================================================

def print_terminal_summary(
    raw_stats: dict,
    processed_df: pd.DataFrame,
    ml_df: pd.DataFrame,
    ml_stats: dict,
    rpeak_summary: pd.DataFrame,
) -> None:

    raw_epochs = int(
        raw_stats[
            "raw_one_minute_epochs"
        ]
    )

    processed_epochs = int(
        len(processed_df)
    )

    low_quality_removed = max(
        raw_epochs
        - processed_epochs,
        0,
    )

    print(
        "\n"
        + "=" * 60
    )

    print(
        "                 PHASE 1 COMPLETE"
    )

    print(
        "=" * 60
    )

    print(
        "\nRAW PHYSIONET DATA"
    )

    print(
        "-" * 60
    )

    print(
        f"Learning records considered       : "
        f"{raw_stats['records_successfully_loaded']}"
    )

    print(
        f"Total raw ECG samples             : "
        f"{raw_stats['total_samples']:,}"
    )

    print(
        f"Total recording duration          : "
        f"{raw_stats['total_duration_minutes']:,.2f} minutes"
    )

    print(
        f"Raw one-minute epochs             : "
        f"{raw_epochs:,}"
    )

    print(
        "\nPROCESSED DATA"
    )

    print(
        "-" * 60
    )

    print(
        f"Processed dataset shape           : "
        f"{processed_df.shape}"
    )

    print(
        f"Usable epochs                     : "
        f"{processed_epochs:,}"
    )

    print(
        f"Low-quality epochs removed        : "
        f"{low_quality_removed:,}"
    )

    print(
        "\nR-PEAK SUMMARY"
    )

    print(
        "-" * 60
    )

    print(
        f"Records summarized               : "
        f"{len(rpeak_summary)}"
    )

    print(
        f"Total epochs                     : "
        f"{int(rpeak_summary['total_epochs'].sum()):,}"
    )

    print(
        f"Usable epochs                    : "
        f"{int(rpeak_summary['usable_epochs'].sum()):,}"
    )

    print(
        f"Unusable epochs                  : "
        f"{int(rpeak_summary['unusable_epochs'].sum()):,}"
    )

    print(
        "\nML-READY DATASET"
    )

    print(
        "-" * 60
    )

    print(
        f"ML-ready epochs                  : "
        f"{len(ml_df):,}"
    )

    print(
        f"Non-apnea epochs                 : "
        f"{ml_stats['non_apnea_epochs']:,}"
    )

    print(
        f"Apnea epochs                     : "
        f"{ml_stats['apnea_epochs']:,}"
    )

    print(
        f"Invalid labels removed           : "
        f"{ml_stats['invalid_label_rows_removed']:,}"
    )

    print(
        f"Feature count                    : "
        f"{ml_stats['feature_count']}"
    )

    print(
        f"Subject groups                   : "
        f"{ml_stats['subject_group_count']}"
    )

    print(
        f"X shape                          : "
        f"{tuple(ml_stats['X_shape'])}"
    )

    print(
        f"y shape                          : "
        f"{tuple(ml_stats['y_shape'])}"
    )

    print(
        f"groups shape                     : "
        f"{tuple(ml_stats['groups_shape'])}"
    )

    print(
        "\nPHASE 1 OUTPUTS"
    )

    print(
        "-" * 60
    )

    print(
        f"✓ {RPEAK_SUMMARY_FILE}"
    )

    print(
        f"✓ {RR_HRV_SUMMARY_FILE}"
    )

    print(
        f"✓ {ML_DATASET_FILE}"
    )

    print(
        f"✓ {X_FILE}"
    )

    print(
        f"✓ {Y_FILE}"
    )

    print(
        f"✓ {GROUPS_FILE}"
    )

    print(
        f"✓ {FIGURES_DIR}"
    )

    print(
        f"✓ {PHASE1_TXT_FILE}"
    )

    print(
        "\n"
        + "=" * 60
        + "\n"
        + "       PHASE 1 — DATA PIPELINE COMPLETE ✓"
        + "\n"
        + "=" * 60
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print(
        "\n"
        + "=" * 60
        + "\n"
        + "       FINALIZING PHASE 1 DATA PIPELINE"
        + "\n"
        + "=" * 60
    )

    # --------------------------------------------------------
    # Setup
    # --------------------------------------------------------

    create_directories()

    remove_obsolete_figures()

    # --------------------------------------------------------
    # Load existing processed data
    # --------------------------------------------------------

    processed_df = (
        load_processed_dataset()
    )

    # --------------------------------------------------------
    # Identify learning records
    # --------------------------------------------------------

    records = get_learning_records(
        keep_duplicate=False
    )

    log.info(
        "Learning records considered: %d",
        len(records),
    )

    # --------------------------------------------------------
    # Raw statistics
    # --------------------------------------------------------

    raw_stats = (
        calculate_raw_statistics(
            records
        )
    )

    # --------------------------------------------------------
    # R-peak summary
    # --------------------------------------------------------

    rpeak_summary = (
        create_rpeak_summary(
            processed_df
        )
    )

    # --------------------------------------------------------
    # RR / HRV summary
    # --------------------------------------------------------

    rr_hrv_summary = (
        create_rr_hrv_summary(
            processed_df
        )
    )

    # --------------------------------------------------------
    # Create ML-ready data
    # --------------------------------------------------------

    ml_df, ml_stats = (
        create_ml_ready_dataset(
            processed_df
        )
    )

    # --------------------------------------------------------
    # Feature statistics
    # --------------------------------------------------------

    create_feature_statistics(
        ml_df,
        ml_stats[
            "feature_columns"
        ],
    )

    # --------------------------------------------------------
    # Dataset summary
    # --------------------------------------------------------

    summary = (
        create_dataset_summary(
            processed_df=processed_df,
            ml_df=ml_df,
            raw_stats=raw_stats,
            ml_stats=ml_stats,
            rpeak_summary=rpeak_summary,
            rr_hrv_summary=rr_hrv_summary,
        )
    )

    SUMMARY_JSON_FILE.write_text(
        json.dumps(
            summary,
            indent=4,
        ),
        encoding="utf-8",
    )

    PHASE1_JSON_FILE.write_text(
        json.dumps(
            summary,
            indent=4,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Visualizations
    # --------------------------------------------------------

    log.info(
        "Generating Phase 1 visualizations..."
    )

    plot_raw_vs_cleaned_ecg(
        record_name="a01",
        seconds=10,
    )

    plot_rpeak_detection(
        record_name="a01",
        seconds=10,
    )

    plot_rr_tachogram(
        record_name="a01",
        epoch_index=0,
    )

    plot_class_distribution(
        ml_df
    )

    plot_apnea_burden_by_record(
        ml_df
    )

    plot_feature_distributions(
        ml_df
    )

    '''
    plot_dataset_quality(
        raw_stats=raw_stats,
        processed_df=processed_df,
        ml_df=ml_df,
        ml_stats=ml_stats,
    )
    '''

    plot_dataset_records(
        processed_df
    )

    plot_preprocessing_summary(
        raw_stats=raw_stats,
        processed_df=processed_df,
        ml_df=ml_df,
        ml_stats=ml_stats,
    )

    # --------------------------------------------------------
    # Text report
    # --------------------------------------------------------

    write_text_report(
        raw_stats=raw_stats,
        processed_df=processed_df,
        ml_df=ml_df,
        ml_stats=ml_stats,
        rpeak_summary=rpeak_summary,
        rr_hrv_summary=rr_hrv_summary,
    )

    # --------------------------------------------------------
    # Terminal summary
    # --------------------------------------------------------

    print_terminal_summary(
        raw_stats=raw_stats,
        processed_df=processed_df,
        ml_df=ml_df,
        ml_stats=ml_stats,
        rpeak_summary=rpeak_summary,
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()