from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.load_ecg import load_record
from src.features.edr import compute_edr_proxy
from src.features.hrv_features import compute_hrv_features
from src.features.rpeak_detection import detect_r_peaks
from src.features.rr_intervals import rr_intervals_from_peaks
from src.features.windowing import (
    EPOCH_SEC,
    get_epoch_boundaries,
    assign_apnea_label,
)
from src.preprocessing.signal_cleaning import clean_ecg


log = logging.getLogger(__name__)

# R-peak quality-control settings
MIN_BEATS = 20
PADDING_SEC = 3


def _get_epoch_r_peaks(
    ecg_clean: np.ndarray,
    start: int,
    end: int,
    fs: int,
    qrs_samples: np.ndarray | None = None,
) -> tuple[np.ndarray, str]:
    """
    Detect R-peaks for one epoch using a padded local ECG segment.

    Primary method:
        NeuroKit2-based detector through detect_r_peaks().

    Fallback:
        PhysioNet database-provided QRS annotations are used only
        when the independent detector produces fewer than MIN_BEATS.

    Returns:
        peaks_in_epoch, source

    source is one of:
        "neurokit"
        "qrs_fallback"
        "unusable"
    """

    pad = int(PADDING_SEC * fs)

    left = max(0, start - pad)
    right = min(len(ecg_clean), end + pad)

    segment = ecg_clean[left:right]

    try:
        detected = detect_r_peaks(
            segment,
            fs,
            None,
        )
    except Exception as exc:
        log.warning(
            "R-peak detection failed for epoch %d-%d: %s",
            start,
            end,
            exc,
        )
        detected = np.array([], dtype=int)

    detected = np.asarray(
        detected,
        dtype=int,
    )

    if detected.size:
        global_peaks = detected + left

        epoch_peaks = global_peaks[
            (global_peaks >= start)
            & (global_peaks < end)
        ]

        epoch_peaks = np.unique(
            epoch_peaks
        )
    else:
        epoch_peaks = np.array(
            [],
            dtype=int,
        )

    if len(epoch_peaks) >= MIN_BEATS:
        return epoch_peaks, "neurokit"

    # ---------------------------------------------------------
    # QRS fallback
    # ---------------------------------------------------------

    if qrs_samples is not None:
        qrs_samples = np.asarray(
            qrs_samples,
            dtype=int,
        )

        qrs_epoch = qrs_samples[
            (qrs_samples >= start)
            & (qrs_samples < end)
        ]

        qrs_epoch = np.unique(
            qrs_epoch
        )

        if len(qrs_epoch) >= MIN_BEATS:
            return qrs_epoch, "qrs_fallback"

    return (
        np.array([], dtype=int),
        "unusable",
    )


def build_epoch_features(record) -> pd.DataFrame:
    """
    Convert one ECG recording into one feature row per
    one-minute epoch.

    R-peaks are detected independently for each padded epoch.
    Epochs with insufficient primary detections may use the
    database-provided QRS annotations as a fallback.
    """

    ecg_clean = clean_ecg(
        record.ecg,
        record.fs,
    )

    boundaries = get_epoch_boundaries(
        len(record.ecg),
        record.fs,
    )

    rows = []

    source_counts = {
        "neurokit": 0,
        "qrs_fallback": 0,
        "unusable": 0,
    }

    for epoch_idx, (start, end) in enumerate(
        boundaries
    ):
        epoch_peaks, peak_source = _get_epoch_r_peaks(
            ecg_clean=ecg_clean,
            start=start,
            end=end,
            fs=record.fs,
            qrs_samples=record.qrs_samples,
        )

        source_counts[peak_source] += 1

        local_peaks = (
            epoch_peaks - start
        )

        rr, rr_times = rr_intervals_from_peaks(
            local_peaks,
            record.fs,
        )

        features = {
            "record": record.name,
            "epoch_idx": epoch_idx,
            "n_beats": int(
                len(local_peaks)
            ),
            "peak_source": peak_source,
        }

        features.update(
            compute_hrv_features(rr)
        )

        features["edr_amp_std"] = compute_edr_proxy(
            ecg_clean[start:end],
            record.fs,
        )

        features["label"] = assign_apnea_label(
            record.apn_labels,
            epoch_idx,
        )

        rows.append(features)

    log.info(
        "%s R-peak sources: NeuroKit=%d | "
        "QRS fallback=%d | unusable=%d",
        record.name,
        source_counts["neurokit"],
        source_counts["qrs_fallback"],
        source_counts["unusable"],
    )

    return pd.DataFrame(rows)


def assemble_dataset(
    record_names: list[str],
    data_dir: str | Path,
    test_labels: dict[str, list[str]] | None = None,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """
    Load and process multiple ECG recordings.

    Each recording remains identifiable through the `record`
    column so GroupKFold can perform subject-wise validation.
    """

    frames = []

    for name in record_names:
        try:
            record = load_record(
                name,
                data_dir,
            )

            if (
                record.apn_labels is None
                and test_labels is not None
                and name in test_labels
            ):
                record.apn_labels = test_labels[name]

            df = build_epoch_features(
                record
            )

            frames.append(df)

            _log_record_summary(
                name,
                df,
            )

        except Exception as exc:
            log.exception(
                "Failed to process %s: %s",
                name,
                exc,
            )

    if not frames:
        raise RuntimeError(
            "No records were successfully processed."
        )

    return pd.concat(
        frames,
        ignore_index=True,
    )


def drop_missing_beats(
    df: pd.DataFrame,
    min_beats: int = MIN_BEATS,
) -> pd.DataFrame:
    """
    Remove epochs with insufficient detected/fallback beats.
    """

    before = len(df)

    result = df[
        df["n_beats"] >= min_beats
    ].reset_index(drop=True)

    log.info(
        "Dropped %d epochs with fewer than %d beats "
        "(%d -> %d)",
        before - len(result),
        min_beats,
        before,
        len(result),
    )

    return result


def _log_record_summary(
    name: str,
    df: pd.DataFrame,
) -> None:
    """
    Log basic per-record processing information.
    """

    n_apnea = int(
        (df["label"] == 1).sum()
    )

    total = len(df)

    log.info(
        "%s: %4d epochs | %4d apnea (%.1f%%)",
        name,
        total,
        n_apnea,
        100 * n_apnea / max(total, 1),
    )