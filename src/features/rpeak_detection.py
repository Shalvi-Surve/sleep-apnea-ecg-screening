from __future__ import annotations

import logging

import neurokit2 as nk
import numpy as np
from scipy import signal

log = logging.getLogger(__name__)


MIN_RR_SEC = 0.30
MAX_RR_SEC = 2.00


def _neurokit_r_peaks(ecg: np.ndarray, fs: int) -> np.ndarray:
    """Detect candidate R-peaks using NeuroKit2."""

    ecg = np.asarray(ecg, dtype=float)

    if ecg.size < fs * 5:
        return np.array([], dtype=int)

    cleaned = nk.ecg_clean(
        ecg,
        sampling_rate=int(fs),
        method="neurokit",
    )

    _, info = nk.ecg_peaks(
        cleaned,
        sampling_rate=int(fs),
        method="neurokit",
    )

    peaks = np.asarray(
        info.get("ECG_R_Peaks", []),
        dtype=int,
    )

    peaks = peaks[
        (peaks >= 0) &
        (peaks < len(ecg))
    ]

    return np.unique(peaks)


def _filter_refractory_peaks(
    peaks: np.ndarray,
    ecg: np.ndarray,
    fs: int,
) -> np.ndarray:
    """
    Remove physiologically implausible peak-to-peak detections.

    A minimum R-R interval of 300 ms corresponds to a maximum
    heart rate of approximately 200 bpm.

    When multiple candidate peaks occur within the refractory
    period, retain the candidate with the greatest local ECG
    amplitude.
    """

    if len(peaks) <= 1:
        return peaks.astype(int)

    peaks = np.sort(
        np.asarray(peaks, dtype=int)
    )

    min_distance = int(
        MIN_RR_SEC * fs
    )

    filtered = [int(peaks[0])]

    for candidate in peaks[1:]:
        candidate = int(candidate)

        if candidate - filtered[-1] >= min_distance:
            filtered.append(candidate)
            continue

        # Two candidates are too close together.
        # Keep the one with greater absolute ECG amplitude.
        previous = filtered[-1]

        previous_amp = abs(
            float(ecg[previous])
        )

        candidate_amp = abs(
            float(ecg[candidate])
        )

        if candidate_amp > previous_amp:
            filtered[-1] = candidate

    return np.asarray(
        filtered,
        dtype=int,
    )


def _filter_rr_outliers(
    peaks: np.ndarray,
    fs: int,
) -> np.ndarray:
    """
    Remove isolated implausible RR intervals.

    This is deliberately conservative. The main purpose is to
    remove obvious detection artifacts rather than aggressively
    edit physiological variability.
    """

    if len(peaks) < 3:
        return peaks.astype(int)

    rr = np.diff(peaks) / fs

    valid = (
        (rr >= MIN_RR_SEC) &
        (rr <= MAX_RR_SEC)
    )

    keep = np.ones(
        len(peaks),
        dtype=bool,
    )

    # A peak participates in two neighboring RR intervals.
    # Remove a peak when either neighboring interval is
    # physiologically impossible.
    for i, is_valid in enumerate(valid):
        if not is_valid:
            if i + 1 < len(keep):
                keep[i + 1] = False

    return peaks[keep]


def _is_plausible(
    peaks: np.ndarray,
    signal_length: int,
    fs: int,
) -> bool:
    """Check whether the resulting peak sequence is usable."""

    if peaks is None or len(peaks) < 5:
        return False

    duration_minutes = (
        signal_length / fs / 60.0
    )

    if duration_minutes <= 0:
        return False

    peaks_per_minute = (
        len(peaks) / duration_minutes
    )

    return (
        30.0 <= peaks_per_minute <= 150.0
    )


def _scipy_fallback_r_peaks(
    ecg: np.ndarray,
    fs: int,
) -> np.ndarray:
    """Conservative SciPy QRS fallback."""

    ecg = np.asarray(
        ecg,
        dtype=float,
    )

    if ecg.size < fs * 5:
        return np.array([], dtype=int)

    low = 5.0
    high = min(
        25.0,
        0.45 * fs,
    )

    b, a = signal.butter(
        2,
        [low, high],
        btype="bandpass",
        fs=fs,
    )

    filtered = signal.filtfilt(
        b,
        a,
        ecg,
    )

    derivative = np.diff(
        filtered,
        prepend=filtered[0],
    )

    squared = derivative ** 2

    integration_window = max(
        1,
        int(0.15 * fs),
    )

    integrated = np.convolve(
        squared,
        np.ones(
            integration_window
        ) / integration_window,
        mode="same",
    )

    threshold = np.percentile(
        integrated,
        90,
    )

    peaks, _ = signal.find_peaks(
        integrated,
        height=threshold,
        distance=max(
            1,
            int(MIN_RR_SEC * fs),
        ),
    )

    return np.asarray(
        peaks,
        dtype=int,
    )


def detect_r_peaks(
    ecg: np.ndarray,
    fs: int,
    qrs_samples: np.ndarray | None = None,
) -> np.ndarray:
    """
    Detect R-peaks directly from a single-lead ECG.

    NeuroKit2 is used as the primary detector. Detected peaks
    are subsequently passed through physiological refractory
    and RR-interval filtering.

    PhysioNet QRS annotations are NOT used as the primary
    detection source. They remain available for optional
    quality-control/reference analysis.
    """

    ecg = np.asarray(
        ecg,
        dtype=float,
    )

    if ecg.ndim != 1:
        ecg = np.squeeze(ecg)

    if ecg.ndim != 1:
        raise ValueError(
            "ECG input must be one-dimensional."
        )

    if ecg.size == 0:
        return np.array([], dtype=int)

    # ---------------------------------------------------------
    # Primary: NeuroKit2
    # ---------------------------------------------------------
    try:
        candidates = _neurokit_r_peaks(
            ecg,
            fs,
        )

        filtered = _filter_refractory_peaks(
            candidates,
            ecg,
            fs,
        )

        filtered = _filter_rr_outliers(
            filtered,
            fs,
        )

        if _is_plausible(
            filtered,
            len(ecg),
            fs,
        ):
            log.debug(
                "NeuroKit2 + physiological filtering: "
                "%d peaks.",
                len(filtered),
            )

            return filtered

        log.warning(
            "Filtered NeuroKit2 result was implausible: "
            "%d peaks.",
            len(filtered),
        )

    except Exception as exc:
        log.warning(
            "NeuroKit2 detection failed: %s",
            exc,
        )

    # ---------------------------------------------------------
    # Fallback: SciPy
    # ---------------------------------------------------------
    try:
        candidates = _scipy_fallback_r_peaks(
            ecg,
            fs,
        )

        filtered = _filter_refractory_peaks(
            candidates,
            ecg,
            fs,
        )

        filtered = _filter_rr_outliers(
            filtered,
            fs,
        )

        if _is_plausible(
            filtered,
            len(ecg),
            fs,
        ):
            log.debug(
                "SciPy fallback: %d peaks.",
                len(filtered),
            )

            return filtered

    except Exception as exc:
        log.warning(
            "SciPy fallback failed: %s",
            exc,
        )

    log.error(
        "Unable to obtain a plausible R-peak sequence."
    )

    return np.array([], dtype=int)