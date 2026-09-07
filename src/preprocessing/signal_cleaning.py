from __future__ import annotations

import numpy as np
from scipy import signal


def bandpass_filter(
    x: np.ndarray,
    fs: int,
    low: float = 0.5,
    high: float = 40.0,
    order: int = 4,
) -> np.ndarray:
    """
    Zero-phase Butterworth bandpass filter.
    """
    nyquist = fs / 2.0

    if high >= nyquist:
        high = nyquist * 0.95

    b, a = signal.butter(
        order,
        [low / nyquist, high / nyquist],
        btype="band",
    )

    return signal.filtfilt(b, a, x)


def notch_filter(
    x: np.ndarray,
    fs: int,
    freq: float = 50.0,
    q: float = 30.0,
) -> np.ndarray:
    """
    Apply a powerline notch filter when the target frequency
    is below the Nyquist frequency.

    At fs=100 Hz, 50 Hz is exactly Nyquist, so the notch
    is skipped.
    """
    nyquist = fs / 2.0

    if freq >= nyquist:
        return x

    b, a = signal.iirnotch(
        freq / nyquist,
        q,
    )

    return signal.filtfilt(b, a, x)


def remove_baseline_wander(
    x: np.ndarray,
    fs: int,
    cutoff: float = 0.5,
) -> np.ndarray:
    """
    Remove low-frequency baseline drift.

    This is optional because the bandpass filter already
    removes frequencies below 0.5 Hz.
    """
    nyquist = fs / 2.0

    b, a = signal.butter(
        2,
        cutoff / nyquist,
        btype="high",
    )

    return signal.filtfilt(b, a, x)


def clean_ecg(
    raw: np.ndarray,
    fs: int,
) -> np.ndarray:
    """
    Complete ECG cleaning pipeline:
        bandpass -> notch (when applicable) -> z-score normalization
    """
    x = bandpass_filter(
        raw,
        fs,
        low=0.5,
        high=40.0,
    )

    x = notch_filter(
        x,
        fs,
        freq=50.0,
    )

    mean = np.mean(x)
    std = np.std(x)

    if std > 1e-8:
        x = (x - mean) / std
    else:
        x = x - mean

    return x