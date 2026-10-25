from __future__ import annotations

from typing import Iterable

import numpy as np


# PhysioNet Apnea-ECG uses 100 Hz ECG
FS = 100

# Apnea annotations are provided per minute
EPOCH_SEC = 60


def get_epoch_boundaries(
    signal_length: int,
    fs: int = FS,
    epoch_sec: int = EPOCH_SEC,
) -> list[tuple[int, int]]:
    """
    Split an ECG recording into consecutive non-overlapping epochs.

    Parameters
    ----------
    signal_length : int
        Number of ECG samples.
    fs : int
        Sampling frequency in Hz.
    epoch_sec : int
        Epoch duration in seconds.

    Returns
    -------
    list of (start_sample, end_sample)
    """
    epoch_len = int(fs * epoch_sec)

    if signal_length <= 0 or epoch_len <= 0:
        return []

    boundaries = []

    for start in range(0, signal_length, epoch_len):
        end = min(start + epoch_len, signal_length)

        # Ignore incomplete final epochs.
        if end - start < epoch_len:
            break

        boundaries.append((start, end))

    return boundaries


def assign_apnea_label(
    apnea_labels: Iterable | None,
    epoch_idx: int,
) -> float:
    """
    Assign the PhysioNet apnea annotation to an epoch.

    The .apn annotation is provided at one label per minute.
    Labels are interpreted as:
        1 -> apnea
        0 -> non-apnea

    Returns NaN when a label is unavailable.
    """
    if apnea_labels is None:
        return np.nan

    try:
        labels = list(apnea_labels)

        if epoch_idx < 0 or epoch_idx >= len(labels):
            return np.nan

        value = labels[epoch_idx]

        if isinstance(value, str):
            value = value.strip().upper()

            if value == "A":
                return 1.0

            if value in {"N", "0"}:
                return 0.0

            return np.nan

        if value in (0, 1):
            return float(value)

        return np.nan

    except Exception:
        return np.nan