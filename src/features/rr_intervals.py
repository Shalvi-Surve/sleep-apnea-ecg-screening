from __future__ import annotations

import numpy as np


RR_MIN_SEC = 0.30
RR_MAX_SEC = 1.80


def rr_intervals_from_peaks(
    r_peaks: np.ndarray,
    fs: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert R-peak locations into RR intervals.

    Returns:
        rr_seconds
        rr_times_seconds
    """

    if len(r_peaks) < 2:
        return (
            np.array([], dtype=float),
            np.array([], dtype=float),
        )

    rr = np.diff(r_peaks) / fs
    times = r_peaks[1:] / fs

    valid = (
        (rr >= RR_MIN_SEC)
        & (rr <= RR_MAX_SEC)
    )

    rr = rr[valid]
    times = times[valid]

    # Robust artifact rejection using MAD.
    if len(rr) >= 5:
        median = np.median(rr)
        mad = np.median(np.abs(rr - median))

        if mad > 1e-8:
            robust_z = (
                np.abs(rr - median)
                / (1.4826 * mad)
            )

            keep = robust_z <= 5.0

            rr = rr[keep]
            times = times[keep]

    return rr, times