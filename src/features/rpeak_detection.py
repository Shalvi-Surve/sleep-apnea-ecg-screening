from __future__ import annotations

import numpy as np
from scipy import signal


def detect_r_peaks(
    ecg_clean: np.ndarray,
    fs: int,
    provided_qrs: np.ndarray | None = None,
) -> np.ndarray:
    """
    Detect R-peaks from the cleaned ECG.

    A lightweight signal-processing detector is used as the
    primary method. Provided PhysioNet QRS annotations are
    retained only as optional reference information.
    """

    if len(ecg_clean) < fs * 2:
        return np.array([], dtype=int)

    # Emphasize rapid QRS transitions.
    derivative = np.diff(
        ecg_clean,
        prepend=ecg_clean[0],
    )

    squared = derivative ** 2

    # Moving-window integration.
    window = max(int(0.15 * fs), 1)

    integrated = np.convolve(
        squared,
        np.ones(window) / window,
        mode="same",
    )

    # Adaptive threshold based on the signal distribution.
    threshold = np.mean(integrated) + 0.5 * np.std(integrated)

    candidate_peaks, _ = signal.find_peaks(
        integrated,
        height=threshold,
        distance=int(0.30 * fs),
    )

    if len(candidate_peaks) == 0:
        return np.array([], dtype=int)

    # Refine each candidate to the largest ECG amplitude
    # in a short neighborhood.
    search_radius = int(0.08 * fs)

    refined = []

    for peak in candidate_peaks:
        start = max(0, peak - search_radius)
        end = min(
            len(ecg_clean),
            peak + search_radius + 1,
        )

        local_peak = start + int(
            np.argmax(
                np.abs(ecg_clean[start:end])
            )
        )

        refined.append(local_peak)

    refined = np.asarray(
        refined,
        dtype=int,
    )

    # Remove duplicates while maintaining order.
    refined = np.unique(refined)

    # Final refractory-period cleanup.
    if len(refined) > 1:
        keep = [refined[0]]

        min_distance = int(0.30 * fs)

        for peak in refined[1:]:
            if peak - keep[-1] >= min_distance:
                keep.append(peak)

        refined = np.asarray(
            keep,
            dtype=int,
        )

    return refined