from __future__ import annotations

import numpy as np
from scipy import signal as sp_signal


def compute_edr_proxy(ecg_epoch: np.ndarray, fs: int) -> float:
    """
    Compute a simple ECG-derived respiration (EDR) proxy.

    The proxy is based on variation in ECG peak amplitudes.
    It is used only as an ECG-derived respiratory feature and
    is not treated as a direct respiration measurement.
    """
    try:
        ecg_epoch = np.asarray(ecg_epoch, dtype=float)

        if ecg_epoch.size < 3:
            return np.nan

        peaks, _ = sp_signal.find_peaks(
            ecg_epoch,
            distance=max(1, int(0.3 * fs))
        )

        if len(peaks) < 2:
            return np.nan

        peak_values = ecg_epoch[peaks]

        return float(np.std(peak_values))

    except Exception:
        return np.nan