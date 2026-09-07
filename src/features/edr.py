import numpy as np
import pandas as pd
from scipy import signal as sp_signal


def compute_edr_proxy(ecg_epoch: np.ndarray, fs: int) -> float:
    """
    Compute a simple ECG-derived respiration (EDR) proxy.

    The proxy is based on variation in ECG R-peak amplitudes.
    It is used only as an ECG-derived respiratory feature and
    is not treated as a direct respiration measurement.
    """
    try:
        peaks, _ = sp_signal.find_peaks(
            ecg_epoch,
            distance=int(0.3 * fs)
        )

        if len(peaks) < 2:
            return np.nan

        peak_values = ecg_epoch[peaks]

        return float(np.std(peak_values))

    except Exception:
        return np.nan

def build_epoch_features(record: RawRecord) -> pd.DataFrame:
    """Cleans the signal, detects beats, and returns one feature row per
    1-minute epoch aligned to the .apn labels (if present)."""
    ecg_clean = clean_ecg(record.ecg, record.fs)
    r_peaks = detect_r_peaks(ecg_clean, record.fs, record.qrs_samples)

    n_epochs = len(record.ecg) // EPOCH_LEN
    rows = []
    for i in range(n_epochs):
        start, end = i * EPOCH_LEN, (i + 1) * EPOCH_LEN
        epoch_peaks = r_peaks[(r_peaks >= start) & (r_peaks < end)] - start
        rr, rr_t = rr_intervals_from_peaks(epoch_peaks, record.fs)

        feats = {"record": record.name, "epoch_idx": i}
        feats.update(hrv_time_domain_features(rr))
        feats.update(hrv_freq_domain_features(rr, rr_t))
        feats.update(edr_amplitude_feature(ecg_clean[start:end], record.fs))
        feats["n_beats"] = len(epoch_peaks)

        if record.apn_labels is not None and i < len(record.apn_labels):
            feats["label"] = 1 if record.apn_labels[i] == "A" else 0
        else:
            feats["label"] = np.nan

        rows.append(feats)

    return pd.DataFrame(rows)

