import wfdb
import numpy as np
from pathlib import Path

from src.preprocessing.signal_cleaning import clean_ecg
from src.features.rpeak_detection import detect_r_peaks

p = Path("data/raw")

r = wfdb.rdrecord(str(p / "b04"))
qrs = wfdb.rdann(str(p / "b04"), "qrs")

ecg = r.p_signal[:, 0]
fs = int(r.fs)

epoch_len = fs * 60
pad = fs * 3

ref = np.asarray(qrs.sample)

print("Epoch | NeuroKit | QRS reference")
print("-" * 35)

for epoch_idx, start in enumerate(
    range(0, len(ecg) - epoch_len + 1, epoch_len)
):

    left = max(0, start - pad)
    right = min(len(ecg), start + epoch_len + pad)

    segment = clean_ecg(ecg[left:right], fs)

    peaks = detect_r_peaks(
        segment,
        fs,
        None,
    )

    global_peaks = peaks + left

    nk_inside = global_peaks[
        (global_peaks >= start) &
        (global_peaks < start + epoch_len)
    ]

    qrs_inside = ref[
        (ref >= start) &
        (ref < start + epoch_len)
    ]

    if len(nk_inside) < 20:
        print(
            f"{epoch_idx:5d} | "
            f"{len(nk_inside):8d} | "
            f"{len(qrs_inside):13d}"
        )
