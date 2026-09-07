import wfdb
import numpy as np
from pathlib import Path
from src.preprocessing.signal_cleaning import clean_ecg
from src.features.rpeak_detection import detect_r_peaks

p = Path("data/raw")

r = wfdb.rdrecord(str(p / "b04"))
ecg = r.p_signal[:, 0]
fs = int(r.fs)

epoch_len = fs * 60
pad = fs * 3

counts = []

for epoch_idx, start in enumerate(range(0, len(ecg) - epoch_len + 1, epoch_len)):
    left = max(0, start - pad)
    right = min(len(ecg), start + epoch_len + pad)

    segment = clean_ecg(ecg[left:right], fs)

    peaks = detect_r_peaks(
        segment,
        fs,
        None,
    )

    global_peaks = peaks + left

    inside = global_peaks[
        (global_peaks >= start) &
        (global_peaks < start + epoch_len)
    ]

    counts.append(len(inside))

counts = np.asarray(counts)

print("Total epochs:", len(counts))
print("Epochs with 0 beats:", int(np.sum(counts == 0)))
print("Min beats:", int(np.min(counts)))
print("Median beats:", float(np.median(counts)))
print("Mean beats:", float(np.mean(counts)))
print("Max beats:", int(np.max(counts)))

print()
print("First 10 epoch counts:")
print(counts[:10])

print()
print("First 20 epoch counts:")
print(counts[:20])
