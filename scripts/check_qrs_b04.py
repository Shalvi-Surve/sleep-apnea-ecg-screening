import wfdb
import numpy as np
from pathlib import Path

from src.preprocessing.signal_cleaning import clean_ecg
from src.features.rpeak_detection import detect_r_peaks


p = Path("data/raw")

r = wfdb.rdrecord(str(p / "b04"))
qrs = wfdb.rdann(str(p / "b04"), "qrs")

ecg = clean_ecg(
    r.p_signal[:, 0],
    r.fs,
)

pred = detect_r_peaks(
    ecg,
    r.fs,
    None,
)

ref = np.asarray(qrs.sample)

tolerance = int(0.10 * r.fs)

matched_ref = set()
matched_errors = []

for peak in pred:
    distances = np.abs(ref - peak)

    if len(distances) == 0:
        continue

    idx = int(np.argmin(distances))

    if distances[idx] <= tolerance and idx not in matched_ref:
        matched_ref.add(idx)
        matched_errors.append(
            int(distances[idx])
        )

print("Predicted peaks:", len(pred))
print("Reference QRS:", len(ref))
print("Matched within 100 ms:", len(matched_errors))
print(
    "Match rate vs predicted:",
    len(matched_errors) / len(pred)
)
print(
    "Match rate vs reference:",
    len(matched_errors) / len(ref)
)

if matched_errors:
    print(
        "Median timing error (ms):",
        np.median(matched_errors) * 1000 / r.fs,
    )
    print(
        "Mean timing error (ms):",
        np.mean(matched_errors) * 1000 / r.fs,
    )
