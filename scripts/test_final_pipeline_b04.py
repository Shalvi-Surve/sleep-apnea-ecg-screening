import wfdb
from pathlib import Path

from src.data.load_ecg import load_record
from src.preprocessing.preprocessing_pipeline import build_epoch_features


data_dir = Path("data/raw")

record = load_record(
    "b04",
    data_dir,
)

df = build_epoch_features(record)

print()
print("Shape:", df.shape)
print()
print("Peak source counts:")
print(df["peak_source"].value_counts())
print()
print("Beat statistics:")
print(df["n_beats"].describe())
print()
print("Unusable epochs:")
print((df["peak_source"] == "unusable").sum())
print()
print("First 20 rows:")
print(
    df[
        [
            "epoch_idx",
            "n_beats",
            "peak_source",
            "mean_rr",
            "sdnn",
            "rmssd",
            "mean_hr",
            "label",
        ]
    ].head(20).to_string(index=False)
)
