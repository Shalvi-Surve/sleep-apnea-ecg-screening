from __future__ import annotations

import numpy as np


FS = 100
EPOCH_SECONDS = 60
EPOCH_SAMPLES = FS * EPOCH_SECONDS


def create_epoch_boundaries(
    signal_length: int,
    fs: int = FS,
) -> list[tuple[int, int]]:
    """
    Divide an ECG recording into complete 60-second epochs.
    """

    epoch_samples = fs * EPOCH_SECONDS

    n_epochs = signal_length // epoch_samples

    return [
        (
            i * epoch_samples,
            (i + 1) * epoch_samples,
        )
        for i in range(n_epochs)
    ]


def assign_apnea_label(
    epoch_index: int,
    labels: list[str] | None,
) -> float:
    """
    Convert PhysioNet minute-level labels to binary labels.

    A = 1 (apnea)
    N = 0 (non-apnea)
    unknown/missing = NaN
    """

    if labels is None:
        return np.nan

    if epoch_index >= len(labels):
        return np.nan

    label = str(labels[epoch_index]).strip().upper()

    if label == "A":
        return 1.0

    if label == "N":
        return 0.0

    return np.nan

def drop_missing_beats(df: pd.DataFrame, min_beats: int = 20) -> pd.DataFrame:
    """Epochs with too few detected beats (signal dropout / lead-off) can't
    produce trustworthy HRV features - drop rather than impute blindly."""
    before = len(df)
    df = df[df["n_beats"] >= min_beats].reset_index(drop=True)
    log.info(f"Dropped {before - len(df)} epochs with <{min_beats} detected beats "
              f"({before} -> {len(df)})")
    return df