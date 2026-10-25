from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class RawRecord:
    """
    Container for one Apnea-ECG recording.
    """

    name: str
    ecg: np.ndarray
    fs: int
    apn_labels: list[str] | None = None
    qrs_samples: np.ndarray | None = None


def load_record(
    name: str,
    data_dir: str | Path,
) -> RawRecord:
    """
    Load one PhysioNet Apnea-ECG record from local files.

    Expected files:
        <name>.hea
        <name>.dat

    Optional:
        <name>.apn
        <name>.qrs
    """
    import wfdb

    data_dir = Path(data_dir)

    record_path = data_dir / name

    rec = wfdb.rdrecord(str(record_path))

    ecg = rec.p_signal[:, 0].astype(np.float64)

    apn_labels = None
    apn_path = data_dir / f"{name}.apn"

    if apn_path.exists():
        try:
            ann = wfdb.rdann(str(record_path), "apn")
            apn_labels = list(ann.symbol)
        except Exception:
            apn_labels = None

    qrs_samples = None
    qrs_path = data_dir / f"{name}.qrs"

    if qrs_path.exists():
        try:
            qrs_ann = wfdb.rdann(str(record_path), "qrs")
            qrs_samples = np.asarray(qrs_ann.sample)
        except Exception:
            qrs_samples = None

    fs = int(rec.fs)

    if fs != 100:
        raise ValueError(
            f"{name}: expected sampling rate 100 Hz, got {fs} Hz"
        )

    return RawRecord(
        name=name,
        ecg=ecg,
        fs=fs,
        apn_labels=apn_labels,
        qrs_samples=qrs_samples,
    )