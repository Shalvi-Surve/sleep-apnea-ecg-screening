"""
Apnea-ECG (PhysioNet) preprocessing pipeline
=============================================

Source: https://physionet.org/content/apnea-ecg/1.0.0/

What this does, end to end
---------------------------
1. Downloads (or loads locally cached) records via the `wfdb` package.
2. Cleans the raw ECG: bandpass filter, powerline notch, baseline-wander
   removal, z-score normalization.
3. Detects R-peaks / RR-intervals (uses the dataset's own .qrs annotations
   when available, falls back to NeuroKit2's detector otherwise) and
   removes physiologically implausible RR intervals (artifact rejection).
4. Segments each recording into the same 1-minute epochs used by the
   official apnea annotations (.apn files: 'N' = normal, 'A' = apnea).
5. Extracts a standard HRV + ECG-derived-respiration feature set per
   1-minute epoch (this is the representation almost all Apnea-ECG
   literature trains on, since raw 100Hz waveforms are rarely fed
   directly into classical ML models).
6. Assembles a labeled feature table, respecting the dataset's own
   train/test split (a/b/c = learning set, x = test set), flags the
   known c05/c06 duplicate-recording issue, and reports class balance.
7. Fits a StandardScaler on the training split only, applies it to both
   splits, and writes everything to disk (CSV + a fitted scaler.pkl).

Usage
-----
    pip install wfdb neurokit2 scipy scikit-learn pandas numpy

    # Full run (downloads from PhysioNet the first time, caches locally)
    python preprocess_apnea_ecg.py --data-dir ./physionet_cache --out-dir ./processed

    # Quick smoke test with no internet / no real data required
    python preprocess_apnea_ecg.py --self-test --out-dir ./processed_selftest

Notes
-----
- Only the 35 "learning set" records (a01-a20, b01-b05, c01-c10) ship with
  .apn labels, so only those can be used for supervised training/validation.
  The 35 "test set" records (x01-x35) originally had labels withheld for
  the CinC 2000 challenge; PhysioNet has since published them in
  `test-dataset-annos.txt`. This script supports both.
- c05 and c06 are known to be the same underlying recording (c05 starts
  ~80s later than c06) - see PhysioNet's notes. By default this script
  drops c05 to avoid data leakage between "different" records that are
  actually the same patient segment.
"""

from __future__ import annotations

import argparse
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal as sp_signal

warnings.filterwarnings("ignore", category=FutureWarning)

# np.trapz was removed in NumPy >= 2.0 in favor of np.trapezoid; support both.
_trapz = getattr(np, "trapezoid", None) or np.trapz

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("apnea_ecg")

# --------------------------------------------------------------------------- #
# Dataset constants
# --------------------------------------------------------------------------- #

FS = 100  # Hz, native sampling rate of the Apnea-ECG signals
EPOCH_SEC = 60  # each apnea annotation covers a 1-minute epoch
EPOCH_LEN = FS * EPOCH_SEC  # 6000 samples

LEARNING_SET = (
    [f"a{i:02d}" for i in range(1, 21)]
    + [f"b{i:02d}" for i in range(1, 6)]
    + [f"c{i:02d}" for i in range(1, 11)]
)
TEST_SET = [f"x{i:02d}" for i in range(1, 36)]

# c05 and c06 are the same underlying recording (c05 begins ~80s later).
# Drop c05 by default to avoid leakage; override with --keep-duplicate.
KNOWN_DUPLICATE_RECORD = "c05"

# Physiologically plausible RR interval bounds (seconds) -> 33-200 bpm
RR_MIN_SEC, RR_MAX_SEC = 0.3, 1.8


# --------------------------------------------------------------------------- #
# 1. Loading
# --------------------------------------------------------------------------- #

@dataclass
class RawRecord:
    name: str
    ecg: np.ndarray            # raw ECG samples, shape (n_samples,)
    fs: int
    apn_labels: list[str] | None  # one 'A'/'N' per minute, or None if unlabeled
    qrs_samples: np.ndarray | None  # sample indices of QRS detections, if provided


def load_record(name: str, data_dir: Path, pn_dir: str = "apnea-ecg/1.0.0") -> RawRecord:
    """Load one record, downloading it via wfdb into data_dir if not cached."""
    import wfdb

    data_dir.mkdir(parents=True, exist_ok=True)
    local_hea = data_dir / f"{name}.hea"

    if not local_hea.exists():
        log.info(f"Downloading {name} from PhysioNet -> {data_dir}")
        wfdb.dl_files("apnea-ecg", str(data_dir), [
            f"{name}.dat", f"{name}.hea", f"{name}.apn", f"{name}.qrs"
        ])

    rec = wfdb.rdrecord(str(data_dir / name))
    ecg = rec.p_signal[:, 0].astype(np.float64)

    apn_labels = None
    apn_path = data_dir / f"{name}.apn"
    if apn_path.exists():
        try:
            ann = wfdb.rdann(str(data_dir / name), "apn")
            apn_labels = list(ann.symbol)
        except Exception as e:
            log.warning(f"{name}: could not read .apn annotations ({e})")

    qrs_samples = None
    qrs_path = data_dir / f"{name}.qrs"
    if qrs_path.exists():
        try:
            qrs_ann = wfdb.rdann(str(data_dir / name), "qrs")
            qrs_samples = np.asarray(qrs_ann.sample)
        except Exception as e:
            log.warning(f"{name}: could not read .qrs annotations ({e})")

    return RawRecord(name=name, ecg=ecg, fs=int(rec.fs), apn_labels=apn_labels,
                      qrs_samples=qrs_samples)


def load_test_labels_from_txt(path: Path) -> dict[str, list[str]]:
    """Parse PhysioNet's test-dataset-annos.txt into {record_name: [labels]}."""
    labels: dict[str, list[str]] = {}
    if not path.exists():
        return labels
    current = None
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("x") and len(line.split()) == 1:
            current = line
            labels[current] = []
        elif current is not None:
            labels[current].extend(list(line.replace(" ", "")))
    return labels


# --------------------------------------------------------------------------- #
# 2. Signal cleaning
# --------------------------------------------------------------------------- #

def bandpass_filter(x: np.ndarray, fs: int, low: float = 0.5, high: float = 40.0,
                     order: int = 4) -> np.ndarray:
    nyq = fs / 2.0
    b, a = sp_signal.butter(order, [low / nyq, high / nyq], btype="band")
    return sp_signal.filtfilt(b, a, x)


def notch_filter(x: np.ndarray, fs: int, freq: float = 50.0, q: float = 30.0) -> np.ndarray:
    nyq = fs / 2.0
    if freq / nyq >= 1.0:
        return x  # sampling rate too low for this notch to be meaningful
    b, a = sp_signal.iirnotch(freq / nyq, q)
    return sp_signal.filtfilt(b, a, x)


def remove_baseline_wander(x: np.ndarray, fs: int, cutoff: float = 0.5) -> np.ndarray:
    """High-pass to strip baseline wander below `cutoff` Hz (redundant with
    bandpass above but kept explicit/separate for clarity and reuse)."""
    nyq = fs / 2.0
    b, a = sp_signal.butter(2, cutoff / nyq, btype="high")
    return sp_signal.filtfilt(b, a, x)


def zscore(x: np.ndarray) -> np.ndarray:
    mu, sd = np.mean(x), np.std(x)
    return (x - mu) / sd if sd > 1e-8 else x - mu


def clean_ecg(raw: np.ndarray, fs: int) -> np.ndarray:
    x = bandpass_filter(raw, fs, 0.5, 40.0)
    x = notch_filter(x, fs, 50.0)
    x = zscore(x)
    return x


# --------------------------------------------------------------------------- #
# 3. QRS detection / RR intervals
# --------------------------------------------------------------------------- #

def detect_r_peaks(ecg_clean: np.ndarray, fs: int,
                    provided_qrs: np.ndarray | None = None) -> np.ndarray:
    """Prefer the dataset's own machine-generated QRS annotations (audited by
    nobody, but consistent); fall back to NeuroKit2 if unavailable."""
    if provided_qrs is not None and len(provided_qrs) > 5:
        return provided_qrs

    import neurokit2 as nk
    _, info = nk.ecg_peaks(ecg_clean, sampling_rate=fs, method="neurokit")
    return np.asarray(info["ECG_R_Peaks"])


def rr_intervals_from_peaks(r_peaks: np.ndarray, fs: int) -> tuple[np.ndarray, np.ndarray]:
    """Returns (rr_seconds, rr_peak_times_seconds) with physiologically
    implausible intervals removed (artifact rejection)."""
    if len(r_peaks) < 2:
        return np.array([]), np.array([])
    rr = np.diff(r_peaks) / fs
    t = r_peaks[1:] / fs
    valid = (rr >= RR_MIN_SEC) & (rr <= RR_MAX_SEC)
    return rr[valid], t[valid]


# --------------------------------------------------------------------------- #
# 4-5. Segmentation + per-epoch HRV feature extraction
# --------------------------------------------------------------------------- #

def hrv_time_domain_features(rr: np.ndarray) -> dict:
    if len(rr) < 2:
        return {k: np.nan for k in
                ["mean_rr", "sdnn", "rmssd", "pnn50", "cvrr", "mean_hr", "min_rr", "max_rr"]}
    diffs = np.diff(rr)
    return {
        "mean_rr": float(np.mean(rr)),
        "sdnn": float(np.std(rr, ddof=1)) if len(rr) > 1 else np.nan,
        "rmssd": float(np.sqrt(np.mean(diffs ** 2))) if len(diffs) else np.nan,
        "pnn50": float(np.mean(np.abs(diffs) > 0.05) * 100) if len(diffs) else np.nan,
        "cvrr": float(np.std(rr, ddof=1) / np.mean(rr)) if np.mean(rr) > 0 else np.nan,
        "mean_hr": float(60.0 / np.mean(rr)) if np.mean(rr) > 0 else np.nan,
        "min_rr": float(np.min(rr)),
        "max_rr": float(np.max(rr)),
    }


def hrv_freq_domain_features(rr: np.ndarray, rr_times: np.ndarray) -> dict:
    """LF/HF power via Lomb-Scargle (handles the unevenly-sampled RR series
    directly, no need to resample -> interpolation artifacts)."""
    out = {"lf_power": np.nan, "hf_power": np.nan, "lf_hf_ratio": np.nan}
    if len(rr) < 4:
        return out
    try:
        freqs = np.linspace(0.003, 0.4, 200)
        rr_detrended = rr - np.mean(rr)
        pgram = sp_signal.lombscargle(rr_times, rr_detrended, freqs * 2 * np.pi, normalize=True)
        lf_mask = (freqs >= 0.04) & (freqs < 0.15)
        hf_mask = (freqs >= 0.15) & (freqs < 0.4)
        lf = float(_trapz(pgram[lf_mask], freqs[lf_mask]))
        hf = float(_trapz(pgram[hf_mask], freqs[hf_mask]))
        out["lf_power"] = lf
        out["hf_power"] = hf
        out["lf_hf_ratio"] = lf / hf if hf > 1e-8 else np.nan
    except Exception as e:
        log.debug(f"LF/HF computation failed for one epoch: {e}")
    return out


def edr_amplitude_feature(ecg_epoch: np.ndarray, fs: int) -> dict:
    """Crude ECG-derived-respiration proxy: R-peak amplitude modulation
    variance, a cheap stand-in for chest-effort signals that are only
    available for 8 of the 70 records."""
    try:
        peak_vals = ecg_epoch[sp_signal.find_peaks(ecg_epoch, distance=int(0.3 * fs))[0]]
        return {"edr_amp_std": float(np.std(peak_vals)) if len(peak_vals) > 1 else np.nan}
    except Exception:
        return {"edr_amp_std": np.nan}


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


# --------------------------------------------------------------------------- #
# 6. Dataset assembly
# --------------------------------------------------------------------------- #

def assemble_dataset(record_names: list[str], data_dir: Path,
                      test_labels: dict[str, list[str]] | None = None,
                      n_jobs: int = 1) -> pd.DataFrame:
    """Loads + featurizes each record. With n_jobs > 1, records are processed
    in parallel threads — this is I/O + numpy/scipy bound work (file reads,
    filtfilt, QRS detection), so threads help without the pickling overhead
    a process pool would add, and each record is independent so there's no
    shared-state risk."""

    def _process_one(name: str):
        rec = load_record(name, data_dir)
        if rec.apn_labels is None and test_labels and name in test_labels:
            rec.apn_labels = test_labels[name]
        df = build_epoch_features(rec)
        return name, df

    frames = []
    if n_jobs == 1:
        for name in record_names:
            try:
                name, df = _process_one(name)
            except Exception as e:
                log.error(f"Skipping {name}: failed to load ({e})")
                continue
            frames.append(df)
            _log_record_summary(name, df)
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=n_jobs) as pool:
            futures = {pool.submit(_process_one, name): name for name in record_names}
            for fut in as_completed(futures):
                name = futures[fut]
                try:
                    name, df = fut.result()
                except Exception as e:
                    log.error(f"Skipping {name}: failed to load ({e})")
                    continue
                frames.append(df)
                _log_record_summary(name, df)

    if not frames:
        raise RuntimeError("No records were successfully loaded.")
    return pd.concat(frames, ignore_index=True)


def _log_record_summary(name: str, df: pd.DataFrame) -> None:
    n_apnea = int((df["label"] == 1).sum())
    log.info(f"{name}: {len(df):4d} epochs | {n_apnea:4d} apnea "
             f"({100 * n_apnea / max(len(df), 1):.1f}%)")


def drop_missing_beats(df: pd.DataFrame, min_beats: int = 20) -> pd.DataFrame:
    """Epochs with too few detected beats (signal dropout / lead-off) can't
    produce trustworthy HRV features - drop rather than impute blindly."""
    before = len(df)
    df = df[df["n_beats"] >= min_beats].reset_index(drop=True)
    log.info(f"Dropped {before - len(df)} epochs with <{min_beats} detected beats "
              f"({before} -> {len(df)})")
    return df


def impute_and_scale(train_df: pd.DataFrame, test_df: pd.DataFrame,
                      feature_cols: list[str]):
    """Median-impute then z-scale, fit only on train. Columns that are
    entirely NaN in the training split (e.g. lf/hf power when RR series
    are too short for a spectral estimate) can't be median-imputed, so
    they're constant-filled with 0 first to keep column count stable."""
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    train_feats = train_df[feature_cols].copy()
    all_nan_cols = [c for c in feature_cols if train_feats[c].isna().all()]
    if all_nan_cols:
        log.warning(f"Columns entirely NaN in training data, filling with 0: {all_nan_cols}")
        train_feats[all_nan_cols] = 0.0

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    train_X = imputer.fit_transform(train_feats)
    train_X = scaler.fit_transform(train_X)

    test_X = None
    if test_df is not None and len(test_df):
        test_feats = test_df[feature_cols].copy()
        if all_nan_cols:
            test_feats[all_nan_cols] = test_feats[all_nan_cols].fillna(0.0)
        test_X = imputer.transform(test_feats)
        test_X = scaler.transform(test_X)

    return train_X, test_X, imputer, scaler


# --------------------------------------------------------------------------- #
# Self-test: synthetic data, no internet/PhysioNet access required
# --------------------------------------------------------------------------- #

def _make_synthetic_record(name: str, minutes: int = 20, apnea_rate: float = 0.3,
                            seed: int = 0) -> RawRecord:
    """Generates a fake but structurally-realistic ECG-like record + labels,
    purely so the pipeline mechanics (filtering, QRS detection, epoching,
    feature extraction, scaling) can be validated without network access."""
    rng = np.random.default_rng(seed)
    n_samples = minutes * EPOCH_LEN
    t = np.arange(n_samples) / FS

    labels = [("A" if rng.random() < apnea_rate else "N") for _ in range(minutes)]

    ecg = np.zeros(n_samples)
    for i, lab in enumerate(labels):
        start, end = i * EPOCH_LEN, (i + 1) * EPOCH_LEN
        hr_bpm = rng.uniform(75, 95) if lab == "A" else rng.uniform(55, 70)
        rr_mean = 60.0 / hr_bpm
        beat_times = []
        tcur = rng.uniform(0, rr_mean)
        while tcur < EPOCH_SEC:
            beat_times.append(tcur)
            jitter = rng.normal(0, 0.02 if lab == "N" else 0.05)
            tcur += max(0.3, rr_mean + jitter)
        local_t = t[start:end] - t[start]
        seg = np.zeros_like(local_t)
        for bt in beat_times:
            seg += np.exp(-((local_t - bt) ** 2) / (2 * 0.008 ** 2))
        ecg[start:end] = seg

    ecg += rng.normal(0, 0.05, size=n_samples)  # sensor noise
    ecg += 0.3 * np.sin(2 * np.pi * 0.2 * t)  # baseline wander
    ecg += 0.1 * np.sin(2 * np.pi * 50 * t)  # powerline interference

    return RawRecord(name=name, ecg=ecg, fs=FS, apn_labels=labels, qrs_samples=None)


def run_self_test(out_dir: Path):
    log.info("Running self-test with synthetic data (no PhysioNet access needed)...")
    train_records = [_make_synthetic_record(f"syn_train_{i:02d}", minutes=30, seed=i)
                      for i in range(4)]
    test_records = [_make_synthetic_record(f"syn_test_{i:02d}", minutes=15, seed=100 + i)
                     for i in range(2)]

    train_df = pd.concat([build_epoch_features(r) for r in train_records], ignore_index=True)
    test_df = pd.concat([build_epoch_features(r) for r in test_records], ignore_index=True)

    train_df = drop_missing_beats(train_df, min_beats=10)
    test_df = drop_missing_beats(test_df, min_beats=10)

    feature_cols = [c for c in train_df.columns if c not in ("record", "epoch_idx", "label")]
    train_X, test_X, imputer, scaler = impute_and_scale(train_df, test_df, feature_cols)

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(train_X, columns=feature_cols).assign(
        label=train_df["label"].values, record=train_df["record"].values
    ).to_csv(out_dir / "train_features_selftest.csv", index=False)
    pd.DataFrame(test_X, columns=feature_cols).assign(
        label=test_df["label"].values, record=test_df["record"].values
    ).to_csv(out_dir / "test_features_selftest.csv", index=False)

    log.info(f"Self-test OK. Train shape={train_X.shape}, Test shape={test_X.shape}")
    log.info(f"Train class balance:\n{train_df['label'].value_counts(normalize=True)}")
    log.info(f"Wrote outputs to {out_dir}/")
    return train_df, test_df


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("./physionet_cache"),
                         help="Where raw PhysioNet files are cached/downloaded")
    parser.add_argument("--out-dir", type=Path, default=Path("./processed"),
                         help="Where processed feature CSVs + scaler are written")
    parser.add_argument("--records", nargs="*", default=None,
                         help="Subset of record names to process (default: all)")
    parser.add_argument("--keep-duplicate", action="store_true",
                         help="Keep c05 despite it duplicating c06")
    parser.add_argument("--min-beats", type=int, default=20,
                         help="Drop epochs with fewer detected beats than this")
    parser.add_argument("--n-jobs", type=int, default=4,
                         help="Parallel worker threads for loading/featurizing records")
    parser.add_argument("--self-test", action="store_true",
                         help="Run on synthetic data only; no internet/PhysioNet needed")
    args = parser.parse_args()

    if args.self_test:
        run_self_test(args.out_dir)
        return

    learning_records = list(LEARNING_SET)
    if not args.keep_duplicate and KNOWN_DUPLICATE_RECORD in learning_records:
        learning_records.remove(KNOWN_DUPLICATE_RECORD)
        log.info(f"Dropping {KNOWN_DUPLICATE_RECORD} (duplicate of c06) - "
                 f"use --keep-duplicate to override")

    test_records = list(TEST_SET)
    if args.records:
        wanted = set(args.records)
        learning_records = [r for r in learning_records if r in wanted]
        test_records = [r for r in test_records if r in wanted]

    test_labels = load_test_labels_from_txt(args.data_dir / "test-dataset-annos.txt")

    log.info(f"Building TRAIN set from {len(learning_records)} learning-set records...")
    train_df = assemble_dataset(learning_records, args.data_dir, n_jobs=args.n_jobs)

    test_df = pd.DataFrame()
    if test_records:
        log.info(f"Building TEST set from {len(test_records)} test-set records...")
        test_df = assemble_dataset(test_records, args.data_dir, test_labels=test_labels,
                                    n_jobs=args.n_jobs)

    train_df = drop_missing_beats(train_df, min_beats=args.min_beats)
    if len(test_df):
        test_df = drop_missing_beats(test_df, min_beats=args.min_beats)

    feature_cols = [c for c in train_df.columns if c not in ("record", "epoch_idx", "label")]
    train_X, test_X, imputer, scaler = impute_and_scale(train_df, test_df, feature_cols)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(train_X, columns=feature_cols).assign(
        label=train_df["label"].values, record=train_df["record"].values
    ).to_csv(args.out_dir / "train_features.csv", index=False)

    if len(test_df):
        pd.DataFrame(test_X, columns=feature_cols).assign(
            label=test_df["label"].values, record=test_df["record"].values
        ).to_csv(args.out_dir / "test_features.csv", index=False)

    import joblib
    joblib.dump({"imputer": imputer, "scaler": scaler, "feature_cols": feature_cols},
                args.out_dir / "preprocessing_pipeline.pkl")

    log.info(f"Done. Train: {train_df.shape}, Test: {test_df.shape if len(test_df) else '(none)'}")
    if train_df["label"].notna().any():
        log.info(f"Train class balance:\n{train_df['label'].value_counts(normalize=True)}")


if __name__ == "__main__":
    main()
