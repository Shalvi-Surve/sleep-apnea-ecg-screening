from __future__ import annotations

import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

from scipy.signal import butter, filtfilt, find_peaks


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

RAW_DIR = BASE_DIR / "data" / "raw" / "apnea-ecg-test"
OUT_DIR = BASE_DIR / "results" / "experiments"

RAW_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = (
    "https://physionet.org/files/apnea-ecg/1.0.0/"
)

FS = 100
EPOCH_SECONDS = 60
WINDOW_LENGTH = 5
N_FEATURES = 11

TEST_RECORDS = [
    f"x{i:02d}" for i in range(1, 36)
]

FEATURE_COLUMNS = [
    "mean_rr",
    "sdnn",
    "rmssd",
    "pnn50",
    "mean_hr",
    "min_rr",
    "max_rr",
    "rr_range",
    "cvrr",
    "n_beats",
    "edr_amp_std",
]


# ============================================================
# DOWNLOAD
# ============================================================

def download_file(record: str, extension: str):

    destination = (
        RAW_DIR /
        f"{record}.{extension}"
    )

    if destination.exists() and destination.stat().st_size > 0:
        return destination

    url = (
        f"{BASE_URL}"
        f"{record}.{extension}"
    )

    print(
        f"Downloading {record}.{extension} ..."
    )

    try:

        urllib.request.urlretrieve(
            url,
            destination,
        )

    except Exception as exc:

        if destination.exists():
            destination.unlink()

        raise RuntimeError(
            f"Failed downloading {url}\n"
            f"{exc}"
        )

    return destination


def download_record(record: str):

    download_file(
        record,
        "hea",
    )

    download_file(
        record,
        "dat",
    )

    # Public test annotations.
    # If unavailable, the script still creates
    # feature data and uses -1 labels.
    try:

        download_file(
            record,
            "apn",
        )

    except Exception:

        print(
            f"WARNING: {record}.apn unavailable."
        )


# ============================================================
# ANNOTATIONS
# ============================================================

def load_apnea_labels(record: str):

    annotation_path = (
        RAW_DIR /
        f"{record}.apn"
    )

    if not annotation_path.exists():

        return {}

    try:

        annotation = wfdb.rdann(
            str(
                RAW_DIR / record
            ),
            "apn",
        )

    except Exception as exc:

        print(
            f"WARNING: Could not read "
            f"{record}.apn: {exc}"
        )

        return {}

    labels = {}

    for sample, symbol in zip(
        annotation.sample,
        annotation.symbol,
    ):

        minute = int(
            round(
                float(sample) / FS / 60.0
            )
        )

        symbol = str(
            symbol
        ).upper()

        if symbol == "A":
            labels[minute] = 1

        elif symbol in {
            "N",
            "O",
        }:
            labels[minute] = 0

    return labels


# ============================================================
# SIGNAL PROCESSING
# ============================================================

def bandpass_filter(
    signal,
    fs=100,
    lowcut=0.5,
    highcut=40.0,
):

    nyquist = fs / 2.0

    highcut = min(
        highcut,
        nyquist - 1.0,
    )

    low = lowcut / nyquist
    high = highcut / nyquist

    b, a = butter(
        4,
        [low, high],
        btype="band",
    )

    return filtfilt(
        b,
        a,
        signal,
    )


def clean_ecg(signal):

    x = np.asarray(
        signal,
        dtype=np.float64,
    )

    if len(x) == 0:
        return x

    finite = np.isfinite(x)

    if not finite.all():

        median = (
            np.nanmedian(x[finite])
            if finite.any()
            else 0.0
        )

        x = np.nan_to_num(
            x,
            nan=median,
            posinf=median,
            neginf=median,
        )

    if len(x) >= FS * 10:

        try:

            x = bandpass_filter(
                x,
                FS,
            )

        except Exception:
            pass

    std = np.std(x)

    if std > 1e-8:

        x = (
            x - np.mean(x)
        ) / std

    else:

        x = x - np.mean(x)

    return x


# ============================================================
# R-PEAK DETECTION
# ============================================================

def detect_r_peaks(signal):

    x = np.asarray(
        signal,
        dtype=np.float64,
    )

    if len(x) < FS * 10:

        return np.array(
            [],
            dtype=int,
        )

    derivative = np.diff(
        x,
        prepend=x[0],
    )

    squared = derivative ** 2

    integration_window = int(
        0.12 * FS
    )

    kernel = np.ones(
        integration_window,
        dtype=np.float64,
    ) / integration_window

    integrated = np.convolve(
        squared,
        kernel,
        mode="same",
    )

    threshold = np.percentile(
        integrated,
        85,
    )

    candidates, _ = find_peaks(
        integrated,
        height=threshold,
        distance=int(
            0.30 * FS
        ),
    )

    if len(candidates) == 0:

        return candidates

    refined = []

    radius = int(
        0.08 * FS
    )

    for candidate in candidates:

        left = max(
            0,
            candidate - radius,
        )

        right = min(
            len(x),
            candidate + radius + 1,
        )

        local_index = np.argmax(
            np.abs(
                x[left:right]
            )
        )

        refined.append(
            left + local_index
        )

    return np.unique(
        np.asarray(
            refined,
            dtype=int,
        )
    )


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(
    ecg,
    r_peaks,
):

    rr = np.diff(
        r_peaks
    ) / FS

    rr = rr[
        (rr >= 0.30)
        &
        (rr <= 2.00)
    ]

    if len(rr) < 2:

        return None

    mean_rr = float(
        np.mean(rr)
    )

    sdnn = float(
        np.std(
            rr,
            ddof=1,
        )
    )

    diff_rr = np.diff(
        rr
    )

    rmssd = float(
        np.sqrt(
            np.mean(
                diff_rr ** 2
            )
        )
    )

    pnn50 = float(
        np.mean(
            np.abs(diff_rr) > 0.05
        ) * 100
    )

    mean_hr = float(
        60.0 / mean_rr
    )

    min_rr = float(
        np.min(rr)
    )

    max_rr = float(
        np.max(rr)
    )

    rr_range = (
        max_rr - min_rr
    )

    cvrr = (
        sdnn / mean_rr
        if mean_rr > 0
        else 0.0
    )

    n_beats = int(
        len(r_peaks)
    )

    amplitudes = []

    radius = int(
        0.04 * FS
    )

    for peak in r_peaks:

        left = max(
            0,
            peak - radius,
        )

        right = min(
            len(ecg),
            peak + radius + 1,
        )

        if right > left:

            amplitudes.append(
                float(
                    np.max(
                        np.abs(
                            ecg[
                                left:right
                            ]
                        )
                    )
                )
            )

    if len(amplitudes) > 1:

        edr_amp_std = float(
            np.std(
                amplitudes,
                ddof=1,
            )
        )

    else:

        edr_amp_std = 0.0

    values = {

        "mean_rr": mean_rr,

        "sdnn": sdnn,

        "rmssd": rmssd,

        "pnn50": pnn50,

        "mean_hr": mean_hr,

        "min_rr": min_rr,

        "max_rr": max_rr,

        "rr_range": rr_range,

        "cvrr": cvrr,

        "n_beats": n_beats,

        "edr_amp_std": edr_amp_std,
    }

    if not all(
        np.isfinite(
            list(values.values())
        )
    ):

        return None

    if not (
        30 <= mean_hr <= 220
    ):

        return None

    if not (
        20 <= n_beats <= 180
    ):

        return None

    return values


# ============================================================
# PROCESS ONE RECORD
# ============================================================

def process_record(record):

    print()
    print(
        "=" * 70
    )

    print(
        f"PROCESSING {record.upper()}"
    )

    print(
        "=" * 70
    )

    download_record(
        record
    )

    try:

        wf = wfdb.rdrecord(
            str(
                RAW_DIR / record
            )
        )

    except Exception as exc:

        print(
            f"ERROR reading {record}: {exc}"
        )

        return None

    signal = np.asarray(
        wf.p_signal[:, 0],
        dtype=np.float64,
    )

    samples_per_epoch = (
        FS * EPOCH_SECONDS
    )

    number_of_epochs = (
        len(signal)
        // samples_per_epoch
    )

    labels = load_apnea_labels(
        record
    )

    print(
        f"Samples       : {len(signal):,}"
    )

    print(
        f"Raw epochs    : {number_of_epochs}"
    )

    print(
        f"Labels found  : {len(labels)}"
    )

    rows = []

    for epoch in range(
        number_of_epochs
    ):

        start = (
            epoch *
            samples_per_epoch
        )

        end = (
            start +
            samples_per_epoch
        )

        raw_epoch = signal[
            start:end
        ]

        cleaned = clean_ecg(
            raw_epoch
        )

        r_peaks = detect_r_peaks(
            cleaned
        )

        features = extract_features(
            cleaned,
            r_peaks,
        )

        if features is None:
            continue

        label = labels.get(
            epoch,
            -1,
        )

        rows.append({

            "record": record,

            "epoch": epoch,

            "label": label,

            **features,
        })

    if not rows:

        print(
            "NO VALID EPOCHS"
        )

        return None

    df = pd.DataFrame(
        rows
    )

    print(
        f"Valid epochs : {len(df)}"
    )

    return df


# ============================================================
# TEMPORAL WINDOWING
# ============================================================

def make_windows(
    epoch_df,
):

    windows = []

    labels = []

    groups = []

    metadata = []

    for record, group in epoch_df.groupby(
        "record",
        sort=False,
    ):

        group = group.sort_values(
            "epoch"
        ).reset_index(
            drop=True
        )

        epoch_numbers = group[
            "epoch"
        ].to_numpy()

        features = group[
            FEATURE_COLUMNS
        ].to_numpy(
            dtype=np.float32
        )

        epoch_labels = group[
            "label"
        ].to_numpy(
            dtype=np.int64
        )

        for i in range(
            len(group)
            - WINDOW_LENGTH
            + 1
        ):

            selected_epochs = (
                epoch_numbers[
                    i:
                    i + WINDOW_LENGTH
                ]
            )

            # Require five consecutive minutes.
            if not np.all(
                np.diff(
                    selected_epochs
                ) == 1
            ):

                continue

            window = features[
                i:
                i + WINDOW_LENGTH
            ]

            center_index = (
                i +
                WINDOW_LENGTH // 2
            )

            center_label = int(
                epoch_labels[
                    center_index
                ]
            )

            windows.append(
                window
            )

            labels.append(
                center_label
            )

            groups.append(
                record
            )

            metadata.append({

                "record": record,

                "start_epoch": int(
                    selected_epochs[0]
                ),

                "center_epoch": int(
                    selected_epochs[
                        WINDOW_LENGTH // 2
                    ]
                ),

                "end_epoch": int(
                    selected_epochs[-1]
                ),

                "label": center_label,
            })

    return (
        np.asarray(
            windows,
            dtype=np.float32,
        ),
        np.asarray(
            labels,
            dtype=np.int64,
        ),
        np.asarray(
            groups
        ),
        pd.DataFrame(
            metadata
        ),
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "=" * 70
    )

    print(
        "PHYSIONET APNEA-ECG X01-X35 TEST PREPARATION"
    )

    print(
        "=" * 70
    )

    all_epochs = []

    failed = []

    start_time = time.time()

    for record in TEST_RECORDS:

        try:

            df = process_record(
                record
            )

            if df is not None:

                all_epochs.append(
                    df
                )

            else:

                failed.append(
                    record
                )

        except Exception as exc:

            print()
            print(
                f"ERROR {record}: {exc}"
            )

            failed.append(
                record
            )

    if not all_epochs:

        raise RuntimeError(
            "No X01-X35 records were processed."
        )

    epoch_df = pd.concat(
        all_epochs,
        ignore_index=True,
    )

    print()
    print(
        "=" * 70
    )

    print(
        "CREATING 5-EPOCH TEMPORAL WINDOWS"
    )

    print(
        "=" * 70
    )

    X, y, groups, metadata = (
        make_windows(
            epoch_df
        )
    )

    print(
        f"X shape       : {X.shape}"
    )

    print(
        f"y shape       : {y.shape}"
    )

    print(
        f"Subjects      : {len(np.unique(groups))}"
    )

    valid_labels = (
        y >= 0
    )

    print(
        f"Valid labels  : {valid_labels.sum()}"
    )

    print(
        f"Unknown       : {(y < 0).sum()}"
    )

    if valid_labels.any():

        print(
            f"Class 0      : "
            f"{(y[valid_labels] == 0).sum()}"
        )

        print(
            f"Class 1      : "
            f"{(y[valid_labels] == 1).sum()}"
        )

    # ========================================================
    # SAVE
    # ========================================================

    epoch_df.to_csv(
        OUT_DIR /
        "test_epoch_features.csv",
        index=False,
    )

    metadata.to_csv(
        OUT_DIR /
        "test_temporal_metadata.csv",
        index=False,
    )

    np.save(
        OUT_DIR /
        "test_temporal_X.npy",
        X,
    )

    np.save(
        OUT_DIR /
        "test_temporal_y.npy",
        y,
    )

    np.save(
        OUT_DIR /
        "test_temporal_groups.npy",
        groups,
    )

    records = pd.DataFrame(
        {
            "record": sorted(
                np.unique(
                    groups
                )
            )
        }
    )

    records.to_csv(
        OUT_DIR /
        "test_records.csv",
        index=False,
    )

    print()
    print(
        "=" * 70
    )

    print(
        "TEST DATASET READY"
    )

    print(
        "=" * 70
    )

    print(
        f"Processed records : "
        f"{len(np.unique(groups))}"
    )

    print(
        f"Temporal windows  : "
        f"{len(X)}"
    )

    print(
        f"Elapsed           : "
        f"{time.time() - start_time:.1f}s"
    )

    if failed:

        print()
        print(
            "FAILED RECORDS:"
        )

        print(
            ", ".join(
                failed
            )
        )

    print()
    print(
        "NEXT COMMAND:"
    )

    print(
        "python scripts/run_test_inference.py"
    )


if __name__ == "__main__":
    main()