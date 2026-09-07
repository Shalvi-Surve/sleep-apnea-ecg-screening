from src.data.load_ecg import load_record
from src.features.edr import compute_edr_proxy
from src.features.hrv_features import compute_hrv_features
from src.features.rpeak_detection import detect_r_peaks
from src.features.rr_intervals import rr_intervals_from_peaks
from src.features.windowing import (
    EPOCH_SEC,
    get_epoch_boundaries,
    assign_apnea_label,
)
from src.preprocessing.signal_cleaning import clean_ecg

log = logging.getLogger(__name__)


def build_epoch_features(record) -> pd.DataFrame:
    """
    Convert one ECG recording into one feature row per
    one-minute epoch.
    """

    ecg_clean = clean_ecg(
        record.ecg,
        record.fs,
    )

    r_peaks = detect_r_peaks(
        ecg_clean,
        record.fs,
        record.qrs_samples,
    )

    boundaries = get_epoch_boundaries(
        len(record.ecg),
        record.fs,
    )

    rows = []

    for epoch_idx, (start, end) in enumerate(
        boundaries
    ):
        epoch_peaks = r_peaks[
            (r_peaks >= start)
            & (r_peaks < end)
        ]

        local_peaks = epoch_peaks - start

        rr, rr_times = rr_intervals_from_peaks(
            local_peaks,
            record.fs,
        )

        features = {
            "record": record.name,
            "epoch_idx": epoch_idx,
            "n_beats": int(len(local_peaks)),
        }

        features.update(
            compute_hrv_features(rr)
        )

        features["edr_amp_std"] = compute_edr_proxy(
            ecg_clean[start:end],
            record.fs,
        )

        features["label"] = assign_apnea_label(
            record.apn_labels,
            epoch_idx,
        )

        rows.append(features)

    return pd.DataFrame(rows)


def assemble_dataset(
    record_names: list[str],
    data_dir: str | Path,
    test_labels: dict[str, list[str]] | None = None,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """
    Load and process multiple ECG recordings.

    Each recording remains identifiable through the `record`
    column so GroupKFold can perform subject-wise validation.
    """

    frames = []

    for name in record_names:
        try:
            record = load_record(
                name,
                data_dir,
            )

            if (
                record.apn_labels is None
                and test_labels is not None
                and name in test_labels
            ):
                record.apn_labels = test_labels[name]

            df = build_epoch_features(
                record
            )

            frames.append(df)

            _log_record_summary(
                name,
                df,
            )

        except Exception as exc:
            log.exception(
                "Failed to process %s: %s",
                name,
                exc,
            )

    if not frames:
        raise RuntimeError(
            "No records were successfully processed."
        )

    return pd.concat(
        frames,
        ignore_index=True,
    )


def drop_missing_beats(
    df: pd.DataFrame,
    min_beats: int = 20,
) -> pd.DataFrame:
    """
    Remove epochs with insufficient detected beats.
    """

    before = len(df)

    result = df[
        df["n_beats"] >= min_beats
    ].reset_index(drop=True)

    log.info(
        "Dropped %d epochs with fewer than %d beats "
        "(%d -> %d)",
        before - len(result),
        min_beats,
        before,
        len(result),
    )

    return result


def _log_record_summary(
    name: str,
    df: pd.DataFrame,
) -> None:
    """
    Log basic per-record processing information.
    """

    n_apnea = int(
        (df["label"] == 1).sum()
    )

    total = len(df)

    log.info(
        "%s: %4d epochs | %4d apnea (%.1f%%)",
        name,
        total,
        n_apnea,
        100 * n_apnea / max(total, 1),
    )