from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "n_beats",
    "mean_rr",
    "sdnn",
    "rmssd",
    "pnn50",
    "cvrr",
    "mean_hr",
    "min_rr",
    "max_rr",
    "rr_range",
    "edr_amp_std",
]

WINDOW_SIZE = 5


def load_ml_ready_dataset(
    path: str | Path = "data/processed/ml_ready_dataset.csv",
) -> pd.DataFrame:

    df = pd.read_csv(path)

    required = (
        ["record", "epoch_idx", "label"]
        + FEATURE_COLUMNS
    )

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    df = df.copy()

    df["epoch_idx"] = df["epoch_idx"].astype(int)
    df["label"] = df["label"].astype(int)

    df = df.sort_values(
        ["record", "epoch_idx"]
    ).reset_index(drop=True)

    return df


def build_temporal_windows(
    df: pd.DataFrame,
    window_size: int = WINDOW_SIZE,
):
    """
    Build subject-wise consecutive temporal windows.

    A window contains `window_size` consecutive one-minute
    epochs from the same recording.

    The label corresponds to the center epoch.
    """

    if window_size % 2 == 0:
        raise ValueError(
            "window_size must be odd."
        )

    half = window_size // 2

    X = []
    y = []
    groups = []
    metadata = []

    for record, record_df in df.groupby(
        "record",
        sort=True,
    ):

        record_df = record_df.sort_values(
            "epoch_idx"
        ).reset_index(drop=True)

        for i in range(
            half,
            len(record_df) - half,
        ):

            window = record_df.iloc[
                i - half : i + half + 1
            ]

            epoch_ids = (
                window["epoch_idx"]
                .to_numpy()
            )

            expected = np.arange(
                epoch_ids[0],
                epoch_ids[0] + window_size,
            )

            if not np.array_equal(
                epoch_ids,
                expected,
            ):
                continue

            values = window[
                FEATURE_COLUMNS
            ].to_numpy(
                dtype=np.float32
            )

            if not np.isfinite(values).all():
                continue

            center_label = int(
                record_df.iloc[i]["label"]
            )

            X.append(values)
            y.append(center_label)
            groups.append(record)

            metadata.append(
                {
                    "record": record,
                    "center_epoch": int(
                        record_df.iloc[i][
                            "epoch_idx"
                        ]
                    ),
                }
            )

    if not X:
        raise RuntimeError(
            "No valid temporal windows were created."
        )

    return (
        np.stack(X),
        np.asarray(y, dtype=np.int64),
        np.asarray(groups),
        pd.DataFrame(metadata),
    )


def main():

    df = load_ml_ready_dataset()

    X, y, groups, metadata = (
        build_temporal_windows(df)
    )

    print("\nTEMPORAL DATASET")
    print("-" * 50)

    print("Original epochs :", len(df))
    print("Temporal windows:", len(X))
    print("Window shape    :", X.shape)
    print("Labels          :", y.shape)
    print("Subjects        :", len(np.unique(groups)))

    print("\nClass distribution:")
    print(
        pd.Series(y).value_counts()
    )

    print("\nWindow example:")
    print(
        metadata.head()
    )

    output_dir = Path(
        "results/experiments"
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        output_dir / "temporal_X.npy",
        X,
    )

    np.save(
        output_dir / "temporal_y.npy",
        y,
    )

    np.save(
        output_dir / "temporal_groups.npy",
        groups,
    )

    metadata.to_csv(
        output_dir / "temporal_metadata.csv",
        index=False,
    )

    print("\nSaved:")
    print(
        "results/experiments/temporal_X.npy"
    )
    print(
        "results/experiments/temporal_y.npy"
    )
    print(
        "results/experiments/temporal_groups.npy"
    )


if __name__ == "__main__":
    main()