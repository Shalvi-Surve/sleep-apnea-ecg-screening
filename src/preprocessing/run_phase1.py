from __future__ import annotations

import logging
from pathlib import Path

from src.data.dataset_utils import get_learning_records
from src.preprocessing.preprocessing_pipeline import (
    assemble_dataset,
    drop_missing_beats,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

log = logging.getLogger(__name__)


def main() -> None:
    data_dir = Path("data/raw")
    output_dir = Path("data/processed")

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    records = get_learning_records(
        keep_duplicate=False
    )

    log.info(
        "Processing %d learning-set records",
        len(records),
    )

    df = assemble_dataset(
        records,
        data_dir,
        n_jobs=1,
    )

    df = drop_missing_beats(
        df,
        min_beats=20,
    )

    output_path = (
        output_dir /
        "epoch_features.csv"
    )

    df.to_csv(
        output_path,
        index=False,
    )

    log.info(
        "Saved epoch features to %s",
        output_path,
    )

    log.info(
        "Final shape: %s",
        df.shape,
    )

    log.info(
        "Records processed: %d",
        df["record"].nunique(),
    )

    log.info(
        "Class distribution:\n%s",
        df["label"].value_counts(
            normalize=True,
            dropna=False,
        ),
    )


if __name__ == "__main__":
    main()