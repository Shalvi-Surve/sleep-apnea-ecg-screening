from __future__ import annotations

from pathlib import Path


LEARNING_SET = (
    [f"a{i:02d}" for i in range(1, 21)]
    + [f"b{i:02d}" for i in range(1, 6)]
    + [f"c{i:02d}" for i in range(1, 11)]
)

TEST_SET = [f"x{i:02d}" for i in range(1, 36)]

KNOWN_DUPLICATE_RECORD = "c05"


def get_learning_records(
    keep_duplicate: bool = False,
) -> list[str]:
    """
    Return the labeled learning-set records.

    c05 is excluded by default because it overlaps with c06.
    """
    records = list(LEARNING_SET)

    if not keep_duplicate and KNOWN_DUPLICATE_RECORD in records:
        records.remove(KNOWN_DUPLICATE_RECORD)

    return records


def get_test_records() -> list[str]:
    """Return the PhysioNet test-set record names."""
    return list(TEST_SET)


def load_test_labels_from_txt(
    path: str | Path,
) -> dict[str, list[str]]:
    """
    Parse PhysioNet test-dataset-annos.txt.

    Returns:
        {
            "x01": ["N", "A", ...],
            ...
        }
    """
    path = Path(path)

    labels: dict[str, list[str]] = {}

    if not path.exists():
        return labels

    current_record = None

    for line in path.read_text().splitlines():
        line = line.strip()

        if not line:
            continue

        if line.startswith("x") and len(line.split()) == 1:
            current_record = line
            labels[current_record] = []

        elif current_record is not None:
            labels[current_record].extend(
                list(line.replace(" ", ""))
            )

    return labels