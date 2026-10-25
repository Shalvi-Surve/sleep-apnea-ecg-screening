from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]

RESULTS = ROOT / "results"
PRED_DIR = RESULTS / "predictions"
ANALYSIS_DIR = RESULTS / "analysis"
FIG_DIR = RESULTS / "figures"

ANALYSIS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FIG_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def find_column(df, candidates):

    for c in candidates:
        if c in df.columns:
            return c

    return None


def analyse_model(filename, model_name):

    path = PRED_DIR / filename

    if not path.exists():
        print(f"Missing: {path}")
        return None

    df = pd.read_csv(path)

    print("\n" + "=" * 70)
    print(model_name)
    print("=" * 70)

    print("Columns:", list(df.columns))

    y_col = find_column(
        df,
        ["y_true", "true_label", "label", "y"],
    )

    p_col = find_column(
        df,
        ["y_pred", "predicted_label", "prediction"],
    )

    record_col = find_column(
        df,
        ["record", "record_name", "subject"],
    )

    if y_col is None or p_col is None:
        print(
            "Could not identify true/prediction columns."
        )
        return None

    df["error_type"] = "Correct"

    df.loc[
        (df[y_col] == 0) &
        (df[p_col] == 1),
        "error_type",
    ] = "False Positive"

    df.loc[
        (df[y_col] == 1) &
        (df[p_col] == 0),
        "error_type",
    ] = "False Negative"

    counts = (
        df["error_type"]
        .value_counts()
        .reset_index()
    )

    counts.columns = [
        "error_type",
        "count",
    ]

    counts.to_csv(
        ANALYSIS_DIR /
        f"{model_name.lower().replace(' ', '_')}_error_counts.csv",
        index=False,
    )

    if record_col:

        per_record = (
            df.groupby(record_col)
            .agg(
                total_windows=(y_col, "size"),
                actual_apnea=(y_col, "sum"),
                predicted_apnea=(p_col, "sum"),
            )
            .reset_index()
        )

        per_record["predicted_apnea_percentage"] = (
            per_record["predicted_apnea"]
            / per_record["total_windows"]
            * 100
        )

        per_record.to_csv(
            ANALYSIS_DIR /
            f"{model_name.lower().replace(' ', '_')}_per_record.csv",
            index=False,
        )

        print("\nPer-record analysis saved.")

    print("\nError counts:")
    print(counts.to_string(index=False))

    return df


def main():

    transformer = analyse_model(
        "transformer_predictions.csv",
        "Transformer V1",
    )

    rf = analyse_model(
        "random_forest_predictions.csv",
        "Random Forest",
    )

    cnn = analyse_model(
        "cnn_predictions.csv",
        "CNN",
    )

    # Subject-level summary using Transformer V1.
    if transformer is not None:

        record_col = find_column(
            transformer,
            ["record", "record_name", "subject"],
        )

        y_col = find_column(
            transformer,
            ["y_true", "true_label", "label", "y"],
        )

        p_col = find_column(
            transformer,
            ["y_pred", "predicted_label", "prediction"],
        )

        if record_col:

            summary = (
                transformer.groupby(record_col)
                .agg(
                    total_windows=(y_col, "size"),
                    actual_apnea_windows=(y_col, "sum"),
                    predicted_apnea_windows=(p_col, "sum"),
                )
                .reset_index()
            )

            summary[
                "predicted_apnea_burden_percent"
            ] = (
                summary["predicted_apnea_windows"]
                / summary["total_windows"]
                * 100
            )

            summary[
                "actual_apnea_burden_percent"
            ] = (
                summary["actual_apnea_windows"]
                / summary["total_windows"]
                * 100
            )

            summary.to_csv(
                ANALYSIS_DIR /
                "subject_level_screening_summary.csv",
                index=False,
            )

            print("\n" + "=" * 70)
            print("SUBJECT-LEVEL SCREENING SUMMARY")
            print("=" * 70)

            print(
                summary.to_string(index=False)
            )

            print(
                "\nSaved:",
                ANALYSIS_DIR /
                "subject_level_screening_summary.csv",
            )

    print("\n" + "=" * 70)
    print("ERROR ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()