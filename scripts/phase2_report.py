from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]

RESULTS = ROOT / "results"

METRICS_DIR = RESULTS / "metrics"
PRED_DIR = RESULTS / "predictions"
COMPARISON_DIR = RESULTS / "model_comparison"
COMPLEXITY_DIR = RESULTS / "complexity"
XAI_DIR = RESULTS / "explainability"
ANALYSIS_DIR = RESULTS / "analysis"
FIG_DIR = RESULTS / "figures"
EXPERIMENT_DIR = RESULTS / "experiments"

for directory in [
    METRICS_DIR,
    PRED_DIR,
    COMPARISON_DIR,
    COMPLEXITY_DIR,
    XAI_DIR,
    ANALYSIS_DIR,
    FIG_DIR,
    EXPERIMENT_DIR,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# HELPERS
# ============================================================

def load_json(path):

    if not path.exists():
        return None

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def metric_value(summary, metric):

    if summary is None:
        return np.nan

    value = summary.get(metric)

    if isinstance(value, dict):
        return value.get("mean", np.nan)

    if value is None:
        return np.nan

    return value


def metric_std(summary, metric):

    if summary is None:
        return np.nan

    value = summary.get(metric)

    if isinstance(value, dict):
        return value.get("std", np.nan)

    return np.nan


def find_column(df, candidates):

    for column in candidates:
        if column in df.columns:
            return column

    return None


# ============================================================
# DATASET SUMMARY
# ============================================================

def build_dataset_summary():

    temporal_x_path = (
        RESULTS /
        "experiments" /
        "temporal_X.npy"
    )

    temporal_y_path = (
        RESULTS /
        "experiments" /
        "temporal_y.npy"
    )

    temporal_groups_path = (
        RESULTS /
        "experiments" /
        "temporal_groups.npy"
    )

    X = np.load(temporal_x_path)
    y = np.load(temporal_y_path)
    groups = np.load(temporal_groups_path)

    summary = {
        "temporal_windows": int(X.shape[0]),
        "window_length": int(X.shape[1]),
        "feature_count_per_timestep": int(X.shape[2]),
        "subjects": int(len(np.unique(groups))),
        "non_apnea_windows": int((y == 0).sum()),
        "apnea_windows": int((y == 1).sum()),
    }

    return summary


# ============================================================
# MODEL PERFORMANCE
# ============================================================

def build_model_performance():

    comparison_path = (
        COMPARISON_DIR /
        "complete_model_comparison.csv"
    )

    if comparison_path.exists():

        df = pd.read_csv(
            comparison_path
        )

        # Keep the three core models only.
        core_models = [
            "Random Forest",
            "1D CNN",
            "Transformer V1",
            "Lightweight Transformer",
        ]

        df = df[
            df["model"].isin(core_models)
        ].copy()

        return df

    # Fallback if comparison file does not exist.
    summaries = {
        "Random Forest":
            load_json(
                METRICS_DIR /
                "random_forest_summary.json"
            ),
        "1D CNN":
            load_json(
                METRICS_DIR /
                "cnn_metrics.json"
            ),
        "Transformer V1":
            load_json(
                METRICS_DIR /
                "transformer_metrics.json"
            ),
    }

    rows = []

    metrics = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "sensitivity",
        "specificity",
        "f1",
        "roc_auc",
        "pr_auc",
    ]

    for model, summary in summaries.items():

        row = {
            "model": model
        }

        for metric in metrics:

            row[metric] = metric_value(
                summary,
                metric,
            )

            row[f"{metric}_std"] = metric_std(
                summary,
                metric,
            )

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# ERROR ANALYSIS
# ============================================================

def error_analysis():

    models = {
        "Random Forest":
            "random_forest_predictions.csv",
        "1D CNN":
            "cnn_predictions.csv",
        "Transformer V1":
            "transformer_predictions.csv",
    }

    rows = []

    for model_name, filename in models.items():

        path = PRED_DIR / filename

        if not path.exists():
            continue

        df = pd.read_csv(path)

        y_true_col = find_column(
            df,
            [
                "y_true",
                "true_label",
                "label",
                "y",
            ],
        )

        y_pred_col = find_column(
            df,
            [
                "y_pred",
                "prediction",
                "predicted_label",
            ],
        )

        if y_true_col is None:
            continue

        if y_pred_col is None:

            probability_col = find_column(
                df,
                [
                    "probability",
                    "y_prob",
                ],
            )

            if probability_col is None:
                continue

            df["prediction"] = (
                df[probability_col] >= 0.5
            ).astype(int)

            y_pred_col = "prediction"

        y_true = df[y_true_col].astype(int)
        y_pred = df[y_pred_col].astype(int)

        tp = int(
            ((y_true == 1) & (y_pred == 1)).sum()
        )

        tn = int(
            ((y_true == 0) & (y_pred == 0)).sum()
        )

        fp = int(
            ((y_true == 0) & (y_pred == 1)).sum()
        )

        fn = int(
            ((y_true == 1) & (y_pred == 0)).sum()
        )

        total = len(df)

        rows.append(
            {
                "model": model_name,
                "total_samples": total,
                "true_positive": tp,
                "true_negative": tn,
                "false_positive": fp,
                "false_negative": fn,
            }
        )

    result = pd.DataFrame(rows)

    if not result.empty:

        result.to_csv(
            ANALYSIS_DIR /
            "model_error_analysis.csv",
            index=False,
        )

    return result


# ============================================================
# SUBJECT-LEVEL SCREENING
# ============================================================

def subject_screening():

    path = (
        PRED_DIR /
        "transformer_predictions.csv"
    )

    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)

    group_col = find_column(
        df,
        [
            "group",
            "record",
            "record_name",
            "subject",
        ],
    )

    true_col = find_column(
        df,
        [
            "y_true",
            "true_label",
            "label",
            "y",
        ],
    )

    pred_col = find_column(
        df,
        [
            "prediction",
            "y_pred",
            "predicted_label",
        ],
    )

    if group_col is None or pred_col is None:
        return pd.DataFrame()

    if true_col is None:
        true_col = pred_col

    result = (
        df.groupby(group_col)
        .agg(
            total_windows=(pred_col, "size"),
            predicted_apnea_windows=(
                pred_col,
                "sum",
            ),
            actual_apnea_windows=(
                true_col,
                "sum",
            ),
        )
        .reset_index()
    )

    result[
        "predicted_apnea_burden_percent"
    ] = (
        result["predicted_apnea_windows"]
        / result["total_windows"]
        * 100
    )

    result[
        "actual_apnea_burden_percent"
    ] = (
        result["actual_apnea_windows"]
        / result["total_windows"]
        * 100
    )

    result.to_csv(
        ANALYSIS_DIR /
        "subject_level_screening_summary.csv",
        index=False,
    )

    return result


# ============================================================
# XAI SUMMARY
# ============================================================

def xai_summary():

    path = (
        XAI_DIR /
        "random_forest_feature_importance.csv"
    )

    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)

    df = df.sort_values(
        "importance",
        ascending=False,
    )

    df.to_csv(
        XAI_DIR /
        "final_feature_importance.csv",
        index=False,
    )

    return df


# ============================================================
# COMPLEXITY
# ============================================================

def complexity_summary():

    path = (
        COMPARISON_DIR /
        "final_model_complexity.csv"
    )

    if path.exists():

        return pd.read_csv(path)

    path = (
        COMPLEXITY_DIR /
        "deep_model_complexity.csv"
    )

    if path.exists():

        return pd.read_csv(path)

    return pd.DataFrame()


# ============================================================
# GENERATE MASTER FIGURE
# ============================================================

def generate_performance_figure(performance):

    if performance.empty:
        return

    metrics = [
        "accuracy",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "f1",
        "roc_auc",
        "pr_auc",
    ]

    available = [
        metric
        for metric in metrics
        if metric in performance.columns
    ]

    if not available:
        return

    plot_df = performance[
        ["model"] + available
    ].set_index("model")

    ax = plot_df.plot(
        kind="bar",
        figsize=(12, 6),
    )

    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_title(
        "Subject-wise Model Performance"
    )

    plt.xticks(
        rotation=0
    )

    plt.tight_layout()

    plt.savefig(
        FIG_DIR /
        "12_final_model_performance.png",
        dpi=200,
    )

    plt.close()


def generate_error_figure(errors):

    if errors.empty:
        return

    plot_df = errors.set_index(
        "model"
    )[
        [
            "true_positive",
            "true_negative",
            "false_positive",
            "false_negative",
        ]
    ]

    ax = plot_df.plot(
        kind="bar",
        figsize=(11, 6),
    )

    ax.set_ylabel("Number of Windows")
    ax.set_title(
        "Model Prediction Error Analysis"
    )

    plt.xticks(
        rotation=0
    )

    plt.tight_layout()

    plt.savefig(
        FIG_DIR /
        "13_error_analysis.png",
        dpi=200,
    )

    plt.close()


def generate_complexity_figure(complexity):

    if complexity.empty:
        return

    if "model" not in complexity.columns:
        return

    if "parameters" not in complexity.columns:
        return

    plot_df = complexity[
        [
            "model",
            "parameters",
        ]
    ].set_index("model")

    ax = plot_df.plot(
        kind="bar",
        figsize=(9, 5),
        legend=False,
    )

    ax.set_ylabel("Number of Parameters")
    ax.set_title(
        "Model Parameter Complexity"
    )

    plt.xticks(
        rotation=0
    )

    plt.tight_layout()

    plt.savefig(
        FIG_DIR /
        "14_model_parameter_complexity.png",
        dpi=200,
    )

    plt.close()


# ============================================================
# MASTER REPORT
# ============================================================

def write_master_report(
    dataset,
    performance,
    errors,
    subjects,
    xai,
    complexity,
):

    report_path = (
        EXPERIMENT_DIR /
        "phase2_summary.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "=" * 70 + "\n"
        )

        f.write(
            "PHASE 2 — MODELING, EVALUATION & XAI SUMMARY\n"
        )

        f.write(
            "=" * 70 + "\n\n"
        )

        # ----------------------------------------------------
        # DATASET
        # ----------------------------------------------------

        f.write(
            "1. TEMPORAL DATASET\n"
        )

        f.write(
            "-" * 70 + "\n"
        )

        for key, value in dataset.items():

            f.write(
                f"{key:<35}: {value}\n"
            )

        f.write("\n")

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        f.write(
            "2. VALIDATION STRATEGY\n"
        )

        f.write(
            "-" * 70 + "\n"
        )

        f.write(
            "Subject-wise 5-fold GroupKFold validation\n"
        )

        f.write(
            "Validation groups are recording subjects.\n"
        )

        f.write(
            "This reduces patient-specific data leakage and "
            "tests generalization to unseen subjects.\n\n"
        )

        # ----------------------------------------------------
        # PERFORMANCE
        # ----------------------------------------------------

        f.write(
            "3. MODEL PERFORMANCE\n"
        )

        f.write(
            "-" * 70 + "\n"
        )

        if not performance.empty:

            columns = [
                "model",
                "accuracy",
                "balanced_accuracy",
                "precision",
                "sensitivity",
                "specificity",
                "f1",
                "roc_auc",
                "pr_auc",
            ]

            columns = [
                c
                for c in columns
                if c in performance.columns
            ]

            f.write(
                performance[
                    columns
                ].to_string(index=False)
            )

            f.write("\n")

        f.write("\n")

        # ----------------------------------------------------
        # ERROR ANALYSIS
        # ----------------------------------------------------

        f.write(
            "4. ERROR ANALYSIS\n"
        )

        f.write(
            "-" * 70 + "\n"
        )

        if not errors.empty:

            f.write(
                errors.to_string(
                    index=False
                )
            )

            f.write("\n")

        f.write("\n")

        # ----------------------------------------------------
        # XAI
        # ----------------------------------------------------

        f.write(
            "5. EXPLAINABLE AI\n"
        )

        f.write(
            "-" * 70 + "\n"
        )

        f.write(
            "Global Random Forest feature importance "
            "was calculated using the ECG-derived feature set.\n"
        )

        f.write(
            "This analysis identifies the relative contribution "
            "of physiological features used by the classifier.\n\n"
        )

        if not xai.empty:

            f.write(
                xai.to_string(
                    index=False
                )
            )

            f.write("\n")

        f.write("\n")

        f.write(
            "Temporal saliency is treated as an auxiliary "
            "analysis and is not presented as direct clinical "
            "interpretation of Transformer attention.\n\n"
        )

        # ----------------------------------------------------
        # COMPLEXITY
        # ----------------------------------------------------

        f.write(
            "6. COMPUTATIONAL EFFICIENCY\n"
        )

        f.write(
            "-" * 70 + "\n"
        )

        if not complexity.empty:

            f.write(
                complexity.to_string(
                    index=False
                )
            )

            f.write("\n")

        f.write("\n")

        # ----------------------------------------------------
        # SUBJECT SCREENING
        # ----------------------------------------------------

        f.write(
            "7. SUBJECT-LEVEL SCREENING OUTPUT\n"
        )

        f.write(
            "-" * 70 + "\n"
        )

        if not subjects.empty:

            f.write(
                f"Subjects summarized: "
                f"{len(subjects)}\n"
            )

            f.write(
                "Transformer V1 prediction burden is "
                "reported as the proportion of analyzed "
                "temporal windows classified as apnea.\n"
            )

            f.write(
                "This value is NOT a clinical AHI estimate.\n"
            )

        f.write("\n")

        # ----------------------------------------------------
        # ARTIFACTS
        # ----------------------------------------------------

        f.write(
            "8. GENERATED ARTIFACTS\n"
        )

        f.write(
            "-" * 70 + "\n"
        )

        artifacts = [
            "results/model_comparison/complete_model_comparison.csv",
            "results/model_comparison/best_model_by_metric.csv",
            "results/model_comparison/final_evaluation_summary.csv",
            "results/model_comparison/final_model_complexity.csv",
            "results/analysis/model_error_analysis.csv",
            "results/analysis/subject_level_screening_summary.csv",
            "results/explainability/final_feature_importance.csv",
            "results/figures/12_final_model_performance.png",
            "results/figures/13_error_analysis.png",
            "results/figures/14_model_parameter_complexity.png",
        ]

        for artifact in artifacts:

            f.write(
                f"✓ {artifact}\n"
            )

        f.write("\n")

        f.write(
            "9. INTERPRETATION NOTE\n"
        )

        f.write(
            "-" * 70 + "\n"
        )

        f.write(
            "The system is intended for preliminary "
            "sleep-apnea screening research.\n"
        )

        f.write(
            "It does not replace polysomnography and "
            "does not provide an independent medical diagnosis.\n"
        )

    return report_path


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print(
        "          FINALIZING PHASE 2"
    )
    print(
        "     MODELING • EVALUATION • XAI"
    )
    print("=" * 70)

    print("\nLoading temporal dataset...")

    dataset = build_dataset_summary()

    print(
        f"Temporal windows       : "
        f"{dataset['temporal_windows']}"
    )

    print(
        f"Window length          : "
        f"{dataset['window_length']} minutes"
    )

    print(
        f"Features per timestep  : "
        f"{dataset['feature_count_per_timestep']}"
    )

    print(
        f"Subjects               : "
        f"{dataset['subjects']}"
    )

    print("\nLoading model performance...")

    performance = build_model_performance()

    print(
        performance.to_string(
            index=False
        )
    )

    print("\nRunning error analysis...")

    errors = error_analysis()

    if not errors.empty:

        print(
            errors.to_string(
                index=False
            )
        )

    print("\nBuilding subject-level screening summary...")

    subjects = subject_screening()

    if not subjects.empty:

        print(
            f"Subjects summarized: "
            f"{len(subjects)}"
        )

    print("\nLoading XAI results...")

    xai = xai_summary()

    if not xai.empty:

        print(
            xai.to_string(
                index=False
            )
        )

    print("\nLoading complexity results...")

    complexity = complexity_summary()

    if not complexity.empty:

        print(
            complexity.to_string(
                index=False
            )
        )

    print("\nGenerating consolidated figures...")

    generate_performance_figure(
        performance
    )

    generate_error_figure(
        errors
    )

    generate_complexity_figure(
        complexity
    )

    print(
        "\nWriting master Phase-2 report..."
    )

    report_path = write_master_report(
        dataset,
        performance,
        errors,
        subjects,
        xai,
        complexity,
    )

    print("\n" + "=" * 70)
    print(
        "              PHASE 2 COMPLETE"
    )
    print("=" * 70)

    print("\nDATASET")
    print("-" * 70)

    print(
        f"Temporal windows        : "
        f"{dataset['temporal_windows']}"
    )

    print(
        f"Subjects                : "
        f"{dataset['subjects']}"
    )

    print("\nMODELS")
    print("-" * 70)

    if not performance.empty:

        for model in performance["model"]:

            print(
                f"✓ {model}"
            )

    print("\nEXPERIMENTS")
    print("-" * 70)

    print("✓ Subject-wise 5-fold validation")
    print("✓ Performance evaluation")
    print("✓ Confusion matrix analysis")
    print("✓ ROC analysis")
    print("✓ Precision-recall analysis")
    print("✓ Error analysis")
    print("✓ Feature importance")
    print("✓ Computational complexity")
    print("✓ Subject-level screening summary")

    print("\nMASTER OUTPUT")
    print("-" * 70)

    print(
        f"✓ {report_path}"
    )

    print(
        f"✓ {FIG_DIR / '12_final_model_performance.png'}"
    )

    print(
        f"✓ {FIG_DIR / '13_error_analysis.png'}"
    )

    print(
        f"✓ {FIG_DIR / '14_model_parameter_complexity.png'}"
    )

    print("\n" + "=" * 70)
    print(
        "       PHASE 2 — CONSOLIDATED ✓"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()