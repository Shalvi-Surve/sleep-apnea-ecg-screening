from __future__ import annotations

import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    roc_auc_score,
    average_precision_score,
)


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_DIR = (
    BASE_DIR /
    "models" /
    "test_ready"
)

DATA_DIR = (
    BASE_DIR /
    "results" /
    "experiments"
)

PRED_DIR = (
    BASE_DIR /
    "results" /
    "test_predictions"
)

RESULT_DIR = (
    BASE_DIR /
    "results" /
    "test_results"
)

PRED_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

WINDOW_LENGTH = 5
N_FEATURES = 11

DEVICE = torch.device("cpu")


# ============================================================
# CNN
# ============================================================

class SmallCNN(nn.Module):

    def __init__(
        self,
        n_features=11,
    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.Conv1d(
                in_channels=n_features,
                out_channels=32,
                kernel_size=3,
                padding=1,
            ),

            nn.ReLU(),

            nn.BatchNorm1d(32),

            nn.Conv1d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                padding=1,
            ),

            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1),

            nn.Flatten(),

            nn.Linear(32, 16),

            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(16, 1),
        )

    def forward(self, x):

        x = x.transpose(1, 2)

        return self.network(x).squeeze(1)


# ============================================================
# TRANSFORMER V1
# ============================================================

class LightweightTransformer(nn.Module):

    def __init__(
        self,
        n_features=11,
        d_model=64,
        n_heads=4,
        n_layers=2,
        dropout=0.2,
    ):

        super().__init__()

        self.input_projection = nn.Linear(
            n_features,
            d_model,
        )

        self.position_embedding = nn.Parameter(
            torch.zeros(
                1,
                WINDOW_LENGTH,
                d_model,
            )
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=128,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )

        self.classifier = nn.Sequential(

            nn.LayerNorm(d_model),

            nn.Linear(
                d_model,
                32,
            ),

            nn.GELU(),

            nn.Dropout(0.2),

            nn.Linear(
                32,
                1,
            ),
        )

    def forward(self, x):

        x = self.input_projection(x)

        x = x + self.position_embedding[
            :,
            :x.size(1),
            :
        ]

        x = self.encoder(x)

        x = x.mean(dim=1)

        return self.classifier(x).squeeze(1)


# ============================================================
# DATA LOADING
# ============================================================

def load_test_data():

    x_path = (
        DATA_DIR /
        "test_temporal_X.npy"
    )

    y_path = (
        DATA_DIR /
        "test_temporal_y.npy"
    )

    groups_path = (
        DATA_DIR /
        "test_temporal_groups.npy"
    )

    metadata_path = (
        DATA_DIR /
        "test_temporal_metadata.csv"
    )

    if not x_path.exists():

        raise FileNotFoundError(
            f"Missing:\n{x_path}\n\n"
            "Run:\n"
            "python scripts/prepare_test_dataset.py"
        )

    if not groups_path.exists():

        raise FileNotFoundError(
            f"Missing:\n{groups_path}"
        )

    X = np.load(
        x_path
    ).astype(
        np.float32
    )

    y = None

    if y_path.exists():

        y = np.load(
            y_path
        ).astype(
            np.int64
        )

    groups = np.load(
        groups_path,
        allow_pickle=True,
    )

    metadata = None

    if metadata_path.exists():

        metadata = pd.read_csv(
            metadata_path
        )

    return X, y, groups, metadata


# ============================================================
# RECORD NORMALIZATION
# ============================================================

def normalize_record(value):

    text = str(
        value
    ).strip().lower()

    match = re.search(
        r"x\d{1,2}",
        text,
    )

    if match:

        number = int(
            re.search(
                r"\d+",
                match.group()
            ).group()
        )

        return f"x{number:02d}"

    return text


# ============================================================
# DEEP MODEL PREDICTION
# ============================================================

def predict_deep(
    model,
    X,
):

    model.eval()

    probabilities = []

    tensor = torch.tensor(
        X,
        dtype=torch.float32,
    )

    loader = torch.utils.data.DataLoader(
        tensor,
        batch_size=256,
        shuffle=False,
        num_workers=0,
    )

    with torch.no_grad():

        for batch in loader:

            logits = model(
                batch.to(DEVICE)
            )

            probs = torch.sigmoid(
                logits
            )

            probabilities.extend(
                probs.cpu().numpy()
            )

    return np.asarray(
        probabilities
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    probabilities,
):

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    sensitivity = (
        tp / (tp + fn)
        if tp + fn > 0
        else 0.0
    )

    specificity = (
        tn / (tn + fp)
        if tn + fp > 0
        else 0.0
    )

    return {

        "accuracy":
            accuracy_score(
                y_true,
                predictions,
            ),

        "balanced_accuracy":
            balanced_accuracy_score(
                y_true,
                predictions,
            ),

        "precision":
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            ),

        "sensitivity":
            sensitivity,

        "specificity":
            specificity,

        "f1":
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            ),

        "roc_auc":
            roc_auc_score(
                y_true,
                probabilities,
            ),

        "pr_auc":
            average_precision_score(
                y_true,
                probabilities,
            ),

        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


# ============================================================
# PREDICTION TABLE
# ============================================================

def create_prediction_table(
    groups,
    probability,
    y,
):

    df = pd.DataFrame(
        {
            "group": [
                normalize_record(x)
                for x in groups
            ],

            "probability": probability,

            "prediction": (
                probability >= 0.5
            ).astype(int),
        }
    )

    if y is not None:

        df["y_true"] = y

    else:

        df["y_true"] = -1

    return df


# ============================================================
# RECORD SCREENING
# ============================================================

def create_record_screening(
    prediction_tables,
):

    records = sorted(
        set(
            record
            for df in prediction_tables.values()
            for record in df["group"].unique()
        )
    )

    rows = []

    for record in records:

        row = {
            "Record": record,
        }

        for model_name, df in prediction_tables.items():

            subset = df[
                df["group"] == record
            ]

            if subset.empty:
                continue

            total = len(
                subset
            )

            positive = int(
                subset["prediction"].sum()
            )

            percentage = (
                positive /
                total *
                100
            )

            mean_probability = float(
                subset["probability"].mean()
            )

            row[
                f"{model_name} Windows"
            ] = total

            row[
                f"{model_name} Positive Windows"
            ] = positive

            row[
                f"{model_name} Positive %"
            ] = percentage

            row[
                f"{model_name} Mean Probability"
            ] = mean_probability

            row[
                f"{model_name} Screening Flag"
            ] = (
                "Apnea-risk"
                if percentage >= 50
                else "No apnea-risk"
            )

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("HELD-OUT TEST INFERENCE — PHYSIONET X01-X35")
    print("=" * 70)

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    X, y, groups, metadata = (
        load_test_data()
    )

    print()
    print(
        f"X shape       : {X.shape}"
    )

    print(
        f"Test records  : "
        f"{len(np.unique(groups))}"
    )

    print(
        "Records:"
    )

    print(
        ", ".join(
            sorted(
                set(
                    normalize_record(x)
                    for x in groups
                )
            )
        )
    )

    # --------------------------------------------------------
    # VALID LABEL MASK
    # --------------------------------------------------------

    if y is None:

        valid_mask = np.zeros(
            len(X),
            dtype=bool,
        )

        print()
        print(
            "WARNING: No test labels found."
        )

    else:

        valid_mask = (
            y >= 0
        )

        print()
        print(
            f"Total windows      : {len(y)}"
        )

        print(
            f"Valid labelled     : "
            f"{valid_mask.sum()}"
        )

        print(
            f"Unknown/unlabelled : "
            f"{(~valid_mask).sum()}"
        )

        print(
            f"Class 0            : "
            f"{(y[valid_mask] == 0).sum()}"
        )

        print(
            f"Class 1            : "
            f"{(y[valid_mask] == 1).sum()}"
        )

    # --------------------------------------------------------
    # SCALER
    # --------------------------------------------------------

    scaler_path = (
        MODEL_DIR /
        "final_scaler.pkl"
    )

    if not scaler_path.exists():

        raise FileNotFoundError(
            f"Missing:\n{scaler_path}\n\n"
            "Run:\n"
            "python scripts/train_final_models.py"
        )

    scaler = joblib.load(
        scaler_path
    )

    X_scaled = scaler.transform(
        X.reshape(
            -1,
            N_FEATURES,
        )
    ).reshape(
        X.shape
    ).astype(
        np.float32
    )

    # ========================================================
    # RANDOM FOREST
    # ========================================================

    print()
    print("=" * 70)
    print("RANDOM FOREST")
    print("=" * 70)

    rf = joblib.load(
        MODEL_DIR /
        "random_forest_final.pkl"
    )

    rf_probability = rf.predict_proba(
        X_scaled.reshape(
            len(X_scaled),
            -1,
        )
    )[:, 1]

    rf_df = create_prediction_table(
        groups,
        rf_probability,
        y,
    )

    rf_df.to_csv(
        PRED_DIR /
        "random_forest_test_predictions.csv",
        index=False,
    )

    # ========================================================
    # CNN
    # ========================================================

    print()
    print("=" * 70)
    print("1D CNN")
    print("=" * 70)

    cnn = SmallCNN(
        n_features=N_FEATURES
    )

    cnn.load_state_dict(
        torch.load(
            MODEL_DIR /
            "cnn_final.pt",
            map_location=DEVICE,
        )
    )

    cnn.to(DEVICE)

    cnn_probability = predict_deep(
        cnn,
        X_scaled,
    )

    cnn_df = create_prediction_table(
        groups,
        cnn_probability,
        y,
    )

    cnn_df.to_csv(
        PRED_DIR /
        "cnn_test_predictions.csv",
        index=False,
    )

    # ========================================================
    # TRANSFORMER
    # ========================================================

    print()
    print("=" * 70)
    print("LIGHTWEIGHT TRANSFORMER V1")
    print("=" * 70)

    transformer = LightweightTransformer(
        n_features=N_FEATURES
    )

    transformer.load_state_dict(
        torch.load(
            MODEL_DIR /
            "transformer_final.pt",
            map_location=DEVICE,
        )
    )

    transformer.to(DEVICE)

    transformer_probability = predict_deep(
        transformer,
        X_scaled,
    )

    transformer_df = create_prediction_table(
        groups,
        transformer_probability,
        y,
    )

    transformer_df.to_csv(
        PRED_DIR /
        "transformer_test_predictions.csv",
        index=False,
    )

    prediction_tables = {

        "Random Forest":
            rf_df,

        "1D CNN":
            cnn_df,

        "Transformer V1":
            transformer_df,
    }

    # ========================================================
    # METRICS ON VALID-LABEL SUBSET
    # ========================================================

    comparison_rows = []

    if y is not None and valid_mask.any():

        y_valid = y[
            valid_mask
        ]

        for model_name, df in prediction_tables.items():

            probability_valid = (
                df["probability"]
                .to_numpy()
                [valid_mask]
            )

            metrics = calculate_metrics(
                y_valid,
                probability_valid,
            )

            comparison_rows.append({

                "Model":
                    model_name,

                "Accuracy":
                    metrics["accuracy"],

                "Balanced Accuracy":
                    metrics["balanced_accuracy"],

                "Precision":
                    metrics["precision"],

                "Sensitivity":
                    metrics["sensitivity"],

                "Specificity":
                    metrics["specificity"],

                "F1":
                    metrics["f1"],

                "ROC-AUC":
                    metrics["roc_auc"],

                "PR-AUC":
                    metrics["pr_auc"],

                "TN":
                    metrics["tn"],

                "FP":
                    metrics["fp"],

                "FN":
                    metrics["fn"],

                "TP":
                    metrics["tp"],

                "Evaluated Windows":
                    int(valid_mask.sum()),

                "Excluded Unknown Labels":
                    int((~valid_mask).sum()),
            })

    comparison = pd.DataFrame(
        comparison_rows
    )

    if not comparison.empty:

        comparison.to_csv(
            RESULT_DIR /
            "test_model_comparison.csv",
            index=False,
        )

        comparison.to_csv(
            RESULT_DIR /
            "test_evaluation_summary.csv",
            index=False,
        )

        print()
        print("=" * 70)
        print("HELD-OUT TEST RESULTS")
        print("=" * 70)

        display_df = comparison.copy()

        metric_columns = [

            "Accuracy",
            "Balanced Accuracy",
            "Precision",
            "Sensitivity",
            "Specificity",
            "F1",
            "ROC-AUC",
            "PR-AUC",
        ]

        for column in metric_columns:

            display_df[column] = (
                display_df[column]
                * 100
            ).round(2)

        print(
            display_df.to_string(
                index=False
            )
        )

        # ----------------------------------------------------
        # BEST MODELS
        # ----------------------------------------------------

        print()
        print(
            "BEST TEST-SET RESULTS"
        )

        print(
            f"Accuracy : "
            f"{comparison.loc[comparison['Accuracy'].idxmax(), 'Model']}"
        )

        print(
            f"F1       : "
            f"{comparison.loc[comparison['F1'].idxmax(), 'Model']}"
        )

        print(
            f"PR-AUC   : "
            f"{comparison.loc[comparison['PR-AUC'].idxmax(), 'Model']}"
        )

    # ========================================================
    # RECORD SCREENING
    # ========================================================

    screening = create_record_screening(
        prediction_tables
    )

    screening.to_csv(
        RESULT_DIR /
        "test_record_screening.csv",
        index=False,
    )

    # ========================================================
    # RECORD SUMMARY
    # ========================================================

    record_rows = []

    for record in sorted(
        set(
            normalize_record(x)
            for x in groups
        )
    ):

        row = {
            "Record": record,
        }

        for model_name, df in prediction_tables.items():

            subset = df[
                df["group"] == record
            ]

            if subset.empty:
                continue

            positive_percentage = (
                subset["prediction"].mean()
                * 100
            )

            mean_probability = (
                subset["probability"].mean()
            )

            row[
                f"{model_name} Positive %"
            ] = float(
                positive_percentage
            )

            row[
                f"{model_name} Mean Probability"
            ] = float(
                mean_probability
            )

            row[
                f"{model_name} Screening Flag"
            ] = (
                "Apnea-risk"
                if positive_percentage >= 50
                else "No apnea-risk"
            )

        record_rows.append(
            row
        )

    record_summary = pd.DataFrame(
        record_rows
    )

    record_summary.to_csv(
        RESULT_DIR /
        "test_record_summary.csv",
        index=False,
    )

    # ========================================================
    # JSON SUMMARY
    # ========================================================

    summary = {

        "dataset":
            "PhysioNet Apnea-ECG",

        "test_records":
            sorted(
                set(
                    normalize_record(x)
                    for x in groups
                )
            ),

        "number_of_test_records":
            int(
                len(
                    np.unique(
                        groups
                    )
                )
            ),

        "total_temporal_windows":
            int(
                len(X)
            ),

        "evaluated_labelled_windows":
            int(
                valid_mask.sum()
            ),

        "excluded_unknown_label_windows":
            int(
                (~valid_mask).sum()
            ),

        "window_length_minutes":
            WINDOW_LENGTH,

        "features_per_timestep":
            N_FEATURES,

        "models":
            [
                "Random Forest",
                "1D CNN",
                "Transformer V1",
            ],

        "prediction_threshold":
            0.5,

        "record_screening_threshold_percent":
            50,

        "training_records":
            34,

        "test_records":
            35,

        "test_labels_partially_available":
            y is not None,

        "evaluation_note":
            (
                "Performance metrics are calculated "
                "only on test windows with valid "
                "reference labels. Unlabelled windows "
                "are retained for prediction and "
                "record-level screening."
            ),
    }

    if not comparison.empty:

        summary[
            "best_accuracy_model"
        ] = comparison.loc[
            comparison["Accuracy"].idxmax(),
            "Model",
        ]

        summary[
            "best_f1_model"
        ] = comparison.loc[
            comparison["F1"].idxmax(),
            "Model",
        ]

        summary[
            "best_pr_auc_model"
        ] = comparison.loc[
            comparison["PR-AUC"].idxmax(),
            "Model",
        ]

    with open(
        RESULT_DIR /
        "test_inference_summary.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    # ========================================================
    # FINISH
    # ========================================================

    print()
    print("=" * 70)
    print("TEST INFERENCE COMPLETE")
    print("=" * 70)

    print(
        f"Test records       : "
        f"{len(np.unique(groups))}"
    )

    print(
        f"Total windows      : "
        f"{len(X)}"
    )

    print(
        f"Evaluated windows  : "
        f"{valid_mask.sum()}"
    )

    print(
        f"Unknown labels     : "
        f"{(~valid_mask).sum()}"
    )

    print()
    print(
        "ALL REQUIRED TEST ARTIFACTS SAVED."
    )


if __name__ == "__main__":
    main()