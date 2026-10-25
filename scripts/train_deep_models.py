from __future__ import annotations

import json
import time
from pathlib import Path

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
    recall_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

N_SPLITS = 5
BATCH_SIZE = 256
EPOCHS = 15
PATIENCE = 3
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

WINDOW_LENGTH = 5
N_FEATURES = 11

DEVICE = torch.device("cpu")

BASE_DIR = Path(__file__).resolve().parents[1]

X_PATH = BASE_DIR / "results" / "experiments" / "temporal_X.npy"
Y_PATH = BASE_DIR / "results" / "experiments" / "temporal_y.npy"
GROUPS_PATH = BASE_DIR / "results" / "experiments" / "temporal_groups.npy"

RESULTS_DIR = BASE_DIR / "results"
PRED_DIR = RESULTS_DIR / "predictions"
METRICS_DIR = RESULTS_DIR / "metrics"
COMPLEXITY_DIR = RESULTS_DIR / "complexity"

MODEL_DIR = BASE_DIR / "models" / "trained"

for directory in [
    PRED_DIR,
    METRICS_DIR,
    COMPLEXITY_DIR,
    MODEL_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================
# MODELS
# ============================================================

class SmallCNN(nn.Module):
    """
    Lightweight 1D CNN baseline.

    Input:
        [batch, sequence_length, features]

    Internally:
        [batch, features, sequence_length]
    """

    def __init__(self, n_features: int = 11):
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


class LightweightTransformer(nn.Module):
    """
    Small Transformer for temporal HRV feature windows.

    2 encoder layers
    d_model = 64
    4 attention heads
    """

    def __init__(
        self,
        n_features: int = 11,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.input_projection = nn.Linear(
            n_features,
            d_model,
        )

        self.position_embedding = nn.Parameter(
            torch.zeros(1, WINDOW_LENGTH, d_model)
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
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):

        x = self.input_projection(x)

        x = x + self.position_embedding[:, :x.size(1)]

        x = self.encoder(x)

        # Mean temporal pooling
        x = x.mean(dim=1)

        return self.classifier(x).squeeze(1)


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    probabilities,
    threshold=0.5,
):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    sensitivity = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    metrics = {
        "accuracy": accuracy_score(
            y_true,
            predictions,
        ),

        "balanced_accuracy": balanced_accuracy_score(
            y_true,
            predictions,
        ),

        "precision": precision_score(
            y_true,
            predictions,
            zero_division=0,
        ),

        "sensitivity": sensitivity,

        "specificity": specificity,

        "f1": f1_score(
            y_true,
            predictions,
            zero_division=0,
        ),

        "roc_auc": roc_auc_score(
            y_true,
            probabilities,
        ),

        "pr_auc": average_precision_score(
            y_true,
            probabilities,
        ),
    }

    return metrics


# ============================================================
# DATA SCALING
# ============================================================

def scale_fold_data(
    X_train,
    X_val,
):
    """
    Fit scaler ONLY on training subjects.

    This prevents validation leakage.
    """

    scaler = StandardScaler()

    n_train = X_train.shape[0]
    n_val = X_val.shape[0]

    train_2d = X_train.reshape(
        n_train * WINDOW_LENGTH,
        N_FEATURES,
    )

    val_2d = X_val.reshape(
        n_val * WINDOW_LENGTH,
        N_FEATURES,
    )

    scaler.fit(train_2d)

    train_scaled = scaler.transform(
        train_2d
    ).reshape(
        n_train,
        WINDOW_LENGTH,
        N_FEATURES,
    )

    val_scaled = scaler.transform(
        val_2d
    ).reshape(
        n_val,
        WINDOW_LENGTH,
        N_FEATURES,
    )

    return train_scaled, val_scaled, scaler


# ============================================================
# DATA LOADERS
# ============================================================

def make_loader(
    X,
    y,
    shuffle=False,
):

    X_tensor = torch.tensor(
        X,
        dtype=torch.float32,
    )

    y_tensor = torch.tensor(
        y,
        dtype=torch.float32,
    )

    dataset = torch.utils.data.TensorDataset(
        X_tensor,
        y_tensor,
    )

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
    )


# ============================================================
# TRAINING
# ============================================================

def train_model(
    model,
    train_loader,
    val_loader,
    model_name,
):

    model = model.to(DEVICE)

    # Calculate class weighting from training data
    train_labels = []

    for _, y_batch in train_loader:
        train_labels.extend(
            y_batch.numpy().tolist()
        )

    train_labels = np.asarray(
        train_labels
    )

    positives = train_labels.sum()
    negatives = len(train_labels) - positives

    pos_weight_value = (
        negatives / positives
        if positives > 0
        else 1.0
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            pos_weight_value,
            dtype=torch.float32,
            device=DEVICE,
        )
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_state = None
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        model.train()

        train_losses = []

        for X_batch, y_batch in train_loader:

            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            optimizer.zero_grad()

            logits = model(X_batch)

            loss = criterion(
                logits,
                y_batch,
            )

            loss.backward()

            optimizer.step()

            train_losses.append(
                loss.item()
            )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        model.eval()

        val_losses = []

        with torch.no_grad():

            for X_batch, y_batch in val_loader:

                X_batch = X_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)

                logits = model(X_batch)

                loss = criterion(
                    logits,
                    y_batch,
                )

                val_losses.append(
                    loss.item()
                )

        train_loss = np.mean(
            train_losses
        )

        val_loss = np.mean(
            val_losses
        )

        print(
            f"{model_name} | "
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train={train_loss:.4f} | "
            f"Val={val_loss:.4f}"
        )

        # ----------------------------------------------------
        # EARLY STOPPING
        # ----------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

            patience_counter = 0

        else:

            patience_counter += 1

            if patience_counter >= PATIENCE:

                print(
                    f"{model_name} | "
                    "Early stopping."
                )

                break

    if best_state is not None:

        model.load_state_dict(
            best_state
        )

    return model


# ============================================================
# PREDICTION
# ============================================================

def predict_model(
    model,
    X,
):

    loader = make_loader(
        X,
        np.zeros(len(X)),
        shuffle=False,
    )

    model.eval()

    probabilities = []

    with torch.no_grad():

        for X_batch, _ in loader:

            X_batch = X_batch.to(DEVICE)

            logits = model(
                X_batch
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
# MODEL COMPLEXITY
# ============================================================

def model_complexity(
    model,
    model_name,
):

    parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    parameter_size_mb = (
        parameters * 4
    ) / (
        1024 ** 2
    )

    model.eval()

    dummy = torch.randn(
        1,
        WINDOW_LENGTH,
        N_FEATURES,
    )

    # Warm-up
    with torch.no_grad():
        for _ in range(5):
            model(dummy)

    start = time.perf_counter()

    with torch.no_grad():
        for _ in range(50):
            model(dummy)

    elapsed = (
        time.perf_counter() - start
    )

    latency_ms = (
        elapsed / 50
    ) * 1000

    return {
        "model": model_name,
        "parameters": parameters,
        "trainable_parameters": trainable_parameters,
        "parameter_size_mb": parameter_size_mb,
        "cpu_inference_latency_ms": latency_ms,
    }


# ============================================================
# CROSS-VALIDATION
# ============================================================

def run_cross_validation(
    X,
    y,
    groups,
    model_name,
):

    group_kfold = GroupKFold(
        n_splits=N_SPLITS
    )

    fold_metrics = []

    all_predictions = []

    all_probabilities = []

    all_targets = []

    all_groups = []

    best_model = None

    best_fold_f1 = -1

    for fold, (
        train_idx,
        val_idx,
    ) in enumerate(
        group_kfold.split(
            X,
            y,
            groups,
        ),
        start=1,
    ):

        print()
        print(
            "=" * 60
        )
        print(
            f"{model_name.upper()} — FOLD {fold}/{N_SPLITS}"
        )
        print(
            "=" * 60
        )

        X_train = X[
            train_idx
        ]

        X_val = X[
            val_idx
        ]

        y_train = y[
            train_idx
        ]

        y_val = y[
            val_idx
        ]

        # IMPORTANT:
        # Fit scaler only on training subjects.
        (
            X_train,
            X_val,
            scaler,
        ) = scale_fold_data(
            X_train,
            X_val,
        )

        train_loader = make_loader(
            X_train,
            y_train,
            shuffle=True,
        )

        val_loader = make_loader(
            X_val,
            y_val,
            shuffle=False,
        )

        if model_name == "cnn":

            model = SmallCNN(
                n_features=N_FEATURES
            )

        elif model_name == "transformer":

            model = LightweightTransformer(
                n_features=N_FEATURES
            )

        else:

            raise ValueError(
                f"Unknown model: {model_name}"
            )

        model = train_model(
            model,
            train_loader,
            val_loader,
            model_name,
        )

        probabilities = predict_model(
            model,
            X_val,
        )

        metrics = calculate_metrics(
            y_val,
            probabilities,
        )

        metrics["fold"] = fold

        fold_metrics.append(
            metrics
        )

        all_predictions.extend(
            (
                probabilities >= 0.5
            ).astype(int)
        )

        all_probabilities.extend(
            probabilities
        )

        all_targets.extend(
            y_val
        )

        all_groups.extend(
            groups[val_idx]
        )

        print()
        print(
            f"Fold {fold}: "
            f"F1={metrics['f1']:.4f} | "
            f"Sensitivity={metrics['sensitivity']:.4f} | "
            f"Specificity={metrics['specificity']:.4f} | "
            f"ROC-AUC={metrics['roc_auc']:.4f}"
        )

        if metrics["f1"] > best_fold_f1:

            best_fold_f1 = metrics["f1"]

            best_model = model

    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    torch.save(
        best_model.state_dict(),
        MODEL_DIR
        / f"{model_name}_best.pt",
    )

    # --------------------------------------------------------
    # SAVE PREDICTIONS
    # --------------------------------------------------------

    prediction_df = pd.DataFrame(
        {
            "group": all_groups,
            "y_true": all_targets,
            "probability": all_probabilities,
            "prediction": all_predictions,
        }
    )

    prediction_df.to_csv(
        PRED_DIR
        / f"{model_name}_predictions.csv",
        index=False,
    )

    # --------------------------------------------------------
    # AGGREGATE METRICS
    # --------------------------------------------------------

    metric_names = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "sensitivity",
        "specificity",
        "f1",
        "roc_auc",
        "pr_auc",
    ]

    summary = {}

    for metric in metric_names:

        values = [
            fold[metric]
            for fold in fold_metrics
        ]

        summary[
            metric
        ] = float(
            np.mean(values)
        )

        summary[
            f"{metric}_std"
        ] = float(
            np.std(values)
        )

    summary[
        "fold_results"
    ] = fold_metrics

    with open(
        METRICS_DIR
        / f"{model_name}_metrics.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # COMPLEXITY
    # --------------------------------------------------------

    complexity = model_complexity(
        best_model,
        model_name,
    )

    with open(
        COMPLEXITY_DIR
        / f"{model_name}_complexity.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            complexity,
            f,
            indent=2,
        )

    return summary, complexity


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed()

    print()
    print("=" * 60)
    print("DEEP MODEL TRAINING")
    print("=" * 60)

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"PyTorch: {torch.__version__}"
    )

    # --------------------------------------------------------
    # LOAD TEMPORAL DATA
    # --------------------------------------------------------

    X = np.load(
        X_PATH
    ).astype(
        np.float32
    )

    y = np.load(
        Y_PATH
    ).astype(
        np.int64
    )

    groups = np.load(
        GROUPS_PATH,
        allow_pickle=True,
    )

    print()
    print(
        f"X shape      : {X.shape}"
    )

    print(
        f"y shape      : {y.shape}"
    )

    print(
        f"Groups       : {len(np.unique(groups))}"
    )

    print(
        f"Class 0      : {(y == 0).sum()}"
    )

    print(
        f"Class 1      : {(y == 1).sum()}"
    )

    # --------------------------------------------------------
    # CNN
    # --------------------------------------------------------

    print()
    print(
        "#" * 60
    )
    print(
        "# 1D CNN"
    )
    print(
        "#" * 60
    )

    cnn_metrics, cnn_complexity = (
        run_cross_validation(
            X,
            y,
            groups,
            "cnn",
        )
    )

    # --------------------------------------------------------
    # TRANSFORMER
    # --------------------------------------------------------

    print()
    print(
        "#" * 60
    )
    print(
        "# LIGHTWEIGHT TRANSFORMER"
    )
    print(
        "#" * 60
    )

    transformer_metrics, transformer_complexity = (
        run_cross_validation(
            X,
            y,
            groups,
            "transformer",
        )
    )

    # --------------------------------------------------------
    # PRINT FINAL RESULTS
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("DEEP MODEL SUMMARY")
    print("=" * 60)

    for name, metrics in [
        ("CNN", cnn_metrics),
        (
            "Transformer",
            transformer_metrics,
        ),
    ]:

        print()
        print(name)

        print(
            f"Accuracy       : "
            f"{metrics['accuracy']:.4f} ± "
            f"{metrics['accuracy_std']:.4f}"
        )

        print(
            f"Balanced Acc.  : "
            f"{metrics['balanced_accuracy']:.4f} ± "
            f"{metrics['balanced_accuracy_std']:.4f}"
        )

        print(
            f"Precision      : "
            f"{metrics['precision']:.4f} ± "
            f"{metrics['precision_std']:.4f}"
        )

        print(
            f"Sensitivity    : "
            f"{metrics['sensitivity']:.4f} ± "
            f"{metrics['sensitivity_std']:.4f}"
        )

        print(
            f"Specificity    : "
            f"{metrics['specificity']:.4f} ± "
            f"{metrics['specificity_std']:.4f}"
        )

        print(
            f"F1             : "
            f"{metrics['f1']:.4f} ± "
            f"{metrics['f1_std']:.4f}"
        )

        print(
            f"ROC-AUC        : "
            f"{metrics['roc_auc']:.4f} ± "
            f"{metrics['roc_auc_std']:.4f}"
        )

        print(
            f"PR-AUC         : "
            f"{metrics['pr_auc']:.4f} ± "
            f"{metrics['pr_auc_std']:.4f}"
        )

    # --------------------------------------------------------
    # COMBINED COMPLEXITY
    # --------------------------------------------------------

    complexity_df = pd.DataFrame(
        [
            cnn_complexity,
            transformer_complexity,
        ]
    )

    complexity_df.to_csv(
        COMPLEXITY_DIR
        / "deep_model_complexity.csv",
        index=False,
    )

    print()
    print(
        "=" * 60
    )
    print(
        "COMPUTATIONAL COMPLEXITY"
    )
    print(
        "=" * 60
    )

    print(
        complexity_df.to_string(
            index=False
        )
    )

    print()
    print(
        "Saved CNN predictions."
    )

    print(
        "Saved Transformer predictions."
    )

    print(
        "Saved CNN/Transformer metrics."
    )

    print(
        "Saved model complexity."
    )

    print()
    print(
        "DEEP MODEL TRAINING COMPLETE."
    )


if __name__ == "__main__":
    main()