# scripts/train_final_models.py

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


# ============================================================
# CONFIG
# ============================================================

SEED = 42

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

MODEL_DIR = BASE_DIR / "models" / "test_ready"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# REPRODUCIBILITY
# ============================================================

np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# CNN — EXACT TRAINING ARCHITECTURE
# ============================================================

class SmallCNN(nn.Module):

    def __init__(self, n_features=11):

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
# TRANSFORMER — EXACT TRAINING ARCHITECTURE
# ============================================================

class FinalTransformer(nn.Module):

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

            nn.Dropout(dropout),

            nn.Linear(
                32,
                1,
            ),
        )

    def forward(self, x):

        x = self.input_projection(x)

        x = x + self.position_embedding[
            :, :x.size(1)
        ]

        x = self.encoder(x)

        x = x.mean(dim=1)

        return self.classifier(x).squeeze(1)


# ============================================================
# LOADER
# ============================================================

def make_loader(X, y, shuffle=False):

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
# TRAIN DEEP MODEL
# ============================================================

def train_deep_model(model, X, y, name):

    loader = make_loader(
        X,
        y,
        shuffle=True,
    )

    positives = float(y.sum())
    negatives = float(len(y) - positives)

    pos_weight = (
        negatives / positives
        if positives > 0
        else 1.0
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            pos_weight,
            dtype=torch.float32,
        ),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_loss = float("inf")
    best_state = None
    patience_counter = 0

    model.to(DEVICE)

    for epoch in range(1, EPOCHS + 1):

        model.train()

        losses = []

        for xb, yb in loader:

            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()

            logits = model(xb)

            loss = criterion(
                logits,
                yb,
            )

            loss.backward()

            optimizer.step()

            losses.append(loss.item())

        epoch_loss = float(
            np.mean(losses)
        )

        print(
            f"{name} | "
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Loss={epoch_loss:.5f}"
        )

        if epoch_loss < best_loss:

            best_loss = epoch_loss

            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

            patience_counter = 0

        else:

            patience_counter += 1

            if patience_counter >= PATIENCE:

                break

    if best_state is not None:

        model.load_state_dict(
            best_state
        )

    return model


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FINAL TRAINING FOR HELD-OUT TEST SET")
    print("=" * 70)

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

    print(
        f"X shape       : {X.shape}"
    )

    print(
        f"y shape       : {y.shape}"
    )

    print(
        f"Subjects      : {len(np.unique(groups))}"
    )

    print(
        f"Class 0       : {(y == 0).sum()}"
    )

    print(
        f"Class 1       : {(y == 1).sum()}"
    )

    # ========================================================
    # FINAL SCALER
    # ========================================================

    X_2d = X.reshape(
        -1,
        N_FEATURES,
    )

    scaler = StandardScaler()

    scaler.fit(X_2d)

    X_scaled = scaler.transform(
        X_2d
    ).reshape(
        X.shape
    ).astype(
        np.float32
    )

    joblib.dump(
        scaler,
        MODEL_DIR / "final_scaler.pkl",
    )

    # ========================================================
    # CNN
    # ========================================================

    print()
    print("=" * 70)
    print("TRAINING FINAL CNN")
    print("=" * 70)

    cnn = SmallCNN(
        n_features=N_FEATURES
    )

    cnn = train_deep_model(
        cnn,
        X_scaled,
        y,
        "CNN",
    )

    torch.save(
        cnn.state_dict(),
        MODEL_DIR / "cnn_final.pt",
    )

    # ========================================================
    # TRANSFORMER
    # ========================================================

    print()
    print("=" * 70)
    print("TRAINING FINAL TRANSFORMER")
    print("=" * 70)

    transformer = FinalTransformer(
        n_features=N_FEATURES
    )

    transformer = train_deep_model(
        transformer,
        X_scaled,
        y,
        "Transformer",
    )

    torch.save(
        transformer.state_dict(),
        MODEL_DIR / "transformer_final.pt",
    )

    # ========================================================
    # RANDOM FOREST
    # ========================================================

    print()
    print("=" * 70)
    print("TRAINING FINAL RANDOM FOREST")
    print("=" * 70)

    rf = RandomForestClassifier(
        n_estimators=300,
        random_state=SEED,
        n_jobs=-1,
        class_weight="balanced",
        max_features="sqrt",
    )

    rf.fit(
        X_scaled.reshape(
            len(X_scaled),
            -1,
        ),
        y,
    )

    joblib.dump(
        rf,
        MODEL_DIR / "random_forest_final.pkl",
    )

    # ========================================================
    # METADATA
    # ========================================================

    metadata = {

        "dataset": "PhysioNet Apnea-ECG",

        "training_records": int(
            len(np.unique(groups))
        ),

        "training_windows": int(
            len(X)
        ),

        "window_length": WINDOW_LENGTH,

        "features_per_timestep": N_FEATURES,

        "models": [
            "Random Forest",
            "1D CNN",
            "Lightweight Transformer V1",
        ],

        "scaler": "StandardScaler",

        "scaler_fitted_on": (
            "34 learning subjects only"
        ),

        "test_set_used": False,

        "seed": SEED,
    }

    with open(
        MODEL_DIR / "final_training_metadata.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
        )

    print()
    print("=" * 70)
    print("FINAL MODELS READY")
    print("=" * 70)

    print(
        MODEL_DIR / "random_forest_final.pkl"
    )

    print(
        MODEL_DIR / "cnn_final.pt"
    )

    print(
        MODEL_DIR / "transformer_final.pt"
    )

    print(
        MODEL_DIR / "final_scaler.pkl"
    )


if __name__ == "__main__":
    main()