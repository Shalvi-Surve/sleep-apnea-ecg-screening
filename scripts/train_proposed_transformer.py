import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import time
import random

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from torch.utils.data import TensorDataset, DataLoader

from src.models.proposed_transformer import (
    ProposedLightweightTransformer,
)

ROOT = Path(__file__).resolve().parents[1]

X_PATH = ROOT / "results" / "experiments" / "temporal_X.npy"
Y_PATH = ROOT / "results" / "experiments" / "temporal_y.npy"
GROUP_PATH = (
    ROOT
    / "results"
    / "experiments"
    / "temporal_groups.npy"
)

MODEL_DIR = ROOT / "models" / "trained"
METRIC_DIR = ROOT / "results" / "metrics"
PRED_DIR = ROOT / "results" / "predictions"
COMPLEXITY_DIR = ROOT / "results" / "complexity"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
METRIC_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR.mkdir(parents=True, exist_ok=True)
COMPLEXITY_DIR.mkdir(parents=True, exist_ok=True)


SEED = 42
EPOCHS = 20
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
PATIENCE = 4


def seed_everything(seed=SEED):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def metrics_from_predictions(
    y_true,
    y_prob,
):

    y_pred = (
        y_prob >= 0.5
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
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

    return {
        "accuracy": accuracy_score(
            y_true,
            y_pred,
        ),
        "balanced_accuracy": balanced_accuracy_score(
            y_true,
            y_pred,
        ),
        "precision": precision_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(
            y_true,
            y_prob,
        ),
        "pr_auc": average_precision_score(
            y_true,
            y_prob,
        ),
    }


def make_loader(
    X,
    y,
    shuffle,
):

    dataset = torch.utils.data.TensorDataset(
        torch.tensor(
            X,
            dtype=torch.float32,
        ),
        torch.tensor(
            y,
            dtype=torch.float32,
        ),
    )

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
    )


def train_fold(
    X_train,
    y_train,
    X_val,
    y_val,
    device,
):

    model = ProposedLightweightTransformer(
        input_dim=X_train.shape[-1],
        d_model=64,
        n_heads=4,
        num_layers=2,
        ff_dim=128,
        dropout=0.20,
        max_len=X_train.shape[1],
    ).to(device)

    positive = np.sum(y_train == 1)
    negative = np.sum(y_train == 0)

    pos_weight = (
        negative / max(positive, 1)
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            pos_weight,
            dtype=torch.float32,
            device=device,
        )
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
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

    best_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        model.train()

        train_losses = []

        for xb, yb in train_loader:

            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()

            logits = model(xb)

            loss = criterion(
                logits,
                yb,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            optimizer.step()

            train_losses.append(
                loss.item()
            )

        model.eval()

        val_losses = []
        val_probs = []

        with torch.no_grad():

            for xb, yb in val_loader:

                xb = xb.to(device)
                yb = yb.to(device)

                logits = model(xb)

                loss = criterion(
                    logits,
                    yb,
                )

                val_losses.append(
                    loss.item()
                )

                probs = torch.sigmoid(
                    logits
                )

                val_probs.extend(
                    probs.cpu().numpy()
                )

        train_loss = np.mean(
            train_losses
        )

        val_loss = np.mean(
            val_losses
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train={train_loss:.4f} | "
            f"Val={val_loss:.4f}"
        )

        if val_loss < best_loss:

            best_loss = val_loss

            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

            patience_counter = 0

        else:

            patience_counter += 1

            if patience_counter >= PATIENCE:

                print(
                    "Early stopping."
                )

                break

    model.load_state_dict(
        best_state
    )

    model.eval()

    probabilities = []

    with torch.no_grad():

        for xb, _ in val_loader:

            xb = xb.to(device)

            logits = model(
                xb
            )

            probs = torch.sigmoid(
                logits
            )

            probabilities.extend(
                probs.cpu().numpy()
            )

    return (
        model,
        np.asarray(
            probabilities
        ),
    )


def main():

    seed_everything()

    device = torch.device(
        "cpu"
    )

    print("=" * 70)
    print(
        "PROPOSED LIGHTWEIGHT TRANSFORMER V2"
    )
    print("=" * 70)

    print(
        f"Device: {device}"
    )

    X = np.load(
        X_PATH
    )

    y = np.load(
        Y_PATH
    )

    groups = np.load(
        GROUP_PATH
    )

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
        f"Class 0      : {np.sum(y == 0)}"
    )

    print(
        f"Class 1      : {np.sum(y == 1)}"
    )

    cv = GroupKFold(
        n_splits=5
    )

    fold_metrics = []

    all_predictions = []

    final_model = None

    for fold, (
        train_idx,
        val_idx,
    ) in enumerate(
        cv.split(
            X,
            y,
            groups,
        ),
        start=1,
    ):

        print()
        print("=" * 70)
        print(
            f"PROPOSED TRANSFORMER — FOLD {fold}/5"
        )
        print("=" * 70)

        model, probs = train_fold(
            X[train_idx],
            y[train_idx],
            X[val_idx],
            y[val_idx],
            device,
        )

        fold_result = metrics_from_predictions(
            y[val_idx],
            probs,
        )

        fold_result["fold"] = fold

        fold_metrics.append(
            fold_result
        )

        print(
            f"Fold {fold}: "
            f"F1={fold_result['f1']:.4f} | "
            f"Sensitivity={fold_result['sensitivity']:.4f} | "
            f"Specificity={fold_result['specificity']:.4f} | "
            f"ROC-AUC={fold_result['roc_auc']:.4f}"
        )

        fold_predictions = pd.DataFrame(
            {
                "fold": fold,
                "index": val_idx,
                "group": groups[val_idx],
                "y_true": y[val_idx],
                "y_prob": probs,
                "y_pred": (
                    probs >= 0.5
                ).astype(int),
            }
        )

        all_predictions.append(
            fold_predictions
        )

        final_model = model

    metrics_df = pd.DataFrame(
        fold_metrics
    )

    summary = {}

    for metric in [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "sensitivity",
        "specificity",
        "f1",
        "roc_auc",
        "pr_auc",
    ]:

        summary[metric] = float(
            metrics_df[metric].mean()
        )

        summary[
            f"{metric}_std"
        ] = float(
            metrics_df[metric].std(
                ddof=1
            )
        )

    print()
    print("=" * 70)
    print(
        "PROPOSED TRANSFORMER V2 — FINAL SUMMARY"
    )
    print("=" * 70)

    for metric in [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "sensitivity",
        "specificity",
        "f1",
        "roc_auc",
        "pr_auc",
    ]:

        print(
            f"{metric:<20}: "
            f"{summary[metric]:.4f} ± "
            f"{summary[f'{metric}_std']:.4f}"
        )

    # Save fold metrics
    metrics_df.to_csv(
        METRIC_DIR
        / "proposed_transformer_v2_folds.csv",
        index=False,
    )

    # Save summary
    with open(
        METRIC_DIR
        / "proposed_transformer_v2_summary.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    # Save predictions
    prediction_df = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    prediction_df.to_csv(
        PRED_DIR
        / "proposed_transformer_v2_predictions.csv",
        index=False,
    )

    # Save final model
    model_path = (
        MODEL_DIR
        / "proposed_transformer_v2_best.pt"
    )

    torch.save(
        final_model.state_dict(),
        model_path,
    )

    # Complexity
    parameters = sum(
        p.numel()
        for p in final_model.parameters()
    )

    trainable = sum(
        p.numel()
        for p in final_model.parameters()
        if p.requires_grad
    )

    parameter_size_mb = (
        parameters * 4
    ) / (
        1024 ** 2
    )

    sample = torch.randn(
        1,
        X.shape[1],
        X.shape[2],
    ).to(device)

    for _ in range(10):

        with torch.no_grad():
            final_model(sample)

    start = time.perf_counter()

    with torch.no_grad():

        for _ in range(100):

            final_model(sample)

    elapsed = (
        time.perf_counter()
        - start
    )

    latency_ms = (
        elapsed
        / 100
        * 1000
    )

    complexity = {
        "model": "Proposed Lightweight Transformer V2",
        "parameters": parameters,
        "trainable_parameters": trainable,
        "parameter_size_mb": parameter_size_mb,
        "cpu_inference_latency_ms": latency_ms,
    }

    with open(
        COMPLEXITY_DIR
        / "proposed_transformer_v2_complexity.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            complexity,
            f,
            indent=2,
        )

    print()
    print("=" * 70)
    print("COMPLEXITY")
    print("=" * 70)

    print(
        f"Parameters       : {parameters:,}"
    )

    print(
        f"Trainable        : {trainable:,}"
    )

    print(
        f"Model size       : {parameter_size_mb:.4f} MB"
    )

    print(
        f"CPU latency      : {latency_ms:.4f} ms"
    )

    print()
    print("Saved:")
    print(
        model_path
    )

    print(
        METRIC_DIR
        / "proposed_transformer_v2_summary.json"
    )

    print(
        PRED_DIR
        / "proposed_transformer_v2_predictions.csv"
    )

    print("=" * 70)
    print(
        "PROPOSED TRANSFORMER V2 COMPLETE"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()