from __future__ import annotations

import torch
import torch.nn as nn


class ProposedLightweightTransformer(nn.Module):
    """
    Feature-Gated Attention-Pooled Lightweight Transformer.

    Input:
        (batch, sequence_length, feature_count)

    Default:
        sequence_length = 5 minutes
        feature_count = 11 ECG/HRV features

    Design:
        ECG/HRV features
            -> feature gating
            -> linear projection
            -> positional encoding
            -> lightweight Transformer encoder
            -> attention pooling
            -> binary classifier
    """

    def __init__(
        self,
        input_dim: int = 11,
        seq_len: int = 5,
        max_len: int | None = None,
        d_model: int = 48,
        n_heads: int = 4,
        num_layers: int = 2,
        ff_dim: int = 96,
        dropout: float = 0.20,
    ):

        super().__init__()

        self.input_dim = input_dim
        self.seq_len = seq_len
        self.max_len = seq_len
        self.d_model = d_model

        # ---------------------------------------------------------
        # 1. Feature gating
        # ---------------------------------------------------------
        self.feature_gate = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid(),
        )

        # ---------------------------------------------------------
        # 2. Input projection
        # ---------------------------------------------------------
        self.input_projection = nn.Linear(
            input_dim,
            d_model
        )

        # ---------------------------------------------------------
        # 3. Learnable positional encoding
        # ---------------------------------------------------------
        self.positional_encoding = nn.Parameter(
            torch.zeros(1, seq_len, d_model)
        )

        # ---------------------------------------------------------
        # 4. Lightweight Transformer encoder
        # ---------------------------------------------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # ---------------------------------------------------------
        # 5. Temporal attention pooling
        # ---------------------------------------------------------
        self.attention_pool = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )

        # ---------------------------------------------------------
        # 6. Classification head
        # ---------------------------------------------------------
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.normal_(
            self.positional_encoding,
            mean=0.0,
            std=0.02,
        )

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns logits with shape (batch,).
        """

        # ---------------------------------------------------------
        # Feature gating
        # ---------------------------------------------------------
        gate = self.feature_gate(x)
        x = x * gate

        # ---------------------------------------------------------
        # Project features
        # ---------------------------------------------------------
        x = self.input_projection(x)

        # ---------------------------------------------------------
        # Add temporal position information
        # ---------------------------------------------------------
        x = x + self.positional_encoding[:, :x.size(1), :]

        # ---------------------------------------------------------
        # Transformer encoder
        # ---------------------------------------------------------
        x = self.encoder(x)

        # ---------------------------------------------------------
        # Attention-based temporal pooling
        # ---------------------------------------------------------
        attention_scores = self.attention_pool(x)

        attention_weights = torch.softmax(
            attention_scores,
            dim=1,
        )

        pooled = torch.sum(
            x * attention_weights,
            dim=1,
        )

        # ---------------------------------------------------------
        # Binary classification
        # ---------------------------------------------------------
        logits = self.classifier(pooled)

        return logits.squeeze(-1)