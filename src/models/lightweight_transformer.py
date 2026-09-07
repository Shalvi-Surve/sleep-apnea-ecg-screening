from __future__ import annotations

import torch
import torch.nn as nn


class LightweightTransformer(
    nn.Module
):

    def __init__(
        self,
        n_features=11,
        d_model=64,
        n_heads=4,
        n_layers=2,
        dim_feedforward=128,
        dropout=0.2,
    ):

        super().__init__()

        self.input_projection = nn.Linear(
            n_features,
            d_model,
        )

        self.positional_encoding = nn.Parameter(
            torch.zeros(
                1,
                5,
                d_model,
            )
        )

        encoder_layer = (
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
        )

        self.encoder = (
            nn.TransformerEncoder(
                encoder_layer,
                num_layers=n_layers,
            )
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(
                d_model
            ),

            nn.Linear(
                d_model,
                16,
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                16,
                1,
            ),
        )

    def forward(self, x):

        x = self.input_projection(
            x
        )

        x = x + self.positional_encoding[
            :, :x.size(1), :
        ]

        x = self.encoder(
            x
        )

        x = x.mean(
            dim=1
        )

        return self.classifier(
            x
        ).squeeze(1)