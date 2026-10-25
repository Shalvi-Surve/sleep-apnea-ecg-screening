from __future__ import annotations

import torch
import torch.nn as nn


class LightweightCNN1D(nn.Module):

    def __init__(
        self,
        n_features=11,
        hidden_channels=32,
        dropout=0.2,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv1d(
                in_channels=n_features,
                out_channels=hidden_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm1d(
                hidden_channels
            ),
            nn.ReLU(),

            nn.Conv1d(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1),

            nn.Flatten(),

            nn.Linear(
                hidden_channels,
                16,
            ),

            nn.ReLU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                16,
                1,
            ),
        )

    def forward(self, x):

        return self.network(
            x
        ).squeeze(1)