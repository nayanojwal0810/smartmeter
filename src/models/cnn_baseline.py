"""Simple 1D CNN Baseline Model Architecture (Decision D-008).

Deliberately minimal 1D CNN for weakly supervised binary classification of
510-point aggregate electricity consumption windows:
- Input: (B, 1, 510) or (B, 510)
- Sequence: Conv1d(32, k=9) -> ReLU -> MaxPool1d(2) -> Conv1d(64, k=9) -> ReLU -> AdaptiveAvgPool1d(1) -> Flatten -> Linear(64, 1)
- Output: (B, 1) single raw logit per window
- Strictly NO BatchNorm, Dropout, attention, residual connections, or CAM logic in this baseline.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Simple1DCNN(nn.Module):
    """Minimal 1D CNN baseline for weakly supervised appliance presence classification."""

    def __init__(
        self,
        in_channels: int = 1,
        conv1_channels: int = 32,
        conv1_kernel: int = 9,
        pool_kernel: int = 2,
        conv2_channels: int = 64,
        conv2_kernel: int = 9,
        out_features: int = 1,
    ) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=conv1_channels,
                kernel_size=conv1_kernel,
                stride=1,
                padding=0,
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=pool_kernel),
            nn.Conv1d(
                in_channels=conv1_channels,
                out_channels=conv2_channels,
                kernel_size=conv2_kernel,
                stride=1,
                padding=0,
            ),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(output_size=1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=conv2_channels, out_features=out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass computing raw classification logits.

        Args:
            x: Input tensor of shape (B, 1, 510) or (B, 510).

        Returns:
            Logit tensor of shape (B, 1).
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 510) -> (B, 1, 510)
        elif x.dim() != 3:
            raise ValueError(f"Expected 2D or 3D input tensor, got shape {x.shape}")

        feat = self.features(x)
        logits = self.classifier(feat)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Compute predicted class probabilities via sigmoid."""
        logits = self.forward(x)
        return torch.sigmoid(logits)
