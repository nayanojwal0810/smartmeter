"""1D ResNet Reference Model Architecture.

Implements the standard 3-block 1D ResNet for time-series classification
as defined in the project execution plan:
- Input: (B, 1, 510) or (B, 510) aggregate power in Watts
- 3 residual blocks with filter counts {64, 128, 128}
- 3 convolutional layers per block with kernel pattern {8, 5, 3} and 'same' padding
- Batch normalization and ReLU activations on each conv layer
- Shortcut projection (1x1 Conv + BatchNorm) when channel dimensions change; BatchNorm when channels match
- Global Average Pooling (AdaptiveAvgPool1d) reducing temporal dimension
- Final Linear layer outputting a single raw logit per window
- Sigmoid applied only for probability estimation / reporting
"""

from __future__ import annotations

from typing import Sequence
import torch
import torch.nn as nn


class ResNetBlock1D(nn.Module):
    """1D Residual Block comprising 3 convolutional layers and a shortcut connection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: Sequence[int] = (8, 5, 3),
    ) -> None:
        super().__init__()
        k1, k2, k3 = kernel_sizes

        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=k1,
            stride=1,
            padding="same",
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=k2,
            stride=1,
            padding="same",
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=k3,
            stride=1,
            padding="same",
            bias=False,
        )
        self.bn3 = nn.BatchNorm1d(out_channels)

        # Shortcut projection if channel dimension changes, else BatchNorm identity
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.BatchNorm1d(out_channels)

        self.relu_out = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through residual block with skip connection."""
        res = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = out + res
        return self.relu_out(out)


class ResNet1D(nn.Module):
    """3-Block 1D ResNet Reference Model for weakly supervised appliance presence classification."""

    def __init__(
        self,
        in_channels: int = 1,
        filter_counts: Sequence[int] = (64, 128, 128),
        kernel_sizes: Sequence[int] = (8, 5, 3),
        out_features: int = 1,
    ) -> None:
        super().__init__()
        if len(filter_counts) != 3:
            raise ValueError(f"Expected exactly 3 filter counts, got {len(filter_counts)}")
        if len(kernel_sizes) != 3:
            raise ValueError(f"Expected exactly 3 kernel sizes, got {len(kernel_sizes)}")

        f1, f2, f3 = filter_counts

        self.block1 = ResNetBlock1D(in_channels=in_channels, out_channels=f1, kernel_sizes=kernel_sizes)
        self.block2 = ResNetBlock1D(in_channels=f1, out_channels=f2, kernel_sizes=kernel_sizes)
        self.block3 = ResNetBlock1D(in_channels=f2, out_channels=f3, kernel_sizes=kernel_sizes)

        self.gap = nn.AdaptiveAvgPool1d(output_size=1)
        self.classifier = nn.Linear(in_features=f3, out_features=out_features)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass up to final convolutional feature map before GAP.

        Args:
            x: Input tensor of shape (B, 1, 510) or (B, 510).

        Returns:
            Feature map tensor of shape (B, 128, 510).
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.dim() != 3:
            raise ValueError(f"Expected 2D or 3D input tensor, got shape {x.shape}")

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x

    def extract_cam(self, x: torch.Tensor) -> torch.Tensor:
        """Extract raw Class Activation Map (CAM) for presence class before GAP.

        Args:
            x: Input tensor of shape (B, 1, 510) or (B, 510).

        Returns:
            CAM tensor of shape (B, 510).
        """
        features = self.forward_features(x)  # (B, C, T)
        # weights: (out_features, in_features) -> (1, C) -> squeeze to (C,)
        w = self.classifier.weight[0]  # (C,)
        # Weighted sum across channels: (B, C, T) * (C, 1) -> sum along C -> (B, T)
        cam = torch.einsum("bct,c->bt", features, w)
        return cam

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass computing scalar logit per window.

        Args:
            x: Input tensor of shape (B, 1, 510) or (B, 510).

        Returns:
            Raw logit tensor of shape (B, 1).
        """
        features = self.forward_features(x)
        feat = self.gap(features)  # (B, 128, 1)
        feat = feat.squeeze(-1)  # (B, 128)
        logits = self.classifier(feat)  # (B, 1)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Compute predicted class probabilities via sigmoid."""
        logits = self.forward(x)
        return torch.sigmoid(logits)

