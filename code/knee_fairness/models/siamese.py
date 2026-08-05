from __future__ import annotations

import torch
from torch import nn

from knee_fairness.models.blocks import ConvNormActivation


class CondyleEncoder(nn.Module):
    def __init__(self, input_channels: int = 1, width: int = 32) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvNormActivation(input_channels, width, 5, 2),
            nn.MaxPool2d(2),
            ConvNormActivation(width, width * 2, 3),
            ConvNormActivation(width * 2, width * 2, 3),
            nn.MaxPool2d(2),
            ConvNormActivation(width * 2, width * 4, 3),
            ConvNormActivation(width * 4, width * 4, 3),
            nn.AdaptiveAvgPool2d(1),
        )
        self.output_dimension = width * 4

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.features(inputs).flatten(1)


class SiameseKneeClassifier(nn.Module):
    def __init__(self, num_classes: int = 5, width: int = 32, dropout: float = 0.3) -> None:
        super().__init__()
        self.encoder = CondyleEncoder(1, width)
        dimension = self.encoder.output_dimension
        self.classifier = nn.Sequential(
            nn.Linear(dimension * 3, dimension * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dimension * 2, num_classes),
        )

    def split_condyles(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        midpoint = inputs.shape[-1] // 2
        medial = inputs[..., :midpoint]
        lateral = torch.flip(inputs[..., midpoint:], dims=(-1,))
        minimum_width = min(medial.shape[-1], lateral.shape[-1])
        return medial[..., :minimum_width], lateral[..., :minimum_width]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        medial, lateral = self.split_condyles(inputs)
        medial_features = self.encoder(medial)
        lateral_features = self.encoder(lateral)
        difference = torch.abs(medial_features - lateral_features)
        return self.classifier(torch.cat([medial_features, lateral_features, difference], dim=1))
