from __future__ import annotations

import torch
from torch import nn

from knee_fairness.models.blocks import DenseStage, Transition


class KneeDenseNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 5,
        growth_rate: int = 24,
        block_counts: tuple[int, ...] = (6, 12, 24, 16),
        initial_channels: int = 48,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, initial_channels, 7, 2, 3, bias=False),
            nn.BatchNorm2d(initial_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
        )
        stages: list[nn.Module] = []
        channels = initial_channels
        for index, count in enumerate(block_counts):
            stage = DenseStage(channels, count, growth_rate)
            stages.append(stage)
            channels = stage.output_channels
            if index < len(block_counts) - 1:
                output_channels = channels // 2
                stages.append(Transition(channels, output_channels))
                channels = output_channels
        self.features = nn.Sequential(*stages)
        self.normalization = nn.BatchNorm2d(channels)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(channels, num_classes)

    def forward_features(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.stem(inputs)
        output = torch.relu(self.normalization(self.features(output)))
        return self.pool(output).flatten(1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.dropout(self.forward_features(inputs)))
