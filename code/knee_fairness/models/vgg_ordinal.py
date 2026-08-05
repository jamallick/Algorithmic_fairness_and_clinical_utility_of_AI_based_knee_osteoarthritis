from __future__ import annotations

import torch
from torch import nn


def convolution_stack(input_channels: int, output_channels: int, count: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    channels = input_channels
    for _ in range(count):
        layers.extend(
            [
                nn.Conv2d(channels, output_channels, 3, padding=1),
                nn.ReLU(inplace=True),
            ]
        )
        channels = output_channels
    layers.append(nn.MaxPool2d(2, 2))
    return nn.Sequential(*layers)


class OrdinalVGG19(nn.Module):
    def __init__(self, num_classes: int = 5, dropout: float = 0.5) -> None:
        super().__init__()
        self.features = nn.Sequential(
            convolution_stack(1, 64, 2),
            convolution_stack(64, 128, 2),
            convolution_stack(128, 256, 4),
            convolution_stack(256, 512, 4),
            convolution_stack(512, 512, 4),
        )
        self.pool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(4096, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(inputs)).flatten(1))
