from __future__ import annotations

import torch
from torch import nn

from knee_fairness.models.blocks import (
    AttentionPool,
    ConvNormActivation,
    PatchEmbedding,
    ResidualDepthwiseBlock,
    TransformerBlock,
)


class EfficientBranch(nn.Module):
    def __init__(self, dimension: int = 256) -> None:
        super().__init__()
        channels = (32, 48, 80, 128, dimension)
        self.stem = ConvNormActivation(1, channels[0], 3, 2)
        blocks: list[nn.Module] = []
        for input_channels, output_channels in zip(channels[:-1], channels[1:], strict=True):
            blocks.append(ResidualDepthwiseBlock(input_channels, output_channels, 2))
            blocks.append(ResidualDepthwiseBlock(output_channels, output_channels, 1))
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.pool(self.blocks(self.stem(inputs))).flatten(1)


class SwinBranch(nn.Module):
    def __init__(
        self,
        dimension: int = 256,
        depth: int = 4,
        heads: int = 8,
    ) -> None:
        super().__init__()
        self.embedding = PatchEmbedding(1, dimension, 4)
        self.blocks = nn.Sequential(
            *[TransformerBlock(dimension, heads, 4, 0.1) for _ in range(depth)]
        )
        self.pool = AttentionPool(dimension, heads)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        tokens, _, _ = self.embedding(inputs)
        return self.pool(self.blocks(tokens))


class PluginInteraction(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.efficient_gate = nn.Sequential(nn.Linear(dimension, dimension), nn.Sigmoid())
        self.transformer_gate = nn.Sequential(nn.Linear(dimension, dimension), nn.Sigmoid())
        self.mixer = nn.Sequential(
            nn.Linear(dimension * 4, dimension * 2),
            nn.GELU(),
            nn.LayerNorm(dimension * 2),
            nn.Linear(dimension * 2, dimension),
        )

    def forward(self, efficient: torch.Tensor, transformer: torch.Tensor) -> torch.Tensor:
        efficient_context = efficient * self.transformer_gate(transformer)
        transformer_context = transformer * self.efficient_gate(efficient)
        product = efficient * transformer
        difference = torch.abs(efficient - transformer)
        return self.mixer(
            torch.cat([efficient_context, transformer_context, product, difference], dim=1)
        )


class PluginInteractionModel(nn.Module):
    def __init__(self, num_classes: int = 5, dimension: int = 256) -> None:
        super().__init__()
        self.efficient_branch = EfficientBranch(dimension)
        self.swin_branch = SwinBranch(dimension)
        self.interaction = PluginInteraction(dimension)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(dimension, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        efficient = self.efficient_branch(inputs)
        transformer = self.swin_branch(inputs)
        return self.classifier(self.interaction(efficient, transformer))
