from __future__ import annotations

import torch
from torch import nn


class ConvNormActivation(nn.Sequential):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
    ) -> None:
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size,
                stride,
                padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
            nn.SiLU(inplace=True),
        )


class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(8, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.reduce = nn.Conv2d(channels, hidden, 1)
        self.activate = nn.SiLU(inplace=True)
        self.expand = nn.Conv2d(hidden, channels, 1)
        self.gate = nn.Sigmoid()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        weights = self.gate(self.expand(self.activate(self.reduce(self.pool(inputs)))))
        return inputs * weights


class ResidualDepthwiseBlock(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        stride: int = 1,
        expansion: int = 4,
    ) -> None:
        super().__init__()
        hidden = input_channels * expansion
        self.expand = ConvNormActivation(input_channels, hidden, 1)
        self.depthwise = ConvNormActivation(hidden, hidden, 3, stride, hidden)
        self.excitation = SqueezeExcitation(hidden)
        self.project = nn.Sequential(
            nn.Conv2d(hidden, output_channels, 1, bias=False),
            nn.BatchNorm2d(output_channels),
        )
        self.skip = stride == 1 and input_channels == output_channels

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.project(self.excitation(self.depthwise(self.expand(inputs))))
        return inputs + output if self.skip else output


class DenseUnit(nn.Module):
    def __init__(self, input_channels: int, growth_rate: int, bottleneck: int = 4) -> None:
        super().__init__()
        hidden = growth_rate * bottleneck
        self.layers = nn.Sequential(
            nn.BatchNorm2d(input_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, growth_rate, 3, padding=1, bias=False),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.cat([inputs, self.layers(inputs)], dim=1)


class DenseStage(nn.Module):
    def __init__(self, input_channels: int, count: int, growth_rate: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        channels = input_channels
        for _ in range(count):
            layers.append(DenseUnit(channels, growth_rate))
            channels += growth_rate
        self.layers = nn.Sequential(*layers)
        self.output_channels = channels

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class Transition(nn.Sequential):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__(
            nn.BatchNorm2d(input_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_channels, output_channels, 1, bias=False),
            nn.AvgPool2d(2, 2),
        )


class PatchEmbedding(nn.Module):
    def __init__(self, input_channels: int, dimension: int, patch_size: int = 4) -> None:
        super().__init__()
        self.projection = nn.Conv2d(input_channels, dimension, patch_size, patch_size)
        self.normalization = nn.LayerNorm(dimension)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        features = self.projection(inputs)
        height, width = features.shape[-2:]
        tokens = features.flatten(2).transpose(1, 2)
        return self.normalization(tokens), height, width


class FeedForward(nn.Module):
    def __init__(self, dimension: int, expansion: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        hidden = dimension * expansion
        self.layers = nn.Sequential(
            nn.Linear(dimension, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dimension),
            nn.Dropout(dropout),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dimension: int,
        heads: int,
        expansion: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.normalization_one = nn.LayerNorm(dimension)
        self.attention = nn.MultiheadAttention(
            dimension,
            heads,
            dropout=dropout,
            batch_first=True,
        )
        self.normalization_two = nn.LayerNorm(dimension)
        self.feed_forward = FeedForward(dimension, expansion, dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        normalized = self.normalization_one(inputs)
        attended, _ = self.attention(normalized, normalized, normalized, need_weights=False)
        output = inputs + attended
        return output + self.feed_forward(self.normalization_two(output))


class AttentionPool(nn.Module):
    def __init__(self, dimension: int, heads: int) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, dimension))
        nn.init.trunc_normal_(self.query, std=0.02)
        self.attention = nn.MultiheadAttention(dimension, heads, batch_first=True)
        self.normalization = nn.LayerNorm(dimension)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        query = self.query.expand(tokens.shape[0], -1, -1)
        output, _ = self.attention(query, tokens, tokens, need_weights=False)
        return self.normalization(output[:, 0])
