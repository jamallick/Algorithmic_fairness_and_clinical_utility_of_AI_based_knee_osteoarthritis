from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class LogitEnsemble(nn.Module):
    def __init__(
        self,
        members: Sequence[nn.Module],
        weights: Sequence[float] | None = None,
    ) -> None:
        super().__init__()
        if not members:
            raise ValueError("An ensemble needs at least one member")
        self.members = nn.ModuleList(members)
        raw = list(weights) if weights is not None else [1.0] * len(members)
        if len(raw) != len(members):
            raise ValueError("Weights must align with members")
        tensor = torch.tensor(raw, dtype=torch.float32)
        self.register_buffer("weights", tensor / tensor.sum())

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = torch.stack([member(inputs) for member in self.members], dim=0)
        shape = (self.weights.shape[0],) + (1,) * (outputs.ndim - 1)
        return torch.sum(outputs * self.weights.view(shape), dim=0)


class ProbabilityEnsemble(nn.Module):
    def __init__(
        self,
        members: Sequence[nn.Module],
        weights: Sequence[float] | None = None,
    ) -> None:
        super().__init__()
        if not members:
            raise ValueError("An ensemble needs at least one member")
        self.members = nn.ModuleList(members)
        raw = list(weights) if weights is not None else [1.0] * len(members)
        tensor = torch.tensor(raw, dtype=torch.float32)
        self.register_buffer("weights", tensor / tensor.sum())

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        probabilities = torch.stack(
            [torch.softmax(member(inputs), dim=1) for member in self.members], dim=0
        )
        shape = (self.weights.shape[0],) + (1,) * (probabilities.ndim - 1)
        mixture = torch.sum(probabilities * self.weights.view(shape), dim=0)
        return torch.log(mixture.clamp_min(1e-12))
