from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as functional


class OrdinalDistanceLoss(nn.Module):
    def __init__(self, num_classes: int = 5, power: float = 1.0) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("Ordinal loss needs at least two classes")
        if power <= 0.0:
            raise ValueError("Distance power must be positive")
        self.num_classes = num_classes
        self.power = power

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probabilities = torch.softmax(logits, dim=1)
        classes = torch.arange(
            self.num_classes,
            device=logits.device,
            dtype=logits.dtype,
        )
        distances = torch.abs(classes.unsqueeze(0) - targets.unsqueeze(1)).pow(self.power)
        return torch.mean(torch.sum(probabilities * distances, dim=1))


class OrdinalCrossEntropy(nn.Module):
    def __init__(
        self,
        num_classes: int = 5,
        ordinal_weight: float = 0.5,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        if not 0.0 <= ordinal_weight <= 1.0:
            raise ValueError("Ordinal weight must be in [0, 1]")
        self.ordinal_weight = ordinal_weight
        self.label_smoothing = label_smoothing
        self.ordinal = OrdinalDistanceLoss(num_classes)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        categorical = functional.cross_entropy(
            logits,
            targets,
            label_smoothing=self.label_smoothing,
        )
        ordinal = self.ordinal(logits, targets)
        return (1.0 - self.ordinal_weight) * categorical + self.ordinal_weight * ordinal


class CumulativeOrdinalLoss(nn.Module):
    def __init__(self, num_classes: int = 5) -> None:
        super().__init__()
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.shape[1] != self.num_classes - 1:
            raise ValueError("Cumulative logits need num_classes - 1 columns")
        thresholds = torch.arange(1, self.num_classes, device=targets.device)
        cumulative_targets = (targets.unsqueeze(1) >= thresholds.unsqueeze(0)).to(logits.dtype)
        return functional.binary_cross_entropy_with_logits(logits, cumulative_targets)


class FairnessRegularizedLoss(nn.Module):
    def __init__(self, base_loss: nn.Module, fairness_weight: float = 0.1) -> None:
        super().__init__()
        if fairness_weight < 0.0:
            raise ValueError("Fairness weight cannot be negative")
        self.base_loss = base_loss
        self.fairness_weight = fairness_weight

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        group_indicator: torch.Tensor,
    ) -> torch.Tensor:
        base = self.base_loss(logits, targets)
        probabilities = torch.softmax(logits, dim=1)
        left = group_indicator == 0
        right = group_indicator == 1
        if not torch.any(left) or not torch.any(right):
            return base
        disparity = torch.mean(
            torch.abs(probabilities[left].mean(0) - probabilities[right].mean(0))
        )
        return base + self.fairness_weight * disparity
