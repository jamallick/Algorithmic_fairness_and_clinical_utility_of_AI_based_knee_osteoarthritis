from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler


def build_optimizer(
    parameters: Iterable[nn.Parameter],
    name: str,
    learning_rate: float,
    weight_decay: float,
) -> Optimizer:
    if learning_rate <= 0.0:
        raise ValueError("Learning rate must be positive")
    if weight_decay < 0.0:
        raise ValueError("Weight decay cannot be negative")
    if name == "adamw":
        return torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(parameters, lr=learning_rate, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=learning_rate,
            momentum=0.9,
            nesterov=True,
            weight_decay=weight_decay,
        )
    raise ValueError(f"Unknown optimizer: {name}")


def build_scheduler(
    optimizer: Optimizer,
    name: str,
    total_steps: int,
    warmup_steps: int = 0,
) -> LRScheduler:
    if total_steps < 1:
        raise ValueError("Total steps must be positive")

    def multiplier(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return max(1e-8, step / warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        if name == "constant":
            return 1.0
        if name == "linear":
            return 1.0 - progress
        if name == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        raise ValueError(f"Unknown scheduler: {name}")

    return LambdaLR(optimizer, multiplier)
