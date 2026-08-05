from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar


@dataclass(frozen=True)
class TemperatureFit:
    temperature: float
    negative_log_likelihood: float


@dataclass(frozen=True)
class GroupThresholds:
    values: dict[str, float]
    target_grade: int


def temperature_scale(logits: NDArray[np.float64], temperature: float) -> NDArray[np.float64]:
    if temperature <= 0.0:
        raise ValueError("Temperature must be positive")
    shifted = logits / temperature
    shifted -= shifted.max(axis=1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=1, keepdims=True)


def negative_log_likelihood(
    logits: NDArray[np.float64], labels: NDArray[np.int64], temperature: float
) -> float:
    probabilities = temperature_scale(logits, temperature)
    selected = np.clip(probabilities[np.arange(labels.size), labels], 1e-12, 1.0)
    return float(-np.log(selected).mean())


def fit_temperature(logits: NDArray[np.float64], labels: NDArray[np.int64]) -> TemperatureFit:
    def objective(value: float) -> float:
        return negative_log_likelihood(logits, labels, float(value))

    result = minimize_scalar(objective, bounds=(0.05, 20.0), method="bounded")
    if not result.success:
        raise RuntimeError("Temperature optimization failed")
    return TemperatureFit(float(result.x), float(result.fun))


def youden_threshold(labels: NDArray[np.int64], scores: NDArray[np.float64]) -> float:
    candidates = np.unique(scores)
    best_threshold = 0.5
    best_index = -np.inf
    positives = labels == 1
    negatives = ~positives
    for threshold in candidates:
        predicted = scores >= threshold
        sensitivity = _fraction(predicted, positives)
        specificity = _fraction(~predicted, negatives)
        index = sensitivity + specificity - 1.0
        if index > best_index:
            best_index = index
            best_threshold = float(threshold)
    return best_threshold


def _fraction(event: NDArray[np.bool_], population: NDArray[np.bool_]) -> float:
    count = int(population.sum())
    return float(np.logical_and(event, population).sum() / count) if count else 0.0


def fit_group_thresholds(
    labels: NDArray[np.int64],
    probabilities: NDArray[np.float64],
    groups: NDArray[np.str_],
    target_grade: int = 2,
) -> GroupThresholds:
    binary_labels = (labels >= target_grade).astype(np.int64)
    scores = probabilities[:, target_grade:].sum(axis=1)
    values: dict[str, float] = {}
    for group in np.unique(groups):
        selected = groups == group
        values[str(group)] = youden_threshold(binary_labels[selected], scores[selected])
    return GroupThresholds(values, target_grade)


def apply_group_thresholds(
    probabilities: NDArray[np.float64],
    groups: NDArray[np.str_],
    thresholds: GroupThresholds,
) -> NDArray[np.int64]:
    scores = probabilities[:, thresholds.target_grade :].sum(axis=1)
    default_predictions = probabilities.argmax(axis=1).astype(np.int64)
    output = default_predictions.copy()
    for index, group in enumerate(groups):
        threshold = thresholds.values[str(group)]
        positive = scores[index] >= threshold
        if positive and output[index] < thresholds.target_grade:
            output[index] = thresholds.target_grade
        if not positive and output[index] >= thresholds.target_grade:
            output[index] = thresholds.target_grade - 1
    return output


class TorchTemperatureScaler(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.log_temperature = torch.nn.Parameter(torch.zeros(()))

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp()

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

    def fit(self, logits: torch.Tensor, labels: torch.Tensor, iterations: int = 100) -> float:
        optimizer = torch.optim.LBFGS([self.log_temperature], max_iter=iterations)
        loss_function = torch.nn.CrossEntropyLoss()

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            loss = loss_function(self(logits), labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        return float(self.temperature.detach().cpu())
