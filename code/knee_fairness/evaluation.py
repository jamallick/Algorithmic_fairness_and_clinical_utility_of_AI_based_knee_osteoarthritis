from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
)
from torch import nn
from torch.utils.data import DataLoader

from knee_fairness.fairness import audit_binary_groups, macro_auc
from knee_fairness.schema import FairnessSummary, PredictionBatch


@dataclass(frozen=True)
class PerformanceSummary:
    accuracy: float
    weighted_kappa: float
    macro_auc: float
    macro_f1: float
    mean_absolute_error: float
    sensitivity: tuple[float, ...]
    specificity: tuple[float, ...]
    confusion: tuple[tuple[int, ...], ...]


def performance_summary(batch: PredictionBatch, num_classes: int = 5) -> PerformanceSummary:
    batch.validate(num_classes)
    predictions = batch.predictions
    matrix = confusion_matrix(batch.labels, predictions, labels=np.arange(num_classes))
    sensitivity: list[float] = []
    specificity: list[float] = []
    for grade in range(num_classes):
        true_positive = matrix[grade, grade]
        false_negative = matrix[grade, :].sum() - true_positive
        false_positive = matrix[:, grade].sum() - true_positive
        true_negative = matrix.sum() - true_positive - false_negative - false_positive
        sensitivity.append(_divide(true_positive, true_positive + false_negative))
        specificity.append(_divide(true_negative, true_negative + false_positive))
    return PerformanceSummary(
        accuracy=float(accuracy_score(batch.labels, predictions)),
        weighted_kappa=float(cohen_kappa_score(batch.labels, predictions, weights="quadratic")),
        macro_auc=macro_auc(batch.labels, batch.probabilities, num_classes),
        macro_f1=float(f1_score(batch.labels, predictions, average="macro")),
        mean_absolute_error=float(mean_absolute_error(batch.labels, predictions)),
        sensitivity=tuple(sensitivity),
        specificity=tuple(specificity),
        confusion=tuple(tuple(int(value) for value in row) for row in matrix),
    )


def _divide(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else float("nan")


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader[Mapping[str, Any]],
    device: torch.device,
    group_key: str = "race",
) -> PredictionBatch:
    model.eval()
    model.to(device)
    labels: list[NDArray[np.int64]] = []
    probabilities: list[NDArray[np.float64]] = []
    groups: list[NDArray[np.str_]] = []
    participants: list[NDArray[np.str_]] = []
    for batch in loader:
        images = batch["image"].to(device)
        logits = model(images)
        labels.append(batch["grade"].numpy().astype(np.int64))
        probabilities.append(torch.softmax(logits, dim=1).cpu().numpy().astype(np.float64))
        groups.append(np.asarray(batch[group_key], dtype=str))
        participants.append(np.asarray(batch["participant_id"], dtype=str))
    return PredictionBatch(
        labels=np.concatenate(labels),
        probabilities=np.concatenate(probabilities),
        groups=np.concatenate(groups),
        participant_ids=np.concatenate(participants),
    )


def multi_axis_audit(
    labels: NDArray[np.int64],
    probabilities: NDArray[np.float64],
    axes: Mapping[str, NDArray[np.str_]],
    comparisons: Mapping[str, tuple[str, str]],
    participant_ids: NDArray[np.str_] | None = None,
) -> dict[str, FairnessSummary]:
    output: dict[str, FairnessSummary] = {}
    for axis, groups in axes.items():
        if axis not in comparisons:
            continue
        left, right = comparisons[axis]
        batch = PredictionBatch(labels, probabilities, groups, participant_ids)
        output[axis] = audit_binary_groups(batch, left, right)
    return output


def summaries_to_records(
    performance: PerformanceSummary,
    fairness: Mapping[str, FairnessSummary],
) -> dict[str, Any]:
    return {
        "performance": asdict(performance),
        "fairness": {axis: asdict(summary) for axis, summary in fairness.items()},
    }


def temporal_gap_matrix(
    batches: Mapping[str, PredictionBatch],
    left_group: str,
    right_group: str,
) -> tuple[tuple[str, float], ...]:
    values: list[tuple[str, float]] = []
    for timepoint, batch in batches.items():
        summary = audit_binary_groups(batch, left_group, right_group)
        values.append((timepoint, summary.worst_group_gap))
    return tuple(values)


def enforce_minimum_cells(
    groups: NDArray[np.str_],
    minimum: int = 30,
) -> tuple[str, ...]:
    accepted: list[str] = []
    for group in np.unique(groups):
        if int(np.sum(groups == group)) >= minimum:
            accepted.append(str(group))
    return tuple(accepted)
