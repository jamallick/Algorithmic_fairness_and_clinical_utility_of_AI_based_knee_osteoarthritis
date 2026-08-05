from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class DecisionCurve:
    thresholds: NDArray[np.float64]
    net_benefit: NDArray[np.float64]
    treat_all: NDArray[np.float64]
    treat_none: NDArray[np.float64]


@dataclass(frozen=True)
class Reclassification:
    event_improvement: float
    nonevent_improvement: float
    net_improvement: float


def net_benefit(
    labels: NDArray[np.int64],
    probabilities: NDArray[np.float64],
    threshold: float,
) -> float:
    if not 0.0 < threshold < 1.0:
        raise ValueError("Threshold must be in (0, 1)")
    predictions = probabilities >= threshold
    positives = labels == 1
    true_positives = int(np.logical_and(predictions, positives).sum())
    false_positives = int(np.logical_and(predictions, ~positives).sum())
    odds = threshold / (1.0 - threshold)
    return true_positives / labels.size - false_positives / labels.size * odds


def decision_curve(
    labels: NDArray[np.int64],
    probabilities: NDArray[np.float64],
    thresholds: NDArray[np.float64] | None = None,
) -> DecisionCurve:
    grid = thresholds if thresholds is not None else np.linspace(0.01, 0.99, 99)
    prevalence = float(labels.mean())
    model = np.asarray([net_benefit(labels, probabilities, value) for value in grid])
    all_values = prevalence - (1.0 - prevalence) * grid / (1.0 - grid)
    none_values = np.zeros_like(grid)
    return DecisionCurve(grid, model, all_values, none_values)


def subgroup_decision_curves(
    labels: NDArray[np.int64],
    probabilities: NDArray[np.float64],
    groups: NDArray[np.str_],
    thresholds: NDArray[np.float64] | None = None,
) -> dict[str, DecisionCurve]:
    output: dict[str, DecisionCurve] = {}
    for group in np.unique(groups):
        selected = groups == group
        output[str(group)] = decision_curve(labels[selected], probabilities[selected], thresholds)
    return output


def net_reclassification_improvement(
    labels: NDArray[np.int64],
    reference_probability: NDArray[np.float64],
    candidate_probability: NDArray[np.float64],
    threshold: float = 0.5,
) -> Reclassification:
    if not (labels.shape == reference_probability.shape == candidate_probability.shape):
        raise ValueError("Inputs must align")
    reference = reference_probability >= threshold
    candidate = candidate_probability >= threshold
    events = labels == 1
    nonevents = ~events
    event_up = _conditional_rate(candidate & ~reference, events)
    event_down = _conditional_rate(reference & ~candidate, events)
    nonevent_down = _conditional_rate(reference & ~candidate, nonevents)
    nonevent_up = _conditional_rate(candidate & ~reference, nonevents)
    event_improvement = event_up - event_down
    nonevent_improvement = nonevent_down - nonevent_up
    return Reclassification(
        event_improvement,
        nonevent_improvement,
        event_improvement + nonevent_improvement,
    )


def _conditional_rate(events: NDArray[np.bool_], population: NDArray[np.bool_]) -> float:
    count = int(population.sum())
    if count == 0:
        return float("nan")
    return float(np.logical_and(events, population).sum() / count)


def clinical_consequence_matrix(num_classes: int = 5) -> NDArray[np.float64]:
    grades = np.arange(num_classes)
    distance = np.abs(grades[:, None] - grades[None, :]).astype(np.float64)
    treatment_side = grades >= 2
    boundary_crossing = treatment_side[:, None] != treatment_side[None, :]
    cost = distance
    cost[boundary_crossing] *= 2.0
    np.fill_diagonal(cost, 0.0)
    return cost


def expected_clinical_cost(
    labels: NDArray[np.int64],
    predictions: NDArray[np.int64],
    cost_matrix: NDArray[np.float64] | None = None,
) -> float:
    costs = cost_matrix if cost_matrix is not None else clinical_consequence_matrix()
    return float(np.mean(costs[labels, predictions]))
