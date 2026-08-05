from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import roc_auc_score

from knee_fairness.schema import FairnessSummary, GradeMetric, PredictionBatch


def _safe_ratio(left: float, right: float) -> float:
    if left == 0.0 and right == 0.0:
        return 1.0
    if left == 0.0 or right == 0.0:
        return 0.0
    return min(left / right, right / left)


def _rate(mask: NDArray[np.bool_], denominator: NDArray[np.bool_]) -> float:
    count = int(denominator.sum())
    if count == 0:
        return float("nan")
    return float(np.logical_and(mask, denominator).sum() / count)


def confusion_rates(
    labels: NDArray[np.int64],
    predictions: NDArray[np.int64],
    grade: int,
) -> tuple[float, float, float]:
    positives = labels == grade
    negatives = ~positives
    predicted_positive = predictions == grade
    tpr = _rate(predicted_positive, positives)
    fpr = _rate(predicted_positive, negatives)
    fnr = _rate(~predicted_positive, positives)
    return tpr, fpr, fnr


def grade_metric(batch: PredictionBatch, group: str, grade: int) -> GradeMetric:
    selected = batch.groups == group
    labels = batch.labels[selected]
    predictions = batch.predictions[selected]
    tpr, fpr, fnr = confusion_rates(labels, predictions, grade)
    true_grade = labels == grade
    prediction_rate = float(np.mean(predictions == grade)) if labels.size else float("nan")
    undergrading_rate = _rate(predictions < grade, true_grade)
    overgrading_rate = _rate(predictions > grade, true_grade)
    return GradeMetric(
        grade=grade,
        group=group,
        true_positive_rate=tpr,
        false_positive_rate=fpr,
        false_negative_rate=fnr,
        prediction_rate=prediction_rate,
        undergrading_rate=undergrading_rate,
        overgrading_rate=overgrading_rate,
        support=int(true_grade.sum()),
    )


def expected_calibration_error(
    labels: NDArray[np.int64],
    probabilities: NDArray[np.float64],
    bins: int = 10,
) -> float:
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = predictions == labels
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    total = labels.size
    if total == 0:
        return float("nan")
    value = 0.0
    for index in range(bins):
        lower = boundaries[index]
        upper = boundaries[index + 1]
        selected = (confidence > lower) & (confidence <= upper)
        if index == 0:
            selected |= confidence == 0.0
        count = int(selected.sum())
        if count:
            accuracy = float(correct[selected].mean())
            mean_confidence = float(confidence[selected].mean())
            value += count / total * abs(accuracy - mean_confidence)
    return value


def brier_score(
    labels: NDArray[np.int64],
    probabilities: NDArray[np.float64],
    num_classes: int = 5,
) -> float:
    targets = np.eye(num_classes, dtype=np.float64)[labels]
    return float(np.mean(np.sum(np.square(probabilities - targets), axis=1)))


def macro_auc(
    labels: NDArray[np.int64],
    probabilities: NDArray[np.float64],
    num_classes: int = 5,
) -> float:
    values: list[float] = []
    for grade in range(num_classes):
        target = (labels == grade).astype(np.int64)
        if np.unique(target).size < 2:
            continue
        values.append(float(roc_auc_score(target, probabilities[:, grade])))
    return float(np.mean(values)) if values else float("nan")


def subgroup_auc(batch: PredictionBatch, group: str, num_classes: int = 5) -> float:
    selected = batch.groups == group
    return macro_auc(batch.labels[selected], batch.probabilities[selected], num_classes)


def equalized_odds_ratio(left: Sequence[GradeMetric], right: Sequence[GradeMetric]) -> float:
    ratios = [
        _safe_ratio(a.true_positive_rate, b.true_positive_rate)
        for a, b in zip(left, right, strict=True)
    ]
    return float(np.nanmean(ratios))


def equalized_odds_difference(left: Sequence[GradeMetric], right: Sequence[GradeMetric]) -> float:
    values = [
        abs(a.false_positive_rate - b.false_positive_rate)
        + abs(a.false_negative_rate - b.false_negative_rate)
        for a, b in zip(left, right, strict=True)
    ]
    return float(np.nanmean(values))


def demographic_parity_difference(
    left: Sequence[GradeMetric], right: Sequence[GradeMetric]
) -> float:
    return float(
        np.nanmean(
            [abs(a.prediction_rate - b.prediction_rate) for a, b in zip(left, right, strict=True)]
        )
    )


def false_negative_rate_disparity(
    left: Sequence[GradeMetric], right: Sequence[GradeMetric]
) -> float:
    return float(
        np.nanmean(
            [
                abs(a.false_negative_rate - b.false_negative_rate)
                for a, b in zip(left, right, strict=True)
            ]
        )
    )


def ordinal_disparity_index(
    left: Sequence[GradeMetric],
    right: Sequence[GradeMetric],
    weights: Sequence[float] = (1.0, 2.0, 2.0, 1.0, 1.0),
    selector: Callable[[GradeMetric], float] | None = None,
) -> float:
    extract = selector or (lambda metric: metric.false_negative_rate)
    disparities = [
        weight * abs(extract(a) - extract(b))
        for a, b, weight in zip(left, right, weights, strict=True)
    ]
    return float(np.nansum(disparities))


def undergrading_disparity(left: GradeMetric, right: GradeMetric) -> float:
    return right.undergrading_rate - left.undergrading_rate


def overgrading_disparity(left: GradeMetric, right: GradeMetric) -> float:
    return right.overgrading_rate - left.overgrading_rate


def compounding_penalty(intersectional_gap: float, axis_gaps: Sequence[float]) -> float:
    if not axis_gaps:
        raise ValueError("At least one single-axis gap is required")
    return intersectional_gap - max(axis_gaps)


def audit_binary_groups(
    batch: PredictionBatch,
    left_group: str,
    right_group: str,
    num_classes: int = 5,
    bins: int = 10,
    weights: Sequence[float] = (1.0, 2.0, 2.0, 1.0, 1.0),
) -> FairnessSummary:
    batch.validate(num_classes)
    left = tuple(grade_metric(batch, left_group, grade) for grade in range(num_classes))
    right = tuple(grade_metric(batch, right_group, grade) for grade in range(num_classes))
    left_selected = batch.groups == left_group
    right_selected = batch.groups == right_group
    left_ece = expected_calibration_error(
        batch.labels[left_selected], batch.probabilities[left_selected], bins
    )
    right_ece = expected_calibration_error(
        batch.labels[right_selected], batch.probabilities[right_selected], bins
    )
    left_auc = subgroup_auc(batch, left_group, num_classes)
    right_auc = subgroup_auc(batch, right_group, num_classes)
    return FairnessSummary(
        equalized_odds_ratio=equalized_odds_ratio(left, right),
        equalized_odds_difference=equalized_odds_difference(left, right),
        demographic_parity_difference=demographic_parity_difference(left, right),
        calibration_difference=abs(left_ece - right_ece),
        worst_group_gap=abs(left_auc - right_auc),
        false_negative_rate_disparity=false_negative_rate_disparity(left, right),
        ordinal_disparity_index=ordinal_disparity_index(left, right, weights),
        grade_metrics=left + right,
    )


def cumulative_threshold_labels(labels: NDArray[np.int64], threshold: int) -> NDArray[np.int64]:
    if threshold not in range(1, 5):
        raise ValueError("Cumulative threshold must be in [1, 4]")
    return (labels >= threshold).astype(np.int64)


def cumulative_threshold_probability(
    probabilities: NDArray[np.float64], threshold: int
) -> NDArray[np.float64]:
    if threshold not in range(1, probabilities.shape[1]):
        raise ValueError("Threshold is outside the class range")
    return probabilities[:, threshold:].sum(axis=1)


def intersection_labels(*axes: NDArray[np.str_]) -> NDArray[np.str_]:
    if not axes:
        raise ValueError("At least one axis is required")
    size = axes[0].size
    if any(axis.size != size for axis in axes):
        raise ValueError("All axes must have equal length")
    output = axes[0].astype(str)
    for axis in axes[1:]:
        output = np.char.add(np.char.add(output, "|"), axis.astype(str))
    return output


def worst_group_gap(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size < 2:
        raise ValueError("At least two subgroup values are required")
    return float(np.nanmax(array) - np.nanmin(array))
