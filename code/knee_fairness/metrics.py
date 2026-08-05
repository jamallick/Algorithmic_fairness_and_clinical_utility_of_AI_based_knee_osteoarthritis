from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import roc_auc_score


@dataclass(frozen=True)
class BinaryCounts:
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int

    @property
    def total(self) -> int:
        return self.true_positive + self.false_positive + self.true_negative + self.false_negative

    @property
    def actual_positive(self) -> int:
        return self.true_positive + self.false_negative

    @property
    def actual_negative(self) -> int:
        return self.true_negative + self.false_positive

    @property
    def predicted_positive(self) -> int:
        return self.true_positive + self.false_positive

    @property
    def predicted_negative(self) -> int:
        return self.true_negative + self.false_negative


@dataclass(frozen=True)
class BinaryMetrics:
    sensitivity: float
    specificity: float
    precision: float
    negative_predictive_value: float
    false_positive_rate: float
    false_negative_rate: float
    accuracy: float
    balanced_accuracy: float
    f1: float
    prevalence: float
    prediction_rate: float


@dataclass(frozen=True)
class OrdinalMetrics:
    exact_accuracy: float
    within_one_accuracy: float
    mean_absolute_error: float
    root_mean_squared_error: float
    signed_error: float
    undergrading_rate: float
    overgrading_rate: float
    quadratic_weighted_kappa: float


@dataclass(frozen=True)
class CalibrationBin:
    lower: float
    upper: float
    count: int
    accuracy: float
    confidence: float
    gap: float


@dataclass(frozen=True)
class CalibrationMetrics:
    expected_calibration_error: float
    maximum_calibration_error: float
    adaptive_calibration_error: float
    brier_score: float
    negative_log_likelihood: float
    bins: tuple[CalibrationBin, ...]


def binary_counts(labels: NDArray[np.int64], predictions: NDArray[np.int64]) -> BinaryCounts:
    if labels.shape != predictions.shape:
        raise ValueError("Labels and predictions must align")
    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("Labels must be binary")
    if not set(np.unique(predictions)).issubset({0, 1}):
        raise ValueError("Predictions must be binary")
    positive = labels == 1
    predicted_positive = predictions == 1
    return BinaryCounts(
        true_positive=int(np.logical_and(positive, predicted_positive).sum()),
        false_positive=int(np.logical_and(~positive, predicted_positive).sum()),
        true_negative=int(np.logical_and(~positive, ~predicted_positive).sum()),
        false_negative=int(np.logical_and(positive, ~predicted_positive).sum()),
    )


def divide(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def binary_metrics(counts: BinaryCounts) -> BinaryMetrics:
    sensitivity = divide(counts.true_positive, counts.actual_positive)
    specificity = divide(counts.true_negative, counts.actual_negative)
    precision = divide(counts.true_positive, counts.predicted_positive)
    negative_predictive_value = divide(counts.true_negative, counts.predicted_negative)
    false_positive_rate = divide(counts.false_positive, counts.actual_negative)
    false_negative_rate = divide(counts.false_negative, counts.actual_positive)
    accuracy = divide(counts.true_positive + counts.true_negative, counts.total)
    balanced_accuracy = (sensitivity + specificity) / 2.0
    f1 = divide(
        2 * counts.true_positive,
        2 * counts.true_positive + counts.false_positive + counts.false_negative,
    )
    prevalence = divide(counts.actual_positive, counts.total)
    prediction_rate = divide(counts.predicted_positive, counts.total)
    return BinaryMetrics(
        sensitivity,
        specificity,
        precision,
        negative_predictive_value,
        false_positive_rate,
        false_negative_rate,
        accuracy,
        balanced_accuracy,
        f1,
        prevalence,
        prediction_rate,
    )


def one_vs_rest_metrics(
    labels: NDArray[np.int64],
    predictions: NDArray[np.int64],
    num_classes: int = 5,
) -> tuple[BinaryMetrics, ...]:
    output: list[BinaryMetrics] = []
    for grade in range(num_classes):
        binary_labels = (labels == grade).astype(np.int64)
        binary_predictions = (predictions == grade).astype(np.int64)
        output.append(binary_metrics(binary_counts(binary_labels, binary_predictions)))
    return tuple(output)


def ordinal_confusion_matrix(
    labels: NDArray[np.int64],
    predictions: NDArray[np.int64],
    num_classes: int = 5,
) -> NDArray[np.int64]:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(matrix, (labels, predictions), 1)
    return matrix


def quadratic_weighted_kappa(
    labels: NDArray[np.int64],
    predictions: NDArray[np.int64],
    num_classes: int = 5,
) -> float:
    observed = ordinal_confusion_matrix(labels, predictions, num_classes).astype(np.float64)
    count = observed.sum()
    if count == 0:
        return float("nan")
    actual = observed.sum(axis=1)
    predicted = observed.sum(axis=0)
    expected = np.outer(actual, predicted) / count
    grades = np.arange(num_classes, dtype=np.float64)
    weights = np.square(grades[:, None] - grades[None, :]) / (num_classes - 1) ** 2
    observed_disagreement = float(np.sum(weights * observed))
    expected_disagreement = float(np.sum(weights * expected))
    if expected_disagreement == 0.0:
        return 1.0
    return 1.0 - observed_disagreement / expected_disagreement


def ordinal_metrics(
    labels: NDArray[np.int64],
    predictions: NDArray[np.int64],
    num_classes: int = 5,
) -> OrdinalMetrics:
    errors = predictions.astype(np.float64) - labels.astype(np.float64)
    absolute = np.abs(errors)
    return OrdinalMetrics(
        exact_accuracy=float(np.mean(absolute == 0)),
        within_one_accuracy=float(np.mean(absolute <= 1)),
        mean_absolute_error=float(absolute.mean()),
        root_mean_squared_error=float(np.sqrt(np.mean(np.square(errors)))),
        signed_error=float(errors.mean()),
        undergrading_rate=float(np.mean(errors < 0)),
        overgrading_rate=float(np.mean(errors > 0)),
        quadratic_weighted_kappa=quadratic_weighted_kappa(labels, predictions, num_classes),
    )


def per_grade_auc(
    labels: NDArray[np.int64],
    probabilities: NDArray[np.float64],
    num_classes: int = 5,
) -> tuple[float, ...]:
    output: list[float] = []
    for grade in range(num_classes):
        target = (labels == grade).astype(np.int64)
        if np.unique(target).size < 2:
            output.append(float("nan"))
        else:
            output.append(float(roc_auc_score(target, probabilities[:, grade])))
    return tuple(output)


def cumulative_auc(
    labels: NDArray[np.int64],
    probabilities: NDArray[np.float64],
    num_classes: int = 5,
) -> tuple[float, ...]:
    output: list[float] = []
    for threshold in range(1, num_classes):
        target = (labels >= threshold).astype(np.int64)
        score = probabilities[:, threshold:].sum(axis=1)
        if np.unique(target).size < 2:
            output.append(float("nan"))
        else:
            output.append(float(roc_auc_score(target, score)))
    return tuple(output)


def calibration_bins(
    labels: NDArray[np.int64],
    probabilities: NDArray[np.float64],
    count: int = 10,
) -> tuple[CalibrationBin, ...]:
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = predictions == labels
    boundaries = np.linspace(0.0, 1.0, count + 1)
    output: list[CalibrationBin] = []
    for index in range(count):
        lower = float(boundaries[index])
        upper = float(boundaries[index + 1])
        selected = (confidence > lower) & (confidence <= upper)
        if index == 0:
            selected |= confidence == 0.0
        size = int(selected.sum())
        accuracy = float(correct[selected].mean()) if size else float("nan")
        mean_confidence = float(confidence[selected].mean()) if size else float("nan")
        gap = abs(accuracy - mean_confidence) if size else float("nan")
        output.append(CalibrationBin(lower, upper, size, accuracy, mean_confidence, gap))
    return tuple(output)


def adaptive_calibration_bins(
    labels: NDArray[np.int64],
    probabilities: NDArray[np.float64],
    count: int = 10,
) -> tuple[CalibrationBin, ...]:
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = predictions == labels
    ordered = np.argsort(confidence)
    pieces = np.array_split(ordered, count)
    output: list[CalibrationBin] = []
    for piece in pieces:
        if piece.size == 0:
            continue
        values = confidence[piece]
        accuracy = float(correct[piece].mean())
        mean_confidence = float(values.mean())
        output.append(
            CalibrationBin(
                float(values.min()),
                float(values.max()),
                int(piece.size),
                accuracy,
                mean_confidence,
                abs(accuracy - mean_confidence),
            )
        )
    return tuple(output)


def calibration_metrics(
    labels: NDArray[np.int64],
    probabilities: NDArray[np.float64],
    count: int = 10,
) -> CalibrationMetrics:
    fixed = calibration_bins(labels, probabilities, count)
    adaptive = adaptive_calibration_bins(labels, probabilities, count)
    total = labels.size
    expected = sum(item.count / total * item.gap for item in fixed if item.count)
    maximum = max((item.gap for item in fixed if item.count), default=float("nan"))
    adaptive_error = sum(item.count / total * item.gap for item in adaptive if item.count)
    targets = np.eye(probabilities.shape[1], dtype=np.float64)[labels]
    brier = float(np.mean(np.sum(np.square(probabilities - targets), axis=1)))
    selected = np.clip(probabilities[np.arange(total), labels], 1e-12, 1.0)
    nll = float(-np.log(selected).mean())
    return CalibrationMetrics(expected, maximum, adaptive_error, brier, nll, fixed)


def sensitivity_gap(left: BinaryMetrics, right: BinaryMetrics) -> float:
    return abs(left.sensitivity - right.sensitivity)


def specificity_gap(left: BinaryMetrics, right: BinaryMetrics) -> float:
    return abs(left.specificity - right.specificity)


def precision_gap(left: BinaryMetrics, right: BinaryMetrics) -> float:
    return abs(left.precision - right.precision)


def selection_rate_gap(left: BinaryMetrics, right: BinaryMetrics) -> float:
    return abs(left.prediction_rate - right.prediction_rate)


def equal_opportunity_difference(left: BinaryMetrics, right: BinaryMetrics) -> float:
    return left.sensitivity - right.sensitivity


def average_odds_difference(left: BinaryMetrics, right: BinaryMetrics) -> float:
    tpr_gap = left.sensitivity - right.sensitivity
    fpr_gap = left.false_positive_rate - right.false_positive_rate
    return 0.5 * (tpr_gap + fpr_gap)


def disparate_impact_ratio(left: BinaryMetrics, right: BinaryMetrics) -> float:
    return divide(left.prediction_rate, right.prediction_rate)


def predictive_parity_ratio(left: BinaryMetrics, right: BinaryMetrics) -> float:
    return divide(left.precision, right.precision)
