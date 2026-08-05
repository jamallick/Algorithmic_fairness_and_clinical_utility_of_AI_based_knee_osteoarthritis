from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from knee_fairness.calibration import fit_group_thresholds, fit_temperature, temperature_scale
from knee_fairness.clinical import decision_curve, net_reclassification_improvement
from knee_fairness.fairness import (
    audit_binary_groups,
    compounding_penalty,
    intersection_labels,
    macro_auc,
)
from knee_fairness.metrics import calibration_metrics, ordinal_metrics, per_grade_auc
from knee_fairness.reporting import atomic_json
from knee_fairness.schema import FairnessSummary, PredictionBatch
from knee_fairness.statistics import (
    bootstrap_interval,
    cohens_d,
    holm_adjust,
    permutation_gap_test,
)


@dataclass(frozen=True)
class DemographicAxes:
    race: NDArray[np.str_]
    sex: NDArray[np.str_]
    age: NDArray[np.float64]
    bmi: NDArray[np.float64]

    def validate(self, size: int) -> None:
        if self.race.size != size:
            raise ValueError("Race axis does not align")
        if self.sex.size != size:
            raise ValueError("Sex axis does not align")
        if self.age.size != size:
            raise ValueError("Age axis does not align")
        if self.bmi.size != size:
            raise ValueError("BMI axis does not align")
        if not np.all(np.isfinite(self.age)):
            raise ValueError("Age contains non-finite values")
        if not np.all(np.isfinite(self.bmi)):
            raise ValueError("BMI contains non-finite values")

    def binary_groups(self) -> dict[str, NDArray[np.str_]]:
        age_group = np.where(self.age < 65.0, "<65", ">=65").astype(str)
        bmi_group = np.where(self.bmi < 30.0, "<30", ">=30").astype(str)
        return {
            "race": self.race,
            "sex": self.sex,
            "age": age_group,
            "bmi": bmi_group,
        }


@dataclass(frozen=True)
class AxisComparison:
    axis: str
    reference: str
    comparison: str
    summary: FairnessSummary
    auc_reference: float
    auc_comparison: float
    auc_gap: float
    effect_size: float
    permutation_p: float
    adjusted_p: float | None
    significant: bool | None


@dataclass(frozen=True)
class IntersectionResult:
    groups: tuple[str, ...]
    auc_values: tuple[float, ...]
    best_group: str
    worst_group: str
    gap: float
    compounding_penalty: float


@dataclass(frozen=True)
class RecalibrationResult:
    temperature: float
    before_nll: float
    after_nll: float
    before_ece: float
    after_ece: float
    thresholds: dict[str, float]


@dataclass(frozen=True)
class TemporalResult:
    timepoints: tuple[str, ...]
    gaps: tuple[float, ...]
    coefficient_of_variation: float
    baseline_rank_correlation: float | None


def group_macro_auc(batch: PredictionBatch, group: str) -> float:
    selected = batch.groups == group
    return macro_auc(batch.labels[selected], batch.probabilities[selected])


def confidence_scores(batch: PredictionBatch) -> NDArray[np.float64]:
    return batch.probabilities[np.arange(batch.labels.size), batch.labels]


def axis_comparison(
    labels: NDArray[np.int64],
    probabilities: NDArray[np.float64],
    groups: NDArray[np.str_],
    participant_ids: NDArray[np.str_] | None,
    axis: str,
    reference: str,
    comparison: str,
    permutations: int,
    seed: int,
) -> AxisComparison:
    batch = PredictionBatch(labels, probabilities, groups, participant_ids)
    summary = audit_binary_groups(batch, reference, comparison)
    reference_auc = group_macro_auc(batch, reference)
    comparison_auc = group_macro_auc(batch, comparison)
    confidence = confidence_scores(batch)
    selected_reference = confidence[groups == reference]
    selected_comparison = confidence[groups == comparison]
    test = permutation_gap_test(
        confidence,
        groups,
        reference,
        comparison,
        permutations,
        seed,
    )
    return AxisComparison(
        axis,
        reference,
        comparison,
        summary,
        reference_auc,
        comparison_auc,
        abs(reference_auc - comparison_auc),
        cohens_d(selected_reference, selected_comparison),
        test.p_value,
        None,
        None,
    )


def primary_audit(
    labels: NDArray[np.int64],
    probabilities: NDArray[np.float64],
    axes: DemographicAxes,
    participant_ids: NDArray[np.str_] | None = None,
    permutations: int = 10000,
    seed: int = 1701,
) -> tuple[AxisComparison, ...]:
    axes.validate(labels.size)
    definitions = {
        "race": ("White", "African American"),
        "sex": ("female", "male"),
        "age": ("<65", ">=65"),
        "bmi": ("<30", ">=30"),
    }
    raw: list[AxisComparison] = []
    for axis, groups in axes.binary_groups().items():
        reference, comparison = definitions[axis]
        available = set(np.unique(groups))
        if reference not in available or comparison not in available:
            continue
        raw.append(
            axis_comparison(
                labels,
                probabilities,
                groups,
                participant_ids,
                axis,
                reference,
                comparison,
                permutations,
                seed,
            )
        )
    hypothesis = [
        permutation_gap_test(
            confidence_scores(
                PredictionBatch(labels, probabilities, axes.binary_groups()[item.axis])
            ),
            axes.binary_groups()[item.axis],
            item.reference,
            item.comparison,
            permutations,
            seed,
        )
        for item in raw
    ]
    adjusted = holm_adjust(hypothesis)
    output: list[AxisComparison] = []
    for item, result in zip(raw, adjusted, strict=True):
        output.append(
            AxisComparison(
                item.axis,
                item.reference,
                item.comparison,
                item.summary,
                item.auc_reference,
                item.auc_comparison,
                item.auc_gap,
                item.effect_size,
                item.permutation_p,
                result.adjusted_p_value,
                result.rejected,
            )
        )
    return tuple(output)


def intersection_audit(
    labels: NDArray[np.int64],
    probabilities: NDArray[np.float64],
    race: NDArray[np.str_],
    sex: NDArray[np.str_],
    race_gap: float,
    sex_gap: float,
    minimum_cell: int = 30,
) -> IntersectionResult:
    intersections = intersection_labels(race, sex)
    names: list[str] = []
    values: list[float] = []
    for group in np.unique(intersections):
        selected = intersections == group
        if int(selected.sum()) < minimum_cell:
            continue
        names.append(str(group))
        values.append(macro_auc(labels[selected], probabilities[selected]))
    if len(names) < 2:
        raise ValueError("Intersectional audit requires at least two eligible cells")
    best = int(np.nanargmax(values))
    worst = int(np.nanargmin(values))
    gap = values[best] - values[worst]
    return IntersectionResult(
        tuple(names),
        tuple(values),
        names[best],
        names[worst],
        gap,
        compounding_penalty(gap, (race_gap, sex_gap)),
    )


def recalibration_audit(
    logits: NDArray[np.float64],
    labels: NDArray[np.int64],
    groups: NDArray[np.str_],
    bins: int = 10,
) -> RecalibrationResult:
    before_probability = temperature_scale(logits, 1.0)
    before = calibration_metrics(labels, before_probability, bins)
    fitted = fit_temperature(logits, labels)
    after_probability = temperature_scale(logits, fitted.temperature)
    after = calibration_metrics(labels, after_probability, bins)
    thresholds = fit_group_thresholds(labels, after_probability, groups, 2)
    return RecalibrationResult(
        fitted.temperature,
        before.negative_log_likelihood,
        after.negative_log_likelihood,
        before.expected_calibration_error,
        after.expected_calibration_error,
        thresholds.values,
    )


def temporal_audit(
    batches: dict[str, PredictionBatch],
    reference: str,
    comparison: str,
    baseline_order: NDArray[np.float64] | None = None,
) -> TemporalResult:
    if not batches:
        raise ValueError("Temporal audit requires timepoints")
    timepoints: list[str] = []
    gaps: list[float] = []
    for timepoint, batch in batches.items():
        summary = audit_binary_groups(batch, reference, comparison)
        timepoints.append(timepoint)
        gaps.append(summary.worst_group_gap)
    array = np.asarray(gaps)
    coefficient = float(array.std(ddof=1) / array.mean()) if len(gaps) > 1 else 0.0
    correlation = None
    if baseline_order is not None and baseline_order.size == array.size:
        correlation = float(stats.spearmanr(baseline_order, array).statistic)
    return TemporalResult(tuple(timepoints), tuple(gaps), coefficient, correlation)


def bootstrap_auc_gap(
    batch: PredictionBatch,
    reference: str,
    comparison: str,
    resamples: int = 1000,
    seed: int = 1701,
) -> dict[str, float]:
    def statistic(value: PredictionBatch) -> float:
        return abs(group_macro_auc(value, reference) - group_macro_auc(value, comparison))

    interval = bootstrap_interval(batch, statistic, resamples, 0.95, seed)
    return asdict(interval)


def clinical_boundary_audit(
    labels: NDArray[np.int64],
    candidate: NDArray[np.float64],
    reference: NDArray[np.float64],
    groups: NDArray[np.str_],
    threshold: float = 0.5,
) -> dict[str, Any]:
    binary_labels = (labels >= 2).astype(np.int64)
    output: dict[str, Any] = {}
    for group in np.unique(groups):
        selected = groups == group
        nri = net_reclassification_improvement(
            binary_labels[selected],
            reference[selected],
            candidate[selected],
            threshold,
        )
        curve = decision_curve(binary_labels[selected], candidate[selected])
        output[str(group)] = {
            "reclassification": asdict(nri),
            "thresholds": curve.thresholds.tolist(),
            "net_benefit": curve.net_benefit.tolist(),
        }
    return output


def per_grade_report(batch: PredictionBatch) -> dict[str, Any]:
    predictions = batch.predictions
    return {
        "ordinal": asdict(ordinal_metrics(batch.labels, predictions)),
        "auc": per_grade_auc(batch.labels, batch.probabilities),
        "groups": {
            str(group): per_grade_auc(
                batch.labels[batch.groups == group],
                batch.probabilities[batch.groups == group],
            )
            for group in np.unique(batch.groups)
        },
    }


def export_audit(path: Path, payload: dict[str, Any]) -> None:
    atomic_json(path, payload)
