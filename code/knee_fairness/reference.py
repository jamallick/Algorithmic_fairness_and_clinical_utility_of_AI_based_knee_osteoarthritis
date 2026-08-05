from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPerformance:
    model: str
    accuracy: float
    weighted_kappa: float
    macro_auc: float
    sensitivity: tuple[float, float, float, float, float]


@dataclass(frozen=True)
class SubgroupAuc:
    cohort: str
    subgroup: str
    deepknee: float
    vgg19: float
    densenet: float
    ensemble: float
    ensemble_pim: float


@dataclass(frozen=True)
class FairnessReference:
    cohort: str
    model: str
    eor: float
    eod: float
    dpd: float
    calibration_difference: float
    worst_group_gap: float
    fnr_disparity: float
    odi: float


@dataclass(frozen=True)
class TemporalGap:
    model: str
    baseline: float
    month_12: float
    month_24: float
    month_48: float
    month_96: float


def overall_performance() -> tuple[ModelPerformance, ...]:
    return (
        ModelPerformance("DeepKnee", 0.664, 0.82, 0.84, (0.86, 0.35, 0.59, 0.72, 0.88)),
        ModelPerformance("VGG-19+Ord.", 0.698, 0.84, 0.86, (0.88, 0.38, 0.64, 0.76, 0.90)),
        ModelPerformance("DenseNet", 0.713, 0.86, 0.87, (0.89, 0.43, 0.67, 0.79, 0.92)),
        ModelPerformance("Ensemble", 0.768, 0.87, 0.89, (0.91, 0.46, 0.73, 0.84, 0.93)),
        ModelPerformance("Ensemble PIM", 0.772, 0.87, 0.89, (0.90, 0.49, 0.71, 0.82, 0.95)),
    )


def subgroup_auc_reference() -> tuple[SubgroupAuc, ...]:
    return (
        SubgroupAuc("OAI", "White", 0.852, 0.870, 0.882, 0.901, 0.905),
        SubgroupAuc("OAI", "African American", 0.817, 0.839, 0.840, 0.847, 0.837),
        SubgroupAuc("OAI", "Female", 0.841, 0.862, 0.877, 0.894, 0.897),
        SubgroupAuc("OAI", "Male", 0.829, 0.847, 0.854, 0.871, 0.863),
        SubgroupAuc("OAI", "Age <65", 0.850, 0.867, 0.879, 0.898, 0.902),
        SubgroupAuc("OAI", "Age >=65", 0.825, 0.838, 0.843, 0.854, 0.851),
        SubgroupAuc("OAI", "BMI <30", 0.845, 0.860, 0.873, 0.891, 0.895),
        SubgroupAuc("OAI", "BMI >=30", 0.827, 0.842, 0.849, 0.861, 0.853),
        SubgroupAuc("MOST", "White", 0.838, 0.855, 0.869, 0.887, 0.890),
        SubgroupAuc("MOST", "African American", 0.804, 0.823, 0.824, 0.830, 0.816),
        SubgroupAuc("MOST", "Female", 0.829, 0.847, 0.860, 0.876, 0.874),
        SubgroupAuc("MOST", "Male", 0.818, 0.836, 0.843, 0.858, 0.848),
    )


def oai_fairness_reference() -> tuple[FairnessReference, ...]:
    return (
        FairnessReference("OAI", "DeepKnee", 0.78, 0.07, 0.06, 0.03, 0.048, 0.05, 22.3),
        FairnessReference("OAI", "VGG-19+Ord.", 0.82, 0.05, 0.05, 0.02, 0.039, 0.04, 21.1),
        FairnessReference("OAI", "DenseNet", 0.80, 0.06, 0.07, 0.04, 0.054, 0.05, 29.3),
        FairnessReference("OAI", "Ensemble", 0.75, 0.09, 0.09, 0.04, 0.071, 0.07, 35.1),
        FairnessReference("OAI", "Ensemble PIM", 0.72, 0.14, 0.12, 0.06, 0.082, 0.09, 47.3),
    )


def most_fairness_reference() -> tuple[FairnessReference, ...]:
    return (
        FairnessReference("MOST", "DeepKnee", 0.76, 0.08, 0.07, 0.04, 0.052, 0.06, 24.1),
        FairnessReference("MOST", "VGG-19+Ord.", 0.81, 0.06, 0.06, 0.03, 0.043, 0.05, 22.7),
        FairnessReference("MOST", "DenseNet", 0.79, 0.07, 0.08, 0.05, 0.058, 0.06, 31.4),
        FairnessReference("MOST", "Ensemble", 0.73, 0.10, 0.10, 0.05, 0.076, 0.08, 38.2),
        FairnessReference("MOST", "Ensemble PIM", 0.69, 0.16, 0.14, 0.07, 0.091, 0.10, 52.6),
    )


def temporal_reference() -> tuple[TemporalGap, ...]:
    return (
        TemporalGap("DeepKnee", 3.5, 3.3, 3.7, 3.9, 4.2),
        TemporalGap("VGG-19+Ord.", 3.1, 2.9, 3.3, 3.2, 3.5),
        TemporalGap("DenseNet", 4.2, 4.0, 4.4, 4.6, 5.0),
        TemporalGap("Ensemble", 5.4, 5.1, 5.6, 5.8, 6.3),
        TemporalGap("Ensemble PIM", 6.8, 6.5, 7.1, 7.3, 7.8),
    )


def undergrading_reference() -> dict[str, tuple[float, float, float, float]]:
    return {
        "DeepKnee": (5.2, 3.8, 2.9, 1.4),
        "VGG-19+Ord.": (5.8, 3.1, 2.4, 0.9),
        "DenseNet": (7.4, 4.6, 3.7, 1.8),
        "Ensemble": (9.1, 5.3, 4.2, 2.1),
        "Ensemble PIM": (11.8, 7.6, 5.8, 2.7),
    }


def thresholds() -> dict[str, float]:
    return {
        "equalized_odds_ratio": 0.80,
        "equalized_odds_difference": 0.10,
        "demographic_parity_difference": 0.10,
        "calibration_difference": 0.05,
        "worst_group_gap": 0.05,
        "false_negative_rate_disparity": 0.05,
        "minimum_cell_size": 30.0,
    }


def tolerance_checks(
    observed: tuple[ModelPerformance, ...],
    tolerance: float = 0.02,
) -> dict[str, bool]:
    expected = {item.model: item for item in overall_performance()}
    output: dict[str, bool] = {}
    for item in observed:
        if item.model not in expected:
            output[item.model] = False
            continue
        output[item.model] = abs(item.accuracy - expected[item.model].accuracy) <= tolerance
    return output
