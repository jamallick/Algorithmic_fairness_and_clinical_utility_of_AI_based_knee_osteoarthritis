from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

AxisName = Literal["race", "sex", "age", "bmi"]
ModelName = Literal["siamese", "vgg_ordinal", "densenet", "ensemble", "ensemble_pim"]


@dataclass(frozen=True)
class AuditConfig:
    seed: int = 1701
    num_classes: int = 5
    bootstrap_resamples: int = 1000
    permutation_resamples: int = 10000
    calibration_bins: int = 10
    minimum_cell_size: int = 30
    treatment_weights: tuple[float, ...] = (1.0, 2.0, 2.0, 1.0, 1.0)
    threshold_grade: int = 2

    def validate(self) -> None:
        if self.num_classes != 5:
            raise ValueError("Kellgren-Lawrence grading requires five classes")
        if len(self.treatment_weights) != self.num_classes:
            raise ValueError("Treatment weights must match the number of classes")
        if self.bootstrap_resamples < 1 or self.permutation_resamples < 1:
            raise ValueError("Resample counts must be positive")
        if self.calibration_bins < 2:
            raise ValueError("Calibration needs at least two bins")


@dataclass(frozen=True)
class CohortRecord:
    image_path: Path
    participant_id: str
    knee: Literal["left", "right"]
    visit: str
    grade: int
    race: str
    sex: str
    age: float
    bmi: float
    cohort: str

    def validate(self) -> None:
        if self.grade not in range(5):
            raise ValueError("Grade must be in [0, 4]")
        if self.age <= 0 or self.bmi <= 0:
            raise ValueError("Age and BMI must be positive")
        if not self.participant_id:
            raise ValueError("Participant identifier cannot be empty")


@dataclass(frozen=True)
class PredictionBatch:
    labels: NDArray[np.int64]
    probabilities: NDArray[np.float64]
    groups: NDArray[np.str_]
    participant_ids: NDArray[np.str_] | None = None

    def validate(self, num_classes: int = 5) -> None:
        if self.labels.ndim != 1:
            raise ValueError("Labels must be one-dimensional")
        if self.probabilities.shape != (self.labels.size, num_classes):
            raise ValueError("Probability matrix has an invalid shape")
        if self.groups.shape != self.labels.shape:
            raise ValueError("Groups must align with labels")
        if not np.allclose(self.probabilities.sum(axis=1), 1.0, atol=1e-6):
            raise ValueError("Each probability row must sum to one")
        if np.any(self.probabilities < 0.0):
            raise ValueError("Probabilities cannot be negative")

    @property
    def predictions(self) -> NDArray[np.int64]:
        return self.probabilities.argmax(axis=1).astype(np.int64)


@dataclass(frozen=True)
class GradeMetric:
    grade: int
    group: str
    true_positive_rate: float
    false_positive_rate: float
    false_negative_rate: float
    prediction_rate: float
    undergrading_rate: float
    overgrading_rate: float
    support: int


@dataclass(frozen=True)
class FairnessSummary:
    equalized_odds_ratio: float
    equalized_odds_difference: float
    demographic_parity_difference: float
    calibration_difference: float
    worst_group_gap: float
    false_negative_rate_disparity: float
    ordinal_disparity_index: float
    grade_metrics: tuple[GradeMetric, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class Interval:
    estimate: float
    lower: float
    upper: float


@dataclass(frozen=True)
class HypothesisResult:
    statistic: float
    p_value: float
    adjusted_p_value: float | None = None
    rejected: bool | None = None
