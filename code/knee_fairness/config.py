from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class RuntimeConfig:
    seed: int
    model: str
    num_classes: int
    image_size: int
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    optimizer: str
    scheduler: str
    warmup_epochs: int
    grad_clip_norm: float
    precision: str
    workers: int
    bootstrap_resamples: int
    permutation_resamples: int
    calibration_bins: int
    minimum_cell_size: int
    test_fraction: float
    validation_fraction: float
    paper_training_parameters_reported: bool

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> RuntimeConfig:
        return cls(
            seed=int(values["seed"]),
            model=str(values["model"]),
            num_classes=int(values["num_classes"]),
            image_size=int(values["image_size"]),
            batch_size=int(values["batch_size"]),
            epochs=int(values["epochs"]),
            learning_rate=float(values["learning_rate"]),
            weight_decay=float(values["weight_decay"]),
            optimizer=str(values["optimizer"]),
            scheduler=str(values["scheduler"]),
            warmup_epochs=int(values["warmup_epochs"]),
            grad_clip_norm=float(values["grad_clip_norm"]),
            precision=str(values["precision"]),
            workers=int(values["workers"]),
            bootstrap_resamples=int(values["bootstrap_resamples"]),
            permutation_resamples=int(values["permutation_resamples"]),
            calibration_bins=int(values["calibration_bins"]),
            minimum_cell_size=int(values["minimum_cell_size"]),
            test_fraction=float(values["test_fraction"]),
            validation_fraction=float(values["validation_fraction"]),
            paper_training_parameters_reported=bool(values["paper_training_parameters_reported"]),
        )

    def validate(self) -> None:
        if self.num_classes != 5:
            raise ValueError("This audit is defined for five KL grades")
        if self.image_size < 16 or self.batch_size < 1:
            raise ValueError("Image size and batch size are invalid")
        if self.epochs < 1 or self.learning_rate <= 0.0:
            raise ValueError("Training duration and learning rate are invalid")
        if self.precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("Unsupported precision")


def load_config(path: str | Path) -> RuntimeConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        values = yaml.safe_load(handle)
    if not isinstance(values, dict):
        raise ValueError("Configuration root must be a mapping")
    config = RuntimeConfig.from_mapping(values)
    config.validate()
    return config
