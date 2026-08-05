from __future__ import annotations

import csv
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from knee_fairness.schema import FairnessSummary


class ScientificEncoder(json.JSONEncoder):
    def default(self, value: object) -> object:
        if is_dataclass(value) and not isinstance(value, type):
            return asdict(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, Path):
            return str(value)
        return super().default(value)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, cls=ScientificEncoder, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def write_fairness_csv(path: Path, summaries: Mapping[str, FairnessSummary]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = (
        "axis",
        "equalized_odds_ratio",
        "equalized_odds_difference",
        "demographic_parity_difference",
        "calibration_difference",
        "worst_group_gap",
        "false_negative_rate_disparity",
        "ordinal_disparity_index",
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for axis, summary in summaries.items():
            writer.writerow(
                {
                    "axis": axis,
                    "equalized_odds_ratio": summary.equalized_odds_ratio,
                    "equalized_odds_difference": summary.equalized_odds_difference,
                    "demographic_parity_difference": summary.demographic_parity_difference,
                    "calibration_difference": summary.calibration_difference,
                    "worst_group_gap": summary.worst_group_gap,
                    "false_negative_rate_disparity": summary.false_negative_rate_disparity,
                    "ordinal_disparity_index": summary.ordinal_disparity_index,
                }
            )
    os.replace(temporary, path)


def write_grade_csv(path: Path, summaries: Mapping[str, FairnessSummary]) -> None:
    columns = (
        "axis",
        "grade",
        "group",
        "true_positive_rate",
        "false_positive_rate",
        "false_negative_rate",
        "prediction_rate",
        "undergrading_rate",
        "overgrading_rate",
        "support",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for axis, summary in summaries.items():
            for metric in summary.grade_metrics:
                row = asdict(metric)
                row["axis"] = axis
                writer.writerow(row)
    os.replace(temporary, path)


def read_prediction_csv(path: Path) -> dict[str, np.ndarray[Any, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    required = {"label", "group", "participant_id", "p0", "p1", "p2", "p3", "p4"}
    if not rows:
        raise ValueError("Prediction CSV is empty")
    if not required.issubset(rows[0]):
        raise ValueError(f"Prediction CSV requires columns: {sorted(required)}")
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    groups = np.asarray([row["group"] for row in rows], dtype=str)
    participant_ids = np.asarray([row["participant_id"] for row in rows], dtype=str)
    probabilities = np.asarray(
        [[float(row[f"p{grade}"]) for grade in range(5)] for row in rows],
        dtype=np.float64,
    )
    return {
        "labels": labels,
        "groups": groups,
        "participant_ids": participant_ids,
        "probabilities": probabilities,
    }


def write_prediction_csv(
    path: Path,
    labels: np.ndarray[Any, Any],
    probabilities: np.ndarray[Any, Any],
    groups: Sequence[str],
    participant_ids: Sequence[str],
) -> None:
    columns = ("participant_id", "label", "group", "p0", "p1", "p2", "p3", "p4")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for index in range(labels.size):
            writer.writerow(
                {
                    "participant_id": participant_ids[index],
                    "label": int(labels[index]),
                    "group": groups[index],
                    **{f"p{grade}": float(probabilities[index, grade]) for grade in range(5)},
                }
            )
    os.replace(temporary, path)
