from __future__ import annotations

import argparse
from pathlib import Path

from knee_fairness.evaluation import performance_summary
from knee_fairness.fairness import audit_binary_groups
from knee_fairness.reporting import atomic_json, read_prediction_csv
from knee_fairness.schema import PredictionBatch


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(prog="knee-fairness-evaluate")
    value.add_argument("--predictions", type=Path, required=True)
    value.add_argument("--left-group", required=True)
    value.add_argument("--right-group", required=True)
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--bins", type=int, default=10)
    return value


def main() -> None:
    arguments = parser().parse_args()
    data = read_prediction_csv(arguments.predictions)
    batch = PredictionBatch(
        data["labels"],
        data["probabilities"],
        data["groups"],
        data["participant_ids"],
    )
    performance = performance_summary(batch)
    fairness = audit_binary_groups(
        batch,
        arguments.left_group,
        arguments.right_group,
        bins=arguments.bins,
    )
    atomic_json(arguments.output, {"performance": performance, "fairness": fairness})


if __name__ == "__main__":
    main()
