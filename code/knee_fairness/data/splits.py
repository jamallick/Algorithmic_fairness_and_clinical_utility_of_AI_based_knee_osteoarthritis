from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from knee_fairness.schema import CohortRecord


@dataclass(frozen=True)
class DatasetSplit:
    train: tuple[CohortRecord, ...]
    validation: tuple[CohortRecord, ...]
    test: tuple[CohortRecord, ...]


def participant_split(
    records: Sequence[CohortRecord],
    test_fraction: float = 0.2,
    validation_fraction: float = 0.1,
    seed: int = 1701,
) -> DatasetSplit:
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("Test fraction must be in (0, 1)")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("Validation fraction must be in (0, 1)")
    if test_fraction + validation_fraction >= 1.0:
        raise ValueError("Test and validation fractions leave no training data")
    participants = np.asarray(sorted({record.participant_id for record in records}))
    rng = np.random.default_rng(seed)
    rng.shuffle(participants)
    test_count = max(1, round(participants.size * test_fraction))
    validation_count = max(1, round(participants.size * validation_fraction))
    test_ids = set(participants[:test_count].tolist())
    validation_ids = set(participants[test_count : test_count + validation_count].tolist())
    train_ids = set(participants[test_count + validation_count :].tolist())
    train = tuple(record for record in records if record.participant_id in train_ids)
    validation = tuple(record for record in records if record.participant_id in validation_ids)
    test = tuple(record for record in records if record.participant_id in test_ids)
    if not train or not validation or not test:
        raise ValueError("Participant split produced an empty partition")
    return DatasetSplit(train, validation, test)


def stratified_participant_split(
    records: Sequence[CohortRecord],
    test_fraction: float = 0.2,
    validation_fraction: float = 0.1,
    seed: int = 1701,
) -> DatasetSplit:
    participant_grade: dict[str, int] = {}
    for record in records:
        participant_grade[record.participant_id] = max(
            record.grade,
            participant_grade.get(record.participant_id, 0),
        )
    rng = np.random.default_rng(seed)
    train_ids: set[str] = set()
    validation_ids: set[str] = set()
    test_ids: set[str] = set()
    for grade in range(5):
        members = np.asarray(
            sorted(key for key, value in participant_grade.items() if value == grade)
        )
        rng.shuffle(members)
        test_count = round(members.size * test_fraction)
        validation_count = round(members.size * validation_fraction)
        test_ids.update(members[:test_count].tolist())
        validation_ids.update(members[test_count : test_count + validation_count].tolist())
        train_ids.update(members[test_count + validation_count :].tolist())
    return DatasetSplit(
        tuple(record for record in records if record.participant_id in train_ids),
        tuple(record for record in records if record.participant_id in validation_ids),
        tuple(record for record in records if record.participant_id in test_ids),
    )


def verify_disjoint(split: DatasetSplit) -> None:
    train = {record.participant_id for record in split.train}
    validation = {record.participant_id for record in split.validation}
    test = {record.participant_id for record in split.test}
    if train & validation or train & test or validation & test:
        raise ValueError("Participant leakage detected across partitions")
