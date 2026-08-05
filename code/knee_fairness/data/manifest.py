from __future__ import annotations

import csv
import hashlib
from collections.abc import Iterable, Sequence
from pathlib import Path

from knee_fairness.schema import CohortRecord

MANIFEST_COLUMNS = (
    "image_path",
    "participant_id",
    "knee",
    "visit",
    "grade",
    "race",
    "sex",
    "age",
    "bmi",
    "cohort",
)


def parse_row(row: dict[str, str], root: Path) -> CohortRecord:
    missing = set(MANIFEST_COLUMNS).difference(row)
    if missing:
        raise ValueError(f"Manifest row lacks columns: {sorted(missing)}")
    image_path = Path(row["image_path"])
    if not image_path.is_absolute():
        image_path = root / image_path
    record = CohortRecord(
        image_path=image_path,
        participant_id=row["participant_id"],
        knee=_parse_knee(row["knee"]),
        visit=row["visit"],
        grade=int(row["grade"]),
        race=row["race"],
        sex=row["sex"],
        age=float(row["age"]),
        bmi=float(row["bmi"]),
        cohort=row["cohort"],
    )
    record.validate()
    return record


def _parse_knee(value: str) -> str:
    normalized = value.lower().strip()
    if normalized not in {"left", "right"}:
        raise ValueError("Knee must be left or right")
    return normalized


def read_manifest(path: Path) -> list[CohortRecord]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [parse_row(dict(row), path.parent) for row in reader]


def write_manifest(path: Path, records: Iterable[CohortRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "image_path": str(record.image_path),
                    "participant_id": record.participant_id,
                    "knee": record.knee,
                    "visit": record.visit,
                    "grade": record.grade,
                    "race": record.race,
                    "sex": record.sex,
                    "age": record.age,
                    "bmi": record.bmi,
                    "cohort": record.cohort,
                }
            )
    temporary.replace(path)


def manifest_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_manifest(records: Sequence[CohortRecord], require_images: bool = True) -> None:
    if not records:
        raise ValueError("Manifest cannot be empty")
    keys: set[tuple[str, str, str]] = set()
    for record in records:
        record.validate()
        key = (record.participant_id, record.knee, record.visit)
        if key in keys:
            raise ValueError(f"Duplicate participant-knee-visit record: {key}")
        keys.add(key)
        if require_images and not record.image_path.is_file():
            raise FileNotFoundError(record.image_path)


def cohort_counts(records: Sequence[CohortRecord]) -> dict[str, int]:
    output: dict[str, int] = {}
    for record in records:
        output[record.cohort] = output.get(record.cohort, 0) + 1
    return output


def grade_counts(records: Sequence[CohortRecord]) -> dict[int, int]:
    output = {grade: 0 for grade in range(5)}
    for record in records:
        output[record.grade] += 1
    return output


def subgroup_counts(records: Sequence[CohortRecord], attribute: str) -> dict[str, int]:
    allowed = {"race", "sex", "visit", "cohort", "knee"}
    if attribute not in allowed:
        raise ValueError(f"Unsupported categorical attribute: {attribute}")
    output: dict[str, int] = {}
    for record in records:
        value = str(getattr(record, attribute))
        output[value] = output.get(value, 0) + 1
    return output


def filter_records(
    records: Sequence[CohortRecord],
    cohort: str | None = None,
    visit: str | None = None,
    race: str | None = None,
    sex: str | None = None,
) -> list[CohortRecord]:
    output: list[CohortRecord] = []
    for record in records:
        if cohort is not None and record.cohort != cohort:
            continue
        if visit is not None and record.visit != visit:
            continue
        if race is not None and record.race != race:
            continue
        if sex is not None and record.sex != sex:
            continue
        output.append(record)
    return output
