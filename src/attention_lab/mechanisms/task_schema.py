from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


REQUIRED_RECORD_FIELDS = ("x_pos", "x_neg", "x_para", "x_decoy", "pair_id", "template_id", "family_id")
PROVENANCE_FIELDS = (
    "generator_name",
    "generator_version",
    "template_set",
    "filler_set",
    "generation_seed",
    "created_at",
)
CONFIRMATORY_MIN_PAIRS_PER_FAMILY = 50


@dataclass(frozen=True)
class TaskRecord:
    x_pos: str
    x_neg: str
    x_para: str
    x_decoy: str
    pair_id: str
    template_id: str
    family_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskSuite:
    records: tuple[TaskRecord, ...]
    metadata: dict[str, Any]
    source_path: Path | None = None

    def pair_counts_by_family(self) -> dict[str, int]:
        pairs: dict[str, set[str]] = {}
        for record in self.records:
            pairs.setdefault(record.family_id, set()).add(record.pair_id)
        return {family: len(pair_ids) for family, pair_ids in sorted(pairs.items())}

    def is_deterministic(self) -> bool:
        return all(self.metadata.get(field) not in {None, ""} for field in PROVENANCE_FIELDS)

    def families(self) -> list[str]:
        return sorted({record.family_id for record in self.records})

    def records_for_family(self, family_id: str) -> list[TaskRecord]:
        return [record for record in self.records if record.family_id == family_id]


@dataclass(frozen=True)
class TaskExample:
    text: str
    label: int
    variant: str
    pair_id: str
    template_id: str
    family_id: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class TaskValidationResult:
    valid: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    pair_counts_by_family: dict[str, int]
    deterministic_provenance: bool
    confirmatory_floor_met: bool


def load_task_suite(path: str | Path) -> TaskSuite:
    task_path = Path(path)
    if not task_path.exists():
        raise ValueError(f"task file does not exist: {task_path}")
    raw = task_path.read_text(encoding="utf-8")
    if task_path.suffix.lower() == ".json":
        payload = json.loads(raw)
    else:
        payload = yaml.safe_load(raw)
    if not isinstance(payload, dict):
        raise ValueError("task file must contain a mapping with metadata and records")

    metadata = dict(payload.get("metadata") or {})
    raw_records = payload.get("records")
    if not isinstance(raw_records, list):
        raise ValueError("task file must contain a records list")

    records = []
    for index, row in enumerate(raw_records):
        if not isinstance(row, dict):
            raise ValueError(f"task record {index} must be a mapping")
        missing = [field for field in REQUIRED_RECORD_FIELDS if field not in row]
        if missing:
            raise ValueError(f"task record {index} missing required fields: {', '.join(missing)}")
        records.append(
            TaskRecord(
                x_pos=_string_field(row, "x_pos", index),
                x_neg=_string_field(row, "x_neg", index),
                x_para=_string_field(row, "x_para", index),
                x_decoy=_string_field(row, "x_decoy", index),
                pair_id=_string_field(row, "pair_id", index),
                template_id=_string_field(row, "template_id", index),
                family_id=_string_field(row, "family_id", index),
                metadata=dict(row.get("metadata") or {}),
            )
        )
    return TaskSuite(records=tuple(records), metadata=metadata, source_path=task_path)


def validate_task_suite(
    suite: TaskSuite,
    *,
    confirmatory: bool,
    exploratory: bool,
    min_n: int,
    require_decoys: bool = True,
) -> TaskValidationResult:
    errors: list[str] = []
    warnings: list[str] = []
    if min_n <= 0:
        errors.append("--min-n must be positive")
    if confirmatory and min_n < CONFIRMATORY_MIN_PAIRS_PER_FAMILY:
        errors.append(
            f"confirmatory --min-n={min_n} is below the committed suite floor "
            f"{CONFIRMATORY_MIN_PAIRS_PER_FAMILY}"
        )

    seen_pairs: set[tuple[str, str]] = set()
    for record in suite.records:
        if require_decoys and not record.x_decoy.strip():
            errors.append(f"pair {record.pair_id} in family {record.family_id} is missing x_decoy")
        pair_key = (record.family_id, record.pair_id)
        if pair_key in seen_pairs:
            warnings.append(f"duplicate pair_id {record.pair_id} in family {record.family_id}")
        seen_pairs.add(pair_key)

    pair_counts = suite.pair_counts_by_family()
    if not pair_counts:
        errors.append("task suite contains no contrast pairs")
    for family, count in pair_counts.items():
        if count < min_n:
            errors.append(f"family {family} has {count} pairs, below --min-n={min_n}")
        if confirmatory and count < CONFIRMATORY_MIN_PAIRS_PER_FAMILY:
            errors.append(
                f"family {family} has {count} pairs, below confirmatory floor "
                f"{CONFIRMATORY_MIN_PAIRS_PER_FAMILY}"
            )

    deterministic = suite.is_deterministic()
    if confirmatory and not exploratory and not deterministic:
        errors.append("confirmatory task suite lacks deterministic generator provenance")
    if exploratory and not deterministic:
        warnings.append("exploratory task suite lacks deterministic generator provenance; claims are capped")

    return TaskValidationResult(
        valid=not errors,
        errors=tuple(errors),
        warnings=tuple(warnings),
        pair_counts_by_family=pair_counts,
        deterministic_provenance=deterministic,
        confirmatory_floor_met=all(count >= CONFIRMATORY_MIN_PAIRS_PER_FAMILY for count in pair_counts.values()),
    )


def examples_for_probe(records: list[TaskRecord]) -> list[TaskExample]:
    examples: list[TaskExample] = []
    for record in records:
        base = {
            "pair_id": record.pair_id,
            "template_id": record.template_id,
            "family_id": record.family_id,
            "metadata": record.metadata,
        }
        examples.append(TaskExample(text=record.x_pos, label=1, variant="pos", **base))
        examples.append(TaskExample(text=record.x_para, label=1, variant="para", **base))
        examples.append(TaskExample(text=record.x_neg, label=0, variant="neg", **base))
        examples.append(TaskExample(text=record.x_decoy, label=0, variant="decoy", **base))
    return examples


def examples_for_primary_probe(records: list[TaskRecord]) -> list[TaskExample]:
    examples: list[TaskExample] = []
    for record in records:
        base = {
            "pair_id": record.pair_id,
            "template_id": record.template_id,
            "family_id": record.family_id,
            "metadata": record.metadata,
        }
        examples.append(TaskExample(text=record.x_pos, label=1, variant="pos", **base))
        examples.append(TaskExample(text=record.x_neg, label=0, variant="neg", **base))
    return examples


def examples_for_decoy_probe(records: list[TaskRecord]) -> list[TaskExample]:
    examples: list[TaskExample] = []
    for record in records:
        base = {
            "pair_id": record.pair_id,
            "template_id": record.template_id,
            "family_id": record.family_id,
            "metadata": record.metadata,
        }
        examples.append(TaskExample(text=record.x_pos, label=1, variant="pos", **base))
        examples.append(TaskExample(text=record.x_decoy, label=0, variant="decoy", **base))
    return examples


def _string_field(row: dict[str, Any], field_name: str, index: int) -> str:
    value = row[field_name]
    if value is None:
        raise ValueError(f"task record {index} field {field_name} may not be null")
    text = str(value)
    if not text:
        raise ValueError(f"task record {index} field {field_name} may not be empty")
    return text
