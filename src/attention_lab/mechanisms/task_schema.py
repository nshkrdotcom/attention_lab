from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


REQUIRED_TEXT_FIELDS = ("x_pos", "x_neg", "x_para", "x_decoy")
REQUIRED_GROUP_FIELDS = ("pair_id", "template_id", "family_id")


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
class TaskExample:
    text: str
    label: int
    pair_id: str
    template_id: str
    family_id: str
    variant: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class TaskSuite:
    records: tuple[TaskRecord, ...]
    metadata: dict[str, Any]
    source_path: Path | None = None

    @property
    def deterministic_provenance(self) -> bool:
        required = {
            "generator_name",
            "generator_version",
            "template_set",
            "filler_set",
            "generation_seed",
            "created_at",
        }
        return required.issubset(self.metadata)

    @property
    def pair_counts_per_family(self) -> dict[str, int]:
        counts: dict[str, set[str]] = {}
        for record in self.records:
            counts.setdefault(record.family_id, set()).add(record.pair_id)
        return {family: len(pair_ids) for family, pair_ids in sorted(counts.items())}

    @property
    def has_decoys(self) -> bool:
        return all(bool(record.x_decoy) for record in self.records)

    def examples(self) -> tuple[TaskExample, ...]:
        examples: list[TaskExample] = []
        for record in self.records:
            for variant, label in (("x_pos", 1), ("x_para", 1), ("x_neg", 0), ("x_decoy", 0)):
                examples.append(
                    TaskExample(
                        text=getattr(record, variant),
                        label=label,
                        pair_id=record.pair_id,
                        template_id=record.template_id,
                        family_id=record.family_id,
                        variant=variant,
                        metadata=dict(record.metadata),
                    )
                )
        return tuple(examples)

    def validate_confirmatory_floor(self, *, min_pairs_per_family: int) -> tuple[bool, list[str]]:
        reasons = []
        for family, count in self.pair_counts_per_family.items():
            if count < min_pairs_per_family:
                reasons.append(f"family {family!r} has {count} pairs < required {min_pairs_per_family}")
        if not self.deterministic_provenance:
            reasons.append("task suite lacks deterministic template/filler provenance")
        if not self.has_decoys:
            reasons.append("task suite is missing decoys")
        return (not reasons, reasons)


def load_task_suite(path: str | Path) -> TaskSuite:
    task_path = Path(path)
    if not task_path.exists():
        raise ValueError(f"task file does not exist: {task_path}")
    if task_path.suffix.lower() in {".yaml", ".yml"}:
        payload = yaml.safe_load(task_path.read_text(encoding="utf-8"))
    else:
        payload = json.loads(task_path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        metadata: dict[str, Any] = {}
        rows = payload
    elif isinstance(payload, dict):
        metadata = dict(payload.get("metadata", {}))
        rows = payload.get("records")
    else:
        raise ValueError("task file must contain a list of records or an object with records")
    if not isinstance(rows, list):
        raise ValueError("task file must contain a records list")
    records = tuple(_normalize_record(row, index=index) for index, row in enumerate(rows))
    if not records:
        raise ValueError("task suite must contain at least one record")
    return TaskSuite(records=records, metadata=metadata, source_path=task_path)


def _normalize_record(row: Any, *, index: int) -> TaskRecord:
    if not isinstance(row, dict):
        raise ValueError(f"task record {index} must be an object")
    missing = [field for field in (*REQUIRED_TEXT_FIELDS, *REQUIRED_GROUP_FIELDS) if field not in row]
    if missing:
        raise ValueError(f"task record {index} missing required fields: {', '.join(missing)}")
    for field_name in REQUIRED_TEXT_FIELDS:
        if not isinstance(row[field_name], str) or not row[field_name].strip():
            raise ValueError(f"task record {index} field {field_name} must be a non-empty string")
    for field_name in REQUIRED_GROUP_FIELDS:
        if not isinstance(row[field_name], str) or not row[field_name].strip():
            raise ValueError(f"task record {index} field {field_name} must be a non-empty string")
    metadata = row.get("metadata", {})
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict):
        raise ValueError(f"task record {index} metadata must be an object")
    return TaskRecord(
        x_pos=row["x_pos"],
        x_neg=row["x_neg"],
        x_para=row["x_para"],
        x_decoy=row["x_decoy"],
        pair_id=row["pair_id"],
        template_id=row["template_id"],
        family_id=row["family_id"],
        metadata=dict(metadata),
    )
