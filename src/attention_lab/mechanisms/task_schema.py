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
RESTORATION_TOKEN_ID_FIELDS = ("target_token_id", "foil_token_id")
RESTORATION_TOKEN_TEXT_FIELDS = ("target_token_text", "foil_token_text")
RESTORATION_ALIGNMENT_FIELDS = (
    "clean_answer_position",
    "corrupted_answer_position",
    "patch_token_indices",
    "clean_corrupt_token_alignment",
)
RESTORATION_ALIGNMENT_MODES = ("same_length", "explicit_patch_indices")


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
    restoration_token_metadata_valid: bool = True


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
    require_restoration_tokens: bool = False,
    tokenizer_name: str = "gpt2",
    vocab_size: int = 50304,
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

    token_errors = validate_restoration_token_metadata(
        suite,
        require=require_restoration_tokens,
        tokenizer_name=tokenizer_name,
        vocab_size=vocab_size,
    )
    errors.extend(token_errors)

    return TaskValidationResult(
        valid=not errors,
        errors=tuple(errors),
        warnings=tuple(warnings),
        pair_counts_by_family=pair_counts,
        deterministic_provenance=deterministic,
        confirmatory_floor_met=bool(pair_counts)
        and all(count >= CONFIRMATORY_MIN_PAIRS_PER_FAMILY for count in pair_counts.values()),
        restoration_token_metadata_valid=not token_errors,
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


def validate_restoration_token_metadata(
    suite: TaskSuite,
    *,
    require: bool,
    tokenizer_name: str = "gpt2",
    vocab_size: int = 50304,
) -> list[str]:
    if tokenizer_name != "gpt2":
        return [f"restoration token metadata validation only supports GPT-2 tokenizer, got {tokenizer_name!r}"]
    if not require and not any(_has_any_restoration_token_field(record.metadata) for record in suite.records):
        return []

    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    errors: list[str] = []
    for record in suite.records:
        metadata = record.metadata
        missing = [
            field
            for field in (*RESTORATION_TOKEN_ID_FIELDS, *RESTORATION_TOKEN_TEXT_FIELDS, *RESTORATION_ALIGNMENT_FIELDS)
            if field not in metadata
        ]
        if missing:
            if require:
                errors.append(
                    f"pair {record.pair_id} in family {record.family_id} lacks restoration token metadata: "
                    f"{', '.join(missing)}"
                )
            continue

        target_id = metadata.get("target_token_id")
        foil_id = metadata.get("foil_token_id")
        target_text = metadata.get("target_token_text")
        foil_text = metadata.get("foil_token_text")
        for field_name, value in (("target_token_id", target_id), ("foil_token_id", foil_id)):
            if isinstance(value, bool) or not isinstance(value, int):
                errors.append(
                    f"pair {record.pair_id} in family {record.family_id} has non-integer {field_name}"
                )
            elif value < 0 or value >= vocab_size:
                errors.append(
                    f"pair {record.pair_id} in family {record.family_id} has out-of-range {field_name}={value}"
                )
        for text_field, token_text, id_field, token_id in (
            ("target_token_text", target_text, "target_token_id", target_id),
            ("foil_token_text", foil_text, "foil_token_id", foil_id),
        ):
            if not isinstance(token_text, str) or not token_text:
                errors.append(
                    f"pair {record.pair_id} in family {record.family_id} has invalid {text_field}"
                )
                continue
            encoded = enc.encode(token_text)
            if len(encoded) != 1:
                errors.append(
                    f"pair {record.pair_id} in family {record.family_id} has {text_field} "
                    f"that is not a single GPT-2 token"
                )
                continue
            if isinstance(token_id, int) and not isinstance(token_id, bool) and encoded[0] != token_id:
                errors.append(
                    f"pair {record.pair_id} in family {record.family_id} {text_field} encodes to "
                    f"{encoded[0]}, not {id_field}={token_id}"
                )
        if target_id == foil_id and isinstance(target_id, int):
            errors.append(
                f"pair {record.pair_id} in family {record.family_id} uses identical target and foil token ids"
            )
        errors.extend(_validate_restoration_alignment(record, enc, vocab_size=vocab_size))
    return errors


def _has_any_restoration_token_field(metadata: dict[str, Any]) -> bool:
    return any(
        field in metadata
        for field in (*RESTORATION_TOKEN_ID_FIELDS, *RESTORATION_TOKEN_TEXT_FIELDS, *RESTORATION_ALIGNMENT_FIELDS)
    )


def _validate_restoration_alignment(record: TaskRecord, enc: Any, *, vocab_size: int) -> list[str]:
    _ = vocab_size
    metadata = record.metadata
    errors: list[str] = []
    clean_tokens = enc.encode(record.x_pos)
    corrupted_tokens = enc.encode(record.x_neg)
    clean_position = metadata.get("clean_answer_position")
    corrupted_position = metadata.get("corrupted_answer_position")
    alignment_mode = metadata.get("clean_corrupt_token_alignment")
    patch_indices = metadata.get("patch_token_indices")
    clean_patch_indices = metadata.get("clean_patch_token_indices", patch_indices)
    corrupted_patch_indices = metadata.get("corrupted_patch_token_indices", patch_indices)

    if isinstance(clean_position, bool) or not isinstance(clean_position, int):
        errors.append(
            f"pair {record.pair_id} in family {record.family_id} has invalid clean_answer_position"
        )
    elif clean_position < 0 or clean_position >= len(clean_tokens):
        errors.append(
            f"pair {record.pair_id} in family {record.family_id} has out-of-range clean_answer_position"
        )

    if isinstance(corrupted_position, bool) or not isinstance(corrupted_position, int):
        errors.append(
            f"pair {record.pair_id} in family {record.family_id} has invalid corrupted_answer_position"
        )
    elif corrupted_position < 0 or corrupted_position >= len(corrupted_tokens):
        errors.append(
            f"pair {record.pair_id} in family {record.family_id} has out-of-range corrupted_answer_position"
        )

    if alignment_mode not in RESTORATION_ALIGNMENT_MODES:
        errors.append(
            f"pair {record.pair_id} in family {record.family_id} has invalid clean_corrupt_token_alignment"
        )
    elif alignment_mode == "same_length" and len(clean_tokens) != len(corrupted_tokens):
        errors.append(
            f"pair {record.pair_id} in family {record.family_id} declares same_length alignment but "
            f"clean/corrupted token lengths differ ({len(clean_tokens)} != {len(corrupted_tokens)})"
        )
    elif alignment_mode == "explicit_patch_indices" and len(clean_tokens) == len(corrupted_tokens):
        # Same-length prompts may still use explicit indices, but this note keeps the metadata honest.
        pass

    errors.extend(
        _validate_index_list(
            record=record,
            field_name="patch_token_indices",
            values=patch_indices,
            max_len=min(len(clean_tokens), len(corrupted_tokens)),
        )
    )
    errors.extend(
        _validate_index_list(
            record=record,
            field_name="clean_patch_token_indices",
            values=clean_patch_indices,
            max_len=len(clean_tokens),
        )
    )
    errors.extend(
        _validate_index_list(
            record=record,
            field_name="corrupted_patch_token_indices",
            values=corrupted_patch_indices,
            max_len=len(corrupted_tokens),
        )
    )
    if isinstance(clean_patch_indices, list) and isinstance(corrupted_patch_indices, list):
        if len(clean_patch_indices) != len(corrupted_patch_indices):
            errors.append(
                f"pair {record.pair_id} in family {record.family_id} has mismatched clean/corrupted "
                "patch index counts"
            )
    if len(clean_tokens) != len(corrupted_tokens) and alignment_mode != "explicit_patch_indices":
        errors.append(
            f"pair {record.pair_id} in family {record.family_id} has different clean/corrupted token lengths "
            "without explicit_patch_indices alignment"
        )
    return errors


def _validate_index_list(
    *,
    record: TaskRecord,
    field_name: str,
    values: Any,
    max_len: int,
) -> list[str]:
    if not isinstance(values, list) or not values:
        return [f"pair {record.pair_id} in family {record.family_id} has invalid {field_name}"]
    errors: list[str] = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, int):
            errors.append(
                f"pair {record.pair_id} in family {record.family_id} has non-integer "
                f"{field_name}[{index}]"
            )
        elif value < 0 or value >= max_len:
            errors.append(
                f"pair {record.pair_id} in family {record.family_id} has out-of-range "
                f"{field_name}[{index}]={value}"
            )
    return errors


def restoration_alignment_metadata(
    record: TaskRecord,
    *,
    tokenizer_name: str,
    block_size: int,
    vocab_size: int,
) -> dict[str, Any]:
    """Return validated restoration alignment metadata for one task record.

    This helper is intentionally strict because full-suite patching must not
    silently patch whole clean caches into corrupted prompts when token positions
    are not aligned.
    """
    if tokenizer_name != "gpt2":
        raise ValueError(f"restoration alignment only supports GPT-2 tokenizer, got {tokenizer_name!r}")
    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    errors = validate_restoration_token_metadata(
        TaskSuite(records=(record,), metadata={field: "x" for field in PROVENANCE_FIELDS}),
        require=True,
        tokenizer_name=tokenizer_name,
        vocab_size=vocab_size,
    )
    if errors:
        raise ValueError("; ".join(errors))
    clean_tokens = enc.encode(record.x_pos)[:block_size] or [0]
    corrupted_tokens = enc.encode(record.x_neg)[:block_size] or [0]
    metadata = record.metadata
    patch_indices = list(metadata["patch_token_indices"])
    clean_patch_indices = list(metadata.get("clean_patch_token_indices", patch_indices))
    corrupted_patch_indices = list(metadata.get("corrupted_patch_token_indices", patch_indices))
    if len(clean_tokens) != len(corrupted_tokens) and metadata["clean_corrupt_token_alignment"] != "explicit_patch_indices":
        raise ValueError("different clean/corrupted token lengths require explicit_patch_indices alignment")
    if len(clean_patch_indices) != len(corrupted_patch_indices):
        raise ValueError("clean_patch_token_indices and corrupted_patch_token_indices must have matching lengths")
    for field_name, indices, length in (
        ("clean_patch_token_indices", clean_patch_indices, len(clean_tokens)),
        ("corrupted_patch_token_indices", corrupted_patch_indices, len(corrupted_tokens)),
    ):
        for value in indices:
            if value < 0 or value >= length:
                raise ValueError(f"{field_name} contains out-of-range index {value}")
    return {
        "target_token_id": int(metadata["target_token_id"]),
        "foil_token_id": int(metadata["foil_token_id"]),
        "target_token_text": metadata["target_token_text"],
        "foil_token_text": metadata["foil_token_text"],
        "clean_answer_position": int(metadata["clean_answer_position"]),
        "corrupted_answer_position": int(metadata["corrupted_answer_position"]),
        "clean_patch_token_indices": clean_patch_indices,
        "corrupted_patch_token_indices": corrupted_patch_indices,
        "patch_token_indices": patch_indices,
        "clean_corrupt_token_alignment": metadata["clean_corrupt_token_alignment"],
        "clean_token_length": len(clean_tokens),
        "corrupted_token_length": len(corrupted_tokens),
    }
