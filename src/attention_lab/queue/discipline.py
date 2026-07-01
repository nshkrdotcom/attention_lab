from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

REQUIRED_HYPOTHESIS_FIELDS = (
    "CLAIM",
    "KILL_CONDITION",
    "MECHANISM_PROOF",
    "NEAREST_BORING_EXPLANATION",
    "CONTROL_THAT_RULES_IT_OUT",
)


@dataclass(frozen=True)
class HypothesisValidation:
    ok: bool
    missing_fields: list[str]
    path: Path


def default_hypothesis_path(config_path: str | Path, config: dict[str, Any]) -> Path:
    queue_config = config.get("queue", {})
    if queue_config.get("hypothesis_doc"):
        return Path(queue_config["hypothesis_doc"])
    parts = Path(config_path).parts
    experiment_id = None
    if "experiments" in parts:
        index = parts.index("experiments")
        if index + 1 < len(parts):
            experiment_id = parts[index + 1]
    run_name = config["run"]["name"]
    if experiment_id is None:
        return Path("docs") / "experiments" / f"hypothesis_{run_name}.md"
    return Path("docs") / "experiments" / experiment_id / f"hypothesis_{run_name}.md"


def _field_blocks(text: str) -> dict[str, str]:
    blocks: dict[str, list[str]] = {}
    current_field = None
    for line in text.splitlines():
        stripped = line.strip()
        field = stripped[:-1] if stripped.endswith(":") else None
        if field in REQUIRED_HYPOTHESIS_FIELDS:
            current_field = field
            blocks.setdefault(field, [])
            continue
        if current_field is not None:
            blocks[current_field].append(line)
    return {field: "\n".join(lines).strip() for field, lines in blocks.items()}


def validate_hypothesis_doc(path: str | Path) -> HypothesisValidation:
    path = Path(path)
    if not path.is_file():
        return HypothesisValidation(False, list(REQUIRED_HYPOTHESIS_FIELDS), path)
    blocks = _field_blocks(path.read_text(encoding="utf-8"))
    missing = [field for field in REQUIRED_HYPOTHESIS_FIELDS if not blocks.get(field)]
    return HypothesisValidation(not missing, missing, path)
