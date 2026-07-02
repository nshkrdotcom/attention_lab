from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


HYPOTHESIS_DOC_ROOT = Path("docs/mechanisms/hypotheses")
REQUIRED_HYPOTHESIS_FIELDS = (
    "CLAIM",
    "KILL_CONDITION",
    "MECHANISM_PROOF",
    "NEAREST_BORING_EXPLANATION",
    "CONTROL_THAT_RULES_IT_OUT",
    "TARGET_SITES",
    "TASK_CONTRASTS",
    "PRIMARY_METRIC",
    "STATISTICAL_TEST",
    "MIN_N",
    "FDR_SCOPE",
    "EXPECTED_DIRECTION",
)


@dataclass(frozen=True)
class HypothesisDoc:
    path: Path
    fields: dict[str, Any]


def validate_hypothesis_doc(path: str | Path, *, repo_root: str | Path = ".") -> HypothesisDoc:
    doc_path = Path(path)
    if not doc_path.exists():
        raise ValueError(f"hypothesis doc does not exist: {doc_path}")
    if doc_path.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError("mechanism hypothesis docs must use YAML under docs/mechanisms/hypotheses")

    repo = Path(repo_root).resolve()
    resolved = doc_path.resolve()
    try:
        relative = resolved.relative_to(repo)
    except ValueError:
        relative = doc_path
    if not _is_under_hypothesis_root(relative):
        raise ValueError(
            "mechanism hypothesis docs must live under docs/mechanisms/hypotheses "
            "to avoid split conventions"
        )

    data = yaml.safe_load(doc_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("hypothesis doc must be a YAML mapping")
    missing = [field for field in REQUIRED_HYPOTHESIS_FIELDS if _empty(data.get(field))]
    if missing:
        raise ValueError(f"hypothesis doc missing required fields: {', '.join(missing)}")
    return HypothesisDoc(path=doc_path, fields=dict(data))


def exploratory_hypothesis_label() -> dict[str, Any]:
    return {
        "mode": "exploratory",
        "claim_cap": "exploratory_probe_signal",
        "reason": "no pre-registered hypothesis doc was supplied",
    }


def _is_under_hypothesis_root(path: Path) -> bool:
    parts = path.parts
    root_parts = HYPOTHESIS_DOC_ROOT.parts
    return len(parts) >= len(root_parts) and parts[: len(root_parts)] == root_parts


def _empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) == 0
    return False
