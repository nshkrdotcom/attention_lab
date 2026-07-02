from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


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
    payload: dict[str, Any]


def load_hypothesis_doc(path: str | Path) -> HypothesisDoc:
    doc_path = Path(path)
    if not doc_path.exists():
        raise ValueError(f"hypothesis doc does not exist: {doc_path}")
    if doc_path.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError("mechanism hypothesis docs must use docs/mechanisms/hypotheses/<name>.yaml")
    payload = yaml.safe_load(doc_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("hypothesis doc must be a YAML object")
    missing = [field for field in REQUIRED_HYPOTHESIS_FIELDS if field not in payload or payload[field] in (None, "")]
    if missing:
        raise ValueError(f"missing required hypothesis fields: {', '.join(missing)}")
    if not isinstance(payload["TARGET_SITES"], list) or not payload["TARGET_SITES"]:
        raise ValueError("TARGET_SITES must be a non-empty list")
    if not isinstance(payload["TASK_CONTRASTS"], list) or not payload["TASK_CONTRASTS"]:
        raise ValueError("TASK_CONTRASTS must be a non-empty list")
    if int(payload["MIN_N"]) <= 0:
        raise ValueError("MIN_N must be positive")
    if str(payload["EXPECTED_DIRECTION"]) not in {"positive", "negative"}:
        raise ValueError("EXPECTED_DIRECTION must be positive or negative")
    return HypothesisDoc(path=doc_path, payload=dict(payload))
