from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def collect_attention_diagnostics(model: torch.nn.Module, step: int) -> list[dict[str, Any]]:
    rows = []
    blocks = getattr(getattr(model, "transformer", None), "h", [])
    for layer, block in enumerate(blocks):
        attention = getattr(block, "attn", None)
        diagnostics_fn = getattr(attention, "attention_diagnostics", None)
        if diagnostics_fn is None:
            continue
        row = diagnostics_fn(step=step, layer=layer)
        if row is not None:
            rows.append(row)
    return rows


def _experiment_id_from_out_dir(out_dir: Path) -> str | None:
    parts = out_dir.parts
    try:
        index = parts.index("experiments")
    except ValueError:
        return None
    if index + 1 >= len(parts):
        return None
    return parts[index + 1]


def _enrich_row(
    row: dict[str, Any],
    *,
    out_dir: Path,
    experiment_id: str | None,
    run_name: str | None,
) -> dict[str, Any]:
    enriched = dict(row)
    enriched.setdefault("schema_version", 1)
    resolved_experiment_id = experiment_id or _experiment_id_from_out_dir(out_dir)
    if resolved_experiment_id is not None:
        enriched.setdefault("experiment_id", resolved_experiment_id)
    enriched.setdefault("run_name", run_name or out_dir.name)
    return enriched


def append_attention_diagnostics(
    out_dir: str | Path,
    rows: list[dict[str, Any]],
    *,
    experiment_id: str | None = None,
    run_name: str | None = None,
) -> Path | None:
    if not rows:
        return None
    out_dir = Path(out_dir)
    path = out_dir / "evals" / "attention_diagnostics.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(
                json.dumps(
                    _enrich_row(
                        row,
                        out_dir=out_dir,
                        experiment_id=experiment_id,
                        run_name=run_name,
                    ),
                    sort_keys=True,
                )
                + "\n"
            )
    return path
