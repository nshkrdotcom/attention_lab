from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import yaml

from attention_lab.training.verify_run import RunVerificationError, load_jsonl_metrics


def _nonnull_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _metric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values = [_nonnull_float(row.get(key)) for row in rows]
    return [value for value in values if value is not None]


def summarize_run(run_dir: str | Path, *, write_json: bool = True) -> dict[str, Any]:
    run_dir = Path(run_dir)
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        raise RunVerificationError(f"Missing metrics.jsonl: {metrics_path}")

    config_path = run_dir / "config.yaml"
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    else:
        config = {}

    metrics = load_jsonl_metrics(metrics_path)
    train_rows = [row for row in metrics if row.get("event") == "train"]
    val_rows = [row for row in metrics if row.get("event") == "val"]
    checkpoint_rows = [row for row in metrics if row.get("event") == "checkpoint"]
    val_losses = _metric_values(val_rows, "val_loss")
    val_perplexities = _metric_values(val_rows, "val_perplexity")
    tokens_per_sec = _metric_values(train_rows, "tokens_per_sec")
    peak_vram_allocated_values = _metric_values(metrics, "peak_vram_allocated_mb")
    peak_vram_allocated_values.extend(_metric_values(metrics, "peak_vram_mb"))
    peak_vram_reserved_values = _metric_values(metrics, "peak_vram_reserved_mb")
    nvidia_smi_values = _metric_values(metrics, "nvidia_smi_memory_mb")
    steps = [int(row["step"]) for row in metrics if row.get("step") is not None]
    peak_vram_allocated = max(peak_vram_allocated_values) if peak_vram_allocated_values else None

    summary = {
        "run_dir": str(run_dir),
        "run_name": config.get("run", {}).get("name", run_dir.name),
        "max_step": max(steps) if steps else None,
        "train_event_count": len(train_rows),
        "val_event_count": len(val_rows),
        "initial_val_loss": val_losses[0] if val_losses else None,
        "final_val_loss": val_losses[-1] if val_losses else None,
        "best_val_loss": min(val_losses) if val_losses else None,
        "initial_val_perplexity": val_perplexities[0] if val_perplexities else None,
        "final_val_perplexity": val_perplexities[-1] if val_perplexities else None,
        "median_tokens_per_sec": statistics.median(tokens_per_sec) if tokens_per_sec else None,
        "peak_vram_mb": peak_vram_allocated,
        "peak_vram_allocated_mb": peak_vram_allocated,
        "peak_vram_reserved_mb": max(peak_vram_reserved_values) if peak_vram_reserved_values else None,
        "nvidia_smi_memory_mb": max(nvidia_smi_values) if nvidia_smi_values else None,
        "checkpoint_count": len(checkpoint_rows),
    }

    if write_json:
        output_path = run_dir / "evals" / "run_summary.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    args = parser.parse_args()
    summary = summarize_run(args.run_dir)
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
