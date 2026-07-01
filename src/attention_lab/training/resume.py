from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from attention_lab.training.data_manifest import (
    load_data_manifest,
    manifest_mismatch_message,
    manifest_payloads_match,
    read_data_root_manifest,
)


class ResumeCompatibilityError(ValueError):
    pass


def _require_equal(current: Any, checkpoint: Any, label: str) -> None:
    if current != checkpoint:
        raise ResumeCompatibilityError(
            f"Resume config mismatch for {label}: current={current!r}, checkpoint={checkpoint!r}"
        )


def validate_resume_compatibility(
    current_config: dict[str, Any],
    checkpoint_config: dict[str, Any],
    *,
    checkpoint_step: int,
) -> None:
    _require_equal(current_config["model"], checkpoint_config["model"], "model config")
    for key in ("tokenizer", "vocab_size"):
        _require_equal(current_config["data"].get(key), checkpoint_config["data"].get(key), f"data.{key}")
    for key in ("B", "T", "total_batch_size"):
        _require_equal(current_config["train"].get(key), checkpoint_config["train"].get(key), f"train.{key}")
    for key in ("weight_decay", "learning_rate", "min_lr", "warmup_steps", "grad_clip"):
        _require_equal(current_config["train"].get(key), checkpoint_config["train"].get(key), f"train.{key}")
    if int(current_config["train"]["max_steps"]) <= checkpoint_step:
        raise ResumeCompatibilityError(
            f"train.max_steps must be greater than checkpoint step {checkpoint_step} when resuming"
        )


def assert_model_state_compatible(model: torch.nn.Module, checkpoint_state: dict[str, torch.Tensor]) -> None:
    current_state = model.state_dict()
    current_keys = set(current_state)
    checkpoint_keys = set(checkpoint_state)
    missing = sorted(current_keys - checkpoint_keys)
    unexpected = sorted(checkpoint_keys - current_keys)
    if missing or unexpected:
        raise ResumeCompatibilityError(f"Checkpoint model keys differ: missing={missing}, unexpected={unexpected}")
    mismatched = [
        key
        for key in sorted(current_keys)
        if tuple(current_state[key].shape) != tuple(checkpoint_state[key].shape)
    ]
    if mismatched:
        details = ", ".join(
            f"{key}: current={tuple(current_state[key].shape)} checkpoint={tuple(checkpoint_state[key].shape)}"
            for key in mismatched[:5]
        )
        raise ResumeCompatibilityError(f"Checkpoint model shape mismatch: {details}")


def validate_resume_data_manifest(
    run_dir: str | Path,
    data_root: str | Path,
    checkpoint: dict[str, Any] | None = None,
) -> None:
    current_manifest_metadata = read_data_root_manifest(data_root)
    if current_manifest_metadata is None:
        return
    current_manifest = current_manifest_metadata[0]

    checkpoint_manifest = checkpoint.get("data_manifest") if checkpoint is not None else None
    if checkpoint_manifest is not None:
        if not manifest_payloads_match(checkpoint_manifest, current_manifest):
            raise ResumeCompatibilityError(
                manifest_mismatch_message("checkpoint", checkpoint_manifest, "current", current_manifest)
            )
        return

    run_manifest_path = Path(run_dir) / "data_manifest.json"
    if not run_manifest_path.is_file():
        return
    run_manifest = load_data_manifest(run_manifest_path)
    if not manifest_payloads_match(run_manifest, current_manifest):
        raise ResumeCompatibilityError(
            manifest_mismatch_message("run", run_manifest, "current", current_manifest)
        )
