import os
from pathlib import Path
from typing import Any

import torch

from attention_lab.training.data_manifest import read_run_manifest


def save_checkpoint(
    out_dir: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    step: int,
    train_loss: float | None = None,
    val_loss: float | None = None,
    train_loader_state: dict[str, Any] | None = None,
    val_loader_state: dict[str, Any] | None = None,
) -> Path:
    checkpoint_dir = Path(out_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    manifest_metadata = read_run_manifest(out_dir)
    data_manifest = manifest_metadata[0] if manifest_metadata is not None else None
    data_manifest_sha256 = manifest_metadata[1] if manifest_metadata is not None else None
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config,
        "step": step,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_loader_state": train_loader_state,
        "val_loader_state": val_loader_state,
        "rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "data_manifest": data_manifest,
        "data_manifest_sha256": data_manifest_sha256,
    }
    step_path = checkpoint_dir / f"ckpt_step_{step:06d}.pt"
    last_path = checkpoint_dir / "ckpt_last.pt"
    tmp_path = step_path.with_suffix(".tmp")
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, step_path)
    torch.save(checkpoint, last_path)
    return step_path


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(path, map_location=device, weights_only=False)


def restore_rng_state(checkpoint: dict[str, Any]) -> None:
    rng_state = checkpoint.get("rng_state")
    if rng_state is not None:
        torch.set_rng_state(rng_state.cpu() if hasattr(rng_state, "cpu") else rng_state)
    cuda_rng_state_all = checkpoint.get("cuda_rng_state_all")
    if cuda_rng_state_all is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_rng_state_all)
