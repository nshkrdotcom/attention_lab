from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch

from attention_lab.models.attention.multi_qkv_common import (
    MultiQKVDebugRouteOverride,
    is_multi_qkv_attention,
    override_multi_qkv_routes,
)
from attention_lab.models.gpt import GPT, config_from_dict
from attention_lab.training.checkpointing import load_checkpoint
from attention_lab.training.config import load_config
from attention_lab.training.data_loader import TokenShardLoader
from attention_lab.training.runtime import autocast_context, device_type_from_device, dtype_from_name

EXPERIMENT_ID = "E002_multitrack_qkv_shift_register"


def _default_output(checkpoint_path: Path) -> Path:
    if checkpoint_path.parent.name == "checkpoints":
        return checkpoint_path.parent.parent / "evals" / "qkv_track_destructive_test.json"
    return Path("reports") / "qkv_track_destructive_test.json"


def _run_name_from_checkpoint(checkpoint_path: Path) -> str:
    if checkpoint_path.parent.name == "checkpoints":
        return checkpoint_path.parent.parent.name
    return checkpoint_path.stem


def _load_config(config_path: str | None, checkpoint: dict[str, Any]) -> dict[str, Any]:
    if config_path is not None:
        return load_config(config_path)
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise ValueError("checkpoint does not contain a config; pass --config")
    return config


@torch.no_grad()
def _loss_and_logits(
    model: GPT,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    device_type: str,
    dtype: torch.dtype,
    schedule_mode: str,
) -> tuple[float, torch.Tensor]:
    with autocast_context(device_type, dtype):
        logits, loss = model(x, y, step=None, schedule_mode=schedule_mode)
    assert loss is not None
    return float(loss.item()), logits.detach().float()


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _max(values: list[float]) -> float:
    return max(values) if values else float("nan")


def _evaluate_perturbation(
    *,
    model: GPT,
    loader: TokenShardLoader,
    num_batches: int,
    device: str,
    device_type: str,
    dtype: torch.dtype,
    schedule_mode: str,
    override: MultiQKVDebugRouteOverride,
    name: str,
) -> dict[str, Any]:
    natural_losses: list[float] = []
    perturbed_losses: list[float] = []
    mean_logit_deltas: list[float] = []
    max_logit_deltas: list[float] = []
    loader.reset()
    for _ in range(num_batches):
        x, y = loader.next_batch()
        x = x.to(device)
        y = y.to(device)
        natural_loss, natural_logits = _loss_and_logits(
            model,
            x,
            y,
            device_type=device_type,
            dtype=dtype,
            schedule_mode=schedule_mode,
        )
        with override_multi_qkv_routes(model, override):
            perturbed_loss, perturbed_logits = _loss_and_logits(
                model,
                x,
                y,
                device_type=device_type,
                dtype=dtype,
                schedule_mode=schedule_mode,
            )
        delta = (perturbed_logits - natural_logits).abs()
        natural_losses.append(natural_loss)
        perturbed_losses.append(perturbed_loss)
        mean_logit_deltas.append(float(delta.mean().item()))
        max_logit_deltas.append(float(delta.max().item()))

    natural_loss = _mean(natural_losses)
    perturbed_loss = _mean(perturbed_losses)
    return {
        "name": name,
        "natural_loss": natural_loss,
        "perturbed_loss": perturbed_loss,
        "loss_delta": perturbed_loss - natural_loss,
        "mean_abs_logit_delta": _mean(mean_logit_deltas),
        "max_abs_logit_delta": _max(max_logit_deltas),
    }


def _passes_destructive_gate(perturbations: list[dict[str, Any]]) -> bool:
    for row in perturbations:
        natural_loss = float(row["natural_loss"])
        perturbed_loss = float(row["perturbed_loss"])
        if not math.isfinite(natural_loss) or not math.isfinite(perturbed_loss):
            return False
        if float(row["mean_abs_logit_delta"]) > 1e-5 or abs(float(row["loss_delta"])) > 1e-5:
            return True
    return False


def _perturbation_specs(args: argparse.Namespace, track_count: int) -> list[tuple[str, MultiQKVDebugRouteOverride]]:
    requested = args.perturbation or ["rotate_tracks", "force_track", "zero_selected"]
    specs: list[tuple[str, MultiQKVDebugRouteOverride]] = []
    for perturbation in requested:
        if perturbation == "rotate_tracks":
            specs.append(("rotate_tracks", MultiQKVDebugRouteOverride(mode="rotate_tracks")))
        elif perturbation == "zero_selected":
            specs.append(("zero_selected", MultiQKVDebugRouteOverride(mode="zero_selected")))
        elif perturbation == "force_track":
            forced_track = 0 if args.forced_track is None else int(args.forced_track)
            if forced_track < 0 or forced_track >= track_count:
                raise ValueError(f"--forced-track must be in [0, {track_count})")
            specs.append(
                (
                    f"force_track_{forced_track}",
                    MultiQKVDebugRouteOverride(mode="force_track", forced_track=forced_track),
                )
            )
        else:
            raise ValueError(f"unknown perturbation: {perturbation}")
    return specs


def run_destructive_test(args: argparse.Namespace) -> dict[str, Any]:
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but torch.cuda.is_available() is False.")
    device_type = device_type_from_device(device)

    checkpoint_path = Path(args.checkpoint)
    checkpoint = load_checkpoint(checkpoint_path, device=device)
    config = _load_config(args.config, checkpoint)
    model_config = config_from_dict(config["model"], config["data"])
    if not is_multi_qkv_attention(model_config.attention_type):
        raise ValueError(f"qkv_track_destructive_test requires a Multi-QKV checkpoint, got {model_config.attention_type}")

    model = GPT(model_config)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    dtype = dtype_from_name(args.dtype or config["train"].get("dtype", "bfloat16"))

    loader = TokenShardLoader(
        args.data_root or config["data"]["data_root"],
        args.B,
        args.T,
        0,
        1,
        args.split,
        True,
    )

    bank = model.multi_qkv_bank
    if bank is None:
        raise ValueError("Loaded model has no multi_qkv_bank")

    perturbations = [
        _evaluate_perturbation(
            model=model,
            loader=loader,
            num_batches=args.num_batches,
            device=device,
            device_type=device_type,
            dtype=dtype,
            schedule_mode=args.mode,
            override=override,
            name=name,
        )
        for name, override in _perturbation_specs(args, bank.track_count)
    ]

    result = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "run_name": config["run"]["name"] if "run" in config else _run_name_from_checkpoint(checkpoint_path),
        "checkpoint": str(checkpoint_path),
        "config": str(args.config) if args.config else None,
        "attention_type": model_config.attention_type,
        "split": args.split,
        "B": args.B,
        "T": args.T,
        "num_batches": args.num_batches,
        "mode": args.mode,
        "perturbations": perturbations,
        "destructive_test_passed": _passes_destructive_gate(perturbations),
    }
    out_path = Path(args.out) if args.out else _default_output(checkpoint_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--B", type=int, default=2)
    parser.add_argument("--T", type=int, default=128)
    parser.add_argument("--num-batches", "--max-batches", dest="num_batches", type=int, default=4)
    parser.add_argument("--dtype", default=None, choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mode", default="eval", choices=["eval", "generate"])
    parser.add_argument(
        "--perturbation",
        action="append",
        choices=["force_track", "rotate_tracks", "zero_selected"],
        default=None,
    )
    parser.add_argument("--forced-track", type=int, default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    print(json.dumps(run_destructive_test(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
