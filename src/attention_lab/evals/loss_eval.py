import argparse
import json
import math
from pathlib import Path

import torch

from attention_lab.models.gpt import GPT, config_from_dict
from attention_lab.training.checkpointing import load_checkpoint
from attention_lab.training.data_manifest import (
    manifest_mismatch_message,
    manifest_payloads_match,
    read_data_root_manifest,
    read_run_manifest,
)
from attention_lab.training.data_loader import TokenShardLoader
from attention_lab.training.runtime import device_type_from_device, dtype_from_name
from attention_lab.training.train import evaluate_loss


def default_eval_output(checkpoint_path: Path) -> Path:
    if checkpoint_path.parent.name == "checkpoints":
        return checkpoint_path.parent.parent / "evals" / "val_loss.json"
    return Path("reports") / "val_loss.json"


def _checkpoint_manifest(checkpoint: dict, checkpoint_path: Path) -> dict | None:
    if checkpoint.get("data_manifest") is not None:
        return checkpoint["data_manifest"]
    if checkpoint_path.parent.name == "checkpoints":
        run_manifest = read_run_manifest(checkpoint_path.parent.parent)
        if run_manifest is not None:
            return run_manifest[0]
    return None


def validate_eval_data_manifest(
    checkpoint: dict,
    checkpoint_path: Path,
    data_root: str | Path,
    *,
    allow_mismatch: bool,
) -> dict:
    if allow_mismatch:
        return {"status": "explicitly_skipped"}
    expected_manifest = _checkpoint_manifest(checkpoint, checkpoint_path)
    if expected_manifest is None:
        return {"status": "unavailable_checkpoint_manifest"}
    current_manifest = read_data_root_manifest(data_root)
    if current_manifest is None:
        raise ValueError(
            f"Checkpoint has a data manifest, but current data root has no manifest.json: {data_root}. "
            "Pass --allow-data-manifest-mismatch only for intentional cross-data evaluation."
        )
    if not manifest_payloads_match(expected_manifest, current_manifest[0]):
        raise ValueError(manifest_mismatch_message("checkpoint", expected_manifest, "current", current_manifest[0]))
    return {"status": "matched", "data_manifest_sha256": current_manifest[1]}


def run_eval(args: argparse.Namespace) -> dict:
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but torch.cuda.is_available() is False.")
    device_type = device_type_from_device(device)

    checkpoint_path = Path(args.checkpoint)
    checkpoint = load_checkpoint(checkpoint_path, device=device)
    config = checkpoint["config"]
    data_config = dict(config["data"])
    if args.data_root:
        data_config["data_root"] = args.data_root
    manifest_check = validate_eval_data_manifest(
        checkpoint,
        checkpoint_path,
        data_config["data_root"],
        allow_mismatch=bool(getattr(args, "allow_data_manifest_mismatch", False)),
    )

    model_config = config_from_dict(config["model"], data_config)
    model = GPT(model_config)
    model.load_state_dict(checkpoint["model"])
    model.to(device)

    train_config = config["train"]
    B = args.B or int(train_config["B"])
    T = args.T or int(train_config["T"])
    val_steps = args.val_steps or int(train_config["val_steps"])
    dtype = dtype_from_name(args.dtype or train_config.get("dtype", "bfloat16"))

    loader = TokenShardLoader(data_config["data_root"], B, T, 0, 1, args.split, True)
    loss = evaluate_loss(model, loader, val_steps, device, device_type, dtype, ddp=False)
    result = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "steps": val_steps,
        "loss": loss,
        "perplexity": math.exp(loss),
        "manifest_check": manifest_check,
    }

    out_path = Path(args.out) if args.out else default_eval_output(checkpoint_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--val_steps", type=int, default=None)
    parser.add_argument("--B", type=int, default=None)
    parser.add_argument("--T", type=int, default=None)
    parser.add_argument("--dtype", default=None, choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default=None)
    parser.add_argument("--allow-data-manifest-mismatch", action="store_true")
    args = parser.parse_args()
    result = run_eval(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
