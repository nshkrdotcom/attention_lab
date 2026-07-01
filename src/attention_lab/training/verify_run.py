from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import yaml

from attention_lab.training.checkpointing import load_checkpoint
from attention_lab.training.data_manifest import DataManifestError, sha256_file, verify_data_manifest


class RunVerificationError(ValueError):
    pass


def _require_file(path: Path) -> None:
    if not path.is_file():
        raise RunVerificationError(f"Missing required file: {path}")


def _require_dir(path: Path) -> None:
    if not path.is_dir():
        raise RunVerificationError(f"Missing required directory: {path}")


def load_jsonl_metrics(path: Path) -> list[dict[str, Any]]:
    _require_file(path)
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RunVerificationError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
    if not rows:
        raise RunVerificationError(f"No metric rows found in {path}")
    return rows


def load_csv_metrics(path: Path) -> list[dict[str, str]]:
    _require_file(path)
    with path.open(newline="", encoding="utf-8") as f:
        try:
            rows = list(csv.DictReader(f))
        except csv.Error as exc:
            raise RunVerificationError(f"Invalid CSV in {path}: {exc}") from exc
    if not rows:
        raise RunVerificationError(f"No metric rows found in {path}")
    return rows


def verify_metrics(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    events = [row.get("event") for row in metrics]
    for event in ("checkpoint", "train", "val"):
        if event not in events:
            raise RunVerificationError(f"Missing required metric event: {event}")

    for row in metrics:
        if row.get("event") == "val" and row.get("val_loss") is not None and row.get("val_perplexity") is not None:
            expected = math.exp(float(row["val_loss"]))
            observed = float(row["val_perplexity"])
            if not math.isclose(observed, expected, rel_tol=1e-6, abs_tol=1e-6):
                raise RunVerificationError(
                    f"val_perplexity mismatch at step {row.get('step')}: "
                    f"expected {expected}, got {observed}"
                )
        if row.get("event") == "train":
            tokens_per_sec = row.get("tokens_per_sec")
            if tokens_per_sec is None or float(tokens_per_sec) <= 0:
                raise RunVerificationError(f"Non-positive tokens_per_sec for train row at step {row.get('step')}")

    max_step = max(int(row["step"]) for row in metrics if row.get("step") is not None)
    return {
        "max_step": max_step,
        "train_event_count": events.count("train"),
        "val_event_count": events.count("val"),
        "checkpoint_event_count": events.count("checkpoint"),
    }


def verify_run(
    run_dir: str | Path,
    *,
    expect_complete_training: bool = False,
    expect_sample: bool = False,
    expect_eval_loss: bool = False,
    expect_hellaswag: bool = False,
    expect_data_manifest: bool = False,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise RunVerificationError(f"Run directory does not exist: {run_dir}")

    required_files = [
        "config.yaml",
        "config_source.txt",
        "environment.txt",
        "git_commit.txt",
        "metrics.jsonl",
        "metrics.csv",
        "checkpoints/ckpt_last.pt",
    ]
    for relative_path in required_files:
        _require_file(run_dir / relative_path)
    _require_dir(run_dir / "samples")
    _require_dir(run_dir / "evals")

    with (run_dir / "config.yaml").open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise RunVerificationError(f"Run config is not a mapping: {run_dir / 'config.yaml'}")

    metrics = load_jsonl_metrics(run_dir / "metrics.jsonl")
    load_csv_metrics(run_dir / "metrics.csv")
    summary = verify_metrics(metrics)
    data_manifest_ok = False

    if expect_data_manifest:
        manifest_path = run_dir / "data_manifest.json"
        sha_path = run_dir / "data_manifest.sha256"
        _require_file(manifest_path)
        _require_file(sha_path)
        expected_sha = sha_path.read_text(encoding="utf-8").strip()
        actual_sha = sha256_file(manifest_path)
        if expected_sha != actual_sha:
            raise RunVerificationError(
                f"data_manifest.sha256 mismatch: expected {expected_sha}, got {actual_sha}"
            )
        data_root = Path(config["data"]["data_root"])
        if data_root.is_dir():
            try:
                verify_data_manifest(data_root, manifest_path, verify_hashes=True)
            except DataManifestError as exc:
                raise RunVerificationError(str(exc)) from exc
        data_manifest_ok = True

    if expect_complete_training:
        max_steps = int(config["train"]["max_steps"])
        if summary["max_step"] != max_steps:
            raise RunVerificationError(f"Final metric step {summary['max_step']} does not equal max_steps {max_steps}")
        checkpoint = load_checkpoint(run_dir / "checkpoints" / "ckpt_last.pt", device="cpu")
        if int(checkpoint["step"]) != max_steps:
            raise RunVerificationError(f"ckpt_last.pt step {checkpoint['step']} does not equal max_steps {max_steps}")
        _require_file(run_dir / "samples" / "sample_step_last.txt")

    if expect_sample:
        _require_file(run_dir / "samples" / "sample_step_last.txt")
    if expect_eval_loss:
        _require_file(run_dir / "evals" / "val_loss.json")
        json.loads((run_dir / "evals" / "val_loss.json").read_text(encoding="utf-8"))
    if expect_hellaswag:
        _require_file(run_dir / "evals" / "hellaswag.json")
        json.loads((run_dir / "evals" / "hellaswag.json").read_text(encoding="utf-8"))

    return {
        "run_dir": str(run_dir),
        **summary,
        "data_manifest": data_manifest_ok,
        "ok": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--expect-complete-training", action="store_true")
    parser.add_argument("--expect-sample", action="store_true")
    parser.add_argument("--expect-eval-loss", action="store_true")
    parser.add_argument("--expect-hellaswag", action="store_true")
    parser.add_argument("--expect-data-manifest", action="store_true")
    args = parser.parse_args()
    result = verify_run(
        args.run_dir,
        expect_complete_training=args.expect_complete_training,
        expect_sample=args.expect_sample,
        expect_eval_loss=args.expect_eval_loss,
        expect_hellaswag=args.expect_hellaswag,
        expect_data_manifest=args.expect_data_manifest,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
