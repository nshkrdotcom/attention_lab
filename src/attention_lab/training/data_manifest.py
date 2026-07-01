from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_DATASET = "HuggingFaceFW/fineweb-edu"
DEFAULT_DATASET_CONFIG = "sample-10BT"
DEFAULT_TOKENIZER = "gpt2"


class DataManifestError(ValueError):
    pass


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _split_from_name(path: Path) -> str:
    name = path.name
    if "val" in name:
        return "val"
    if "train" in name:
        return "train"
    raise DataManifestError(f"Cannot infer split from shard name: {path}")


def _shard_sort_key(path: Path) -> tuple[int, str]:
    split = _split_from_name(path)
    split_rank = {"val": 0, "train": 1}[split]
    return split_rank, path.name


def build_data_manifest(
    data_root: str | Path,
    *,
    tokenizer: str = DEFAULT_TOKENIZER,
    dataset: str = DEFAULT_DATASET,
    dataset_config: str = DEFAULT_DATASET_CONFIG,
) -> dict[str, Any]:
    data_root = Path(data_root)
    shards = sorted(data_root.glob("*.npy"), key=_shard_sort_key)
    if not shards:
        raise DataManifestError(f"No .npy shards found in {data_root}")

    shard_records = []
    totals = {"train": 0, "val": 0}
    for path in shards:
        tokens = np.load(path)
        if tokens.size == 0:
            raise DataManifestError(f"Empty shard: {path}")
        split = _split_from_name(path)
        num_tokens = int(tokens.size)
        totals[split] += num_tokens
        shard_records.append(
            {
                "path": path.name,
                "split": split,
                "num_tokens": num_tokens,
                "dtype": str(tokens.dtype),
                "min_token": int(tokens.min()),
                "max_token": int(tokens.max()),
                "sha256": sha256_file(path),
            }
        )

    return {
        "data_root": str(data_root),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tokenizer": tokenizer,
        "dataset": dataset,
        "dataset_config": dataset_config,
        "split": "train",
        "shards": shard_records,
        "total_train_tokens": totals["train"],
        "total_val_tokens": totals["val"],
    }


def write_data_manifest(
    data_root: str | Path,
    out: str | Path,
    *,
    tokenizer: str = DEFAULT_TOKENIZER,
    dataset: str = DEFAULT_DATASET,
    dataset_config: str = DEFAULT_DATASET_CONFIG,
) -> dict[str, Any]:
    manifest = build_data_manifest(
        data_root,
        tokenizer=tokenizer,
        dataset=dataset,
        dataset_config=dataset_config,
    )
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def load_data_manifest(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("shards"), list):
        raise DataManifestError(f"Invalid data manifest: {path}")
    return manifest


def manifest_comparison_payload(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "tokenizer": manifest.get("tokenizer"),
        "dataset": manifest.get("dataset"),
        "dataset_config": manifest.get("dataset_config"),
        "split": manifest.get("split"),
        "shards": manifest.get("shards"),
        "total_train_tokens": manifest.get("total_train_tokens"),
        "total_val_tokens": manifest.get("total_val_tokens"),
    }


def manifest_payloads_match(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return manifest_comparison_payload(left) == manifest_comparison_payload(right)


def manifest_mismatch_message(left_label: str, left: dict[str, Any], right_label: str, right: dict[str, Any]) -> str:
    return (
        "data manifest mismatch:\n"
        f"{left_label}={json.dumps(manifest_comparison_payload(left), sort_keys=True)}\n"
        f"{right_label}={json.dumps(manifest_comparison_payload(right), sort_keys=True)}"
    )


def read_run_manifest(run_dir: str | Path) -> tuple[dict[str, Any], str] | None:
    manifest_path = Path(run_dir) / "data_manifest.json"
    if not manifest_path.is_file():
        return None
    text = manifest_path.read_text(encoding="utf-8")
    return load_data_manifest(manifest_path), sha256_text(text)


def read_data_root_manifest(data_root: str | Path) -> tuple[dict[str, Any], str] | None:
    manifest_path = Path(data_root) / "manifest.json"
    if not manifest_path.is_file():
        return None
    text = manifest_path.read_text(encoding="utf-8")
    return load_data_manifest(manifest_path), sha256_text(text)


def verify_data_manifest(
    data_root: str | Path,
    manifest_path: str | Path,
    *,
    verify_hashes: bool = False,
) -> dict[str, Any]:
    data_root = Path(data_root)
    manifest = load_data_manifest(manifest_path)
    actual_by_name = {path.name: path for path in data_root.glob("*.npy")}
    expected_names = {record["path"] for record in manifest["shards"]}
    actual_names = set(actual_by_name)
    missing = sorted(expected_names - actual_names)
    extra = sorted(actual_names - expected_names)
    if missing:
        raise DataManifestError(f"Manifest references missing shards: {missing}")
    if extra:
        raise DataManifestError(f"Data root contains shards not present in manifest: {extra}")

    for record in manifest["shards"]:
        path = actual_by_name[record["path"]]
        tokens = np.load(path)
        checks = {
            "split": _split_from_name(path),
            "num_tokens": int(tokens.size),
            "dtype": str(tokens.dtype),
            "min_token": int(tokens.min()),
            "max_token": int(tokens.max()),
        }
        for key, actual_value in checks.items():
            if record.get(key) != actual_value:
                raise DataManifestError(
                    f"Manifest mismatch for {path.name} field {key}: "
                    f"expected {record.get(key)!r}, got {actual_value!r}"
                )
        if verify_hashes:
            actual_hash = sha256_file(path)
            if record.get("sha256") != actual_hash:
                raise DataManifestError(
                    f"Manifest hash mismatch for {path.name}: expected {record.get('sha256')}, got {actual_hash}"
                )

    return manifest


def copy_manifest_to_run(data_root: str | Path, run_dir: str | Path) -> str | None:
    manifest_path = Path(data_root) / "manifest.json"
    if not manifest_path.is_file():
        return None
    run_dir = Path(run_dir)
    run_manifest_path = run_dir / "data_manifest.json"
    manifest_text = manifest_path.read_text(encoding="utf-8")
    run_manifest_path.write_text(manifest_text, encoding="utf-8")
    digest = sha256_text(manifest_text)
    (run_dir / "data_manifest.sha256").write_text(digest + "\n", encoding="utf-8")
    return digest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--dataset-config", default=DEFAULT_DATASET_CONFIG)
    args = parser.parse_args()
    manifest = write_data_manifest(
        args.data_root,
        args.out,
        tokenizer=args.tokenizer,
        dataset=args.dataset,
        dataset_config=args.dataset_config,
    )
    print(json.dumps(manifest, indent=2))
