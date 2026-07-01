from __future__ import annotations

import argparse
import json

import numpy as np
import pytest

from attention_lab.training.data_manifest import sha256_file, write_data_manifest
from attention_lab.training.verify_data import verify_data_root


def test_manifest_writes_shard_hashes(tmp_path, write_tiny_shards):
    data_root = tmp_path / "data"
    write_tiny_shards(data_root)
    manifest_path = data_root / "manifest.json"

    manifest = write_data_manifest(data_root, manifest_path)

    assert manifest_path.exists()
    assert manifest["dataset"] == "HuggingFaceFW/fineweb-edu"
    assert manifest["dataset_config"] == "sample-10BT"
    assert manifest["tokenizer"] == "gpt2"
    assert manifest["split"] == "train"
    assert manifest["total_train_tokens"] == 256
    assert manifest["total_val_tokens"] == 128
    by_path = {record["path"]: record for record in manifest["shards"]}
    train_record = by_path["edufineweb_train_000001.npy"]
    assert train_record["sha256"] == sha256_file(data_root / train_record["path"])

    disk_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert disk_manifest["shards"] == manifest["shards"]


def test_verify_data_passes_with_correct_manifest(tmp_path, write_tiny_shards):
    data_root = tmp_path / "data"
    write_tiny_shards(data_root)
    manifest_path = data_root / "manifest.json"
    write_data_manifest(data_root, manifest_path)

    verify_data_root(
        argparse.Namespace(data_root=str(data_root), manifest=str(manifest_path), verify_hashes=True)
    )


def test_verify_data_fails_when_shard_changes_after_manifest(tmp_path, write_tiny_shards):
    data_root = tmp_path / "data"
    write_tiny_shards(data_root)
    manifest_path = data_root / "manifest.json"
    write_data_manifest(data_root, manifest_path)
    np.save(data_root / "edufineweb_train_000001.npy", np.arange(256, dtype=np.uint16) + 1)

    with pytest.raises(SystemExit, match="hash mismatch|field min_token"):
        verify_data_root(
            argparse.Namespace(data_root=str(data_root), manifest=str(manifest_path), verify_hashes=True)
        )
