from __future__ import annotations

import json
import math

import pytest

from attention_lab.training.config import save_config
from attention_lab.training.data_manifest import write_data_manifest
from attention_lab.training.train import train
from attention_lab.training.verify_run import RunVerificationError, verify_metrics, verify_run


def make_tiny_run(tmp_path, write_tiny_shards, tiny_config):
    data_root = tmp_path / "data"
    write_tiny_shards(data_root)
    config = tiny_config(tmp_path, data_root)
    config_path = tmp_path / "tiny.yaml"
    save_config(config, config_path)
    train(config_path, overwrite=True)
    return tmp_path / "runs" / "tiny_test_run"


def test_verify_run_passes_on_tiny_run(tmp_path, write_tiny_shards, tiny_config):
    run_dir = make_tiny_run(tmp_path, write_tiny_shards, tiny_config)
    result = verify_run(run_dir, expect_complete_training=True, expect_sample=True)
    assert result["ok"] is True
    assert result["max_step"] == 2


def test_verify_run_fails_when_metrics_missing(tmp_path, write_tiny_shards, tiny_config):
    run_dir = make_tiny_run(tmp_path, write_tiny_shards, tiny_config)
    (run_dir / "metrics.jsonl").unlink()
    with pytest.raises(RunVerificationError, match="metrics.jsonl"):
        verify_run(run_dir)


def test_verify_run_fails_when_ckpt_last_missing(tmp_path, write_tiny_shards, tiny_config):
    run_dir = make_tiny_run(tmp_path, write_tiny_shards, tiny_config)
    (run_dir / "checkpoints" / "ckpt_last.pt").unlink()
    with pytest.raises(RunVerificationError, match="ckpt_last"):
        verify_run(run_dir)


def test_verify_run_checks_val_perplexity_consistency(tmp_path, write_tiny_shards, tiny_config):
    run_dir = make_tiny_run(tmp_path, write_tiny_shards, tiny_config)
    rows = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    for row in rows:
        if row.get("event") == "val":
            row["val_loss"] = 1.0
            row["val_perplexity"] = math.e + 1.0
            break
    (run_dir / "metrics.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    with pytest.raises(RunVerificationError, match="val_perplexity"):
        verify_run(run_dir)


def test_verify_run_checks_positive_tokens_per_sec(tmp_path, write_tiny_shards, tiny_config):
    run_dir = make_tiny_run(tmp_path, write_tiny_shards, tiny_config)
    rows = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    for row in rows:
        if row.get("event") == "train":
            row["tokens_per_sec"] = 0
            break
    (run_dir / "metrics.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    with pytest.raises(RunVerificationError, match="tokens_per_sec"):
        verify_run(run_dir)


def test_verify_run_can_require_data_manifest(tmp_path, write_tiny_shards, tiny_config):
    data_root = tmp_path / "data"
    write_tiny_shards(data_root)
    write_data_manifest(data_root, data_root / "manifest.json")
    config = tiny_config(tmp_path, data_root)
    config_path = tmp_path / "tiny.yaml"
    save_config(config, config_path)
    train(config_path, overwrite=True)

    run_dir = tmp_path / "runs" / "tiny_test_run"
    result = verify_run(run_dir, expect_data_manifest=True)
    assert result["ok"] is True
    assert result["data_manifest"] is True


def test_verify_run_fails_when_data_manifest_sha_mismatches(tmp_path, write_tiny_shards, tiny_config):
    data_root = tmp_path / "data"
    write_tiny_shards(data_root)
    write_data_manifest(data_root, data_root / "manifest.json")
    config = tiny_config(tmp_path, data_root)
    config_path = tmp_path / "tiny.yaml"
    save_config(config, config_path)
    train(config_path, overwrite=True)

    run_dir = tmp_path / "runs" / "tiny_test_run"
    (run_dir / "data_manifest.sha256").write_text("bad\n", encoding="utf-8")
    with pytest.raises(RunVerificationError, match="data_manifest.sha256"):
        verify_run(run_dir, expect_data_manifest=True)


def test_verify_metrics_accepts_new_memory_fields():
    summary = verify_metrics(
        [
            {
                "event": "train",
                "step": 1,
                "tokens_per_sec": 1.0,
                "peak_vram_allocated_mb": 2.0,
                "peak_vram_reserved_mb": 3.0,
            },
            {"event": "val", "step": 1, "val_loss": 1.0, "val_perplexity": math.e},
            {"event": "checkpoint", "step": 1, "checkpoint": "ckpt"},
        ]
    )
    assert summary["max_step"] == 1
