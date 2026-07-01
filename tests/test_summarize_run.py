from __future__ import annotations

import json

import pytest

from attention_lab.training.config import save_config
from attention_lab.training.summarize_run import summarize_run
from attention_lab.training.train import train
from attention_lab.training.verify_run import RunVerificationError


def test_summary_works_on_tiny_run(tmp_path, write_tiny_shards, tiny_config):
    data_root = tmp_path / "data"
    write_tiny_shards(data_root)
    config = tiny_config(tmp_path, data_root)
    config_path = tmp_path / "tiny.yaml"
    save_config(config, config_path)
    train(config_path, overwrite=True)

    run_dir = tmp_path / "runs" / "tiny_test_run"
    summary = summarize_run(run_dir)
    assert summary["run_name"] == "tiny_test_run"
    assert summary["max_step"] == 2
    assert summary["train_event_count"] >= 2
    assert summary["val_event_count"] >= 2
    assert summary["checkpoint_count"] >= 1
    assert (run_dir / "evals" / "run_summary.json").exists()


def test_summary_fails_clearly_on_missing_metrics(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with pytest.raises(RunVerificationError, match="Missing metrics"):
        summarize_run(run_dir)


def test_summary_supports_old_and_new_memory_fields(tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "evals").mkdir(parents=True)
    rows = [
        {"event": "train", "step": 1, "tokens_per_sec": 10.0, "peak_vram_mb": 5.0},
        {
            "event": "train",
            "step": 2,
            "tokens_per_sec": 20.0,
            "peak_vram_allocated_mb": 7.0,
            "peak_vram_reserved_mb": 9.0,
            "nvidia_smi_memory_mb": 11.0,
        },
        {"event": "val", "step": 2, "val_loss": 1.0, "val_perplexity": 2.718281828459045},
        {"event": "checkpoint", "step": 2, "checkpoint": "ckpt"},
    ]
    (run_dir / "metrics.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    summary = summarize_run(run_dir)

    assert summary["median_tokens_per_sec"] == 15.0
    assert summary["peak_vram_mb"] == 7.0
    assert summary["peak_vram_allocated_mb"] == 7.0
    assert summary["peak_vram_reserved_mb"] == 9.0
    assert summary["nvidia_smi_memory_mb"] == 11.0
