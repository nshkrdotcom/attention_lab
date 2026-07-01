from __future__ import annotations

import json

import numpy as np
import pytest

from attention_lab.training.checkpointing import load_checkpoint
from attention_lab.training.config import save_config
from attention_lab.training.data_manifest import write_data_manifest
from attention_lab.training.train import train


def test_resume_continues_from_checkpoint_and_appends_metrics(tmp_path, write_tiny_shards, tiny_config):
    data_root = tmp_path / "data"
    write_tiny_shards(data_root, train_tokens=512, val_tokens=128)

    config = tiny_config(tmp_path, data_root, max_steps=1)
    config["train"]["save_every"] = 1
    config["sample"]["sample_every"] = 1
    config_path = tmp_path / "tiny.yaml"
    save_config(config, config_path)
    train(config_path, overwrite=True)

    run_dir = tmp_path / "runs" / "tiny_test_run"
    checkpoint_path = run_dir / "checkpoints" / "ckpt_last.pt"
    checkpoint = load_checkpoint(checkpoint_path)
    assert checkpoint["step"] == 1
    assert checkpoint["train_loader_state"] is not None
    metric_lines_before = (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()

    resumed_config = tiny_config(tmp_path, data_root, max_steps=2)
    resumed_config["train"]["save_every"] = 1
    resumed_config["sample"]["sample_every"] = 2
    save_config(resumed_config, config_path)
    train(config_path, resume_path=str(checkpoint_path))

    resumed_checkpoint = load_checkpoint(checkpoint_path)
    assert resumed_checkpoint["step"] == 2
    metric_lines_after = (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(metric_lines_after) > len(metric_lines_before)
    rows = [json.loads(line) for line in metric_lines_after]
    resume_rows = [row for row in rows if row.get("event") == "resume"]
    assert resume_rows
    assert resume_rows[-1]["step"] == 1
    assert (run_dir / "resume_from.txt").read_text(encoding="utf-8").strip() == str(checkpoint_path)


def test_resume_refuses_overwrite(tmp_path, write_tiny_shards, tiny_config):
    data_root = tmp_path / "data"
    write_tiny_shards(data_root, train_tokens=512, val_tokens=128)
    config = tiny_config(tmp_path, data_root, max_steps=1)
    config["train"]["save_every"] = 1
    config["sample"]["sample_every"] = 1
    config_path = tmp_path / "tiny.yaml"
    save_config(config, config_path)
    train(config_path, overwrite=True)

    checkpoint_path = tmp_path / "runs" / "tiny_test_run" / "checkpoints" / "ckpt_last.pt"
    with pytest.raises(ValueError, match="overwrite.*resume"):
        train(config_path, overwrite=True, resume_path=str(checkpoint_path))


def test_resume_refuses_incompatible_model_config(tmp_path, write_tiny_shards, tiny_config):
    data_root = tmp_path / "data"
    write_tiny_shards(data_root, train_tokens=512, val_tokens=128)
    config = tiny_config(tmp_path, data_root, max_steps=1)
    config["train"]["save_every"] = 1
    config["sample"]["sample_every"] = 1
    config_path = tmp_path / "tiny.yaml"
    save_config(config, config_path)
    train(config_path, overwrite=True)

    checkpoint_path = tmp_path / "runs" / "tiny_test_run" / "checkpoints" / "ckpt_last.pt"
    resumed_config = tiny_config(tmp_path, data_root, max_steps=2)
    resumed_config["model"]["n_embd"] = 64
    save_config(resumed_config, config_path)
    with pytest.raises(ValueError, match="model config"):
        train(config_path, resume_path=str(checkpoint_path))


def test_resume_refuses_batch_shape_change(tmp_path, write_tiny_shards, tiny_config):
    data_root = tmp_path / "data"
    write_tiny_shards(data_root, train_tokens=512, val_tokens=128)
    config = tiny_config(tmp_path, data_root, max_steps=1)
    config["train"]["save_every"] = 1
    config["sample"]["sample_every"] = 1
    config_path = tmp_path / "tiny.yaml"
    save_config(config, config_path)
    train(config_path, overwrite=True)

    checkpoint_path = tmp_path / "runs" / "tiny_test_run" / "checkpoints" / "ckpt_last.pt"
    resumed_config = tiny_config(tmp_path, data_root, max_steps=2)
    resumed_config["train"]["B"] = 1
    resumed_config["train"]["total_batch_size"] = 16
    save_config(resumed_config, config_path)
    with pytest.raises(ValueError, match="train.B"):
        train(config_path, resume_path=str(checkpoint_path))


def test_resume_refuses_data_manifest_change(tmp_path, write_tiny_shards, tiny_config):
    data_root = tmp_path / "data"
    write_tiny_shards(data_root, train_tokens=512, val_tokens=128)
    write_data_manifest(data_root, data_root / "manifest.json")
    config = tiny_config(tmp_path, data_root, max_steps=1)
    config["train"]["save_every"] = 1
    config["sample"]["sample_every"] = 1
    config_path = tmp_path / "tiny.yaml"
    save_config(config, config_path)
    train(config_path, overwrite=True)

    checkpoint_path = tmp_path / "runs" / "tiny_test_run" / "checkpoints" / "ckpt_last.pt"
    np.save(data_root / "edufineweb_train_000001.npy", np.arange(512, dtype=np.uint16) + 1)
    write_data_manifest(data_root, data_root / "manifest.json")
    resumed_config = tiny_config(tmp_path, data_root, max_steps=2)
    resumed_config["train"]["save_every"] = 1
    save_config(resumed_config, config_path)
    with pytest.raises(ValueError, match="data manifest mismatch"):
        train(config_path, resume_path=str(checkpoint_path))


def test_resume_uses_checkpoint_manifest_when_run_manifest_missing(tmp_path, write_tiny_shards, tiny_config):
    data_root = tmp_path / "data"
    write_tiny_shards(data_root, train_tokens=512, val_tokens=128)
    write_data_manifest(data_root, data_root / "manifest.json")
    config = tiny_config(tmp_path, data_root, max_steps=1)
    config["train"]["save_every"] = 1
    config["sample"]["sample_every"] = 1
    config_path = tmp_path / "tiny.yaml"
    save_config(config, config_path)
    train(config_path, overwrite=True)

    run_dir = tmp_path / "runs" / "tiny_test_run"
    checkpoint_path = run_dir / "checkpoints" / "ckpt_last.pt"
    (run_dir / "data_manifest.json").unlink()
    (run_dir / "data_manifest.sha256").unlink()
    np.save(data_root / "edufineweb_train_000001.npy", np.arange(512, dtype=np.uint16) + 1)
    write_data_manifest(data_root, data_root / "manifest.json")
    resumed_config = tiny_config(tmp_path, data_root, max_steps=2)
    resumed_config["train"]["save_every"] = 1
    save_config(resumed_config, config_path)
    with pytest.raises(ValueError, match="data manifest mismatch"):
        train(config_path, resume_path=str(checkpoint_path))
