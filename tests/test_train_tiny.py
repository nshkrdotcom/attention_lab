from __future__ import annotations

from attention_lab.training.config import save_config
from attention_lab.training.data_manifest import write_data_manifest
from attention_lab.training.train import train


def test_tiny_training_run_creates_expected_artifacts(tmp_path, write_tiny_shards, tiny_config):
    data_root = tmp_path / "data"
    write_tiny_shards(data_root)
    write_data_manifest(data_root, data_root / "manifest.json")
    config = tiny_config(tmp_path, data_root)
    config_path = tmp_path / "tiny.yaml"
    save_config(config, config_path)

    train(config_path, overwrite=True)

    run_dir = tmp_path / "runs" / "tiny_test_run"
    expected = [
        "config.yaml",
        "config_source.txt",
        "environment.txt",
        "git_commit.txt",
        "metrics.jsonl",
        "metrics.csv",
        "checkpoints/ckpt_last.pt",
        "samples",
        "evals",
    ]
    for relative_path in expected:
        assert (run_dir / relative_path).exists(), relative_path
    assert (run_dir / "data_manifest.json").exists()
    assert (run_dir / "data_manifest.sha256").exists()
    assert (run_dir / "samples" / "sample_step_last.txt").read_text(encoding="utf-8").strip()
