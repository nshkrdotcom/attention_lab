from __future__ import annotations

import argparse

import numpy as np
import pytest

from attention_lab.evals.generation_eval import run_generate
from attention_lab.evals.loss_eval import run_eval
from attention_lab.models.gpt import GPT, config_from_dict
from attention_lab.training.checkpointing import save_checkpoint
from attention_lab.training.config import save_config
from attention_lab.training.data_manifest import write_data_manifest
from attention_lab.training.optim import build_optimizer
from attention_lab.training.train import train


def make_tiny_checkpoint(tmp_path, tiny_config):
    model_config = config_from_dict(tiny_config["model"], tiny_config["data"])
    model = GPT(model_config)
    optimizer = build_optimizer(model, weight_decay=0.1, learning_rate=0.001, device_type="cpu", master_process=False)
    return save_checkpoint(tmp_path, model, optimizer, tiny_config, step=0)


def test_eval_loss_can_load_tiny_checkpoint(tmp_path, write_tiny_shards, tiny_config):
    data_root = tmp_path / "data"
    write_tiny_shards(data_root)
    config = tiny_config(tmp_path, data_root)
    checkpoint = make_tiny_checkpoint(tmp_path, config)

    result = run_eval(
        argparse.Namespace(
            checkpoint=str(checkpoint),
            data_root=str(data_root),
            split="val",
            val_steps=1,
            B=2,
            T=8,
            dtype="float32",
            device="cpu",
            out=str(tmp_path / "eval.json"),
            allow_data_manifest_mismatch=False,
        )
    )
    assert result["loss"] > 0
    assert result["perplexity"] > 1
    assert (tmp_path / "eval.json").exists()


def test_generation_eval_writes_nonempty_sample(tmp_path, write_tiny_shards, tiny_config):
    data_root = tmp_path / "data"
    write_tiny_shards(data_root)
    config = tiny_config(tmp_path, data_root)
    checkpoint = make_tiny_checkpoint(tmp_path, config)
    out_path = tmp_path / "sample.txt"

    samples = run_generate(
        argparse.Namespace(
            checkpoint=str(checkpoint),
            prompt="!",
            num_samples=1,
            max_new_tokens=2,
            top_k=10,
            temperature=1.0,
            seed=42,
            dtype="float32",
            device="cpu",
            out=str(out_path),
        )
    )
    assert samples
    assert out_path.read_text(encoding="utf-8").strip()


def test_eval_loss_refuses_data_manifest_mismatch(tmp_path, write_tiny_shards, tiny_config):
    data_root = tmp_path / "data"
    write_tiny_shards(data_root, train_tokens=512, val_tokens=128)
    write_data_manifest(data_root, data_root / "manifest.json")
    config = tiny_config(tmp_path, data_root, max_steps=1)
    config["train"]["save_every"] = 1
    config["sample"]["sample_every"] = 1
    config_path = tmp_path / "tiny.yaml"

    save_config(config, config_path)
    train(config_path, overwrite=True)

    other_root = tmp_path / "other_data"
    write_tiny_shards(other_root, vocab_size=64, train_tokens=512, val_tokens=128)

    np.save(other_root / "edufineweb_val_000000.npy", np.arange(128, dtype=np.uint16) + 1)
    write_data_manifest(other_root, other_root / "manifest.json")

    with pytest.raises(ValueError, match="manifest mismatch"):
        run_eval(
            argparse.Namespace(
                checkpoint=str(tmp_path / "runs" / "tiny_test_run" / "checkpoints" / "ckpt_last.pt"),
                data_root=str(other_root),
                split="val",
                val_steps=1,
                B=2,
                T=8,
                dtype="float32",
                device="cpu",
                out=str(tmp_path / "eval.json"),
                allow_data_manifest_mismatch=False,
            )
        )


def test_eval_loss_allows_explicit_data_manifest_mismatch(tmp_path, write_tiny_shards, tiny_config):
    data_root = tmp_path / "data"
    write_tiny_shards(data_root, train_tokens=512, val_tokens=128)
    write_data_manifest(data_root, data_root / "manifest.json")
    config = tiny_config(tmp_path, data_root, max_steps=1)
    config["train"]["save_every"] = 1
    config["sample"]["sample_every"] = 1
    config_path = tmp_path / "tiny.yaml"

    save_config(config, config_path)
    train(config_path, overwrite=True)

    other_root = tmp_path / "other_data"
    write_tiny_shards(other_root, train_tokens=512, val_tokens=128)
    write_data_manifest(other_root, other_root / "manifest.json")
    result = run_eval(
        argparse.Namespace(
            checkpoint=str(tmp_path / "runs" / "tiny_test_run" / "checkpoints" / "ckpt_last.pt"),
            data_root=str(other_root),
            split="val",
            val_steps=1,
            B=2,
            T=8,
            dtype="float32",
            device="cpu",
            out=str(tmp_path / "eval.json"),
            allow_data_manifest_mismatch=True,
        )
    )
    assert result["manifest_check"]["status"] == "explicitly_skipped"
