from __future__ import annotations

import argparse

import pytest

from attention_lab.evals.loss_eval import run_eval
from attention_lab.training.config import save_config
from attention_lab.training.data_manifest import write_data_manifest
from attention_lab.training.summarize_run import summarize_run
from attention_lab.training.train import train
from attention_lab.training.verify_run import verify_run


def cp_tiny_config(tiny_config, tmp_path, data_root, attention_type: str, run_name: str, lambda_fixed: bool = False):
    config = tiny_config(tmp_path, data_root)
    config["run"]["name"] = run_name
    config["run"]["out_dir"] = str(tmp_path / "runs" / run_name)
    config["model"]["attention_type"] = attention_type
    config["model"]["cp_rank"] = 4
    config["model"]["cp_lambda_init"] = 0.0
    config["model"]["cp_lambda_trainable"] = not lambda_fixed
    config["model"]["cp_lambda_fixed"] = lambda_fixed
    config["diagnostics"] = {"attention_diagnostics_every": 1}
    return config


@pytest.mark.parametrize(
    ("attention_type", "run_name", "lambda_fixed"),
    [
        ("cp_bilinear", "tiny_cp_bilinear", False),
        ("cp_trilinear", "tiny_cp_trilinear", False),
        ("cp_trilinear", "tiny_cp_trilinear_lambda0", True),
    ],
)
def test_tiny_cp_training_run_creates_verifiable_artifacts(
    tmp_path,
    write_tiny_shards,
    tiny_config,
    attention_type,
    run_name,
    lambda_fixed,
):
    data_root = tmp_path / "data"
    write_tiny_shards(data_root)
    write_data_manifest(data_root, data_root / "manifest.json")
    config = cp_tiny_config(tiny_config, tmp_path, data_root, attention_type, run_name, lambda_fixed)
    config_path = tmp_path / f"{run_name}.yaml"
    save_config(config, config_path)

    train(config_path, overwrite=True)

    run_dir = tmp_path / "runs" / run_name
    result = verify_run(
        run_dir,
        expect_complete_training=True,
        expect_sample=True,
        expect_data_manifest=True,
    )
    assert result["ok"] is True
    summary = summarize_run(run_dir)
    assert summary["max_step"] == config["train"]["max_steps"]

    diagnostics_path = run_dir / "evals" / "attention_diagnostics.jsonl"
    assert diagnostics_path.exists()
    assert diagnostics_path.read_text(encoding="utf-8").strip()

    eval_result = run_eval(
        argparse.Namespace(
            checkpoint=str(run_dir / "checkpoints" / "ckpt_last.pt"),
            data_root=str(data_root),
            split="val",
            val_steps=1,
            B=2,
            T=8,
            dtype="float32",
            device="cpu",
            out=str(run_dir / "evals" / "val_loss.json"),
            allow_data_manifest_mismatch=False,
        )
    )
    assert eval_result["loss"] > 0
