from __future__ import annotations

import json
from pathlib import Path

import torch
import yaml

from attention_lab.mechanisms.backfill import build_experiment_inventory, write_backfill_outputs


def write_config(path: Path, *, run_name: str, attention_type: str, out_dir: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "run": {"name": run_name, "out_dir": out_dir, "seed": 1},
                "data": {
                    "data_root": "data/fineweb_edu_100m",
                    "tokenizer": "gpt2",
                    "vocab_size": 64,
                    "train_tokens": 100,
                    "val_tokens": 20,
                },
                "model": {
                    "attention_type": attention_type,
                    "block_size": 8,
                    "n_layer": 1,
                    "n_head": 1,
                    "n_embd": 8,
                    "dropout": 0.0,
                    "bias": False,
                },
                "train": {
                    "device": "cpu",
                    "dtype": "float32",
                    "compile": False,
                    "eval_at_start": True,
                    "B": 1,
                    "T": 8,
                    "total_batch_size": 8,
                    "max_steps": 2,
                    "grad_clip": 1.0,
                    "weight_decay": 0.1,
                    "learning_rate": 0.001,
                    "min_lr": 0.0001,
                    "warmup_steps": 1,
                    "val_every": 1,
                    "val_steps": 1,
                    "save_every": 2,
                    "log_every": 1,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def test_backfill_inventory_reports_missing_and_existing_artifacts(tmp_path):
    root = tmp_path
    experiment_id = "E999_test"
    config_dir = root / "configs" / "experiments" / experiment_id
    run_root = root / "runs" / "experiments" / experiment_id
    report_dir = root / "reports" / "experiments" / experiment_id
    report_dir.mkdir(parents=True)
    write_config(
        config_dir / "standard.yaml",
        run_name="standard",
        attention_type="standard",
        out_dir=f"runs/experiments/{experiment_id}/standard",
    )
    write_config(
        config_dir / "candidate.yaml",
        run_name="candidate",
        attention_type="operator_valued_attention",
        out_dir=f"runs/experiments/{experiment_id}/candidate",
    )
    ckpt_dir = run_root / "candidate" / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    torch.save({"model": {}, "config": {}}, ckpt_dir / "ckpt_last.pt")
    eval_dir = run_root / "candidate" / "evals"
    eval_dir.mkdir()
    (eval_dir / "attention_diagnostics.jsonl").write_text('{"attention_type":"operator_valued_attention"}\n')
    (eval_dir / "run_summary.json").write_text('{"final_val_loss": 1.0}\n')

    inventory = build_experiment_inventory(
        experiment_id=experiment_id,
        repo_root=root,
        config_dir=config_dir,
        report_dir=report_dir,
        run_dir=run_root,
    )

    rows = {row["run_name"]: row for row in inventory["candidates"]}
    assert rows["candidate"]["checkpoint_status"] == "available"
    assert rows["candidate"]["evidence_level"] == "checkpoint_recompute"
    assert rows["standard"]["checkpoint_status"] == "checkpoint_unavailable"
    assert rows["standard"]["posthoc_probe_status"] == "checkpoint_unavailable"
    assert "operator_probs[layer]" in rows["candidate"]["supported_hook_sites"]


def test_backfill_outputs_validate_json_and_mark_missing_reasons(tmp_path):
    root = tmp_path
    experiment_id = "E999_test"
    config_dir = root / "configs" / "experiments" / experiment_id
    run_root = root / "runs" / "experiments" / experiment_id
    report_dir = root / "reports" / "experiments" / experiment_id
    write_config(
        config_dir / "candidate.yaml",
        run_name="candidate",
        attention_type="scope_gated_qkv",
        out_dir=f"runs/experiments/{experiment_id}/candidate",
    )
    before = sorted(report_dir.glob("**/*")) if report_dir.exists() else []

    inventory = build_experiment_inventory(
        experiment_id=experiment_id,
        repo_root=root,
        config_dir=config_dir,
        report_dir=report_dir,
        run_dir=run_root,
    )
    out_dir = root / "reports" / "mechanisms" / "backfill" / experiment_id
    write_backfill_outputs(inventory, out_dir)

    loaded = json.loads((out_dir / "inventory.json").read_text(encoding="utf-8"))
    assert loaded["schema_version"] == 1
    assert "candidate" in (out_dir / "inventory.md").read_text(encoding="utf-8")
    assert "checkpoint_unavailable" in (out_dir / "missing_artifacts.md").read_text(encoding="utf-8")
    after = sorted(report_dir.glob("**/*")) if report_dir.exists() else []
    assert before == after


def test_backfill_does_not_attach_prefixed_screen_artifacts_to_base_config(tmp_path):
    root = tmp_path
    experiment_id = "E999_test"
    config_dir = root / "configs" / "experiments" / experiment_id
    run_root = root / "runs" / "experiments" / experiment_id
    report_dir = root / "reports" / "experiments" / experiment_id
    write_config(
        config_dir / "candidate.yaml",
        run_name="candidate",
        attention_type="operator_valued_attention",
        out_dir=f"runs/experiments/{experiment_id}/candidate",
    )
    screen_dir = root / "runs" / "screen" / "candidate_rung020_abc123"
    screen_dir.mkdir(parents=True)
    write_config(
        screen_dir / "config.yaml",
        run_name="candidate_rung020",
        attention_type="operator_valued_attention",
        out_dir=str(screen_dir.relative_to(root)),
    )
    (screen_dir / "checkpoints").mkdir()
    torch.save({"model": {}, "config": {}}, screen_dir / "checkpoints" / "ckpt_last.pt")

    inventory = build_experiment_inventory(
        experiment_id=experiment_id,
        repo_root=root,
        config_dir=config_dir,
        report_dir=report_dir,
        run_dir=run_root,
    )

    row = inventory["candidates"][0]
    assert row["run_name"] == "candidate"
    assert row["checkpoint_status"] == "checkpoint_unavailable"
