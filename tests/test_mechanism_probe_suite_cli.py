from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch
import yaml

from attention_lab.mechanisms.hypotheses import validate_hypothesis_doc
from attention_lab.mechanisms.task_schema import load_task_suite, validate_task_suite
from attention_lab.models.gpt import GPT, config_from_dict


def _write_tiny_config(path: Path, *, attention_type: str, name: str) -> None:
    config = {
        "run": {"name": name, "out_dir": str(path.parent / name), "seed": 1},
        "data": {
            "data_root": "data/fineweb_edu_100m",
            "tokenizer": "gpt2",
            "vocab_size": 50304,
            "train_tokens": 100,
            "val_tokens": 20,
        },
        "model": {
            "attention_type": attention_type,
            "block_size": 16,
            "n_layer": 1,
            "n_head": 2,
            "n_embd": 16,
            "dropout": 0.0,
            "bias": False,
        },
        "train": {
            "device": "cpu",
            "dtype": "float32",
            "compile": False,
            "eval_at_start": True,
            "B": 1,
            "T": 16,
            "total_batch_size": 16,
            "max_steps": 1,
            "grad_clip": 1.0,
            "weight_decay": 0.1,
            "learning_rate": 0.001,
            "min_lr": 0.0001,
            "warmup_steps": 1,
            "val_every": 1,
            "val_steps": 1,
            "save_every": 1,
            "log_every": 1,
        },
    }
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def _write_tiny_checkpoint(config_path: Path, checkpoint_path: Path) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    torch.manual_seed(0)
    model = GPT(config_from_dict(config["model"], config["data"]))
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict()}, checkpoint_path)


def _write_task_file(path: Path, *, pairs: int = 6) -> None:
    records = []
    for index in range(pairs):
        records.append(
            {
                "pair_id": f"pair_{index}",
                "template_id": f"template_{index % 3}",
                "family_id": "negation",
                "x_pos": f"The marker is present good {index}.",
                "x_neg": f"The marker is absent bad {index}.",
                "x_para": f"A present marker is visible good {index}.",
                "x_decoy": f"The nearby topic is present neutral {index}.",
                "metadata": {},
            }
        )
    path.write_text(yaml.safe_dump({"schema_version": 1, "metadata": {}, "records": records}), encoding="utf-8")


def test_probe_suite_cli_exploratory_probe_only_real_path(tmp_path):
    candidate_config = tmp_path / "candidate.yaml"
    control_config = tmp_path / "control.yaml"
    candidate_checkpoint = tmp_path / "candidate" / "ckpt_last.pt"
    control_checkpoint = tmp_path / "control" / "ckpt_last.pt"
    task_file = tmp_path / "tasks.yaml"
    output_dir = tmp_path / "suite_out"
    _write_tiny_config(candidate_config, attention_type="differential_qkv_anti_value", name="tiny_candidate")
    _write_tiny_config(control_config, attention_type="standard", name="tiny_control")
    _write_tiny_checkpoint(candidate_config, candidate_checkpoint)
    _write_tiny_checkpoint(control_config, control_checkpoint)
    _write_task_file(task_file)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_mechanism_probe_suite.py",
            "--experiment-id",
            "E003_qkv_architecture_gauntlet",
            "--candidate",
            "differential",
            "--config",
            str(candidate_config),
            "--checkpoint",
            str(candidate_checkpoint),
            "--task-file",
            str(task_file),
            "--output-dir",
            str(output_dir),
            "--exploratory",
            "--probe-only",
            "--sites",
            "branch_delta",
            "--control-mode",
            "matched",
            "--control-config",
            str(control_config),
            "--control-checkpoint",
            str(control_checkpoint),
            "--min-n",
            "2",
            "--bootstrap-samples",
            "10",
            "--fdr-alpha",
            "0.2",
            "--seed",
            "1",
            "--batch-size",
            "4",
        ],
        cwd=Path.cwd(),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    gates = json.loads((output_dir / "claim_gates.json").read_text(encoding="utf-8"))
    summary = (output_dir / "summary.md").read_text(encoding="utf-8")

    assert metrics["mode"]["probe_only"] is True
    assert metrics["control"]["override_used"] is True
    assert metrics["control"]["canonical"] is False
    assert metrics["cells"]
    cell = next(iter(metrics["cells"].values()))
    assert cell["linear_probe_auc"] is not None
    assert cell["shuffled_label_auc"] is not None
    assert "random_site_null_available" in cell["random_site_null"]
    assert "probe_direction_cosine_to_control" in cell["alignment_to_control"]
    assert cell["patching"]["reason"] == "probe-only mode"
    assert any("linear_probe_auc_minus_0_5" in item for item in metrics["fdr_bh"]["tested_cells"])
    assert any("auc_minus_shuffled_auc" in item for item in metrics["fdr_bh"]["tested_cells"])
    assert any("target_vs_decoy_specificity" in item for item in metrics["fdr_bh"]["tested_cells"])
    assert gates["overall_status"] == "exploratory_probe_signal"
    assert "Probe-only mode skipped" in summary


def test_probe_suite_cli_requires_hypothesis_doc_for_confirmatory(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_mechanism_probe_suite.py",
            "--experiment-id",
            "E003_qkv_architecture_gauntlet",
            "--candidate",
            "differential",
            "--checkpoint",
            "missing.pt",
            "--task-file",
            "missing.yaml",
            "--output-dir",
            str(tmp_path / "out"),
        ],
        cwd=Path.cwd(),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "--hypothesis-doc" in result.stderr


def test_invalid_hypothesis_doc_fails_schema_and_convention(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("CLAIM: x\n", encoding="utf-8")

    try:
        validate_hypothesis_doc(bad, repo_root=tmp_path)
    except ValueError as exc:
        assert "docs/mechanisms/hypotheses" in str(exc)
    else:
        raise AssertionError("invalid hypothesis path should fail")

    good_path = tmp_path / "docs" / "mechanisms" / "hypotheses" / "h.yaml"
    good_path.parent.mkdir(parents=True)
    good_path.write_text("CLAIM: x\n", encoding="utf-8")
    try:
        validate_hypothesis_doc(good_path, repo_root=tmp_path)
    except ValueError as exc:
        assert "missing required fields" in str(exc)
    else:
        raise AssertionError("invalid hypothesis schema should fail")


def test_task_suite_validation_blocks_small_hand_authored_confirmatory_suite(tmp_path):
    task_file = tmp_path / "tasks.yaml"
    _write_task_file(task_file, pairs=4)
    suite = load_task_suite(task_file)
    result = validate_task_suite(suite, confirmatory=True, exploratory=False, min_n=50)

    assert not result.valid
    assert not result.deterministic_provenance
    assert not result.confirmatory_floor_met
    assert any("deterministic generator provenance" in error for error in result.errors)
    assert any("below confirmatory floor" in error for error in result.errors)
