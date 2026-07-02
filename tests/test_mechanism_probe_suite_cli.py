from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

from attention_lab.models.gpt import GPT, config_from_dict
from attention_lab.training.checkpointing import save_checkpoint
from attention_lab.training.optim import build_optimizer


def _write_task_file(path: Path, n_pairs: int = 4) -> None:
    records = []
    marks = ["!", ".", "?", ","]
    for idx in range(n_pairs):
        mark = marks[idx % len(marks)]
        records.append(
            {
                "x_pos": f"{mark} !",
                "x_neg": f"{mark} .",
                "x_para": f"! {mark}",
                "x_decoy": f". {mark}",
                "pair_id": f"pair_{idx}",
                "template_id": f"template_{idx // 2}",
                "family_id": "punctuation",
                "metadata": {},
            }
        )
    path.write_text(
        json.dumps(
            {
                "metadata": {
                    "generator_name": "unit_test",
                    "generator_version": "1",
                    "template_set": "punctuation",
                    "filler_set": "punctuation",
                    "generation_seed": 1,
                    "created_at": "2026-07-02",
                },
                "records": records,
            }
        ),
        encoding="utf-8",
    )


def _write_hypothesis_doc(path: Path, *, valid: bool = True) -> None:
    payload = {
        "CLAIM": "Branch delta encodes the punctuation contrast.",
        "KILL_CONDITION": "AUC fails to beat nulls after correction.",
        "MECHANISM_PROOF": "Linear probe and patching gates pass.",
        "NEAREST_BORING_EXPLANATION": "Any same-dimensional site encodes punctuation.",
        "CONTROL_THAT_RULES_IT_OUT": "standard_refactor_control_30m_seed1_rung500",
        "TARGET_SITES": ["branch_delta"],
        "TASK_CONTRASTS": ["x_pos", "x_neg", "x_para", "x_decoy"],
        "PRIMARY_METRIC": "auc_minus_matched_control_auc",
        "STATISTICAL_TEST": "bootstrap_ci_with_fdr_bh",
        "MIN_N": 50,
        "FDR_SCOPE": "all computed cells",
        "EXPECTED_DIRECTION": "positive",
    }
    if not valid:
        payload.pop("KILL_CONDITION")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def _write_tiny_checkpoint(tmp_path: Path, attention_type: str = "differential_qkv_anti_value") -> tuple[Path, Path]:
    config = {
        "run": {"name": "tiny_probe_suite", "out_dir": str(tmp_path / "run"), "seed": 1},
        "data": {"data_root": str(tmp_path / "data"), "tokenizer": "gpt2", "vocab_size": 50304},
        "model": {
            "attention_type": attention_type,
            "block_size": 8,
            "n_layer": 1,
            "n_head": 1,
            "n_embd": 16,
            "dropout": 0.0,
            "bias": False,
        },
        "train": {
            "device": "cpu",
            "dtype": "float32",
            "compile": False,
            "B": 1,
            "T": 8,
            "total_batch_size": 8,
            "max_steps": 1,
            "grad_clip": 1.0,
            "weight_decay": 0.0,
            "learning_rate": 0.001,
            "min_lr": 0.0001,
            "warmup_steps": 1,
            "val_every": 1,
            "val_steps": 1,
            "save_every": 1,
            "log_every": 1,
        },
        "sample": {
            "sample_every": 1,
            "prompt": "!",
            "num_samples": 1,
            "max_new_tokens": 1,
            "top_k": 10,
            "temperature": 1.0,
            "seed": 1,
        },
    }
    config_path = tmp_path / "candidate.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    model = GPT(config_from_dict(config["model"], config["data"]))
    optimizer = build_optimizer(model, weight_decay=0.0, learning_rate=0.001, device_type="cpu", master_process=False)
    checkpoint = save_checkpoint(tmp_path / "checkpoint", model, optimizer, config, step=0)
    return config_path, checkpoint


def test_suite_cli_requires_hypothesis_doc_unless_exploratory(tmp_path, repo_root):
    task_file = tmp_path / "tasks.json"
    _write_task_file(task_file)
    config_path, checkpoint = _write_tiny_checkpoint(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "run_mechanism_probe_suite.py"),
            "--experiment-id",
            "E003_qkv_architecture_gauntlet",
            "--candidate",
            "differential",
            "--config",
            str(config_path),
            "--checkpoint",
            str(checkpoint),
            "--task-file",
            str(task_file),
            "--output-dir",
            str(tmp_path / "out"),
            "--control-mode",
            "none",
            "--min-n",
            "2",
            "--bootstrap-samples",
            "25",
            "--seed",
            "1",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "--hypothesis-doc is required" in result.stderr


def test_suite_cli_rejects_invalid_hypothesis_doc(tmp_path, repo_root):
    task_file = tmp_path / "tasks.json"
    hypothesis_doc = tmp_path / "docs" / "hypothesis.yaml"
    _write_task_file(task_file)
    _write_hypothesis_doc(hypothesis_doc, valid=False)
    config_path, checkpoint = _write_tiny_checkpoint(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "run_mechanism_probe_suite.py"),
            "--experiment-id",
            "E003_qkv_architecture_gauntlet",
            "--candidate",
            "differential",
            "--config",
            str(config_path),
            "--checkpoint",
            str(checkpoint),
            "--task-file",
            str(task_file),
            "--hypothesis-doc",
            str(hypothesis_doc),
            "--output-dir",
            str(tmp_path / "out"),
            "--control-mode",
            "none",
            "--min-n",
            "2",
            "--bootstrap-samples",
            "25",
            "--seed",
            "1",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "missing required hypothesis fields" in result.stderr


def test_suite_cli_exploratory_probe_only_writes_real_artifacts(tmp_path, repo_root):
    task_file = tmp_path / "tasks.json"
    output_dir = tmp_path / "out"
    _write_task_file(task_file)
    config_path, checkpoint = _write_tiny_checkpoint(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "run_mechanism_probe_suite.py"),
            "--experiment-id",
            "E003_qkv_architecture_gauntlet",
            "--candidate",
            "differential",
            "--config",
            str(config_path),
            "--checkpoint",
            str(checkpoint),
            "--task-file",
            str(task_file),
            "--output-dir",
            str(output_dir),
            "--exploratory",
            "--probe-only",
            "--sites",
            "branch_delta",
            "--control-mode",
            "none",
            "--min-n",
            "2",
            "--bootstrap-samples",
            "25",
            "--fdr-alpha",
            "0.05",
            "--seed",
            "1",
            "--device",
            "cpu",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    gates = json.loads((output_dir / "claim_gates.json").read_text(encoding="utf-8"))
    summary = (output_dir / "summary.md").read_text(encoding="utf-8")

    assert metrics["probe_only"] is True
    assert metrics["exploratory"] is True
    assert metrics["patching_restoration"] == {"skipped": True, "reason": "probe-only mode"}
    assert "linear_probe_auc" in metrics["probe_metrics"]["branch_delta[0]"]["punctuation"]
    assert gates["status"] != "candidate_mechanism_evidence"
    assert "single-seed" in summary
