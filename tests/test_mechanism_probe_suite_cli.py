from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch
import yaml

from attention_lab.mechanisms.hypotheses import validate_hypothesis_doc
from attention_lab.mechanisms.task_generation import gpt2_single_token_id
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
            "block_size": 64,
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


def _write_task_file(
    path: Path,
    *,
    pairs: int = 6,
    provenance: bool = False,
    restoration_tokens: bool = False,
    families: tuple[str, ...] = ("negation",),
) -> None:
    records = []
    target_id = gpt2_single_token_id(" true")
    foil_id = gpt2_single_token_id(" false")
    template_mod = min(max(pairs, 2), 8)
    for family in families:
        for index in range(pairs):
            metadata = {}
            if restoration_tokens:
                metadata.update(
                    {
                        "target_token_text": " true",
                        "foil_token_text": " false",
                        "target_token_id": target_id,
                        "foil_token_id": foil_id,
                    }
                )
            records.append(
                    {
                        "pair_id": f"{family}_pair_{index}",
                        "template_id": f"{family}_template_{index % template_mod}",
                    "family_id": family,
                    "x_pos": f"Sentence: The analyst did not approve report {index}. Answer:",
                    "x_neg": f"Sentence: The analyst approved report {index}. Answer:",
                    "x_para": f"Sentence: The analyst never approved report {index}. Answer:",
                    "x_decoy": f"Sentence: The analyst carefully approved report {index}. Answer:",
                    "metadata": metadata,
                }
            )
    suite_metadata = {}
    if provenance:
        suite_metadata = {
            "generator_name": "test_generator",
            "generator_version": "1",
            "template_set": "test_templates",
            "filler_set": "test_fillers",
            "generation_seed": 1,
            "created_at": "1970-01-01T00:00:00Z",
        }
    path.write_text(
        yaml.safe_dump({"schema_version": 1, "metadata": suite_metadata, "records": records}, sort_keys=False),
        encoding="utf-8",
    )


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


def test_confirmatory_small_suite_fails_before_checkpoint_loading(tmp_path):
    task_file = tmp_path / "small.yaml"
    _write_task_file(task_file, pairs=4, provenance=True, restoration_tokens=True)

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
            str(task_file),
            "--hypothesis-doc",
            "docs/mechanisms/hypotheses/E003_differential_negation_tier1.yaml",
            "--output-dir",
            str(tmp_path / "out"),
            "--min-n",
            "50",
        ],
        cwd=Path.cwd(),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "confirmatory task suite invalid" in result.stderr
    assert "candidate checkpoint does not exist" not in result.stderr


def test_confirmatory_missing_provenance_fails_before_checkpoint_loading(tmp_path):
    task_file = tmp_path / "missing_provenance.yaml"
    _write_task_file(task_file, pairs=50, provenance=False, restoration_tokens=True)

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
            str(task_file),
            "--hypothesis-doc",
            "docs/mechanisms/hypotheses/E003_differential_negation_tier1.yaml",
            "--output-dir",
            str(tmp_path / "out"),
            "--min-n",
            "50",
        ],
        cwd=Path.cwd(),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "deterministic generator provenance" in result.stderr
    assert "candidate checkpoint does not exist" not in result.stderr


def test_confirmatory_missing_restoration_tokens_fails_before_checkpoint_loading(tmp_path):
    task_file = tmp_path / "missing_tokens.yaml"
    _write_task_file(task_file, pairs=50, provenance=True, restoration_tokens=False)

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
            str(task_file),
            "--hypothesis-doc",
            "docs/mechanisms/hypotheses/E003_differential_negation_tier1.yaml",
            "--output-dir",
            str(tmp_path / "out"),
            "--min-n",
            "50",
        ],
        cwd=Path.cwd(),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "lacks restoration token metadata" in result.stderr
    assert "candidate checkpoint does not exist" not in result.stderr


def test_exploratory_full_run_is_capped_and_labels_patching_limitation(tmp_path):
    candidate_config = tmp_path / "candidate.yaml"
    control_config = tmp_path / "control.yaml"
    candidate_checkpoint = tmp_path / "candidate" / "ckpt_last.pt"
    control_checkpoint = tmp_path / "control" / "ckpt_last.pt"
    task_file = tmp_path / "tasks.yaml"
    output_dir = tmp_path / "suite_full_exploratory"
    _write_tiny_config(candidate_config, attention_type="differential_qkv_anti_value", name="tiny_candidate")
    _write_tiny_config(control_config, attention_type="standard", name="tiny_control")
    _write_tiny_checkpoint(candidate_config, candidate_checkpoint)
    _write_tiny_checkpoint(control_config, control_checkpoint)
    _write_task_file(task_file, pairs=6)

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
            "5",
            "--fdr-alpha",
            "0.2",
            "--seed",
            "2",
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
    assert metrics["mode"] == {"exploratory": True, "probe_only": False}
    assert gates["overall_status"] == "exploratory_probe_signal"
    cell = next(iter(metrics["cells"].values()))
    assert cell["patching"]["valid"] is False
    assert "target_token_id/foil_token_id" in cell["patching"]["reason"]
    assert "Exploratory mode capped" in summary


def test_confirmatory_full_tiny_run_writes_real_artifacts_and_caps_noncanonical_control(tmp_path):
    candidate_config = tmp_path / "candidate.yaml"
    control_config = tmp_path / "control.yaml"
    candidate_checkpoint = tmp_path / "candidate" / "ckpt_last.pt"
    control_checkpoint = tmp_path / "control" / "ckpt_last.pt"
    task_file = tmp_path / "generated_tasks.yaml"
    output_dir = tmp_path / "suite_full_confirmatory"
    _write_tiny_config(candidate_config, attention_type="differential_qkv_anti_value", name="tiny_candidate")
    _write_tiny_config(control_config, attention_type="standard", name="tiny_control")
    _write_tiny_checkpoint(candidate_config, candidate_checkpoint)
    _write_tiny_checkpoint(control_config, control_checkpoint)

    generated = subprocess.run(
        [
            sys.executable,
            "scripts/generate_tier1_mechanism_tasks.py",
            "--output",
            str(task_file),
            "--candidate",
            "e003_differential",
            "--pairs-per-family",
            "50",
            "--seed",
            "3",
        ],
        cwd=Path.cwd(),
        check=False,
        capture_output=True,
        text=True,
    )
    assert generated.returncode == 0, generated.stderr

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
            "--hypothesis-doc",
            "docs/mechanisms/hypotheses/E003_differential_negation_tier1.yaml",
            "--output-dir",
            str(output_dir),
            "--sites",
            "branch_delta",
            "--control-mode",
            "matched",
            "--control-config",
            str(control_config),
            "--control-checkpoint",
            str(control_checkpoint),
            "--force-noncanonical-control",
            "--min-n",
            "50",
            "--bootstrap-samples",
            "5",
            "--fdr-alpha",
            "0.2",
            "--seed",
            "3",
            "--batch-size",
            "8",
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
    assert metrics["mode"] == {"exploratory": False, "probe_only": False}
    assert metrics["hypothesis"]["path"].endswith("E003_differential_negation_tier1.yaml")
    assert metrics["control"]["override_used"] is True
    assert metrics["control"]["canonical"] is False
    assert metrics["control"]["force_noncanonical"] is True
    assert metrics["task_suite"]["restoration_token_metadata_valid"] is True
    assert metrics["task_suite"]["confirmatory_floor_met"] is True
    assert metrics["cells"]
    cell = next(iter(metrics["cells"].values()))
    assert cell["linear_probe_auc"] is not None
    assert "patching" in cell
    assert "mediation_fraction" in cell
    assert "fdr_bh" in metrics
    assert any("component_patch_restoration" in item for item in metrics["fdr_bh"]["tested_cells"]) or not cell[
        "patching"
    ]["valid"]
    assert gates["overall_status"] != "candidate_mechanism_evidence"
    assert any("control pairing" in blocker for gate in gates["cells"].values() for blocker in gate["blockers"])
    assert "mode: `confirmatory`" in summary
    assert "canonical_control: `False`" in summary
    assert "FDR-BH" in summary


def test_fdr_scope_includes_all_computed_site_family_metric_cells(tmp_path):
    candidate_config = tmp_path / "candidate.yaml"
    control_config = tmp_path / "control.yaml"
    candidate_checkpoint = tmp_path / "candidate" / "ckpt_last.pt"
    control_checkpoint = tmp_path / "control" / "ckpt_last.pt"
    task_file = tmp_path / "tasks.yaml"
    output_dir = tmp_path / "suite_fdr_scope"
    _write_tiny_config(candidate_config, attention_type="differential_qkv_anti_value", name="tiny_candidate")
    _write_tiny_config(control_config, attention_type="standard", name="tiny_control")
    _write_tiny_checkpoint(candidate_config, candidate_checkpoint)
    _write_tiny_checkpoint(control_config, control_checkpoint)
    _write_task_file(task_file, pairs=6, families=("negation", "adverb_control"))

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
            "branch_delta,pos_out",
            "--control-mode",
            "matched",
            "--control-config",
            str(control_config),
            "--control-checkpoint",
            str(control_checkpoint),
            "--min-n",
            "2",
            "--bootstrap-samples",
            "5",
            "--fdr-alpha",
            "0.2",
            "--seed",
            "4",
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
    tested = set(metrics["fdr_bh"]["tested_cells"])
    for site in ("branch_delta[0]", "pos_out[0]"):
        for family in ("negation", "adverb_control"):
            prefix = f"{site}|family={family}|metric="
            assert prefix + "linear_probe_auc_minus_0_5" in tested
            assert prefix + "auc_minus_shuffled_auc" in tested
            assert prefix + "target_vs_decoy_specificity" in tested
    assert "every computed (site x layer x task_family x metric) cell" in metrics["fdr_bh"]["comparison_family"]


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


def test_task_suite_validation_rejects_multitoken_restoration_text(tmp_path):
    task_file = tmp_path / "tasks.yaml"
    _write_task_file(task_file, pairs=50, provenance=True, restoration_tokens=True)
    payload = yaml.safe_load(task_file.read_text(encoding="utf-8"))
    payload["records"][0]["metadata"]["target_token_text"] = " definitely true"
    task_file.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    suite = load_task_suite(task_file)
    result = validate_task_suite(
        suite,
        confirmatory=True,
        exploratory=False,
        min_n=50,
        require_restoration_tokens=True,
    )

    assert not result.valid
    assert any("not a single GPT-2 token" in error for error in result.errors)
