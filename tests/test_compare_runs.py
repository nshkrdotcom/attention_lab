from __future__ import annotations

import json

import pytest
import yaml

from attention_lab.training.compare_runs import compare_runs, compare_runs_for_experiment


def _write_summary(run_dir, run_name: str, loss: float) -> None:
    evals_dir = run_dir / "evals"
    evals_dir.mkdir(parents=True)
    summary = {
        "run_dir": str(run_dir),
        "run_name": run_name,
        "max_step": 2,
        "train_event_count": 2,
        "val_event_count": 2,
        "initial_val_loss": 2.0,
        "final_val_loss": loss,
        "best_val_loss": loss,
        "initial_val_perplexity": 7.0,
        "final_val_perplexity": 3.0,
        "median_tokens_per_sec": 10.0,
        "peak_vram_mb": 5.0,
        "checkpoint_count": 1,
    }
    (evals_dir / "run_summary.json").write_text(json.dumps(summary), encoding="utf-8")


def _write_config(run_dir, config: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")


def test_compare_runs_reads_summary_contract(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_summary(baseline, "baseline", 1.0)
    _write_summary(candidate, "candidate", 0.9)

    rows = compare_runs(baseline, candidate)

    assert [row["role"] for row in rows] == ["baseline", "candidate"]
    assert rows[0]["run_name"] == "baseline"
    assert rows[1]["final_val_loss"] == 0.9
    assert rows[1]["peak_vram_allocated_mb"] == 5.0


def test_compare_runs_includes_multi_qkv_mechanism_and_destructive_fields(tmp_path, tiny_config):
    data_root = tmp_path / "data"
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_summary(baseline, "baseline", 1.0)
    _write_summary(candidate, "candidate", 0.9)
    baseline_config = tiny_config(tmp_path, data_root)
    candidate_config = tiny_config(tmp_path, data_root)
    candidate_config["run"]["name"] = "candidate"
    candidate_config["run"]["out_dir"] = str(candidate)
    candidate_config["model"].update(
        {
            "attention_type": "multi_qkv_static_3track_global",
            "qkv_track_count": 3,
            "qkv_global_bank": True,
            "qkv_route_formula": "layer_mod",
            "n_layer": 3,
            "n_head": 2,
        }
    )
    _write_config(baseline, baseline_config)
    _write_config(candidate, candidate_config)
    (candidate / "evals" / "hellaswag.json").write_text(
        json.dumps({"accuracy_norm": 0.42, "num_total": 100}),
        encoding="utf-8",
    )
    rows = [
        {
            "attention_type": "multi_qkv_static_3track_global",
            "route_formula": "layer_idx % track_count",
            "uses_global_bank": True,
            "track_count": 3,
            "layer_idx": layer_idx,
            "layer": layer_idx,
            "step": 3000,
            "last_forward_step": 3000,
            "schedule_mode": "train",
            "active_track_index": layer_idx,
            "active_track_counts": {
                "0": 8 if layer_idx == 0 else 0,
                "1": 8 if layer_idx == 1 else 0,
                "2": 8 if layer_idx == 2 else 0,
            },
            "track_gradient_norm": 0.1,
            "per_track_gradient_norm": {
                "0": 0.1 if layer_idx == 0 else 0.0,
                "1": 0.1 if layer_idx == 1 else 0.0,
                "2": 0.1 if layer_idx == 2 else 0.0,
            },
            "per_track_qkv_weight_norm": {"0": 1.0, "1": 1.0, "2": 1.0},
            "position_routing_enabled": False,
            "eval_freeze_mode": False,
        }
        for layer_idx in range(3)
    ]
    (candidate / "evals" / "attention_diagnostics.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    (candidate / "evals" / "qkv_track_destructive_test.json").write_text(
        json.dumps({"destructive_test_passed": True, "perturbations": [{"name": "rotate_tracks"}]}),
        encoding="utf-8",
    )

    comparison_rows = compare_runs(baseline, candidate)
    candidate_row = comparison_rows[1]

    assert candidate_row["attention_type"] == "multi_qkv_static_3track_global"
    assert candidate_row["hellaswag_acc"] == 0.42
    assert candidate_row["parameters_including_positional"] is not None
    assert candidate_row["global_qkv_bank_parameters"] > 0
    assert candidate_row["mechanism_check_passed"] is True
    assert candidate_row["destructive_test_passed"] is True
    assert candidate_row["evidence_level"] == "full_run_verified"


def test_compare_runs_for_experiment_adds_metadata_and_derived_fields(tmp_path, monkeypatch):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "runs" / "experiments" / "E001" / "candidate"
    _write_summary(baseline, "baseline", 1.0)
    _write_summary(candidate, "candidate", 0.9)
    monkeypatch.setattr(
        "attention_lab.training.compare_runs.get_experiment",
        lambda experiment_id: {"id": experiment_id, "run_dir": str(tmp_path / "runs" / "experiments" / "E001")},
    )

    result = compare_runs_for_experiment("E001", baseline, candidate)

    assert result["experiment_id"] == "E001"
    assert result["candidate"]["run_name"] == "candidate"
    assert result["derived"]["delta_final_val_loss"] == pytest.approx(-0.1)
    assert result["derived"]["tokens_per_sec_ratio"] == 1.0
