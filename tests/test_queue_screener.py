from __future__ import annotations

import json

from attention_lab.queue.screener import (
    ScreenVerdict,
    classify_screen_result,
    mechanism_active_from_diagnostics,
    screen_config_with_overrides,
)


def test_screen_config_overrides_training_budget(tiny_config, tmp_path):
    config = tiny_config(tmp_path, tmp_path / "data")
    screened = screen_config_with_overrides(config, tmp_path / "runs" / "screen" / "candidate")
    assert screened["run"]["out_dir"].endswith("runs/screen/candidate")
    assert screened["train"]["max_steps"] == 150
    assert screened["train"]["val_every"] == 50
    assert screened["train"]["save_every"] == 150
    assert "diagnostics" not in screened


def test_screen_config_forces_diagnostics_during_nonstandard_screen(tiny_config, tmp_path):
    config = tiny_config(tmp_path, tmp_path / "data")
    config["model"]["attention_type"] = "cp_trilinear"
    config["model"]["cp_rank"] = 8
    config["model"]["cp_lambda_init"] = 0.0
    config["model"]["cp_lambda_trainable"] = True
    config["model"]["cp_lambda_fixed"] = False
    config["diagnostics"] = {"attention_diagnostics_every": 250}
    screened = screen_config_with_overrides(config, tmp_path / "runs" / "screen" / "candidate")
    assert screened["diagnostics"]["attention_diagnostics_every"] == 50

    config["diagnostics"] = {"attention_diagnostics_every": 25}
    screened = screen_config_with_overrides(config, tmp_path / "runs" / "screen" / "candidate")
    assert screened["diagnostics"]["attention_diagnostics_every"] == 25


def test_screen_config_adds_diagnostics_for_nonstandard_without_diagnostics(tiny_config, tmp_path):
    config = tiny_config(tmp_path, tmp_path / "data")
    config["model"]["attention_type"] = "cp_bilinear"
    config["model"]["cp_rank"] = 8
    config["model"]["cp_lambda_init"] = 0.0
    config["model"]["cp_lambda_trainable"] = True
    config["model"]["cp_lambda_fixed"] = False
    screened = screen_config_with_overrides(config, tmp_path / "runs" / "screen" / "candidate")
    assert screened["diagnostics"]["attention_diagnostics_every"] == 50


def test_screen_classifier_kill_criteria():
    metrics = [
        {"event": "val", "step": 10, "val_loss": 10.0},
        {"event": "val", "step": 150, "val_loss": 9.8},
    ]
    assert classify_screen_result(returncode=1, stderr="import error", metrics=[], attention_type="standard").failure_class == "COMPILE_ERROR"
    assert classify_screen_result(returncode=1, stderr="CUDA out of memory", metrics=[], attention_type="standard").failure_class == "OOM"
    assert classify_screen_result(
        returncode=0,
        stderr="",
        metrics=[{"event": "val", "step": 50, "val_loss": float("nan")}],
        attention_type="standard",
    ).failure_class == "NAN"
    assert classify_screen_result(returncode=0, stderr="", metrics=metrics, attention_type="standard").failure_class == "FLAT_LOSS"
    slow = classify_screen_result(
        returncode=0,
        stderr="",
        metrics=[{"event": "train", "step": 1, "tokens_per_sec": 10}, {"event": "val", "step": 150, "val_loss": 5.0}],
        attention_type="standard",
        baseline_tokens_per_sec=100.0,
    )
    assert slow.failure_class == "SLOW"


def test_screen_classifier_nonzero_and_incomplete_runs_do_not_pass():
    metrics = [
        {"event": "train", "step": 100, "tokens_per_sec": 100.0, "train_loss": 5.0},
        {"event": "val", "step": 100, "val_loss": 5.0},
    ]
    nonzero = classify_screen_result(returncode=1, stderr="late crash", metrics=metrics, attention_type="standard")
    assert nonzero.passed is False
    assert nonzero.failure_class == "UNKNOWN"

    missing_metrics = classify_screen_result(returncode=0, stderr="", metrics=[], attention_type="standard")
    assert missing_metrics.passed is False
    assert missing_metrics.failure_class == "VERIFY_FAIL"

    partial = classify_screen_result(returncode=0, stderr="", metrics=metrics, attention_type="standard")
    assert partial.passed is False
    assert partial.failure_class == "UNKNOWN"


def test_screen_classifier_detects_train_and_val_nan():
    train_nan = classify_screen_result(
        returncode=0,
        stderr="",
        metrics=[{"event": "train", "step": 150, "train_loss": float("nan")}],
        attention_type="standard",
    )
    assert train_nan.failure_class == "NAN"

    val_nan = classify_screen_result(
        returncode=0,
        stderr="",
        metrics=[{"event": "val", "step": 150, "val_loss": float("nan")}],
        attention_type="standard",
    )
    assert val_nan.failure_class == "NAN"


def test_screen_classifier_passes_nonflat_standard_run():
    verdict = classify_screen_result(
        returncode=0,
        stderr="",
        metrics=[
            {"event": "train", "step": 1, "tokens_per_sec": 100.0},
            {"event": "val", "step": 10, "val_loss": 10.0},
            {"event": "val", "step": 150, "val_loss": 9.0},
        ],
        attention_type="standard",
        baseline_tokens_per_sec=100.0,
    )
    assert verdict == ScreenVerdict(passed=True, failure_class=None, mechanism_active=None, step_reached=150)


def test_mechanism_active_from_diagnostics(tmp_path):
    missing = mechanism_active_from_diagnostics("cp_trilinear", tmp_path / "missing.jsonl")
    assert missing is None

    path = tmp_path / "attention_diagnostics.jsonl"
    path.write_text(json.dumps({"cp_gradient_norm": 0.0}) + "\n", encoding="utf-8")
    assert mechanism_active_from_diagnostics("cp_trilinear", path) is False

    path.write_text(json.dumps({"cp_gradient_norm": 1e-3}) + "\n", encoding="utf-8")
    assert mechanism_active_from_diagnostics("cp_trilinear", path) is True
    assert mechanism_active_from_diagnostics("standard", path) is None


def test_cp_mechanism_missing_or_inactive_blocks_full_promotion(tmp_path):
    metrics = [
        {"event": "train", "step": 150, "tokens_per_sec": 100.0, "train_loss": 5.0},
        {"event": "val", "step": 50, "val_loss": 6.0},
        {"event": "val", "step": 150, "val_loss": 5.0},
    ]
    missing = classify_screen_result(
        returncode=0,
        stderr="",
        metrics=metrics,
        attention_type="cp_trilinear",
        diagnostics_path=tmp_path / "missing.jsonl",
        queue_config={},
    )
    assert missing.passed is False
    assert missing.failure_class == "VERIFY_FAIL"
    assert missing.mechanism_active is None

    diagnostics = tmp_path / "attention_diagnostics.jsonl"
    diagnostics.write_text(json.dumps({"cp_gradient_norm": 0.0}) + "\n", encoding="utf-8")
    inactive = classify_screen_result(
        returncode=0,
        stderr="",
        metrics=metrics,
        attention_type="cp_bilinear",
        diagnostics_path=diagnostics,
        queue_config={},
    )
    assert inactive.passed is False
    assert inactive.failure_class == "DEAD_GRAD"
    assert inactive.mechanism_active is False


def test_qkv_mechanism_check_uses_future_diagnostic_fields(tmp_path):
    metrics = [
        {"event": "train", "step": 150, "tokens_per_sec": 100.0, "train_loss": 5.0},
        {"event": "val", "step": 50, "val_loss": 6.0},
        {"event": "val", "step": 150, "val_loss": 5.0},
    ]
    diagnostics = tmp_path / "attention_diagnostics.jsonl"
    diagnostics.write_text(json.dumps({"per_track_gradient_norm": [0.0, 2e-3]}) + "\n", encoding="utf-8")
    verdict = classify_screen_result(
        returncode=0,
        stderr="",
        metrics=metrics,
        attention_type="multi_qkv_static",
        diagnostics_path=diagnostics,
        queue_config={"mechanism_check": "qkv_track_activity"},
    )
    assert verdict.passed is True
    assert verdict.mechanism_active is True

    diagnostics.write_text(json.dumps({"track_gradient_norm": 0.0, "track_output_delta": 0.0}) + "\n", encoding="utf-8")
    inactive = classify_screen_result(
        returncode=0,
        stderr="",
        metrics=metrics,
        attention_type="multi_qkv_static",
        diagnostics_path=diagnostics,
        queue_config={"mechanism_check": "qkv_track_activity"},
    )
    assert inactive.passed is False
    assert inactive.failure_class == "DEAD_GRAD"


def test_nonstandard_screen_with_step_50_diagnostics_can_pass(tmp_path):
    metrics = [
        {"event": "train", "step": 50, "tokens_per_sec": 100.0, "train_loss": 5.5},
        {"event": "val", "step": 50, "val_loss": 6.0},
        {"event": "train", "step": 150, "tokens_per_sec": 100.0, "train_loss": 4.5},
        {"event": "val", "step": 150, "val_loss": 5.0},
    ]
    diagnostics = tmp_path / "attention_diagnostics.jsonl"
    diagnostics.write_text(json.dumps({"step": 50, "cp_gradient_norm": 1e-3}) + "\n", encoding="utf-8")
    verdict = classify_screen_result(
        returncode=0,
        stderr="",
        metrics=metrics,
        attention_type="cp_trilinear",
        diagnostics_path=diagnostics,
        queue_config={"mechanism_check": "cp_gradient_norm"},
    )
    assert verdict.passed is True
    assert verdict.mechanism_active is True
