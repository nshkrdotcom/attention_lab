from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from attention_lab.queue.cli import main as queue_main
from attention_lab.queue.ledger import QueueLedger
from attention_lab.queue.promotion import (
    REQUIRED_PROMOTION_REPORT_FIELDS,
    approval_blockers,
    build_promotion_report,
    build_screen_destructive_test_command,
    validate_promotion_report,
    write_promotion_report,
)
from attention_lab.queue.runner import CommandResult
from attention_lab.queue.screener import run_screen


def test_standard_screen_descending_loss_promotes(tmp_path, tiny_config):
    config, row, screen_dir = _screen_case(tmp_path, tiny_config)
    _write_metrics(screen_dir / "metrics.jsonl")
    report = build_promotion_report(row=row, config=config, screen_run_dir=screen_dir, repo_root=tmp_path)

    assert validate_promotion_report(report) == []
    assert report["promotion_recommendation"] == "promote"
    assert report["promotion_blockers"] == []
    assert report["checkpoint_present"] is True
    assert report["final_screen_loss"] == 5.0
    assert report["diagnostics_present"] is False
    assert approval_blockers(report) == []


def test_cp_screen_missing_diagnostics_blocks_promotion(tmp_path, tiny_config):
    config, row, screen_dir = _screen_case(tmp_path, tiny_config, attention_type="cp_trilinear")
    _write_metrics(screen_dir / "metrics.jsonl")
    report = build_promotion_report(row=row, config=config, screen_run_dir=screen_dir, repo_root=tmp_path)

    assert report["promotion_recommendation"] == "needs_investigation"
    assert "non-standard attention is missing required diagnostics" in report["promotion_blockers"]
    assert approval_blockers(report)


def test_allow_missing_diagnostics_needs_investigation_not_promote(tmp_path, tiny_config):
    config, row, screen_dir = _screen_case(tmp_path, tiny_config, attention_type="cp_trilinear")
    config["queue"]["allow_missing_diagnostics"] = True
    _write_metrics(screen_dir / "metrics.jsonl")
    report = build_promotion_report(row=row, config=config, screen_run_dir=screen_dir, repo_root=tmp_path)

    assert report["promotion_recommendation"] == "needs_investigation"
    assert "queue.allow_missing_diagnostics" in " ".join(report["promotion_blockers"])
    assert "non-standard attention lacks non-degenerate diagnostics" in approval_blockers(report)


def test_cp_screen_positive_gradient_allows_promotion(tmp_path, tiny_config):
    config, row, screen_dir = _screen_case(tmp_path, tiny_config, attention_type="cp_bilinear")
    _write_metrics(screen_dir / "metrics.jsonl")
    _write_jsonl(screen_dir / "evals" / "attention_diagnostics.jsonl", [{"step": 50, "cp_gradient_norm": 1e-3}])
    report = build_promotion_report(row=row, config=config, screen_run_dir=screen_dir, repo_root=tmp_path)

    assert report["promotion_recommendation"] == "promote"
    assert report["diagnostics_non_degenerate"] is True
    assert report["mechanism_activity_summary"]["cp_gradient_norm_exceeded_threshold"] is True


def test_cp_screen_zero_gradient_is_killed(tmp_path, tiny_config):
    config, row, screen_dir = _screen_case(tmp_path, tiny_config, attention_type="cp_trilinear")
    _write_metrics(screen_dir / "metrics.jsonl")
    _write_jsonl(screen_dir / "evals" / "attention_diagnostics.jsonl", [{"step": 50, "cp_gradient_norm": 0.0}])
    report = build_promotion_report(row=row, config=config, screen_run_dir=screen_dir, repo_root=tmp_path)

    assert report["promotion_recommendation"] == "kill"
    assert "mechanism diagnostics were degenerate" in report["promotion_blockers"]


def test_multi_qkv_screen_valid_diagnostics_marks_mechanism_active(tmp_path, tiny_config):
    config, row, screen_dir = _screen_case(
        tmp_path,
        tiny_config,
        attention_type="multi_qkv_static_3track_global",
    )
    _write_metrics(screen_dir / "metrics.jsonl")
    _write_jsonl(
        screen_dir / "evals" / "attention_diagnostics.jsonl",
        [_qkv_row(layer_idx=layer_idx, active_track=layer_idx) for layer_idx in range(3)],
    )
    _write_destructive(screen_dir / "evals" / "qkv_track_destructive_test.json")
    report = build_promotion_report(row=row, config=config, screen_run_dir=screen_dir, repo_root=tmp_path)

    assert report["mechanism_active"] is True
    assert report["diagnostics_non_degenerate"] is True
    assert report["destructive_test_present"] is True
    assert report["promotion_recommendation"] == "promote"


def test_multi_qkv_screen_missing_diagnostics_blocks(tmp_path, tiny_config):
    config, row, screen_dir = _screen_case(
        tmp_path,
        tiny_config,
        attention_type="multi_qkv_static_3track_global",
    )
    _write_metrics(screen_dir / "metrics.jsonl")
    report = build_promotion_report(row=row, config=config, screen_run_dir=screen_dir, repo_root=tmp_path)

    assert report["promotion_recommendation"] == "needs_investigation"
    assert "non-standard attention is missing required diagnostics" in report["promotion_blockers"]


def test_multi_qkv_train_rotation_requires_step_routing_and_eval_freeze(tmp_path, tiny_config):
    config, row, screen_dir = _screen_case(
        tmp_path,
        tiny_config,
        attention_type="multi_qkv_train_rotation_3track_global",
    )
    _write_metrics(screen_dir / "metrics.jsonl")
    _write_destructive(screen_dir / "evals" / "qkv_track_destructive_test.json")
    _write_jsonl(
        screen_dir / "evals" / "attention_diagnostics.jsonl",
        [
            _qkv_train_rotation_row(layer_idx=0, active_track=0, last_forward_step=0, schedule_mode="train"),
            _qkv_train_rotation_row(layer_idx=0, active_track=0, last_forward_step=0, schedule_mode="eval"),
        ],
    )
    single_step = build_promotion_report(row=row, config=config, screen_run_dir=screen_dir, repo_root=tmp_path)
    assert single_step["promotion_recommendation"] == "kill"
    assert "multiple training steps" in single_step["mechanism_activity_summary"]["check_note"]

    _write_jsonl(
        screen_dir / "evals" / "attention_diagnostics.jsonl",
        [
            _qkv_train_rotation_row(layer_idx=0, active_track=0, last_forward_step=0, schedule_mode="train"),
            _qkv_train_rotation_row(layer_idx=0, active_track=1, last_forward_step=1, schedule_mode="train"),
        ],
    )
    no_eval_freeze = build_promotion_report(row=row, config=config, screen_run_dir=screen_dir, repo_root=tmp_path)
    assert no_eval_freeze["promotion_recommendation"] == "kill"
    assert "eval/generate freeze evidence" in no_eval_freeze["mechanism_activity_summary"]["check_note"]

    _write_jsonl(
        screen_dir / "evals" / "attention_diagnostics.jsonl",
        [
            _qkv_train_rotation_row(layer_idx=0, active_track=0, last_forward_step=0, schedule_mode="train"),
            _qkv_train_rotation_row(layer_idx=0, active_track=1, last_forward_step=1, schedule_mode="train"),
            _qkv_train_rotation_row(layer_idx=0, active_track=0, last_forward_step=1, schedule_mode="eval"),
        ],
    )
    valid = build_promotion_report(row=row, config=config, screen_run_dir=screen_dir, repo_root=tmp_path)
    assert valid["promotion_recommendation"] == "promote"


def test_multi_qkv_position_rotation_rejects_scalar_routing(tmp_path, tiny_config):
    config, row, screen_dir = _screen_case(
        tmp_path,
        tiny_config,
        attention_type="multi_qkv_position_rotation_3track_global",
    )
    _write_metrics(screen_dir / "metrics.jsonl")
    _write_destructive(screen_dir / "evals" / "qkv_track_destructive_test.json")
    _write_jsonl(
        screen_dir / "evals" / "attention_diagnostics.jsonl",
        [_qkv_position_rotation_row(layer_idx=0, active_track=0)],
    )
    scalar = build_promotion_report(row=row, config=config, screen_run_dir=screen_dir, repo_root=tmp_path)

    assert scalar["promotion_recommendation"] == "kill"
    assert "scalar active_track_index" in scalar["mechanism_activity_summary"]["check_note"]

    _write_jsonl(
        screen_dir / "evals" / "attention_diagnostics.jsonl",
        [_qkv_position_rotation_row(layer_idx=0, active_track=None, counts={"0": 3, "1": 3, "2": 2})],
    )
    valid = build_promotion_report(row=row, config=config, screen_run_dir=screen_dir, repo_root=tmp_path)
    assert valid["promotion_recommendation"] == "promote"


def test_screen_missing_checkpoint_is_killed(tmp_path, tiny_config):
    config, row, screen_dir = _screen_case(tmp_path, tiny_config, checkpoint=False)
    _write_metrics(screen_dir / "metrics.jsonl")
    report = build_promotion_report(row=row, config=config, screen_run_dir=screen_dir, repo_root=tmp_path)

    assert report["checkpoint_present"] is False
    assert report["promotion_recommendation"] == "kill"
    assert "final screen checkpoint is missing" in report["promotion_blockers"]
    assert approval_blockers(report)


def test_nan_flat_and_slow_screens_do_not_promote(tmp_path, tiny_config):
    config, row, screen_dir = _screen_case(tmp_path, tiny_config)
    _write_metrics(screen_dir / "metrics.jsonl", val_losses=(6.0, float("nan")))
    nan_report = build_promotion_report(row=row, config=config, screen_run_dir=screen_dir, repo_root=tmp_path)
    assert nan_report["promotion_recommendation"] == "kill"

    config, row, screen_dir = _screen_case(tmp_path, tiny_config, name="flat")
    _write_metrics(screen_dir / "metrics.jsonl", val_losses=(6.0, 5.9))
    flat_report = build_promotion_report(row=row, config=config, screen_run_dir=screen_dir, repo_root=tmp_path)
    assert flat_report["promotion_recommendation"] == "kill"
    assert "loss did not descend during the screen" in flat_report["promotion_blockers"]

    config, row, screen_dir = _screen_case(tmp_path, tiny_config, name="slow")
    _write_metrics(screen_dir / "metrics.jsonl")
    slow_report = build_promotion_report(
        row=row,
        config=config,
        screen_run_dir=screen_dir,
        repo_root=tmp_path,
        failure_class="SLOW",
    )
    assert slow_report["promotion_recommendation"] == "needs_investigation"


def test_destructive_test_summary_uses_actual_fields(tmp_path, tiny_config):
    config, row, screen_dir = _screen_case(
        tmp_path,
        tiny_config,
        attention_type="multi_qkv_static_3track_global",
    )
    _write_metrics(screen_dir / "metrics.jsonl")
    _write_jsonl(
        screen_dir / "evals" / "attention_diagnostics.jsonl",
        [_qkv_row(layer_idx=layer_idx, active_track=layer_idx) for layer_idx in range(3)],
    )
    _write_destructive(screen_dir / "evals" / "qkv_track_destructive_test.json")
    report = build_promotion_report(row=row, config=config, screen_run_dir=screen_dir, repo_root=tmp_path)

    assert report["destructive_test_present"] is True
    assert report["destructive_test_command_failed"] is False
    assert report["destructive_test_effect_summary"] == {
        "destructive_test_passed": True,
        "max_loss_delta": 0.2,
        "max_logit_delta": 0.3,
        "perturbation_count": 2,
    }


def test_screen_destructive_command_uses_screen_artifacts(tmp_path):
    cmd = build_screen_destructive_test_command(
        screen_config_path=tmp_path / "screen_config.yaml",
        checkpoint_path=tmp_path / "checkpoints" / "ckpt_last.pt",
        out_path=tmp_path / "evals" / "qkv_track_destructive_test.json",
        num_batches=1,
    )

    assert cmd == [
        "uv",
        "run",
        "scripts/qkv_track_destructive_test.py",
        "--config",
        str(tmp_path / "screen_config.yaml"),
        "--checkpoint",
        str(tmp_path / "checkpoints" / "ckpt_last.pt"),
        "--out",
        str(tmp_path / "evals" / "qkv_track_destructive_test.json"),
        "--num-batches",
        "1",
    ]


def test_screen_destructive_command_uses_repo_root_project(tmp_path):
    cmd = build_screen_destructive_test_command(
        screen_config_path=tmp_path / "screen_config.yaml",
        checkpoint_path=tmp_path / "checkpoints" / "ckpt_last.pt",
        out_path=tmp_path / "evals" / "qkv_track_destructive_test.json",
        num_batches=1,
        repo_root=tmp_path,
    )

    assert cmd[:4] == ["uv", "--project", str(tmp_path), "run"]
    assert cmd[4] == str(tmp_path / "scripts" / "qkv_track_destructive_test.py")


def test_schema_required_fields_match_python_validator(repo_root):
    schema = json.loads((repo_root / "reports" / "schema" / "promotion_report.schema.json").read_text(encoding="utf-8"))

    assert set(schema["required"]) == REQUIRED_PROMOTION_REPORT_FIELDS


def test_approve_requires_clean_promotion_report(tmp_path, tiny_config, capsys):
    db_path = tmp_path / "queue.db"
    config, row, screen_dir = _screen_case(tmp_path, tiny_config)
    config_path = tmp_path / "candidate.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    ledger = QueueLedger(db_path)
    ledger.initialize()
    run_id = ledger.enqueue_config(config_path, config, config_path.read_bytes())
    ledger.mark_promotion_candidate(run_id)
    ledger.close()

    with pytest.raises(SystemExit) as missing:
        queue_main(["--db", str(db_path), "approve", run_id])
    assert missing.value.code == 2
    assert "promotion report missing" in capsys.readouterr().err

    _write_metrics(screen_dir / "metrics.jsonl")
    report = build_promotion_report(row={**row, "id": run_id, "config_path": str(config_path)}, config=config, screen_run_dir=screen_dir, repo_root=tmp_path)
    report["promotion_blockers"] = ["manual blocker"]
    report_path = write_promotion_report(report, repo_root=tmp_path)
    ledger = QueueLedger(db_path)
    ledger.initialize()
    ledger.record_promotion_report(run_id, report_path, report)
    ledger.close()
    with pytest.raises(SystemExit) as blocked:
        queue_main(["--db", str(db_path), "--root", str(tmp_path), "approve", run_id])
    assert blocked.value.code == 2
    assert "manual blocker" in capsys.readouterr().err

    report["promotion_blockers"] = []
    report["promotion_recommendation"] = "needs_investigation"
    report_path = write_promotion_report(report, repo_root=tmp_path)
    ledger = QueueLedger(db_path)
    ledger.initialize()
    ledger.record_promotion_report(run_id, report_path, report)
    ledger.close()
    with pytest.raises(SystemExit) as not_promote:
        queue_main(["--db", str(db_path), "--root", str(tmp_path), "approve", run_id])
    assert not_promote.value.code == 2

    report["promotion_recommendation"] = "promote"
    report_path = write_promotion_report(report, repo_root=tmp_path)
    assert (Path(report["screen_run_dir"]) / "promotion_report.json").exists()
    ledger = QueueLedger(db_path)
    ledger.initialize()
    ledger.record_promotion_report(run_id, report_path, report)
    ledger.close()
    queue_main(["--db", str(db_path), "--root", str(tmp_path), "approve", run_id])
    assert "approved:" in capsys.readouterr().out
    ledger = QueueLedger(db_path)
    ledger.initialize()
    approved = ledger.get_run(run_id)
    assert approved["stage"] == "FULL"
    assert approved["full_run_approved"] == 1
    assert approved["promotion_approved_at"] is not None


def test_run_screen_preserves_artifacts_and_writes_report(tmp_path, tiny_config, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = tiny_config(tmp_path, tmp_path / "data")
    config["model"].update(
        {
            "attention_type": "cp_trilinear",
            "cp_rank": 8,
            "cp_lambda_init": 0.0,
            "cp_lambda_trainable": True,
            "cp_lambda_fixed": False,
        }
    )
    config["queue"] = {"mechanism_check": "cp_gradient_norm"}
    config_path = tmp_path / "candidate.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    ledger = QueueLedger(tmp_path / "queue.db")
    ledger.initialize()
    run_id = ledger.enqueue_config(config_path, config, config_path.read_bytes())
    row = ledger.get_run(run_id)

    def fake_screen_command(cmd, log_path):  # noqa: ARG001
        screen_dir = Path(log_path).parent
        _write_metrics(screen_dir / "metrics.jsonl")
        _write_jsonl(screen_dir / "evals" / "attention_diagnostics.jsonl", [{"step": 50, "cp_gradient_norm": 1e-3}])
        (screen_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (screen_dir / "checkpoints" / "ckpt_last.pt").write_bytes(b"test checkpoint")
        return CommandResult(returncode=0, stdout="ok", stderr="")

    result = run_screen(row, ledger, command_runner=fake_screen_command)
    updated = ledger.get_run(run_id)
    screen_dir = Path(updated["screen_run_dir"])
    report = json.loads(Path(updated["promotion_report_path"]).read_text(encoding="utf-8"))

    assert result["ok"] is True
    assert updated["stage"] == "PROMOTION_CANDIDATE"
    assert (screen_dir / "screen_config.yaml").exists()
    assert (screen_dir / "resolved_config.yaml").exists()
    assert (screen_dir / "promotion_report.json").exists()
    assert (screen_dir / "metrics.jsonl").exists()
    assert (screen_dir / "evals" / "attention_diagnostics.jsonl").exists()
    assert Path(report["artifact_paths"]["metrics"]).exists()
    assert Path(report["artifact_paths"]["attention_diagnostics"]).exists()


def test_run_screen_records_multi_qkv_destructive_failure(tmp_path, tiny_config, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = tiny_config(tmp_path, tmp_path / "data")
    config["model"].update(
        {
            "attention_type": "multi_qkv_static_3track_global",
            "qkv_track_count": 3,
            "qkv_global_bank": True,
            "qkv_route_formula": "layer_mod",
        }
    )
    config["queue"] = {"mechanism_check": "qkv_track_activity"}
    config_path = tmp_path / "candidate.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    ledger = QueueLedger(tmp_path / "queue.db")
    ledger.initialize()
    run_id = ledger.enqueue_config(config_path, config, config_path.read_bytes())
    row = ledger.get_run(run_id)

    def fake_command(cmd, log_path):
        screen_dir = Path(log_path).parent
        if _command_has_script(cmd, "scripts/qkv_track_destructive_test.py"):
            return CommandResult(returncode=2, stdout="", stderr="route perturbation failed")
        _write_metrics(screen_dir / "metrics.jsonl")
        _write_jsonl(
            screen_dir / "evals" / "attention_diagnostics.jsonl",
            [_qkv_row(layer_idx=layer_idx, active_track=layer_idx) for layer_idx in range(3)],
        )
        (screen_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (screen_dir / "checkpoints" / "ckpt_last.pt").write_bytes(b"test checkpoint")
        return CommandResult(returncode=0, stdout="ok", stderr="")

    result = run_screen(row, ledger, command_runner=fake_command, repo_root=tmp_path)
    report = json.loads(Path(ledger.get_run(run_id)["promotion_report_path"]).read_text(encoding="utf-8"))

    assert result["destructive_test_failed"] is True
    assert report["destructive_test_command_failed"] is True
    assert report["destructive_test_failure_summary"]["returncode"] == 2
    assert report["promotion_recommendation"] == "needs_investigation"
    assert "destructive test command failed" in " ".join(approval_blockers(report))


def test_run_screen_uses_repo_root_when_cwd_differs(tmp_path, tiny_config, monkeypatch):
    repo_root = tmp_path / "repo"
    outside = tmp_path / "outside"
    repo_root.mkdir()
    outside.mkdir()
    monkeypatch.chdir(outside)
    config = tiny_config(repo_root, repo_root / "data")
    config_path = repo_root / "queue" / "inbox" / "candidate.yaml"
    config_path.parent.mkdir(parents=True)
    (repo_root / "data").mkdir(parents=True)
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    ledger = QueueLedger(repo_root / "data" / "queue.db")
    ledger.initialize()
    run_id = ledger.enqueue_config(config_path, config, config_path.read_bytes())
    row = ledger.get_run(run_id)
    commands = []

    def fake_command(cmd, log_path):
        commands.append(cmd)
        screen_dir = Path(log_path).parent
        _write_metrics(screen_dir / "metrics.jsonl")
        (screen_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (screen_dir / "checkpoints" / "ckpt_last.pt").write_bytes(b"test checkpoint")
        return CommandResult(returncode=0, stdout="ok", stderr="")

    run_screen(row, ledger, command_runner=fake_command, repo_root=repo_root)

    updated = ledger.get_run(run_id)
    assert commands[0][:4] == ["uv", "--project", str(repo_root), "run"]
    assert commands[0][4] == str(repo_root / "scripts" / "train.py")
    assert Path(updated["promotion_report_path"]).is_relative_to(repo_root)
    assert Path(updated["screen_run_dir"]).is_relative_to(repo_root)
    assert not (outside / "reports").exists()


def _screen_case(
    tmp_path: Path,
    tiny_config,
    *,
    attention_type: str = "standard",
    name: str = "candidate",
    checkpoint: bool = True,
):
    config = tiny_config(tmp_path, tmp_path / "data")
    config["run"]["name"] = name
    config["run"]["out_dir"] = str(tmp_path / "runs" / name)
    if attention_type in {"cp_bilinear", "cp_trilinear"}:
        config["model"].update(
            {
                "attention_type": attention_type,
                "cp_rank": 8,
                "cp_lambda_init": 0.0,
                "cp_lambda_trainable": True,
                "cp_lambda_fixed": False,
            }
        )
        config["queue"] = {"mechanism_check": "cp_gradient_norm"}
    elif attention_type.startswith("multi_qkv_"):
        config["model"].update(
            {
                "attention_type": attention_type,
                "qkv_track_count": 3,
                "qkv_global_bank": True,
                "qkv_route_formula": "layer_mod",
            }
        )
        config["queue"] = {"mechanism_check": "qkv_track_activity"}
    row = {
        "id": f"{name}_id",
        "config_name": name,
        "config_path": str(tmp_path / f"{name}.yaml"),
        "run_dir": config["run"]["out_dir"],
        "failure_class": None,
    }
    screen_dir = tmp_path / "runs" / "screen" / name
    screen_dir.mkdir(parents=True)
    (screen_dir / "screen_config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    (screen_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    if checkpoint:
        (screen_dir / "checkpoints").mkdir(parents=True)
        (screen_dir / "checkpoints" / "ckpt_last.pt").write_bytes(b"test checkpoint")
    return config, row, screen_dir


def _write_metrics(path: Path, *, val_losses=(6.0, 5.0)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"event": "train", "step": 50, "tokens_per_sec": 100.0, "train_loss": 5.5},
        {"event": "val", "step": 50, "val_loss": val_losses[0]},
        {"event": "train", "step": 150, "tokens_per_sec": 120.0, "train_loss": 4.5},
        {"event": "val", "step": 150, "val_loss": val_losses[1], "peak_vram_allocated_mb": 123.0},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _qkv_row(*, layer_idx: int, active_track: int) -> dict:
    return {
        "schema_version": 1,
        "attention_type": "multi_qkv_static_3track_global",
        "route_formula": "layer_idx % track_count",
        "uses_global_bank": True,
        "track_count": 3,
        "layer_idx": layer_idx,
        "layer": layer_idx,
        "step": 50,
        "last_forward_step": 50,
        "schedule_mode": "train",
        "active_track_index": active_track,
        "active_track_counts": {str(track): (8 if track == active_track else 0) for track in range(3)},
        "track_gradient_norm": 0.1,
        "per_track_gradient_norm": {str(track): (0.1 if track == active_track else 0.0) for track in range(3)},
        "per_track_qkv_weight_norm": {"0": 1.0, "1": 1.0, "2": 1.0},
        "position_routing_enabled": False,
        "eval_freeze_mode": False,
    }


def _qkv_train_rotation_row(
    *,
    layer_idx: int,
    active_track: int,
    last_forward_step: int,
    schedule_mode: str,
) -> dict:
    return {
        "schema_version": 1,
        "attention_type": "multi_qkv_train_rotation_3track_global",
        "route_formula": "(layer_idx + step) % track_count during train; layer_idx % track_count during eval/generate",
        "uses_global_bank": True,
        "track_count": 3,
        "layer_idx": layer_idx,
        "layer": layer_idx,
        "step": last_forward_step,
        "last_forward_step": last_forward_step,
        "schedule_mode": schedule_mode,
        "active_track_index": active_track,
        "active_track_counts": {str(track): (8 if track == active_track else 0) for track in range(3)},
        "track_gradient_norm": 0.1,
        "per_track_gradient_norm": {"0": 0.1, "1": 0.1, "2": 0.1},
        "per_track_qkv_weight_norm": {"0": 1.0, "1": 1.0, "2": 1.0},
        "position_routing_enabled": False,
        "eval_freeze_mode": True,
    }


def _qkv_position_rotation_row(
    *,
    layer_idx: int,
    active_track: int | None,
    counts: dict[str, int] | None = None,
) -> dict:
    return {
        "schema_version": 1,
        "attention_type": "multi_qkv_position_rotation_3track_global",
        "route_formula": "(layer_idx + position) % track_count",
        "uses_global_bank": True,
        "track_count": 3,
        "layer_idx": layer_idx,
        "layer": layer_idx,
        "step": 50,
        "last_forward_step": 50,
        "schedule_mode": "train",
        "active_track_index": active_track,
        "active_track_counts": counts or {"0": 8, "1": 0, "2": 0},
        "track_gradient_norm": 0.1,
        "per_track_gradient_norm": {"0": 0.1, "1": 0.1, "2": 0.1},
        "per_track_qkv_weight_norm": {"0": 1.0, "1": 1.0, "2": 1.0},
        "position_routing_enabled": True,
        "eval_freeze_mode": False,
    }


def _write_destructive(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "destructive_test_passed": True,
                "perturbations": [
                    {"loss_delta": 0.2, "max_abs_logit_delta": 0.1},
                    {"loss_delta": -0.1, "max_abs_logit_delta": 0.3},
                ],
            }
        ),
        encoding="utf-8",
    )


def _command_has_script(cmd: list[str], script_path: str) -> bool:
    return any(str(part).endswith(script_path) for part in cmd)
