from __future__ import annotations

import json
from pathlib import Path

from attention_lab.queue.cli import main as queue_main
from attention_lab.queue.discipline import default_hypothesis_path, validate_hypothesis_doc
from attention_lab.queue.leaderboard import render_leaderboard
from attention_lab.queue.ledger import QueueLedger
from attention_lab.queue.watchdog import Watchdog


def write_hypothesis(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """CLAIM:
This run will lower validation loss versus control.

KILL_CONDITION:
Final val loss is not at least 0.02 lower than control.

MECHANISM_PROOF:
cp_gradient_norm > 1e-4 by step 500.

NEAREST_BORING_EXPLANATION:
Extra low-rank parameters.

CONTROL_THAT_RULES_IT_OUT:
cp_bilinear_r8_30m_seed1.
""",
        encoding="utf-8",
    )


def test_hypothesis_doc_validation_and_default_path(tmp_path, tiny_config):
    config = tiny_config(tmp_path, tmp_path / "data")
    config_path = Path("configs/experiments/E999_example/candidate.yaml")
    assert default_hypothesis_path(config_path, config) == Path("docs/experiments/E999_example/hypothesis_tiny_test_run.md")

    hypothesis_path = tmp_path / "hypothesis.md"
    write_hypothesis(hypothesis_path)
    result = validate_hypothesis_doc(hypothesis_path)
    assert result.ok is True
    assert result.missing_fields == []


def test_leaderboard_renders_mechanism_and_missing_values():
    rows = [
        {
            "config_name": "cp_trilinear_r8_30m_seed1",
            "attention_type": "cp_trilinear",
            "stage": "FULL",
            "status": "PASSED",
            "final_val_loss": 4.0,
            "final_ppl": 54.6,
            "median_tokens_per_sec": 100000.0,
            "peak_vram_allocated_mb": 3200.0,
            "mechanism_active": 1,
            "full_run_approved": 1,
            "notes": "SHOWS: active",
        },
        {
            "config_name": "candidate_pending",
            "attention_type": "cp_bilinear",
            "stage": "SCREEN",
            "status": "PENDING",
            "final_val_loss": None,
            "final_ppl": None,
            "median_tokens_per_sec": None,
            "peak_vram_allocated_mb": None,
            "mechanism_active": None,
            "full_run_approved": 0,
            "notes": "",
        },
    ]
    output = render_leaderboard(rows, now="2026-06-30 07:14")
    assert "QUEUE STATUS  2026-06-30 07:14" in output
    assert "cp_trilinear_r8_30m_seed1" in output
    assert "4.000" in output
    assert "100k" in output
    assert "candidate_pending" in output
    assert "approved=N" in output
    assert "---" in output


def test_leaderboard_filter_and_sort_options():
    rows = [
        {"config_name": "screen", "attention_type": "standard", "stage": "SCREEN", "status": "PENDING"},
        {
            "config_name": "slow",
            "attention_type": "standard",
            "stage": "FULL",
            "status": "PASSED",
            "final_val_loss": 5.0,
            "final_ppl": 150.0,
            "median_tokens_per_sec": 10.0,
        },
        {
            "config_name": "fast",
            "attention_type": "standard",
            "stage": "FULL",
            "status": "PASSED",
            "final_val_loss": 4.0,
            "final_ppl": 100.0,
            "median_tokens_per_sec": 20.0,
        },
    ]
    full_only = render_leaderboard(rows, min_stage="FULL")
    assert "screen" not in full_only
    by_loss = render_leaderboard(rows, min_stage="FULL", sort="loss")
    assert by_loss.find("fast") < by_loss.find("slow")
    by_speed = render_leaderboard(rows, min_stage="FULL", sort="speed")
    assert by_speed.find("fast") < by_speed.find("slow")


def test_cli_ls_note_requeue_and_status(tmp_path, tiny_config, capsys):
    db_path = tmp_path / "queue.db"
    ledger = QueueLedger(db_path)
    ledger.initialize()
    config = tiny_config(tmp_path, tmp_path / "data")
    config_path = tmp_path / "candidate.yaml"
    import yaml

    content = yaml.safe_dump(config).encode()
    config_path.write_bytes(content)
    run_id = ledger.enqueue_config(config_path, config, content)
    ledger.promote_to_full(run_id)
    ledger.mark_failed(run_id, failure_class="NAN")
    ledger.close()

    queue_main(["--db", str(db_path), "ls"])
    assert "candidate" in capsys.readouterr().out

    queue_main(["--db", str(db_path), "note", run_id, "SHOWS: failed with NAN"])
    queue_main(["--db", str(db_path), "show", run_id])
    assert "SHOWS: failed with NAN" in capsys.readouterr().out

    queue_main(["--db", str(db_path), "requeue", run_id])
    queue_main(["--db", str(db_path), "unapprove", run_id])
    queue_main(["--db", str(db_path), "status"])
    output = capsys.readouterr().out
    assert "QUEUE STATUS" in output
    assert "PENDING" in output
    assert "approved=N" in output


def test_watchdog_screens_before_full_and_skips_missing_hypothesis(tmp_path, tiny_config):
    db_path = tmp_path / "queue.db"
    ledger = QueueLedger(db_path)
    ledger.initialize()
    config = tiny_config(tmp_path, tmp_path / "data")
    config_path = tmp_path / "candidate.yaml"
    import yaml

    content = yaml.safe_dump(config).encode()
    config_path.write_bytes(content)
    run_id = ledger.enqueue_config(config_path, config, content)

    events = []

    def screen_runner(row, ledger_obj):
        events.append(("screen", row["id"]))
        report_path = tmp_path / "promotion.json"
        _write_clean_promotion_report(report_path, row, config)
        ledger_obj.record_promotion_report(row["id"], report_path, json.loads(report_path.read_text(encoding="utf-8")))
        ledger_obj.mark_promotion_candidate(row["id"])
        return {"ok": True}

    def full_runner(row, ledger_obj):
        events.append(("full", row["id"]))
        ledger_obj.mark_passed(row["id"])
        return {"ok": True}

    watchdog = Watchdog(ledger, screen_runner=screen_runner, full_runner=full_runner, sleep_seconds=0)
    watchdog.run_once()
    assert events == [("screen", run_id)]

    # Promotion candidates do not become full work until explicitly approved.
    watchdog.run_once()
    assert events == [("screen", run_id)]
    assert ledger.get_run(run_id)["stage"] == "PROMOTION_CANDIDATE"

    hypothesis_path = default_hypothesis_path(config_path, config)
    write_hypothesis(hypothesis_path)
    try:
        ledger.approve_full_run(run_id)
        watchdog.run_once()
        assert events == [("screen", run_id), ("full", run_id)]
    finally:
        hypothesis_path.unlink(missing_ok=True)
        try:
            hypothesis_path.parent.rmdir()
        except OSError:
            pass


def test_cli_add_validates_and_copies(tmp_path, tiny_config, capsys):
    root = tmp_path / "root"
    root.mkdir()
    db_path = root / "data" / "queue.db"
    source = tmp_path / "candidate.yaml"
    import yaml

    source.write_text(yaml.safe_dump(tiny_config(tmp_path, tmp_path / "data")), encoding="utf-8")
    queue_main(["--root", str(root), "--db", str(db_path), "add", str(source)])
    output = capsys.readouterr().out
    assert "added:" in output
    assert (root / "queue" / "inbox" / "candidate.yaml").exists()
    assert json.dumps({"ok": True}) != ""


def test_cli_export_report_and_morning_note_commands(tmp_path, capsys, monkeypatch):
    db_path = tmp_path / "queue.db"

    def fake_export_report(*, experiment_id, ledger, repo_root):  # noqa: ARG001
        return {
            "json_path": tmp_path / "run_index.json",
            "markdown_path": tmp_path / "run_index.md",
            "row_count": 0,
        }

    def fake_append_decision_log(*, experiment_id, shows, not_shows, next_step, repo_root):  # noqa: ARG001
        assert experiment_id == "E001_cp_trilinear_attention"
        assert shows == "A"
        assert not_shows == "B"
        assert next_step == "C"
        return tmp_path / "decision_log.md"

    monkeypatch.setattr("attention_lab.queue.cli.export_queue_report", fake_export_report)
    monkeypatch.setattr("attention_lab.queue.cli.append_decision_log", fake_append_decision_log)
    queue_main(["--db", str(db_path), "export-report", "--experiment", "E001_cp_trilinear_attention"])
    assert "rows: 0" in capsys.readouterr().out
    queue_main(
        [
            "--db",
            str(db_path),
            "morning-note",
            "--experiment",
            "E001_cp_trilinear_attention",
            "--shows",
            "A",
            "--not-shows",
            "B",
            "--next",
            "C",
        ]
    )
    assert "decision_log.md" in capsys.readouterr().out


def test_watchdog_requires_control_dependency_for_nonstandard_full(tmp_path, tiny_config):
    db_path = tmp_path / "queue.db"
    ledger = QueueLedger(db_path)
    ledger.initialize()
    config = tiny_config(tmp_path, tmp_path / "data")
    config["model"]["attention_type"] = "cp_trilinear"
    config["model"]["cp_rank"] = 8
    config["model"]["cp_lambda_trainable"] = True
    config["model"]["cp_lambda_fixed"] = False
    config["model"]["cp_lambda_init"] = 0.0
    config_path = tmp_path / "candidate.yaml"
    import yaml

    content = yaml.safe_dump(config).encode()
    config_path.write_bytes(content)
    run_id = ledger.enqueue_config(config_path, config, content)
    ledger.promote_to_full(run_id)
    report_path = tmp_path / "candidate_promotion.json"
    _write_clean_promotion_report(report_path, ledger.get_run(run_id), config)
    ledger.record_promotion_report(run_id, report_path, json.loads(report_path.read_text(encoding="utf-8")))
    ledger.approve_full_run(run_id)

    events = []

    def full_runner(row, ledger_obj):
        events.append(("full", row["id"]))
        ledger_obj.mark_passed(row["id"])
        return {"ok": True}

    watchdog = Watchdog(ledger, screen_runner=lambda row, ledger_obj: {"ok": True}, full_runner=full_runner, sleep_seconds=0)
    watchdog.run_once()
    assert events == []
    assert "requires_run" in (ledger.get_run(run_id)["notes"] or "")

    config["queue"] = {"skip_control_check": True, "skip_hypothesis_check": True}
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    watchdog.run_once()
    assert events == [("full", run_id)]
    assert "control dependency explicitly skipped" in (ledger.get_run(run_id)["notes"] or "")


def test_watchdog_blocks_full_rows_without_promotion_report(tmp_path, tiny_config):
    ledger, run_id, _config, _config_path = _full_row(tmp_path, tiny_config)
    ledger.set_full_run_approved(run_id, True)
    events = []
    watchdog = Watchdog(
        ledger,
        screen_runner=lambda row, ledger_obj: {"ok": True},
        full_runner=lambda row, ledger_obj: events.append(row["id"]) or {"ok": True},
        sleep_seconds=0,
    )

    watchdog.run_once()

    assert events == []
    assert "promotion report" in (ledger.get_run(run_id)["notes"] or "")


def test_watchdog_blocks_full_rows_with_blocked_promotion_report(tmp_path, tiny_config):
    ledger, run_id, config, _config_path = _full_row(tmp_path, tiny_config)
    report_path = tmp_path / "blocked_promotion.json"
    _write_clean_promotion_report(report_path, ledger.get_run(run_id), config)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["promotion_blockers"] = ["needs review"]
    report_path.write_text(json.dumps(report), encoding="utf-8")
    ledger.record_promotion_report(run_id, report_path, report)
    ledger.approve_full_run(run_id)
    events = []
    watchdog = Watchdog(
        ledger,
        screen_runner=lambda row, ledger_obj: {"ok": True},
        full_runner=lambda row, ledger_obj: events.append(row["id"]) or {"ok": True},
        sleep_seconds=0,
    )

    watchdog.run_once()

    assert events == []
    assert "promotion report blocked" in (ledger.get_run(run_id)["notes"] or "")


def test_watchdog_blocks_full_rows_without_full_approval(tmp_path, tiny_config):
    ledger, run_id, config, _config_path = _full_row(tmp_path, tiny_config)
    report_path = tmp_path / "promotion.json"
    _write_clean_promotion_report(report_path, ledger.get_run(run_id), config)
    ledger.record_promotion_report(run_id, report_path, json.loads(report_path.read_text(encoding="utf-8")))
    events = []
    watchdog = Watchdog(
        ledger,
        screen_runner=lambda row, ledger_obj: {"ok": True},
        full_runner=lambda row, ledger_obj: events.append(row["id"]) or {"ok": True},
        sleep_seconds=0,
    )

    watchdog.run_once()

    assert events == []
    assert "full_run_approved" in (ledger.get_run(run_id)["notes"] or "")


def test_watchdog_waits_for_required_run_to_pass(tmp_path, tiny_config):
    db_path = tmp_path / "queue.db"
    ledger = QueueLedger(db_path)
    ledger.initialize()
    standard = tiny_config(tmp_path, tmp_path / "data")
    standard_path = tmp_path / "standard.yaml"
    candidate = tiny_config(tmp_path, tmp_path / "data")
    candidate["run"]["name"] = "candidate"
    candidate["run"]["out_dir"] = str(tmp_path / "runs" / "candidate")
    candidate["model"]["attention_type"] = "cp_bilinear"
    candidate["model"]["cp_rank"] = 8
    candidate["model"]["cp_lambda_trainable"] = True
    candidate["model"]["cp_lambda_fixed"] = False
    candidate["model"]["cp_lambda_init"] = 0.0
    candidate["queue"] = {
        "requires_run": "standard",
        "skip_hypothesis_check": True,
    }
    candidate_path = tmp_path / "candidate.yaml"
    import yaml

    standard_path.write_text(yaml.safe_dump(standard), encoding="utf-8")
    candidate_path.write_text(yaml.safe_dump(candidate), encoding="utf-8")
    standard_id = ledger.enqueue_config(standard_path, standard, standard_path.read_bytes())
    candidate_id = ledger.enqueue_config(candidate_path, candidate, candidate_path.read_bytes())
    ledger.promote_to_full(standard_id)
    ledger.mark_failed(standard_id, failure_class="UNKNOWN")
    ledger.promote_to_full(candidate_id)
    report_path = tmp_path / "candidate_promotion.json"
    _write_clean_promotion_report(report_path, ledger.get_run(candidate_id), candidate)
    ledger.record_promotion_report(candidate_id, report_path, json.loads(report_path.read_text(encoding="utf-8")))
    ledger.approve_full_run(candidate_id)

    events = []

    def full_runner(row, ledger_obj):
        events.append(row["config_name"])
        ledger_obj.mark_passed(row["id"])
        return {"ok": True}

    watchdog = Watchdog(ledger, screen_runner=lambda row, ledger_obj: {"ok": True}, full_runner=full_runner, sleep_seconds=0)
    watchdog.run_once()
    assert events == []
    assert "required run" in (ledger.get_run(candidate_id)["notes"] or "")

    ledger.mark_passed(standard_id)
    watchdog.run_once()
    assert events == ["candidate"]


def _full_row(tmp_path: Path, tiny_config):
    db_path = tmp_path / "queue.db"
    ledger = QueueLedger(db_path)
    ledger.initialize()
    config = tiny_config(tmp_path, tmp_path / "data")
    config_path = tmp_path / "candidate.yaml"
    import yaml

    content = yaml.safe_dump(config).encode()
    config_path.write_bytes(content)
    run_id = ledger.enqueue_config(config_path, config, content)
    ledger.promote_to_full(run_id)
    return ledger, run_id, config, config_path


def _write_clean_promotion_report(path: Path, row: dict, config: dict) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "experiment_id": None,
                "run_id": row["id"],
                "run_name": config["run"]["name"],
                "config_path": str(row["config_path"]),
                "screen_config_path": str(path.parent / "screen_config.yaml"),
                "screen_run_dir": str(path.parent),
                "attention_type": config["model"].get("attention_type", "standard"),
                "stage": "SCREEN",
                "max_step_reached": 150,
                "expected_screen_steps": 150,
                "loss_descended": True,
                "nan_or_inf_seen": False,
                "final_screen_train_loss": 4.0,
                "final_screen_val_loss": 4.0,
                "first_val_loss": 5.0,
                "final_val_loss": 4.0,
                "median_tokens_per_sec": 100.0,
                "peak_vram_mb": None,
                "peak_vram_allocated_mb": None,
                "diagnostics_present": config["model"].get("attention_type", "standard") != "standard",
                "diagnostics_non_degenerate": config["model"].get("attention_type", "standard") != "standard",
                "mechanism_check_name": None,
                "mechanism_active": None if config["model"].get("attention_type", "standard") == "standard" else True,
                "mechanism_activity_summary": {},
                "destructive_test_present": False,
                "destructive_test_effect_summary": None,
                "promotion_recommendation": "promote",
                "promotion_blockers": [],
                "promotion_reason": "test report",
                "created_at": "2026-07-01T00:00:00+00:00",
                "source_git_commit": None,
                "source_dirty": False,
            }
        ),
        encoding="utf-8",
    )
