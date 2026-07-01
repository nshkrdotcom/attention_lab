from __future__ import annotations

import json
from pathlib import Path

from attention_lab.queue.ledger import QueueLedger
from attention_lab.queue.reporting import append_decision_log, export_queue_report


def _fake_experiment(tmp_path: Path) -> dict:
    return {
        "id": "E999_test",
        "run_dir": "runs/experiments/E999_test",
        "config_dir": str(tmp_path / "configs" / "experiments" / "E999_test"),
        "report_dir": str(tmp_path / "reports" / "experiments" / "E999_test"),
    }


def test_export_queue_report_writes_json_and_markdown(tmp_path, tiny_config, monkeypatch):
    import yaml

    monkeypatch.setattr("attention_lab.queue.reporting.get_experiment", lambda experiment_id: _fake_experiment(tmp_path))
    ledger = QueueLedger(tmp_path / "queue.db")
    ledger.initialize()
    e001_config = tiny_config(tmp_path, tmp_path / "data")
    e001_config["run"]["name"] = "standard_30m_seed1"
    e001_config["run"]["out_dir"] = "runs/experiments/E999_test/standard_30m_seed1"
    e001_config["queue"] = {
        "full_run_approved": True,
        "allow_overwrite_existing_run_dir": False,
        "requires_run": "control",
        "mechanism_check": "cp_gradient_norm",
    }
    e001_path = tmp_path / "standard.yaml"
    e001_path.write_text(yaml.safe_dump(e001_config), encoding="utf-8")
    run_id = ledger.enqueue_config(e001_path, e001_config, e001_path.read_bytes())
    ledger.promote_to_full(run_id)
    ledger.mark_passed(
        run_id,
        step_reached=3000,
        final_val_loss=4.0,
        best_val_loss=3.9,
        final_ppl=54.6,
        median_tokens_per_sec=100000.0,
        peak_vram_allocated_mb=3200.0,
        mechanism_active=None,
    )

    other = tiny_config(tmp_path, tmp_path / "other_data")
    other["run"]["out_dir"] = "runs/experiments/OTHER/candidate"
    other_path = tmp_path / "other.yaml"
    other_path.write_text(yaml.safe_dump(other), encoding="utf-8")
    other_id = ledger.enqueue_config(other_path, other, other_path.read_bytes())
    ledger.mark_failed(other_id, failure_class="NAN")

    result = export_queue_report(experiment_id="E999_test", ledger=ledger, repo_root=Path.cwd())
    json_path = Path(result["json_path"])
    md_path = Path(result["markdown_path"])
    assert json_path.exists()
    assert md_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["experiment_id"] == "E999_test"
    assert [row["config_name"] for row in payload["runs"]] == ["standard"]
    row = payload["runs"][0]
    assert row["full_run_approved"] == 1
    assert row["allow_overwrite_existing_run_dir"] == 0
    assert row["queue_requires_run"] == "control"
    assert row["queue_mechanism_check"] == "cp_gradient_norm"
    text = md_path.read_text(encoding="utf-8")
    assert "standard" in text
    assert "approved" in text
    assert "overwrite" in text
    assert "---" in text


def test_export_queue_report_uses_config_backed_rows_with_queue_fields(tmp_path, tiny_config, monkeypatch):
    import yaml

    monkeypatch.setattr("attention_lab.queue.reporting.get_experiment", lambda experiment_id: _fake_experiment(tmp_path))
    config_dir = tmp_path / "configs" / "experiments" / "E999_test"
    config_dir.mkdir(parents=True)
    config = tiny_config(tmp_path, tmp_path / "data")
    config["run"]["name"] = "candidate"
    config["run"]["out_dir"] = "runs/experiments/E999_test/candidate"
    config["queue"] = {
        "full_run_approved": False,
        "allow_overwrite_existing_run_dir": True,
        "requires_run": "standard",
        "mechanism_check": "qkv_track_activity",
    }
    (config_dir / "candidate.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    ledger = QueueLedger(tmp_path / "queue.db")
    ledger.initialize()
    result = export_queue_report(experiment_id="E999_test", ledger=ledger, repo_root=Path.cwd())
    payload = json.loads(Path(result["json_path"]).read_text(encoding="utf-8"))
    assert len(payload["runs"]) == 1
    row = payload["runs"][0]
    assert row["status"] == "NOT_QUEUED"
    assert row["full_run_approved"] is False
    assert row["allow_overwrite_existing_run_dir"] is True
    assert row["queue_requires_run"] == "standard"
    assert row["queue_mechanism_check"] == "qkv_track_activity"
    markdown = Path(result["markdown_path"]).read_text(encoding="utf-8")
    assert "| candidate |" in markdown
    assert "| no | yes | standard | qkv_track_activity |" in markdown


def test_morning_note_creates_and_appends(tmp_path, monkeypatch):
    monkeypatch.setattr("attention_lab.queue.reporting.get_experiment", lambda experiment_id: _fake_experiment(tmp_path))
    path = append_decision_log(
        experiment_id="E999_test",
        shows="A passed.",
        not_shows="No claim yet.",
        next_step="Run B.",
        repo_root=Path.cwd(),
    )
    append_decision_log(
        experiment_id="E999_test",
        shows="B failed.",
        not_shows="No improvement.",
        next_step="Inspect diagnostics.",
        repo_root=Path.cwd(),
    )
    text = path.read_text(encoding="utf-8")
    assert text.count("SHOWS:") >= 2
    assert "Inspect diagnostics." in text

    try:
        append_decision_log(
            experiment_id="E999_test",
            shows="",
            not_shows="No claim.",
            next_step="Stop.",
            repo_root=Path.cwd(),
        )
    except ValueError as exc:
        assert "nonempty" in str(exc)
    else:
        raise AssertionError("empty morning-note field was accepted")
