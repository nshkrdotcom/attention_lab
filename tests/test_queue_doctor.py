from __future__ import annotations

from pathlib import Path

from attention_lab.queue.cli import main as queue_main
from attention_lab.queue.doctor import run_doctor
from attention_lab.queue.ledger import QueueLedger


def test_doctor_passes_for_e001_current_configs(tmp_path):
    ledger = QueueLedger(tmp_path / "queue.db")
    ledger.initialize()
    report = run_doctor(experiment_id="E001_cp_trilinear_attention", ledger=ledger, root=Path.cwd())
    assert report.ok is True
    text = "\n".join(message.text for message in report.messages)
    assert "experiment exists: E001_cp_trilinear_attention" in text
    assert "all run.out_dir values are unique" in text


def test_doctor_passes_for_e002_skeleton(tmp_path):
    ledger = QueueLedger(tmp_path / "queue.db")
    ledger.initialize()
    report = run_doctor(experiment_id="E002_multitrack_qkv_shift_register", ledger=ledger, root=Path.cwd())
    assert report.ok is True
    text = "\n".join(message.text for message in report.messages)
    assert "explicitly experimental/unimplemented" in text


def test_doctor_fails_duplicate_run_dirs(tmp_path, tiny_config, monkeypatch):
    import yaml

    experiment = _fake_experiment(tmp_path)
    monkeypatch.setattr("attention_lab.queue.doctor.get_experiment", lambda experiment_id: experiment)
    config_dir = Path(experiment["config_dir"])
    config_dir.mkdir(parents=True)
    Path(experiment["report_dir"]).mkdir(parents=True)
    Path(experiment["dataset_manifest"]).parent.mkdir(parents=True)
    Path(experiment["dataset_manifest"]).write_text("{}", encoding="utf-8")
    config = tiny_config(tmp_path, tmp_path / "data")
    (config_dir / "a.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    config["run"]["name"] = "b"
    (config_dir / "b.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")

    ledger = QueueLedger(tmp_path / "queue.db")
    ledger.initialize()
    report = run_doctor(experiment_id="E999", ledger=ledger, root=Path.cwd())
    assert report.ok is False
    assert any("duplicate run.out_dir" in message.text for message in report.messages)


def test_doctor_warns_missing_hypothesis_only_for_approved_full_rows(tmp_path, tiny_config, monkeypatch):
    import yaml

    experiment = _fake_experiment(tmp_path)
    monkeypatch.setattr("attention_lab.queue.doctor.get_experiment", lambda experiment_id: experiment)
    config_dir = Path(experiment["config_dir"])
    config_dir.mkdir(parents=True)
    Path(experiment["report_dir"]).mkdir(parents=True)
    Path(experiment["dataset_manifest"]).parent.mkdir(parents=True)
    Path(experiment["dataset_manifest"]).write_text("{}", encoding="utf-8")
    config = tiny_config(tmp_path, tmp_path / "data")
    config_path = config_dir / "candidate.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    ledger = QueueLedger(tmp_path / "queue.db")
    ledger.initialize()
    run_id = ledger.enqueue_config(config_path, config, config_path.read_bytes())
    ledger.promote_to_full(run_id)
    ledger.set_full_run_approved(run_id, True)

    report = run_doctor(experiment_id="E999", ledger=ledger, root=Path.cwd())
    assert report.ok is True
    assert any(message.level == "WARN" and "missing hypothesis" in message.text for message in report.messages)


def test_cli_doctor_prints_report(tmp_path, capsys):
    queue_main(["--db", str(tmp_path / "queue.db"), "doctor", "--experiment", "E001_cp_trilinear_attention"])
    output = capsys.readouterr().out
    assert "OK: experiment exists: E001_cp_trilinear_attention" in output


def _fake_experiment(tmp_path: Path) -> dict:
    return {
        "id": "E999",
        "status": "planned",
        "plan": str(tmp_path / "plan.md"),
        "config_dir": str(tmp_path / "configs"),
        "run_dir": str(tmp_path / "runs"),
        "report_dir": str(tmp_path / "reports"),
        "baseline_config": str(tmp_path / "configs" / "a.yaml"),
        "baseline_reference_run": str(tmp_path / "baseline"),
        "accurate_baseline_alias": "configs/baseline_30m_fineweb100m.yaml",
        "dataset_manifest": str(tmp_path / "data" / "manifest.json"),
    }
