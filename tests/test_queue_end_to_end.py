from __future__ import annotations

import json
from pathlib import Path

import yaml

from attention_lab.queue.discipline import REQUIRED_HYPOTHESIS_FIELDS
from attention_lab.queue.ledger import QueueLedger
from attention_lab.queue.paths import ensure_queue_dirs
from attention_lab.queue.reporting import export_queue_report
from attention_lab.queue.runner import CommandResult, run_full
from attention_lab.queue.screener import run_screen
from attention_lab.queue.watchdog import Watchdog


def test_queue_end_to_end_dry_run_with_fake_commands(tmp_path, tiny_config, monkeypatch):
    root = tmp_path
    paths = ensure_queue_dirs(root)
    data_root = root / "data_shards"
    hypothesis_path = root / "hypothesis_tiny_standard.md"
    _write_hypothesis(hypothesis_path)

    config = tiny_config(root, data_root)
    config["run"]["name"] = "tiny_standard"
    config["run"]["out_dir"] = str(root / "runs" / "experiments" / "E999" / "tiny_standard")
    config["queue"] = {
        "hypothesis_doc": str(hypothesis_path),
        "full_run_approved": False,
        "allow_overwrite_existing_run_dir": False,
    }
    config_path = paths["inbox"] / "tiny_standard.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    ledger = QueueLedger(root / "data" / "queue.db")
    ledger.initialize()

    def fake_screen_command(cmd, log_path):  # noqa: ARG001
        screen_run_dir = Path(log_path).parent
        _write_screen_metrics(screen_run_dir / "metrics.jsonl")
        return CommandResult(returncode=0, stdout="screen ok", stderr="")

    def screen_runner(row, ledger_obj):
        return run_screen(row, ledger_obj, command_runner=fake_screen_command, keep_screens=True)

    full_calls = []

    def fake_full_command(cmd, log_path):
        full_calls.append(cmd)
        run_dir = Path(config["run"]["out_dir"])
        _write_full_artifact_for_command(cmd, run_dir)
        return CommandResult(returncode=0, stdout="full ok", stderr="")

    def full_runner(row, ledger_obj):
        return run_full(row, ledger_obj, command_runner=fake_full_command, repo_root=Path.cwd())

    watchdog = Watchdog(
        ledger,
        inbox_dir=paths["inbox"],
        screen_runner=screen_runner,
        full_runner=full_runner,
        sleep_seconds=0,
    )

    first = watchdog.run_once()
    rows = ledger.list_runs()
    assert first["action"] == "screen"
    assert len(rows) == 1
    row = rows[0]
    assert row["stage"] == "FULL"
    assert row["status"] == "PENDING"
    assert row["full_run_approved"] == 0
    assert not config_path.exists()
    pending_path = paths["full_pending"] / "tiny_standard.yaml"
    assert pending_path.exists()

    second = watchdog.run_once()
    row = ledger.get_run(row["id"])
    assert second["action"] == "sleep"
    assert "full_run_approved" in (row["notes"] or "")
    assert full_calls == []

    ledger.set_full_run_approved(row["id"], True)
    third = watchdog.run_once()
    row = ledger.get_run(row["id"])
    assert third["action"] == "full"
    assert row["status"] == "PASSED"
    assert row["final_val_loss"] == 4.0
    assert not pending_path.exists()
    assert (paths["done"] / "tiny_standard.yaml").exists()
    assert len(full_calls) == 8

    experiment = {
        "id": "E999",
        "run_dir": str(root / "runs" / "experiments" / "E999"),
        "config_dir": str(root / "configs" / "experiments" / "E999"),
        "report_dir": str(root / "reports" / "experiments" / "E999"),
    }
    monkeypatch.setattr("attention_lab.queue.reporting.get_experiment", lambda experiment_id: experiment)
    result = export_queue_report(experiment_id="E999", ledger=ledger, repo_root=Path.cwd())
    payload = json.loads(Path(result["json_path"]).read_text(encoding="utf-8"))
    assert payload["runs"][0]["status"] == "PASSED"
    assert payload["runs"][0]["full_run_approved"] == 1


def _write_screen_metrics(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"event": "train", "step": 50, "tokens_per_sec": 100.0, "train_loss": 5.5},
        {"event": "val", "step": 50, "val_loss": 6.0},
        {"event": "train", "step": 150, "tokens_per_sec": 100.0, "train_loss": 4.5},
        {"event": "val", "step": 150, "val_loss": 5.0},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _write_full_artifact_for_command(cmd: list[str], run_dir: Path) -> None:
    if "scripts/train.py" in cmd:
        (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (run_dir / "checkpoints" / "ckpt_last.pt").write_bytes(b"tiny")
    if "scripts/eval_loss.py" in cmd:
        (run_dir / "evals").mkdir(parents=True, exist_ok=True)
        (run_dir / "evals" / "val_loss.json").write_text(json.dumps({"val_loss": 4.0}), encoding="utf-8")
    if "scripts/eval_hellaswag.py" in cmd:
        (run_dir / "evals").mkdir(parents=True, exist_ok=True)
        (run_dir / "evals" / "hellaswag.json").write_text(json.dumps({"accuracy_norm": 0.25}), encoding="utf-8")
    if "scripts/summarize_run.py" in cmd:
        (run_dir / "evals").mkdir(parents=True, exist_ok=True)
        (run_dir / "evals" / "run_summary.json").write_text(
            json.dumps(
                {
                    "max_step": 2,
                    "final_val_loss": 4.0,
                    "best_val_loss": 3.9,
                    "final_val_perplexity": 54.6,
                    "median_tokens_per_sec": 123.0,
                    "peak_vram_allocated_mb": 456.0,
                }
            ),
            encoding="utf-8",
        )


def _write_hypothesis(path: Path) -> None:
    path.write_text(
        "\n\n".join(f"{field}:\nTest {field.lower()}." for field in REQUIRED_HYPOTHESIS_FIELDS) + "\n",
        encoding="utf-8",
    )
