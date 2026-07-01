from __future__ import annotations

import json
from pathlib import Path

from attention_lab.queue.ledger import QueueLedger
from attention_lab.queue.runner import (
    CommandResult,
    build_full_pipeline,
    capture_git_state,
    classify_failure,
    existing_run_artifacts,
    run_full,
)


def test_full_pipeline_command_contract():
    steps = build_full_pipeline(
        config_path="queue/inbox/candidate.yaml",
        run_dir="runs/experiments/E999/candidate",
        data_root="data/fineweb_edu_100m",
        manifest_path="data/fineweb_edu_100m/manifest.json",
    )
    assert steps[0] == [
        "uv",
        "run",
        "scripts/verify_data.py",
        "--data_root",
        "data/fineweb_edu_100m",
        "--manifest",
        "data/fineweb_edu_100m/manifest.json",
        "--verify_hashes",
    ]
    assert steps[-1][-5:] == [
        "--expect-complete-training",
        "--expect-sample",
        "--expect-eval-loss",
        "--expect-hellaswag",
        "--expect-data-manifest",
    ]
    assert len(steps) == 8


def test_failure_classifier_prefers_known_patterns():
    assert classify_failure(1, "CUDA out of memory") == "OOM"
    assert classify_failure(1, "loss is nan") == "NAN"
    assert classify_failure(1, "verifier failed", verify_step=True) == "VERIFY_FAIL"
    assert classify_failure(1, "anything else") == "UNKNOWN"


def test_capture_git_state_writes_commit_and_diff(tmp_path):
    warning = capture_git_state(tmp_path, repo_root=Path.cwd())
    text = (tmp_path / "git_state.txt").read_text(encoding="utf-8")
    assert "commit:" in text
    assert "diff:" in text
    assert warning is None or "dirty" in warning.lower()


def test_run_full_stops_on_failure_and_updates_ledger(tmp_path, tiny_config):
    ledger = QueueLedger(tmp_path / "queue.db")
    ledger.initialize()
    config = tiny_config(tmp_path, tmp_path / "data")
    config_path = tmp_path / "candidate.yaml"
    import yaml

    content = yaml.safe_dump(config).encode()
    config_path.write_bytes(content)
    run_id = ledger.enqueue_config(config_path, config, content)
    ledger.promote_to_full(run_id)

    calls = []

    def fake_runner(cmd, log_path):
        calls.append(cmd)
        return CommandResult(returncode=1, stdout="", stderr="CUDA out of memory")

    result = run_full(ledger.get_run(run_id), ledger, command_runner=fake_runner, repo_root=Path.cwd())
    row = ledger.get_run(run_id)
    assert result["ok"] is False
    assert row["status"] == "FAILED"
    assert row["failure_class"] == "OOM"
    assert len(calls) == 1


def test_run_full_ingests_summary_and_hellaswag_on_success(tmp_path, tiny_config):
    ledger = QueueLedger(tmp_path / "queue.db")
    ledger.initialize()
    config = tiny_config(tmp_path, tmp_path / "data")
    config["queue"] = {"allow_overwrite_existing_run_dir": True}
    config_path = tmp_path / "candidate.yaml"
    import yaml

    content = yaml.safe_dump(config).encode()
    config_path.write_bytes(content)
    run_id = ledger.enqueue_config(config_path, config, content)
    ledger.promote_to_full(run_id)
    run_dir = Path(config["run"]["out_dir"])
    (run_dir / "evals").mkdir(parents=True)
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "checkpoints" / "ckpt_last.pt").write_bytes(b"tiny")
    (run_dir / "evals" / "val_loss.json").write_text(json.dumps({"val_loss": 4.0}), encoding="utf-8")
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
    (run_dir / "evals" / "hellaswag.json").write_text(json.dumps({"accuracy_norm": 0.25}), encoding="utf-8")

    def fake_runner(cmd, log_path):
        return CommandResult(returncode=0, stdout="ok", stderr="")

    result = run_full(ledger.get_run(run_id), ledger, command_runner=fake_runner, repo_root=Path.cwd())
    row = ledger.get_run(run_id)
    assert result["ok"] is True
    assert row["status"] == "PASSED"
    assert row["final_val_loss"] == 4.0
    assert row["hellaswag_acc"] == 0.25


def test_run_full_refuses_existing_artifacts_without_allow(tmp_path, tiny_config):
    ledger = QueueLedger(tmp_path / "queue.db")
    ledger.initialize()
    config = tiny_config(tmp_path, tmp_path / "data")
    config_path = tmp_path / "candidate.yaml"
    import yaml

    content = yaml.safe_dump(config).encode()
    config_path.write_bytes(content)
    run_id = ledger.enqueue_config(config_path, config, content)
    ledger.promote_to_full(run_id)
    run_dir = Path(config["run"]["out_dir"])
    (run_dir / "checkpoints").mkdir(parents=True)
    assert existing_run_artifacts(run_dir)

    calls = []

    def fake_runner(cmd, log_path):
        calls.append(cmd)
        return CommandResult(returncode=0, stdout="ok", stderr="")

    result = run_full(ledger.get_run(run_id), ledger, command_runner=fake_runner, repo_root=Path.cwd())
    row = ledger.get_run(run_id)
    assert result["ok"] is False
    assert row["failure_class"] == "RUN_DIR_EXISTS"
    assert calls == []


def test_run_full_requires_summary_eval_and_checkpoint_artifacts(tmp_path, tiny_config):
    ledger = QueueLedger(tmp_path / "queue.db")
    ledger.initialize()
    config = tiny_config(tmp_path, tmp_path / "data")
    config_path = tmp_path / "candidate.yaml"
    import yaml

    content = yaml.safe_dump(config).encode()
    config_path.write_bytes(content)
    run_id = ledger.enqueue_config(config_path, config, content)
    ledger.promote_to_full(run_id)

    def fake_runner(cmd, log_path):
        return CommandResult(returncode=0, stdout="ok", stderr="")

    result = run_full(ledger.get_run(run_id), ledger, command_runner=fake_runner, repo_root=Path.cwd())
    row = ledger.get_run(run_id)
    assert result["ok"] is False
    assert row["failure_class"] == "VERIFY_FAIL"
    assert "missing required full-run artifacts" in row["notes"]


def test_run_full_rejects_missing_required_summary_field(tmp_path, tiny_config):
    ledger = QueueLedger(tmp_path / "queue.db")
    ledger.initialize()
    config = tiny_config(tmp_path, tmp_path / "data")
    config["queue"] = {"allow_overwrite_existing_run_dir": True}
    config_path = tmp_path / "candidate.yaml"
    import yaml

    content = yaml.safe_dump(config).encode()
    config_path.write_bytes(content)
    run_id = ledger.enqueue_config(config_path, config, content)
    ledger.promote_to_full(run_id)
    run_dir = Path(config["run"]["out_dir"])
    (run_dir / "evals").mkdir(parents=True)
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "checkpoints" / "ckpt_last.pt").write_bytes(b"tiny")
    (run_dir / "evals" / "val_loss.json").write_text(json.dumps({"val_loss": 4.0}), encoding="utf-8")
    (run_dir / "evals" / "hellaswag.json").write_text(json.dumps({"accuracy_norm": 0.25}), encoding="utf-8")
    (run_dir / "evals" / "run_summary.json").write_text(json.dumps({"max_step": 2}), encoding="utf-8")

    def fake_runner(cmd, log_path):
        return CommandResult(returncode=0, stdout="ok", stderr="")

    result = run_full(ledger.get_run(run_id), ledger, command_runner=fake_runner, repo_root=Path.cwd())
    row = ledger.get_run(run_id)
    assert result["ok"] is False
    assert row["failure_class"] == "VERIFY_FAIL"
    assert "run_summary.json missing required fields" in row["notes"]
