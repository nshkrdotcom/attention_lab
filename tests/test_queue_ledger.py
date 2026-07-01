from __future__ import annotations

from copy import deepcopy

from attention_lab.queue.ledger import QueueLedger, hash_config_bytes
from attention_lab.queue.paths import ensure_queue_dirs
from attention_lab.queue.state_files import clear_active, copy_to_active, finalize_config, move_to_full_pending
from attention_lab.training.config import validate_config


def test_queue_dirs_and_gitignore_contract(repo_root, tmp_path):
    ensure_queue_dirs(tmp_path)
    assert (tmp_path / "queue" / "inbox").is_dir()
    assert (tmp_path / "queue" / "active").is_dir()
    assert (tmp_path / "queue" / "full_pending").is_dir()
    assert (tmp_path / "queue" / "done").is_dir()
    assert (tmp_path / "queue" / "failed").is_dir()

    gitignore = (repo_root / ".gitignore").read_text(encoding="utf-8")
    for entry in (
        "queue/active/",
        "queue/full_pending/",
        "queue/done/",
        "queue/failed/",
        "data/queue.db",
        "data/queue.pid",
    ):
        assert entry in gitignore

    pyproject = (repo_root / "pyproject.toml").read_text(encoding="utf-8")
    assert 'attention-lab-queue = "attention_lab.queue.cli:main"' in pyproject
    assert 'attn-queue = "attention_lab.queue.cli:main"' in pyproject


def test_config_validation_accepts_strict_queue_section(tiny_config, tmp_path):
    config = tiny_config(tmp_path, tmp_path / "data")
    config["queue"] = {
        "requires_run": "standard_30m_seed1",
        "hypothesis_doc": "docs/experiments/E999/hypothesis_candidate.md",
        "skip_hypothesis_check": False,
        "family": "toy_family",
        "full_run_approved": False,
        "allow_overwrite_existing_run_dir": False,
        "mechanism_check": "cp_gradient_norm",
        "allow_missing_diagnostics": False,
        "skip_control_check": False,
    }
    assert validate_config(config)["queue"]["requires_run"] == "standard_30m_seed1"

    bad = deepcopy(config)
    bad["queue"]["requires"] = "typo"
    try:
        validate_config(bad)
    except ValueError as exc:
        assert "Unknown queue config keys" in str(exc)
    else:
        raise AssertionError("unknown queue key was accepted")

    bad_bool = deepcopy(config)
    bad_bool["queue"]["full_run_approved"] = "yes"
    try:
        validate_config(bad_bool)
    except ValueError as exc:
        assert "queue.full_run_approved must be a boolean" in str(exc)
    else:
        raise AssertionError("non-boolean full_run_approved was accepted")

    bad_check = deepcopy(config)
    bad_check["queue"]["mechanism_check"] = "made_up_check"
    try:
        validate_config(bad_check)
    except ValueError as exc:
        assert "queue.mechanism_check" in str(exc)
    else:
        raise AssertionError("unknown mechanism_check was accepted")


def test_ledger_schema_insert_deduplicate_and_list(tmp_path, tiny_config):
    ledger = QueueLedger(tmp_path / "queue.db")
    ledger.initialize()
    config = tiny_config(tmp_path, tmp_path / "data")
    config_path = tmp_path / "candidate.yaml"
    config_text = "run:\n  name: tiny_test_run\n"
    config_path.write_text(config_text, encoding="utf-8")

    run_id = ledger.enqueue_config(config_path, config, config_text.encode())
    duplicate = ledger.enqueue_config(config_path, config, config_text.encode())
    assert run_id == hash_config_bytes(config_text.encode())
    assert duplicate == run_id

    rows = ledger.list_runs(stage="SCREEN", status="PENDING")
    assert len(rows) == 1
    assert rows[0]["id"] == run_id
    assert rows[0]["config_name"] == "candidate"
    assert rows[0]["attention_type"] == "standard"
    assert rows[0]["full_run_approved"] == 0
    assert rows[0]["allow_overwrite_existing_run_dir"] == 0


def test_ledger_approval_and_run_dir_collision(tmp_path, tiny_config):
    ledger = QueueLedger(tmp_path / "queue.db")
    ledger.initialize()
    config = tiny_config(tmp_path, tmp_path / "data")
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    first.write_text("first", encoding="utf-8")
    second.write_text("second", encoding="utf-8")
    run_id = ledger.enqueue_config(first, config, b"first")

    ledger.set_full_run_approved(run_id, True)
    assert ledger.get_run(run_id)["full_run_approved"] == 1
    ledger.set_full_run_approved(run_id, False)
    assert ledger.get_run(run_id)["full_run_approved"] == 0

    try:
        ledger.enqueue_config(second, config, b"second")
    except ValueError as exc:
        assert "run.out_dir collision" in str(exc)
    else:
        raise AssertionError("duplicate run.out_dir was accepted")

    ledger.mark_failed(run_id, failure_class="UNKNOWN", killed=True)
    try:
        ledger.enqueue_config(second, config, b"second")
    except ValueError as exc:
        assert "run.out_dir collision" in str(exc)
    else:
        raise AssertionError("failed run.out_dir reuse was accepted without explicit allow")

    config["queue"] = {"allow_overwrite_existing_run_dir": True}
    second_id = ledger.enqueue_config(second, config, b"second")
    assert second_id != run_id


def test_ledger_notes_requeue_baseline_and_interrupted_reset(tmp_path, tiny_config):
    ledger = QueueLedger(tmp_path / "queue.db")
    ledger.initialize()
    config = tiny_config(tmp_path, tmp_path / "data")
    config_path = tmp_path / "candidate.yaml"
    content = b"candidate"
    config_path.write_bytes(content)
    run_id = ledger.enqueue_config(config_path, config, content)

    ledger.update_notes(run_id, "SHOWS: nothing yet")
    assert ledger.get_run(run_id)["notes"] == "SHOWS: nothing yet"

    ledger.promote_to_full(run_id)
    ledger.mark_failed(run_id, failure_class="NAN", notes="nan observed")
    ledger.requeue(run_id)
    row = ledger.get_run(run_id)
    assert row["stage"] == "FULL"
    assert row["status"] == "PENDING"
    assert row["failure_class"] is None

    ledger.update_baseline_screen_tokens_per_sec(12345.0)
    assert ledger.get_baseline_screen_tokens_per_sec() == 12345.0

    ledger.mark_started(run_id)
    ledger.reset_interrupted()
    assert ledger.get_run(run_id)["status"] == "PENDING"


def test_scan_inbox_validates_and_skips_bad_configs(tmp_path, tiny_config):
    root = tmp_path
    ensure_queue_dirs(root)
    inbox = root / "queue" / "inbox"
    good_config = tiny_config(tmp_path, tmp_path / "data")
    good_path = inbox / "good.yaml"
    bad_path = inbox / "bad.yaml"

    import yaml

    good_path.write_text(yaml.safe_dump(good_config), encoding="utf-8")
    bad_path.write_text("not: [valid", encoding="utf-8")

    ledger = QueueLedger(tmp_path / "queue.db")
    ledger.initialize()
    result = ledger.scan_inbox(inbox)
    assert result["inserted"] == 1
    assert result["skipped"] == 0
    assert len(result["errors"]) == 1
    assert ledger.list_runs(stage="SCREEN", status="PENDING")[0]["config_name"] == "good"


def test_queue_state_file_transitions(tmp_path):
    ensure_queue_dirs(tmp_path)
    config_path = tmp_path / "queue" / "inbox" / "candidate.yaml"
    config_path.write_text("run: {}\n", encoding="utf-8")

    active_path = copy_to_active(config_path)
    assert active_path is not None
    assert active_path.exists()
    clear_active(config_path)
    assert not active_path.exists()

    active_path = copy_to_active(config_path)
    assert active_path is not None
    done_path = finalize_config(config_path, "done")
    assert done_path is not None
    assert done_path.exists()
    assert not active_path.exists()
    assert not config_path.exists()


def test_queue_full_pending_state_transition(tmp_path):
    ensure_queue_dirs(tmp_path)
    config_path = tmp_path / "queue" / "inbox" / "candidate.yaml"
    config_path.write_text("run: {}\n", encoding="utf-8")
    active_path = copy_to_active(config_path)
    assert active_path is not None

    pending_path = move_to_full_pending(config_path)
    assert pending_path is not None
    assert pending_path.exists()
    assert pending_path.parent.name == "full_pending"
    assert not config_path.exists()
    assert not active_path.exists()

    active_path = copy_to_active(pending_path)
    assert active_path is not None
    failed_path = finalize_config(pending_path, "failed")
    assert failed_path is not None
    assert failed_path.exists()
    assert not pending_path.exists()
    assert not active_path.exists()
