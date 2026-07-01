from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from attention_lab.queue.discipline import default_hypothesis_path, validate_hypothesis_doc
from attention_lab.queue.ledger import QueueLedger
from attention_lab.queue.paths import ensure_queue_dirs
from attention_lab.queue.promotion import approval_blockers, load_promotion_report
from attention_lab.training.config import (
    EXPERIMENTAL_UNIMPLEMENTED_STATUS,
    MECHANISM_CHECKS,
    QUEUE_KEYS,
    load_config,
)
from attention_lab.training.experiments import get_experiment, validate_experiment_entry
from attention_lab.training.validate_experiment import load_yaml


@dataclass(frozen=True)
class DoctorMessage:
    level: str
    text: str


@dataclass(frozen=True)
class DoctorReport:
    ok: bool
    messages: list[DoctorMessage]


def run_doctor(
    *,
    experiment_id: str,
    ledger: QueueLedger,
    root: str | Path = ".",
) -> DoctorReport:
    root = Path(root)
    messages: list[DoctorMessage] = []

    try:
        experiment = get_experiment(experiment_id)
        validate_experiment_entry(experiment)
        _ok(messages, f"experiment exists: {experiment_id}")
    except Exception as exc:  # noqa: BLE001 - doctor reports instead of crashing
        _fail(messages, f"experiment lookup failed: {exc}")
        return _report(messages)

    config_dir = root / experiment["config_dir"]
    report_dir = root / experiment["report_dir"]
    dataset_manifest = root / experiment["dataset_manifest"]
    for path, label in (
        (config_dir, "experiment config dir"),
        (report_dir, "experiment report dir"),
        (dataset_manifest, "data manifest"),
    ):
        if path.exists():
            _ok(messages, f"{label} exists: {path}")
        else:
            _fail(messages, f"{label} missing: {path}")

    configs = _load_experiment_configs(config_dir, messages)
    _check_run_dirs(configs, messages)
    _check_configs(configs, messages)
    _check_ledger_approval(configs, ledger, messages)
    _check_queue_state_classification(ledger, messages)
    _check_promotion_state(ledger, messages)
    _check_approved_hypotheses(configs, ledger, messages)

    ensure_queue_dirs(root)
    _ok(messages, "queue dirs exist")
    ledger.initialize()
    _ok(messages, "queue DB initializes")
    _ok(messages, "attn-queue entrypoint is active")
    return _report(messages)


def render_doctor_report(report: DoctorReport) -> str:
    return "\n".join(f"{message.level}: {message.text}" for message in report.messages) + "\n"


def _load_experiment_configs(config_dir: Path, messages: list[DoctorMessage]) -> list[tuple[Path, dict[str, Any]]]:
    configs = []
    if not config_dir.exists():
        return configs
    for config_path in sorted(config_dir.glob("*.yaml")):
        try:
            config = load_yaml(config_path)
        except Exception as exc:  # noqa: BLE001
            _fail(messages, f"config YAML invalid: {config_path}: {exc}")
            continue
        configs.append((config_path, config))
    if configs:
        _ok(messages, f"configs discovered: {len(configs)}")
    else:
        _fail(messages, f"no configs found in {config_dir}")
    return configs


def _check_run_dirs(configs: list[tuple[Path, dict[str, Any]]], messages: list[DoctorMessage]) -> None:
    out_dirs = [str(config.get("run", {}).get("out_dir")) for _, config in configs]
    duplicates = sorted({run_dir for run_dir in out_dirs if out_dirs.count(run_dir) > 1})
    if duplicates:
        _fail(messages, f"duplicate run.out_dir values: {duplicates}")
    elif out_dirs:
        _ok(messages, "all run.out_dir values are unique")


def _check_configs(configs: list[tuple[Path, dict[str, Any]]], messages: list[DoctorMessage]) -> None:
    for config_path, config in configs:
        if config.get("status") == EXPERIMENTAL_UNIMPLEMENTED_STATUS:
            _check_queue_section(config_path, config, messages)
            _ok(messages, f"explicitly experimental/unimplemented: {config_path}")
            continue
        try:
            loaded = load_config(config_path)
        except Exception as exc:  # noqa: BLE001
            _fail(messages, f"runnable config failed validation: {config_path}: {exc}")
            continue
        _ok(messages, f"runnable config validates: {config_path}")
        queue_config = loaded.get("queue", {})
        if queue_config.get("full_run_approved", False):
            _fail(messages, f"full_run_approved defaults true in config: {config_path}")
        attention_type = loaded["model"].get("attention_type", "standard")
        if attention_type != "standard":
            if not queue_config.get("requires_run") and not queue_config.get("skip_control_check", False):
                _fail(messages, f"non-standard config missing queue.requires_run: {config_path}")
            else:
                _ok(messages, f"non-standard control dependency configured: {config_path}")
            if not queue_config.get("mechanism_check"):
                _fail(messages, f"non-standard config missing queue.mechanism_check: {config_path}")
            else:
                _ok(messages, f"mechanism_check configured: {config_path}")
            if loaded.get("diagnostics") or attention_type != "standard":
                _ok(messages, f"screen diagnostics available or injected: {config_path}")


def _check_queue_section(config_path: Path, config: dict[str, Any], messages: list[DoctorMessage]) -> None:
    queue = config.get("queue", {})
    if not isinstance(queue, dict):
        _fail(messages, f"queue section must be a mapping: {config_path}")
        return
    unknown = sorted(set(queue) - QUEUE_KEYS)
    if unknown:
        _fail(messages, f"unknown queue keys in {config_path}: {unknown}")
    for key in (
        "skip_hypothesis_check",
        "full_run_approved",
        "allow_overwrite_existing_run_dir",
        "allow_missing_diagnostics",
        "skip_control_check",
    ):
        if key in queue and not isinstance(queue[key], bool):
            _fail(messages, f"queue.{key} must be boolean in {config_path}")
    if "mechanism_check" in queue and queue["mechanism_check"] not in MECHANISM_CHECKS:
        _fail(messages, f"unknown mechanism_check in {config_path}: {queue['mechanism_check']}")


def _check_ledger_approval(
    configs: list[tuple[Path, dict[str, Any]]],
    ledger: QueueLedger,
    messages: list[DoctorMessage],
) -> None:
    for config_path, config in configs:
        if config.get("status") == EXPERIMENTAL_UNIMPLEMENTED_STATUS:
            continue
        row = _find_ledger_row(config_path, config, ledger)
        if row and row.get("full_run_approved"):
            _ok(messages, f"ledger-approved full run: {config.get('run', {}).get('name')}")
        elif not config.get("queue", {}).get("full_run_approved", False):
            _ok(messages, f"full_run_approved defaults false: {config_path}")


def _check_promotion_state(ledger: QueueLedger, messages: list[DoctorMessage]) -> None:
    for row in ledger.list_runs():
        stage = row.get("stage")
        status = row.get("status")
        report_path = row.get("promotion_report_path")
        if stage == "PROMOTION_CANDIDATE" or (stage == "SCREEN" and status == "PASSED"):
            if not report_path:
                _warn(messages, f"screen passed but promotion report missing: {row.get('config_name')}")
                continue
            report, blockers = _load_report_and_blockers(report_path)
            if report is None:
                _warn(messages, f"promotion report unreadable: {report_path}: {'; '.join(blockers)}")
            elif blockers:
                _warn(messages, f"promotion candidate has blockers: {row.get('config_name')}: {'; '.join(blockers)}")
            else:
                _ok(messages, f"promotion report exists: {report_path}")
        if stage == "FULL" and row.get("full_run_approved"):
            if not report_path:
                _fail(messages, f"full-approved row lacks promotion report: {row.get('config_name')}")
                continue
            report, blockers = _load_report_and_blockers(report_path)
            if report is None:
                _fail(messages, f"full-approved row has unreadable promotion report: {report_path}: {'; '.join(blockers)}")
            elif blockers:
                _fail(messages, f"full-approved row has promotion blockers: {row.get('config_name')}: {'; '.join(blockers)}")
            else:
                _ok(messages, f"full-approved row has promotion evidence: {row.get('config_name')}")


def _check_queue_state_classification(ledger: QueueLedger, messages: list[DoctorMessage]) -> None:
    for row in ledger.list_runs():
        stage = row.get("stage")
        status = row.get("status")
        name = row.get("config_name")
        report_path = row.get("promotion_report_path")
        if stage == "SCREEN" and status == "PENDING":
            _ok(messages, f"screen-ready: {name}")
        elif stage == "SCREEN" and status in {"PASSED", "FAILED", "KILLED"}:
            _ok(messages, f"screen-complete: {name} status={status}")
        elif stage == "PROMOTION_CANDIDATE":
            if not report_path:
                _warn(messages, f"promotion-blocked: {name}: missing promotion report")
                continue
            _report, blockers = _load_report_and_blockers(report_path)
            if blockers:
                _warn(messages, f"promotion-blocked: {name}: {'; '.join(blockers)}")
            else:
                _ok(messages, f"promotion-candidate: {name}")
        elif stage == "FULL" and row.get("full_run_approved"):
            if status == "PASSED":
                _ok(messages, f"full-complete: {name}")
            else:
                _ok(messages, f"full-approved: {name} status={status}")
        elif stage == "KILLED" or status == "KILLED":
            reason = row.get("kill_reason") or row.get("failure_class") or "unspecified"
            _ok(messages, f"killed: {name}: {reason}")


def _load_report_and_blockers(report_path: str | Path) -> tuple[dict[str, Any] | None, list[str]]:
    try:
        report = load_promotion_report(report_path)
    except Exception as exc:  # noqa: BLE001 - doctor reports all promotion issues
        return None, [str(exc)]
    return report, approval_blockers(report)


def _check_approved_hypotheses(
    configs: list[tuple[Path, dict[str, Any]]],
    ledger: QueueLedger,
    messages: list[DoctorMessage],
) -> None:
    for config_path, config in configs:
        row = _find_ledger_row(config_path, config, ledger)
        if not row or row.get("stage") != "FULL" or not row.get("full_run_approved"):
            continue
        hypothesis_path = default_hypothesis_path(config_path, config)
        hypothesis = validate_hypothesis_doc(hypothesis_path)
        if hypothesis.ok:
            _ok(messages, f"approved FULL hypothesis exists: {hypothesis.path}")
        else:
            _warn(messages, f"approved FULL row missing hypothesis doc: {hypothesis.path}")


def _find_ledger_row(config_path: Path, config: dict[str, Any], ledger: QueueLedger) -> dict[str, Any] | None:
    candidates = [
        str(config.get("run", {}).get("name") or ""),
        config_path.stem,
    ]
    for candidate in candidates:
        if not candidate:
            continue
        row = ledger.get_run(candidate)
        if row is not None:
            return row
    run_dir = str(config.get("run", {}).get("out_dir") or "")
    for row in ledger.list_runs():
        if row.get("run_dir") == run_dir:
            return row
    return None


def _ok(messages: list[DoctorMessage], text: str) -> None:
    messages.append(DoctorMessage("OK", text))


def _warn(messages: list[DoctorMessage], text: str) -> None:
    messages.append(DoctorMessage("WARN", text))


def _fail(messages: list[DoctorMessage], text: str) -> None:
    messages.append(DoctorMessage("FAIL", text))


def _report(messages: list[DoctorMessage]) -> DoctorReport:
    return DoctorReport(ok=not any(message.level == "FAIL" for message in messages), messages=messages)
