from __future__ import annotations

import json
import math
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from attention_lab.queue.ledger import QueueLedger
from attention_lab.queue.promotion import load_promotion_report
from attention_lab.queue.runner import CommandRunner, default_command_runner
from attention_lab.queue.screener import run_screen
from attention_lab.training.config import load_config, save_config


@dataclass(frozen=True)
class GauntletRung:
    name: str
    max_steps: int
    val_every: int
    save_every: int
    diagnostics_every: int
    allow_full: bool = False


@dataclass(frozen=True)
class GauntletPolicy:
    experiment_id: str
    control_run_name: str
    base_configs: list[str]
    rungs: list[GauntletRung]
    require_loss_descended: bool = True
    require_checkpoint: bool = True
    require_mechanism_active: bool = True
    require_no_nan_or_inf: bool = True
    max_loss_ratio_vs_control: float = 1.20
    min_speed_ratio_vs_control: float = 0.20
    max_vram_ratio_vs_control: float = 2.75


def load_gauntlet_policy(path: str | Path) -> GauntletPolicy:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Gauntlet policy must be a mapping: {path}")

    rungs_payload = payload.get("rungs")
    if not isinstance(rungs_payload, list) or not rungs_payload:
        raise ValueError("gauntlet policy requires a nonempty rungs list")
    rungs = [
        GauntletRung(
            name=_nonempty_string(rung, "name"),
            max_steps=_positive_int(rung, "max_steps"),
            val_every=_positive_int(rung, "val_every"),
            save_every=_positive_int(rung, "save_every"),
            diagnostics_every=_positive_int(rung, "diagnostics_every"),
            allow_full=bool(rung.get("allow_full", False)),
        )
        for rung in rungs_payload
    ]

    gates = payload.get("gates") or {}
    if not isinstance(gates, dict):
        raise ValueError("gauntlet policy gates must be a mapping")
    base_configs = payload.get("base_configs") or []
    if not isinstance(base_configs, list) or not all(isinstance(item, str) and item.strip() for item in base_configs):
        raise ValueError("gauntlet policy base_configs must be a list of config paths")

    return GauntletPolicy(
        experiment_id=_nonempty_string(payload, "experiment_id"),
        control_run_name=_nonempty_string(payload, "control_run_name"),
        base_configs=list(base_configs),
        rungs=rungs,
        require_loss_descended=bool(gates.get("require_loss_descended", True)),
        require_checkpoint=bool(gates.get("require_checkpoint", True)),
        require_mechanism_active=bool(gates.get("require_mechanism_active", True)),
        require_no_nan_or_inf=bool(gates.get("require_no_nan_or_inf", True)),
        max_loss_ratio_vs_control=float(gates.get("max_loss_ratio_vs_control", 1.20)),
        min_speed_ratio_vs_control=float(gates.get("min_speed_ratio_vs_control", 0.20)),
        max_vram_ratio_vs_control=float(gates.get("max_vram_ratio_vs_control", 2.75)),
    )


def make_rung_config(base_config: dict[str, Any], rung: GauntletRung, *, experiment_id: str) -> dict[str, Any]:
    config = deepcopy(base_config)
    base_name = str(config["run"]["name"])
    rung_name = f"{base_name}_{rung.name}"
    config["run"]["name"] = rung_name
    config["run"]["out_dir"] = f"runs/experiments/{experiment_id}/{rung_name}"
    config["train"]["max_steps"] = rung.max_steps
    config["train"]["val_every"] = rung.val_every
    config["train"]["save_every"] = rung.save_every
    if config["model"].get("attention_type", "standard") != "standard":
        config.setdefault("diagnostics", {})["attention_diagnostics_every"] = rung.diagnostics_every
    queue_config = config.setdefault("queue", {})
    queue_config["screen_steps"] = rung.max_steps
    queue_config["screen_val_every"] = rung.val_every
    queue_config["screen_save_every"] = rung.save_every
    queue_config["screen_diagnostics_every"] = rung.diagnostics_every
    queue_config["family"] = queue_config.get("family", "qkv_gauntlet")
    queue_config["allow_overwrite_existing_run_dir"] = False
    queue_config["full_run_approved"] = False
    return config


def score_promotion_report(
    report: dict[str, Any],
    control_report: dict[str, Any] | None,
    policy: GauntletPolicy,
) -> dict[str, Any]:
    blockers: list[str] = []
    attention_type = str(report.get("attention_type") or "")
    if policy.require_no_nan_or_inf and report.get("nan_or_inf_seen") is True:
        blockers.append("NaN or Inf observed")
    if report.get("max_step_reached") is None or int(report["max_step_reached"]) < int(report.get("expected_screen_steps") or 0):
        blockers.append("expected screen steps were not reached")
    if policy.require_checkpoint and report.get("checkpoint_present") is not True:
        blockers.append("checkpoint is missing")
    if policy.require_loss_descended and report.get("loss_descended") is not True:
        blockers.append("loss did not descend")
    if attention_type != "standard" and policy.require_mechanism_active:
        if report.get("mechanism_active") is not True or report.get("diagnostics_non_degenerate") is not True:
            blockers.append("mechanism diagnostics are missing or degenerate")
    recommendation = report.get("promotion_recommendation")
    if recommendation != "promote":
        blockers.append(f"promotion report recommendation is {recommendation!r}")
    blockers.extend(str(blocker) for blocker in report.get("promotion_blockers") or [])

    final_val_loss = _metric(report, "final_val_loss", "final_screen_val_loss", "final_screen_loss")
    control_val_loss = _metric(control_report, "final_val_loss", "final_screen_val_loss", "final_screen_loss")
    loss_ratio = _ratio(final_val_loss, control_val_loss)
    if loss_ratio is not None and loss_ratio > policy.max_loss_ratio_vs_control:
        blockers.append(f"loss ratio vs control {loss_ratio:.4f} exceeds {policy.max_loss_ratio_vs_control:.4f}")

    speed = _metric(report, "median_tokens_per_sec")
    control_speed = _metric(control_report, "median_tokens_per_sec")
    speed_ratio = _ratio(speed, control_speed)
    if speed_ratio is not None and speed_ratio < policy.min_speed_ratio_vs_control:
        blockers.append(f"speed ratio vs control {speed_ratio:.4f} is below {policy.min_speed_ratio_vs_control:.4f}")

    vram = _metric(report, "peak_vram_allocated_mb", "peak_vram_mb")
    control_vram = _metric(control_report, "peak_vram_allocated_mb", "peak_vram_mb")
    vram_ratio = _ratio(vram, control_vram)
    if vram_ratio is not None and vram_ratio > policy.max_vram_ratio_vs_control:
        blockers.append(f"VRAM ratio vs control {vram_ratio:.4f} exceeds {policy.max_vram_ratio_vs_control:.4f}")

    status = "pass" if not blockers else _blocked_status(recommendation, blockers)
    return {
        "candidate": report.get("run_name"),
        "rung": _rung_from_run_name(str(report.get("run_name") or "")),
        "attention_type": attention_type,
        "status": status,
        "promotion_recommendation": recommendation,
        "machine_decision": "advance" if status == "pass" else status,
        "decision_reason": "passed gauntlet policy" if status == "pass" else "; ".join(dict.fromkeys(blockers)),
        "final_val_loss": final_val_loss,
        "loss_ratio_vs_control": loss_ratio,
        "median_tokens_per_sec": speed,
        "speed_ratio_vs_control": speed_ratio,
        "peak_vram_allocated_mb": vram,
        "vram_ratio_vs_control": vram_ratio,
        "mechanism_check_name": report.get("mechanism_check_name"),
        "mechanism_active": report.get("mechanism_active"),
        "diagnostics_non_degenerate": report.get("diagnostics_non_degenerate"),
        "next_action": "advance_to_next_rung" if status == "pass" else status,
    }


def write_gauntlet_report(
    *,
    experiment_id: str,
    policy_path: str | Path,
    control_run_name: str,
    decisions: list[dict[str, Any]],
    repo_root: str | Path = ".",
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    report_dir = repo_root / "reports" / "experiments" / experiment_id
    report_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment_id": experiment_id,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "policy_path": str(policy_path),
        "control_run_name": control_run_name,
        "decisions": decisions,
    }
    json_path = report_dir / "gauntlet_report.json"
    md_path = report_dir / "gauntlet_report.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(render_gauntlet_markdown(payload), encoding="utf-8")
    return {"json_path": str(json_path), "markdown_path": str(md_path), "report": payload}


def render_gauntlet_markdown(payload: dict[str, Any]) -> str:
    lines = [
        f"# {payload['experiment_id']} Gauntlet Report",
        "",
        f"Created: {payload['created_at']}",
        f"Policy: `{payload['policy_path']}`",
        f"Control: `{payload['control_run_name']}`",
        "",
        "| Candidate | Rung | Attention | Decision | Reason | Next |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in payload.get("decisions", []):
        lines.append(
            "| {candidate} | {rung} | {attention_type} | {machine_decision} | {decision_reason} | {next_action} |".format(
                candidate=row.get("candidate") or "",
                rung=row.get("rung") or "",
                attention_type=row.get("attention_type") or "",
                machine_decision=row.get("machine_decision") or "",
                decision_reason=str(row.get("decision_reason") or "").replace("|", "/"),
                next_action=row.get("next_action") or "",
            )
        )
    return "\n".join(lines) + "\n"


def gauntlet_plan(policy: GauntletPolicy, *, repo_root: str | Path = ".") -> dict[str, Any]:
    repo_root = Path(repo_root)
    entries = []
    missing = []
    for config_path_text in policy.base_configs:
        config_path = _resolve(repo_root, config_path_text)
        if not config_path.exists():
            missing.append(str(config_path))
            continue
        config = load_config(config_path)
        base_name = str(config["run"]["name"])
        entries.append(
            {
                "base_config": str(config_path_text),
                "run_name": base_name,
                "attention_type": config["model"].get("attention_type", "standard"),
                "rung_configs": [
                    str(config_path.parent / f"{base_name}_{rung.name}.yaml") for rung in policy.rungs
                ],
            }
        )
    return {
        "experiment_id": policy.experiment_id,
        "control_run_name": policy.control_run_name,
        "missing_prerequisites": missing,
        "entries": entries,
        "ok": not missing,
    }


def run_gauntlet_once(
    *,
    policy_path: str | Path,
    ledger: QueueLedger,
    repo_root: str | Path = ".",
    command_runner: CommandRunner = default_command_runner,
    allow_full: bool = False,
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    policy = load_gauntlet_policy(policy_path)
    decisions = _load_existing_decisions(policy.experiment_id, repo_root)

    for base_config_path_text in policy.base_configs:
        base_config_path = _resolve(repo_root, base_config_path_text)
        base_config = load_config(base_config_path)
        base_name = str(base_config["run"]["name"])
        for rung_index, rung in enumerate(policy.rungs):
            rung_name = f"{base_name}_{rung.name}"
            row = ledger.get_run(rung_name)
            rung_config_path = base_config_path.parent / f"{rung_name}.yaml"
            if row is None:
                rung_config = make_rung_config(base_config, rung, experiment_id=policy.experiment_id)
                save_config(rung_config, rung_config_path)
                run_id = ledger.enqueue_config(rung_config_path, rung_config, rung_config_path.read_bytes(), stage="SCREEN")
                _write_current_report(policy, policy_path, decisions, repo_root)
                return {"action": "queued", "run_id": run_id, "run_name": rung_name, "config_path": str(rung_config_path)}

            if row["stage"] == "SCREEN" and row["status"] == "PENDING":
                result = run_screen(row, ledger, command_runner=command_runner, repo_root=repo_root)
                _write_current_report(policy, policy_path, decisions, repo_root)
                return {"action": "screened", "run_name": rung_name, **result}

            if row.get("promotion_report_path") and row["stage"] in {"PROMOTION_CANDIDATE", "KILLED"}:
                existing_decision = _decision_for(decisions, rung_name)
                if existing_decision is not None:
                    if (
                        existing_decision.get("status") != "pass"
                        or row["stage"] == "KILLED"
                        or row["status"] in {"FAILED", "KILLED"}
                    ):
                        break
                    continue
                report = _load_report(row["promotion_report_path"], repo_root)
                control_report = _control_report_for_rung(policy, ledger, rung.name, repo_root)
                decision = score_promotion_report(report, control_report, policy)
                decision["candidate"] = rung_name
                decision["rung"] = rung.name
                if decision["status"] == "pass":
                    if rung_index + 1 < len(policy.rungs):
                        decision["next_action"] = f"queue_{policy.rungs[rung_index + 1].name}"
                    else:
                        decision["ready_for_manual_full_promotion"] = not allow_full
                        decision["next_action"] = "ready_for_manual_full_promotion"
                        if allow_full:
                            decision.update(_try_approve_full(base_name, ledger, repo_root))
                decisions.append(decision)
                _write_current_report(policy, policy_path, decisions, repo_root)
                return {"action": "decided", "run_name": rung_name, "decision": decision}

            if row["stage"] == "KILLED" or row["status"] in {"FAILED", "KILLED"}:
                break

    _write_current_report(policy, policy_path, decisions, repo_root)
    return {"action": "blocked_or_complete", "reason": "no pending automatic gauntlet action"}


def run_gauntlet_until_blocked_or_complete(
    *,
    policy_path: str | Path,
    ledger: QueueLedger,
    repo_root: str | Path = ".",
    command_runner: CommandRunner = default_command_runner,
    allow_full: bool = False,
    max_iterations: int = 100,
) -> dict[str, Any]:
    actions = []
    for _ in range(max_iterations):
        result = run_gauntlet_once(
            policy_path=policy_path,
            ledger=ledger,
            repo_root=repo_root,
            command_runner=command_runner,
            allow_full=allow_full,
        )
        actions.append(result)
        if result["action"] == "blocked_or_complete":
            return {"ok": True, "actions": actions}
    return {"ok": False, "actions": actions, "reason": f"stopped after {max_iterations} gauntlet iterations"}


def render_latest_gauntlet_report(*, experiment_id: str, repo_root: str | Path = ".") -> str:
    report_path = Path(repo_root) / "reports" / "experiments" / experiment_id / "gauntlet_report.json"
    if not report_path.exists():
        return f"No gauntlet report found for {experiment_id}: {report_path}\n"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    return render_gauntlet_markdown(payload)


def _write_current_report(
    policy: GauntletPolicy,
    policy_path: str | Path,
    decisions: list[dict[str, Any]],
    repo_root: Path,
) -> None:
    write_gauntlet_report(
        experiment_id=policy.experiment_id,
        policy_path=policy_path,
        control_run_name=policy.control_run_name,
        decisions=decisions,
        repo_root=repo_root,
    )


def _load_existing_decisions(experiment_id: str, repo_root: Path) -> list[dict[str, Any]]:
    path = repo_root / "reports" / "experiments" / experiment_id / "gauntlet_report.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    decisions = payload.get("decisions") or []
    return list(decisions) if isinstance(decisions, list) else []


def _decision_for(decisions: list[dict[str, Any]], candidate: str) -> dict[str, Any] | None:
    for row in decisions:
        if row.get("candidate") == candidate:
            return row
    return None


def _control_report_for_rung(
    policy: GauntletPolicy,
    ledger: QueueLedger,
    rung_name: str,
    repo_root: Path,
) -> dict[str, Any] | None:
    row = ledger.get_run(f"{policy.control_run_name}_{rung_name}")
    if row is None or not row.get("promotion_report_path"):
        return None
    return _load_report(row["promotion_report_path"], repo_root)


def _load_report(path_text: str, repo_root: Path) -> dict[str, Any]:
    path = Path(path_text)
    if not path.is_absolute():
        path = repo_root / path
    return load_promotion_report(path)


def _try_approve_full(base_name: str, ledger: QueueLedger, repo_root: Path) -> dict[str, Any]:
    row = ledger.get_run(base_name)
    if row is None:
        return {
            "next_action": "manual_full_row_required",
            "full_auto_approval": "blocked",
            "full_auto_approval_reason": "base full-run row is not present in the queue ledger",
        }
    try:
        ledger.approve_full_run(row["id"], repo_root=repo_root)
    except Exception as exc:  # noqa: BLE001 - report the safety blocker as data
        return {
            "next_action": "manual_full_approval_blocked",
            "full_auto_approval": "blocked",
            "full_auto_approval_reason": str(exc),
        }
    return {"next_action": "full_run_approved", "full_auto_approval": "approved"}


def _resolve(repo_root: Path, path_text: str | Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return repo_root / path


def _metric(report: dict[str, Any] | None, *keys: str) -> float | None:
    if report is None:
        return None
    for key in keys:
        value = report.get(key)
        if value is None:
            continue
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(parsed):
            return parsed
    return None


def _ratio(value: float | None, control: float | None) -> float | None:
    if value is None or control is None or control <= 0.0:
        return None
    return value / control


def _blocked_status(recommendation: Any, blockers: list[str]) -> str:
    if recommendation == "kill" or any("NaN" in blocker or "loss did not descend" in blocker for blocker in blockers):
        return "killed"
    return "needs_investigation"


def _rung_from_run_name(run_name: str) -> str | None:
    parts = run_name.rsplit("_", 1)
    if len(parts) == 2 and parts[1].startswith("rung"):
        return parts[1]
    return None


def _nonempty_string(mapping: dict[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a nonempty string")
    return value


def _positive_int(mapping: dict[str, Any], key: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{key} must be a positive integer")
    return value
