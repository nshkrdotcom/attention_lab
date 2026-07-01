from __future__ import annotations

import hashlib
import json
import math
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from attention_lab.queue.mechanism_checks import (
    DEFAULT_ACTIVITY_THRESHOLD,
    evaluate_mechanism_activity,
    load_diagnostic_rows,
    mechanism_check_name,
)
from attention_lab.training.experiments import list_experiments

PROMOTION_SCHEMA_VERSION = 1
PROMOTION_RECOMMENDATIONS = {"promote", "kill", "needs_investigation"}
QKV_FAMILY_PREFIXES = ("multi_qkv_", "qkv_shift_")


@dataclass(frozen=True)
class ScreenProfile:
    max_steps: int = 150
    val_every: int = 50
    save_every: int = 150
    diagnostics_every_nonstandard: int = 50


@dataclass(frozen=True)
class PromotionDecision:
    recommendation: str
    blockers: list[str]
    reason: str


DEFAULT_SCREEN_PROFILE = ScreenProfile()

REQUIRED_PROMOTION_REPORT_FIELDS = {
    "schema_version",
    "experiment_id",
    "run_id",
    "run_name",
    "config_path",
    "screen_config_path",
    "resolved_config_path",
    "screen_run_dir",
    "source_config_hash",
    "attention_type",
    "stage",
    "max_step_reached",
    "expected_screen_steps",
    "loss_descended",
    "nan_or_inf_seen",
    "final_screen_train_loss",
    "final_screen_val_loss",
    "final_screen_loss",
    "first_val_loss",
    "final_val_loss",
    "median_tokens_per_sec",
    "peak_vram_mb",
    "peak_vram_allocated_mb",
    "diagnostics_present",
    "diagnostics_non_degenerate",
    "mechanism_check_name",
    "mechanism_check_passed",
    "mechanism_active",
    "mechanism_activity_summary",
    "checkpoint_present",
    "train_points_seen",
    "eval_points_seen",
    "destructive_test_present",
    "destructive_test_effect_summary",
    "destructive_test_command_failed",
    "destructive_test_failure_summary",
    "promotion_recommendation",
    "promotion_blockers",
    "promotion_reason",
    "created_at",
    "source_git_commit",
    "source_dirty",
}


def resolve_screen_profile(queue_config: dict[str, Any] | None = None) -> ScreenProfile:
    queue_config = queue_config or {}
    max_steps = _positive_int(queue_config.get("screen_steps"), DEFAULT_SCREEN_PROFILE.max_steps)
    return ScreenProfile(
        max_steps=max_steps,
        val_every=_positive_int(queue_config.get("screen_val_every"), DEFAULT_SCREEN_PROFILE.val_every),
        save_every=_positive_int(queue_config.get("screen_save_every"), max_steps),
        diagnostics_every_nonstandard=_positive_int(
            queue_config.get("screen_diagnostics_every"),
            DEFAULT_SCREEN_PROFILE.diagnostics_every_nonstandard,
        ),
    )


def is_multi_qkv_attention_type(attention_type: str) -> bool:
    return attention_type.startswith(QKV_FAMILY_PREFIXES)


def build_screen_destructive_test_command(
    *,
    screen_config_path: str | Path,
    checkpoint_path: str | Path,
    out_path: str | Path,
    num_batches: int = 1,
) -> list[str]:
    return [
        "uv",
        "run",
        "scripts/qkv_track_destructive_test.py",
        "--config",
        str(screen_config_path),
        "--checkpoint",
        str(checkpoint_path),
        "--out",
        str(out_path),
        "--num-batches",
        str(num_batches),
    ]


def build_promotion_report(
    *,
    row: dict[str, Any],
    config: dict[str, Any],
    screen_run_dir: Path,
    repo_root: str | Path = ".",
    screen_config_path: str | Path | None = None,
    failure_class: str | None = None,
    mechanism_active: bool | None = None,
) -> dict[str, Any]:
    repo_root = Path(repo_root)
    screen_run_dir = Path(screen_run_dir)
    screen_config_path = Path(screen_config_path) if screen_config_path is not None else screen_run_dir / "screen_config.yaml"
    resolved_config_path = screen_run_dir / "resolved_config.yaml"
    metrics_path = screen_run_dir / "metrics.jsonl"
    diagnostics_path = screen_run_dir / "evals" / "attention_diagnostics.jsonl"
    destructive_path = screen_run_dir / "evals" / "qkv_track_destructive_test.json"
    destructive_error_path = screen_run_dir / "evals" / "qkv_track_destructive_test_error.json"
    checkpoint_path = screen_run_dir / "checkpoints" / "ckpt_last.pt"
    metrics = load_metrics(metrics_path)
    attention_type = config["model"].get("attention_type", "standard")
    queue_config = config.get("queue", {})
    expected_steps = _expected_screen_steps(screen_config_path, queue_config)

    val_losses = _val_losses(metrics)
    train_losses = _train_losses(metrics)
    final_train_loss = train_losses[-1][1] if train_losses else None
    first_val_loss = val_losses[0][1] if val_losses else None
    final_val_loss = val_losses[-1][1] if val_losses else None
    final_screen_loss = final_val_loss if final_val_loss is not None else final_train_loss
    loss_descended = _loss_descended(val_losses, train_losses)
    nan_or_inf_seen = _nan_or_inf_seen(metrics)
    max_step = step_reached(metrics)
    tokens_per_sec = [
        float(row["tokens_per_sec"])
        for row in metrics
        if row.get("event") == "train" and row.get("tokens_per_sec") is not None
    ]
    peak_vram_mb = _max_metric(metrics, "peak_vram_mb")
    peak_vram_allocated_mb = _max_metric(metrics, "peak_vram_allocated_mb")

    diagnostics_rows = load_diagnostic_rows(diagnostics_path)
    check_name = mechanism_check_name(attention_type, queue_config)
    mechanism_result = (
        evaluate_mechanism_activity(
            attention_type=attention_type,
            diagnostics_path=diagnostics_path,
            queue_config=queue_config,
        )
        if check_name is not None
        else None
    )
    resolved_mechanism_active = (
        mechanism_active if mechanism_active is not None else mechanism_result.active if mechanism_result is not None else None
    )
    missing_diagnostics_allowed = bool(queue_config.get("allow_missing_diagnostics", False))
    diagnostics_non_degenerate = bool(mechanism_result and mechanism_result.passed)
    if attention_type == "standard":
        diagnostics_non_degenerate = False

    destructive_summary = _destructive_test_summary(
        destructive_path=destructive_path,
        destructive_error_path=destructive_error_path,
        checkpoint_path=checkpoint_path,
        attention_type=attention_type,
    )
    decision = decide_promotion(
        attention_type=attention_type,
        expected_screen_steps=expected_steps,
        max_step_reached=max_step,
        loss_descended=loss_descended,
        nan_or_inf_seen=nan_or_inf_seen,
        diagnostics_non_degenerate=diagnostics_non_degenerate,
        mechanism_active=resolved_mechanism_active,
        missing_diagnostics_allowed=missing_diagnostics_allowed,
        destructive_test_present=destructive_summary["present"],
        destructive_test_effect_summary=destructive_summary["summary"],
        checkpoint_present=checkpoint_path.exists(),
        failure_class=failure_class or row.get("failure_class"),
    )
    experiment_id = resolve_experiment_id(
        config_path=row.get("config_path"),
        run_dir=config["run"].get("out_dir"),
        repo_root=repo_root,
    )
    commit, dirty = _source_state(repo_root)
    report = {
        "schema_version": PROMOTION_SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "run_id": row["id"],
        "run_name": config["run"].get("name") or row.get("config_name"),
        "config_path": str(row.get("config_path") or ""),
        "screen_config_path": str(screen_config_path),
        "resolved_config_path": str(resolved_config_path if resolved_config_path.exists() else screen_config_path),
        "screen_run_dir": str(screen_run_dir),
        "source_config_hash": _file_sha256(row.get("config_path"), repo_root),
        "artifact_paths": {
            "metrics": str(metrics_path) if metrics_path.exists() else None,
            "attention_diagnostics": str(diagnostics_path) if diagnostics_path.exists() else None,
            "destructive_test": str(destructive_path) if destructive_path.exists() else None,
            "destructive_test_error": str(destructive_error_path) if destructive_error_path.exists() else None,
            "checkpoint": str(checkpoint_path) if checkpoint_path.exists() else None,
            "promotion_report": str(screen_run_dir / "promotion_report.json"),
        },
        "attention_type": attention_type,
        "stage": "SCREEN",
        "max_step_reached": max_step,
        "expected_screen_steps": expected_steps,
        "loss_descended": loss_descended,
        "nan_or_inf_seen": nan_or_inf_seen,
        "final_screen_train_loss": final_train_loss,
        "final_screen_val_loss": final_val_loss,
        "final_screen_loss": final_screen_loss,
        "first_val_loss": first_val_loss,
        "final_val_loss": final_val_loss,
        "median_tokens_per_sec": _median(tokens_per_sec),
        "peak_vram_mb": peak_vram_mb,
        "peak_vram_allocated_mb": peak_vram_allocated_mb,
        "diagnostics_present": bool(diagnostics_rows),
        "diagnostics_non_degenerate": diagnostics_non_degenerate,
        "mechanism_check_name": check_name,
        "mechanism_check_passed": bool(mechanism_result and mechanism_result.passed) if mechanism_result is not None else None,
        "mechanism_active": resolved_mechanism_active,
        "mechanism_activity_summary": _mechanism_activity_summary(
            attention_type=attention_type,
            rows=diagnostics_rows,
            mechanism_result=mechanism_result,
            missing_diagnostics_allowed=missing_diagnostics_allowed,
        ),
        "checkpoint_present": checkpoint_path.exists(),
        "train_points_seen": len(train_losses),
        "eval_points_seen": len(val_losses),
        "destructive_test_present": destructive_summary["present"],
        "destructive_test_effect_summary": destructive_summary["summary"],
        "destructive_test_command_failed": destructive_summary["command_failed"],
        "destructive_test_failure_summary": destructive_summary["failure_summary"],
        "promotion_recommendation": decision.recommendation,
        "promotion_blockers": decision.blockers,
        "promotion_reason": decision.reason,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "source_git_commit": commit,
        "source_dirty": dirty,
    }
    return report


def decide_promotion(
    *,
    attention_type: str,
    expected_screen_steps: int,
    max_step_reached: int | None,
    loss_descended: bool,
    nan_or_inf_seen: bool,
    diagnostics_non_degenerate: bool,
    mechanism_active: bool | None,
    missing_diagnostics_allowed: bool,
    destructive_test_present: bool,
    destructive_test_effect_summary: dict[str, Any] | None,
    checkpoint_present: bool,
    failure_class: str | None = None,
) -> PromotionDecision:
    blockers: list[str] = []
    if nan_or_inf_seen or failure_class == "NAN":
        return PromotionDecision("kill", ["NaN or Inf observed during screen"], "screen was numerically unstable")
    if failure_class == "OOM":
        return PromotionDecision("kill", ["screen hit OOM"], "screen did not fit the available device")
    if failure_class == "COMPILE_ERROR":
        return PromotionDecision("kill", ["screen failed before useful metrics"], "candidate did not compile or start")
    if failure_class == "SLOW":
        blockers.append("screen throughput was below the configured baseline threshold")
    if max_step_reached is None or max_step_reached < expected_screen_steps:
        blockers.append(f"screen did not reach expected step {expected_screen_steps}")
    if not checkpoint_present:
        return PromotionDecision("kill", ["final screen checkpoint is missing"], "screen did not preserve a checkpoint")
    if not loss_descended:
        blockers.append("loss did not descend during the screen")
    if attention_type != "standard":
        if mechanism_active is False:
            return PromotionDecision("kill", ["mechanism diagnostics were degenerate"], "mechanism check failed")
        if mechanism_active is None:
            if missing_diagnostics_allowed:
                blockers.append(
                    "non-standard attention is missing diagnostics under queue.allow_missing_diagnostics"
                )
            else:
                blockers.append("non-standard attention is missing required diagnostics")
        if not diagnostics_non_degenerate:
            if missing_diagnostics_allowed:
                blockers.append(
                    "queue.allow_missing_diagnostics exception requires human investigation"
                )
            else:
                blockers.append("non-standard attention lacks non-degenerate mechanism diagnostics")
    if is_multi_qkv_attention_type(attention_type):
        destructive_status = (destructive_test_effect_summary or {}).get("status")
        destructive_passed = (destructive_test_effect_summary or {}).get("destructive_test_passed")
        if destructive_test_present and destructive_passed is not True:
            blockers.append("Multi-QKV screen destructive test did not pass")
        elif not destructive_test_present and destructive_status != "not_feasible_for_screen":
            blockers.append("Multi-QKV screen is missing destructive-test evidence")

    if blockers:
        if failure_class == "SLOW":
            return PromotionDecision("needs_investigation", blockers, "screen needs throughput review before promotion")
        if any("loss did not descend" in blocker for blocker in blockers):
            return PromotionDecision("kill", blockers, "screen did not show early learning")
        return PromotionDecision("needs_investigation", blockers, "screen needs review before promotion")
    return PromotionDecision("promote", [], "screen passed stability, loss, and mechanism gates")


def write_promotion_report(report: dict[str, Any], repo_root: str | Path = ".") -> Path:
    errors = validate_promotion_report(report)
    if errors:
        raise ValueError(f"invalid promotion report: {errors}")
    repo_root = Path(repo_root)
    experiment_id = report.get("experiment_id")
    if experiment_id:
        report_dir = repo_root / "reports" / "experiments" / str(experiment_id) / "promotion"
    else:
        report_dir = repo_root / "reports" / "promotion"
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / f"{_safe_filename(report['run_name'])}_{_safe_filename(report['run_id'])}.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    screen_report_dir = Path(str(report["screen_run_dir"]))
    if not screen_report_dir.is_absolute():
        screen_report_dir = repo_root / screen_report_dir
    screen_report_path = screen_report_dir / "promotion_report.json"
    screen_report_path.parent.mkdir(parents=True, exist_ok=True)
    screen_report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load_promotion_report(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_promotion_report(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    missing = sorted(REQUIRED_PROMOTION_REPORT_FIELDS - set(report))
    if missing:
        errors.append(f"missing required fields: {missing}")
    if report.get("schema_version") != PROMOTION_SCHEMA_VERSION:
        errors.append(f"schema_version must be {PROMOTION_SCHEMA_VERSION}")
    recommendation = report.get("promotion_recommendation")
    if recommendation not in PROMOTION_RECOMMENDATIONS:
        errors.append(f"promotion_recommendation must be one of {sorted(PROMOTION_RECOMMENDATIONS)}")
    if not isinstance(report.get("promotion_blockers", []), list):
        errors.append("promotion_blockers must be a list")
    if report.get("stage") != "SCREEN":
        errors.append("stage must be SCREEN")
    for key in (
        "run_id",
        "run_name",
        "config_path",
        "screen_config_path",
        "resolved_config_path",
        "screen_run_dir",
        "attention_type",
    ):
        if key in report and (not isinstance(report[key], str) or not report[key].strip()):
            errors.append(f"{key} must be a nonempty string")
    for key in (
        "loss_descended",
        "nan_or_inf_seen",
        "diagnostics_present",
        "diagnostics_non_degenerate",
        "checkpoint_present",
    ):
        if key in report and not isinstance(report[key], bool):
            errors.append(f"{key} must be a boolean")
    if "destructive_test_present" in report and not isinstance(report["destructive_test_present"], bool):
        errors.append("destructive_test_present must be a boolean")
    if "destructive_test_command_failed" in report and not isinstance(
        report["destructive_test_command_failed"], bool
    ):
        errors.append("destructive_test_command_failed must be a boolean")
    return errors


def approval_blockers(report: dict[str, Any]) -> list[str]:
    blockers = validate_promotion_report(report)
    if report.get("promotion_recommendation") != "promote":
        blockers.append(f"promotion_recommendation is {report.get('promotion_recommendation')!r}, not 'promote'")
    blockers.extend(str(item) for item in report.get("promotion_blockers") or [])
    if report.get("checkpoint_present") is not True:
        blockers.append("final screen checkpoint is missing")
    attention_type = str(report.get("attention_type") or "")
    if attention_type != "standard" and not bool(report.get("diagnostics_non_degenerate")):
        blockers.append("non-standard attention lacks non-degenerate diagnostics")
    if is_multi_qkv_attention_type(attention_type):
        if report.get("destructive_test_command_failed"):
            summary = report.get("destructive_test_failure_summary") or {}
            blockers.append(f"Multi-QKV destructive test command failed: {summary.get('reason') or 'see screen log'}")
        summary = report.get("destructive_test_effect_summary") or {}
        if report.get("destructive_test_present"):
            if summary.get("destructive_test_passed") is not True:
                blockers.append("Multi-QKV destructive test did not pass")
        elif summary.get("status") != "not_feasible_for_screen" or not summary.get("reason"):
            blockers.append("Multi-QKV destructive test is missing without a not_feasible_for_screen reason")
    return blockers


def load_metrics(metrics_path: str | Path) -> list[dict[str, Any]]:
    metrics_path = Path(metrics_path)
    if not metrics_path.exists():
        return []
    rows = []
    with metrics_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def step_reached(metrics: list[dict[str, Any]]) -> int | None:
    steps = [int(row["step"]) for row in metrics if row.get("step") is not None]
    return max(steps) if steps else None


def resolve_experiment_id(
    *,
    config_path: str | Path | None,
    run_dir: str | Path | None,
    repo_root: str | Path = ".",
) -> str | None:
    repo_root = Path(repo_root)
    config_text = _repo_relative_text(config_path, repo_root)
    run_text = _repo_relative_text(run_dir, repo_root)
    try:
        experiments = list_experiments(repo_root / "docs" / "experiments" / "experiments.yaml")
    except Exception:  # noqa: BLE001 - report generation should still work outside registered experiments
        return None
    for experiment in experiments:
        config_dir = str(experiment.get("config_dir") or "")
        exp_run_dir = str(experiment.get("run_dir") or "")
        if config_text and config_text.startswith(config_dir):
            return str(experiment["id"])
        if run_text and run_text.startswith(exp_run_dir):
            return str(experiment["id"])
    return None


def _expected_screen_steps(screen_config_path: Path, queue_config: dict[str, Any]) -> int:
    if screen_config_path.exists():
        try:
            import yaml

            with screen_config_path.open("r", encoding="utf-8") as f:
                screen_config = yaml.safe_load(f) or {}
            max_steps = screen_config.get("train", {}).get("max_steps")
            if max_steps is not None:
                return int(max_steps)
        except Exception:  # noqa: BLE001 - fall back to profile default
            pass
    return resolve_screen_profile(queue_config).max_steps


def _mechanism_activity_summary(
    *,
    attention_type: str,
    rows: list[dict[str, Any]],
    mechanism_result: Any,
    missing_diagnostics_allowed: bool,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "rows_seen": len(rows),
        "missing_diagnostics_allowed": missing_diagnostics_allowed,
    }
    if mechanism_result is not None:
        summary["check_note"] = mechanism_result.note
        summary.update(mechanism_result.details)
    if attention_type in {"cp_bilinear", "cp_trilinear"}:
        values = [float(row["cp_gradient_norm"]) for row in rows if row.get("cp_gradient_norm") is not None]
        summary.update(
            {
                "cp_gradient_norm_seen": bool(values),
                "cp_gradient_norm_max": max(values) if values else None,
                "cp_gradient_norm_exceeded_threshold": (
                    max(values) > DEFAULT_ACTIVITY_THRESHOLD if values else False
                ),
            }
        )
    if is_multi_qkv_attention_type(attention_type):
        summary.setdefault(
            "route_formula_seen",
            sorted({str(row.get("route_formula")) for row in rows if row.get("route_formula") is not None}),
        )
        summary.setdefault(
            "tracks_with_nonzero_gradients",
            sorted(
                {
                    str(track)
                    for row in rows
                    for track, value in _track_dict(row.get("per_track_gradient_norm")).items()
                    if float(value) > DEFAULT_ACTIVITY_THRESHOLD
                }
            ),
        )
        summary.setdefault(
            "active_tracks_seen",
            sorted({str(row.get("active_track_index")) for row in rows if row.get("active_track_index") is not None}),
        )
    return summary


def _destructive_test_summary(
    *,
    destructive_path: Path,
    destructive_error_path: Path,
    checkpoint_path: Path,
    attention_type: str,
) -> dict[str, Any]:
    if destructive_path.exists():
        payload = json.loads(destructive_path.read_text(encoding="utf-8"))
        perturbations = payload.get("perturbations") or []
        return {
            "present": True,
            "command_failed": False,
            "failure_summary": None,
            "summary": {
                "destructive_test_passed": payload.get("destructive_test_passed"),
                "max_loss_delta": _max_abs([row.get("loss_delta") for row in perturbations]),
                "max_logit_delta": _max_abs([row.get("max_abs_logit_delta") for row in perturbations]),
                "perturbation_count": len(perturbations),
            },
        }
    if destructive_error_path.exists():
        payload = json.loads(destructive_error_path.read_text(encoding="utf-8"))
        failure_summary = {
            "status": "failed",
            "reason": "screen destructive test command failed",
            "returncode": payload.get("returncode"),
            "stderr_tail": payload.get("stderr_tail"),
        }
        return {
            "present": False,
            "command_failed": True,
            "failure_summary": failure_summary,
            "summary": failure_summary,
        }
    if is_multi_qkv_attention_type(attention_type):
        if not checkpoint_path.exists():
            reason = "screen checkpoint is missing"
        else:
            reason = "screen destructive test artifact was not generated"
        return {
            "present": False,
            "command_failed": False,
            "failure_summary": None,
            "summary": {
                "status": "not_feasible_for_screen" if not checkpoint_path.exists() else "missing",
                "reason": reason,
            },
        }
    return {"present": False, "command_failed": False, "failure_summary": None, "summary": None}


def _val_losses(metrics: list[dict[str, Any]]) -> list[tuple[int, float]]:
    return [
        (int(row.get("step", 0)), float(row["val_loss"]))
        for row in metrics
        if row.get("event") == "val" and row.get("val_loss") is not None
    ]


def _train_losses(metrics: list[dict[str, Any]]) -> list[tuple[int, float]]:
    return [
        (int(row.get("step", 0)), float(row["train_loss"]))
        for row in metrics
        if row.get("event") == "train" and row.get("train_loss") is not None
    ]


def _loss_descended(val_losses: list[tuple[int, float]], train_losses: list[tuple[int, float]]) -> bool:
    losses = val_losses if len(val_losses) >= 2 else train_losses
    if len(losses) < 2:
        return False
    early = next((loss for step, loss in losses if step >= 10), losses[0][1])
    final = losses[-1][1]
    return math.isfinite(early) and math.isfinite(final) and final <= early * 0.97


def _nan_or_inf_seen(metrics: list[dict[str, Any]]) -> bool:
    for row in metrics:
        for key in ("train_loss", "val_loss", "loss"):
            if row.get(key) is not None and not math.isfinite(float(row[key])):
                return True
    return False


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    values = sorted(values)
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2.0


def _max_metric(metrics: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in metrics if row.get(key) is not None]
    return max(values) if values else None


def _max_abs(values: list[Any]) -> float | None:
    numbers = [abs(float(value)) for value in values if value is not None]
    return max(numbers) if numbers else None


def _positive_int(value: Any, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        raise ValueError("screen profile values must be positive integers")
    parsed = int(value)
    if parsed <= 0:
        raise ValueError("screen profile values must be positive integers")
    return parsed


def _track_dict(value: Any) -> dict[str, float]:
    if isinstance(value, dict):
        return {str(key): float(item or 0.0) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return {str(index): float(item or 0.0) for index, item in enumerate(value)}
    return {}


def _source_state(repo_root: Path) -> tuple[str | None, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    dirty = subprocess.run(
        ["git", "diff", "--quiet", "HEAD"],
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    commit_text = commit.stdout.strip() if commit.returncode == 0 else None
    return commit_text, dirty.returncode != 0


def _repo_relative_text(path: str | Path | None, repo_root: Path) -> str:
    if path is None:
        return ""
    path = Path(path)
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except Exception:  # noqa: BLE001
        return path.as_posix()


def _safe_filename(value: Any) -> str:
    text = str(value)
    return "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in text)


def _file_sha256(path: Any, repo_root: Path) -> str | None:
    if not path:
        return None
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = repo_root / config_path
    if not config_path.exists():
        return None
    return hashlib.sha256(config_path.read_bytes()).hexdigest()
