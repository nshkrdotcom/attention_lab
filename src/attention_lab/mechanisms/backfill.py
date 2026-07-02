from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import yaml

from attention_lab.mechanisms.hook_sites import get_hook_site_specs, supported_site_names, unsupported_site_names


EXPERIMENT_ALIASES = {
    "E001": "E001_cp_trilinear_attention",
    "E002": "E002_multitrack_qkv_shift_register",
    "E003": "E003_qkv_architecture_gauntlet",
    "E004": "E004_operator_binding_qkv_gauntlet",
}


def load_registered_experiments(repo_root: Path) -> dict[str, dict[str, Any]]:
    path = repo_root / "docs" / "experiments" / "experiments.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {item["id"]: item for item in data.get("experiments", [])}


def resolve_experiment_ids(values: list[str], repo_root: Path) -> list[str]:
    registered = load_registered_experiments(repo_root)
    resolved = []
    for value in values:
        for part in value.split(","):
            key = part.strip()
            if not key:
                continue
            experiment_id = EXPERIMENT_ALIASES.get(key, key)
            if experiment_id not in registered:
                raise ValueError(f"unknown experiment id: {key}")
            resolved.append(experiment_id)
    return resolved


def build_registered_experiment_inventory(experiment_id: str, repo_root: Path) -> dict[str, Any]:
    registered = load_registered_experiments(repo_root)[experiment_id]
    return build_experiment_inventory(
        experiment_id=experiment_id,
        repo_root=repo_root,
        config_dir=repo_root / registered["config_dir"],
        report_dir=repo_root / registered["report_dir"],
        run_dir=repo_root / registered["run_dir"],
    )


def build_experiment_inventory(
    *,
    experiment_id: str,
    repo_root: Path,
    config_dir: Path,
    report_dir: Path,
    run_dir: Path,
) -> dict[str, Any]:
    promotion_reports = _load_promotion_reports(report_dir)
    gauntlet_decisions = _load_gauntlet_decisions(report_dir)
    candidates = []
    for config_path in sorted(config_dir.glob("*.yaml")):
        if config_path.name == "gauntlet_policy.yaml":
            continue
        raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        run = raw_config.get("run", {})
        model = raw_config.get("model", {})
        run_name = run.get("name", config_path.stem)
        attention_type = model.get("attention_type", "unknown")
        artifact_dirs = _candidate_artifact_dirs(repo_root, run.get("out_dir"), run_dir, run_name, promotion_reports)
        checkpoint_path = _first_existing([path / "checkpoints" / "ckpt_last.pt" for path in artifact_dirs])
        diagnostics_path = _first_existing([path / "evals" / "attention_diagnostics.jsonl" for path in artifact_dirs])
        run_summary_path = _first_existing([path / "evals" / "run_summary.json" for path in artifact_dirs])
        promotion = promotion_reports.get(run_name, {})
        gauntlet = gauntlet_decisions.get(run_name, {})
        checkpoint_status = "available" if checkpoint_path else "checkpoint_unavailable"
        diagnostics_status = _diagnostics_status(attention_type, diagnostics_path)
        run_summary_status = "available" if run_summary_path else "missing"
        supported, unsupported = _hook_sites_for(attention_type)
        evidence_level = _evidence_level(checkpoint_path, diagnostics_path, run_summary_path, promotion, gauntlet)
        notes = _notes(checkpoint_status, diagnostics_status, run_summary_status)
        candidates.append(
            {
                "experiment_id": experiment_id,
                "run_name": run_name,
                "attention_type": attention_type,
                "config_path": _rel(repo_root, config_path),
                "artifact_dirs": [_rel(repo_root, path) for path in artifact_dirs if path.exists()],
                "checkpoint_path": _rel(repo_root, checkpoint_path) if checkpoint_path else None,
                "checkpoint_status": checkpoint_status,
                "diagnostics_path": _rel(repo_root, diagnostics_path) if diagnostics_path else None,
                "diagnostics_status": diagnostics_status,
                "run_summary_path": _rel(repo_root, run_summary_path) if run_summary_path else None,
                "run_summary_status": run_summary_status,
                "gauntlet_status": _gauntlet_status(gauntlet),
                "promotion_status": promotion.get("promotion_recommendation", "missing"),
                "supported_hook_sites": supported,
                "unsupported_hook_sites": unsupported,
                "posthoc_probe_status": "checkpoint_recompute_available" if checkpoint_path else "checkpoint_unavailable",
                "evidence_level": evidence_level,
                "notes": "; ".join(notes) if notes else "",
            }
        )
    return {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "config_dir": _rel(repo_root, config_dir),
        "report_dir": _rel(repo_root, report_dir),
        "run_dir": _rel(repo_root, run_dir),
        "candidates": candidates,
    }


def write_backfill_outputs(inventory: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "inventory.json").write_text(json.dumps(inventory, indent=2, sort_keys=True), encoding="utf-8")
    _write_inventory_md(inventory, out_dir / "inventory.md")
    _write_candidate_matrix(inventory, out_dir / "candidate_matrix.csv")
    _write_missing_artifacts(inventory, out_dir / "missing_artifacts.md")


def _load_promotion_reports(report_dir: Path) -> dict[str, dict[str, Any]]:
    reports = {}
    promotion_dir = report_dir / "promotion"
    if not promotion_dir.exists():
        return reports
    for path in sorted(promotion_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        run_name = data.get("run_name")
        if run_name:
            reports[run_name] = data
    return reports


def _load_gauntlet_decisions(report_dir: Path) -> dict[str, dict[str, Any]]:
    path = report_dir / "gauntlet_report.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return {item.get("candidate"): item for item in data.get("decisions", []) if item.get("candidate")}


def _candidate_artifact_dirs(
    repo_root: Path,
    configured_out_dir: str | None,
    experiment_run_dir: Path,
    run_name: str,
    promotion_reports: dict[str, dict[str, Any]],
) -> list[Path]:
    dirs: list[Path] = []
    if configured_out_dir:
        dirs.append(repo_root / configured_out_dir)
    dirs.append(experiment_run_dir / run_name)
    promotion = promotion_reports.get(run_name)
    if promotion:
        screen_run_dir = promotion.get("screen_run_dir")
        if screen_run_dir and _artifact_dir_matches_run_name(repo_root / screen_run_dir, run_name):
            dirs.append(repo_root / screen_run_dir)
        artifact_checkpoint = promotion.get("artifact_paths", {}).get("checkpoint")
        if artifact_checkpoint:
            artifact_dir = (repo_root / artifact_checkpoint).parents[1]
            if _artifact_dir_matches_run_name(artifact_dir, run_name):
                dirs.append(artifact_dir)
    screen_root = repo_root / "runs" / "screen"
    if screen_root.exists():
        dirs.extend(
            path
            for path in sorted(screen_root.glob(f"{run_name}_*"))
            if _artifact_dir_matches_run_name(path, run_name)
        )
    unique = []
    seen = set()
    for path in dirs:
        resolved = path.resolve()
        if resolved not in seen:
            unique.append(path)
            seen.add(resolved)
    return unique


def _artifact_dir_matches_run_name(path: Path, run_name: str) -> bool:
    config_path = path / "config.yaml"
    if not config_path.exists():
        return path.name == run_name
    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        return False
    return data.get("run", {}).get("name") == run_name


def _first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _diagnostics_status(attention_type: str, diagnostics_path: Path | None) -> str:
    if diagnostics_path is not None:
        return "available"
    if attention_type == "standard":
        return "not_recorded"
    return "missing"


def _hook_sites_for(attention_type: str) -> tuple[list[str], list[str]]:
    try:
        get_hook_site_specs(attention_type)
    except ValueError:
        return [], []
    return supported_site_names(attention_type), sorted(unsupported_site_names(attention_type))


def _evidence_level(
    checkpoint_path: Path | None,
    diagnostics_path: Path | None,
    run_summary_path: Path | None,
    promotion: dict[str, Any],
    gauntlet: dict[str, Any],
) -> str:
    if checkpoint_path is not None:
        return "checkpoint_recompute"
    if diagnostics_path is not None or run_summary_path is not None or promotion or gauntlet:
        return "artifact_summary"
    return "not_available"


def _gauntlet_status(decision: dict[str, Any]) -> str:
    if not decision:
        return "missing"
    if decision.get("promotion_recommendation") == "kill":
        return "kill"
    return str(decision.get("status") or decision.get("machine_decision") or "missing")


def _notes(checkpoint_status: str, diagnostics_status: str, run_summary_status: str) -> list[str]:
    notes = []
    if checkpoint_status != "available":
        notes.append("checkpoint_unavailable")
    if diagnostics_status in {"missing", "not_recorded"}:
        notes.append(diagnostics_status)
    if run_summary_status != "available":
        notes.append("run_summary_missing")
    return notes


def _write_inventory_md(inventory: dict[str, Any], path: Path) -> None:
    lines = [
        f"# Mechanism Backfill Inventory: {inventory['experiment_id']}",
        "",
        "| run | attention | checkpoint | diagnostics | evidence | posthoc |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in inventory["candidates"]:
        lines.append(
            "| {run_name} | {attention_type} | {checkpoint_status} | {diagnostics_status} | "
            "{evidence_level} | {posthoc_probe_status} |".format(**row)
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_candidate_matrix(inventory: dict[str, Any], path: Path) -> None:
    fieldnames = [
        "experiment_id",
        "run_name",
        "attention_type",
        "config_path",
        "checkpoint_status",
        "diagnostics_status",
        "run_summary_status",
        "gauntlet_status",
        "promotion_status",
        "supported_hook_sites",
        "unsupported_hook_sites",
        "posthoc_probe_status",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in inventory["candidates"]:
            writer.writerow(
                {
                    key: ";".join(row[key]) if isinstance(row.get(key), list) else row.get(key, "")
                    for key in fieldnames
                }
            )


def _write_missing_artifacts(inventory: dict[str, Any], path: Path) -> None:
    lines = [f"# Missing Mechanism Artifacts: {inventory['experiment_id']}", ""]
    for row in inventory["candidates"]:
        reasons = []
        if row["checkpoint_status"] != "available":
            reasons.append("checkpoint_unavailable")
        if row["diagnostics_status"] in {"missing", "not_recorded"}:
            reasons.append(row["diagnostics_status"])
        if row["run_summary_status"] != "available":
            reasons.append("missing")
        if reasons:
            lines.append(f"- `{row['run_name']}`: {', '.join(reasons)}")
    if len(lines) == 2:
        lines.append("- none")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _rel(root: Path, path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)
