from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_suite_artifacts(output_dir: str | Path, metrics: dict[str, Any], claim_gates: dict[str, Any]) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    (out / "claim_gates.json").write_text(json.dumps(claim_gates, indent=2, sort_keys=True), encoding="utf-8")
    (out / "summary.md").write_text(render_summary(metrics, claim_gates), encoding="utf-8")


def render_summary(metrics: dict[str, Any], claim_gates: dict[str, Any]) -> str:
    run = metrics.get("run", {})
    task = metrics.get("task_suite", {})
    control = metrics.get("control", {})
    mode = metrics.get("mode", {})
    fdr = metrics.get("fdr_bh", {})
    cells = metrics.get("cells", {})
    pooling = metrics.get("feature_pooling", {})
    preflight = metrics.get("preflight", {})
    overall = claim_gates.get("overall_status", "insufficient_evidence")
    overall_claim_gate_passed = bool(claim_gates.get("overall_claim_gate_passed", False))
    random_pool = preflight.get("random_site_null_pool", {})

    lines = [
        "# Tier-1 Mechanism Probe Suite Summary",
        "",
        "This report uses a mechanism-probe-specific claim ladder, distinct from the repository-wide "
        "experiment status vocabulary.",
        "An exploratory signal is not a passed confirmatory claim gate.",
        "",
        "## Run",
        f"- experiment_id: `{run.get('experiment_id')}`",
        f"- candidate: `{run.get('candidate')}`",
        f"- checkpoint: `{run.get('checkpoint')}`",
        f"- task_file: `{run.get('task_file')}`",
        f"- mode: `{'exploratory' if mode.get('exploratory') else 'confirmatory'}`",
        f"- probe_only: `{mode.get('probe_only')}`",
        f"- feature_pooling: `{pooling.get('strategy')}`",
        f"- task_aligned_pooling: `{pooling.get('task_aligned')}`",
        f"- overall_mechanism_probe_status: `{overall}`",
        f"- overall_claim_gate_passed: `{overall_claim_gate_passed}`",
        "",
        "## Control",
        f"- expected_control_checkpoint: `{control.get('expected_control_checkpoint')}`",
        f"- actual_control_checkpoint: `{control.get('actual_control_checkpoint')}`",
        f"- canonical_control: `{control.get('canonical')}`",
        f"- override_used: `{control.get('override_used')}`",
        f"- control_available: `{control.get('available')}`",
        f"- reason: {control.get('reason') or 'none'}",
        "",
        "## Task Suite",
        f"- deterministic_provenance: `{task.get('deterministic_provenance')}`",
        f"- deterministic_fingerprint_valid: `{task.get('deterministic_fingerprint_valid')}`",
        f"- deterministic_fingerprint_reason: {task.get('deterministic_fingerprint_reason') or 'none'}",
        f"- confirmatory_floor_met: `{task.get('confirmatory_floor_met')}`",
        f"- restoration_token_metadata_valid: `{task.get('restoration_token_metadata_valid')}`",
        f"- pair_counts_by_family: `{task.get('pair_counts_by_family')}`",
        f"- validation_errors: `{task.get('validation_errors')}`",
        f"- validation_warnings: `{task.get('validation_warnings')}`",
        "",
        "## FDR-BH",
        "- comparison_family: every computed `(site x layer x task_family x metric)` cell in this run, "
        "including probe, null, matched-control, specificity, restoration, and mediation metrics when present.",
        f"- alpha: `{fdr.get('alpha')}`",
        f"- tested_cells: `{fdr.get('tested_cells')}`",
        f"- invalid_or_unavailable_cells: `{fdr.get('invalid_or_unavailable_cells')}`",
        "",
        "## Random-Site Null Pool",
        f"- scope: `{random_pool.get('scope')}`",
        f"- selection_policy: {random_pool.get('selection_policy') or 'not recorded'}",
        f"- declared_sites: `{[site.get('key') for site in random_pool.get('sites', [])]}`",
        "",
        "## Site Results",
    ]
    if not cells:
        lines.append("- none")
    for cell_id, cell in sorted(cells.items()):
        gate = claim_gates.get("cells", {}).get(cell_id, {})
        random_site = cell.get("random_site_null", {})
        alignment = cell.get("alignment_to_control", {})
        random_status = _random_site_status(cell, gate)
        patching = cell.get("patching", {})
        mediation = cell.get("mediation_fraction", {})
        cell_pooling = cell.get("feature_pooling", {})
        lines.extend(
            [
                f"### `{cell_id}`",
                f"- claim_gate: `{gate.get('status')}`",
                f"- claim_gate_passed: `{gate.get('claim_gate_passed')}`",
                f"- status_kind: `{gate.get('status_kind')}`",
                f"- blockers: `{gate.get('blockers')}`",
                f"- feature_pooling: `{cell_pooling.get('strategy')}`",
                f"- task_aligned_pooling: `{cell_pooling.get('task_aligned')}`",
                f"- linear_probe_auc: `{cell.get('linear_probe_auc')}`",
                f"- auc_minus_shuffled_auc: `{cell.get('auc_minus_shuffled_auc')}`",
                f"- auc_minus_random_site_auc: `{cell.get('auc_minus_random_site_auc')}`",
                f"- auc_minus_matched_control_auc: `{cell.get('auc_minus_matched_control_auc')}`",
                f"- target_vs_decoy_specificity: `{cell.get('target_vs_decoy_specificity')}`",
                f"- random_site_status: `{random_status}`",
                f"- random_site_null_available: `{random_site.get('random_site_null_available')}`",
                f"- selected_random_site: `{random_site.get('selected_site')}`",
                f"- random_site_reason: {random_site.get('reason') or 'none'}",
                f"- patching_valid: `{patching.get('valid')}`",
                f"- restoration_alignment_valid: `{patching.get('restoration_alignment_valid')}`",
                f"- patching_reason: {patching.get('reason') or 'none'}",
                f"- mediation_fraction_valid: `{mediation.get('valid')}`",
                f"- mediation_fraction: `{mediation.get('mediation_fraction')}`",
                f"- probe_direction_cosine_to_control: `{alignment.get('probe_direction_cosine_to_control')}`",
                f"- alignment_available: `{alignment.get('available')}`",
                "",
            ]
        )

    lines.extend(
        [
            "## Limitations",
            "- Exploratory runs and probe-only runs cannot support confirmatory mechanism claims.",
            "- Missing matched controls, missing decoys, missing random-site nulls, invalid statistics, "
            "or noncanonical controls cap the affected claim gates.",
            "- Missing random-site nulls are feasibility limits for the affected `(site x layer)` cell, "
            "not automatic implementation failures and not a run-wide cap.",
            "- Mean-sequence pooling is exploratory/diagnostic for Tier-1 and cannot support "
            "`candidate_mechanism_evidence`; confirmatory claims require task-aligned pooling.",
            "- FDR-BH reports both tested metric cells and invalid/unavailable cells with reasons; unavailable "
            "cells are not assigned meaningful p-values.",
            "- Candidate-to-control alignment is not cross-architecture universality evidence.",
            "- Low alignment is not representational novelty evidence by itself.",
            "- This Tier-1 status is single-seed, checkpoint-backed, statistically controlled evidence when gates pass; "
            "it is not a replicated finding.",
            "- This report is not evidence that the mechanism is universal, replicated, solved a task family, "
            "lowered superposition, or proves a causal mechanism in general.",
            "",
        ]
    )
    if mode.get("probe_only"):
        lines.append("Probe-only mode skipped interventions, causal patching, restoration, and mediation metrics.")
    if mode.get("exploratory"):
        lines.append("Exploratory mode capped the claim ladder below confirmatory evidence.")
    return "\n".join(lines).rstrip() + "\n"


def _random_site_status(cell: dict[str, Any], gate: dict[str, Any]) -> str:
    random_site = cell.get("random_site_null", {})
    if not random_site.get("random_site_null_available"):
        return "unavailable_no_compatible_matched_dimensionality_site"
    blockers = gate.get("blockers") or []
    if any("random-site null comparison failed" in blocker for blocker in blockers):
        return "available_but_candidate_did_not_beat_null_after_correction"
    if cell.get("auc_minus_random_site_auc") is None:
        return "available_but_metric_unavailable"
    return "available"


def load_suite_artifacts(output_dir: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    out = Path(output_dir)
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    claim_gates = json.loads((out / "claim_gates.json").read_text(encoding="utf-8"))
    return metrics, claim_gates


def validate_suite_artifacts(output_dir: str | Path) -> list[str]:
    out = Path(output_dir)
    errors: list[str] = []
    metrics_path = out / "metrics.json"
    gates_path = out / "claim_gates.json"
    summary_path = out / "summary.md"
    for path in (metrics_path, gates_path, summary_path):
        if not path.exists():
            errors.append(f"missing artifact: {path.name}")
    if errors:
        return errors
    metrics, claim_gates = load_suite_artifacts(out)
    for key in ("schema_version", "run", "mode", "control", "task_suite", "sites_evaluated", "cells", "fdr_bh"):
        if key not in metrics:
            errors.append(f"metrics.json missing {key}")
    for key in ("overall_status", "status_vocabulary", "status_vocabulary_scope", "cells"):
        if key not in claim_gates:
            errors.append(f"claim_gates.json missing {key}")
    if not isinstance(metrics.get("cells"), dict):
        errors.append("metrics.json cells must be a mapping")
    else:
        required_cell_keys = (
            "site",
            "layer",
            "family_id",
            "feature_pooling",
            "linear_probe_auc",
            "random_site_null",
            "matched_control",
            "alignment_to_control",
            "patching",
            "mediation_fraction",
        )
        for cell_id, cell in metrics["cells"].items():
            if not isinstance(cell, dict):
                errors.append(f"cell {cell_id} must be a mapping")
                continue
            for key in required_cell_keys:
                if key not in cell:
                    errors.append(f"cell {cell_id} missing {key}")
    fdr = metrics.get("fdr_bh", {})
    if isinstance(fdr, dict):
        for key in ("comparison_family", "tested_cells", "invalid_or_unavailable_cells", "results"):
            if key not in fdr:
                errors.append(f"fdr_bh missing {key}")
    try:
        regenerated = render_summary(metrics, claim_gates)
        if not regenerated.strip():
            errors.append("summary.md regeneration produced empty output")
    except Exception as exc:  # pragma: no cover - defensive artifact validation
        errors.append(f"summary.md could not be regenerated: {exc}")
    return errors
