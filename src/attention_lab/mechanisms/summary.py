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
    overall = claim_gates.get("overall_status", "insufficient_evidence")

    lines = [
        "# Tier-1 Mechanism Probe Suite Summary",
        "",
        "This report uses a mechanism-probe-specific claim ladder, distinct from the repository-wide "
        "experiment status vocabulary.",
        "",
        "## Run",
        f"- experiment_id: `{run.get('experiment_id')}`",
        f"- candidate: `{run.get('candidate')}`",
        f"- checkpoint: `{run.get('checkpoint')}`",
        f"- task_file: `{run.get('task_file')}`",
        f"- mode: `{'exploratory' if mode.get('exploratory') else 'confirmatory'}`",
        f"- probe_only: `{mode.get('probe_only')}`",
        f"- overall_mechanism_probe_status: `{overall}`",
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
        f"- confirmatory_floor_met: `{task.get('confirmatory_floor_met')}`",
        f"- pair_counts_by_family: `{task.get('pair_counts_by_family')}`",
        f"- validation_errors: `{task.get('validation_errors')}`",
        f"- validation_warnings: `{task.get('validation_warnings')}`",
        "",
        "## FDR-BH",
        "- comparison_family: every computed `(site x layer x task_family x metric)` cell in this run, "
        "including probe, null, matched-control, specificity, restoration, and mediation metrics when present.",
        f"- alpha: `{fdr.get('alpha')}`",
        f"- tested_cells: `{fdr.get('tested_cells')}`",
        "",
        "## Site Results",
    ]
    if not cells:
        lines.append("- none")
    for cell_id, cell in sorted(cells.items()):
        gate = claim_gates.get("cells", {}).get(cell_id, {})
        random_site = cell.get("random_site_null", {})
        alignment = cell.get("alignment_to_control", {})
        lines.extend(
            [
                f"### `{cell_id}`",
                f"- claim_gate: `{gate.get('status')}`",
                f"- blockers: `{gate.get('blockers')}`",
                f"- linear_probe_auc: `{cell.get('linear_probe_auc')}`",
                f"- auc_minus_shuffled_auc: `{cell.get('auc_minus_shuffled_auc')}`",
                f"- auc_minus_random_site_auc: `{cell.get('auc_minus_random_site_auc')}`",
                f"- auc_minus_matched_control_auc: `{cell.get('auc_minus_matched_control_auc')}`",
                f"- target_vs_decoy_specificity: `{cell.get('target_vs_decoy_specificity')}`",
                f"- random_site_null_available: `{random_site.get('random_site_null_available')}`",
                f"- random_site_reason: {random_site.get('reason') or 'none'}",
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


def load_suite_artifacts(output_dir: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    out = Path(output_dir)
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    claim_gates = json.loads((out / "claim_gates.json").read_text(encoding="utf-8"))
    return metrics, claim_gates
