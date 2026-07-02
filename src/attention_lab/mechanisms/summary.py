from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from attention_lab.mechanisms.claim_gates import ClaimGateResult


def write_suite_artifacts(output_dir: str | Path, metrics: dict[str, Any], gate: ClaimGateResult) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    (out / "claim_gates.json").write_text(
        json.dumps(gate.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (out / "summary.md").write_text(render_summary_markdown(metrics, gate), encoding="utf-8")


def render_summary_markdown(metrics: dict[str, Any], gate: ClaimGateResult) -> str:
    lines = [
        "# Tier-1 Mechanism Probe Suite Summary",
        "",
        "This report uses a mechanism-probe-specific status vocabulary. It is distinct from the "
        "global experiment status vocabulary used elsewhere in Attention Lab.",
        "",
        "## Run",
        f"- experiment_id: `{metrics.get('experiment_id')}`",
        f"- candidate: `{metrics.get('candidate')}`",
        f"- checkpoint: `{metrics.get('checkpoint')}`",
        f"- task_file: `{metrics.get('task_file')}`",
        f"- hypothesis_doc: `{metrics.get('hypothesis_doc') or 'exploratory/no hypothesis doc'}`",
        f"- exploratory: `{bool(metrics.get('exploratory'))}`",
        f"- probe-only: `{bool(metrics.get('probe_only'))}`",
        f"- sites_evaluated: {', '.join(metrics.get('sites_evaluated', [])) or 'none'}",
        "",
        "## Controls",
        f"- canonical_control_checkpoint: `{metrics.get('canonical_control_checkpoint')}`",
        f"- actual_control_checkpoint: `{metrics.get('actual_control_checkpoint')}`",
        f"- control_is_canonical: `{bool(metrics.get('control_is_canonical'))}`",
    ]
    if metrics.get("control_noncanonical_reason"):
        lines.append(f"- noncanonical control reason: {metrics['control_noncanonical_reason']}")
    lines.extend(
        [
            "",
            "## Task Suite",
            f"- pair_counts_per_family: `{metrics.get('pair_counts_per_family', {})}`",
            f"- task_suite_provenance: `{metrics.get('task_suite_provenance', {})}`",
            f"- fdr_scope: {metrics.get('fdr_scope')}",
            "",
            "## Claim Gate",
            f"- status: `{gate.status}`",
            f"- status_vocabulary: `{gate.status_vocabulary}`",
        ]
    )
    if gate.reasons:
        lines.append("- reasons:")
        lines.extend(f"  - {reason}" for reason in gate.reasons)
    if gate.caps:
        lines.append(f"- caps: {', '.join(gate.caps)}")

    random_site = metrics.get("random_site_null", {})
    if random_site:
        lines.extend(
            [
                "",
                "## Random-Site Null",
                f"- available: `{bool(random_site.get('available'))}`",
            ]
        )
        if not random_site.get("available"):
            lines.append(
                "- random-site null feasibility limit: "
                f"{random_site.get('reason', 'no matched-dimensionality compatible site exists')}"
            )

    alignment = metrics.get("alignment_to_control", {})
    lines.extend(
        [
            "",
            "## Alignment",
            f"- alignment_to_control: `{alignment}`",
            "- high candidate-to-control alignment suggests an already-present or universal-ish feature may be surfaced differently.",
            "- low candidate-to-control alignment may motivate a decomposition question, but it is not representational novelty evidence by itself.",
            "- candidate-to-control alignment is not cross-architecture universality evidence.",
        ]
    )

    if metrics.get("missing_decoys"):
        lines.append("- missing decoys: this blocks non-decoy specificity claims.")
    provenance = metrics.get("task_suite_provenance", {})
    if provenance and not provenance.get("deterministic", False):
        lines.append("- hand-authored or non-provenance task file: confirmatory claims are capped unless exploratory.")
    if metrics.get("probe_only"):
        lines.append("- probe-only: patching/restoration/mediation were skipped; this is not causal.")
    if metrics.get("exploratory"):
        lines.append("- exploratory: this cannot be presented as confirmatory evidence.")
    if metrics.get("control_is_canonical") is False:
        lines.append("- noncanonical control: full canonical evidence claims are capped.")

    lines.extend(
        [
            "",
            "## Limitations",
            "- single-seed: candidate_mechanism_evidence means single-seed, checkpoint-backed, statistically controlled evidence.",
            "- not replicated: this is not a replicated finding.",
            "- not causal unless full patch/restoration metrics are valid and the gate reaches candidate_mechanism_evidence.",
            "- not cross-architecture universality evidence.",
            "- not representational novelty evidence by itself.",
            "- not evidence that the architecture has solved the task family or reduced superposition.",
        ]
    )
    return "\n".join(lines) + "\n"
