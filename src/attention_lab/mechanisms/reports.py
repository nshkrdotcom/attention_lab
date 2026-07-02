from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


REPORT_ORDER = [
    "promote_full_mechanism_run",
    "diagnostic_rescue",
    "profiling_redesign",
    "route_specialization_workbench",
    "cp_diagnostic_followup",
    "unsupported_or_incomplete",
    "not_evaluated",
]


def classify_candidate(row: dict[str, Any]) -> str:
    experiment_id = row.get("experiment_id", "")
    run_name = row.get("run_name", "")
    attention_type = row.get("attention_type", "")
    checkpoint_available = row.get("checkpoint_status") == "available"
    gauntlet_status = row.get("gauntlet_status")
    promotion_status = row.get("promotion_status")
    has_any_evidence = (
        checkpoint_available
        or row.get("diagnostics_status") == "available"
        or row.get("run_summary_status") == "available"
        or gauntlet_status not in {None, "missing"}
        or promotion_status not in {None, "missing"}
    )

    if (
        experiment_id == "E004_operator_binding_qkv_gauntlet"
        and attention_type == "operator_valued_attention"
        and "rung500" in run_name
        and gauntlet_status == "pass"
        and (checkpoint_available or promotion_status == "promote")
    ):
        return "promote_full_mechanism_run"
    if (
        experiment_id == "E003_qkv_architecture_gauntlet"
        and attention_type in {"differential_qkv_anti_value", "scope_gated_qkv"}
        and "rung500" in run_name
        and gauntlet_status == "pass"
    ):
        return "promote_full_mechanism_run"
    if (
        experiment_id == "E004_operator_binding_qkv_gauntlet"
        and attention_type == "dynamic_value_query_conditioned_attention"
        and ("rung500" in run_name or promotion_status == "kill")
    ):
        return "diagnostic_rescue"
    if (
        experiment_id == "E004_operator_binding_qkv_gauntlet"
        and attention_type == "q3k3v3_role_routed_attention"
        and has_any_evidence
    ):
        return "profiling_redesign"
    if experiment_id == "E002_multitrack_qkv_shift_register" and attention_type in {
        "multi_qkv_static_3track_global",
        "multi_qkv_position_rotation_3track_global",
    }:
        return "route_specialization_workbench"
    if experiment_id == "E001_cp_trilinear_attention" and attention_type in {"cp_bilinear", "cp_trilinear"}:
        return "cp_diagnostic_followup"
    if not has_any_evidence:
        return "not_evaluated"
    return "unsupported_or_incomplete"


def generate_cross_experiment_report(backfill_root: Path) -> str:
    rows = []
    for inventory_path in sorted(backfill_root.glob("*/inventory.json")):
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
        rows.extend(inventory.get("candidates", []))
    if not rows:
        raise ValueError(f"no inventory.json files found under {backfill_root}")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sorted(rows, key=lambda item: (item.get("experiment_id", ""), item.get("run_name", ""))):
        grouped[classify_candidate(row)].append(row)

    lines = [
        "# Cross-Experiment Mechanism Candidate Report",
        "",
        "Generated from structured backfill inventories. It is not a training-result claim.",
        "",
    ]
    for classification in REPORT_ORDER:
        title = classification.replace("_", " ").title()
        lines.append(f"## {title} (`{classification}`)")
        items = grouped.get(classification, [])
        if not items:
            lines.append("- none")
        for row in items:
            lines.append(
                "- `{run_name}` ({experiment_id}, `{attention_type}`): evidence={evidence_level}; "
                "checkpoint={checkpoint_status}; next={next_action}".format(
                    **row,
                    next_action=_next_action(classification, row),
                )
            )
        lines.append("")
    lines.append("## What Cannot Be Concluded")
    lines.extend(
        [
            "- Survival-screen pass does not establish semantic mechanism roles.",
            "- Missing historical activations cannot be reconstructed without saved tensors.",
            "- Checkpoint availability only means post-hoc recomputation is possible.",
            "- Validation-loss differences are not architecture evidence without matched controls and diagnostics.",
            "",
        ]
    )
    return "\n".join(lines)


def write_cross_experiment_report(backfill_root: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(generate_cross_experiment_report(backfill_root), encoding="utf-8")


def _next_action(classification: str, row: dict[str, Any]) -> str:
    if classification == "promote_full_mechanism_run":
        return "run matched full mechanism probe from checkpoint before full promotion"
    if classification == "diagnostic_rescue":
        return "run gate/delta post-hoc probe and causal ablation"
    if classification == "profiling_redesign":
        return "profile low-batch role streams and redesign throughput"
    if classification == "route_specialization_workbench":
        return "run route replacement and track ablation matrix"
    if classification == "cp_diagnostic_followup":
        return "run lambda/null and CP contribution probes"
    if classification == "not_evaluated":
        return "do not classify scientifically until artifacts exist"
    return "complete missing checkpoint, diagnostics, or report artifacts"
