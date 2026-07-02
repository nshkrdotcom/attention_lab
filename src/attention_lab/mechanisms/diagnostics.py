from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def normalize_diagnostic_row(row: dict[str, Any]) -> dict[str, Any]:
    attention_type = str(row.get("attention_type", "unknown"))
    normalized: dict[str, Any] = {
        "attention_type": attention_type,
        "layer": row.get("layer", row.get("layer_idx")),
        "step": row.get("step"),
        "metrics": {},
    }
    metrics = normalized["metrics"]
    if attention_type in {"cp_bilinear", "cp_trilinear"}:
        _copy(row, metrics, "lambda_value", "lambda")
        _copy(row, metrics, "cp_score_std", "cp_score_std")
        _copy(row, metrics, "cp_to_standard_score_std_ratio", "cp_to_standard_score_std_ratio")
        _copy(row, metrics, "cp_gradient_norm", "cp_gradient_norm")
    elif attention_type.startswith("multi_qkv_"):
        _copy(row, metrics, "active_track_index", "selected_track")
        _copy(row, metrics, "active_track_counts", "track_counts")
        _copy(row, metrics, "track_entropy", "track_entropy")
        _copy(row, metrics, "track_gradient_norm", "selected_track_gradient_norm")
        _copy(row, metrics, "per_track_gradient_norm", "per_track_gradient_norm")
    elif attention_type == "differential_qkv_anti_value":
        _copy(row, metrics, "diff_lambda", "lambda")
        _copy(row, metrics, "pos_output_norm", "pos_out_norm")
        _copy(row, metrics, "neg_output_norm", "neg_out_norm")
        _copy(row, metrics, "branch_output_delta", "branch_delta_norm")
    elif attention_type == "scope_gated_qkv":
        _copy(row, metrics, "content_output_norm", "content_out_norm")
        _copy(row, metrics, "scope_output_norm", "scope_out_norm")
        _copy(row, metrics, "gate_mean", "gate_mean")
        _copy(row, metrics, "gate_std", "gate_std")
        _copy(row, metrics, "scope_content_interaction_norm", "content_scope_product_norm")
    elif attention_type == "operator_valued_attention":
        _copy(row, metrics, "operator_prob_entropy_mean", "router_entropy_mean")
        _copy(row, metrics, "operator_combined_output_norm", "combined_out_norm")
        for name in ("add", "suppress", "gate", "transform", "bind"):
            _copy(row, metrics, f"operator_prob_{name}_mean", f"prob_{name}_mean")
            _copy(row, metrics, f"operator_{name}_output_norm", f"{name}_out_norm")
    elif attention_type == "dynamic_value_query_conditioned_attention":
        _copy(row, metrics, "dynamic_value_gate_mean", "gate_mean")
        _copy(row, metrics, "dynamic_value_gate_std", "gate_std")
        _copy(row, metrics, "dynamic_value_delta_norm", "delta_norm")
        _copy(row, metrics, "dynamic_value_delta_to_static_ratio", "delta_to_static_ratio")
    elif attention_type == "q3k3v3_role_routed_attention":
        for name in ("content", "operator", "binding"):
            _copy(row, metrics, f"q3_{name}_output_norm", f"{name}_out_norm")
            _copy(row, metrics, f"q3_{name}_to_total_norm_ratio", f"{name}_to_total_norm_ratio")
        _copy(row, metrics, "q3_content_operator_interaction_norm", "content_operator_product_norm")
        _copy(row, metrics, "q3_content_binding_interaction_norm", "content_binding_product_norm")
        _copy(row, metrics, "q3_operator_binding_interaction_norm", "operator_binding_product_norm")
    return normalized


def normalize_diagnostics_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(normalize_diagnostic_row(json.loads(line)))
    return rows


def _copy(source: dict[str, Any], target: dict[str, Any], source_key: str, target_key: str) -> None:
    if source_key in source:
        target[target_key] = source[source_key]
