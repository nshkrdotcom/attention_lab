from __future__ import annotations

import json
import math
from pathlib import Path

from attention_lab.queue.mechanism_checks import evaluate_mechanism_activity, mechanism_check_name


def test_e004_attention_types_map_to_mechanism_checks():
    assert mechanism_check_name("operator_valued_attention") == "operator_valued_activity"
    assert mechanism_check_name("q3k3v3_role_routed_attention") == "q3k3v3_role_activity"
    assert mechanism_check_name("dynamic_value_query_conditioned_attention") == "dynamic_value_activity"


def test_operator_valued_activity_missing_zero_nan_collapse_and_pass(tmp_path):
    missing = evaluate_mechanism_activity(
        attention_type="operator_valued_attention",
        diagnostics_path=tmp_path / "missing.jsonl",
    )
    assert missing.passed is False
    assert "missing diagnostics" in missing.reason

    zero = _eval(tmp_path, "operator_valued_attention", [_operator_row(combined=0.0)])
    assert zero.passed is False
    assert "combined output norm" in zero.reason

    nonfinite = _eval(
        tmp_path,
        "operator_valued_attention",
        [_operator_row(operator_prob_add_mean=math.nan)],
    )
    assert nonfinite.passed is False
    assert "non-finite" in nonfinite.reason

    collapsed = _eval(
        tmp_path,
        "operator_valued_attention",
        [
            _operator_row(
                operator_prob_entropy_mean=0.0,
                operator_prob_add_mean=1.0,
                operator_prob_suppress_mean=0.0,
                operator_prob_gate_mean=0.0,
                operator_prob_transform_mean=0.0,
                operator_prob_bind_mean=0.0,
                operator_argmax_add_frac=1.0,
                operator_argmax_suppress_frac=0.0,
                operator_argmax_gate_frac=0.0,
                operator_argmax_transform_frac=0.0,
                operator_argmax_bind_frac=0.0,
            )
        ],
    )
    assert collapsed.passed is False
    assert "collapsed" in collapsed.reason or "entropy" in collapsed.reason

    non_add_zero = _eval(
        tmp_path,
        "operator_valued_attention",
        [
            _operator_row(
                operator_suppress_output_norm=0.0,
                operator_gate_output_norm=0.0,
                operator_transform_output_norm=0.0,
                operator_bind_output_norm=0.0,
            )
        ],
    )
    assert non_add_zero.passed is False
    assert "non-add" in non_add_zero.reason

    valid = _eval(tmp_path, "operator_valued_attention", [_operator_row()])
    assert valid.passed is True
    assert valid.details["operator_combined_output_norm_max"] > 0
    assert valid.details["operator_prob_entropy_mean"] > 0


def test_q3k3v3_role_activity_failures_and_pass(tmp_path):
    missing = evaluate_mechanism_activity(
        attention_type="q3k3v3_role_routed_attention",
        diagnostics_path=tmp_path / "missing.jsonl",
    )
    assert missing.passed is False
    assert "missing diagnostics" in missing.reason

    zero_role = _eval(
        tmp_path,
        "q3k3v3_role_routed_attention",
        [_q3_row(q3_operator_output_norm=0.0)],
    )
    assert zero_role.passed is False
    assert "operator output norm" in zero_role.reason

    no_interaction = _eval(
        tmp_path,
        "q3k3v3_role_routed_attention",
        [
            _q3_row(
                q3_content_operator_interaction_norm=0.0,
                q3_content_binding_interaction_norm=0.0,
                q3_operator_binding_interaction_norm=0.0,
            )
        ],
    )
    assert no_interaction.passed is False
    assert "pair interactions" in no_interaction.reason

    nonfinite_ratio = _eval(
        tmp_path,
        "q3k3v3_role_routed_attention",
        [_q3_row(q3_binding_to_total_norm_ratio=math.inf)],
    )
    assert nonfinite_ratio.passed is False
    assert "non-finite" in nonfinite_ratio.reason

    valid = _eval(tmp_path, "q3k3v3_role_routed_attention", [_q3_row()])
    assert valid.passed is True
    assert valid.details["q3_operator_output_norm_max"] > 0


def test_dynamic_value_activity_failures_and_pass(tmp_path):
    missing = evaluate_mechanism_activity(
        attention_type="dynamic_value_query_conditioned_attention",
        diagnostics_path=tmp_path / "missing.jsonl",
    )
    assert missing.passed is False
    assert "missing diagnostics" in missing.reason

    saturated = _eval(
        tmp_path,
        "dynamic_value_query_conditioned_attention",
        [_dynamic_row(dynamic_value_gate_min=0.0, dynamic_value_gate_max=1.0)],
    )
    assert saturated.passed is False
    assert "saturated" in saturated.reason

    zero_delta = _eval(
        tmp_path,
        "dynamic_value_query_conditioned_attention",
        [_dynamic_row(dynamic_value_delta_norm=0.0)],
    )
    assert zero_delta.passed is False
    assert "delta norm" in zero_delta.reason

    nonfinite = _eval(
        tmp_path,
        "dynamic_value_query_conditioned_attention",
        [_dynamic_row(dynamic_value_gate_mean=math.nan)],
    )
    assert nonfinite.passed is False
    assert "non-finite" in nonfinite.reason

    valid = _eval(tmp_path, "dynamic_value_query_conditioned_attention", [_dynamic_row()])
    assert valid.passed is True
    assert valid.details["dynamic_value_delta_norm_max"] > 0


def _eval(tmp_path: Path, attention_type: str, rows: list[dict]):
    path = tmp_path / f"{attention_type}.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return evaluate_mechanism_activity(attention_type=attention_type, diagnostics_path=path)


def _operator_row(**overrides):
    row = {
        "attention_type": "operator_valued_attention",
        "operator_prob_add_mean": 0.30,
        "operator_prob_suppress_mean": 0.20,
        "operator_prob_gate_mean": 0.20,
        "operator_prob_transform_mean": 0.15,
        "operator_prob_bind_mean": 0.15,
        "operator_prob_entropy_mean": 1.55,
        "operator_argmax_add_frac": 0.50,
        "operator_argmax_suppress_frac": 0.20,
        "operator_argmax_gate_frac": 0.15,
        "operator_argmax_transform_frac": 0.10,
        "operator_argmax_bind_frac": 0.05,
        "operator_add_output_norm": 0.3,
        "operator_suppress_output_norm": 0.2,
        "operator_gate_output_norm": 0.1,
        "operator_transform_output_norm": 0.1,
        "operator_bind_output_norm": 0.1,
        "operator_combined_output_norm": 0.4,
        "operator_suppress_scale": 0.5,
    }
    if "combined" in overrides:
        row["operator_combined_output_norm"] = overrides.pop("combined")
    row.update(overrides)
    return row


def _q3_row(**overrides):
    row = {
        "attention_type": "q3k3v3_role_routed_attention",
        "q3_content_output_norm": 0.3,
        "q3_operator_output_norm": 0.2,
        "q3_binding_output_norm": 0.25,
        "q3_content_operator_interaction_norm": 0.1,
        "q3_content_binding_interaction_norm": 0.0,
        "q3_operator_binding_interaction_norm": 0.0,
        "q3_content_to_total_norm_ratio": 0.4,
        "q3_operator_to_total_norm_ratio": 0.3,
        "q3_binding_to_total_norm_ratio": 0.3,
        "q3_attention_entropy_content": 1.0,
        "q3_attention_entropy_operator": 1.0,
        "q3_attention_entropy_binding": 1.0,
        "q3_cross_role_grid_enabled": False,
        "q3_pair_products_enabled": True,
    }
    row.update(overrides)
    return row


def _dynamic_row(**overrides):
    row = {
        "attention_type": "dynamic_value_query_conditioned_attention",
        "dynamic_value_gate_mean": 0.5,
        "dynamic_value_gate_std": 0.1,
        "dynamic_value_gate_min": 0.2,
        "dynamic_value_gate_max": 0.8,
        "dynamic_value_gate_entropy_proxy": 0.65,
        "dynamic_value_static_content_norm": 0.3,
        "dynamic_value_gated_content_norm": 0.2,
        "dynamic_value_delta_norm": 0.1,
        "dynamic_value_delta_to_static_ratio": 0.33,
        "dynamic_value_pairwise_gate_enabled": False,
        "dynamic_value_gate_from": "x",
    }
    row.update(overrides)
    return row
