from __future__ import annotations

from attention_lab.mechanisms.diagnostics import normalize_diagnostic_row, normalize_diagnostics_jsonl


def test_normalize_operator_and_dynamic_diagnostics():
    operator = normalize_diagnostic_row(
        {
            "attention_type": "operator_valued_attention",
            "layer": 0,
            "step": 10,
            "operator_prob_entropy_mean": 1.2,
            "operator_combined_output_norm": 3.4,
            "operator_prob_suppress_mean": 0.2,
            "operator_suppress_output_norm": 5.6,
        }
    )
    dynamic = normalize_diagnostic_row(
        {
            "attention_type": "dynamic_value_query_conditioned_attention",
            "layer": 1,
            "dynamic_value_gate_mean": 0.4,
            "dynamic_value_gate_std": 0.1,
            "dynamic_value_delta_to_static_ratio": 0.7,
        }
    )

    assert operator["metrics"]["router_entropy_mean"] == 1.2
    assert operator["metrics"]["suppress_out_norm"] == 5.6
    assert dynamic["metrics"]["gate_mean"] == 0.4
    assert dynamic["metrics"]["delta_to_static_ratio"] == 0.7


def test_normalize_diagnostics_jsonl_reads_real_jsonl(tmp_path):
    path = tmp_path / "attention_diagnostics.jsonl"
    path.write_text(
        '{"attention_type":"scope_gated_qkv","content_output_norm":2.0,"scope_content_interaction_norm":3.0}\n'
        '{"attention_type":"differential_qkv_anti_value","diff_lambda":0.5,"branch_output_delta":4.0}\n',
        encoding="utf-8",
    )

    rows = normalize_diagnostics_jsonl(path)

    assert rows[0]["metrics"]["content_out_norm"] == 2.0
    assert rows[0]["metrics"]["content_scope_product_norm"] == 3.0
    assert rows[1]["metrics"]["lambda"] == 0.5
    assert rows[1]["metrics"]["branch_delta_norm"] == 4.0
