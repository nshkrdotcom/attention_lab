from __future__ import annotations

import json

from attention_lab.models.attention.registry import build_attention
from attention_lab.models.gpt import GPT, GPTConfig
from attention_lab.queue.mechanism_checks import evaluate_mechanism_activity, mechanism_check_name
from attention_lab.training.attention_diagnostics import collect_attention_diagnostics


def tiny_config(attention_type: str) -> GPTConfig:
    return GPTConfig(
        block_size=8,
        vocab_size=64,
        n_layer=2,
        n_head=2,
        n_embd=16,
        dropout=0.0,
        bias=False,
        attention_type=attention_type,
        diff_qkv_lambda_init=0.5,
        diff_qkv_lambda_trainable=True,
        diff_qkv_share_value=False,
        scope_gate_bias_init=0.0,
        scope_stream_scale_init=1.0,
    )


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_qkv_attention_types_are_buildable_from_registry():
    assert build_attention(tiny_config("differential_qkv_anti_value")).__class__.__name__ == (
        "DifferentialQKVAntiValueCausalSelfAttention"
    )
    assert build_attention(tiny_config("scope_gated_qkv")).__class__.__name__ == "ScopeGatedQKVCausalSelfAttention"


def test_collect_attention_diagnostics_includes_new_qkv_fields():
    import torch

    torch.manual_seed(0)
    model = GPT(tiny_config("scope_gated_qkv"))
    idx = torch.randint(0, 64, (2, 8))
    _, loss = model(idx, idx, step=10, schedule_mode="train")
    assert loss is not None
    loss.backward()

    rows = collect_attention_diagnostics(model, step=10)

    assert len(rows) == 2
    assert rows[0]["attention_type"] == "scope_gated_qkv"
    assert rows[0]["scope_output_norm"] > 0
    assert rows[0]["gate_mean"] > 0


def test_mechanism_check_names_for_e003_attention_types():
    assert mechanism_check_name("differential_qkv_anti_value") == "differential_qkv_activity"
    assert mechanism_check_name("scope_gated_qkv") == "scope_gated_qkv_activity"


def test_differential_qkv_activity_requires_nonzero_branches_and_positive_lambda(tmp_path):
    missing = evaluate_mechanism_activity(
        attention_type="differential_qkv_anti_value",
        diagnostics_path=tmp_path / "missing.jsonl",
    )
    assert missing.active is None

    diagnostics = tmp_path / "attention_diagnostics.jsonl"
    _write_jsonl(
        diagnostics,
        [
            {
                "attention_type": "differential_qkv_anti_value",
                "pos_output_norm": 0.0,
                "neg_output_norm": 0.0,
                "branch_output_delta": 0.0,
                "diff_lambda": 0.5,
            }
        ],
    )
    zero = evaluate_mechanism_activity(
        attention_type="differential_qkv_anti_value",
        diagnostics_path=diagnostics,
    )
    assert zero.active is False
    assert "pos_output_norm" in zero.note

    _write_jsonl(
        diagnostics,
        [
            {
                "attention_type": "differential_qkv_anti_value",
                "pos_output_norm": 1e-3,
                "neg_output_norm": 1e-3,
                "branch_output_delta": 1e-3,
                "diff_lambda": 0.5,
            }
        ],
    )
    valid = evaluate_mechanism_activity(
        attention_type="differential_qkv_anti_value",
        diagnostics_path=diagnostics,
    )
    assert valid.passed
    assert valid.details["branch_output_delta_max"] == 1e-3


def test_scope_gated_qkv_activity_rejects_saturated_gate_and_accepts_valid_rows(tmp_path):
    diagnostics = tmp_path / "attention_diagnostics.jsonl"
    _write_jsonl(
        diagnostics,
        [
            {
                "attention_type": "scope_gated_qkv",
                "scope_output_norm": 1e-3,
                "content_output_norm": 1e-3,
                "scope_content_interaction_norm": 1e-3,
                "gate_mean": 1.0,
                "gate_std": 0.0,
            }
        ],
    )
    saturated = evaluate_mechanism_activity(attention_type="scope_gated_qkv", diagnostics_path=diagnostics)
    assert saturated.active is False
    assert "gate_mean" in saturated.note

    _write_jsonl(
        diagnostics,
        [
            {
                "attention_type": "scope_gated_qkv",
                "scope_output_norm": 1e-3,
                "content_output_norm": 1e-3,
                "scope_content_interaction_norm": 1e-3,
                "gate_mean": 0.5,
                "gate_std": 0.1,
            }
        ],
    )
    valid = evaluate_mechanism_activity(attention_type="scope_gated_qkv", diagnostics_path=diagnostics)
    assert valid.passed
    assert valid.details["gate_mean_min"] == 0.5
