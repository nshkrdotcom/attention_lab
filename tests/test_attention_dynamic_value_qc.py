from __future__ import annotations

import pytest
import torch

from attention_lab.models.attention.dynamic_value_qc import DynamicValueQueryConditionedCausalSelfAttention
from attention_lab.models.attention.standard import StandardCausalSelfAttention
from attention_lab.models.gpt import GPTConfig


def tiny_config() -> GPTConfig:
    return GPTConfig(
        block_size=8,
        vocab_size=64,
        n_layer=2,
        n_head=2,
        n_embd=16,
        dropout=0.0,
        bias=False,
        attention_type="dynamic_value_query_conditioned_attention",
        dynamic_value_gate_bias_init=0.0,
        dynamic_value_gate_from="x",
        dynamic_value_pairwise_gate=False,
    )


def test_dynamic_value_constructs_and_forward_shape_and_diagnostics():
    torch.manual_seed(0)
    attention = DynamicValueQueryConditionedCausalSelfAttention(tiny_config())
    x = torch.randn(2, 8, 16)

    y = attention(x, step=4, layer_idx=1)

    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    row = attention.attention_diagnostics(step=4, layer=1)
    assert row is not None
    assert row["attention_type"] == "dynamic_value_query_conditioned_attention"
    for key in (
        "dynamic_value_gate_mean",
        "dynamic_value_gate_std",
        "dynamic_value_gate_min",
        "dynamic_value_gate_max",
        "dynamic_value_gate_entropy_proxy",
        "dynamic_value_static_content_norm",
        "dynamic_value_gated_content_norm",
        "dynamic_value_delta_norm",
        "dynamic_value_delta_to_static_ratio",
        "dynamic_value_pairwise_gate_enabled",
        "dynamic_value_gate_from",
    ):
        assert key in row
    assert 0.0 < row["dynamic_value_gate_mean"] < 1.0
    assert row["dynamic_value_gate_std"] > 0
    assert row["dynamic_value_static_content_norm"] > 0
    assert row["dynamic_value_gated_content_norm"] > 0
    assert row["dynamic_value_delta_norm"] > 0


def test_dynamic_value_causal_mask_prevents_future_token_influence():
    torch.manual_seed(1)
    attention = DynamicValueQueryConditionedCausalSelfAttention(tiny_config())
    attention.eval()
    x1 = torch.randn(1, 8, 16)
    x2 = x1.clone()
    x2[:, 5:, :] = torch.randn_like(x2[:, 5:, :]) * 10.0

    y1 = attention(x1)
    y2 = attention(x2)

    assert torch.allclose(y1[:, :5, :], y2[:, :5, :], atol=1e-6, rtol=1e-5)


def test_dynamic_value_gradients_reach_gate_and_qkv():
    torch.manual_seed(2)
    attention = DynamicValueQueryConditionedCausalSelfAttention(tiny_config())
    x = torch.randn(2, 8, 16)

    loss = attention(x).pow(2).mean()
    loss.backward()

    assert attention.c_attn.weight.grad is not None
    assert attention.c_attn.weight.grad.abs().sum() > 0
    assert attention.c_gate.weight.grad is not None
    assert attention.c_gate.weight.grad.abs().sum() > 0
    assert attention.c_proj.weight.grad is not None
    assert attention.c_proj.weight.grad.abs().sum() > 0


def test_dynamic_value_parameter_count_is_greater_than_standard_but_finite():
    config = tiny_config()
    standard = StandardCausalSelfAttention(GPTConfig(**{**config.__dict__, "attention_type": "standard"}))
    attention = DynamicValueQueryConditionedCausalSelfAttention(config)

    standard_params = sum(param.numel() for param in standard.parameters())
    dynamic_params = sum(param.numel() for param in attention.parameters())

    assert dynamic_params > standard_params
    assert dynamic_params < standard_params * 3


def test_dynamic_value_gate_from_changes_gate_input_and_bias_is_initialized():
    config = tiny_config()
    config.dynamic_value_gate_from = "xq"
    config.dynamic_value_gate_bias_init = 1.5
    attention = DynamicValueQueryConditionedCausalSelfAttention(config)

    assert attention.c_gate.in_features == 2 * config.n_embd
    assert attention.c_gate.bias is not None
    assert torch.allclose(attention.c_gate.bias, torch.full_like(attention.c_gate.bias, 1.5))


def test_dynamic_value_rejects_pairwise_mode_until_safe_implementation_exists():
    config = tiny_config()
    config.dynamic_value_pairwise_gate = True
    with pytest.raises(ValueError, match="pairwise"):
        DynamicValueQueryConditionedCausalSelfAttention(config)
