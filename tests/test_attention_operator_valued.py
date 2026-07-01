from __future__ import annotations

import torch

from attention_lab.models.attention.operator_valued import OperatorValuedCausalSelfAttention
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
        attention_type="operator_valued_attention",
        operator_router_hidden_mult=1.0,
        operator_suppress_scale_init=0.5,
        operator_include_bind=True,
        operator_include_transform=True,
    )


def test_operator_valued_constructs_and_forward_shape_and_diagnostics():
    torch.manual_seed(0)
    attention = OperatorValuedCausalSelfAttention(tiny_config())
    x = torch.randn(2, 8, 16)

    y = attention(x, step=3, layer_idx=1)

    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    row = attention.attention_diagnostics(step=3, layer=1)
    assert row is not None
    assert row["attention_type"] == "operator_valued_attention"
    for key in (
        "operator_prob_add_mean",
        "operator_prob_suppress_mean",
        "operator_prob_gate_mean",
        "operator_prob_transform_mean",
        "operator_prob_bind_mean",
        "operator_prob_entropy_mean",
        "operator_argmax_add_frac",
        "operator_argmax_suppress_frac",
        "operator_argmax_gate_frac",
        "operator_argmax_transform_frac",
        "operator_argmax_bind_frac",
        "operator_add_output_norm",
        "operator_suppress_output_norm",
        "operator_gate_output_norm",
        "operator_transform_output_norm",
        "operator_bind_output_norm",
        "operator_combined_output_norm",
        "operator_suppress_scale",
    ):
        assert key in row
    assert row["operator_combined_output_norm"] > 0
    assert row["operator_suppress_scale"] > 0
    assert row["layer"] == 1
    assert row["step"] == 3


def test_operator_valued_causal_mask_prevents_future_token_influence():
    torch.manual_seed(1)
    attention = OperatorValuedCausalSelfAttention(tiny_config())
    attention.eval()
    x1 = torch.randn(1, 8, 16)
    x2 = x1.clone()
    x2[:, 5:, :] = torch.randn_like(x2[:, 5:, :]) * 10.0

    y1 = attention(x1)
    y2 = attention(x2)

    assert torch.allclose(y1[:, :5, :], y2[:, :5, :], atol=1e-6, rtol=1e-5)


def test_operator_valued_gradients_reach_operator_branches():
    torch.manual_seed(2)
    attention = OperatorValuedCausalSelfAttention(tiny_config())
    x = torch.randn(2, 8, 16)

    loss = attention(x).pow(2).mean()
    loss.backward()

    assert attention.router[-1].weight.grad is not None
    assert attention.router[-1].weight.grad.abs().sum() > 0
    assert attention.suppress_scale_raw.grad is not None
    assert attention.suppress_scale_raw.grad.abs() > 0
    for module in (
        attention.op_add,
        attention.op_suppress,
        attention.op_gate_value,
        attention.op_gate,
        attention.op_transform_1,
        attention.op_transform_2,
        attention.op_bind,
    ):
        assert module.weight.grad is not None
        assert module.weight.grad.abs().sum() > 0


def test_operator_valued_parameter_count_is_greater_than_standard_but_finite():
    config = tiny_config()
    standard = StandardCausalSelfAttention(GPTConfig(**{**config.__dict__, "attention_type": "standard"}))
    attention = OperatorValuedCausalSelfAttention(config)

    standard_params = sum(param.numel() for param in standard.parameters())
    operator_params = sum(param.numel() for param in attention.parameters())

    assert operator_params > standard_params
    assert operator_params < standard_params * 8


def test_operator_valued_respects_disabled_optional_operators():
    config = tiny_config()
    config.operator_include_transform = False
    config.operator_include_bind = False
    attention = OperatorValuedCausalSelfAttention(config)
    x = torch.randn(1, 8, 16)

    y = attention(x)
    row = attention.attention_diagnostics(step=0, layer=0)

    assert y.shape == x.shape
    assert row is not None
    assert row["operator_prob_transform_mean"] == 0.0
    assert row["operator_prob_bind_mean"] == 0.0
    assert row["operator_transform_output_norm"] == 0.0
    assert row["operator_bind_output_norm"] == 0.0
