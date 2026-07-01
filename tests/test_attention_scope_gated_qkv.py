from __future__ import annotations

import torch

from attention_lab.models.attention.scope_gated_qkv import ScopeGatedQKVCausalSelfAttention
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
        attention_type="scope_gated_qkv",
        scope_gate_bias_init=0.0,
        scope_stream_scale_init=1.0,
    )


def test_scope_gated_qkv_constructs_and_forward_shape_and_diagnostics():
    torch.manual_seed(0)
    config = tiny_config()
    attention = ScopeGatedQKVCausalSelfAttention(config)
    x = torch.randn(2, 8, 16)

    y = attention(x, step=4, layer_idx=1)

    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    row = attention.attention_diagnostics(step=4, layer=1)
    assert row is not None
    assert row["attention_type"] == "scope_gated_qkv"
    assert row["content_output_norm"] > 0
    assert row["scope_output_norm"] > 0
    assert row["scope_content_interaction_norm"] > 0
    assert 0.0 < row["gate_mean"] < 1.0
    assert row["gate_std"] >= 0.0
    assert row["scope_stream_scale"] == 1.0
    assert row["layer"] == 1
    assert row["step"] == 4


def test_scope_gated_qkv_causal_mask_prevents_future_token_influence():
    torch.manual_seed(1)
    attention = ScopeGatedQKVCausalSelfAttention(tiny_config())
    attention.eval()
    x1 = torch.randn(1, 8, 16)
    x2 = x1.clone()
    x2[:, 5:, :] = torch.randn_like(x2[:, 5:, :]) * 10.0

    y1 = attention(x1)
    y2 = attention(x2)

    assert torch.allclose(y1[:, :5, :], y2[:, :5, :], atol=1e-6, rtol=1e-5)


def test_scope_gated_qkv_gradients_reach_scope_stream_and_gate():
    torch.manual_seed(2)
    attention = ScopeGatedQKVCausalSelfAttention(tiny_config())
    x = torch.randn(2, 8, 16)

    loss = attention(x).pow(2).mean()
    loss.backward()

    assert attention.scope_scale.grad is not None
    assert attention.scope_scale.grad.abs() > 0
    assert attention.c_gate.weight.grad is not None
    assert attention.c_gate.weight.grad.abs().sum() > 0
    assert attention.c_attn.weight.grad is not None
    q_grad, k_grad, v_grad, scope_grad = attention.c_attn.weight.grad.split(attention.n_embd, dim=0)
    assert q_grad.abs().sum() > 0
    assert k_grad.abs().sum() > 0
    assert v_grad.abs().sum() > 0
    assert scope_grad.abs().sum() > 0


def test_scope_gated_qkv_parameter_count_is_greater_than_standard_but_finite():
    config = tiny_config()
    standard = StandardCausalSelfAttention(GPTConfig(**{**config.__dict__, "attention_type": "standard"}))
    attention = ScopeGatedQKVCausalSelfAttention(config)

    standard_params = sum(param.numel() for param in standard.parameters())
    scope_params = sum(param.numel() for param in attention.parameters())

    assert scope_params > standard_params
    assert scope_params < standard_params * 4


def test_scope_gated_qkv_gate_bias_init_controls_initial_gate_mean():
    config = tiny_config()
    config.scope_gate_bias_init = 2.0
    attention = ScopeGatedQKVCausalSelfAttention(config)

    assert attention.c_gate.bias is not None
    assert torch.allclose(attention.c_gate.bias, torch.full_like(attention.c_gate.bias, 2.0))
