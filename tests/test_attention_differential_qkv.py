from __future__ import annotations

import torch

from attention_lab.models.attention.differential_qkv import DifferentialQKVAntiValueCausalSelfAttention
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
        attention_type="differential_qkv_anti_value",
        diff_qkv_lambda_init=0.5,
        diff_qkv_lambda_trainable=True,
        diff_qkv_share_value=False,
    )


def test_differential_qkv_constructs_and_forward_shape_and_diagnostics():
    torch.manual_seed(0)
    config = tiny_config()
    attention = DifferentialQKVAntiValueCausalSelfAttention(config)
    x = torch.randn(2, 8, 16)

    y = attention(x, step=3, layer_idx=1)

    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    row = attention.attention_diagnostics(step=3, layer=1)
    assert row is not None
    assert row["attention_type"] == "differential_qkv_anti_value"
    assert row["diff_lambda"] > 0
    assert row["pos_output_norm"] > 0
    assert row["neg_output_norm"] > 0
    assert row["branch_output_delta"] > 0
    assert row["layer"] == 1
    assert row["step"] == 3


def test_differential_qkv_causal_mask_prevents_future_token_influence():
    torch.manual_seed(1)
    attention = DifferentialQKVAntiValueCausalSelfAttention(tiny_config())
    attention.eval()
    x1 = torch.randn(1, 8, 16)
    x2 = x1.clone()
    x2[:, 5:, :] = torch.randn_like(x2[:, 5:, :]) * 10.0

    y1 = attention(x1)
    y2 = attention(x2)

    assert torch.allclose(y1[:, :5, :], y2[:, :5, :], atol=1e-6, rtol=1e-5)


def test_differential_qkv_gradients_reach_negative_branch_and_lambda():
    torch.manual_seed(2)
    attention = DifferentialQKVAntiValueCausalSelfAttention(tiny_config())
    x = torch.randn(2, 8, 16)

    loss = attention(x).pow(2).mean()
    loss.backward()

    assert attention.lambda_raw.grad is not None
    assert attention.lambda_raw.grad.abs() > 0
    assert attention.c_attn.weight.grad is not None
    q_pos_grad, k_pos_grad, v_pos_grad, q_neg_grad, k_neg_grad, v_neg_grad = attention.c_attn.weight.grad.split(
        attention.n_embd,
        dim=0,
    )
    assert q_pos_grad.abs().sum() > 0
    assert k_pos_grad.abs().sum() > 0
    assert v_pos_grad.abs().sum() > 0
    assert q_neg_grad.abs().sum() > 0
    assert k_neg_grad.abs().sum() > 0
    assert v_neg_grad.abs().sum() > 0


def test_differential_qkv_parameter_count_is_greater_than_standard_but_finite():
    config = tiny_config()
    standard = StandardCausalSelfAttention(GPTConfig(**{**config.__dict__, "attention_type": "standard"}))
    attention = DifferentialQKVAntiValueCausalSelfAttention(config)

    standard_params = sum(param.numel() for param in standard.parameters())
    differential_params = sum(param.numel() for param in attention.parameters())

    assert differential_params > standard_params
    assert differential_params < standard_params * 3


def test_differential_qkv_share_value_ablation_constructs_and_runs():
    config = tiny_config()
    config.diff_qkv_share_value = True
    attention = DifferentialQKVAntiValueCausalSelfAttention(config)
    x = torch.randn(1, 8, 16)

    y = attention(x)

    assert y.shape == x.shape
    assert attention.c_attn.out_features == 5 * config.n_embd
