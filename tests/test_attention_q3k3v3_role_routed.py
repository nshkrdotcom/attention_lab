from __future__ import annotations

import pytest
import torch

from attention_lab.models.attention.q3k3v3_role_routed import Q3K3V3RoleRoutedCausalSelfAttention
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
        attention_type="q3k3v3_role_routed_attention",
        q3k3v3_role_dim_mode="equal",
        q3k3v3_cross_role_grid=False,
        q3k3v3_include_pair_products=True,
    )


def test_q3k3v3_constructs_and_forward_shape_and_diagnostics():
    torch.manual_seed(0)
    attention = Q3K3V3RoleRoutedCausalSelfAttention(tiny_config())
    x = torch.randn(2, 8, 16)

    y = attention(x, step=4, layer_idx=1)

    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    row = attention.attention_diagnostics(step=4, layer=1)
    assert row is not None
    assert row["attention_type"] == "q3k3v3_role_routed_attention"
    for key in (
        "q3_content_output_norm",
        "q3_operator_output_norm",
        "q3_binding_output_norm",
        "q3_content_operator_interaction_norm",
        "q3_content_binding_interaction_norm",
        "q3_operator_binding_interaction_norm",
        "q3_content_to_total_norm_ratio",
        "q3_operator_to_total_norm_ratio",
        "q3_binding_to_total_norm_ratio",
        "q3_attention_entropy_content",
        "q3_attention_entropy_operator",
        "q3_attention_entropy_binding",
        "q3_cross_role_grid_enabled",
        "q3_pair_products_enabled",
    ):
        assert key in row
    assert row["q3_content_output_norm"] > 0
    assert row["q3_operator_output_norm"] > 0
    assert row["q3_binding_output_norm"] > 0
    assert row["q3_cross_role_grid_enabled"] is False
    assert row["q3_pair_products_enabled"] is True


def test_q3k3v3_causal_mask_prevents_future_token_influence():
    torch.manual_seed(1)
    attention = Q3K3V3RoleRoutedCausalSelfAttention(tiny_config())
    attention.eval()
    x1 = torch.randn(1, 8, 16)
    x2 = x1.clone()
    x2[:, 5:, :] = torch.randn_like(x2[:, 5:, :]) * 10.0

    y1 = attention(x1)
    y2 = attention(x2)

    assert torch.allclose(y1[:, :5, :], y2[:, :5, :], atol=1e-6, rtol=1e-5)


def test_q3k3v3_gradients_reach_all_role_projections():
    torch.manual_seed(2)
    attention = Q3K3V3RoleRoutedCausalSelfAttention(tiny_config())
    x = torch.randn(2, 8, 16)

    loss = attention(x).pow(2).mean()
    loss.backward()

    assert attention.c_roles.weight.grad is not None
    chunks = attention.c_roles.weight.grad.split(attention.n_embd, dim=0)
    assert len(chunks) == 9
    for chunk in chunks:
        assert chunk.abs().sum() > 0
    assert attention.c_proj.weight.grad is not None
    assert attention.c_proj.weight.grad.abs().sum() > 0


def test_q3k3v3_parameter_count_is_greater_than_standard_but_finite():
    config = tiny_config()
    standard = StandardCausalSelfAttention(GPTConfig(**{**config.__dict__, "attention_type": "standard"}))
    attention = Q3K3V3RoleRoutedCausalSelfAttention(config)

    standard_params = sum(param.numel() for param in standard.parameters())
    q3_params = sum(param.numel() for param in attention.parameters())

    assert q3_params > standard_params
    assert q3_params < standard_params * 6


def test_q3k3v3_pair_products_and_cross_grid_flags_change_behavior():
    config = tiny_config()
    config.q3k3v3_include_pair_products = False
    attention = Q3K3V3RoleRoutedCausalSelfAttention(config)
    x = torch.randn(1, 8, 16)

    y = attention(x)
    row = attention.attention_diagnostics(step=0, layer=0)

    assert y.shape == x.shape
    assert row is not None
    assert row["q3_pair_products_enabled"] is False
    assert attention.c_proj.in_features == 3 * config.n_embd

    config.q3k3v3_cross_role_grid = True
    attention_grid = Q3K3V3RoleRoutedCausalSelfAttention(config)
    assert attention_grid.c_proj.in_features == 9 * config.n_embd


def test_q3k3v3_rejects_unknown_role_dim_mode():
    config = tiny_config()
    config.q3k3v3_role_dim_mode = "unequal"
    with pytest.raises(ValueError, match="q3k3v3_role_dim_mode"):
        Q3K3V3RoleRoutedCausalSelfAttention(config)
