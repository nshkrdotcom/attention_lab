from __future__ import annotations

import torch

from attention_lab.mechanisms.capture import capture_activations
from attention_lab.mechanisms.hook_sites import get_hook_site_specs
from attention_lab.models.gpt import GPT, GPTConfig


def tiny_cp_config(attention_type: str) -> GPTConfig:
    return GPTConfig(
        block_size=8,
        vocab_size=64,
        n_layer=2,
        n_head=2,
        n_embd=16,
        dropout=0.0,
        bias=False,
        attention_type=attention_type,
        cp_rank=4,
        cp_lambda_init=1.0,
        cp_lambda_trainable=True,
        cp_lambda_fixed=False,
    )


def tiny_multi_config(attention_type: str, route_formula: str) -> GPTConfig:
    return GPTConfig(
        block_size=8,
        vocab_size=64,
        n_layer=2,
        n_head=2,
        n_embd=16,
        dropout=0.0,
        bias=False,
        attention_type=attention_type,
        qkv_track_count=3,
        qkv_global_bank=True,
        qkv_route_formula=route_formula,
    )


def _assert_valid_causal_attention_distribution(weights: torch.Tensor) -> None:
    row_sums = weights.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    seq_len = weights.shape[-1]
    upper_triangle = weights.triu(diagonal=1)
    assert torch.allclose(upper_triangle, torch.zeros_like(upper_triangle), atol=0.0)
    assert weights.shape[-2] == seq_len


def test_attn_weights_site_is_registered_for_cp_and_multi_qkv_families():
    cp_names = {spec.name for spec in get_hook_site_specs("cp_bilinear")}
    multi_names = {spec.name for spec in get_hook_site_specs("multi_qkv_static_3track_global")}

    assert "attn_weights[layer]" in cp_names
    assert "attn_weights[layer]" in multi_names


def test_cp_bilinear_attn_weights_is_a_real_causal_softmax_distribution():
    torch.manual_seed(0)
    model = GPT(tiny_cp_config("cp_bilinear"))
    model.eval()
    input_ids = torch.randint(0, 64, (2, 8))

    result = capture_activations(model, input_ids, sites=["attn_weights[0]", "attn_weights[1]"], detach=True)

    for layer in (0, 1):
        weights = result.cache.records[f"attn_weights[{layer}]"].tensor
        assert tuple(weights.shape) == (2, 2, 8, 8)
        _assert_valid_causal_attention_distribution(weights)


def test_cp_trilinear_attn_weights_is_a_real_causal_softmax_distribution():
    torch.manual_seed(0)
    model = GPT(tiny_cp_config("cp_trilinear"))
    model.eval()
    input_ids = torch.randint(0, 64, (2, 8))

    result = capture_activations(model, input_ids, sites=["attn_weights[0]"], detach=True)

    weights = result.cache.records["attn_weights[0]"].tensor
    _assert_valid_causal_attention_distribution(weights)


def test_multi_qkv_attn_weights_is_a_real_causal_softmax_distribution():
    variants = [
        ("multi_qkv_static_3track_global", "layer_mod"),
        ("multi_qkv_position_rotation_3track_global", "layer_plus_position"),
    ]
    for seed, (attention_type, route_formula) in enumerate(variants, start=1):
        torch.manual_seed(seed)
        model = GPT(tiny_multi_config(attention_type, route_formula))
        model.eval()
        input_ids = torch.randint(0, 64, (2, 8))

        result = capture_activations(model, input_ids, sites=["attn_weights[0]"], detach=True)

        weights = result.cache.records["attn_weights[0]"].tensor
        _assert_valid_causal_attention_distribution(weights)


def test_attn_weights_capture_does_not_change_model_output():
    torch.manual_seed(0)
    model = GPT(tiny_cp_config("cp_bilinear"))
    model.eval()
    input_ids = torch.randint(0, 64, (1, 8))

    baseline_logits, _ = model(input_ids)
    result = capture_activations(model, input_ids, sites=["attn_weights[0]"], detach=True)

    assert torch.allclose(result.logits, baseline_logits, atol=0.0, rtol=0.0)
