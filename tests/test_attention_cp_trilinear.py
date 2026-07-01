from __future__ import annotations

from copy import deepcopy

import torch

from attention_lab.models.attention.cp_trilinear import CPTrilinearCausalSelfAttention
from attention_lab.models.attention.standard import StandardCausalSelfAttention
from attention_lab.models.gpt import config_from_dict


def cp_trilinear_config(lambda_init: float = 0.0, trainable: bool = True, fixed: bool = False):
    return config_from_dict(
        {
            "attention_type": "cp_trilinear",
            "cp_rank": 4,
            "cp_lambda_init": lambda_init,
            "cp_lambda_trainable": trainable,
            "cp_lambda_fixed": fixed,
            "block_size": 8,
            "n_layer": 1,
            "n_head": 2,
            "n_embd": 16,
            "dropout": 0.0,
            "bias": False,
        },
        {"vocab_size": 64},
    )


def standard_config():
    config = cp_trilinear_config()
    config.attention_type = "standard"
    return config


def copy_standard_weights(target: CPTrilinearCausalSelfAttention, source: StandardCausalSelfAttention) -> None:
    target.c_attn.load_state_dict(deepcopy(source.c_attn.state_dict()))
    target.c_proj.load_state_dict(deepcopy(source.c_proj.state_dict()))


def test_cp_trilinear_forward_shape():
    torch.manual_seed(1)
    attention = CPTrilinearCausalSelfAttention(cp_trilinear_config(lambda_init=1.0))
    x = torch.randn(2, 8, 16)
    y = attention(x)
    assert tuple(y.shape) == (2, 8, 16)


def test_cp_trilinear_causal_mask_prevents_future_token_influence():
    torch.manual_seed(2)
    attention = CPTrilinearCausalSelfAttention(cp_trilinear_config(lambda_init=1.0))
    attention.eval()
    x1 = torch.randn(1, 8, 16)
    x2 = x1.clone()
    x2[:, 5:] = torch.randn_like(x2[:, 5:]) * 10.0

    y1 = attention(x1)
    y2 = attention(x2)
    assert torch.allclose(y1[:, :5], y2[:, :5], atol=1e-6, rtol=1e-5)


def test_cp_trilinear_lambda_zero_matches_standard_attention_with_shared_weights():
    torch.manual_seed(3)
    standard = StandardCausalSelfAttention(standard_config())
    cp = CPTrilinearCausalSelfAttention(cp_trilinear_config(lambda_init=0.0, trainable=False, fixed=True))
    copy_standard_weights(cp, standard)
    standard.eval()
    cp.eval()
    x = torch.randn(2, 8, 16)

    assert torch.allclose(cp(x), standard(x), atol=1e-6, rtol=1e-5)


def test_cp_trilinear_extra_scores_are_nonzero():
    torch.manual_seed(4)
    attention = CPTrilinearCausalSelfAttention(cp_trilinear_config(lambda_init=1.0))
    x = torch.randn(2, 8, 16)
    extra_scores = attention.compute_extra_scores(x)
    assert tuple(extra_scores.shape) == (2, 2, 8, 8)
    assert extra_scores.abs().sum() > 0


def test_cp_trilinear_extra_scores_change_with_value_context():
    torch.manual_seed(5)
    attention = CPTrilinearCausalSelfAttention(cp_trilinear_config(lambda_init=1.0))
    x = torch.randn(2, 8, 16)
    q_low, k_low, v_low = attention.low_rank_factors(x)
    extra_1 = attention.extra_scores_from_factors(q_low, k_low, v_low)
    extra_2 = attention.extra_scores_from_factors(q_low, k_low, v_low + 0.25)
    assert not torch.allclose(extra_1, extra_2)


def test_cp_trilinear_gradients_flow_when_lambda_permits_it():
    torch.manual_seed(6)
    attention = CPTrilinearCausalSelfAttention(cp_trilinear_config(lambda_init=1.0))
    x = torch.randn(2, 8, 16)
    loss = attention(x).pow(2).mean()
    loss.backward()
    assert attention.q_low.weight.grad is not None
    assert attention.k_low.weight.grad is not None
    assert attention.v_low.weight.grad is not None
    assert attention.q_low.weight.grad.abs().sum() > 0
    assert attention.k_low.weight.grad.abs().sum() > 0
    assert attention.v_low.weight.grad.abs().sum() > 0


def test_cp_trilinear_lambda_trainable_and_fixed_modes():
    trainable = CPTrilinearCausalSelfAttention(cp_trilinear_config(lambda_init=0.0, trainable=True, fixed=False))
    fixed = CPTrilinearCausalSelfAttention(cp_trilinear_config(lambda_init=0.0, trainable=False, fixed=True))

    assert isinstance(trainable.cp_lambda, torch.nn.Parameter)
    assert trainable.cp_lambda.requires_grad
    assert not isinstance(fixed.cp_lambda, torch.nn.Parameter)


def test_cp_trilinear_trainable_lambda_receives_gradient_at_zero_init():
    torch.manual_seed(7)
    attention = CPTrilinearCausalSelfAttention(cp_trilinear_config(lambda_init=0.0, trainable=True, fixed=False))
    x = torch.randn(2, 8, 16)
    loss = attention(x).pow(2).mean()
    loss.backward()
    assert attention.cp_lambda.grad is not None
    assert attention.cp_lambda.grad.abs() > 0
