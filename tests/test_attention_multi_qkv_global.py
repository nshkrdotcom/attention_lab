from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from attention_lab.models.attention.multi_qkv_common import MultiQKVGlobalBank, MultiQKVRouteContext
from attention_lab.models.attention.multi_qkv_position_rotation import MultiQKVPositionRotationGlobalCausalSelfAttention
from attention_lab.models.attention.multi_qkv_static import MultiQKVStaticGlobalCausalSelfAttention
from attention_lab.models.attention.multi_qkv_train_rotation import MultiQKVTrainRotationGlobalCausalSelfAttention
from attention_lab.models.attention.standard import StandardCausalSelfAttention
from attention_lab.models.gpt import GPTConfig


ROUTES = {
    "multi_qkv_static_3track_global": "layer_mod",
    "multi_qkv_train_rotation_3track_global": "layer_plus_step_train_layer_eval",
    "multi_qkv_position_rotation_3track_global": "layer_plus_position",
}

ATTENTION_CLASSES = {
    "multi_qkv_static_3track_global": MultiQKVStaticGlobalCausalSelfAttention,
    "multi_qkv_train_rotation_3track_global": MultiQKVTrainRotationGlobalCausalSelfAttention,
    "multi_qkv_position_rotation_3track_global": MultiQKVPositionRotationGlobalCausalSelfAttention,
}


def tiny_standard_config() -> GPTConfig:
    return GPTConfig(
        block_size=8,
        vocab_size=64,
        n_layer=3,
        n_head=2,
        n_embd=16,
        dropout=0.0,
        bias=False,
        attention_type="standard",
    )


def tiny_multi_qkv_config(attention_type: str, track_count: int = 3) -> GPTConfig:
    return GPTConfig(
        block_size=8,
        vocab_size=64,
        n_layer=3,
        n_head=2,
        n_embd=16,
        dropout=0.0,
        bias=False,
        attention_type=attention_type,
        qkv_track_count=track_count,
        qkv_global_bank=True,
        qkv_route_formula=ROUTES[attention_type],
    )


def tiny_one_track_multi_qkv_config() -> GPTConfig:
    return GPTConfig(
        block_size=8,
        vocab_size=64,
        n_layer=1,
        n_head=2,
        n_embd=16,
        dropout=0.0,
        bias=False,
        attention_type="multi_qkv_static_3track_global",
        qkv_track_count=1,
        qkv_global_bank=True,
        qkv_route_formula="layer_mod",
    )


def build_multi_qkv_attention_for_test(
    config: GPTConfig,
    bank: MultiQKVGlobalBank,
    *,
    layer_idx: int,
):
    return ATTENTION_CLASSES[config.attention_type](config, layer_idx=layer_idx, qkv_bank=bank)


def copy_standard_weights(target: MultiQKVStaticGlobalCausalSelfAttention, source: StandardCausalSelfAttention) -> None:
    target.qkv_bank.c_attn_bank[0].load_state_dict(deepcopy(source.c_attn.state_dict()))
    target.c_proj.load_state_dict(deepcopy(source.c_proj.state_dict()))


def test_multi_qkv_global_bank_constructs_three_packed_tracks():
    config = tiny_multi_qkv_config("multi_qkv_static_3track_global")
    bank = MultiQKVGlobalBank(config)

    assert bank.track_count == 3
    assert len(bank.c_attn_bank) == 3
    for layer in bank.c_attn_bank:
        assert tuple(layer.weight.shape) == (3 * config.n_embd, config.n_embd)


def test_multi_qkv_global_bank_project_track_shape():
    config = tiny_multi_qkv_config("multi_qkv_static_3track_global")
    bank = MultiQKVGlobalBank(config)
    x = torch.randn(2, 8, config.n_embd)

    qkv = bank.project_track(x, 1)

    assert qkv.shape == (2, 8, 3 * config.n_embd)


def test_multi_qkv_global_bank_rejects_invalid_track_index():
    config = tiny_multi_qkv_config("multi_qkv_static_3track_global")
    bank = MultiQKVGlobalBank(config)
    x = torch.randn(2, 8, config.n_embd)

    with pytest.raises(IndexError):
        bank.project_track(x, 3)


@pytest.mark.parametrize(
    ("layer_idx", "expected"),
    [(0, 0), (1, 1), (2, 2), (3, 0), (4, 1), (5, 2)],
)
def test_static_global_track_formula_is_layer_mod(layer_idx: int, expected: int):
    config = tiny_multi_qkv_config("multi_qkv_static_3track_global")
    bank = MultiQKVGlobalBank(config)
    attention = MultiQKVStaticGlobalCausalSelfAttention(config, layer_idx=layer_idx, qkv_bank=bank)
    context = MultiQKVRouteContext(layer_idx=layer_idx, step=999, schedule_mode="train")

    assert attention.select_scalar_track(context) == expected


@pytest.mark.parametrize(
    ("layer_idx", "step", "expected"),
    [
        (0, 0, 0),
        (1, 0, 1),
        (2, 0, 2),
        (0, 1, 1),
        (1, 1, 2),
        (2, 1, 0),
        (0, 2, 2),
        (1, 2, 0),
        (2, 2, 1),
    ],
)
def test_train_rotation_uses_layer_plus_step_during_train(layer_idx: int, step: int, expected: int):
    config = tiny_multi_qkv_config("multi_qkv_train_rotation_3track_global")
    bank = MultiQKVGlobalBank(config)
    attention = MultiQKVTrainRotationGlobalCausalSelfAttention(config, layer_idx=layer_idx, qkv_bank=bank)
    context = MultiQKVRouteContext(layer_idx=layer_idx, step=step, schedule_mode="train")

    assert attention.select_scalar_track(context) == expected


def test_train_rotation_requires_step_during_train():
    config = tiny_multi_qkv_config("multi_qkv_train_rotation_3track_global")
    bank = MultiQKVGlobalBank(config)
    attention = MultiQKVTrainRotationGlobalCausalSelfAttention(config, layer_idx=0, qkv_bank=bank)
    context = MultiQKVRouteContext(layer_idx=0, step=None, schedule_mode="train")

    with pytest.raises(ValueError, match="requires step"):
        attention.select_scalar_track(context)


@pytest.mark.parametrize("mode", ["eval", "generate"])
def test_train_rotation_freezes_to_layer_mod_during_eval_and_generate(mode: str):
    config = tiny_multi_qkv_config("multi_qkv_train_rotation_3track_global")
    bank = MultiQKVGlobalBank(config)
    attention = MultiQKVTrainRotationGlobalCausalSelfAttention(config, layer_idx=2, qkv_bank=bank)
    context = MultiQKVRouteContext(layer_idx=2, step=999, schedule_mode=mode)

    assert attention.select_scalar_track(context) == 2


def test_position_rotation_uses_layer_plus_position_vector():
    config = tiny_multi_qkv_config("multi_qkv_position_rotation_3track_global")
    bank = MultiQKVGlobalBank(config)
    attention = MultiQKVPositionRotationGlobalCausalSelfAttention(config, layer_idx=1, qkv_bank=bank)
    context = MultiQKVRouteContext(
        layer_idx=1,
        step=123,
        schedule_mode="train",
        position_ids=torch.arange(0, 8),
    )

    tracks = attention.select_position_tracks(context, seq_len=8, device=torch.device("cpu"))

    assert tracks.tolist() == [1, 2, 0, 1, 2, 0, 1, 2]


def test_position_rotation_rejects_wrong_position_length():
    config = tiny_multi_qkv_config("multi_qkv_position_rotation_3track_global")
    bank = MultiQKVGlobalBank(config)
    attention = MultiQKVPositionRotationGlobalCausalSelfAttention(config, layer_idx=0, qkv_bank=bank)
    context = MultiQKVRouteContext(
        layer_idx=0,
        step=None,
        schedule_mode="eval",
        position_ids=torch.arange(0, 7),
    )

    with pytest.raises(ValueError, match="does not match"):
        attention.select_position_tracks(context, seq_len=8, device=torch.device("cpu"))


@pytest.mark.parametrize(
    "attention_type",
    [
        "multi_qkv_static_3track_global",
        "multi_qkv_train_rotation_3track_global",
        "multi_qkv_position_rotation_3track_global",
    ],
)
def test_multi_qkv_forward_shape(attention_type: str):
    config = tiny_multi_qkv_config(attention_type)
    bank = MultiQKVGlobalBank(config)
    attention = build_multi_qkv_attention_for_test(config, bank, layer_idx=0)
    x = torch.randn(2, 8, config.n_embd)

    y = attention(x, step=0, schedule_mode="train", position_ids=torch.arange(8))

    assert y.shape == x.shape


@pytest.mark.parametrize(
    "attention_type",
    [
        "multi_qkv_static_3track_global",
        "multi_qkv_train_rotation_3track_global",
        "multi_qkv_position_rotation_3track_global",
    ],
)
def test_multi_qkv_causal_mask_prevents_future_token_influence(attention_type: str):
    torch.manual_seed(2)
    config = tiny_multi_qkv_config(attention_type)
    bank = MultiQKVGlobalBank(config)
    attention = build_multi_qkv_attention_for_test(config, bank, layer_idx=0)
    attention.eval()

    x1 = torch.randn(2, 8, config.n_embd)
    x2 = x1.clone()
    x2[:, 5:, :] = torch.randn_like(x2[:, 5:, :]) * 10.0

    y1 = attention(x1, step=None, schedule_mode="eval", position_ids=torch.arange(8))
    y2 = attention(x2, step=None, schedule_mode="eval", position_ids=torch.arange(8))

    assert torch.allclose(y1[:, :5, :], y2[:, :5, :], atol=1e-5)


def test_one_track_multi_qkv_matches_standard_attention_with_shared_weights():
    torch.manual_seed(3)
    standard = StandardCausalSelfAttention(tiny_standard_config())
    config = tiny_one_track_multi_qkv_config()
    bank = MultiQKVGlobalBank(config)
    multi = MultiQKVStaticGlobalCausalSelfAttention(config, layer_idx=0, qkv_bank=bank)
    copy_standard_weights(multi, standard)
    standard.eval()
    multi.eval()
    x = torch.randn(2, 8, config.n_embd)

    assert torch.allclose(multi(x), standard(x), atol=1e-6, rtol=1e-5)


def test_inactive_hard_switch_tracks_receive_no_gradient_for_static_route():
    torch.manual_seed(4)
    config = tiny_multi_qkv_config("multi_qkv_static_3track_global")
    bank = MultiQKVGlobalBank(config)
    attention = MultiQKVStaticGlobalCausalSelfAttention(config, layer_idx=0, qkv_bank=bank)
    loss = attention(torch.randn(2, 8, config.n_embd)).pow(2).mean()
    loss.backward()

    assert bank.c_attn_bank[0].weight.grad is not None and bank.c_attn_bank[0].weight.grad.abs().sum() > 0
    for track in (1, 2):
        grad = bank.c_attn_bank[track].weight.grad
        assert grad is None or grad.abs().sum() == 0
