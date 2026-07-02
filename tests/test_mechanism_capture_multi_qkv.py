from __future__ import annotations

import torch

from attention_lab.mechanisms.capture import capture_activations
from attention_lab.models.gpt import GPT, GPTConfig


def tiny_multi_config(attention_type: str, route_formula: str) -> GPTConfig:
    return GPTConfig(
        block_size=8,
        vocab_size=64,
        n_layer=3,
        n_head=2,
        n_embd=16,
        dropout=0.0,
        bias=False,
        attention_type=attention_type,
        qkv_track_count=3,
        qkv_global_bank=True,
        qkv_route_formula=route_formula,
    )


def test_static_global_selected_track_matches_layer_mod_track_count():
    torch.manual_seed(0)
    model = GPT(tiny_multi_config("multi_qkv_static_3track_global", "layer_mod"))
    model.eval()
    result = capture_activations(model, torch.randint(0, 64, (1, 8)), detach=True)

    assert result.cache.records["selected_track[0]"].tensor.item() == 0
    assert result.cache.records["selected_track[1]"].tensor.item() == 1
    assert result.cache.records["selected_track[2]"].tensor.item() == 2
    assert "track_q[0,0]" in result.cache.records
    assert "track_out[0]" in result.cache.records


def test_train_rotation_selected_track_follows_train_eval_contract():
    torch.manual_seed(1)
    model = GPT(tiny_multi_config("multi_qkv_train_rotation_3track_global", "layer_plus_step_train_layer_eval"))
    input_ids = torch.randint(0, 64, (1, 8))

    train_result = capture_activations(model, input_ids, step=1, schedule_mode="train", detach=True)
    eval_result = capture_activations(model, input_ids, schedule_mode="eval", detach=True)

    assert train_result.cache.records["selected_track[0]"].tensor.item() == 1
    assert eval_result.cache.records["selected_track[0]"].tensor.item() == 0


def test_position_rotation_selected_track_depends_on_token_position():
    torch.manual_seed(2)
    model = GPT(tiny_multi_config("multi_qkv_position_rotation_3track_global", "layer_plus_position"))
    result = capture_activations(model, torch.randint(0, 64, (1, 8)), detach=True)

    selected = result.cache.records["selected_track[0]"].tensor
    assert selected.tolist() == [0, 1, 2, 0, 1, 2, 0, 1]
