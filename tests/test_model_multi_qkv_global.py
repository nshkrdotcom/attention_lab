from __future__ import annotations

import pytest
import torch

from attention_lab.models.gpt import GPT, GPTConfig
from attention_lab.training.inspect_model_config import inspect_model_config


ROUTES = {
    "multi_qkv_static_3track_global": "layer_mod",
    "multi_qkv_train_rotation_3track_global": "layer_plus_step_train_layer_eval",
    "multi_qkv_position_rotation_3track_global": "layer_plus_position",
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


def tiny_multi_qkv_config(attention_type: str) -> GPTConfig:
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
        qkv_route_formula=ROUTES[attention_type],
    )


@pytest.mark.parametrize("attention_type", list(ROUTES))
def test_canonical_multi_qkv_gpt_rejects_missing_required_fields(attention_type: str):
    config = GPTConfig(
        block_size=8,
        vocab_size=64,
        n_layer=3,
        n_head=2,
        n_embd=16,
        dropout=0.0,
        bias=False,
        attention_type=attention_type,
    )

    with pytest.raises(ValueError, match="qkv_track_count"):
        GPT(config)


@pytest.mark.parametrize("attention_type", list(ROUTES))
def test_canonical_multi_qkv_gpt_rejects_wrong_track_count(attention_type: str):
    config = tiny_multi_qkv_config(attention_type)
    config.qkv_track_count = 2

    with pytest.raises(ValueError, match="qkv_track_count"):
        GPT(config)


@pytest.mark.parametrize("attention_type", list(ROUTES))
def test_canonical_multi_qkv_gpt_rejects_non_global_bank(attention_type: str):
    config = tiny_multi_qkv_config(attention_type)
    config.qkv_global_bank = False

    with pytest.raises(ValueError, match="qkv_global_bank"):
        GPT(config)


@pytest.mark.parametrize("attention_type", list(ROUTES))
def test_canonical_multi_qkv_gpt_rejects_wrong_route_formula(attention_type: str):
    config = tiny_multi_qkv_config(attention_type)
    config.qkv_route_formula = "layer_mod" if ROUTES[attention_type] != "layer_mod" else "layer_plus_position"

    with pytest.raises(ValueError, match="qkv_route_formula"):
        GPT(config)


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
        cp_lambda_init=0.1,
        cp_lambda_trainable=True,
        cp_lambda_fixed=False,
    )


def test_multi_qkv_model_blocks_share_one_global_bank():
    config = tiny_multi_qkv_config("multi_qkv_static_3track_global")
    model = GPT(config)

    banks = [block.attn.qkv_bank for block in model.transformer.h]

    assert len(banks) == config.n_layer
    assert all(bank is banks[0] for bank in banks)
    assert model.multi_qkv_bank is banks[0]


def test_multi_qkv_model_blocks_share_bank_parameters():
    config = tiny_multi_qkv_config("multi_qkv_train_rotation_3track_global")
    model = GPT(config)

    first = model.transformer.h[0].attn.qkv_bank.c_attn_bank[0].weight
    second = model.transformer.h[1].attn.qkv_bank.c_attn_bank[0].weight

    assert first is second


@pytest.mark.parametrize(
    "attention_type",
    [
        "multi_qkv_static_3track_global",
        "multi_qkv_train_rotation_3track_global",
        "multi_qkv_position_rotation_3track_global",
    ],
)
def test_multi_qkv_gpt_forward_shape(attention_type: str):
    torch.manual_seed(1)
    config = tiny_multi_qkv_config(attention_type)
    model = GPT(config)
    idx = torch.randint(0, config.vocab_size, (2, 8))
    targets = torch.randint(0, config.vocab_size, (2, 8))

    logits, loss = model(idx, targets, step=0, schedule_mode="train")

    assert logits.shape == (2, 8, config.vocab_size)
    assert loss is not None and loss.ndim == 0


def test_standard_model_output_unchanged_by_step_and_schedule_mode():
    torch.manual_seed(2)
    config = tiny_standard_config()
    model = GPT(config)
    model.eval()
    idx = torch.randint(0, config.vocab_size, (2, 8))

    with torch.no_grad():
        logits_a, _ = model(idx, step=None, schedule_mode="eval")
        logits_b, _ = model(idx, step=7, schedule_mode="eval")
        logits_c, _ = model(idx, step=7, schedule_mode="generate")

    assert torch.allclose(logits_a, logits_b, atol=0.0, rtol=0.0)
    assert torch.allclose(logits_a, logits_c, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("attention_type", ["cp_bilinear", "cp_trilinear"])
def test_cp_attention_accepts_step_and_schedule_context(attention_type: str):
    config = tiny_cp_config(attention_type)
    model = GPT(config)
    idx = torch.randint(0, config.vocab_size, (2, 8))
    targets = torch.randint(0, config.vocab_size, (2, 8))

    logits, loss = model(idx, targets, step=3, schedule_mode="train")

    assert logits.shape == (2, 8, config.vocab_size)
    assert loss is not None and loss.ndim == 0


def test_train_rotation_model_eval_freezes_tracks_to_layer_mod():
    config = tiny_multi_qkv_config("multi_qkv_train_rotation_3track_global")
    model = GPT(config)
    model.eval()
    idx = torch.randint(0, config.vocab_size, (2, 8))

    with torch.no_grad():
        model(idx, step=999, schedule_mode="eval")

    observed = [block.attn._last_active_track_index for block in model.transformer.h]

    assert observed == [0, 1, 2]


def test_train_rotation_model_train_uses_step():
    config = tiny_multi_qkv_config("multi_qkv_train_rotation_3track_global")
    model = GPT(config)
    model.train()
    idx = torch.randint(0, config.vocab_size, (2, 8))
    targets = torch.randint(0, config.vocab_size, (2, 8))

    model(idx, targets, step=1, schedule_mode="train")

    observed = [block.attn._last_active_track_index for block in model.transformer.h]

    assert observed == [1, 2, 0]


def test_position_rotation_model_records_per_position_track_counts():
    config = tiny_multi_qkv_config("multi_qkv_position_rotation_3track_global")
    model = GPT(config)
    idx = torch.randint(0, config.vocab_size, (2, 8))
    targets = torch.randint(0, config.vocab_size, (2, 8))

    model(idx, targets, step=0, schedule_mode="train")

    assert model.transformer.h[0].attn._last_active_track_counts == {"0": 3, "1": 3, "2": 2}
    assert model.transformer.h[0].attn._last_active_track_index is None


def test_position_ids_reach_position_rotation_attention():
    config = tiny_multi_qkv_config("multi_qkv_position_rotation_3track_global")
    model = GPT(config)
    idx = torch.randint(0, config.vocab_size, (2, 8))

    model(idx, step=None, schedule_mode="eval")

    assert model.transformer.h[1].attn._last_active_track_counts == {"0": 2, "1": 3, "2": 3}


def test_e002_candidate_parameter_deltas_are_reported(repo_root):
    config_dir = repo_root / "configs" / "experiments" / "E002_multitrack_qkv_shift_register"
    baseline = config_dir / "standard_refactor_control_30m_seed1.yaml"
    static = inspect_model_config(
        config_dir / "multi_qkv_static_3track_global_30m_seed1.yaml",
        baseline_config_path=baseline,
    )
    train_rotation = inspect_model_config(
        config_dir / "multi_qkv_train_rotation_3track_global_30m_seed1.yaml",
        baseline_config_path=baseline,
    )
    position = inspect_model_config(
        config_dir / "multi_qkv_position_rotation_3track_global_30m_seed1.yaml",
        baseline_config_path=baseline,
    )

    assert static["attention_type"] == "multi_qkv_static_3track_global"
    assert static["qkv_track_count"] == 3
    assert static["qkv_global_bank"] is True
    assert static["parameter_delta_vs_baseline"] != 0
    assert train_rotation["parameter_delta_vs_baseline"] == static["parameter_delta_vs_baseline"]
    assert position["parameter_delta_vs_baseline"] == static["parameter_delta_vs_baseline"]
