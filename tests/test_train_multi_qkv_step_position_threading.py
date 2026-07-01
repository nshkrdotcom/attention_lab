from __future__ import annotations

import json

import pytest
import torch
import yaml

from attention_lab.evals.generation_eval import generate_text
from attention_lab.models.gpt import GPT, config_from_dict
from attention_lab.training.config import save_config
from attention_lab.training.train import train
from attention_lab.training.verify_run import verify_run


@pytest.mark.integration
def test_tiny_train_passes_step_to_multi_qkv_attention(tmp_path, write_tiny_shards, tiny_config):
    data_root = tmp_path / "data"
    write_tiny_shards(data_root)
    config = tiny_config(tmp_path, data_root, max_steps=1)
    config["model"].update(
        {
            "attention_type": "multi_qkv_train_rotation_3track_global",
            "qkv_track_count": 3,
            "qkv_global_bank": True,
            "qkv_route_formula": "layer_plus_step_train_layer_eval",
            "n_layer": 3,
            "n_head": 2,
        }
    )
    config["diagnostics"] = {"attention_diagnostics_every": 1}
    config["run"]["name"] = "tiny_multi_qkv_train_rotation"
    config["run"]["out_dir"] = str(tmp_path / "runs" / "tiny_multi_qkv_train_rotation")
    config_path = tmp_path / "tiny_multi_qkv.yaml"
    save_config(config, config_path)

    train(config_path, overwrite=True)
    verify_run(config["run"]["out_dir"], expect_complete_training=True, expect_sample=True)

    diagnostics_path = tmp_path / "runs" / "tiny_multi_qkv_train_rotation" / "evals" / "attention_diagnostics.jsonl"
    rows = [json.loads(line) for line in diagnostics_path.read_text(encoding="utf-8").splitlines()]
    assert rows
    assert {row["step"] for row in rows} == {1}
    assert any(row["active_track_index"] == (row["layer_idx"] + 1) % 3 for row in rows)


def test_verify_data_config_option_loads_config(tmp_path, write_tiny_shards, tiny_config):
    from attention_lab.training.verify_data import main

    data_root = tmp_path / "data"
    write_tiny_shards(data_root)
    config = tiny_config(tmp_path, data_root)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    import sys

    old_argv = sys.argv
    try:
        sys.argv = ["verify_data.py", "--config", str(config_path)]
        main()
    finally:
        sys.argv = old_argv


def test_generation_path_uses_eval_freeze_for_train_rotation():
    class TinyEncoding:
        def encode(self, prompt):
            return [1, 2]

        def decode(self, tokens):
            return " ".join(str(token) for token in tokens)

    model_config = config_from_dict(
        {
            "attention_type": "multi_qkv_train_rotation_3track_global",
            "qkv_track_count": 3,
            "qkv_global_bank": True,
            "qkv_route_formula": "layer_plus_step_train_layer_eval",
            "block_size": 8,
            "n_layer": 3,
            "n_head": 2,
            "n_embd": 16,
            "dropout": 0.0,
            "bias": False,
        },
        {"vocab_size": 64},
    )
    model = GPT(model_config)
    generate_text(
        model,
        TinyEncoding(),
        prompt="tiny",
        num_samples=1,
        max_new_tokens=1,
        top_k=5,
        temperature=1.0,
        device="cpu",
        device_type="cpu",
        dtype=torch.float32,
        seed=123,
    )
    rows = [block.attn.last_diagnostics for block in model.transformer.h]
    assert rows
    assert all(row["eval_freeze_mode"] is True for row in rows)
    assert [row["active_track_index"] for row in rows] == [0, 1, 2]
    assert all(row["schedule_mode"] == "generate" for row in rows)


def test_position_rotation_generation_uses_cropped_window_relative_positions():
    class TinyEncoding:
        def encode(self, prompt):
            return [1, 2, 3, 4, 5, 6]

        def decode(self, tokens):
            return " ".join(str(token) for token in tokens)

    model_config = config_from_dict(
        {
            "attention_type": "multi_qkv_position_rotation_3track_global",
            "qkv_track_count": 3,
            "qkv_global_bank": True,
            "qkv_route_formula": "layer_plus_position",
            "block_size": 4,
            "n_layer": 3,
            "n_head": 2,
            "n_embd": 16,
            "dropout": 0.0,
            "bias": False,
        },
        {"vocab_size": 64},
    )
    model = GPT(model_config)

    generate_text(
        model,
        TinyEncoding(),
        prompt="long tiny prompt",
        num_samples=1,
        max_new_tokens=1,
        top_k=5,
        temperature=1.0,
        device="cpu",
        device_type="cpu",
        dtype=torch.float32,
        seed=123,
    )

    rows = [block.attn.last_diagnostics for block in model.transformer.h]
    assert rows
    assert all(row["schedule_mode"] == "generate" for row in rows)
    assert rows[0]["active_track_index"] is None
    # The generation path has no KV cache: it crops to a full context window and
    # GPT.forward recomputes positions as 0..T-1 for that window.
    assert rows[0]["active_track_counts"] == {"0": 2, "1": 1, "2": 1}
    assert rows[1]["active_track_counts"] == {"0": 1, "1": 2, "2": 1}
