from __future__ import annotations

import json

import torch

from attention_lab.models.gpt import GPT, config_from_dict
from attention_lab.training.attention_diagnostics import append_attention_diagnostics, collect_attention_diagnostics


def test_attention_diagnostics_collects_cp_rows_after_backward(tmp_path):
    config = config_from_dict(
        {
            "attention_type": "cp_trilinear",
            "cp_rank": 4,
            "cp_lambda_init": 1.0,
            "cp_lambda_trainable": True,
            "cp_lambda_fixed": False,
            "block_size": 8,
            "n_layer": 2,
            "n_head": 2,
            "n_embd": 16,
            "dropout": 0.0,
            "bias": False,
        },
        {"vocab_size": 64},
    )
    model = GPT(config)
    idx = torch.randint(0, 64, (2, 8))
    _, loss = model(idx, idx)
    assert loss is not None
    loss.backward()

    rows = collect_attention_diagnostics(model, step=1)
    assert len(rows) == 2
    required = {
        "attention_type",
        "step",
        "layer",
        "lambda_value",
        "cp_score_mean",
        "cp_score_std",
        "standard_score_mean",
        "standard_score_std",
        "cp_to_standard_score_std_ratio",
        "attention_entropy_mean",
        "attention_entropy_std",
        "cp_parameter_norm",
        "cp_gradient_norm",
    }
    assert required.issubset(rows[0])
    assert rows[0]["attention_type"] == "cp_trilinear"
    assert rows[0]["cp_gradient_norm"] is not None

    path = append_attention_diagnostics(tmp_path, rows)
    assert path is not None
    written = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(written) == 2


def test_append_attention_diagnostics_enriches_run_metadata(tmp_path):
    out_dir = tmp_path / "runs" / "experiments" / "E002_multitrack_qkv_shift_register" / "tiny_qkv_run"
    path = append_attention_diagnostics(
        out_dir,
        [{"attention_type": "multi_qkv_static_3track_global", "step": 1, "layer": 0}],
    )

    assert path is not None
    row = json.loads(path.read_text(encoding="utf-8").strip())
    assert row["schema_version"] == 1
    assert row["experiment_id"] == "E002_multitrack_qkv_shift_register"
    assert row["run_name"] == "tiny_qkv_run"
