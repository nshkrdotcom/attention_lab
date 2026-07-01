from __future__ import annotations

import argparse

import torch

from attention_lab.evals.hellaswag_eval import run_hellaswag
from attention_lab.models.gpt import GPT, config_from_dict
from attention_lab.training.checkpointing import save_checkpoint
from attention_lab.training.optim import build_optimizer


def test_hellaswag_eval_records_data_provenance(tmp_path, monkeypatch, tiny_config):
    data_path = tmp_path / "hellaswag_val.jsonl"
    data_path.write_text('{"label": 0}\n', encoding="utf-8")
    config = tiny_config(tmp_path, tmp_path / "data")
    model_config = config_from_dict(config["model"], config["data"])
    model = GPT(model_config)
    optimizer = build_optimizer(model, weight_decay=0.1, learning_rate=0.001, device_type="cpu", master_process=False)
    checkpoint = save_checkpoint(tmp_path, model, optimizer, config, step=0)

    monkeypatch.setattr("attention_lab.evals.hellaswag_data.download", lambda split: data_path)
    monkeypatch.setattr(
        "attention_lab.evals.hellaswag_data.iterate_examples_from_path",
        lambda path: iter([{"label": 0}]),
    )
    monkeypatch.setitem(
        __import__("attention_lab.evals.hellaswag_data", fromlist=["HELLASWAG_URLS"]).HELLASWAG_URLS,
        "val",
        "https://example.test/hellaswag_val.jsonl",
    )

    def fake_render_example(example):
        tokens = torch.tensor([[1, 2], [1, 3], [1, 4], [1, 5]], dtype=torch.long)
        mask = torch.tensor([[0, 1], [0, 1], [0, 1], [0, 1]], dtype=torch.long)
        return {}, tokens, mask, 0

    monkeypatch.setattr("attention_lab.evals.hellaswag_data.render_example", fake_render_example)

    result = run_hellaswag(
        argparse.Namespace(
            checkpoint=str(checkpoint),
            split="val",
            max_examples=1,
            dtype="float32",
            device="cpu",
            out=str(tmp_path / "hellaswag.json"),
        )
    )

    assert result["data_path"] == str(data_path)
    assert result["data_url"] == "https://example.test/hellaswag_val.jsonl"
    assert len(result["data_sha256"]) == 64
