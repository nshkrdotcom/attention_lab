from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def pytest_addoption(parser):
    parser.addoption("--run-integration", action="store_true", default=False, help="run integration tests")


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: tests that run tiny end-to-end training/eval flows")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-integration"):
        return
    skip_integration = pytest.mark.skip(reason="need --run-integration option to run")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def write_tiny_shards():
    def _write(root: Path, vocab_size: int = 64, train_tokens: int = 256, val_tokens: int = 128) -> None:
        root.mkdir(parents=True, exist_ok=True)
        np.save(root / "edufineweb_train_000001.npy", np.arange(train_tokens, dtype=np.uint16) % vocab_size)
        np.save(root / "edufineweb_val_000000.npy", np.arange(val_tokens, dtype=np.uint16) % vocab_size)

    return _write


@pytest.fixture
def tiny_config():
    def _config(tmp_path: Path, data_root: Path, max_steps: int = 2) -> dict:
        return {
            "run": {
                "name": "tiny_test_run",
                "out_dir": str(tmp_path / "runs" / "tiny_test_run"),
                "seed": 123,
            },
            "data": {
                "data_root": str(data_root),
                "tokenizer": "gpt2",
                "vocab_size": 64,
                "train_tokens": 256,
                "val_tokens": 128,
            },
            "model": {
                "attention_type": "standard",
                "block_size": 16,
                "n_layer": 1,
                "n_head": 1,
                "n_embd": 32,
                "dropout": 0.0,
                "bias": False,
            },
            "train": {
                "device": "cpu",
                "dtype": "float32",
                "compile": False,
                "eval_at_start": True,
                "B": 2,
                "T": 8,
                "total_batch_size": 16,
                "max_steps": max_steps,
                "grad_clip": 1.0,
                "weight_decay": 0.1,
                "learning_rate": 0.0006,
                "min_lr": 0.00006,
                "warmup_steps": 1,
                "val_every": 1,
                "val_steps": 1,
                "save_every": max_steps,
                "log_every": 1,
            },
            "sample": {
                "sample_every": max_steps,
                "prompt": "!",
                "num_samples": 1,
                "max_new_tokens": 4,
                "top_k": 10,
                "temperature": 1.0,
                "seed": 42,
            },
        }

    return _config
