from __future__ import annotations

import argparse

import numpy as np
import pytest
import torch

from attention_lab.training.data_loader import TokenShardLoader, load_tokens
from attention_lab.training.verify_data import verify_data_root


def test_load_tokens_uint16_returns_long_tensor(tmp_path):
    path = tmp_path / "tokens.npy"
    np.save(path, np.arange(10, dtype=np.uint16))
    tokens = load_tokens(path)
    assert tokens.dtype == torch.long
    assert tokens.tolist() == list(range(10))


def test_loader_finds_train_and_val_shards(tmp_path):
    np.save(tmp_path / "edufineweb_train_000001.npy", np.arange(64, dtype=np.uint16))
    np.save(tmp_path / "edufineweb_val_000000.npy", np.arange(64, dtype=np.uint16))
    train_loader = TokenShardLoader(tmp_path, B=2, T=8, process_rank=0, num_processes=1, split="train")
    val_loader = TokenShardLoader(tmp_path, B=2, T=8, process_rank=0, num_processes=1, split="val")
    assert len(train_loader.shards) == 1
    assert len(val_loader.shards) == 1


def test_loader_emits_shifted_batch(tmp_path):
    np.save(tmp_path / "edufineweb_train_000001.npy", np.arange(64, dtype=np.uint16))
    loader = TokenShardLoader(tmp_path, B=2, T=8, process_rank=0, num_processes=1, split="train")
    x, y = loader.next_batch()
    assert tuple(x.shape) == (2, 8)
    assert tuple(y.shape) == (2, 8)
    assert torch.equal(y.reshape(-1), x.reshape(-1) + 1)


def test_loader_advances_across_shard_boundary(tmp_path):
    np.save(tmp_path / "edufineweb_train_000001.npy", np.arange(17, dtype=np.uint16))
    np.save(tmp_path / "edufineweb_train_000002.npy", np.arange(100, 117, dtype=np.uint16))
    loader = TokenShardLoader(tmp_path, B=2, T=8, process_rank=0, num_processes=1, split="train")
    x1, _ = loader.next_batch()
    x2, _ = loader.next_batch()
    assert int(x1[0, 0]) == 0
    assert int(x2[0, 0]) == 100


def test_loader_errors_if_shard_too_short(tmp_path):
    np.save(tmp_path / "edufineweb_train_000001.npy", np.arange(8, dtype=np.uint16))
    with pytest.raises(ValueError, match="at least"):
        TokenShardLoader(tmp_path, B=2, T=8, process_rank=0, num_processes=1, split="train")


def test_loader_state_round_trips(tmp_path):
    np.save(tmp_path / "edufineweb_train_000001.npy", np.arange(64, dtype=np.uint16))
    loader = TokenShardLoader(tmp_path, B=2, T=8, process_rank=0, num_processes=1, split="train")
    first_x, _ = loader.next_batch()
    state = loader.state_dict()
    expected_x, _ = loader.next_batch()

    restored = TokenShardLoader(tmp_path, B=2, T=8, process_rank=0, num_processes=1, split="train")
    restored.load_state_dict(state)
    actual_x, _ = restored.next_batch()
    assert int(first_x[0, 0]) == 0
    assert torch.equal(actual_x, expected_x)


def test_verify_data_fails_on_missing_shards(tmp_path):
    with pytest.raises(SystemExit, match="No .npy shards"):
        verify_data_root(argparse.Namespace(data_root=str(tmp_path)))


def test_verify_data_fails_on_token_ids_outside_uint16(tmp_path):
    np.save(tmp_path / "bad.npy", np.array([0, 70000], dtype=np.int64))
    with pytest.raises(SystemExit, match="Token id too large"):
        verify_data_root(argparse.Namespace(data_root=str(tmp_path)))
