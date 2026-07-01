from __future__ import annotations

from attention_lab.models.gpt import GPT, config_from_dict
from attention_lab.training.checkpointing import load_checkpoint, save_checkpoint
from attention_lab.training.data_manifest import write_data_manifest
from attention_lab.training.data_loader import TokenShardLoader
from attention_lab.training.optim import build_optimizer


def tiny_model():
    config = config_from_dict(
        {
            "attention_type": "standard",
            "block_size": 8,
            "n_layer": 1,
            "n_head": 1,
            "n_embd": 16,
            "dropout": 0.0,
            "bias": False,
        },
        {"vocab_size": 64},
    )
    return GPT(config)


def test_save_and_load_checkpoint_includes_training_state(tmp_path, write_tiny_shards):
    data_root = tmp_path / "data"
    write_tiny_shards(data_root)
    model = tiny_model()
    optimizer = build_optimizer(model, weight_decay=0.1, learning_rate=0.001, device_type="cpu", master_process=False)
    loader = TokenShardLoader(data_root, B=2, T=8, process_rank=0, num_processes=1, split="train")
    loader.next_batch()
    path = save_checkpoint(
        tmp_path,
        model,
        optimizer,
        {"train": {"max_steps": 2}},
        step=1,
        train_loss=1.23,
        val_loss=1.11,
        train_loader_state=loader.state_dict(),
    )

    assert path.exists()
    assert (tmp_path / "checkpoints" / "ckpt_last.pt").exists()
    checkpoint = load_checkpoint(path)
    assert checkpoint["step"] == 1
    assert checkpoint["train_loss"] == 1.23
    assert "optimizer" in checkpoint
    assert "rng_state" in checkpoint
    assert checkpoint["train_loader_state"]["current_position"] == loader.current_position


def test_save_checkpoint_includes_run_data_manifest(tmp_path, write_tiny_shards):
    data_root = tmp_path / "data"
    write_tiny_shards(data_root)
    write_data_manifest(data_root, tmp_path / "data_manifest.json")
    model = tiny_model()
    optimizer = build_optimizer(model, weight_decay=0.1, learning_rate=0.001, device_type="cpu", master_process=False)

    path = save_checkpoint(tmp_path, model, optimizer, {"train": {"max_steps": 2}}, step=1)
    checkpoint = load_checkpoint(path)

    assert checkpoint["data_manifest"]["total_train_tokens"] == 256
    assert len(checkpoint["data_manifest_sha256"]) == 64
