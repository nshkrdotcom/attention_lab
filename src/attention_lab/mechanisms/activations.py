from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from attention_lab.mechanisms.cache import ActivationRecord
from attention_lab.mechanisms.capture import capture_activations
from attention_lab.mechanisms.probe import tokenizer_metadata
from attention_lab.models.gpt import GPT, config_from_dict
from attention_lab.training.checkpointing import load_checkpoint
from attention_lab.training.config import load_config


@dataclass(frozen=True)
class LoadedMechanismModel:
    model: GPT
    config: dict[str, Any]
    tokenizer: dict[str, int | str]
    attention_type: str
    block_size: int
    vocab_size: int


@dataclass(frozen=True)
class EncodedTexts:
    input_ids: torch.Tensor
    lengths: list[int]
    tokenizer: dict[str, int | str]


@dataclass(frozen=True)
class ActivationFeatureSet:
    features: dict[str, np.ndarray]
    record_summaries: dict[str, dict[str, Any]]
    missing_sites: dict[str, str]
    tokenizer: dict[str, int | str]


def load_mechanism_model(config_path: str | Path, checkpoint_path: str | Path, *, device: str) -> LoadedMechanismModel:
    config = load_config(config_path)
    model_config = config_from_dict(config["model"], config["data"])
    tokenizer_name = str(config.get("data", {}).get("tokenizer", "gpt2"))
    tokenizer_info = tokenizer_metadata(
        tokenizer_name=tokenizer_name,
        block_size=model_config.block_size,
        vocab_size=model_config.vocab_size,
    )
    model = GPT(model_config)
    checkpoint = load_checkpoint(checkpoint_path, device=device)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return LoadedMechanismModel(
        model=model,
        config=config,
        tokenizer=tokenizer_info,
        attention_type=model_config.attention_type,
        block_size=model_config.block_size,
        vocab_size=model_config.vocab_size,
    )


def encode_texts(
    texts: list[str],
    *,
    tokenizer_name: str,
    block_size: int,
    vocab_size: int,
) -> EncodedTexts:
    if tokenizer_name != "gpt2":
        raise ValueError(f"unsupported tokenizer for mechanism probe suite: {tokenizer_name!r}")
    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    token_lists = [enc.encode(text)[:block_size] or [0] for text in texts]
    lengths = [len(tokens) for tokens in token_lists]
    max_len = max(lengths) if lengths else 1
    max_token_id = max((max(tokens) for tokens in token_lists), default=0)
    if max_token_id >= vocab_size:
        raise ValueError(f"tokenizer produced token id {max_token_id}, exceeding vocab_size={vocab_size}")
    padded = [tokens + [0] * (max_len - len(tokens)) for tokens in token_lists]
    return EncodedTexts(
        input_ids=torch.tensor(padded, dtype=torch.long),
        lengths=lengths,
        tokenizer=tokenizer_metadata(tokenizer_name=tokenizer_name, block_size=block_size, vocab_size=vocab_size),
    )


def collect_activation_features(
    loaded: LoadedMechanismModel,
    texts: list[str],
    *,
    sites: list[str],
    checkpoint_path: str | Path,
    device: str,
    batch_size: int = 16,
) -> ActivationFeatureSet:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    encoded = encode_texts(
        texts,
        tokenizer_name=str(loaded.tokenizer["tokenizer"]),
        block_size=int(loaded.tokenizer["block_size"]),
        vocab_size=int(loaded.tokenizer["vocab_size"]),
    )
    records: dict[str, list[ActivationRecord]] = {}
    missing: dict[str, str] = {}
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            input_ids = encoded.input_ids[start : start + batch_size].to(device)
            capture = capture_activations(
                loaded.model,
                input_ids,
                sites=sites,
                detach=True,
                cpu=True,
                checkpoint_path=checkpoint_path,
                batch_metadata={"tokenizer": loaded.tokenizer},
                schedule_mode="eval",
            )
            for key, record in capture.cache.records.items():
                records.setdefault(key, []).append(record)
            for key, item in capture.missing_sites.items():
                missing[key] = item.reason

    features: dict[str, np.ndarray] = {}
    summaries: dict[str, dict[str, Any]] = {}
    for key, chunks in records.items():
        tensor = torch.cat([record.tensor for record in chunks], dim=0)
        feature_matrix = tensor_to_feature_matrix(tensor, expected_batch=len(texts))
        if feature_matrix is None:
            missing[key] = f"site {key} tensor shape {tuple(tensor.shape)} is not example-indexed"
            continue
        features[key] = feature_matrix
        summaries[key] = {
            "site": key,
            "shape": list(tensor.shape),
            "feature_shape": list(feature_matrix.shape),
            "dtype": str(tensor.dtype).replace("torch.", ""),
        }
    return ActivationFeatureSet(
        features=features,
        record_summaries=summaries,
        missing_sites=missing,
        tokenizer=loaded.tokenizer,
    )


def tensor_to_feature_matrix(tensor: torch.Tensor, *, expected_batch: int) -> np.ndarray | None:
    value = tensor.detach().cpu().float()
    if value.ndim == 0 or value.shape[0] != expected_batch:
        return None
    if value.ndim == 1:
        return value.reshape(expected_batch, 1).numpy()
    if value.ndim == 2:
        return value.numpy()
    if value.ndim == 3:
        return value.mean(dim=1).reshape(expected_batch, -1).numpy()
    if value.ndim == 4:
        return value.mean(dim=2).reshape(expected_batch, -1).numpy()
    return value.reshape(expected_batch, -1).numpy()
