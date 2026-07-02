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
from attention_lab.mechanisms.task_schema import TaskExample


FEATURE_POOLING_STRATEGIES = ("mean_sequence", "final_token", "answer_position", "patch_positions_mean")
TASK_ALIGNED_FEATURE_POOLING = ("answer_position", "patch_positions_mean")


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
    feature_pooling: str
    task_aligned: bool


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
    examples: list[TaskExample] | None = None,
    feature_pooling: str = "mean_sequence",
) -> ActivationFeatureSet:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if feature_pooling not in FEATURE_POOLING_STRATEGIES:
        raise ValueError(f"unknown feature pooling strategy {feature_pooling!r}")
    if examples is not None and len(examples) != len(texts):
        raise ValueError("examples length must match texts length for task-aligned feature pooling")
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
        feature_matrix = tensor_to_feature_matrix(
            tensor,
            expected_batch=len(texts),
            lengths=encoded.lengths,
            examples=examples,
            feature_pooling=feature_pooling,
        )
        if feature_matrix is None:
            missing[key] = f"site {key} tensor shape {tuple(tensor.shape)} is not example-indexed"
            continue
        features[key] = feature_matrix
        summaries[key] = {
            "site": key,
            "shape": list(tensor.shape),
            "feature_shape": list(feature_matrix.shape),
            "dtype": str(tensor.dtype).replace("torch.", ""),
            "feature_pooling": feature_pooling,
            "task_aligned_pooling": feature_pooling in TASK_ALIGNED_FEATURE_POOLING,
        }
    return ActivationFeatureSet(
        features=features,
        record_summaries=summaries,
        missing_sites=missing,
        tokenizer=loaded.tokenizer,
        feature_pooling=feature_pooling,
        task_aligned=feature_pooling in TASK_ALIGNED_FEATURE_POOLING,
    )


def tensor_to_feature_matrix(
    tensor: torch.Tensor,
    *,
    expected_batch: int,
    lengths: list[int] | None = None,
    examples: list[TaskExample] | None = None,
    feature_pooling: str = "mean_sequence",
) -> np.ndarray | None:
    value = tensor.detach().cpu().float()
    if value.ndim == 0 or value.shape[0] != expected_batch:
        return None
    if value.ndim == 1:
        return value.reshape(expected_batch, 1).numpy()
    if value.ndim == 2:
        return value.numpy()
    if value.ndim == 3:
        return _pool_sequence_tensor(
            value,
            expected_batch=expected_batch,
            token_dim=1,
            lengths=lengths,
            examples=examples,
            feature_pooling=feature_pooling,
        )
    if value.ndim == 4:
        return _pool_sequence_tensor(
            value,
            expected_batch=expected_batch,
            token_dim=2,
            lengths=lengths,
            examples=examples,
            feature_pooling=feature_pooling,
        )
    return value.reshape(expected_batch, -1).numpy()


def _pool_sequence_tensor(
    value: torch.Tensor,
    *,
    expected_batch: int,
    token_dim: int,
    lengths: list[int] | None,
    examples: list[TaskExample] | None,
    feature_pooling: str,
) -> np.ndarray:
    if feature_pooling == "mean_sequence":
        return value.mean(dim=token_dim).reshape(expected_batch, -1).numpy()
    if lengths is None:
        raise ValueError(f"{feature_pooling} pooling requires encoded sequence lengths")
    rows = []
    for index in range(expected_batch):
        positions = _pool_positions(
            index=index,
            length=lengths[index],
            examples=examples,
            feature_pooling=feature_pooling,
        )
        rows.append(_take_token_positions(value[index], token_dim=token_dim - 1, positions=positions).reshape(-1))
    return torch.stack(rows, dim=0).numpy()


def _take_token_positions(value: torch.Tensor, *, token_dim: int, positions: list[int]) -> torch.Tensor:
    index = torch.tensor(positions, dtype=torch.long)
    selected = value.index_select(token_dim, index)
    return selected.mean(dim=token_dim)


def _pool_positions(
    *,
    index: int,
    length: int,
    examples: list[TaskExample] | None,
    feature_pooling: str,
) -> list[int]:
    if length <= 0:
        raise ValueError("cannot pool an empty encoded sequence")
    if feature_pooling == "final_token":
        return [length - 1]
    if examples is None:
        raise ValueError(f"{feature_pooling} pooling requires task examples with metadata")
    example = examples[index]
    metadata = example.metadata
    if feature_pooling == "answer_position":
        if example.variant == "pos" and "clean_answer_position" in metadata:
            positions = [metadata["clean_answer_position"]]
        elif example.variant == "neg" and "corrupted_answer_position" in metadata:
            positions = [metadata["corrupted_answer_position"]]
        else:
            positions = [length - 1]
    elif feature_pooling == "patch_positions_mean":
        if example.variant == "pos":
            positions = metadata.get("clean_patch_token_indices", metadata.get("patch_token_indices"))
        elif example.variant == "neg":
            positions = metadata.get("corrupted_patch_token_indices", metadata.get("patch_token_indices"))
        else:
            positions = [length - 1]
        if not isinstance(positions, list):
            raise ValueError(f"{feature_pooling} pooling requires patch token metadata")
    else:
        raise ValueError(f"unsupported feature pooling strategy {feature_pooling!r}")
    clean_positions: list[int] = []
    for position in positions:
        if isinstance(position, bool) or not isinstance(position, int):
            raise ValueError(f"{feature_pooling} pooling has non-integer token position for example {index}")
        if position < 0 or position >= length:
            raise ValueError(
                f"{feature_pooling} pooling token position {position} out of range for encoded length {length}"
            )
        clean_positions.append(position)
    return clean_positions
