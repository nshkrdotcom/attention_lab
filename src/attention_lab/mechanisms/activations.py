from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from attention_lab.mechanisms.capture import capture_activations
from attention_lab.mechanisms.controls import ActivationMatrix
from attention_lab.mechanisms.hook_sites import format_site_name, get_hook_site_spec, site_base
from attention_lab.mechanisms.linear_probe import LinearProbeDataset
from attention_lab.mechanisms.probe import encode_prompts, tokenizer_metadata
from attention_lab.mechanisms.task_schema import TaskExample
from attention_lab.models.gpt import GPT, config_from_dict
from attention_lab.training.checkpointing import load_checkpoint
from attention_lab.training.config import load_config


@dataclass(frozen=True)
class LoadedMechanismModel:
    model: GPT
    config: dict[str, Any]
    config_path: Path
    checkpoint_path: Path
    tokenizer_name: str
    block_size: int
    vocab_size: int
    attention_type: str


def load_mechanism_model(
    *,
    config_path: str | Path,
    checkpoint_path: str | Path,
    device: str,
) -> LoadedMechanismModel:
    config_file = Path(config_path)
    checkpoint_file = Path(checkpoint_path)
    if not config_file.exists():
        raise ValueError(f"config does not exist: {config_file}")
    if not checkpoint_file.exists():
        raise ValueError(f"checkpoint does not exist: {checkpoint_file}")
    config = load_config(config_file)
    model_config = config_from_dict(config["model"], config["data"])
    model = GPT(model_config)
    checkpoint = load_checkpoint(checkpoint_file, device=device)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    tokenizer_name = str(config.get("data", {}).get("tokenizer", "gpt2"))
    return LoadedMechanismModel(
        model=model,
        config=config,
        config_path=config_file,
        checkpoint_path=checkpoint_file,
        tokenizer_name=tokenizer_name,
        block_size=int(model_config.block_size),
        vocab_size=int(model_config.vocab_size),
        attention_type=str(model_config.attention_type),
    )


def capture_feature_matrices(
    loaded: LoadedMechanismModel,
    examples: tuple[TaskExample, ...],
    *,
    sites: list[str],
    layer: int,
    batch_size: int,
    device: str,
) -> dict[str, ActivationMatrix]:
    if not examples:
        raise ValueError("need at least one task example")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    site_keys = [format_site_name(site_base(site), layer=layer) for site in sites]
    features: dict[str, list[np.ndarray]] = {key: [] for key in site_keys}
    tokenizer_info = tokenizer_metadata(
        tokenizer_name=loaded.tokenizer_name,
        block_size=loaded.block_size,
        vocab_size=loaded.vocab_size,
    )
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            batch = examples[start : start + batch_size]
            input_ids = encode_prompts(
                [example.text for example in batch],
                tokenizer_name=loaded.tokenizer_name,
                block_size=loaded.block_size,
                vocab_size=loaded.vocab_size,
            ).to(device)
            capture = capture_activations(
                loaded.model,
                input_ids,
                sites=[site_base(site) for site in sites],
                detach=True,
                cpu=True,
                checkpoint_path=loaded.checkpoint_path,
                batch_metadata={"tokenizer": tokenizer_info},
                schedule_mode="eval",
            )
            for key in site_keys:
                if key not in capture.cache.records:
                    continue
                record = capture.cache.records[key]
                pooled = pool_activation_tensor(record.tensor, expected_batch=len(batch))
                features[key].append(pooled)
    matrices: dict[str, ActivationMatrix] = {}
    for key, chunks in features.items():
        if not chunks:
            continue
        X = np.concatenate(chunks, axis=0).astype(np.float32)
        spec = get_hook_site_spec(loaded.attention_type, key)
        tensor_kind = spec.tensor_kind if spec is not None else "activation"
        matrices[key] = ActivationMatrix(site=key, X=X, tensor_kind=tensor_kind, shape=tuple(X.shape))
    return matrices


def pool_activation_tensor(tensor: torch.Tensor, *, expected_batch: int) -> np.ndarray:
    if not torch.is_floating_point(tensor):
        raise ValueError("discrete route/index tensors are not continuous probe activations")
    value = tensor.detach().float().cpu()
    if value.shape[0] != expected_batch:
        raise ValueError(
            f"activation tensor first dimension {value.shape[0]} does not match batch size {expected_batch}; "
            "parameter or scalar hook sites are not per-example activations"
        )
    if value.ndim == 2:
        pooled = value
    elif value.ndim == 3:
        pooled = value.mean(dim=1)
    elif value.ndim == 4:
        pooled = value.mean(dim=2).reshape(value.shape[0], -1)
    else:
        pooled = value.reshape(value.shape[0], -1)
    return pooled.numpy()


def probe_dataset_from_matrix(matrix: ActivationMatrix, examples: tuple[TaskExample, ...], *, family_id: str) -> LinearProbeDataset:
    selected = [idx for idx, example in enumerate(examples) if example.family_id == family_id]
    if not selected:
        raise ValueError(f"family {family_id!r} has no examples")
    return LinearProbeDataset(
        X=matrix.X[selected],
        y=np.asarray([examples[idx].label for idx in selected], dtype=np.int64),
        pair_ids=np.asarray([examples[idx].pair_id for idx in selected]),
        template_ids=np.asarray([examples[idx].template_id for idx in selected]),
        family_ids=np.asarray([examples[idx].family_id for idx in selected]),
        variants=np.asarray([examples[idx].variant for idx in selected]),
    )
