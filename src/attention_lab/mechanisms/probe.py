from __future__ import annotations

from pathlib import Path

import torch

from attention_lab.mechanisms.cache import ActivationCache
from attention_lab.mechanisms.interventions import InterventionKind, InterventionSpec


def parse_index_list(value: str | None) -> list[int] | None:
    if value is None or not value.strip():
        return None
    indices: list[int] = []
    for part in value.split(","):
        raw = part.strip()
        if not raw:
            continue
        try:
            indices.append(int(raw))
        except ValueError as exc:
            raise ValueError(f"index lists must contain integers, got {raw!r}") from exc
    return indices or None


def encode_prompts(
    prompts: list[str],
    *,
    tokenizer_name: str,
    block_size: int,
    vocab_size: int,
) -> torch.Tensor:
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    if tokenizer_name != "gpt2":
        raise ValueError(f"unsupported tokenizer for mechanism probe: {tokenizer_name!r}")

    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    encoded = [enc.encode(prompt)[:block_size] for prompt in prompts]
    max_len = max(1, max(len(tokens) for tokens in encoded))
    padded = []
    max_token_id = 0
    for tokens in encoded:
        if not tokens:
            tokens = [0]
        max_token_id = max(max_token_id, max(tokens))
        padded.append(tokens + [0] * (max_len - len(tokens)))
    if max_token_id >= vocab_size:
        raise ValueError(
            f"tokenizer {tokenizer_name!r} produced token id {max_token_id}, "
            f"which exceeds configured vocab_size={vocab_size}"
        )
    return torch.tensor(padded, dtype=torch.long)


def tokenizer_metadata(*, tokenizer_name: str, block_size: int, vocab_size: int) -> dict[str, int | str]:
    return {
        "tokenizer": tokenizer_name,
        "block_size": int(block_size),
        "vocab_size": int(vocab_size),
        "pad_token_id": 0,
    }


def load_source_cache(path: str | Path | None) -> ActivationCache | None:
    if path is None:
        return None
    cache_path = Path(path)
    if not cache_path.exists():
        raise ValueError(f"source cache does not exist: {cache_path}")
    return ActivationCache.load(cache_path)


def load_replacement_tensor(path: str | Path | None) -> torch.Tensor | None:
    if path is None:
        return None
    tensor_path = Path(path)
    if not tensor_path.exists():
        raise ValueError(f"replacement tensor does not exist: {tensor_path}")
    payload = torch.load(tensor_path, map_location="cpu", weights_only=False)
    if isinstance(payload, torch.Tensor):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("tensor"), torch.Tensor):
        return payload["tensor"]
    raise ValueError(
        f"replacement tensor file must contain a torch.Tensor or a dict with tensor key: {tensor_path}"
    )


def build_probe_intervention_specs(
    *,
    sites: list[str],
    intervention_names: list[str],
    layer: int,
    scale: float | None,
    source_cache: ActivationCache | None,
    source_site: str | None,
    replacement_tensor: torch.Tensor | None,
    batch_indices: list[int] | None,
    token_indices: list[int] | None,
) -> list[InterventionSpec]:
    specs: list[InterventionSpec] = []
    for raw_name in intervention_names:
        name = raw_name.strip()
        if not name:
            continue
        try:
            kind = InterventionKind(name)
        except ValueError as exc:
            supported = ", ".join(kind.value for kind in InterventionKind)
            raise ValueError(f"unknown intervention {name!r}; supported interventions: {supported}") from exc

        if kind == InterventionKind.SCALE and scale is None:
            raise ValueError("scale requires --scale")
        if kind == InterventionKind.REPLACE and replacement_tensor is None and source_cache is None:
            raise ValueError("replace requires --replacement-tensor or --source-cache")
        if kind == InterventionKind.PATCH_FROM_CACHE and source_cache is None:
            raise ValueError("patch_from_cache requires --source-cache")

        for site in sites:
            specs.append(
                InterventionSpec(
                    site=site,
                    layer=layer,
                    kind=kind,
                    scale=scale if kind == InterventionKind.SCALE else None,
                    value=replacement_tensor if kind == InterventionKind.REPLACE else None,
                    source_cache=source_cache if kind in {InterventionKind.REPLACE, InterventionKind.PATCH_FROM_CACHE} else None,
                    source_site=source_site,
                    batch_indices=batch_indices if kind == InterventionKind.PATCH_FROM_CACHE else None,
                    token_indices=token_indices if kind == InterventionKind.PATCH_FROM_CACHE else None,
                )
            )
    return specs
