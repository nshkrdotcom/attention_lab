from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from attention_lab.mechanisms.cache import ActivationCache
from attention_lab.mechanisms.hook_sites import is_discrete_hook_site
from attention_lab.mechanisms.interventions import InterventionKind, InterventionSpec


@dataclass(frozen=True)
class ProbeInterventionPlan:
    specs: list[InterventionSpec]
    invalid_interventions: list[dict[str, str]]
    capture_sites: list[str]
    intervention_sites: list[str]


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
    plan = build_probe_intervention_plan(
        attention_type=None,
        capture_sites=sites,
        intervention_sites=sites,
        intervention_names=intervention_names,
        layer=layer,
        scale=scale,
        source_cache=source_cache,
        source_site=source_site,
        replacement_tensor=replacement_tensor,
        batch_indices=batch_indices,
        token_indices=token_indices,
    )
    return plan.specs


def build_probe_intervention_plan(
    *,
    attention_type: str | None,
    capture_sites: list[str],
    intervention_sites: list[str] | None,
    intervention_names: list[str],
    layer: int,
    scale: float | None,
    source_cache: ActivationCache | None,
    source_site: str | None,
    replacement_tensor: torch.Tensor | None,
    batch_indices: list[int] | None,
    token_indices: list[int] | None,
) -> ProbeInterventionPlan:
    requested_names = [name.strip() for name in intervention_names if name.strip()]
    if not requested_names:
        return ProbeInterventionPlan(
            specs=[],
            invalid_interventions=[],
            capture_sites=list(capture_sites),
            intervention_sites=[],
        )
    effective_intervention_sites = list(intervention_sites) if intervention_sites is not None else [
        site for site in capture_sites if not _is_discrete_site(attention_type, site)
    ]
    invalid_interventions = []
    if intervention_sites is None:
        invalid_interventions = [
            {
                "site": site,
                "kind": name,
                "reason": _discrete_site_reason(site),
            }
            for site in capture_sites
            if _is_discrete_site(attention_type, site)
            for name in requested_names
        ]
    else:
        discrete_sites = [site for site in effective_intervention_sites if _is_discrete_site(attention_type, site)]
        if discrete_sites:
            sites_text = ", ".join(discrete_sites)
            raise ValueError(
                f"route-index hook sites are capture-only for mechanism probes: {sites_text}. "
                "Use --sites to capture them and --intervention-sites for continuous sites."
            )
    if not effective_intervention_sites:
        raise ValueError("no intervention-capable sites remain after excluding capture-only route-index sites")

    specs: list[InterventionSpec] = []
    for name in requested_names:
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

        for site in effective_intervention_sites:
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
    return ProbeInterventionPlan(
        specs=specs,
        invalid_interventions=invalid_interventions,
        capture_sites=list(capture_sites),
        intervention_sites=effective_intervention_sites,
    )


def _is_discrete_site(attention_type: str | None, site: str) -> bool:
    if attention_type is None:
        return site.split("[", 1)[0] == "selected_track"
    return is_discrete_hook_site(attention_type, site)


def _discrete_site_reason(site: str) -> str:
    return (
        f"{site} is a discrete route-index hook site. It is captured for diagnostics, "
        "but continuous activation interventions are only applied to continuous sites."
    )
