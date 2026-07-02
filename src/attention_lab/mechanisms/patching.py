from __future__ import annotations

from attention_lab.mechanisms.cache import ActivationCache
from attention_lab.mechanisms.interventions import InterventionKind, InterventionSpec


def make_cache_patch(
    source_cache: ActivationCache,
    *,
    site: str,
    layer: int | None = None,
    source_site: str | None = None,
    batch_indices: list[int] | None = None,
    token_indices: list[int] | None = None,
) -> InterventionSpec:
    return InterventionSpec(
        site=site,
        layer=layer,
        kind=InterventionKind.PATCH_FROM_CACHE,
        source_cache=source_cache,
        source_site=source_site,
        batch_indices=batch_indices,
        token_indices=token_indices,
    )
