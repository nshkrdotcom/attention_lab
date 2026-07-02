from __future__ import annotations

from dataclasses import dataclass

from attention_lab.mechanisms.cache import ActivationCache
from attention_lab.mechanisms.interventions import InterventionKind, InterventionSpec


@dataclass(frozen=True)
class ScalarMetric:
    value: float | None
    valid: bool
    reason: str | None = None


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


def compute_restoration_score(
    *,
    clean_logitdiff: float,
    corrupted_logitdiff: float,
    patched_logitdiff: float,
    min_abs_denominator: float = 1e-6,
) -> ScalarMetric:
    denominator = float(clean_logitdiff - corrupted_logitdiff)
    if abs(denominator) < min_abs_denominator:
        return ScalarMetric(
            value=None,
            valid=False,
            reason=(
                "restoration denominator clean_logitdiff - corrupted_logitdiff is too small; "
                "metric and dependent gates are invalid"
            ),
        )
    value = float((patched_logitdiff - corrupted_logitdiff) / denominator)
    return ScalarMetric(value=value, valid=True)


def compute_mediation_fraction(
    *,
    component_patch_restoration: float,
    full_layer_patch_restoration: float,
    min_abs_denominator: float = 1e-6,
) -> ScalarMetric:
    if abs(float(full_layer_patch_restoration)) < min_abs_denominator:
        return ScalarMetric(
            value=None,
            valid=False,
            reason="full_layer_patch_restoration is too small for a valid mediation_fraction denominator",
        )
    return ScalarMetric(
        value=float(component_patch_restoration / full_layer_patch_restoration),
        valid=True,
    )
