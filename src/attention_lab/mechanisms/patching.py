from __future__ import annotations

from dataclasses import dataclass

from attention_lab.mechanisms.cache import ActivationCache
from attention_lab.mechanisms.interventions import InterventionKind, InterventionSpec


RESTORATION_EPSILON = 1e-6


@dataclass(frozen=True)
class RestorationMetrics:
    valid: bool
    clean_logitdiff: float | None
    corrupted_logitdiff: float | None
    patched_logitdiff: float | None
    restoration_score: float | None
    reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "valid": self.valid,
            "clean_logitdiff": self.clean_logitdiff,
            "corrupted_logitdiff": self.corrupted_logitdiff,
            "patched_logitdiff": self.patched_logitdiff,
            "restoration_score": self.restoration_score,
            "reason": self.reason,
            "formula": "(patched_logitdiff - corrupted_logitdiff) / (clean_logitdiff - corrupted_logitdiff)",
        }


@dataclass(frozen=True)
class MediationMetrics:
    valid: bool
    component_patch_restoration: float | None
    full_layer_patch_restoration: float | None
    mediation_fraction: float | None
    reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "valid": self.valid,
            "component_patch_restoration": self.component_patch_restoration,
            "full_layer_patch_restoration": self.full_layer_patch_restoration,
            "mediation_fraction": self.mediation_fraction,
            "reason": self.reason,
            "formula": "component_patch_restoration / full_layer_patch_restoration",
        }


def make_cache_patch(
    source_cache: ActivationCache,
    *,
    site: str,
    layer: int | None = None,
    source_site: str | None = None,
    batch_indices: list[int] | None = None,
    token_indices: list[int] | None = None,
    source_token_indices: list[int] | None = None,
) -> InterventionSpec:
    return InterventionSpec(
        site=site,
        layer=layer,
        kind=InterventionKind.PATCH_FROM_CACHE,
        source_cache=source_cache,
        source_site=source_site,
        batch_indices=batch_indices,
        token_indices=token_indices,
        source_token_indices=source_token_indices,
    )


def restoration_score(
    *,
    clean_logitdiff: float,
    corrupted_logitdiff: float,
    patched_logitdiff: float,
    epsilon: float = RESTORATION_EPSILON,
) -> RestorationMetrics:
    denominator = clean_logitdiff - corrupted_logitdiff
    if abs(denominator) < epsilon:
        return RestorationMetrics(
            valid=False,
            clean_logitdiff=float(clean_logitdiff),
            corrupted_logitdiff=float(corrupted_logitdiff),
            patched_logitdiff=float(patched_logitdiff),
            restoration_score=None,
            reason=f"invalid restoration denominator abs(clean-corrupted)={abs(denominator):.6g} < {epsilon}",
        )
    score = (patched_logitdiff - corrupted_logitdiff) / denominator
    return RestorationMetrics(
        valid=True,
        clean_logitdiff=float(clean_logitdiff),
        corrupted_logitdiff=float(corrupted_logitdiff),
        patched_logitdiff=float(patched_logitdiff),
        restoration_score=float(score),
    )


def mediation_fraction(
    *,
    component_patch_restoration: float | None,
    full_layer_patch_restoration: float | None,
    epsilon: float = RESTORATION_EPSILON,
) -> MediationMetrics:
    if component_patch_restoration is None or full_layer_patch_restoration is None:
        return MediationMetrics(
            valid=False,
            component_patch_restoration=component_patch_restoration,
            full_layer_patch_restoration=full_layer_patch_restoration,
            mediation_fraction=None,
            reason="component or full-layer restoration metric is unavailable",
        )
    if abs(full_layer_patch_restoration) < epsilon:
        return MediationMetrics(
            valid=False,
            component_patch_restoration=float(component_patch_restoration),
            full_layer_patch_restoration=float(full_layer_patch_restoration),
            mediation_fraction=None,
            reason=f"invalid mediation denominator abs(full_layer_patch_restoration) < {epsilon}",
        )
    return MediationMetrics(
        valid=True,
        component_patch_restoration=float(component_patch_restoration),
        full_layer_patch_restoration=float(full_layer_patch_restoration),
        mediation_fraction=float(component_patch_restoration / full_layer_patch_restoration),
    )
