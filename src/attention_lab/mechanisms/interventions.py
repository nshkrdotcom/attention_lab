from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch

from attention_lab.mechanisms.cache import ActivationCache


class InterventionKind(Enum):
    ZERO = "zero"
    MEAN_ABLATE = "mean_ablate"
    REPLACE = "replace"
    SCALE = "scale"
    PATCH_FROM_CACHE = "patch_from_cache"


@dataclass
class InterventionSpec:
    site: str
    kind: InterventionKind
    layer: int | None = None
    value: torch.Tensor | None = None
    scale: float | None = None
    source_cache: ActivationCache | None = None
    source_site: str | None = None
    batch_indices: list[int] | None = None
    token_indices: list[int] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class InterventionResult:
    logits: torch.Tensor
    loss: torch.Tensor | None
    applied_interventions: list[dict[str, Any]]
    missing_or_failed_interventions: list[dict[str, Any]]
    before_after_summaries: dict[str, dict[str, Any]]
    after_cache: ActivationCache


def run_with_interventions(
    model,
    input_ids: torch.Tensor,
    interventions: list[InterventionSpec],
    *,
    labels: torch.Tensor | None = None,
    capture_sites: list[str] | None = None,
    step: int | None = None,
    schedule_mode: str | None = None,
) -> InterventionResult:
    from attention_lab.mechanisms.capture import ActivationRecorder

    if not interventions and capture_sites is None:
        logits, loss = model(input_ids, labels, step=step, schedule_mode=schedule_mode)
        recorder = ActivationRecorder(model=model, sites=[], interventions=[])
        return InterventionResult(
            logits=logits,
            loss=loss,
            applied_interventions=[],
            missing_or_failed_interventions=[],
            before_after_summaries={},
            after_cache=recorder.to_cache(),
        )

    wanted_sites = list(capture_sites or [])
    for spec in interventions:
        if spec.site not in wanted_sites:
            wanted_sites.append(spec.site)
    recorder = ActivationRecorder(model=model, sites=wanted_sites, interventions=interventions, detach=True)
    logits, loss = model(
        input_ids,
        labels,
        step=step,
        schedule_mode=schedule_mode,
        activation_recorder=recorder,
    )
    recorder.finalize()
    return InterventionResult(
        logits=logits,
        loss=loss,
        applied_interventions=recorder.applied_interventions,
        missing_or_failed_interventions=recorder.failed_interventions + recorder.missing_intervention_summaries(),
        before_after_summaries=recorder.before_after_summaries,
        after_cache=recorder.to_cache(),
    )
