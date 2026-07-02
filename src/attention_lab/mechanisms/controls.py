from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from attention_lab.mechanisms.presets import MechanismPreset


@dataclass(frozen=True)
class ControlResolution:
    expected_run_name: str | None
    canonical_control_checkpoint: Path | None
    canonical_control_config: Path | None
    control_checkpoint: Path | None
    control_config: Path | None
    is_override: bool
    is_canonical: bool
    noncanonical_reason: str | None


@dataclass(frozen=True)
class ActivationMatrix:
    site: str
    X: np.ndarray
    tensor_kind: str
    shape: tuple[int, ...]

    @property
    def feature_dim(self) -> int:
        if self.X.ndim != 2:
            raise ValueError("activation matrix must be 2D")
        return int(self.X.shape[1])


@dataclass(frozen=True)
class RandomSiteNullSelection:
    available: bool
    selected_site: str | None
    reason: str | None
    candidate_site: str
    candidate_dim: int
    selected_dim: int | None
    selected_tensor_kind: str | None


def resolve_control(
    preset: MechanismPreset,
    *,
    control_checkpoint: str | Path | None = None,
    control_config: str | Path | None = None,
    force_noncanonical: bool = False,
) -> ControlResolution:
    canonical = preset.matched_control
    expected_run_name = canonical.run_name if canonical is not None else None
    canonical_checkpoint = canonical.checkpoint if canonical is not None else None
    canonical_config = canonical.config if canonical is not None else None
    actual_checkpoint = Path(control_checkpoint) if control_checkpoint is not None else canonical_checkpoint
    actual_config = Path(control_config) if control_config is not None else canonical_config
    is_override = control_checkpoint is not None or control_config is not None
    is_canonical = canonical is not None and actual_checkpoint == canonical_checkpoint and actual_config == canonical_config
    reason = None
    if canonical is None:
        is_canonical = False
        reason = "preset has no canonical matched control"
    elif is_override and not is_canonical:
        reason = (
            "override does not match canonical matched-control pairing "
            f"{expected_run_name}; full canonical evidence claims remain capped"
        )
        if force_noncanonical:
            reason += " even with force_noncanonical=true"
    return ControlResolution(
        expected_run_name=expected_run_name,
        canonical_control_checkpoint=canonical_checkpoint,
        canonical_control_config=canonical_config,
        control_checkpoint=actual_checkpoint,
        control_config=actual_config,
        is_override=is_override,
        is_canonical=is_canonical,
        noncanonical_reason=reason,
    )


def choose_random_site_null(
    *,
    candidate_site: str,
    candidate: ActivationMatrix,
    available: dict[str, ActivationMatrix],
    seed: int,
) -> RandomSiteNullSelection:
    candidate_dim = candidate.feature_dim
    candidates = [
        matrix
        for site, matrix in available.items()
        if site != candidate_site
        and matrix.feature_dim == candidate_dim
        and matrix.tensor_kind == candidate.tensor_kind
    ]
    if not candidates:
        return RandomSiteNullSelection(
            available=False,
            selected_site=None,
            reason=(
                "no non-candidate site has matched dimensionality and compatible tensor kind; "
                "random-site null feasibility is limited by actual activation shapes"
            ),
            candidate_site=candidate_site,
            candidate_dim=candidate_dim,
            selected_dim=None,
            selected_tensor_kind=None,
        )
    rng = np.random.default_rng(seed)
    selected = candidates[int(rng.integers(0, len(candidates)))]
    return RandomSiteNullSelection(
        available=True,
        selected_site=selected.site,
        reason=None,
        candidate_site=candidate_site,
        candidate_dim=candidate_dim,
        selected_dim=selected.feature_dim,
        selected_tensor_kind=selected.tensor_kind,
    )
