from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from attention_lab.mechanisms.presets import ControlPreset, MechanismProbePreset
from attention_lab.mechanisms.presets import SitePreset


@dataclass(frozen=True)
class ControlResolution:
    expected_control: ControlPreset | None
    config_path: Path | None
    checkpoint_path: Path | None
    mode: str
    canonical: bool
    override_used: bool
    available: bool
    force_noncanonical: bool
    reason: str | None = None


@dataclass(frozen=True)
class RandomSiteSelection:
    available: bool
    selected_site: str | None
    selected_feature_dim: int | None
    candidate_site: str
    candidate_feature_dim: int
    reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "random_site_null_available": self.available,
            "selected_site": self.selected_site,
            "selected_feature_dim": self.selected_feature_dim,
            "candidate_site": self.candidate_site,
            "candidate_feature_dim": self.candidate_feature_dim,
            "reason": self.reason,
        }


def resolve_control(
    preset: MechanismProbePreset,
    *,
    control_mode: str,
    control_config: str | Path | None,
    control_checkpoint: str | Path | None,
    force_noncanonical: bool = False,
) -> ControlResolution:
    if control_mode not in {"matched", "none"}:
        raise ValueError("--control-mode must be one of: matched, none")
    expected = preset.matched_control
    if control_mode == "none":
        return ControlResolution(
            expected_control=expected,
            config_path=None,
            checkpoint_path=None,
            mode=control_mode,
            canonical=False,
            override_used=False,
            available=False,
            force_noncanonical=force_noncanonical,
            reason="matched control disabled; candidate mechanism evidence is unavailable",
        )
    if expected is None:
        return ControlResolution(
            expected_control=None,
            config_path=Path(control_config) if control_config else None,
            checkpoint_path=Path(control_checkpoint) if control_checkpoint else None,
            mode=control_mode,
            canonical=False,
            override_used=bool(control_config or control_checkpoint),
            available=False,
            force_noncanonical=force_noncanonical,
            reason="preset has no canonical matched control",
        )

    config_path = Path(control_config) if control_config else expected.config_path
    checkpoint_path = Path(control_checkpoint) if control_checkpoint else expected.checkpoint_path
    override_used = bool(control_config or control_checkpoint)
    canonical = config_path == expected.config_path and checkpoint_path == expected.checkpoint_path
    available = config_path.exists() and checkpoint_path.exists()
    reason = None
    if not available:
        missing = []
        if not config_path.exists():
            missing.append(f"config missing: {config_path}")
        if not checkpoint_path.exists():
            missing.append(f"checkpoint missing: {checkpoint_path}")
        reason = "; ".join(missing)
    elif not canonical:
        reason = "control override does not match the canonical seed-matched pairing"
    return ControlResolution(
        expected_control=expected,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        mode=control_mode,
        canonical=canonical,
        override_used=override_used,
        available=available,
        force_noncanonical=force_noncanonical,
        reason=reason,
    )


def is_seed_mismatched(preset: MechanismProbePreset, resolution: ControlResolution) -> bool:
    if resolution.canonical or resolution.checkpoint_path is None:
        return False
    expected = resolution.expected_control
    if expected is None:
        return True
    expected_seed = _seed_token(expected.run_name)
    actual_seed = _seed_token(str(resolution.checkpoint_path))
    return expected_seed is not None and actual_seed is not None and expected_seed != actual_seed


def select_random_site_null(
    *,
    candidate: SitePreset,
    candidate_key: str,
    feature_shapes: dict[str, tuple[int, int]],
    pool: tuple[SitePreset, ...],
    seed: int,
) -> RandomSiteSelection:
    candidate_shape = feature_shapes.get(candidate_key)
    if candidate_shape is None:
        return RandomSiteSelection(
            available=False,
            selected_site=None,
            selected_feature_dim=None,
            candidate_site=candidate_key,
            candidate_feature_dim=0,
            reason="candidate site features are unavailable",
        )
    candidate_dim = candidate_shape[1]
    compatible = []
    for site in pool:
        key = site.key
        if key == candidate_key or site.site == candidate.site:
            continue
        shape = feature_shapes.get(key)
        if shape is None:
            continue
        if shape[1] != candidate_dim:
            continue
        if not _compatible_site_type(candidate.tensor_kind, site.tensor_kind):
            continue
        compatible.append((key, shape[1]))
    if not compatible:
        return RandomSiteSelection(
            available=False,
            selected_site=None,
            selected_feature_dim=None,
            candidate_site=candidate_key,
            candidate_feature_dim=candidate_dim,
            reason=(
                "no non-candidate random-site null with matched dimensionality and compatible site type; "
                "this is a null-feasibility limit, not an implementation failure"
            ),
        )
    rng = np.random.default_rng(seed)
    index = int(rng.integers(0, len(compatible)))
    selected_site, selected_dim = compatible[index]
    return RandomSiteSelection(
        available=True,
        selected_site=selected_site,
        selected_feature_dim=selected_dim,
        candidate_site=candidate_key,
        candidate_feature_dim=candidate_dim,
    )


def _seed_token(value: str) -> str | None:
    for part in value.replace("/", "_").split("_"):
        if part.startswith("seed") and part[4:].isdigit():
            return part
    return None


def _compatible_site_type(candidate_kind: str, random_kind: str) -> bool:
    if candidate_kind == random_kind:
        return True
    if candidate_kind == "activation" and random_kind == "activation":
        return True
    return False
