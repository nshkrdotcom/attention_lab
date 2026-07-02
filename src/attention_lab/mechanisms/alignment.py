from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AlignmentResult:
    available: bool
    probe_direction_cosine_to_control: float | None
    probe_direction_alignment_abs: float | None
    reason: str | None = None


def probe_direction_alignment(candidate_weights: np.ndarray, control_weights: np.ndarray) -> AlignmentResult:
    candidate = np.asarray(candidate_weights, dtype=float)
    control = np.asarray(control_weights, dtype=float)
    if candidate.shape != control.shape:
        return AlignmentResult(
            available=False,
            probe_direction_cosine_to_control=None,
            probe_direction_alignment_abs=None,
            reason=f"shape mismatch: candidate {candidate.shape} vs control {control.shape}; no projection/coercion applied",
        )
    candidate_norm = float(np.linalg.norm(candidate))
    control_norm = float(np.linalg.norm(control))
    if candidate_norm <= 1e-12 or control_norm <= 1e-12:
        return AlignmentResult(
            available=False,
            probe_direction_cosine_to_control=None,
            probe_direction_alignment_abs=None,
            reason="zero-norm probe direction prevents cosine alignment",
        )
    cosine = float(np.dot(candidate, control) / (candidate_norm * control_norm))
    cosine = max(-1.0, min(1.0, cosine))
    return AlignmentResult(
        available=True,
        probe_direction_cosine_to_control=cosine,
        probe_direction_alignment_abs=abs(cosine),
    )
