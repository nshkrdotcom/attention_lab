from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AlignmentResult:
    available: bool
    probe_direction_cosine_to_control: float | None
    probe_direction_alignment_abs: float | None
    reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "available": self.available,
            "probe_direction_cosine_to_control": self.probe_direction_cosine_to_control,
            "probe_direction_alignment_abs": self.probe_direction_alignment_abs,
            "reason": self.reason,
            "interpretation": (
                "high alignment suggests an already-present or universal feature direction surfaced differently; "
                "low alignment is only a prompt for scrutiny and is not representational novelty evidence by itself"
            ),
        }


def probe_direction_alignment(candidate_weight: np.ndarray, control_weight: np.ndarray) -> AlignmentResult:
    candidate = np.asarray(candidate_weight, dtype=float).reshape(-1)
    control = np.asarray(control_weight, dtype=float).reshape(-1)
    if candidate.shape != control.shape:
        return AlignmentResult(
            available=False,
            probe_direction_cosine_to_control=None,
            probe_direction_alignment_abs=None,
            reason=f"shape mismatch: candidate={candidate.shape}, control={control.shape}",
        )
    candidate_norm = float(np.linalg.norm(candidate))
    control_norm = float(np.linalg.norm(control))
    if candidate_norm <= 0.0 or control_norm <= 0.0:
        return AlignmentResult(
            available=False,
            probe_direction_cosine_to_control=None,
            probe_direction_alignment_abs=None,
            reason="zero-norm probe direction",
        )
    cosine = float(np.dot(candidate, control) / (candidate_norm * control_norm))
    return AlignmentResult(
        available=True,
        probe_direction_cosine_to_control=cosine,
        probe_direction_alignment_abs=abs(cosine),
    )
