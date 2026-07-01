from __future__ import annotations

import math

import torch

from attention_lab.models.attention.cp_common import CPScoreAugmentedCausalSelfAttention


class CPTrilinearCausalSelfAttention(CPScoreAugmentedCausalSelfAttention):
    """Standard GPT attention plus a value-conditioned low-rank CP score branch."""

    attention_type = "cp_trilinear"
    low_rank_projection_count = 3

    def extra_scores_from_factors(self, *factors: torch.Tensor) -> torch.Tensor:
        q_low, k_low, v_low = factors
        return torch.einsum("bhir,bhjr,bhjr->bhij", q_low, k_low, v_low) * (1.0 / math.sqrt(self.cp_rank))
