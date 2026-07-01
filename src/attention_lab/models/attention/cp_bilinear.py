from __future__ import annotations

import math

import torch

from attention_lab.models.attention.cp_common import CPScoreAugmentedCausalSelfAttention


class CPBilinearCausalSelfAttention(CPScoreAugmentedCausalSelfAttention):
    """Standard GPT attention plus a low-rank bilinear score branch."""

    attention_type = "cp_bilinear"
    low_rank_projection_count = 2

    def extra_scores_from_factors(self, *factors: torch.Tensor) -> torch.Tensor:
        q_low, k_low = factors
        return torch.einsum("bhir,bhjr->bhij", q_low, k_low) * (1.0 / math.sqrt(self.cp_rank))
