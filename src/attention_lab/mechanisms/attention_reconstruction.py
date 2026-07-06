from __future__ import annotations

import math

import torch


def reconstruct_standard_attention_weights(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Reconstruct standard attention's post-softmax weights from captured Q/K.

    standard attention uses F.scaled_dot_product_attention, a fused kernel
    that never exposes the intermediate attention matrix -- unlike the CP
    and multi-QKV families, which already compute it unfused (see
    attn_weights[layer] recorded directly in cp_common.py and
    multi_qkv_common.py). This is safe to reconstruct externally because
    the formula has no extra branch to get wrong: causal
    softmax(q @ k^T / sqrt(head_dim)), the textbook case.
    """
    head_dim = q.shape[-1]
    seq_len = q.shape[-2]
    scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(head_dim))
    causal_mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=q.device).tril()
    scores = scores.masked_fill(~causal_mask, float("-inf"))
    return torch.softmax(scores, dim=-1)
