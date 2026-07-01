from __future__ import annotations

import torch

from attention_lab.models.attention.multi_qkv_common import MultiQKVBaseCausalSelfAttention, MultiQKVRouteContext


class MultiQKVPositionRotationGlobalCausalSelfAttention(MultiQKVBaseCausalSelfAttention):
    attention_type = "multi_qkv_position_rotation_3track_global"
    route_formula = "(layer_idx + position) % track_count"
    position_routing_enabled = True

    def select_position_tracks(
        self,
        context: MultiQKVRouteContext,
        *,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        if context.position_ids is None:
            position_ids = torch.arange(seq_len, dtype=torch.long, device=device)
        else:
            position_ids = context.position_ids.to(device=device, dtype=torch.long)
            if position_ids.ndim != 1:
                raise ValueError("position_ids must be a 1D tensor for Multi-QKV position routing")
            if position_ids.numel() != seq_len:
                raise ValueError(
                    f"position_ids length {position_ids.numel()} does not match seq_len {seq_len}"
                )
        return (context.layer_idx + position_ids) % self.track_count
