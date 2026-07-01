from __future__ import annotations

from attention_lab.models.attention.multi_qkv_common import MultiQKVBaseCausalSelfAttention, MultiQKVRouteContext


class MultiQKVStaticGlobalCausalSelfAttention(MultiQKVBaseCausalSelfAttention):
    attention_type = "multi_qkv_static_3track_global"
    route_formula = "layer_idx % track_count"

    def select_scalar_track(self, context: MultiQKVRouteContext) -> int:
        return context.layer_idx % self.track_count
