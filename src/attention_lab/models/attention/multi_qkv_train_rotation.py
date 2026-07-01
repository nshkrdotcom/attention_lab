from __future__ import annotations

from attention_lab.models.attention.multi_qkv_common import MultiQKVBaseCausalSelfAttention, MultiQKVRouteContext


class MultiQKVTrainRotationGlobalCausalSelfAttention(MultiQKVBaseCausalSelfAttention):
    attention_type = "multi_qkv_train_rotation_3track_global"
    route_formula = "(layer_idx + step) % track_count during train; layer_idx % track_count during eval/generate"
    eval_freeze_mode = True

    def select_scalar_track(self, context: MultiQKVRouteContext) -> int:
        if context.schedule_mode == "train":
            if context.step is None:
                raise ValueError("multi_qkv_train_rotation_3track_global requires step during training")
            return (context.layer_idx + int(context.step)) % self.track_count
        if context.schedule_mode in {"eval", "generate"}:
            return context.layer_idx % self.track_count
        raise ValueError("schedule_mode must be one of: train, eval, generate")
