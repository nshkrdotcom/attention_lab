from attention_lab.models.attention.cp_bilinear import CPBilinearCausalSelfAttention
from attention_lab.models.attention.cp_trilinear import CPTrilinearCausalSelfAttention
from attention_lab.models.attention.multi_qkv_common import MultiQKVGlobalBank
from attention_lab.models.attention.multi_qkv_position_rotation import MultiQKVPositionRotationGlobalCausalSelfAttention
from attention_lab.models.attention.multi_qkv_static import MultiQKVStaticGlobalCausalSelfAttention
from attention_lab.models.attention.multi_qkv_train_rotation import MultiQKVTrainRotationGlobalCausalSelfAttention
from attention_lab.models.attention.standard import StandardCausalSelfAttention
from attention_lab.models.attention.trilinear_cp import TrilinearCPCausalSelfAttention


def build_attention(
    config,
    *,
    layer_idx: int | None = None,
    qkv_bank: MultiQKVGlobalBank | None = None,
    shared_qkv_bank: MultiQKVGlobalBank | None = None,
):
    if qkv_bank is not None and shared_qkv_bank is not None and qkv_bank is not shared_qkv_bank:
        raise ValueError("qkv_bank and shared_qkv_bank must reference the same global bank")
    bank = qkv_bank if qkv_bank is not None else shared_qkv_bank
    attention_type = getattr(config, "attention_type", "standard")
    if attention_type == "standard":
        return StandardCausalSelfAttention(config)
    if attention_type == "cp_bilinear":
        return CPBilinearCausalSelfAttention(config)
    if attention_type == "cp_trilinear":
        return CPTrilinearCausalSelfAttention(config)
    if attention_type == "multi_qkv_static_3track_global":
        if layer_idx is None or bank is None:
            raise ValueError("multi_qkv_static_3track_global requires layer_idx and qkv_bank")
        return MultiQKVStaticGlobalCausalSelfAttention(config, layer_idx=layer_idx, qkv_bank=bank)
    if attention_type == "multi_qkv_train_rotation_3track_global":
        if layer_idx is None or bank is None:
            raise ValueError("multi_qkv_train_rotation_3track_global requires layer_idx and qkv_bank")
        return MultiQKVTrainRotationGlobalCausalSelfAttention(config, layer_idx=layer_idx, qkv_bank=bank)
    if attention_type == "multi_qkv_position_rotation_3track_global":
        if layer_idx is None or bank is None:
            raise ValueError("multi_qkv_position_rotation_3track_global requires layer_idx and qkv_bank")
        return MultiQKVPositionRotationGlobalCausalSelfAttention(config, layer_idx=layer_idx, qkv_bank=bank)
    if attention_type == "trilinear_cp":
        return TrilinearCPCausalSelfAttention(config)
    raise ValueError(f"Unknown attention_type: {attention_type}")
