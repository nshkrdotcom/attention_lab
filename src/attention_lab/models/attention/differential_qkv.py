from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.nn import functional as F


class DifferentialQKVAntiValueCausalSelfAttention(nn.Module):
    """Causal attention with positive and subtractive QKV branches."""

    def __init__(self, config):
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_size = config.n_embd // config.n_head
        self.share_value = bool(getattr(config, "diff_qkv_share_value", False))

        out_features = 5 * config.n_embd if self.share_value else 6 * config.n_embd
        self.c_attn = nn.Linear(config.n_embd, out_features, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.c_proj.NANOGPT_SCALE_INIT = 1

        init = float(getattr(config, "diff_qkv_lambda_init", 0.5))
        raw_init = math.log(math.expm1(max(init, 1e-6)))
        self.lambda_raw = nn.Parameter(torch.tensor(raw_init, dtype=torch.float32))
        if not bool(getattr(config, "diff_qkv_lambda_trainable", True)):
            self.lambda_raw.requires_grad_(False)

        self.attn_dropout = float(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self._last_diag: dict[str, float | int | str] = {}

    def _split_heads(self, tensor: torch.Tensor, batch_size: int, seq_len: int) -> torch.Tensor:
        return tensor.view(batch_size, seq_len, self.n_head, self.head_size).transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        *,
        step: int | None = None,
        positions: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        schedule_mode: str | None = None,
        layer_idx: int | None = None,
        activation_recorder=None,
    ) -> torch.Tensor:
        del positions, position_ids, schedule_mode
        batch_size, seq_len, channels = x.size()

        parts = self.c_attn(x).split(self.n_embd, dim=2)
        if self.share_value:
            q_pos, k_pos, v_shared, q_neg, k_neg = parts
            v_pos = v_shared
            v_neg = v_shared
        else:
            q_pos, k_pos, v_pos, q_neg, k_neg, v_neg = parts

        q_pos = self._split_heads(q_pos, batch_size, seq_len)
        k_pos = self._split_heads(k_pos, batch_size, seq_len)
        v_pos = self._split_heads(v_pos, batch_size, seq_len)
        q_neg = self._split_heads(q_neg, batch_size, seq_len)
        k_neg = self._split_heads(k_neg, batch_size, seq_len)
        v_neg = self._split_heads(v_neg, batch_size, seq_len)
        if activation_recorder is not None:
            q_pos = activation_recorder.record("pos_q", q_pos, layer=layer_idx)
            k_pos = activation_recorder.record("pos_k", k_pos, layer=layer_idx)
            v_pos = activation_recorder.record("pos_v", v_pos, layer=layer_idx)
            q_neg = activation_recorder.record("neg_q", q_neg, layer=layer_idx)
            k_neg = activation_recorder.record("neg_k", k_neg, layer=layer_idx)
            v_neg = activation_recorder.record("neg_v", v_neg, layer=layer_idx)

        y_pos = F.scaled_dot_product_attention(
            q_pos,
            k_pos,
            v_pos,
            attn_mask=None,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=True,
        )
        y_neg = F.scaled_dot_product_attention(
            q_neg,
            k_neg,
            v_neg,
            attn_mask=None,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=True,
        )

        lambda_value = F.softplus(self.lambda_raw).to(dtype=y_pos.dtype)
        if activation_recorder is not None:
            lambda_value = activation_recorder.record("lambda", lambda_value.reshape(()), layer=layer_idx)
        pos_flat = y_pos.transpose(1, 2).contiguous().view(batch_size, seq_len, channels)
        neg_flat = y_neg.transpose(1, 2).contiguous().view(batch_size, seq_len, channels)
        if activation_recorder is not None:
            pos_flat = activation_recorder.record("pos_out", pos_flat, layer=layer_idx)
            neg_flat = activation_recorder.record("neg_out", neg_flat, layer=layer_idx)
        y = pos_flat - lambda_value * neg_flat
        if activation_recorder is not None:
            y = activation_recorder.record("branch_delta", y, layer=layer_idx)
        y = self.resid_dropout(self.c_proj(y))

        pos_norm = y_pos.detach().float().norm()
        neg_norm = y_neg.detach().float().norm()
        self._last_diag = {
            "attention_type": "differential_qkv_anti_value",
            "diff_lambda": float(lambda_value.detach().float().cpu()),
            "pos_output_norm": float(pos_norm.cpu()),
            "neg_output_norm": float(neg_norm.cpu()),
            "neg_to_pos_output_norm_ratio": float((neg_norm / pos_norm.clamp_min(1e-12)).cpu()),
            "branch_output_delta": float((y_pos - y_neg).detach().float().norm().cpu()),
        }
        if layer_idx is not None:
            self._last_diag["layer"] = int(layer_idx)
        if step is not None:
            self._last_diag["step"] = int(step)
        return y

    def attention_diagnostics(self, *, step: int, layer: int) -> dict[str, float | int | str] | None:
        if not self._last_diag:
            return None
        row = dict(self._last_diag)
        row["step"] = int(step)
        row["layer"] = int(layer)
        return row
