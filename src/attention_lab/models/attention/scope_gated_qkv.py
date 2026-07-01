from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import functional as F


class ScopeGatedQKVCausalSelfAttention(nn.Module):
    """Causal attention with content, scope, and receiver-side write-gate streams."""

    def __init__(self, config):
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_size = config.n_embd // config.n_head

        self.c_attn = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_gate = nn.Linear(config.n_embd, config.n_embd, bias=True)
        if self.c_gate.bias is not None:
            nn.init.constant_(self.c_gate.bias, float(getattr(config, "scope_gate_bias_init", 0.0)))

        self.scope_scale = nn.Parameter(
            torch.tensor(float(getattr(config, "scope_stream_scale_init", 1.0)), dtype=torch.float32)
        )
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.c_proj.NANOGPT_SCALE_INIT = 1

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
    ) -> torch.Tensor:
        del positions, position_ids, schedule_mode
        batch_size, seq_len, channels = x.size()

        q, k, v, scope = self.c_attn(x).split(self.n_embd, dim=2)
        q = self._split_heads(q, batch_size, seq_len)
        k = self._split_heads(k, batch_size, seq_len)
        v = self._split_heads(v, batch_size, seq_len)
        scope = self._split_heads(scope, batch_size, seq_len)

        content = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=True,
        )
        scoped = F.scaled_dot_product_attention(
            q,
            k,
            scope,
            attn_mask=None,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=True,
        )

        content_flat = content.transpose(1, 2).contiguous().view(batch_size, seq_len, channels)
        scoped_flat = scoped.transpose(1, 2).contiguous().view(batch_size, seq_len, channels)
        scoped_flat = self.scope_scale.to(dtype=scoped_flat.dtype) * scoped_flat
        gate = torch.sigmoid(self.c_gate(x))
        combined = torch.cat(
            [
                content_flat,
                scoped_flat,
                content_flat * scoped_flat,
                gate * content_flat,
            ],
            dim=-1,
        )
        y = self.resid_dropout(self.c_proj(combined))

        content_norm = content_flat.detach().float().norm()
        scope_norm = scoped_flat.detach().float().norm()
        self._last_diag = {
            "attention_type": "scope_gated_qkv",
            "content_output_norm": float(content_norm.cpu()),
            "scope_output_norm": float(scope_norm.cpu()),
            "scope_to_content_norm_ratio": float((scope_norm / content_norm.clamp_min(1e-12)).cpu()),
            "gate_mean": float(gate.detach().float().mean().cpu()),
            "gate_std": float(gate.detach().float().std(unbiased=False).cpu()),
            "scope_content_interaction_norm": float((content_flat * scoped_flat).detach().float().norm().cpu()),
            "scope_stream_scale": float(self.scope_scale.detach().float().cpu()),
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
