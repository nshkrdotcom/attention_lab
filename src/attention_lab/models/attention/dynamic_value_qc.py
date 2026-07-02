from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch.nn import functional as F


class DynamicValueQueryConditionedCausalSelfAttention(nn.Module):
    """Causal attention with receiver-conditioned value read mode gates."""

    allowed_gate_sources = {"x", "q", "xq"}

    def __init__(self, config: Any):
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if bool(getattr(config, "dynamic_value_pairwise_gate", False)):
            raise ValueError("dynamic_value_pairwise_gate is not implemented safely for E004 configs")

        self.n_head = int(config.n_head)
        self.n_embd = int(config.n_embd)
        self.head_size = self.n_embd // self.n_head
        self.gate_from = str(getattr(config, "dynamic_value_gate_from", "x"))
        if self.gate_from not in self.allowed_gate_sources:
            raise ValueError(f"dynamic_value_gate_from must be one of {sorted(self.allowed_gate_sources)}")
        gate_in = 2 * self.n_embd if self.gate_from == "xq" else self.n_embd

        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd, bias=config.bias)
        self.c_gate = nn.Linear(gate_in, self.n_embd, bias=True)
        if self.c_gate.bias is not None:
            nn.init.constant_(self.c_gate.bias, float(getattr(config, "dynamic_value_gate_bias_init", 0.0)))
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=config.bias)
        self.c_proj.NANOGPT_SCALE_INIT = 1

        self.attn_dropout = float(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self._last_diag: dict[str, float | int | str | bool] = {}

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
        q_flat, k_flat, v_flat = self.c_attn(x).split(self.n_embd, dim=2)
        q = self._split_heads(q_flat, batch_size, seq_len)
        k = self._split_heads(k_flat, batch_size, seq_len)
        v = self._split_heads(v_flat, batch_size, seq_len)

        content_heads = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=True,
        )
        content = content_heads.transpose(1, 2).contiguous().view(batch_size, seq_len, channels)
        gate_input = self._gate_input(x, q_flat)
        gate = torch.sigmoid(self.c_gate(gate_input))
        if activation_recorder is not None:
            content = activation_recorder.record("static_value_content", content, layer=layer_idx)
            gate = activation_recorder.record(
                "dynamic_gate",
                gate,
                layer=layer_idx,
                metadata={"mean": float(gate.detach().float().mean().item())},
            )
        gated_content = gate * content
        dynamic_delta = gated_content - content
        if activation_recorder is not None:
            dynamic_delta = activation_recorder.record("dynamic_delta", dynamic_delta, layer=layer_idx)
            gated_content = activation_recorder.record("dynamic_value_output", content + dynamic_delta, layer=layer_idx)
        y = self.resid_dropout(self.c_proj(gated_content))
        self._record_diagnostics(
            gate=gate,
            content=content,
            gated_content=gated_content,
            step=step,
            layer_idx=layer_idx,
        )
        return y

    def _gate_input(self, x: torch.Tensor, q_flat: torch.Tensor) -> torch.Tensor:
        if self.gate_from == "x":
            return x
        if self.gate_from == "q":
            return q_flat
        if self.gate_from == "xq":
            return torch.cat([x, q_flat], dim=-1)
        raise ValueError(f"dynamic_value_gate_from must be one of {sorted(self.allowed_gate_sources)}")

    def _record_diagnostics(
        self,
        *,
        gate: torch.Tensor,
        content: torch.Tensor,
        gated_content: torch.Tensor,
        step: int | None,
        layer_idx: int | None,
    ) -> None:
        with torch.no_grad():
            gate_f = gate.detach().float()
            entropy_proxy = -(
                gate_f * gate_f.clamp_min(1e-12).log()
                + (1.0 - gate_f) * (1.0 - gate_f).clamp_min(1e-12).log()
            )
            content_norm = content.detach().float().norm()
            gated_norm = gated_content.detach().float().norm()
            delta_norm = (gated_content - content).detach().float().norm()
            row: dict[str, float | int | str | bool] = {
                "attention_type": "dynamic_value_query_conditioned_attention",
                "dynamic_value_gate_mean": float(gate_f.mean().item()),
                "dynamic_value_gate_std": float(gate_f.std(unbiased=False).item()),
                "dynamic_value_gate_min": float(gate_f.min().item()),
                "dynamic_value_gate_max": float(gate_f.max().item()),
                "dynamic_value_gate_entropy_proxy": float(entropy_proxy.mean().item()),
                "dynamic_value_static_content_norm": float(content_norm.item()),
                "dynamic_value_gated_content_norm": float(gated_norm.item()),
                "dynamic_value_delta_norm": float(delta_norm.item()),
                "dynamic_value_delta_to_static_ratio": float((delta_norm / content_norm.clamp_min(1e-12)).item()),
                "dynamic_value_pairwise_gate_enabled": False,
                "dynamic_value_gate_from": self.gate_from,
            }
            if layer_idx is not None:
                row["layer"] = int(layer_idx)
            if step is not None:
                row["step"] = int(step)
            self._last_diag = row

    def attention_diagnostics(self, *, step: int, layer: int) -> dict[str, float | int | str | bool] | None:
        if not self._last_diag:
            return None
        row = dict(self._last_diag)
        row["step"] = int(step)
        row["layer"] = int(layer)
        return row
