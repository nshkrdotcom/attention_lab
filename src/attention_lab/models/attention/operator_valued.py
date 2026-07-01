from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
from torch.nn import functional as F


class OperatorValuedCausalSelfAttention(nn.Module):
    """Causal attention that routes retrieved content through fixed update operators."""

    _operator_names = ("add", "suppress", "gate", "transform", "bind")

    def __init__(self, config: Any):
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")

        self.n_head = int(config.n_head)
        self.n_embd = int(config.n_embd)
        self.head_size = self.n_embd // self.n_head
        self.include_transform = bool(getattr(config, "operator_include_transform", True))
        self.include_bind = bool(getattr(config, "operator_include_bind", True))
        hidden_mult = float(getattr(config, "operator_router_hidden_mult", 1.0))
        if not math.isfinite(hidden_mult) or hidden_mult <= 0.0:
            raise ValueError("operator_router_hidden_mult must be finite and positive")
        router_hidden = max(1, int(round(self.n_embd * hidden_mult)))

        self.active_operator_names = [
            name
            for name in self._operator_names
            if (name != "transform" or self.include_transform) and (name != "bind" or self.include_bind)
        ]
        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd, bias=config.bias)
        self.router = nn.Sequential(
            nn.Linear(2 * self.n_embd, router_hidden, bias=True),
            nn.GELU(),
            nn.Linear(router_hidden, len(self.active_operator_names), bias=True),
        )

        self.op_add = nn.Linear(self.n_embd, self.n_embd, bias=config.bias)
        self.op_suppress = nn.Linear(self.n_embd, self.n_embd, bias=config.bias)
        self.op_gate = nn.Linear(self.n_embd, self.n_embd, bias=True)
        self.op_gate_value = nn.Linear(self.n_embd, self.n_embd, bias=config.bias)
        self.op_transform_1 = nn.Linear(2 * self.n_embd, router_hidden, bias=True)
        self.op_transform_2 = nn.Linear(router_hidden, self.n_embd, bias=config.bias)
        self.op_bind = nn.Linear(self.n_embd, self.n_embd, bias=config.bias)

        suppress_init = float(getattr(config, "operator_suppress_scale_init", 0.5))
        if not math.isfinite(suppress_init) or suppress_init <= 0.0:
            raise ValueError("operator_suppress_scale_init must be finite and positive")
        raw_init = math.log(math.expm1(max(suppress_init, 1e-6)))
        self.suppress_scale_raw = nn.Parameter(torch.tensor(raw_init, dtype=torch.float32))

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

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = self._split_heads(q, batch_size, seq_len)
        k = self._split_heads(k, batch_size, seq_len)
        v = self._split_heads(v, batch_size, seq_len)

        content_heads = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=True,
        )
        content = content_heads.transpose(1, 2).contiguous().view(batch_size, seq_len, channels)

        router_input = torch.cat([x, content], dim=-1)
        active_probs = F.softmax(self.router(router_input), dim=-1)
        probs = self._full_operator_probs(active_probs)

        suppress_scale = F.softplus(self.suppress_scale_raw).to(dtype=content.dtype)
        outputs = {
            "add": self.op_add(content),
            "suppress": -suppress_scale * self.op_suppress(content),
            "gate": torch.sigmoid(self.op_gate(x)) * self.op_gate_value(content),
            "transform": (
                self.op_transform_2(F.gelu(self.op_transform_1(router_input)))
                if self.include_transform
                else torch.zeros_like(content)
            ),
            "bind": self.op_bind(x * content) if self.include_bind else torch.zeros_like(content),
        }

        combined = torch.zeros_like(content)
        for index, name in enumerate(self._operator_names):
            combined = combined + probs[..., index : index + 1] * outputs[name]
        y = self.resid_dropout(combined)

        self._record_diagnostics(
            probs=probs,
            outputs=outputs,
            combined=combined,
            suppress_scale=suppress_scale,
            step=step,
            layer_idx=layer_idx,
        )
        return y

    def _full_operator_probs(self, active_probs: torch.Tensor) -> torch.Tensor:
        full = active_probs.new_zeros(*active_probs.shape[:-1], len(self._operator_names))
        active_index = 0
        for index, name in enumerate(self._operator_names):
            if name in self.active_operator_names:
                full[..., index] = active_probs[..., active_index]
                active_index += 1
        return full

    def _record_diagnostics(
        self,
        *,
        probs: torch.Tensor,
        outputs: dict[str, torch.Tensor],
        combined: torch.Tensor,
        suppress_scale: torch.Tensor,
        step: int | None,
        layer_idx: int | None,
    ) -> None:
        with torch.no_grad():
            probs_f = probs.detach().float()
            entropy = -(probs_f * probs_f.clamp_min(1e-12).log()).sum(dim=-1)
            argmax = probs_f.argmax(dim=-1)
            total = max(1, argmax.numel())
            row: dict[str, float | int | str] = {
                "attention_type": "operator_valued_attention",
                "operator_prob_entropy_mean": float(entropy.mean().item()),
                "operator_combined_output_norm": float(combined.detach().float().norm().item()),
                "operator_suppress_scale": float(suppress_scale.detach().float().item()),
            }
            for index, name in enumerate(self._operator_names):
                row[f"operator_prob_{name}_mean"] = float(probs_f[..., index].mean().item())
                row[f"operator_argmax_{name}_frac"] = float((argmax == index).sum().item() / total)
                row[f"operator_{name}_output_norm"] = float(outputs[name].detach().float().norm().item())
            if layer_idx is not None:
                row["layer"] = int(layer_idx)
            if step is not None:
                row["step"] = int(step)
            self._last_diag = row

    def attention_diagnostics(self, *, step: int, layer: int) -> dict[str, float | int | str] | None:
        if not self._last_diag:
            return None
        row = dict(self._last_diag)
        row["step"] = int(step)
        row["layer"] = int(layer)
        return row
