from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
from torch.nn import functional as F


class Q3K3V3RoleRoutedCausalSelfAttention(nn.Module):
    """Causal attention with content, operator, and binding Q/K/V role streams."""

    role_names = ("content", "operator", "binding")

    def __init__(self, config: Any):
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        role_dim_mode = str(getattr(config, "q3k3v3_role_dim_mode", "equal"))
        if role_dim_mode != "equal":
            raise ValueError("q3k3v3_role_dim_mode must be 'equal'")

        self.n_head = int(config.n_head)
        self.n_embd = int(config.n_embd)
        self.head_size = self.n_embd // self.n_head
        self.cross_role_grid = bool(getattr(config, "q3k3v3_cross_role_grid", False))
        self.include_pair_products = bool(getattr(config, "q3k3v3_include_pair_products", True))

        self.c_roles = nn.Linear(self.n_embd, 9 * self.n_embd, bias=config.bias)
        if self.cross_role_grid:
            projection_in = 9 * self.n_embd
        elif self.include_pair_products:
            projection_in = 6 * self.n_embd
        else:
            projection_in = 3 * self.n_embd
        self.c_proj = nn.Linear(projection_in, self.n_embd, bias=config.bias)
        self.c_proj.NANOGPT_SCALE_INIT = 1

        self.attn_dropout = float(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self._last_diag: dict[str, float | int | str | bool] = {}

    def _split_heads(self, tensor: torch.Tensor, batch_size: int, seq_len: int) -> torch.Tensor:
        return tensor.view(batch_size, seq_len, self.n_head, self.head_size).transpose(1, 2)

    def _merge_heads(self, tensor: torch.Tensor, batch_size: int, seq_len: int) -> torch.Tensor:
        return tensor.transpose(1, 2).contiguous().view(batch_size, seq_len, self.n_embd)

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
        batch_size, seq_len, _ = x.size()
        parts = self.c_roles(x).split(self.n_embd, dim=2)
        qc, kc, vc, qo, ko, vo, qb, kb, vb = [
            self._split_heads(part, batch_size, seq_len) for part in parts
        ]
        queries = {"content": qc, "operator": qo, "binding": qb}
        keys = {"content": kc, "operator": ko, "binding": kb}
        values = {"content": vc, "operator": vo, "binding": vb}

        diagonal_outputs = {
            role: self._attention(queries[role], keys[role], values[role])
            for role in self.role_names
        }
        content = self._merge_heads(diagonal_outputs["content"], batch_size, seq_len)
        operator = self._merge_heads(diagonal_outputs["operator"], batch_size, seq_len)
        binding = self._merge_heads(diagonal_outputs["binding"], batch_size, seq_len)
        if activation_recorder is not None:
            content = activation_recorder.record(
                "content_out",
                content,
                layer=layer_idx,
                metadata={"norm": float(content.detach().float().norm().item())},
            )
            operator = activation_recorder.record(
                "operator_out",
                operator,
                layer=layer_idx,
                metadata={"norm": float(operator.detach().float().norm().item())},
            )
            binding = activation_recorder.record(
                "binding_out",
                binding,
                layer=layer_idx,
                metadata={"norm": float(binding.detach().float().norm().item())},
            )

        if self.cross_role_grid:
            grid_outputs = []
            for query_role in self.role_names:
                for key_role in self.role_names:
                    target_value = values[key_role]
                    grid = self._attention(queries[query_role], keys[key_role], target_value)
                    grid_outputs.append(self._merge_heads(grid, batch_size, seq_len))
            projected_input = torch.cat(grid_outputs, dim=-1)
        else:
            pieces = [content, operator, binding]
            if self.include_pair_products:
                content_operator = content * operator
                content_binding = content * binding
                operator_binding = operator * binding
                if activation_recorder is not None:
                    content_operator = activation_recorder.record(
                        "content_operator_product",
                        content_operator,
                        layer=layer_idx,
                    )
                    content_binding = activation_recorder.record(
                        "content_binding_product",
                        content_binding,
                        layer=layer_idx,
                    )
                    operator_binding = activation_recorder.record(
                        "operator_binding_product",
                        operator_binding,
                        layer=layer_idx,
                    )
                pieces.extend([content_operator, content_binding, operator_binding])
            projected_input = torch.cat(pieces, dim=-1)

        y = self.resid_dropout(self.c_proj(projected_input))
        self._record_diagnostics(
            queries=queries,
            keys=keys,
            content=content,
            operator=operator,
            binding=binding,
            step=step,
            layer_idx=layer_idx,
        )
        return y

    def _attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=True,
        )

    def _attention_entropy(self, q: torch.Tensor, k: torch.Tensor) -> float:
        with torch.no_grad():
            seq_len = q.size(-2)
            scores = (q.detach().float() @ k.detach().float().transpose(-2, -1)) * (
                1.0 / math.sqrt(q.size(-1))
            )
            mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=q.device).tril()
            scores = scores.masked_fill(~mask, float("-inf"))
            attn = F.softmax(scores, dim=-1)
            entropy = -(attn * attn.clamp_min(1e-12).log()).sum(dim=-1)
            return float(entropy.mean().item())

    def _record_diagnostics(
        self,
        *,
        queries: dict[str, torch.Tensor],
        keys: dict[str, torch.Tensor],
        content: torch.Tensor,
        operator: torch.Tensor,
        binding: torch.Tensor,
        step: int | None,
        layer_idx: int | None,
    ) -> None:
        with torch.no_grad():
            content_norm = content.detach().float().norm()
            operator_norm = operator.detach().float().norm()
            binding_norm = binding.detach().float().norm()
            total_norm = (content_norm + operator_norm + binding_norm).clamp_min(1e-12)
            row: dict[str, float | int | str | bool] = {
                "attention_type": "q3k3v3_role_routed_attention",
                "q3_content_output_norm": float(content_norm.item()),
                "q3_operator_output_norm": float(operator_norm.item()),
                "q3_binding_output_norm": float(binding_norm.item()),
                "q3_content_operator_interaction_norm": float((content * operator).detach().float().norm().item()),
                "q3_content_binding_interaction_norm": float((content * binding).detach().float().norm().item()),
                "q3_operator_binding_interaction_norm": float((operator * binding).detach().float().norm().item()),
                "q3_content_to_total_norm_ratio": float((content_norm / total_norm).item()),
                "q3_operator_to_total_norm_ratio": float((operator_norm / total_norm).item()),
                "q3_binding_to_total_norm_ratio": float((binding_norm / total_norm).item()),
                "q3_attention_entropy_content": self._attention_entropy(queries["content"], keys["content"]),
                "q3_attention_entropy_operator": self._attention_entropy(queries["operator"], keys["operator"]),
                "q3_attention_entropy_binding": self._attention_entropy(queries["binding"], keys["binding"]),
                "q3_cross_role_grid_enabled": self.cross_role_grid,
                "q3_pair_products_enabled": self.include_pair_products,
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
