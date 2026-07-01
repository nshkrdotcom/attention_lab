from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
from torch.nn import functional as F


class CPScoreAugmentedCausalSelfAttention(nn.Module):
    """GPT attention with an additive low-rank CP score branch."""

    attention_type = "cp_base"
    low_rank_projection_count = 2

    def __init__(self, config: Any):
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if config.cp_rank is None or int(config.cp_rank) <= 0:
            raise ValueError("cp_rank must be a positive integer")
        if bool(config.cp_lambda_fixed) and bool(config.cp_lambda_trainable):
            raise ValueError("cp_lambda_fixed and cp_lambda_trainable cannot both be true")

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.cp_rank = int(config.cp_rank)

        self.q_low = nn.Linear(config.n_embd, config.n_head * self.cp_rank, bias=config.bias)
        self.k_low = nn.Linear(config.n_embd, config.n_head * self.cp_rank, bias=config.bias)
        if self.low_rank_projection_count == 3:
            self.v_low = nn.Linear(config.n_embd, config.n_head * self.cp_rank, bias=config.bias)

        lambda_init = float(config.cp_lambda_init)
        if bool(config.cp_lambda_fixed):
            self.register_buffer("cp_lambda", torch.tensor(lambda_init, dtype=torch.float32))
        elif bool(config.cp_lambda_trainable):
            self.cp_lambda = nn.Parameter(torch.tensor(lambda_init, dtype=torch.float32))
        else:
            self.register_buffer("cp_lambda", torch.tensor(lambda_init, dtype=torch.float32))

        self.last_diagnostics: dict[str, float] | None = None

    def _split_qkv(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, channels = x.size()
        head_size = channels // self.n_head
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        q = q.view(batch_size, seq_len, self.n_head, head_size).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_head, head_size).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, head_size).transpose(1, 2)
        return q, k, v

    def _low_rank_factor(self, projection: nn.Linear, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        return projection(x).view(batch_size, seq_len, self.n_head, self.cp_rank).transpose(1, 2)

    def low_rank_factors(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        q_low = self._low_rank_factor(self.q_low, x)
        k_low = self._low_rank_factor(self.k_low, x)
        if self.low_rank_projection_count == 3:
            return q_low, k_low, self._low_rank_factor(self.v_low, x)
        return q_low, k_low

    def extra_scores_from_factors(self, *factors: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def compute_extra_scores(self, x: torch.Tensor) -> torch.Tensor:
        return self.extra_scores_from_factors(*self.low_rank_factors(x))

    def cp_parameter_norm(self) -> float:
        total = torch.zeros((), device=self.c_attn.weight.device)
        for parameter in self.cp_parameters():
            total = total + parameter.detach().float().pow(2).sum()
        return float(total.sqrt().item())

    def cp_gradient_norm(self) -> float | None:
        total = None
        for parameter in self.cp_parameters():
            if parameter.grad is None:
                continue
            grad_total = parameter.grad.detach().float().pow(2).sum()
            total = grad_total if total is None else total + grad_total
        if total is None:
            return None
        return float(total.sqrt().item())

    def cp_parameters(self) -> list[nn.Parameter]:
        modules = [self.q_low, self.k_low]
        if hasattr(self, "v_low"):
            modules.append(self.v_low)
        parameters = [parameter for module in modules for parameter in module.parameters()]
        if isinstance(self.cp_lambda, nn.Parameter):
            parameters.append(self.cp_lambda)
        return parameters

    def _record_diagnostics(
        self,
        standard_scores: torch.Tensor,
        extra_scores: torch.Tensor,
        attention: torch.Tensor,
    ) -> None:
        with torch.no_grad():
            standard_std = standard_scores.detach().float().std(unbiased=False)
            cp_std = extra_scores.detach().float().std(unbiased=False)
            entropy = -(attention.detach().float() * attention.detach().float().clamp_min(1e-12).log()).sum(dim=-1)
            self.last_diagnostics = {
                "attention_type": self.attention_type,
                "lambda_value": float(self.cp_lambda.detach().float().item()),
                "cp_score_mean": float(extra_scores.detach().float().mean().item()),
                "cp_score_std": float(cp_std.item()),
                "standard_score_mean": float(standard_scores.detach().float().mean().item()),
                "standard_score_std": float(standard_std.item()),
                "cp_to_standard_score_std_ratio": float((cp_std / standard_std.clamp_min(1e-12)).item()),
                "attention_entropy_mean": float(entropy.mean().item()),
                "attention_entropy_std": float(entropy.std(unbiased=False).item()),
            }

    def attention_diagnostics(self, step: int, layer: int) -> dict[str, float | int | str] | None:
        if self.last_diagnostics is None:
            return None
        return {
            "step": step,
            "layer": layer,
            **self.last_diagnostics,
            "cp_parameter_norm": self.cp_parameter_norm(),
            "cp_gradient_norm": self.cp_gradient_norm(),
        }

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
        del step, positions, position_ids, schedule_mode, layer_idx
        batch_size, seq_len, channels = x.size()
        head_size = channels // self.n_head
        q, k, v = self._split_qkv(x)

        standard_scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(head_size))
        extra_scores = self.compute_extra_scores(x)
        scores = standard_scores + self.cp_lambda.to(dtype=standard_scores.dtype) * extra_scores
        causal_mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device).tril()
        scores = scores.masked_fill(~causal_mask, float("-inf"))
        attention = F.softmax(scores, dim=-1)
        self._record_diagnostics(standard_scores, extra_scores, attention)
        attention = self.attn_dropout(attention)
        y = attention @ v
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, channels)
        return self.resid_dropout(self.c_proj(y))
