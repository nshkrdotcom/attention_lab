from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch.nn import functional as F

from attention_lab.models.attention.multi_qkv_common import (
    MultiQKVGlobalBank,
    is_multi_qkv_attention,
    validate_canonical_multi_qkv_model_config,
)
from attention_lab.models.attention.registry import build_attention


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False
    attention_type: str = "standard"
    cp_rank: int | None = None
    cp_lambda_init: float = 0.0
    cp_lambda_trainable: bool = True
    cp_lambda_fixed: bool = False
    qkv_track_count: int = 1
    qkv_global_bank: bool = False
    qkv_route_formula: str | None = None
    multi_qkv_track_count: int = 3
    multi_qkv_global: bool = True


def config_from_dict(model_config: dict[str, Any], data_config: dict[str, Any] | None = None) -> GPTConfig:
    merged = dict(model_config)
    if "qkv_track_count" not in merged and "multi_qkv_track_count" in merged:
        merged["qkv_track_count"] = merged["multi_qkv_track_count"]
    if "qkv_global_bank" not in merged and "multi_qkv_global" in merged:
        merged["qkv_global_bank"] = merged["multi_qkv_global"]
    if "multi_qkv_track_count" not in merged and "qkv_track_count" in merged:
        merged["multi_qkv_track_count"] = merged["qkv_track_count"]
    if "multi_qkv_global" not in merged and "qkv_global_bank" in merged:
        merged["multi_qkv_global"] = merged["qkv_global_bank"]
    if data_config is not None and "vocab_size" not in merged:
        merged["vocab_size"] = data_config.get("vocab_size", GPTConfig.vocab_size)
    valid_keys = set(GPTConfig.__dataclass_fields__)
    unknown = sorted(set(merged) - valid_keys)
    if unknown:
        raise ValueError(f"Unknown model config keys: {unknown}")
    return GPTConfig(**merged)


class LayerNorm(nn.Module):
    def __init__(self, ndim: int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return self.dropout(x)


class Block(nn.Module):
    def __init__(
        self,
        config: GPTConfig,
        *,
        layer_idx: int,
        qkv_bank: MultiQKVGlobalBank | None = None,
        shared_qkv_bank: MultiQKVGlobalBank | None = None,
    ):
        super().__init__()
        if qkv_bank is not None and shared_qkv_bank is not None and qkv_bank is not shared_qkv_bank:
            raise ValueError("qkv_bank and shared_qkv_bank must reference the same global bank")
        bank = qkv_bank if qkv_bank is not None else shared_qkv_bank
        self.layer_idx = layer_idx
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = build_attention(config, layer_idx=layer_idx, qkv_bank=bank)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        *,
        step: int | None = None,
        positions: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        schedule_mode: str | None = None,
    ) -> torch.Tensor:
        if position_ids is not None:
            positions = position_ids
        x = x + self.attn(
            self.ln_1(x),
            step=step,
            position_ids=positions,
            schedule_mode=schedule_mode,
            layer_idx=self.layer_idx,
        )
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        validate_canonical_multi_qkv_model_config(config)
        self.config = config
        self.multi_qkv_bank = MultiQKVGlobalBank(config) if is_multi_qkv_attention(config.attention_type) else None

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                h=nn.ModuleList(
                    [Block(config, layer_idx=layer_idx, qkv_bank=self.multi_qkv_bank) for layer_idx in range(config.n_layer)]
                ),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        *,
        step: int | None = None,
        positions: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        schedule_mode: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, seq_len = idx.size()
        if seq_len > self.config.block_size:
            raise ValueError(
                f"Cannot forward sequence of length {seq_len}; "
                f"block_size is {self.config.block_size}"
            )

        if position_ids is not None:
            positions = position_ids
        if positions is None:
            pos = torch.arange(0, seq_len, dtype=torch.long, device=idx.device)
        else:
            pos = positions.to(device=idx.device, dtype=torch.long)
        if schedule_mode is None:
            schedule_mode = "train" if self.training else "eval"
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb

        for block in self.transformer.h:
            x = block(x, step=step, position_ids=pos, schedule_mode=schedule_mode)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def num_parameters(self, non_embedding: bool = True) -> int:
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params
