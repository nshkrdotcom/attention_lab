import torch
import torch.nn as nn
from torch.nn import functional as F


class StandardCausalSelfAttention(nn.Module):
    """GPT-style multi-head causal self-attention."""

    def __init__(self, config):
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.attn_dropout = float(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd

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

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        q = q.view(batch_size, seq_len, self.n_head, head_size).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_head, head_size).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, head_size).transpose(1, 2)

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, channels)
        return self.resid_dropout(self.c_proj(y))
