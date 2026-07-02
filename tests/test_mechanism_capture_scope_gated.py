from __future__ import annotations

import torch

from attention_lab.mechanisms.capture import capture_activations
from attention_lab.models.gpt import GPT, GPTConfig


def test_scope_gated_capture_exposes_gate_and_product_relationships():
    torch.manual_seed(0)
    model = GPT(
        GPTConfig(
            block_size=8,
            vocab_size=64,
            n_layer=1,
            n_head=2,
            n_embd=16,
            dropout=0.0,
            bias=False,
            attention_type="scope_gated_qkv",
        )
    )
    model.eval()
    result = capture_activations(model, torch.randint(0, 64, (2, 8)), detach=True)
    records = result.cache.records

    gate = records["gate[0]"].tensor
    content = records["content_out[0]"].tensor
    scope = records["scope_out[0]"].tensor

    assert torch.all((0.0 <= gate) & (gate <= 1.0))
    assert torch.allclose(records["content_scope_product[0]"].tensor, content * scope)
    assert torch.allclose(records["gated_content[0]"].tensor, gate * content)
