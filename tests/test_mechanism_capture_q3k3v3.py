from __future__ import annotations

import torch

from attention_lab.mechanisms.capture import capture_activations
from attention_lab.models.gpt import GPT, GPTConfig


def test_q3k3v3_capture_exposes_role_products():
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
            attention_type="q3k3v3_role_routed_attention",
        )
    )
    model.eval()
    result = capture_activations(model, torch.randint(0, 64, (2, 8)), detach=True)
    records = result.cache.records

    content = records["content_out[0]"].tensor
    operator = records["operator_out[0]"].tensor
    binding = records["binding_out[0]"].tensor

    assert torch.allclose(records["content_operator_product[0]"].tensor, content * operator)
    assert torch.allclose(records["content_binding_product[0]"].tensor, content * binding)
    assert torch.allclose(records["operator_binding_product[0]"].tensor, operator * binding)
    assert records["content_out[0]"].metadata["norm"] > 0
