from __future__ import annotations

import torch

from attention_lab.mechanisms.capture import capture_activations
from attention_lab.models.gpt import GPT, GPTConfig


def test_differential_qkv_capture_exposes_branch_relationship():
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
            attention_type="differential_qkv_anti_value",
        )
    )
    model.eval()
    result = capture_activations(model, torch.randint(0, 64, (2, 8)), detach=True)
    records = result.cache.records

    lam = records["lambda[0]"].tensor
    expected = records["pos_out[0]"].tensor - lam * records["neg_out[0]"].tensor

    assert torch.allclose(records["branch_delta[0]"].tensor, expected, atol=1e-6, rtol=1e-5)
    assert lam.item() > 0
    for key in ("pos_q[0]", "pos_k[0]", "pos_v[0]", "neg_q[0]", "neg_k[0]", "neg_v[0]"):
        assert records[key].tensor.abs().sum() > 0
