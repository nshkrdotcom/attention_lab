from __future__ import annotations

import torch

from attention_lab.mechanisms.capture import capture_activations
from attention_lab.models.gpt import GPT, GPTConfig


def test_dynamic_value_capture_exposes_gate_delta_relationship():
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
            attention_type="dynamic_value_query_conditioned_attention",
        )
    )
    model.eval()
    result = capture_activations(model, torch.randint(0, 64, (2, 8)), detach=True)
    records = result.cache.records

    content = records["static_value_content[0]"].tensor
    gate = records["dynamic_gate[0]"].tensor
    delta = records["dynamic_delta[0]"].tensor
    output = records["dynamic_value_output[0]"].tensor

    assert torch.all((0.0 <= gate) & (gate <= 1.0))
    assert torch.allclose(output, gate * content)
    assert torch.allclose(delta, output - content)
    assert abs(records["dynamic_gate[0]"].metadata["mean"] - float(gate.mean())) < 1e-6
