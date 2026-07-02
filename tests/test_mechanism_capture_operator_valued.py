from __future__ import annotations

import torch

from attention_lab.mechanisms.capture import capture_activations
from attention_lab.models.gpt import GPT, GPTConfig


def tiny_operator_model(**kwargs) -> GPT:
    values = {
        "block_size": 8,
        "vocab_size": 64,
        "n_layer": 1,
        "n_head": 2,
        "n_embd": 16,
        "dropout": 0.0,
        "bias": False,
        "attention_type": "operator_valued_attention",
    }
    values.update(kwargs)
    return GPT(GPTConfig(**values))


def test_operator_valued_noop_capture_preserves_logits():
    torch.manual_seed(10)
    model = tiny_operator_model()
    model.eval()
    input_ids = torch.randint(0, 64, (2, 8))
    baseline_logits, _ = model(input_ids)

    result = capture_activations(model, input_ids, detach=True)

    assert torch.allclose(result.logits, baseline_logits, atol=0.0, rtol=0.0)


def test_operator_valued_capture_exposes_weighted_operator_components():
    torch.manual_seed(0)
    model = tiny_operator_model()
    model.eval()
    input_ids = torch.randint(0, 64, (2, 8))

    result = capture_activations(model, input_ids, detach=True)
    records = result.cache.records

    probs = records["operator_probs[0]"].tensor
    outputs = torch.stack(
        [
            records["operator_add_out[0]"].tensor,
            records["operator_suppress_out[0]"].tensor,
            records["operator_gate_out[0]"].tensor,
            records["operator_transform_out[0]"].tensor,
            records["operator_bind_out[0]"].tensor,
        ],
        dim=-2,
    )
    combined = records["operator_combined_out[0]"].tensor

    assert torch.allclose(probs.sum(dim=-1), torch.ones_like(probs[..., 0]), atol=1e-6)
    assert torch.allclose((probs.unsqueeze(-1) * outputs).sum(dim=-2), combined, atol=1e-6, rtol=1e-5)
    assert records["operator_suppress_out[0]"].metadata["signed"] == "negative"


def test_operator_disabled_components_are_explicitly_reported():
    torch.manual_seed(1)
    model = tiny_operator_model(operator_include_transform=False, operator_include_bind=False)
    result = capture_activations(model, torch.randint(0, 64, (1, 8)), detach=True)

    assert result.cache.records["operator_transform_out[0]"].metadata["disabled"] is True
    assert result.cache.records["operator_bind_out[0]"].metadata["disabled"] is True
    assert torch.count_nonzero(result.cache.records["operator_transform_out[0]"].tensor) == 0
    assert torch.count_nonzero(result.cache.records["operator_bind_out[0]"].tensor) == 0
