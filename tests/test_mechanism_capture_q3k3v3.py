from __future__ import annotations

import torch

from attention_lab.mechanisms.capture import capture_activations
from attention_lab.models.gpt import GPT, GPTConfig


def tiny_q3_model(**kwargs) -> GPT:
    values = {
        "block_size": 8,
        "vocab_size": 64,
        "n_layer": 1,
        "n_head": 2,
        "n_embd": 16,
        "dropout": 0.0,
        "bias": False,
        "attention_type": "q3k3v3_role_routed_attention",
    }
    values.update(kwargs)
    return GPT(GPTConfig(**values))


def test_q3k3v3_noop_capture_preserves_logits():
    torch.manual_seed(10)
    model = tiny_q3_model()
    model.eval()
    input_ids = torch.randint(0, 64, (2, 8))
    baseline_logits, _ = model(input_ids)

    result = capture_activations(model, input_ids, detach=True)

    assert torch.allclose(result.logits, baseline_logits, atol=0.0, rtol=0.0)


def test_q3k3v3_capture_exposes_role_products():
    torch.manual_seed(0)
    model = tiny_q3_model()
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


def test_q3k3v3_strict_capture_reports_disabled_pair_products_as_declared_but_unemitted():
    torch.manual_seed(1)
    model = tiny_q3_model(q3k3v3_include_pair_products=False)
    model.eval()
    result = capture_activations(
        model,
        torch.randint(0, 64, (1, 8)),
        detach=True,
        require_declared_sites=True,
    )

    assert result.declared_but_unemitted_sites["content_operator_product[layer]"].status == "missing"
    assert result.declared_but_unemitted_sites["content_binding_product[layer]"].status == "missing"
    assert result.declared_but_unemitted_sites["operator_binding_product[layer]"].status == "missing"
