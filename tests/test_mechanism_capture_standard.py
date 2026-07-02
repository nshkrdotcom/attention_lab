from __future__ import annotations

import torch

from attention_lab.mechanisms.capture import capture_activations
from attention_lab.models.gpt import GPT, GPTConfig


def tiny_config(attention_type: str = "standard", **kwargs) -> GPTConfig:
    values = {
        "block_size": 8,
        "vocab_size": 64,
        "n_layer": 2,
        "n_head": 2,
        "n_embd": 16,
        "dropout": 0.0,
        "bias": False,
        "attention_type": attention_type,
    }
    values.update(kwargs)
    return GPTConfig(**values)


def test_standard_model_captures_standard_sites_on_real_forward_pass():
    torch.manual_seed(0)
    model = GPT(tiny_config())
    model.eval()
    input_ids = torch.randint(0, 64, (2, 8))

    result = capture_activations(model, input_ids, detach=True)

    assert result.logits.shape == (2, 8, 64)
    for key in (
        "resid_pre[0]",
        "attn_q[0]",
        "attn_k[0]",
        "attn_v[0]",
        "attn_out[0]",
        "resid_mid[0]",
        "mlp_out[0]",
        "resid_post[0]",
        "logits",
    ):
        assert key in result.cache.records
    assert result.cache.records["attn_q[0]"].tensor.shape == (2, 2, 8, 8)
    assert result.cache.records["resid_post[1]"].tensor.shape == (2, 8, 16)
    assert not result.missing_sites


def test_strict_capture_all_reports_no_declared_gaps_for_standard_model():
    torch.manual_seed(4)
    model = GPT(tiny_config())
    model.eval()
    input_ids = torch.randint(0, 64, (2, 8))

    result = capture_activations(model, input_ids, detach=True, require_declared_sites=True)

    assert not result.declared_but_unemitted_sites


def test_capture_disabled_has_no_output_side_effects():
    torch.manual_seed(1)
    model = GPT(tiny_config())
    model.eval()
    input_ids = torch.randint(0, 64, (2, 8))

    baseline_logits, _ = model(input_ids)
    captured = capture_activations(model, input_ids, detach=True)

    assert torch.allclose(captured.logits, baseline_logits)


def test_capture_does_not_detach_silently():
    torch.manual_seed(2)
    model = GPT(tiny_config())
    input_ids = torch.randint(0, 64, (2, 8))

    result = capture_activations(model, input_ids, sites=["logits"], detach=False)

    assert result.cache.records["logits"].tensor.requires_grad


def test_missing_unsupported_sites_are_reported_not_silently_dropped():
    torch.manual_seed(3)
    model = GPT(tiny_config())
    input_ids = torch.randint(0, 64, (1, 8))

    result = capture_activations(model, input_ids, sites=["operator_probs", "resid_pre"], detach=True)

    assert "operator_probs" in result.missing_sites
    assert result.missing_sites["operator_probs"].status == "missing"
    assert "resid_pre[0]" in result.cache.records
