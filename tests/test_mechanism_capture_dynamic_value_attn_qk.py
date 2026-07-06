from __future__ import annotations

import torch

from attention_lab.mechanisms.attention_reconstruction import reconstruct_standard_attention_weights
from attention_lab.mechanisms.capture import capture_activations
from attention_lab.models.gpt import GPT, GPTConfig


def tiny_dynamic_value_config() -> GPTConfig:
    return GPTConfig(
        block_size=8,
        vocab_size=64,
        n_layer=2,
        n_head=2,
        n_embd=16,
        dropout=0.0,
        bias=False,
        attention_type="dynamic_value_query_conditioned_attention",
    )


def test_dynamic_value_now_records_attn_q_and_attn_k():
    # Regression: dynamic_value_query_conditioned_attention computes standard
    # causal content attention via F.scaled_dot_product_attention but never
    # recorded q/k at all -- discovered while spelunking a real checkpoint
    # (attn_q/attn_k were declared sites but silently never emitted, so its
    # content-attention pattern was completely invisible). Fixed by recording
    # them the same way Phase 1 did for CP/multi-QKV.
    torch.manual_seed(0)
    model = GPT(tiny_dynamic_value_config())
    model.eval()
    input_ids = torch.randint(0, 64, (2, 8))

    result = capture_activations(model, input_ids, sites=["attn_q", "attn_k"], detach=True)

    assert "attn_q[0]" in result.cache.records
    assert "attn_k[0]" in result.cache.records
    assert "attn_q[1]" in result.cache.records
    assert "attn_k[1]" in result.cache.records


def test_dynamic_value_content_attention_is_reconstructable_as_a_real_causal_distribution():
    torch.manual_seed(0)
    model = GPT(tiny_dynamic_value_config())
    model.eval()
    input_ids = torch.randint(0, 64, (2, 8))

    result = capture_activations(model, input_ids, sites=["attn_q", "attn_k"], detach=True)
    q = result.cache.records["attn_q[0]"].tensor
    k = result.cache.records["attn_k[0]"].tensor

    weights = reconstruct_standard_attention_weights(q, k)

    row_sums = weights.sum(dim=-1)
    torch.testing.assert_close(row_sums, torch.ones_like(row_sums), atol=1e-5, rtol=0.0)
    upper_triangle = weights.triu(diagonal=1)
    torch.testing.assert_close(upper_triangle, torch.zeros_like(upper_triangle), atol=0.0, rtol=0.0)


def test_recording_attn_q_and_attn_k_does_not_change_model_output():
    torch.manual_seed(0)
    model = GPT(tiny_dynamic_value_config())
    model.eval()
    input_ids = torch.randint(0, 64, (1, 8))

    baseline_logits, _ = model(input_ids)
    result = capture_activations(model, input_ids, sites=["attn_q", "attn_k"], detach=True)

    assert torch.allclose(result.logits, baseline_logits, atol=0.0, rtol=0.0)
