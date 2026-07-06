from __future__ import annotations

import torch

from attention_lab.mechanisms.attention_reconstruction import reconstruct_standard_attention_weights


def test_reconstruction_matches_hand_computed_three_position_example():
    # head_dim=1 so scores are just q*k -- easy to hand-verify.
    # q = [1, 0, 1], k = [1, 1, 0] (batch=1, head=1, seq_len=3, head_dim=1)
    q = torch.tensor([[[[1.0], [0.0], [1.0]]]])
    k = torch.tensor([[[[1.0], [1.0], [0.0]]]])

    weights = reconstruct_standard_attention_weights(q, k)

    assert tuple(weights.shape) == (1, 1, 3, 3)
    expected = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.4223188, 0.4223188, 0.1553624],
        ]
    )
    torch.testing.assert_close(weights[0, 0], expected, atol=1e-5, rtol=1e-4)


def test_reconstruction_rows_sum_to_one_and_is_causal():
    torch.manual_seed(0)
    batch, heads, seq_len, head_dim = 2, 3, 6, 4
    q = torch.randn(batch, heads, seq_len, head_dim)
    k = torch.randn(batch, heads, seq_len, head_dim)

    weights = reconstruct_standard_attention_weights(q, k)

    row_sums = weights.sum(dim=-1)
    torch.testing.assert_close(row_sums, torch.ones_like(row_sums), atol=1e-5, rtol=0.0)

    upper_triangle = weights.triu(diagonal=1)
    torch.testing.assert_close(upper_triangle, torch.zeros_like(upper_triangle), atol=0.0, rtol=0.0)


def test_reconstruction_matches_real_captured_attn_q_and_attn_k_for_a_real_forward_pass():
    from attention_lab.mechanisms.capture import capture_activations
    from attention_lab.models.gpt import GPT, GPTConfig

    torch.manual_seed(1)
    config = GPTConfig(
        block_size=8,
        vocab_size=64,
        n_layer=1,
        n_head=2,
        n_embd=16,
        dropout=0.0,
        bias=False,
        attention_type="standard",
    )
    model = GPT(config)
    model.eval()
    input_ids = torch.randint(0, 64, (1, 8))

    result = capture_activations(model, input_ids, sites=["attn_q[0]", "attn_k[0]"], detach=True)
    q = result.cache.records["attn_q[0]"].tensor
    k = result.cache.records["attn_k[0]"].tensor

    weights = reconstruct_standard_attention_weights(q, k)

    row_sums = weights.sum(dim=-1)
    torch.testing.assert_close(row_sums, torch.ones_like(row_sums), atol=1e-5, rtol=0.0)
    assert tuple(weights.shape) == (1, 2, 8, 8)
