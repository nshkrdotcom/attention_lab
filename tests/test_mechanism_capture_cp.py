from __future__ import annotations

import torch

from attention_lab.mechanisms.capture import capture_activations
from attention_lab.models.gpt import GPT, GPTConfig


def test_cp_capture_exposes_lambda_and_score_contribution_without_changing_logits():
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
            attention_type="cp_trilinear",
            cp_rank=4,
            cp_lambda_init=0.0,
            cp_lambda_trainable=False,
            cp_lambda_fixed=True,
        )
    )
    model.eval()
    input_ids = torch.randint(0, 64, (2, 8))
    baseline_logits, _ = model(input_ids)

    result = capture_activations(model, input_ids, detach=True)
    records = result.cache.records

    assert records["cp_lambda[0]"].tensor.item() == 0.0
    assert records["cp_score[0]"].tensor.abs().sum() > 0
    assert torch.count_nonzero(records["cp_output[0]"].tensor) == 0
    assert torch.allclose(result.logits, baseline_logits)
    assert "cp_rank_component" in result.missing_sites or "cp_rank_component[layer, rank]" not in records
