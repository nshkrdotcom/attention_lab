from __future__ import annotations

import pytest
import torch

from attention_lab.mechanisms.capture import capture_activations
from attention_lab.mechanisms.interventions import InterventionKind, InterventionSpec, run_with_interventions
from attention_lab.models.gpt import GPT, GPTConfig


def tiny_model(attention_type: str = "standard") -> GPT:
    return GPT(
        GPTConfig(
            block_size=8,
            vocab_size=64,
            n_layer=1,
            n_head=2,
            n_embd=16,
            dropout=0.0,
            bias=False,
            attention_type=attention_type,
        )
    )


def test_zeroing_attn_out_changes_logits_on_tiny_standard_model():
    torch.manual_seed(0)
    model = tiny_model()
    model.eval()
    input_ids = torch.randint(0, 64, (2, 8))
    baseline_logits, _ = model(input_ids)

    result = run_with_interventions(
        model,
        input_ids,
        [InterventionSpec(site="attn_out", layer=0, kind=InterventionKind.ZERO)],
    )

    assert not torch.allclose(result.logits, baseline_logits)
    assert result.applied_interventions[0]["site"] == "attn_out[0]"


def test_no_intervention_equals_baseline_logits_exactly_in_eval_mode():
    torch.manual_seed(1)
    model = tiny_model()
    model.eval()
    input_ids = torch.randint(0, 64, (2, 8))
    baseline_logits, _ = model(input_ids)

    result = run_with_interventions(model, input_ids, [])

    assert torch.allclose(result.logits, baseline_logits, atol=0.0, rtol=0.0)


def test_replacing_operator_suppress_out_with_zeros_changes_combined_output_as_expected():
    torch.manual_seed(2)
    model = tiny_model("operator_valued_attention")
    model.eval()
    input_ids = torch.randint(0, 64, (1, 8))

    baseline = capture_activations(model, input_ids, detach=True)
    intervened = run_with_interventions(
        model,
        input_ids,
        [InterventionSpec(site="operator_suppress_out", layer=0, kind=InterventionKind.ZERO)],
        capture_sites=["operator_probs", "operator_combined_out"],
    )

    probs = baseline.cache.records["operator_probs[0]"].tensor
    suppress = baseline.cache.records["operator_suppress_out[0]"].tensor
    expected_delta = probs[..., 1:2] * suppress
    actual_delta = (
        baseline.cache.records["operator_combined_out[0]"].tensor
        - intervened.after_cache.records["operator_combined_out[0]"].tensor
    )

    assert torch.allclose(actual_delta, expected_delta, atol=1e-6, rtol=1e-5)


def test_incompatible_replacement_shapes_fail_clearly():
    torch.manual_seed(3)
    model = tiny_model()
    model.eval()
    input_ids = torch.randint(0, 64, (1, 8))

    with pytest.raises(ValueError, match="replacement shape"):
        run_with_interventions(
            model,
            input_ids,
            [
                InterventionSpec(
                    site="attn_out",
                    layer=0,
                    kind=InterventionKind.REPLACE,
                    value=torch.zeros(1, 1),
                )
            ],
        )


def test_interventions_are_deterministic_under_fixed_seed_eval_mode():
    torch.manual_seed(4)
    model = tiny_model()
    model.eval()
    input_ids = torch.randint(0, 64, (2, 8))
    spec = InterventionSpec(site="attn_out", layer=0, kind=InterventionKind.SCALE, scale=0.5)

    first = run_with_interventions(model, input_ids, [spec])
    second = run_with_interventions(model, input_ids, [spec])

    assert torch.allclose(first.logits, second.logits)
