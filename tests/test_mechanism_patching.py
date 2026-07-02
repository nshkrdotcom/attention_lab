from __future__ import annotations

import pytest
import torch

from attention_lab.mechanisms.capture import capture_activations
from attention_lab.mechanisms.interventions import InterventionKind, InterventionSpec, run_with_interventions
from attention_lab.mechanisms.patching import make_cache_patch, mediation_fraction, restoration_score
from attention_lab.models.gpt import GPT, GPTConfig


def tiny_model() -> GPT:
    return GPT(
        GPTConfig(
            block_size=8,
            vocab_size=64,
            n_layer=1,
            n_head=2,
            n_embd=16,
            dropout=0.0,
            bias=False,
            attention_type="standard",
        )
    )


def test_patching_from_cache_validates_model_config_site_compatibility():
    torch.manual_seed(0)
    model = tiny_model()
    model.eval()
    source_ids = torch.randint(0, 64, (1, 8))
    target_ids = torch.randint(0, 64, (1, 8))
    source_cache = capture_activations(model, source_ids, sites=["attn_out"], detach=True).cache

    spec = make_cache_patch(source_cache, site="attn_out", layer=0)
    patched = run_with_interventions(model, target_ids, [spec], capture_sites=["attn_out"])

    assert patched.applied_interventions[0]["kind"] == "patch_from_cache"
    assert torch.allclose(
        patched.after_cache.records["attn_out[0]"].tensor,
        source_cache.records["attn_out[0]"].tensor,
    )


def test_patching_selected_token_positions_only():
    torch.manual_seed(1)
    model = tiny_model()
    model.eval()
    source_ids = torch.randint(0, 64, (1, 8))
    target_ids = torch.randint(0, 64, (1, 8))
    source_cache = capture_activations(model, source_ids, sites=["attn_out"], detach=True).cache
    baseline = capture_activations(model, target_ids, sites=["attn_out"], detach=True).cache

    spec = InterventionSpec(
        site="attn_out",
        layer=0,
        kind=InterventionKind.PATCH_FROM_CACHE,
        source_cache=source_cache,
        token_indices=[0, 3],
    )
    patched = run_with_interventions(model, target_ids, [spec], capture_sites=["attn_out"])
    tensor = patched.after_cache.records["attn_out[0]"].tensor

    assert torch.allclose(tensor[:, [0, 3], :], source_cache.records["attn_out[0]"].tensor[:, [0, 3], :])
    assert torch.allclose(tensor[:, [1, 2, 4, 5, 6, 7], :], baseline.records["attn_out[0]"].tensor[:, [1, 2, 4, 5, 6, 7], :])


def test_patching_rejects_attention_type_mismatch():
    torch.manual_seed(2)
    model = tiny_model()
    source = capture_activations(model, torch.randint(0, 64, (1, 8)), sites=["attn_out"], detach=True).cache
    source.attention_type = "other"

    with pytest.raises(ValueError, match="attention_type"):
        run_with_interventions(
            model,
            torch.randint(0, 64, (1, 8)),
            [make_cache_patch(source, site="attn_out", layer=0)],
        )


def test_restoration_score_and_mediation_fraction_formulas():
    restoration = restoration_score(clean_logitdiff=5.0, corrupted_logitdiff=1.0, patched_logitdiff=3.0)
    mediation = mediation_fraction(component_patch_restoration=0.25, full_layer_patch_restoration=0.5)

    assert restoration.valid
    assert restoration.restoration_score == 0.5
    assert mediation.valid
    assert mediation.mediation_fraction == 0.5


def test_restoration_and_mediation_denominator_edges_are_invalid():
    restoration = restoration_score(clean_logitdiff=1.0, corrupted_logitdiff=1.0, patched_logitdiff=1.5)
    mediation = mediation_fraction(component_patch_restoration=0.2, full_layer_patch_restoration=0.0)

    assert not restoration.valid
    assert "denominator" in (restoration.reason or "")
    assert not mediation.valid
    assert "denominator" in (mediation.reason or "")
