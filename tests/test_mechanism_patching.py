from __future__ import annotations

from pathlib import Path

import pytest
import torch

from attention_lab.mechanisms.capture import capture_activations
from attention_lab.mechanisms.interventions import InterventionKind, InterventionSpec, run_with_interventions
from attention_lab.mechanisms.patching import make_cache_patch, mediation_fraction, restoration_score
from attention_lab.mechanisms.presets import SitePreset
from attention_lab.mechanisms.suite import _compute_patching_metrics
from attention_lab.mechanisms.task_schema import TaskRecord
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


def tiny_gpt2_vocab_model() -> GPT:
    return GPT(
        GPTConfig(
            block_size=16,
            vocab_size=50304,
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


def test_suite_patching_skips_discrete_route_site():
    result = _compute_patching_metrics(
        records=[],
        site=SitePreset("selected_track", 0, "route", None, continuous=False),
        model=tiny_model(),
        attention_type="multi_qkv_static_3track_global",
        tokenizer_name="gpt2",
        block_size=8,
        vocab_size=64,
        checkpoint_path=Path("dummy.pt"),
        device="cpu",
        batch_size=1,
        bootstrap_samples=5,
        seed=0,
    )

    assert result["patching"]["valid"] is False
    assert "discrete route/index" in result["patching"]["reason"]


def test_suite_patching_rejects_non_integer_token_metadata():
    record = TaskRecord(
        x_pos="Sentence: The analyst did not approve the report. Answer:",
        x_neg="Sentence: The analyst approved the report. Answer:",
        x_para="Sentence: The analyst never approved the report. Answer:",
        x_decoy="Sentence: The analyst carefully approved the report. Answer:",
        pair_id="pair_0",
        template_id="template_0",
        family_id="negation",
        metadata={"target_token_id": "2081", "foil_token_id": 3991},
    )

    result = _compute_patching_metrics(
        records=[record],
        site=SitePreset("attn_out", 0, "activation", "attn_out"),
        model=tiny_gpt2_vocab_model(),
        attention_type="standard",
        tokenizer_name="gpt2",
        block_size=16,
        vocab_size=50304,
        checkpoint_path=Path("dummy.pt"),
        device="cpu",
        batch_size=1,
        bootstrap_samples=5,
        seed=0,
    )

    assert result["patching"]["valid"] is False
    assert "integer GPT-2 token ids" in result["patching"]["reason"]


def test_suite_patching_valid_toy_restoration_emits_artifacts():
    torch.manual_seed(3)
    record = TaskRecord(
        x_pos="Yes:",
        x_neg="No:",
        x_para="Never:",
        x_decoy="Often:",
        pair_id="pair_0",
        template_id="template_0",
        family_id="negation",
        metadata={"target_token_id": 2081, "foil_token_id": 3991},
    )

    result = _compute_patching_metrics(
        records=[record],
        site=SitePreset("attn_out", 0, "activation", "attn_out"),
        model=tiny_gpt2_vocab_model(),
        attention_type="standard",
        tokenizer_name="gpt2",
        block_size=16,
        vocab_size=50304,
        checkpoint_path=Path("dummy.pt"),
        device="cpu",
        batch_size=1,
        bootstrap_samples=5,
        seed=0,
    )

    assert "component_patch_restoration" in result["patching"]
    assert "full_layer_patch_restoration" in result["patching"]
    assert "mediation_fraction" in result
