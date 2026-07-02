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


def _restoration_metadata(x_pos: str, x_neg: str, *, alignment: str | None = None) -> dict[str, object]:
    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    clean_len = len(enc.encode(x_pos))
    corrupted_len = len(enc.encode(x_neg))
    clean_answer = clean_len - 1
    corrupted_answer = corrupted_len - 1
    shared = min(clean_answer, corrupted_answer)
    return {
        "target_token_text": " true",
        "foil_token_text": " false",
        "target_token_id": 2081,
        "foil_token_id": 3991,
        "clean_answer_position": clean_answer,
        "corrupted_answer_position": corrupted_answer,
        "patch_token_indices": [shared],
        "clean_patch_token_indices": [clean_answer],
        "corrupted_patch_token_indices": [corrupted_answer],
        "clean_corrupt_token_alignment": alignment
        or ("same_length" if clean_len == corrupted_len else "explicit_patch_indices"),
    }


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


def test_patching_maps_source_token_positions_to_target_token_positions():
    torch.manual_seed(1)
    model = tiny_model()
    model.eval()
    source_ids = torch.randint(0, 64, (1, 8))
    target_ids = torch.randint(0, 64, (1, 6))
    source_cache = capture_activations(model, source_ids, sites=["attn_out"], detach=True).cache
    baseline = capture_activations(model, target_ids, sites=["attn_out"], detach=True).cache

    spec = InterventionSpec(
        site="attn_out",
        layer=0,
        kind=InterventionKind.PATCH_FROM_CACHE,
        source_cache=source_cache,
        token_indices=[2],
        source_token_indices=[5],
    )
    patched = run_with_interventions(model, target_ids, [spec], capture_sites=["attn_out"])
    tensor = patched.after_cache.records["attn_out[0]"].tensor

    assert torch.allclose(tensor[:, 2, :], source_cache.records["attn_out[0]"].tensor[:, 5, :])
    assert torch.allclose(tensor[:, [0, 1, 3, 4, 5], :], baseline.records["attn_out[0]"].tensor[:, [0, 1, 3, 4, 5], :])


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
        metadata={**_restoration_metadata("Sentence: The analyst did not approve the report. Answer:", "Sentence: The analyst approved the report. Answer:"), "target_token_id": "2081"},
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
    assert "non-integer target_token_id" in result["patching"]["reason"]


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
        metadata=_restoration_metadata("Yes:", "No:"),
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


def test_suite_patching_rejects_different_lengths_without_explicit_patch_indices():
    x_pos = "Sentence: The analyst did not approve the report. Answer:"
    x_neg = "Sentence: The analyst approved the report. Answer:"
    metadata = _restoration_metadata(x_pos, x_neg, alignment="same_length")
    record = TaskRecord(
        x_pos=x_pos,
        x_neg=x_neg,
        x_para="Sentence: The analyst never approved the report. Answer:",
        x_decoy="Sentence: The analyst carefully approved the report. Answer:",
        pair_id="pair_0",
        template_id="template_0",
        family_id="negation",
        metadata=metadata,
    )

    result = _compute_patching_metrics(
        records=[record],
        site=SitePreset("attn_out", 0, "activation", "attn_out"),
        model=tiny_gpt2_vocab_model(),
        attention_type="standard",
        tokenizer_name="gpt2",
        block_size=32,
        vocab_size=50304,
        checkpoint_path=Path("dummy.pt"),
        device="cpu",
        batch_size=1,
        bootstrap_samples=5,
        seed=0,
    )

    assert result["patching"]["valid"] is False
    assert result["patching"]["restoration_alignment_valid"] is False
    assert "same_length alignment" in result["patching"]["reason"]


def test_suite_patching_rejects_out_of_range_patch_indices():
    x_pos = "Yes:"
    x_neg = "No:"
    metadata = _restoration_metadata(x_pos, x_neg)
    metadata["clean_patch_token_indices"] = [999]
    record = TaskRecord(
        x_pos=x_pos,
        x_neg=x_neg,
        x_para="Never:",
        x_decoy="Often:",
        pair_id="pair_0",
        template_id="template_0",
        family_id="negation",
        metadata=metadata,
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
    assert "out-of-range clean_patch_token_indices" in result["patching"]["reason"]


def test_suite_patching_rejects_multitoken_target_text():
    x_pos = "Yes:"
    x_neg = "No:"
    metadata = _restoration_metadata(x_pos, x_neg)
    metadata["target_token_text"] = " definitely true"
    record = TaskRecord(
        x_pos=x_pos,
        x_neg=x_neg,
        x_para="Never:",
        x_decoy="Often:",
        pair_id="pair_0",
        template_id="template_0",
        family_id="negation",
        metadata=metadata,
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
    assert "not a single GPT-2 token" in result["patching"]["reason"]
