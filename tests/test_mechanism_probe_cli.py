from __future__ import annotations

import pytest
import torch

from attention_lab.mechanisms.cache import ActivationCache, ActivationRecord
from attention_lab.mechanisms.interventions import InterventionKind
from attention_lab.mechanisms.probe import (
    build_probe_intervention_specs,
    encode_prompts,
    load_replacement_tensor,
    parse_index_list,
)


def test_probe_builds_replace_intervention_from_real_tensor(tmp_path):
    tensor = torch.randn(1, 4, 8)
    tensor_path = tmp_path / "replacement.pt"
    torch.save(tensor, tensor_path)

    specs = build_probe_intervention_specs(
        sites=["attn_out"],
        intervention_names=["replace"],
        layer=0,
        scale=None,
        source_cache=None,
        source_site=None,
        replacement_tensor=load_replacement_tensor(tensor_path),
        batch_indices=None,
        token_indices=None,
    )

    assert len(specs) == 1
    assert specs[0].kind == InterventionKind.REPLACE
    assert specs[0].site == "attn_out"
    assert specs[0].layer == 0
    assert torch.equal(specs[0].value, tensor)


def test_probe_builds_patch_from_cache_intervention_with_indices():
    source_cache = ActivationCache(
        records={
            "attn_out[0]": ActivationRecord(
                site="attn_out[0]",
                layer=0,
                tensor=torch.randn(2, 4, 8),
                metadata={},
            )
        },
        model_name="GPT",
        attention_type="standard",
        checkpoint_path=None,
        config_hash="abc",
        batch_metadata={},
    )

    specs = build_probe_intervention_specs(
        sites=["attn_out"],
        intervention_names=["patch_from_cache"],
        layer=0,
        scale=None,
        source_cache=source_cache,
        source_site="attn_out",
        replacement_tensor=None,
        batch_indices=[0],
        token_indices=[1, 3],
    )

    assert len(specs) == 1
    assert specs[0].kind == InterventionKind.PATCH_FROM_CACHE
    assert specs[0].source_cache is source_cache
    assert specs[0].source_site == "attn_out"
    assert specs[0].batch_indices == [0]
    assert specs[0].token_indices == [1, 3]


def test_probe_intervention_validation_fails_before_forward_for_missing_inputs():
    with pytest.raises(ValueError, match="replace requires"):
        build_probe_intervention_specs(
            sites=["attn_out"],
            intervention_names=["replace"],
            layer=0,
            scale=None,
            source_cache=None,
            source_site=None,
            replacement_tensor=None,
            batch_indices=None,
            token_indices=None,
        )

    with pytest.raises(ValueError, match="patch_from_cache requires --source-cache"):
        build_probe_intervention_specs(
            sites=["attn_out"],
            intervention_names=["patch_from_cache"],
            layer=0,
            scale=None,
            source_cache=None,
            source_site=None,
            replacement_tensor=None,
            batch_indices=None,
            token_indices=None,
        )

    with pytest.raises(ValueError, match="scale requires --scale"):
        build_probe_intervention_specs(
            sites=["attn_out"],
            intervention_names=["scale"],
            layer=0,
            scale=None,
            source_cache=None,
            source_site=None,
            replacement_tensor=None,
            batch_indices=None,
            token_indices=None,
        )


def test_probe_tokenizer_is_config_driven_and_vocab_checked():
    input_ids = encode_prompts(["The history of mathematics"], tokenizer_name="gpt2", block_size=16, vocab_size=50304)

    assert input_ids.ndim == 2
    assert input_ids.max().item() < 50304
    with pytest.raises(ValueError, match="unsupported tokenizer"):
        encode_prompts(["hello"], tokenizer_name="sentencepiece", block_size=8, vocab_size=50304)
    with pytest.raises(ValueError, match="exceeds configured vocab_size"):
        encode_prompts(["The history of mathematics"], tokenizer_name="gpt2", block_size=16, vocab_size=32)


def test_parse_index_list_accepts_commas_and_empty_values():
    assert parse_index_list(None) is None
    assert parse_index_list("") is None
    assert parse_index_list("0, 2,4") == [0, 2, 4]
    with pytest.raises(ValueError, match="integer"):
        parse_index_list("0,nope")
