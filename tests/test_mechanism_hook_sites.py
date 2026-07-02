from __future__ import annotations

import pytest

from attention_lab.mechanisms.hook_sites import (
    UnknownAttentionTypeError,
    get_hook_site_specs,
    get_hook_site_status,
    render_hook_site_docs,
)


def test_registry_returns_required_standard_sites():
    names = {spec.name for spec in get_hook_site_specs("standard")}

    assert {
        "resid_pre[layer]",
        "attn_q[layer]",
        "attn_k[layer]",
        "attn_v[layer]",
        "attn_out[layer]",
        "resid_mid[layer]",
        "mlp_out[layer]",
        "resid_post[layer]",
        "logits",
    }.issubset(names)


def test_registry_returns_architecture_specific_sites_by_attention_type():
    operator_names = {spec.name for spec in get_hook_site_specs("operator_valued_attention")}
    differential_names = {spec.name for spec in get_hook_site_specs("differential_qkv_anti_value")}
    multi_names = {spec.name for spec in get_hook_site_specs("multi_qkv_static_3track_global")}

    assert "operator_suppress_out[layer]" in operator_names
    assert "operator_combined_out[layer]" in operator_names
    assert "neg_out[layer]" in differential_names
    assert "branch_delta[layer]" in differential_names
    assert "track_q[layer, track]" in multi_names
    assert "selected_track[layer]" in multi_names


def test_unknown_attention_types_fail_clearly():
    with pytest.raises(UnknownAttentionTypeError, match="unknown attention_type"):
        get_hook_site_specs("no_such_attention")


def test_unsupported_declared_sites_are_distinguishable_from_missing_sites():
    status = get_hook_site_status("cp_trilinear", "cp_rank_component[layer, rank]")
    assert status.declared
    assert not status.runtime_supported
    assert "summary" in (status.reason or "")

    missing = get_hook_site_status("cp_trilinear", "not_a_site[layer]")
    assert not missing.declared
    assert not missing.runtime_supported
    assert "not declared" in (missing.reason or "")


def test_docs_generated_from_registry_are_deterministic():
    first = render_hook_site_docs()
    second = render_hook_site_docs()

    assert first == second
    assert "standard" in first
    assert "operator_valued_attention" in first
    assert first.index("standard") < first.index("operator_valued_attention")
