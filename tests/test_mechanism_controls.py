from __future__ import annotations

import numpy as np

from attention_lab.mechanisms.controls import is_seed_mismatched, resolve_control, select_random_site_null
from attention_lab.mechanisms.presets import resolve_preset, site_presets_for_names


def test_tier1_presets_resolve_seed_matched_controls():
    e003 = resolve_preset("E003_qkv_architecture_gauntlet", "differential")
    e004 = resolve_preset("E004_operator_binding_qkv_gauntlet", "operator_valued")

    assert e003.matched_control is not None
    assert e003.matched_control.run_name == "standard_refactor_control_30m_seed1_rung500"
    assert "seed1" in str(e003.matched_control.checkpoint_path)

    assert e004.matched_control is not None
    assert e004.matched_control.run_name == "standard_refactor_control_30m_seed2_rung500"
    assert "seed2" in str(e004.matched_control.checkpoint_path)
    assert "seed1" not in str(e004.matched_control.checkpoint_path)


def test_control_override_is_recorded_as_noncanonical(tmp_path):
    preset = resolve_preset("E004_operator_binding_qkv_gauntlet", "operator_valued")
    config = tmp_path / "control.yaml"
    checkpoint = tmp_path / "seed1_control.pt"
    config.write_text("x: 1\n", encoding="utf-8")
    checkpoint.write_bytes(b"not a real checkpoint for resolution-only test")

    resolved = resolve_control(
        preset,
        control_mode="matched",
        control_config=config,
        control_checkpoint=checkpoint,
    )

    assert resolved.override_used
    assert resolved.available
    assert not resolved.canonical
    assert "override" in (resolved.reason or "")


def test_e004_seed1_override_is_seed_mismatched(tmp_path):
    preset = resolve_preset("E004_operator_binding_qkv_gauntlet", "operator_valued")
    config = tmp_path / "control.yaml"
    checkpoint = tmp_path / "standard_refactor_control_30m_seed1_rung500_fake" / "ckpt_last.pt"
    config.write_text("x: 1\n", encoding="utf-8")
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"not a real checkpoint for resolution-only test")

    resolved = resolve_control(
        preset,
        control_mode="matched",
        control_config=config,
        control_checkpoint=checkpoint,
    )

    assert resolved.override_used
    assert resolved.available
    assert is_seed_mismatched(preset, resolved)


def test_control_mode_none_is_unavailable_for_candidate_evidence():
    preset = resolve_preset("E003_qkv_architecture_gauntlet", "differential")
    resolved = resolve_control(
        preset,
        control_mode="none",
        control_config=None,
        control_checkpoint=None,
    )

    assert not resolved.available
    assert not resolved.canonical
    assert "disabled" in (resolved.reason or "")


def test_random_site_null_requires_matched_dimension_and_excludes_candidate():
    preset = resolve_preset("E003_qkv_architecture_gauntlet", "differential")
    candidate = preset.target_sites[0]
    shapes = {
        "branch_delta[0]": (8, 4),
        "attn_out[0]": (8, 4),
        "mlp_out[0]": (8, 5),
    }
    selected = select_random_site_null(
        candidate=candidate,
        candidate_key="branch_delta[0]",
        feature_shapes=shapes,
        pool=preset.random_site_pool,
        seed=0,
    )

    assert selected.available
    assert selected.selected_site == "attn_out[0]"
    assert selected.selected_site != "branch_delta[0]"
    assert selected.selected_feature_dim == 4
    assert any(item["site"] == "mlp_out[0]" and not item["accepted"] for item in selected.considered_sites)


def test_missing_random_site_null_reports_feasibility_limit_for_low_dim_probability_site():
    preset = resolve_preset("E004_operator_binding_qkv_gauntlet", "operator_valued")
    operator_probs = preset.target_sites[0]
    shapes = {
        "operator_probs[0]": (8, 5),
        "attn_out[0]": (8, 384),
        "resid_mid[0]": (8, 384),
    }
    selected = select_random_site_null(
        candidate=operator_probs,
        candidate_key="operator_probs[0]",
        feature_shapes=shapes,
        pool=preset.random_site_pool,
        seed=0,
    )

    assert not selected.available
    assert selected.candidate_feature_dim == 5
    assert selected.candidate_tensor_kind == "probability"
    assert "feasibility limit" in (selected.reason or "")
    assert any("dimension mismatch" in str(item["reason"]) for item in selected.considered_sites)
    assert np.isfinite(selected.candidate_feature_dim)


def test_e004_operator_probs_is_capture_probe_only_not_continuous_patch_site():
    preset = resolve_preset("E004_operator_binding_qkv_gauntlet", "operator_valued")
    operator_probs = next(site for site in preset.target_sites if site.site == "operator_probs")

    assert operator_probs.tensor_kind == "probability"
    assert not operator_probs.continuous
    assert operator_probs.full_layer_site is None
    assert "capture/probe-only" in (operator_probs.no_full_layer_comparator_reason or "")


def test_random_site_null_rejects_same_site_and_incompatible_tensor_kind():
    preset = resolve_preset("E004_operator_binding_qkv_gauntlet", "operator_valued")
    operator_probs = preset.target_sites[0]
    shapes = {
        "operator_probs[0]": (8, 5),
        "attn_out[0]": (8, 5),
    }
    selected = select_random_site_null(
        candidate=operator_probs,
        candidate_key="operator_probs[0]",
        feature_shapes=shapes,
        pool=(operator_probs, *preset.random_site_pool[:1]),
        seed=0,
    )

    assert not selected.available
    assert any("own random-site null" in str(item["reason"]) for item in selected.considered_sites)
    assert any("incompatible tensor kind" in str(item["reason"]) for item in selected.considered_sites)


def test_e004_full_width_operator_output_can_select_matched_random_site():
    preset = resolve_preset("E004_operator_binding_qkv_gauntlet", "operator_valued")
    operator_combined = next(site for site in preset.target_sites if site.site == "operator_combined_out")
    shapes = {
        "operator_combined_out[0]": (8, 384),
        "operator_probs[0]": (8, 5),
        "attn_out[0]": (8, 384),
        "resid_mid[0]": (8, 384),
    }
    selected = select_random_site_null(
        candidate=operator_combined,
        candidate_key="operator_combined_out[0]",
        feature_shapes=shapes,
        pool=preset.random_site_pool,
        seed=0,
    )

    assert selected.available
    assert selected.selected_feature_dim == 384
    assert selected.selected_site in {"attn_out[0]", "resid_mid[0]"}


def test_tier2_tier3_presets_are_not_executable():
    scope = resolve_preset("E003_qkv_architecture_gauntlet", "scope_gated")
    q3 = resolve_preset("E004_operator_binding_qkv_gauntlet", "q3k3v3")

    assert scope.status == "stub_not_executable"
    assert not scope.executable
    assert q3.status == "stub_not_executable"
    assert not q3.executable


def test_confirmatory_site_selection_rejects_wrong_explicit_layer():
    preset = resolve_preset("E003_qkv_architecture_gauntlet", "differential")

    try:
        site_presets_for_names(preset, ["branch_delta[1]"], exploratory=False)
    except ValueError as exc:
        assert "declared layer" in str(exc)
    else:
        raise AssertionError("explicit wrong layer should not resolve to layer 0")
