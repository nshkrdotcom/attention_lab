from __future__ import annotations

import numpy as np

from attention_lab.mechanisms.controls import (
    ActivationMatrix,
    choose_random_site_null,
    resolve_control,
)
from attention_lab.mechanisms.presets import get_preset


def test_e003_resolves_seed1_canonical_control():
    preset = get_preset("E003_qkv_architecture_gauntlet", "differential")
    control = resolve_control(preset)

    assert control.expected_run_name == "standard_refactor_control_30m_seed1_rung500"
    assert "seed1" in str(control.control_checkpoint)
    assert control.is_canonical is True
    assert control.noncanonical_reason is None


def test_e004_resolves_seed2_canonical_control_not_seed1():
    preset = get_preset("E004_operator_binding_qkv_gauntlet", "operator_valued")
    control = resolve_control(preset)

    assert control.expected_run_name == "standard_refactor_control_30m_seed2_rung500"
    assert "seed2" in str(control.control_checkpoint)
    assert "seed1" not in str(control.control_checkpoint)
    assert control.is_canonical is True


def test_control_override_is_recorded_and_marked_noncanonical():
    preset = get_preset("E004_operator_binding_qkv_gauntlet", "operator_valued")
    control = resolve_control(
        preset,
        control_checkpoint="runs/screen/standard_refactor_control_30m_seed1_rung500/checkpoints/ckpt_last.pt",
        control_config="configs/experiments/E003_qkv_architecture_gauntlet/standard_refactor_control_30m_seed1_rung500.yaml",
    )

    assert control.is_override is True
    assert control.is_canonical is False
    assert "does not match canonical" in (control.noncanonical_reason or "")


def test_random_site_null_selects_matched_dimensional_site_and_excludes_candidate():
    activations = {
        "branch_delta[0]": ActivationMatrix(
            site="branch_delta[0]",
            X=np.zeros((8, 4)),
            tensor_kind="activation",
            shape=(8, 4),
        ),
        "pos_out[0]": ActivationMatrix(
            site="pos_out[0]",
            X=np.ones((8, 4)),
            tensor_kind="activation",
            shape=(8, 4),
        ),
        "operator_probs[0]": ActivationMatrix(
            site="operator_probs[0]",
            X=np.ones((8, 5)),
            tensor_kind="probability",
            shape=(8, 5),
        ),
    }

    selected = choose_random_site_null(
        candidate_site="branch_delta[0]",
        candidate=activations["branch_delta[0]"],
        available=activations,
        seed=13,
    )

    assert selected.available is True
    assert selected.selected_site == "pos_out[0]"
    assert selected.selected_site != "branch_delta[0]"
    assert selected.reason is None


def test_random_site_null_reports_missing_matched_shape_without_coercion():
    activations = {
        "operator_probs[0]": ActivationMatrix(
            site="operator_probs[0]",
            X=np.zeros((8, 5)),
            tensor_kind="probability",
            shape=(8, 5),
        ),
        "operator_combined_out[0]": ActivationMatrix(
            site="operator_combined_out[0]",
            X=np.zeros((8, 16)),
            tensor_kind="activation",
            shape=(8, 16),
        ),
    }

    selected = choose_random_site_null(
        candidate_site="operator_probs[0]",
        candidate=activations["operator_probs[0]"],
        available=activations,
        seed=2,
    )

    assert selected.available is False
    assert selected.selected_site is None
    assert "matched dimensionality" in (selected.reason or "")


def test_tier2_and_tier3_presets_are_not_executable():
    preset = get_preset("E004_operator_binding_qkv_gauntlet", "dynamic_value")

    assert preset.executable is False
    assert preset.status == "stub_not_executable"
