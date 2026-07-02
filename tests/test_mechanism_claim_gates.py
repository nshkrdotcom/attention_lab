from __future__ import annotations

import numpy as np

from attention_lab.mechanisms.alignment import probe_direction_alignment
from attention_lab.mechanisms.claim_gates import evaluate_claim_gate
from attention_lab.mechanisms.patching import compute_mediation_fraction, compute_restoration_score


def _base_gate_input(**overrides):
    data = {
        "exploratory": False,
        "probe_only": False,
        "has_real_probe_metrics": True,
        "minimum_n_passed": True,
        "confirmatory_floor_passed": True,
        "grouped_split_passed": True,
        "hypothesis_doc_valid": True,
        "matched_control_available": True,
        "control_canonical": True,
        "random_site_null_available": True,
        "stats_valid": True,
        "fdr_primary_passed": True,
        "bootstrap_primary_ci_excludes_null": True,
        "decoy_specificity_passed": True,
        "patching_valid": True,
        "restoration_valid": True,
        "mediation_fraction_valid": True,
        "min_n_below_floor": False,
        "raw_delta_only": False,
    }
    data.update(overrides)
    return data


def test_candidate_mechanism_evidence_requires_full_confirmatory_controls_and_causality():
    result = evaluate_claim_gate(_base_gate_input())

    assert result.status == "candidate_mechanism_evidence"
    assert result.status_vocabulary == "mechanism_probe_scoped"


def test_probe_only_and_exploratory_are_capped_below_confirmatory_claims():
    probe_only = evaluate_claim_gate(_base_gate_input(probe_only=True))
    exploratory = evaluate_claim_gate(_base_gate_input(exploratory=True))

    assert probe_only.status == "exploratory_probe_signal"
    assert exploratory.status == "exploratory_probe_signal"


def test_minimum_n_floor_missing_control_noncanonical_and_decoy_fail_block_gates():
    for override in (
        {"minimum_n_passed": False},
        {"confirmatory_floor_passed": False},
        {"min_n_below_floor": True},
        {"matched_control_available": False},
        {"control_canonical": False},
        {"random_site_null_available": False},
        {"decoy_specificity_passed": False},
        {"fdr_primary_passed": False},
        {"raw_delta_only": True},
    ):
        result = evaluate_claim_gate(_base_gate_input(**override))
        assert result.status == "insufficient_evidence"
        assert result.reasons


def test_controlled_probe_signal_can_pass_when_causal_metrics_are_invalid_but_probe_gates_pass():
    result = evaluate_claim_gate(
        _base_gate_input(
            patching_valid=False,
            restoration_valid=False,
            mediation_fraction_valid=False,
        )
    )

    assert result.status == "controlled_probe_signal"
    assert "causal patch/restoration metrics are invalid or unavailable" in result.reasons


def test_restoration_score_formula_and_denominator_guard():
    score = compute_restoration_score(clean_logitdiff=4.0, corrupted_logitdiff=1.0, patched_logitdiff=2.5)
    invalid = compute_restoration_score(clean_logitdiff=1.0, corrupted_logitdiff=1.0, patched_logitdiff=1.5)

    assert score.valid is True
    assert score.value == 0.5
    assert invalid.valid is False
    assert "denominator" in (invalid.reason or "")


def test_mediation_fraction_formula_and_edge_case():
    fraction = compute_mediation_fraction(component_patch_restoration=0.25, full_layer_patch_restoration=0.5)
    invalid = compute_mediation_fraction(component_patch_restoration=0.25, full_layer_patch_restoration=0.0)

    assert fraction.valid is True
    assert fraction.value == 0.5
    assert invalid.valid is False


def test_alignment_metric_emitted_or_unavailable_without_shape_coercion():
    aligned = probe_direction_alignment(np.array([1.0, 0.0]), np.array([0.5, 0.0]))
    mismatch = probe_direction_alignment(np.array([1.0, 0.0]), np.array([1.0, 0.0, 0.0]))

    assert aligned.available is True
    assert aligned.probe_direction_cosine_to_control == 1.0
    assert aligned.probe_direction_alignment_abs == 1.0
    assert mismatch.available is False
    assert "shape mismatch" in (mismatch.reason or "")
