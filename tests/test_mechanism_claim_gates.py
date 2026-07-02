from __future__ import annotations

from attention_lab.mechanisms.claim_gates import (
    CANDIDATE_MECHANISM_EVIDENCE,
    CONTROLLED_PROBE_SIGNAL,
    EXPLORATORY_PROBE_SIGNAL,
    INSUFFICIENT_EVIDENCE,
    CellGateInputs,
    evaluate_cell_claim_gate,
    overall_status,
)


def _passing_inputs(**overrides) -> CellGateInputs:
    data = dict(
        exploratory=False,
        probe_only=False,
        hypothesis_doc_valid=True,
        real_probe_metrics=True,
        min_n_passed=True,
        confirmatory_floor_met=True,
        grouped_split=True,
        matched_control_available=True,
        canonical_control=True,
        noncanonical_control=False,
        shuffled_null_passed=True,
        random_site_null_available=True,
        random_site_null_passed=True,
        matched_control_passed=True,
        primary_fdr_passed=True,
        primary_ci_passed=True,
        specificity_fdr_passed=True,
        specificity_ci_passed=True,
        patching_valid=True,
        mediation_valid=True,
    )
    data.update(overrides)
    return CellGateInputs(**data)


def test_candidate_mechanism_evidence_requires_all_gates():
    result = evaluate_cell_claim_gate(_passing_inputs())

    assert result.status == CANDIDATE_MECHANISM_EVIDENCE


def test_raw_delta_or_missing_stats_cannot_pass_gate():
    result = evaluate_cell_claim_gate(_passing_inputs(primary_fdr_passed=False))

    assert result.status == INSUFFICIENT_EVIDENCE
    assert any("primary probe" in blocker for blocker in result.blockers)


def test_probe_only_and_exploratory_modes_are_capped():
    probe_only = evaluate_cell_claim_gate(_passing_inputs(probe_only=True))
    exploratory = evaluate_cell_claim_gate(_passing_inputs(exploratory=True))

    assert probe_only.status == EXPLORATORY_PROBE_SIGNAL
    assert exploratory.status == EXPLORATORY_PROBE_SIGNAL
    assert not probe_only.claim_gate_passed
    assert not exploratory.claim_gate_passed
    assert probe_only.to_dict()["status_kind"] == "exploratory_signal"
    assert exploratory.to_dict()["status_kind"] == "exploratory_signal"


def test_missing_random_site_null_caps_only_affected_cell():
    missing = evaluate_cell_claim_gate(
        _passing_inputs(random_site_null_available=False, random_site_null_passed=False)
    )
    passing = evaluate_cell_claim_gate(_passing_inputs())

    assert missing.status == INSUFFICIENT_EVIDENCE
    assert "missing random-site null caps only this site-layer cell" in missing.caps
    assert overall_status([missing, passing]) == CANDIDATE_MECHANISM_EVIDENCE


def test_missing_or_seed_mismatched_control_blocks_candidate_evidence():
    missing = evaluate_cell_claim_gate(_passing_inputs(matched_control_available=False))
    mismatched = evaluate_cell_claim_gate(
        _passing_inputs(canonical_control=False, noncanonical_control=True, force_noncanonical_control=True)
    )

    assert missing.status == INSUFFICIENT_EVIDENCE
    assert mismatched.status == INSUFFICIENT_EVIDENCE
    assert any("control" in blocker for blocker in mismatched.blockers)


def test_valid_controlled_probe_without_patching_stops_below_candidate_evidence():
    result = evaluate_cell_claim_gate(_passing_inputs(patching_valid=False, mediation_valid=False))

    assert result.status == CONTROLLED_PROBE_SIGNAL
    assert result.claim_gate_passed
    assert result.to_dict()["status_kind"] == "confirmatory_claim"
    assert any("patching" in blocker for blocker in result.blockers)


def test_candidate_mechanism_evidence_requires_full_layer_restoration_fdr():
    result = evaluate_cell_claim_gate(_passing_inputs(full_layer_patching_fdr_passed=False))

    assert result.status == CONTROLLED_PROBE_SIGNAL
    assert any("restoration/mediation" in blocker for blocker in result.blockers)


def test_candidate_mechanism_evidence_requires_task_aligned_pooling():
    result = evaluate_cell_claim_gate(_passing_inputs(task_aligned_pooling=False))

    assert result.status == CONTROLLED_PROBE_SIGNAL
    assert any("task-aligned" in blocker for blocker in result.blockers)


def test_candidate_mechanism_evidence_requires_valid_restoration_alignment():
    result = evaluate_cell_claim_gate(_passing_inputs(restoration_alignment_valid=False, patching_valid=False))

    assert result.status == CONTROLLED_PROBE_SIGNAL
    assert any("alignment" in blocker for blocker in result.blockers)


def test_min_n_floor_grouping_and_decoy_specificity_block_gates():
    for kwargs in (
        {"min_n_passed": False},
        {"confirmatory_floor_met": False},
        {"grouped_split": False},
        {"specificity_fdr_passed": False},
        {"specificity_ci_passed": False},
    ):
        assert evaluate_cell_claim_gate(_passing_inputs(**kwargs)).status == INSUFFICIENT_EVIDENCE


def test_controlled_probe_signal_requires_random_null_and_fdr_not_raw_auc():
    missing_random = evaluate_cell_claim_gate(_passing_inputs(random_site_null_available=False, random_site_null_passed=False))
    raw_auc_only = evaluate_cell_claim_gate(_passing_inputs(primary_fdr_passed=False, primary_ci_passed=True))

    assert missing_random.status == INSUFFICIENT_EVIDENCE
    assert raw_auc_only.status == INSUFFICIENT_EVIDENCE


def test_specificity_ci_without_fdr_cannot_pass_candidate_evidence():
    result = evaluate_cell_claim_gate(_passing_inputs(specificity_fdr_passed=False, specificity_ci_passed=True))

    assert result.status == INSUFFICIENT_EVIDENCE
    assert any("specificity" in blocker for blocker in result.blockers)


def test_noncanonical_forced_control_still_blocks_candidate_mechanism_evidence():
    result = evaluate_cell_claim_gate(
        _passing_inputs(canonical_control=False, noncanonical_control=True, force_noncanonical_control=True)
    )

    assert result.status == INSUFFICIENT_EVIDENCE
    assert any("noncanonical" in cap for cap in result.caps)


def test_hand_authored_or_small_task_suite_blocks_confirmatory_gate():
    result = evaluate_cell_claim_gate(
        _passing_inputs(
            min_n_passed=False,
            confirmatory_floor_met=False,
            extra_blockers=("confirmatory task suite lacks deterministic generator provenance",),
        )
    )

    assert result.status == INSUFFICIENT_EVIDENCE
    assert any("provenance" in blocker for blocker in result.blockers)
