from __future__ import annotations

from attention_lab.mechanisms.claim_gates import ClaimGateResult
from attention_lab.mechanisms.summary import render_summary_markdown


def test_summary_includes_required_caveats_and_status_vocabulary_boundary():
    metrics = {
        "schema_version": 1,
        "experiment_id": "E003_qkv_architecture_gauntlet",
        "candidate": "differential",
        "checkpoint": "runs/screen/differential/checkpoints/ckpt_last.pt",
        "canonical_control_checkpoint": "runs/screen/standard_seed1/checkpoints/ckpt_last.pt",
        "actual_control_checkpoint": "runs/screen/standard_seed1/checkpoints/ckpt_last.pt",
        "control_is_canonical": True,
        "task_file": "tasks.json",
        "task_suite_provenance": {"deterministic": True, "generator_name": "unit_test"},
        "pair_counts_per_family": {"negation": 50},
        "hypothesis_doc": "docs/mechanisms/hypotheses/test.yaml",
        "exploratory": False,
        "probe_only": True,
        "sites_evaluated": ["branch_delta[0]"],
        "random_site_null": {"available": False, "reason": "no non-candidate site has matched dimensionality"},
        "alignment_to_control": {"available": True, "probe_direction_cosine_to_control": 0.2},
        "fdr_scope": "every computed site x layer x task_family x metric cell in the run",
        "limitations": ["single-seed"],
    }
    gate = ClaimGateResult(
        status="exploratory_probe_signal",
        status_vocabulary="mechanism_probe_scoped",
        reasons=["probe-only runs cannot reach candidate_mechanism_evidence"],
        caps=["probe_only"],
    )

    summary = render_summary_markdown(metrics, gate)

    assert "mechanism-probe-specific" in summary
    assert "global experiment status vocabulary" in summary
    assert "single-seed" in summary
    assert "not replicated" in summary
    assert "probe-only" in summary
    assert "not causal" in summary
    assert "candidate-to-control alignment is not cross-architecture universality evidence" in summary
    assert "random-site null feasibility limit" in summary


def test_summary_distinguishes_noncanonical_controls_and_missing_decoys():
    metrics = {
        "schema_version": 1,
        "experiment_id": "E004_operator_binding_qkv_gauntlet",
        "candidate": "operator_valued",
        "checkpoint": "candidate.pt",
        "canonical_control_checkpoint": "seed2.pt",
        "actual_control_checkpoint": "seed1.pt",
        "control_is_canonical": False,
        "control_noncanonical_reason": "override does not match canonical seed2 control",
        "task_file": "tasks.json",
        "task_suite_provenance": {"deterministic": False},
        "pair_counts_per_family": {"negation": 12},
        "hypothesis_doc": None,
        "exploratory": True,
        "probe_only": False,
        "sites_evaluated": ["operator_probs[0]"],
        "missing_decoys": True,
        "random_site_null": {"available": True},
        "alignment_to_control": {"available": False, "reason": "shape mismatch"},
        "fdr_scope": "every computed site x layer x task_family x metric cell in the run",
    }
    gate = ClaimGateResult(
        status="insufficient_evidence",
        status_vocabulary="mechanism_probe_scoped",
        reasons=["noncanonical control"],
        caps=["exploratory", "noncanonical_control"],
    )

    summary = render_summary_markdown(metrics, gate)

    assert "noncanonical control" in summary
    assert "missing decoys" in summary
    assert "hand-authored or non-provenance task file" in summary
    assert "not representational novelty evidence by itself" in summary
