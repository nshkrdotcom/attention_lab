from __future__ import annotations

from attention_lab.mechanisms.alignment import probe_direction_alignment
from attention_lab.mechanisms.summary import render_summary


def test_summary_includes_single_seed_and_status_vocabulary_caveats():
    metrics = {
        "run": {
            "experiment_id": "E003_qkv_architecture_gauntlet",
            "candidate": "differential",
            "checkpoint": "ckpt.pt",
            "task_file": "tasks.yaml",
        },
        "mode": {"exploratory": True, "probe_only": True},
        "control": {
            "expected_control_checkpoint": "seed1_control.pt",
            "actual_control_checkpoint": "seed1_control.pt",
            "canonical": True,
            "override_used": False,
            "available": True,
        },
        "task_suite": {
            "deterministic_provenance": False,
            "confirmatory_floor_met": False,
            "pair_counts_by_family": {"negation": 4},
            "validation_errors": [],
            "validation_warnings": ["small"],
        },
        "fdr_bh": {"alpha": 0.05, "tested_cells": []},
        "cells": {
            "branch_delta[0]|family=negation": {
                "linear_probe_auc": 0.8,
                "auc_minus_shuffled_auc": 0.2,
                "auc_minus_random_site_auc": None,
                "auc_minus_matched_control_auc": 0.1,
                "target_vs_decoy_specificity": 0.05,
                "random_site_null": {
                    "random_site_null_available": False,
                    "reason": "no matched dimensionality",
                },
                "alignment_to_control": {
                    "available": True,
                    "probe_direction_cosine_to_control": 0.3,
                },
            }
        },
    }
    claim_gates = {
        "overall_status": "exploratory_probe_signal",
        "cells": {"branch_delta[0]|family=negation": {"status": "exploratory_probe_signal", "blockers": []}},
    }

    summary = render_summary(metrics, claim_gates)

    assert "mechanism-probe-specific claim ladder" in summary
    assert "single-seed" in summary
    assert "not a replicated finding" in summary
    assert "Candidate-to-control alignment is not cross-architecture universality evidence" in summary
    assert "feasibility limits" in summary
    assert "Probe-only mode skipped" in summary
    assert "Exploratory mode capped" in summary


def test_alignment_metric_unavailable_on_shape_mismatch_and_not_novelty_claim():
    result = probe_direction_alignment([1.0, 0.0], [1.0, 0.0, 0.0])
    payload = result.to_dict()

    assert not result.available
    assert "shape mismatch" in (result.reason or "")
    assert "not representational novelty evidence" in payload["interpretation"]


def test_alignment_metric_emits_cosine_when_shapes_match():
    result = probe_direction_alignment([1.0, 0.0], [0.0, 1.0])

    assert result.available
    assert result.probe_direction_cosine_to_control == 0.0
    assert result.probe_direction_alignment_abs == 0.0
