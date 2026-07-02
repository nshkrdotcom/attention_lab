from __future__ import annotations

from attention_lab.mechanisms.alignment import probe_direction_alignment
import json

from attention_lab.mechanisms.summary import render_summary, validate_suite_artifacts, write_suite_artifacts


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
        "feature_pooling": {"strategy": "mean_sequence", "task_aligned": False},
        "cells": {
            "branch_delta[0]|family=negation": {
                "site": "branch_delta",
                "layer": 0,
                "family_id": "negation",
                "feature_pooling": {"strategy": "mean_sequence", "task_aligned": False},
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
                "matched_control": {"available": True},
                "patching": {"valid": False, "reason": "probe-only mode"},
                "mediation_fraction": {"valid": False, "mediation_fraction": None},
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
    assert "feature_pooling" in summary
    assert "FDR-BH reports both tested metric cells and invalid/unavailable cells" in summary
    assert "Probe-only mode skipped" in summary
    assert "Exploratory mode capped" in summary


def test_suite_artifact_validation_checks_required_schema(tmp_path):
    metrics = {
        "schema_version": 1,
        "run": {"experiment_id": "E003", "candidate": "differential", "checkpoint": "ckpt.pt", "task_file": "tasks.yaml"},
        "mode": {"exploratory": True, "probe_only": True},
        "control": {"available": True},
        "task_suite": {},
        "sites_evaluated": ["branch_delta[0]"],
        "feature_pooling": {"strategy": "mean_sequence", "task_aligned": False},
        "fdr_bh": {
            "comparison_family": "every computed (site x layer x task_family x metric) cell in the run",
            "tested_cells": [],
            "invalid_or_unavailable_cells": [],
            "results": {},
        },
        "cells": {
            "branch_delta[0]|family=negation": {
                "site": "branch_delta",
                "layer": 0,
                "family_id": "negation",
                "feature_pooling": {"strategy": "mean_sequence", "task_aligned": False},
                "linear_probe_auc": 0.5,
                "random_site_null": {},
                "matched_control": {},
                "alignment_to_control": {},
                "patching": {},
                "mediation_fraction": {},
            }
        },
    }
    claim_gates = {
        "overall_status": "exploratory_probe_signal",
        "status_vocabulary": ["insufficient_evidence", "exploratory_probe_signal"],
        "status_vocabulary_scope": "mechanism-probe scoped",
        "cells": {"branch_delta[0]|family=negation": {"status": "exploratory_probe_signal", "blockers": []}},
    }

    write_suite_artifacts(tmp_path, metrics, claim_gates)
    assert validate_suite_artifacts(tmp_path) == []

    broken = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    del broken["cells"]["branch_delta[0]|family=negation"]["feature_pooling"]
    (tmp_path / "metrics.json").write_text(json.dumps(broken), encoding="utf-8")
    errors = validate_suite_artifacts(tmp_path)
    assert any("feature_pooling" in error for error in errors)


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
