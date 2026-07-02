from __future__ import annotations

import json

from attention_lab.mechanisms.reports import classify_candidate, generate_cross_experiment_report


def row(**kwargs):
    base = {
        "experiment_id": "E999",
        "run_name": "unknown",
        "attention_type": "standard",
        "checkpoint_status": "checkpoint_unavailable",
        "diagnostics_status": "missing",
        "run_summary_status": "missing",
        "gauntlet_status": "missing",
        "promotion_status": "missing",
        "evidence_level": "not_available",
        "notes": "",
    }
    base.update(kwargs)
    return base


def test_candidate_classification_matches_structured_inputs():
    assert (
        classify_candidate(
            row(
                experiment_id="E004_operator_binding_qkv_gauntlet",
                run_name="operator_valued_attention_30m_seed2_rung500",
                attention_type="operator_valued_attention",
                checkpoint_status="available",
                gauntlet_status="pass",
                promotion_status="promote",
                evidence_level="checkpoint_recompute",
            )
        )
        == "promote_full_mechanism_run"
    )
    assert (
        classify_candidate(
            row(
                experiment_id="E004_operator_binding_qkv_gauntlet",
                run_name="dynamic_value_query_conditioned_attention_30m_seed2_rung500",
                attention_type="dynamic_value_query_conditioned_attention",
                gauntlet_status="kill",
                promotion_status="kill",
                evidence_level="artifact_summary",
            )
        )
        == "diagnostic_rescue"
    )
    assert (
        classify_candidate(
            row(
                experiment_id="E002_multitrack_qkv_shift_register",
                run_name="multi_qkv_static_3track_global_30m_seed1",
                attention_type="multi_qkv_static_3track_global",
                checkpoint_status="available",
                evidence_level="checkpoint_recompute",
            )
        )
        == "route_specialization_workbench"
    )


def test_report_generation_is_deterministic_and_includes_cannot_conclude(tmp_path):
    backfill_root = tmp_path / "backfill"
    exp_dir = backfill_root / "E004_operator_binding_qkv_gauntlet"
    exp_dir.mkdir(parents=True)
    inventory = {
        "schema_version": 1,
        "experiment_id": "E004_operator_binding_qkv_gauntlet",
        "candidates": [
            row(
                experiment_id="E004_operator_binding_qkv_gauntlet",
                run_name="operator_valued_attention_30m_seed2_rung500",
                attention_type="operator_valued_attention",
                checkpoint_status="available",
                gauntlet_status="pass",
                promotion_status="promote",
                evidence_level="checkpoint_recompute",
            ),
            row(
                experiment_id="E004_operator_binding_qkv_gauntlet",
                run_name="base_unrun",
                attention_type="operator_valued_attention",
            ),
        ],
    }
    (exp_dir / "inventory.json").write_text(json.dumps(inventory), encoding="utf-8")

    first = generate_cross_experiment_report(backfill_root)
    second = generate_cross_experiment_report(backfill_root)

    assert first == second
    assert "promote_full_mechanism_run" in first
    assert "What Cannot Be Concluded" in first
    assert "base_unrun" in first


def test_report_refuses_to_overstate_missing_evidence():
    classification = classify_candidate(row(attention_type="operator_valued_attention"))

    assert classification == "not_evaluated"
    assert (
        classify_candidate(
            row(
                experiment_id="E004_operator_binding_qkv_gauntlet",
                attention_type="q3k3v3_role_routed_attention",
            )
        )
        == "not_evaluated"
    )


def test_positive_workbench_classifications_require_available_evidence():
    assert (
        classify_candidate(
            row(
                experiment_id="E001_cp_trilinear_attention",
                run_name="cp_trilinear_r8_lambda0_30m_seed1",
                attention_type="cp_trilinear",
                evidence_level="not_available",
            )
        )
        == "not_evaluated"
    )
    assert (
        classify_candidate(
            row(
                experiment_id="E002_multitrack_qkv_shift_register",
                run_name="multi_qkv_static_3track_global_30m_seed1",
                attention_type="multi_qkv_static_3track_global",
                evidence_level="not_available",
            )
        )
        == "not_evaluated"
    )
