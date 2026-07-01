from __future__ import annotations

import os
from pathlib import Path

from attention_lab.training.experiments import get_experiment, list_experiments


def test_e004_is_registered_with_existing_paths():
    ids = [experiment["id"] for experiment in list_experiments()]
    assert "E004_operator_binding_qkv_gauntlet" in ids

    experiment = get_experiment("E004_operator_binding_qkv_gauntlet")
    for key in ("plan", "config_dir", "report_dir", "baseline_config", "dataset_manifest"):
        assert Path(experiment[key]).exists(), key
    assert experiment["policy_configs"] == ["gauntlet_policy.yaml"]
    assert experiment["canonical_first_build_configs"] == [
        "standard_refactor_control_30m_seed2.yaml",
        "operator_valued_attention_30m_seed2.yaml",
        "q3k3v3_role_routed_attention_30m_seed2.yaml",
        "dynamic_value_query_conditioned_attention_30m_seed2.yaml",
    ]


def test_e004_script_is_executable_and_screen_first(repo_root):
    script = repo_root / "scripts" / "experiments" / "E004_operator_binding_qkv_gauntlet" / "run_gauntlet.sh"
    assert script.exists()
    assert os.access(script, os.X_OK)

    text = script.read_text(encoding="utf-8")
    assert "scripts/verify_cuda.py" in text
    assert "scripts/verify_data.py" in text
    assert "scripts/validate_experiment.py --id" in text
    assert "gauntlet-plan" in text
    assert "gauntlet-run" in text
    assert "--until-blocked" in text
    assert "--allow-full" in text
    assert "approve" not in text
