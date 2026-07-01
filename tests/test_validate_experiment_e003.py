from __future__ import annotations

import pytest

from attention_lab.queue.gauntlet import load_gauntlet_policy
from attention_lab.training.config import load_config
from attention_lab.training.validate_experiment import validate_experiment


def test_validate_e003_qkv_gauntlet_configs():
    result = validate_experiment("E003_qkv_architecture_gauntlet")

    assert result["ok"] is True
    assert result["config_count"] == 3
    assert result["runnable_config_count"] == 3
    assert result["unimplemented_config_count"] == 0
    assert result["non_run_config_count"] == 1
    assert result["canonical_first_build_configs"] == [
        "standard_refactor_control_30m_seed1.yaml",
        "differential_qkv_anti_value_30m_seed1.yaml",
        "scope_gated_qkv_30m_seed1.yaml",
    ]


def test_e003_base_configs_load_and_policy_is_not_training_config(repo_root):
    config_dir = repo_root / "configs" / "experiments" / "E003_qkv_architecture_gauntlet"
    for name in (
        "standard_refactor_control_30m_seed1.yaml",
        "differential_qkv_anti_value_30m_seed1.yaml",
        "scope_gated_qkv_30m_seed1.yaml",
    ):
        assert load_config(config_dir / name)

    with pytest.raises(ValueError, match="missing sections"):
        load_config(config_dir / "gauntlet_policy.yaml")
    policy = load_gauntlet_policy(config_dir / "gauntlet_policy.yaml")
    assert policy.experiment_id == "E003_qkv_architecture_gauntlet"
    assert len(policy.rungs) == 3
