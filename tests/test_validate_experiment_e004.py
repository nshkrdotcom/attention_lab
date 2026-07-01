from __future__ import annotations

import pytest

from attention_lab.queue.gauntlet import load_gauntlet_policy
from attention_lab.training.config import load_config
from attention_lab.training.validate_experiment import validate_experiment


def test_validate_e004_operator_binding_qkv_gauntlet_configs():
    result = validate_experiment("E004_operator_binding_qkv_gauntlet")

    assert result["ok"] is True
    assert result["config_count"] == 4
    assert result["runnable_config_count"] == 4
    assert result["unimplemented_config_count"] == 0
    assert result["non_run_config_count"] == 1
    assert result["canonical_first_build_configs"] == [
        "standard_refactor_control_30m_seed2.yaml",
        "operator_valued_attention_30m_seed2.yaml",
        "q3k3v3_role_routed_attention_30m_seed2.yaml",
        "dynamic_value_query_conditioned_attention_30m_seed2.yaml",
    ]


def test_e004_base_configs_load_and_policy_is_not_training_config(repo_root):
    config_dir = repo_root / "configs" / "experiments" / "E004_operator_binding_qkv_gauntlet"
    expected = {
        "standard_refactor_control_30m_seed2.yaml": "standard",
        "operator_valued_attention_30m_seed2.yaml": "operator_valued_attention",
        "q3k3v3_role_routed_attention_30m_seed2.yaml": "q3k3v3_role_routed_attention",
        "dynamic_value_query_conditioned_attention_30m_seed2.yaml": "dynamic_value_query_conditioned_attention",
    }
    for name, attention_type in expected.items():
        config = load_config(config_dir / name)
        assert config["model"]["attention_type"] == attention_type

    with pytest.raises(ValueError, match="missing sections"):
        load_config(config_dir / "gauntlet_policy.yaml")
    policy = load_gauntlet_policy(config_dir / "gauntlet_policy.yaml")
    assert policy.experiment_id == "E004_operator_binding_qkv_gauntlet"
    assert policy.control_run_name == "standard_refactor_control_30m_seed2"
    assert len(policy.rungs) == 3
    assert policy.max_loss_ratio_vs_control == 1.25
