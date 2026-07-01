from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from attention_lab.training.config import load_config, validate_config
from attention_lab.training.experiments import get_experiment


CANONICAL_CONFIGS = [
    "standard_refactor_control_30m_seed1.yaml",
    "multi_qkv_static_3track_global_30m_seed1.yaml",
    "multi_qkv_train_rotation_3track_global_30m_seed1.yaml",
    "multi_qkv_position_rotation_3track_global_30m_seed1.yaml",
]


@pytest.mark.parametrize("config_name", CANONICAL_CONFIGS)
def test_e002_canonical_configs_validate(repo_root, config_name: str):
    config_path = repo_root / "configs" / "experiments" / "E002_multitrack_qkv_shift_register" / config_name
    config = load_config(config_path)
    validate_config(config)


def _multi_qkv_tiny_config(tiny_config, tmp_path, *, attention_type: str) -> dict:
    config = deepcopy(tiny_config(tmp_path, tmp_path / "data"))
    config["model"].update(
        {
            "attention_type": attention_type,
            "qkv_track_count": 3,
            "qkv_global_bank": True,
            "qkv_route_formula": {
                "multi_qkv_static_3track_global": "layer_mod",
                "multi_qkv_train_rotation_3track_global": "layer_plus_step_train_layer_eval",
                "multi_qkv_position_rotation_3track_global": "layer_plus_position",
            }[attention_type],
            "n_layer": 3,
            "n_head": 2,
        }
    )
    config["queue"] = {
        "family": "multitrack_qkv_shift_register",
        "requires_run": "multi_qkv_static_3track_global_30m_seed1",
        "mechanism_check": "qkv_track_activity",
        "full_run_approved": False,
        "allow_overwrite_existing_run_dir": False,
    }
    return config


def test_multi_qkv_config_rejects_non_global_bank(tiny_config, tmp_path):
    config = _multi_qkv_tiny_config(
        tiny_config,
        tmp_path,
        attention_type="multi_qkv_static_3track_global",
    )
    config["model"]["qkv_global_bank"] = False

    with pytest.raises(ValueError, match="qkv_global_bank"):
        validate_config(config)


def test_multi_qkv_config_rejects_wrong_route_formula(tiny_config, tmp_path):
    config = _multi_qkv_tiny_config(
        tiny_config,
        tmp_path,
        attention_type="multi_qkv_train_rotation_3track_global",
    )
    config["model"]["qkv_route_formula"] = "layer_mod"

    with pytest.raises(ValueError, match="qkv_route_formula"):
        validate_config(config)


def test_multi_qkv_config_rejects_wrong_track_count(tiny_config, tmp_path):
    config = _multi_qkv_tiny_config(
        tiny_config,
        tmp_path,
        attention_type="multi_qkv_position_rotation_3track_global",
    )
    config["model"]["qkv_track_count"] = 2

    with pytest.raises(ValueError, match="qkv_track_count"):
        validate_config(config)


def test_e002_experiment_metadata_includes_canonical_initial_configs():
    experiment = get_experiment("E002_multitrack_qkv_shift_register")
    config_dir = Path(experiment["config_dir"])
    config_names = {path.name for path in config_dir.glob("*.yaml")}

    assert set(CANONICAL_CONFIGS).issubset(config_names)
    assert experiment["canonical_first_build_configs"] == CANONICAL_CONFIGS
    assert experiment["legacy_or_auxiliary_runnable_configs"] == ["standard_30m_seed1.yaml"]
    assert "standard_30m_seed1.yaml" not in experiment["canonical_first_build_configs"]
