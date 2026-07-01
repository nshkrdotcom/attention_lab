from __future__ import annotations

from attention_lab.training.validate_experiment import validate_experiment


def test_validate_e001_experiment():
    result = validate_experiment("E001_cp_trilinear_attention")
    assert result["ok"] is True
    assert result["config_count"] == 5
    assert result["runnable_config_count"] == 5
    assert result["unimplemented_config_count"] == 0


def test_validate_e002_experiment_skeleton():
    result = validate_experiment("E002_multitrack_qkv_shift_register")
    assert result["ok"] is True
    assert result["config_count"] == 11
    assert result["runnable_config_count"] == 5
    assert result["unimplemented_config_count"] == 6
