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
    # +4 vs. the original 11/5: seed-replication configs added 2026-07-06
    # (multi_qkv_static/train_rotation seed1338/1339), see test_config.py's
    # test_e002_skeleton_config_contract for why they're runnable, not
    # unimplemented skeletons.
    assert result["config_count"] == 15
    assert result["runnable_config_count"] == 9
    assert result["unimplemented_config_count"] == 6
