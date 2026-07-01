from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from attention_lab.models.gpt import GPT, config_from_dict
from attention_lab.training.config import load_config, validate_config


def test_valid_baseline_sanity_config_loads(repo_root):
    config = load_config(repo_root / "configs" / "baseline_15m_fineweb100m_sanity.yaml")
    assert config["run"]["name"] == "baseline_15m_fineweb100m_sanity_seed1"
    assert config["model"]["attention_type"] == "standard"


def test_baseline_config_ladder_loads(repo_root):
    config_names = [
        "baseline_15m_fineweb100m.yaml",
        "baseline_16m_fineweb100m.yaml",
        "baseline_30m_fineweb100m.yaml",
        "baseline_70m_fineweb300m.yaml",
        "baseline_125m_fineweb1b.yaml",
    ]
    for config_name in config_names:
        config = load_config(repo_root / "configs" / config_name)
        assert config["model"]["attention_type"] == "standard"
        assert config["train"]["total_batch_size"] % (config["train"]["B"] * config["train"]["T"]) == 0


def test_historical_15m_and_30m_alias_have_same_model_shape(repo_root):
    historical = load_config(repo_root / "configs" / "baseline_15m_fineweb100m.yaml")
    canonical = load_config(repo_root / "configs" / "baseline_30m_fineweb100m.yaml")
    assert historical["model"] == canonical["model"]
    assert historical["data"]["vocab_size"] == canonical["data"]["vocab_size"]


def test_missing_required_section_fails(tiny_config, tmp_path):
    config = tiny_config(tmp_path, tmp_path / "data")
    del config["data"]
    with pytest.raises(ValueError, match="missing sections"):
        validate_config(config, source=tmp_path / "bad.yaml")


def test_unknown_model_key_fails(tiny_config, tmp_path):
    config = deepcopy(tiny_config(tmp_path, tmp_path / "data"))
    config["model"]["not_a_real_key"] = 1
    with pytest.raises(ValueError, match="Unknown model config keys"):
        validate_config(config)


def test_unknown_train_key_fails(tiny_config, tmp_path):
    config = deepcopy(tiny_config(tmp_path, tmp_path / "data"))
    config["train"]["learningrate"] = 0.0006
    with pytest.raises(ValueError, match="Unknown train config keys"):
        validate_config(config)


def test_unknown_data_key_fails(tiny_config, tmp_path):
    config = deepcopy(tiny_config(tmp_path, tmp_path / "data"))
    config["data"]["toknizer"] = "gpt2"
    with pytest.raises(ValueError, match="Unknown data config keys"):
        validate_config(config)


def test_unknown_sample_key_fails(tiny_config, tmp_path):
    config = deepcopy(tiny_config(tmp_path, tmp_path / "data"))
    config["sample"]["sample_evey"] = 1
    with pytest.raises(ValueError, match="Unknown sample config keys"):
        validate_config(config)


def test_invalid_dtype_fails(tiny_config, tmp_path):
    config = deepcopy(tiny_config(tmp_path, tmp_path / "data"))
    config["train"]["dtype"] = "fp8"
    with pytest.raises(ValueError, match="train.dtype"):
        validate_config(config)


def test_n_embd_must_be_divisible_by_n_head(tiny_config, tmp_path):
    config = deepcopy(tiny_config(tmp_path, tmp_path / "data"))
    config["model"]["n_head"] = 3
    with pytest.raises(ValueError, match="n_embd"):
        validate_config(config)


def test_total_batch_size_must_be_divisible_by_micro_batch(tiny_config, tmp_path):
    config = deepcopy(tiny_config(tmp_path, tmp_path / "data"))
    config["train"]["total_batch_size"] = 17
    with pytest.raises(ValueError, match="total_batch_size"):
        validate_config(config)


def test_unimplemented_trilinear_config_not_runnable_baseline(repo_root):
    config_path = repo_root / "configs" / "experimental" / "trilinear_cp_15m_fineweb100m.yaml"
    with pytest.raises(ValueError, match="experimental"):
        load_config(config_path)


def test_compile_requires_explicit_experimental_flag(tiny_config, tmp_path):
    config = deepcopy(tiny_config(tmp_path, tmp_path / "data"))
    config["train"]["compile"] = True
    with pytest.raises(ValueError, match="compile"):
        validate_config(config)


def test_e001_runnable_configs_load_and_instantiate(repo_root):
    config_dir = repo_root / "configs" / "experiments" / "E001_cp_trilinear_attention"
    config_names = [
        "standard_30m_seed1.yaml",
        "standard_refactor_control_30m_seed1.yaml",
        "cp_bilinear_r8_30m_seed1.yaml",
        "cp_trilinear_r8_30m_seed1.yaml",
        "cp_trilinear_r8_lambda0_30m_seed1.yaml",
    ]
    for config_name in config_names:
        config = load_config(config_dir / config_name)
        model = GPT(config_from_dict(config["model"], config["data"]))
        assert model.num_parameters() >= 29_938_560


def test_invalid_cp_lambda_flags_fail(tiny_config, tmp_path):
    config = deepcopy(tiny_config(tmp_path, tmp_path / "data"))
    config["model"]["attention_type"] = "cp_trilinear"
    config["model"]["cp_rank"] = 8
    config["model"]["cp_lambda_trainable"] = True
    config["model"]["cp_lambda_fixed"] = True
    with pytest.raises(ValueError, match="cannot both be true"):
        validate_config(config)


def test_unknown_diagnostics_key_fails(tiny_config, tmp_path):
    config = deepcopy(tiny_config(tmp_path, tmp_path / "data"))
    config["diagnostics"] = {"attention_diagnostics_evey": 1}
    with pytest.raises(ValueError, match="Unknown diagnostics config keys"):
        validate_config(config)


def test_e001_configs_have_distinct_experiment_run_dirs(repo_root):
    import yaml

    config_dir = repo_root / "configs" / "experiments" / "E001_cp_trilinear_attention"
    experiment_run_dir = Path("runs/experiments/E001_cp_trilinear_attention")
    configs = []
    for config_path in sorted(config_dir.glob("*.yaml")):
        with config_path.open("r", encoding="utf-8") as f:
            configs.append(yaml.safe_load(f))

    out_dirs = [Path(config["run"]["out_dir"]) for config in configs]
    assert len(out_dirs) == len(set(out_dirs))
    assert all(str(out_dir).startswith(str(experiment_run_dir)) for out_dir in out_dirs)

    fixed_fields = {
        "data": ("data_root", "tokenizer", "vocab_size", "train_tokens", "val_tokens"),
        "model": ("block_size", "n_layer", "n_head", "n_embd", "dropout", "bias"),
        "train": (
            "B",
            "T",
            "total_batch_size",
            "max_steps",
            "grad_clip",
            "weight_decay",
            "learning_rate",
            "min_lr",
            "warmup_steps",
            "val_every",
            "val_steps",
            "save_every",
            "log_every",
        ),
        "sample": ("sample_every", "prompt", "num_samples", "max_new_tokens", "top_k", "temperature", "seed"),
    }
    reference = configs[0]
    for config in configs[1:]:
        for section, keys in fixed_fields.items():
            for key in keys:
                assert config[section][key] == reference[section][key], f"{section}.{key}"


def test_e002_skeleton_config_contract(repo_root):
    import yaml

    config_dir = repo_root / "configs" / "experiments" / "E002_multitrack_qkv_shift_register"
    standard = load_config(config_dir / "standard_30m_seed1.yaml")
    assert standard["model"]["attention_type"] == "standard"
    assert standard["queue"]["family"] == "multitrack_qkv_shift_register"
    runnable_names = {
        "standard_30m_seed1.yaml",
        "standard_refactor_control_30m_seed1.yaml",
        "multi_qkv_static_3track_global_30m_seed1.yaml",
        "multi_qkv_train_rotation_3track_global_30m_seed1.yaml",
        "multi_qkv_position_rotation_3track_global_30m_seed1.yaml",
    }

    configs = []
    for config_path in sorted(config_dir.glob("*.yaml")):
        with config_path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        configs.append(config)
        if config_path.name in runnable_names:
            assert load_config(config_path)
        else:
            with pytest.raises(ValueError, match="experimental"):
                load_config(config_path)
            assert config["status"] == "experimental_unimplemented"
            assert config["queue"]["mechanism_check"] == "qkv_track_activity"

    out_dirs = [Path(config["run"]["out_dir"]) for config in configs]
    assert len(out_dirs) == len(set(out_dirs))
    assert all(str(path).startswith("runs/experiments/E002_multitrack_qkv_shift_register") for path in out_dirs)


def test_multi_qkv_config_validation_requires_global_three_track(tiny_config, tmp_path):
    config = tiny_config(tmp_path, tmp_path / "data")
    config["model"].update(
        {
            "attention_type": "multi_qkv_static_3track_global",
            "qkv_track_count": 2,
            "qkv_global_bank": True,
            "qkv_route_formula": "layer_mod",
        }
    )
    with pytest.raises(ValueError, match="3track_global"):
        validate_config(config)

    config["model"]["qkv_track_count"] = 3
    config["model"]["qkv_global_bank"] = False
    with pytest.raises(ValueError, match="qkv_global_bank"):
        validate_config(config)

    config["model"]["qkv_global_bank"] = True
    config["model"]["qkv_route_formula"] = "wrong"
    with pytest.raises(ValueError, match="qkv_route_formula"):
        validate_config(config)
