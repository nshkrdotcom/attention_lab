from __future__ import annotations

import json
from pathlib import Path

import yaml

from attention_lab.queue.cli import main as queue_main
from attention_lab.queue.gauntlet import load_gauntlet_policy, run_gauntlet_until_blocked_or_complete
from attention_lab.queue.ledger import QueueLedger
from attention_lab.queue.runner import CommandResult


def test_e004_gauntlet_policy_loads_from_repo(repo_root):
    policy = load_gauntlet_policy(
        repo_root / "configs" / "experiments" / "E004_operator_binding_qkv_gauntlet" / "gauntlet_policy.yaml"
    )

    assert policy.experiment_id == "E004_operator_binding_qkv_gauntlet"
    assert policy.control_run_name == "standard_refactor_control_30m_seed2"
    assert [rung.name for rung in policy.rungs] == ["rung020", "rung150", "rung500"]
    assert policy.min_speed_ratio_vs_control == 0.15


def test_e004_gauntlet_run_once_cli_queues_without_training(tiny_config, tmp_path, capsys):
    policy_path = _write_e004_case(tmp_path, tiny_config, variants=("standard",))

    queue_main(
        [
            "--root",
            str(tmp_path),
            "--db",
            str(tmp_path / "queue.db"),
            "gauntlet-run",
            "--experiment",
            "E004_operator_binding_qkv_gauntlet",
            "--policy",
            str(policy_path),
            "--once",
        ]
    )

    output = json.loads(capsys.readouterr().out)
    assert output["action"] == "queued"
    assert output["run_name"] == "standard_refactor_control_30m_seed2_rung020"


def test_e004_gauntlet_until_blocked_screens_all_variants_and_reports_summaries(tiny_config, tmp_path):
    policy_path = _write_e004_case(tmp_path, tiny_config)
    ledger = QueueLedger(tmp_path / "queue.db")
    ledger.initialize()

    result = run_gauntlet_until_blocked_or_complete(
        policy_path=policy_path,
        ledger=ledger,
        repo_root=tmp_path,
        command_runner=_fake_e004_screen_command,
        max_iterations=50,
    )

    report_path = tmp_path / "reports" / "experiments" / "E004_operator_binding_qkv_gauntlet" / "gauntlet_report.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    decisions = {row["candidate"]: row for row in payload["decisions"]}

    assert result["ok"] is True
    assert "operator_valued_attention_30m_seed2_rung020" in decisions
    assert decisions["operator_valued_attention_30m_seed2_rung020"]["operator_combined_output_norm_max"] > 0
    assert decisions["q3k3v3_role_routed_attention_30m_seed2_rung020"]["q3_operator_output_norm_max"] > 0
    assert decisions["dynamic_value_query_conditioned_attention_30m_seed2_rung020"]["dynamic_value_delta_norm_max"] > 0


def _write_e004_case(tmp_path: Path, tiny_config, *, variants: tuple[str, ...] | None = None) -> Path:
    config_dir = tmp_path / "configs" / "experiments" / "E004_operator_binding_qkv_gauntlet"
    config_dir.mkdir(parents=True)
    variants = variants or ("standard", "operator", "q3", "dynamic")
    base_configs = []
    for variant in variants:
        config = tiny_config(tmp_path, tmp_path / "data", max_steps=20)
        config["run"]["seed"] = 1338
        config["run"]["out_dir"] = str(tmp_path / "runs" / variant)
        config.setdefault("queue", {})["family"] = "operator_binding_qkv_gauntlet"
        config["queue"]["full_run_approved"] = False
        config["queue"]["allow_overwrite_existing_run_dir"] = False
        if variant == "standard":
            name = "standard_refactor_control_30m_seed2"
        elif variant == "operator":
            name = "operator_valued_attention_30m_seed2"
            config["model"].update(
                {
                    "attention_type": "operator_valued_attention",
                    "operator_router_hidden_mult": 1.0,
                    "operator_suppress_scale_init": 0.5,
                    "operator_include_bind": True,
                    "operator_include_transform": True,
                }
            )
            _nonstandard_queue(config, "operator_valued_activity")
        elif variant == "q3":
            name = "q3k3v3_role_routed_attention_30m_seed2"
            config["model"].update(
                {
                    "attention_type": "q3k3v3_role_routed_attention",
                    "q3k3v3_role_dim_mode": "equal",
                    "q3k3v3_cross_role_grid": False,
                    "q3k3v3_include_pair_products": True,
                }
            )
            _nonstandard_queue(config, "q3k3v3_role_activity")
        elif variant == "dynamic":
            name = "dynamic_value_query_conditioned_attention_30m_seed2"
            config["model"].update(
                {
                    "attention_type": "dynamic_value_query_conditioned_attention",
                    "dynamic_value_gate_bias_init": 0.0,
                    "dynamic_value_gate_from": "x",
                    "dynamic_value_pairwise_gate": False,
                }
            )
            _nonstandard_queue(config, "dynamic_value_activity")
        else:
            raise AssertionError(variant)
        config["run"]["name"] = name
        config["run"]["out_dir"] = str(tmp_path / "runs" / "experiments" / "E004_operator_binding_qkv_gauntlet" / name)
        path = config_dir / f"{name}.yaml"
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        base_configs.append(str(path.relative_to(tmp_path)))

    policy = {
        "experiment_id": "E004_operator_binding_qkv_gauntlet",
        "control_run_name": "standard_refactor_control_30m_seed2",
        "base_configs": base_configs,
        "rungs": [
            {"name": "rung020", "max_steps": 20, "val_every": 10, "save_every": 20, "diagnostics_every": 10},
        ],
        "gates": {
            "require_loss_descended": True,
            "require_checkpoint": True,
            "require_mechanism_active": True,
            "max_loss_ratio_vs_control": 1.25,
            "min_speed_ratio_vs_control": 0.15,
            "max_vram_ratio_vs_control": 3.25,
            "require_no_nan_or_inf": True,
        },
    }
    policy_path = config_dir / "gauntlet_policy.yaml"
    policy_path.write_text(yaml.safe_dump(policy), encoding="utf-8")
    return policy_path


def _nonstandard_queue(config: dict, mechanism_check: str) -> None:
    config["diagnostics"] = {"attention_diagnostics_every": 10}
    config["queue"].update(
        {
            "requires_run": "standard_refactor_control_30m_seed2",
            "mechanism_check": mechanism_check,
            "allow_missing_diagnostics": False,
        }
    )


def _fake_e004_screen_command(cmd, log_path):  # noqa: ARG001
    screen_dir = Path(log_path).parent
    screen_config = yaml.safe_load((screen_dir / "screen_config.yaml").read_text(encoding="utf-8"))
    max_steps = int(screen_config["train"]["max_steps"])
    val_every = int(screen_config["train"]["val_every"])
    attention_type = screen_config["model"].get("attention_type", "standard")
    metrics = [
        {"event": "train", "step": val_every, "tokens_per_sec": 100.0, "train_loss": 5.5},
        {"event": "val", "step": val_every, "val_loss": 6.0},
        {"event": "train", "step": max_steps, "tokens_per_sec": 100.0, "train_loss": 4.5},
        {"event": "val", "step": max_steps, "val_loss": 5.0, "peak_vram_allocated_mb": 1000.0},
    ]
    (screen_dir / "metrics.jsonl").write_text("\n".join(json.dumps(row) for row in metrics) + "\n", encoding="utf-8")
    (screen_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (screen_dir / "checkpoints" / "ckpt_last.pt").write_bytes(b"checkpoint")
    if attention_type != "standard":
        diag_path = screen_dir / "evals" / "attention_diagnostics.jsonl"
        diag_path.parent.mkdir(parents=True, exist_ok=True)
        diag_path.write_text(json.dumps(_diag_row(attention_type)) + "\n", encoding="utf-8")
    return CommandResult(returncode=0, stdout="ok", stderr="")


def _diag_row(attention_type: str) -> dict:
    if attention_type == "operator_valued_attention":
        return {
            "attention_type": attention_type,
            "operator_prob_add_mean": 0.3,
            "operator_prob_suppress_mean": 0.2,
            "operator_prob_gate_mean": 0.2,
            "operator_prob_transform_mean": 0.15,
            "operator_prob_bind_mean": 0.15,
            "operator_prob_entropy_mean": 1.5,
            "operator_argmax_add_frac": 0.5,
            "operator_argmax_suppress_frac": 0.2,
            "operator_argmax_gate_frac": 0.15,
            "operator_argmax_transform_frac": 0.1,
            "operator_argmax_bind_frac": 0.05,
            "operator_add_output_norm": 0.3,
            "operator_suppress_output_norm": 0.2,
            "operator_gate_output_norm": 0.1,
            "operator_transform_output_norm": 0.1,
            "operator_bind_output_norm": 0.1,
            "operator_combined_output_norm": 0.4,
            "operator_suppress_scale": 0.5,
        }
    if attention_type == "q3k3v3_role_routed_attention":
        return {
            "attention_type": attention_type,
            "q3_content_output_norm": 0.3,
            "q3_operator_output_norm": 0.2,
            "q3_binding_output_norm": 0.25,
            "q3_content_operator_interaction_norm": 0.1,
            "q3_content_binding_interaction_norm": 0.0,
            "q3_operator_binding_interaction_norm": 0.0,
            "q3_content_to_total_norm_ratio": 0.4,
            "q3_operator_to_total_norm_ratio": 0.3,
            "q3_binding_to_total_norm_ratio": 0.3,
            "q3_cross_role_grid_enabled": False,
            "q3_pair_products_enabled": True,
        }
    if attention_type == "dynamic_value_query_conditioned_attention":
        return {
            "attention_type": attention_type,
            "dynamic_value_gate_mean": 0.5,
            "dynamic_value_gate_std": 0.1,
            "dynamic_value_gate_min": 0.2,
            "dynamic_value_gate_max": 0.8,
            "dynamic_value_static_content_norm": 0.3,
            "dynamic_value_gated_content_norm": 0.2,
            "dynamic_value_delta_norm": 0.1,
            "dynamic_value_delta_to_static_ratio": 0.33,
        }
    raise AssertionError(attention_type)
