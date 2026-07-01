from __future__ import annotations

import json
from pathlib import Path

import yaml

from attention_lab.queue.cli import main as queue_main
from attention_lab.queue.gauntlet import (
    GauntletPolicy,
    GauntletRung,
    gauntlet_plan,
    load_gauntlet_policy,
    make_rung_config,
    run_gauntlet_once,
    run_gauntlet_until_blocked_or_complete,
    score_promotion_report,
    write_gauntlet_report,
)
from attention_lab.queue.ledger import QueueLedger
from attention_lab.queue.runner import CommandResult


def test_gauntlet_policy_loads(tmp_path):
    policy_path = _write_policy(tmp_path)

    policy = load_gauntlet_policy(policy_path)

    assert policy.experiment_id == "E003_qkv_architecture_gauntlet"
    assert policy.control_run_name == "standard_control"
    assert [rung.name for rung in policy.rungs] == ["rung020", "rung150"]
    assert policy.max_loss_ratio_vs_control == 1.2


def test_make_rung_config_overrides_budget_and_names(tiny_config, tmp_path):
    base = tiny_config(tmp_path, tmp_path / "data")
    base["run"]["name"] = "differential_candidate"
    base["model"].update(
        {
            "attention_type": "differential_qkv_anti_value",
            "diff_qkv_lambda_init": 0.5,
            "diff_qkv_lambda_trainable": True,
            "diff_qkv_share_value": False,
        }
    )
    rung = GauntletRung("rung020", max_steps=20, val_every=10, save_every=20, diagnostics_every=10)

    config = make_rung_config(base, rung, experiment_id="E003_qkv_architecture_gauntlet")

    assert config["run"]["name"] == "differential_candidate_rung020"
    assert config["run"]["out_dir"].endswith("differential_candidate_rung020")
    assert config["train"]["max_steps"] == 20
    assert config["train"]["val_every"] == 10
    assert config["train"]["save_every"] == 20
    assert config["diagnostics"]["attention_diagnostics_every"] == 10
    assert config["queue"]["screen_steps"] == 20


def test_score_promotion_report_advances_clean_candidate_and_blocks_bad_ratios():
    policy = _policy_object()
    report = _promotion_report("candidate_rung020", attention_type="differential_qkv_anti_value")
    control = _promotion_report("standard_control_rung020", attention_type="standard")

    decision = score_promotion_report(report, control, policy)

    assert decision["machine_decision"] == "advance"
    assert decision["loss_ratio_vs_control"] == 1.0

    report["final_val_loss"] = 20.0
    catastrophic = score_promotion_report(report, control, policy)

    assert catastrophic["machine_decision"] in {"killed", "needs_investigation"}
    assert "loss ratio" in catastrophic["decision_reason"]


def test_score_promotion_report_kills_inactive_mechanism():
    policy = _policy_object()
    report = _promotion_report("candidate_rung020", attention_type="scope_gated_qkv")
    report["mechanism_active"] = False
    report["diagnostics_non_degenerate"] = False

    decision = score_promotion_report(report, None, policy)

    assert decision["machine_decision"] in {"killed", "needs_investigation"}
    assert "mechanism diagnostics" in decision["decision_reason"]


def test_gauntlet_plan_reports_missing_prerequisites(tmp_path):
    policy_path = _write_policy(tmp_path, base_configs=["configs/missing.yaml"])
    policy = load_gauntlet_policy(policy_path)

    plan = gauntlet_plan(policy, repo_root=tmp_path)

    assert plan["ok"] is False
    assert plan["missing_prerequisites"]


def test_run_gauntlet_once_queues_first_rung(tiny_config, tmp_path):
    policy_path, ledger = _gauntlet_case(tmp_path, tiny_config, include_candidate=False)

    result = run_gauntlet_once(policy_path=policy_path, ledger=ledger, repo_root=tmp_path)

    assert result["action"] == "queued"
    assert result["run_name"] == "standard_control_rung020"
    assert (tmp_path / "configs" / "standard_control_rung020.yaml").exists()


def test_run_gauntlet_until_blocked_screens_and_writes_report(tiny_config, tmp_path):
    policy_path, ledger = _gauntlet_case(tmp_path, tiny_config)

    result = run_gauntlet_until_blocked_or_complete(
        policy_path=policy_path,
        ledger=ledger,
        repo_root=tmp_path,
        command_runner=_fake_screen_command,
        max_iterations=20,
    )

    report_path = tmp_path / "reports" / "experiments" / "E003_qkv_architecture_gauntlet" / "gauntlet_report.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))

    assert result["ok"] is True
    assert any(action["action"] == "screened" for action in result["actions"])
    assert any(decision["candidate"] == "differential_candidate_rung020" for decision in payload["decisions"])
    final = next(decision for decision in payload["decisions"] if decision["candidate"] == "differential_candidate_rung020")
    assert final["ready_for_manual_full_promotion"] is True
    assert final["next_action"] == "ready_for_manual_full_promotion"


def test_gauntlet_does_not_advance_after_killed_rung(tiny_config, tmp_path):
    policy_path, ledger = _gauntlet_case(
        tmp_path,
        tiny_config,
        include_control=False,
        rungs=("rung020", "rung150"),
    )

    result = run_gauntlet_until_blocked_or_complete(
        policy_path=policy_path,
        ledger=ledger,
        repo_root=tmp_path,
        command_runner=_fake_inactive_screen_command,
        max_iterations=10,
    )

    report_path = tmp_path / "reports" / "experiments" / "E003_qkv_architecture_gauntlet" / "gauntlet_report.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))

    assert result["ok"] is True
    assert payload["decisions"][0]["machine_decision"] == "killed"
    assert not (tmp_path / "configs" / "differential_candidate_rung150.yaml").exists()


def test_gauntlet_allow_full_uses_ledger_approve_full_run(tiny_config, tmp_path):
    policy_path, ledger = _gauntlet_case(tmp_path, tiny_config, include_control=False)
    candidate_path = tmp_path / "configs" / "differential_candidate.yaml"
    candidate_config = yaml.safe_load(candidate_path.read_text(encoding="utf-8"))
    ledger.enqueue_config(candidate_path, candidate_config, candidate_path.read_bytes(), stage="SCREEN")
    called = {}

    def approve(run_id, *, reason=None, repo_root="."):  # noqa: ARG001
        called["run_id"] = run_id

    ledger.approve_full_run = approve  # type: ignore[method-assign]

    result = run_gauntlet_until_blocked_or_complete(
        policy_path=policy_path,
        ledger=ledger,
        repo_root=tmp_path,
        command_runner=_fake_screen_command,
        allow_full=True,
        max_iterations=10,
    )

    assert result["ok"] is True
    assert called["run_id"] == ledger.get_run("differential_candidate")["id"]


def test_gauntlet_plan_cli_is_read_only(tiny_config, tmp_path, capsys):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    control = tiny_config(tmp_path, tmp_path / "data", max_steps=20)
    control["run"]["name"] = "standard_control"
    control["run"]["out_dir"] = str(tmp_path / "runs" / "standard_control")
    (config_dir / "standard_control.yaml").write_text(yaml.safe_dump(control), encoding="utf-8")
    policy_path = _write_policy(tmp_path, base_configs=["configs/standard_control.yaml"], rungs=("rung020",))

    queue_main(
        [
            "--root",
            str(tmp_path),
            "--db",
            str(tmp_path / "queue.db"),
            "gauntlet-plan",
            "--experiment",
            "E003_qkv_architecture_gauntlet",
            "--policy",
            str(policy_path),
        ]
    )

    assert "standard_control_rung020.yaml" in capsys.readouterr().out
    assert not (tmp_path / "queue.db").exists()


def test_gauntlet_report_cli_renders_existing_report(tmp_path, capsys):
    write_gauntlet_report(
        experiment_id="E003_qkv_architecture_gauntlet",
        policy_path="policy.yaml",
        control_run_name="standard_control",
        decisions=[{"candidate": "candidate_rung020", "rung": "rung020", "machine_decision": "advance"}],
        repo_root=tmp_path,
    )

    queue_main(
        [
            "--root",
            str(tmp_path),
            "gauntlet-report",
            "--experiment",
            "E003_qkv_architecture_gauntlet",
        ]
    )

    assert "candidate_rung020" in capsys.readouterr().out


def _policy_object() -> GauntletPolicy:
    return GauntletPolicy(
        experiment_id="E003_qkv_architecture_gauntlet",
        control_run_name="standard_control",
        base_configs=["configs/standard_control.yaml"],
        rungs=[GauntletRung("rung020", 20, 10, 20, 10)],
    )


def _promotion_report(run_name: str, *, attention_type: str) -> dict:
    return {
        "run_name": run_name,
        "attention_type": attention_type,
        "max_step_reached": 20,
        "expected_screen_steps": 20,
        "loss_descended": True,
        "nan_or_inf_seen": False,
        "checkpoint_present": True,
        "final_val_loss": 5.0,
        "median_tokens_per_sec": 100.0,
        "peak_vram_allocated_mb": 1000.0,
        "mechanism_active": None if attention_type == "standard" else True,
        "diagnostics_non_degenerate": False if attention_type == "standard" else True,
        "mechanism_check_name": None if attention_type == "standard" else "differential_qkv_activity",
        "promotion_recommendation": "promote",
        "promotion_blockers": [],
    }


def _gauntlet_case(
    tmp_path: Path,
    tiny_config,
    *,
    include_control: bool = True,
    include_candidate: bool = True,
    rungs: tuple[str, ...] = ("rung020",),
) -> tuple[Path, QueueLedger]:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    base_configs = []
    if include_control:
        control = tiny_config(tmp_path, tmp_path / "data", max_steps=20)
        control["run"]["name"] = "standard_control"
        control["run"]["out_dir"] = str(tmp_path / "runs" / "standard_control")
        control_path = config_dir / "standard_control.yaml"
        control_path.write_text(yaml.safe_dump(control), encoding="utf-8")
        base_configs.append(str(control_path.relative_to(tmp_path)))
    if include_candidate:
        candidate = tiny_config(tmp_path, tmp_path / "data", max_steps=20)
        candidate["run"]["name"] = "differential_candidate"
        candidate["run"]["out_dir"] = str(tmp_path / "runs" / "differential_candidate")
        candidate["model"].update(
            {
                "attention_type": "differential_qkv_anti_value",
                "diff_qkv_lambda_init": 0.5,
                "diff_qkv_lambda_trainable": True,
                "diff_qkv_share_value": False,
            }
        )
        candidate["diagnostics"] = {"attention_diagnostics_every": 10}
        candidate["queue"] = {"mechanism_check": "differential_qkv_activity"}
        candidate_path = config_dir / "differential_candidate.yaml"
        candidate_path.write_text(yaml.safe_dump(candidate), encoding="utf-8")
        base_configs.append(str(candidate_path.relative_to(tmp_path)))
    policy_path = _write_policy(tmp_path, base_configs=base_configs, rungs=rungs)
    ledger = QueueLedger(tmp_path / "queue.db")
    ledger.initialize()
    return policy_path, ledger


def _write_policy(
    tmp_path: Path,
    *,
    base_configs: list[str] | None = None,
    rungs: tuple[str, ...] = ("rung020", "rung150"),
) -> Path:
    rung_payload = [
        {"name": rung, "max_steps": 20 if rung == "rung020" else 150, "val_every": 10, "save_every": 20, "diagnostics_every": 10}
        for rung in rungs
    ]
    payload = {
        "experiment_id": "E003_qkv_architecture_gauntlet",
        "control_run_name": "standard_control",
        "base_configs": base_configs if base_configs is not None else ["configs/standard_control.yaml"],
        "rungs": rung_payload,
        "gates": {
            "require_loss_descended": True,
            "require_checkpoint": True,
            "require_mechanism_active": True,
            "max_loss_ratio_vs_control": 1.2,
            "min_speed_ratio_vs_control": 0.2,
            "max_vram_ratio_vs_control": 2.75,
            "require_no_nan_or_inf": True,
        },
    }
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return policy_path


def _fake_screen_command(cmd, log_path):  # noqa: ARG001
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
    if attention_type == "differential_qkv_anti_value":
        diag_path = screen_dir / "evals" / "attention_diagnostics.jsonl"
        diag_path.parent.mkdir(parents=True, exist_ok=True)
        diag_path.write_text(
            json.dumps(
                {
                    "attention_type": "differential_qkv_anti_value",
                    "step": val_every,
                    "pos_output_norm": 1e-3,
                    "neg_output_norm": 1e-3,
                    "branch_output_delta": 1e-3,
                    "diff_lambda": 0.5,
                }
            )
            + "\n",
            encoding="utf-8",
        )
    return CommandResult(returncode=0, stdout="ok", stderr="")


def _fake_inactive_screen_command(cmd, log_path):  # noqa: ARG001
    result = _fake_screen_command(cmd, log_path)
    screen_dir = Path(log_path).parent
    diag_path = screen_dir / "evals" / "attention_diagnostics.jsonl"
    if diag_path.exists():
        diag_path.write_text(
            json.dumps(
                {
                    "attention_type": "differential_qkv_anti_value",
                    "step": 10,
                    "pos_output_norm": 0.0,
                    "neg_output_norm": 0.0,
                    "branch_output_delta": 0.0,
                    "diff_lambda": 0.5,
                }
            )
            + "\n",
            encoding="utf-8",
        )
    return result
