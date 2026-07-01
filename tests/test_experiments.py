from __future__ import annotations

import os
from pathlib import Path

from attention_lab.training.experiments import EXPERIMENT_STATUSES, get_experiment, list_experiments, main


def test_experiment_manifest_loads_e001():
    experiments = list_experiments()
    ids = [experiment["id"] for experiment in experiments]
    assert "E001_cp_trilinear_attention" in ids


def test_e001_paths_are_committed_and_status_is_valid():
    experiment = get_experiment("E001_cp_trilinear_attention")
    assert experiment["status"] in EXPERIMENT_STATUSES
    for key in ("plan", "config_dir", "report_dir", "baseline_config", "accurate_baseline_alias", "dataset_manifest"):
        assert Path(experiment[key]).exists(), key


def test_list_experiments_prints_e001(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["list_experiments.py", "--id", "E001_cp_trilinear_attention"])
    main()
    output = capsys.readouterr().out
    assert "experiment id: E001_cp_trilinear_attention" in output
    assert "dataset manifest: data/fineweb_edu_100m/manifest.json" in output


def test_e001_manual_full_run_scripts_are_executable(repo_root):
    script_dir = repo_root / "scripts" / "experiments" / "E001_cp_trilinear_attention"
    script_names = [
        "run_full_standard_30m.sh",
        "run_full_cp_bilinear_r8_30m.sh",
        "run_full_cp_trilinear_r8_30m.sh",
        "run_full_cp_trilinear_r8_lambda0_30m.sh",
        "run_all_full.sh",
        "compare_full_runs.sh",
    ]
    for script_name in script_names:
        script_path = script_dir / script_name
        assert script_path.exists(), script_name
        assert os.access(script_path, os.X_OK), script_name

    full_script = (script_dir / "run_full_cp_trilinear_r8_30m.sh").read_text(encoding="utf-8")
    for command in (
        "scripts/verify_data.py",
        "scripts/train.py",
        "scripts/verify_run.py",
        "scripts/eval_loss.py",
        "scripts/eval_generate.py",
        "scripts/eval_hellaswag.py",
        "scripts/summarize_run.py",
    ):
        assert command in full_script
