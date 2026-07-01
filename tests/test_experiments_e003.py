from __future__ import annotations

import os
import subprocess
from pathlib import Path

from attention_lab.training.experiments import get_experiment, list_experiments


def test_e003_is_registered_with_existing_paths():
    ids = [experiment["id"] for experiment in list_experiments()]
    assert "E003_qkv_architecture_gauntlet" in ids

    experiment = get_experiment("E003_qkv_architecture_gauntlet")
    for key in ("plan", "config_dir", "report_dir", "baseline_config", "dataset_manifest"):
        assert Path(experiment[key]).exists(), key
    assert experiment["policy_configs"] == ["gauntlet_policy.yaml"]


def test_e003_scripts_are_executable_and_screen_first(repo_root):
    script_dir = repo_root / "scripts" / "experiments" / "E003_qkv_architecture_gauntlet"
    for name in ("run_gauntlet.sh", "compare_gauntlet_runs.sh", "summarize_gauntlet.py"):
        path = script_dir / name
        assert path.exists(), name
        assert os.access(path, os.X_OK), name

    run_script = (script_dir / "run_gauntlet.sh").read_text(encoding="utf-8")
    assert "scripts/verify_cuda.py" in run_script
    assert "scripts/verify_data.py" in run_script
    assert "gauntlet-plan" in run_script
    assert "gauntlet-run" in run_script
    assert "--allow-full" in run_script


def test_e003_compare_script_requires_report_or_full_artifacts(repo_root):
    script = repo_root / "scripts" / "experiments" / "E003_qkv_architecture_gauntlet" / "compare_gauntlet_runs.sh"
    result = subprocess.run(
        [str(script)],
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        assert "Missing gauntlet screen report" in result.stderr
    else:
        assert "wrote:" in result.stdout
