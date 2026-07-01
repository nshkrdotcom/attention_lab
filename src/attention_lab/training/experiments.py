from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


EXPERIMENTS_MANIFEST = Path("docs/experiments/experiments.yaml")
EXPERIMENT_STATUSES = {"planned", "running", "completed", "deferred", "abandoned"}


class ExperimentError(ValueError):
    pass


def load_experiments_manifest(path: str | Path = EXPERIMENTS_MANIFEST) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        manifest = yaml.safe_load(f) or {}
    experiments = manifest.get("experiments")
    if not isinstance(experiments, list):
        raise ExperimentError(f"Experiment manifest must contain an experiments list: {path}")
    return manifest


def list_experiments(path: str | Path = EXPERIMENTS_MANIFEST) -> list[dict[str, Any]]:
    return load_experiments_manifest(path)["experiments"]


def get_experiment(experiment_id: str, path: str | Path = EXPERIMENTS_MANIFEST) -> dict[str, Any]:
    for experiment in list_experiments(path):
        if experiment.get("id") == experiment_id:
            return experiment
    raise ExperimentError(f"Unknown experiment id: {experiment_id}")


def validate_experiment_entry(experiment: dict[str, Any]) -> None:
    required = {
        "id",
        "status",
        "plan",
        "config_dir",
        "run_dir",
        "report_dir",
        "baseline_config",
        "baseline_reference_run",
        "accurate_baseline_alias",
        "dataset_manifest",
    }
    missing = sorted(required - set(experiment))
    if missing:
        raise ExperimentError(f"Experiment {experiment.get('id', '<unknown>')} missing fields: {missing}")
    if experiment["status"] not in EXPERIMENT_STATUSES:
        raise ExperimentError(
            f"Experiment {experiment['id']} has invalid status {experiment['status']!r}; "
            f"expected one of {sorted(EXPERIMENT_STATUSES)}"
        )


def format_experiment(experiment: dict[str, Any]) -> str:
    fields = [
        ("id", "experiment id"),
        ("status", "status"),
        ("plan", "plan path"),
        ("config_dir", "config dir"),
        ("run_dir", "run dir"),
        ("report_dir", "report dir"),
        ("baseline_config", "baseline config"),
        ("dataset_manifest", "dataset manifest"),
    ]
    return "\n".join(f"{label}: {experiment[key]}" for key, label in fields)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", default=None)
    parser.add_argument("--manifest", default=str(EXPERIMENTS_MANIFEST))
    args = parser.parse_args()
    if args.id:
        experiment = get_experiment(args.id, args.manifest)
        validate_experiment_entry(experiment)
        print(format_experiment(experiment))
        return
    for index, experiment in enumerate(list_experiments(args.manifest)):
        validate_experiment_entry(experiment)
        if index:
            print()
        print(format_experiment(experiment))
