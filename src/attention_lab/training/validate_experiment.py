from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from attention_lab.training.config import EXPERIMENTAL_UNIMPLEMENTED_STATUS, load_config
from attention_lab.training.experiments import get_experiment, validate_experiment_entry


class ExperimentValidationError(ValueError):
    pass


FIXED_SECTION_KEYS = {
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


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        value = yaml.safe_load(f)
    if not isinstance(value, dict):
        raise ExperimentValidationError(f"Config is not a mapping: {path}")
    return value


def _relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _fixed_values(config: dict[str, Any]) -> dict[str, Any]:
    values = {}
    for section, keys in FIXED_SECTION_KEYS.items():
        for key in keys:
            values[f"{section}.{key}"] = config.get(section, {}).get(key)
    return values


def _manifest_config_names(experiment: dict[str, Any], key: str) -> list[str]:
    values = experiment.get(key, [])
    if values is None:
        return []
    if not isinstance(values, list) or not all(isinstance(value, str) and value.endswith(".yaml") for value in values):
        raise ExperimentValidationError(f"Experiment {experiment['id']} field {key} must be a list of YAML filenames")
    return values


def validate_experiment(experiment_id: str) -> dict[str, Any]:
    experiment = get_experiment(experiment_id)
    validate_experiment_entry(experiment)

    plan_path = Path(experiment["plan"])
    config_dir = Path(experiment["config_dir"])
    report_dir = Path(experiment["report_dir"])
    run_dir = Path(experiment["run_dir"])
    dataset_manifest = Path(experiment["dataset_manifest"])
    for path, label in (
        (plan_path, "plan path"),
        (config_dir, "config dir"),
        (report_dir, "report dir"),
        (dataset_manifest, "dataset manifest"),
    ):
        if not path.exists():
            raise ExperimentValidationError(f"Missing {label}: {path}")

    config_paths = sorted(config_dir.glob("*.yaml"))
    if not config_paths:
        raise ExperimentValidationError(f"No YAML configs found in {config_dir}")

    canonical_first_build_configs = _manifest_config_names(experiment, "canonical_first_build_configs")
    legacy_or_auxiliary_runnable_configs = _manifest_config_names(
        experiment,
        "legacy_or_auxiliary_runnable_configs",
    )
    overlap = sorted(set(canonical_first_build_configs) & set(legacy_or_auxiliary_runnable_configs))
    if overlap:
        raise ExperimentValidationError(f"Configs cannot be both canonical first-build and legacy: {overlap}")
    config_names = {path.name for path in config_paths}
    missing_canonical = sorted(set(canonical_first_build_configs) - config_names)
    if missing_canonical:
        raise ExperimentValidationError(f"Missing canonical first-build configs: {missing_canonical}")
    missing_legacy = sorted(set(legacy_or_auxiliary_runnable_configs) - config_names)
    if missing_legacy:
        raise ExperimentValidationError(f"Missing legacy/auxiliary runnable configs: {missing_legacy}")

    out_dirs = []
    runnable = []
    unimplemented = []
    runnable_names = []
    fixed_reference = None
    for config_path in config_paths:
        config = load_yaml(config_path)
        out_dir = Path(config["run"]["out_dir"])
        out_dirs.append(out_dir)
        if not _relative_to(out_dir, run_dir):
            raise ExperimentValidationError(f"{config_path} run.out_dir is outside experiment run dir: {out_dir}")
        fixed_values = _fixed_values(config)
        if fixed_reference is None:
            fixed_reference = fixed_values
        elif fixed_values != fixed_reference:
            raise ExperimentValidationError(f"{config_path} does not share the fixed baseline fields")

        if config.get("status") == EXPERIMENTAL_UNIMPLEMENTED_STATUS:
            try:
                load_config(config_path)
            except ValueError as exc:
                if "experimental" not in str(exc):
                    raise ExperimentValidationError(f"{config_path} did not fail as experimental") from exc
            else:
                raise ExperimentValidationError(f"{config_path} unexpectedly loaded as runnable")
            unimplemented.append(str(config_path))
        else:
            load_config(config_path)
            runnable.append(str(config_path))
            runnable_names.append(config_path.name)

    if len(out_dirs) != len(set(out_dirs)):
        raise ExperimentValidationError("Experiment configs must have unique run.out_dir values")
    non_runnable_canonical = sorted(set(canonical_first_build_configs) - set(runnable_names))
    if non_runnable_canonical:
        raise ExperimentValidationError(f"Canonical first-build configs must be runnable: {non_runnable_canonical}")
    non_runnable_legacy = sorted(set(legacy_or_auxiliary_runnable_configs) - set(runnable_names))
    if non_runnable_legacy:
        raise ExperimentValidationError(f"Legacy/auxiliary configs must be runnable: {non_runnable_legacy}")
    unclassified_runnable = sorted(set(runnable_names) - set(canonical_first_build_configs) - set(legacy_or_auxiliary_runnable_configs))
    if canonical_first_build_configs and unclassified_runnable:
        raise ExperimentValidationError(
            "Runnable configs must be classified as canonical first-build or legacy/auxiliary: "
            f"{unclassified_runnable}"
        )

    baseline_reference = Path(experiment["baseline_reference_run"])
    baseline_summary = baseline_reference / "evals" / "run_summary.json"
    baseline_summary_checked = False
    if baseline_reference.exists():
        if not baseline_summary.is_file():
            raise ExperimentValidationError(f"Historical baseline run exists but summary is missing: {baseline_summary}")
        baseline_summary_checked = True

    return {
        "id": experiment_id,
        "config_count": len(config_paths),
        "runnable_config_count": len(runnable),
        "unimplemented_config_count": len(unimplemented),
        "canonical_first_build_config_count": len(canonical_first_build_configs),
        "legacy_or_auxiliary_runnable_config_count": len(legacy_or_auxiliary_runnable_configs),
        "canonical_first_build_configs": canonical_first_build_configs,
        "legacy_or_auxiliary_runnable_configs": legacy_or_auxiliary_runnable_configs,
        "baseline_summary_checked": baseline_summary_checked,
        "ok": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", required=True)
    args = parser.parse_args()
    result = validate_experiment(args.id)
    for key, value in result.items():
        print(f"{key}: {value}")
