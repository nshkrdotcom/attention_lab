#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from attention_lab.mechanisms.controls import is_seed_mismatched, resolve_control
from attention_lab.mechanisms.hypotheses import validate_hypothesis_doc
from attention_lab.mechanisms.presets import resolve_preset, site_presets_for_names
from attention_lab.mechanisms.suite import _preflight_report, _resolve_feature_pooling
from attention_lab.mechanisms.summary import validate_suite_artifacts
from attention_lab.mechanisms.task_schema import load_task_suite, validate_task_suite


TIER1_RUNS = (
    {
        "label": "E003",
        "experiment_id": "E003_qkv_architecture_gauntlet",
        "candidate": "differential",
        "task_file": Path("configs/mechanisms/tier1_tasks/E003_differential_negation_tier1.yaml"),
        "hypothesis_doc": Path("docs/mechanisms/hypotheses/E003_differential_negation_tier1.yaml"),
        "probe_only_output": Path("reports/mechanisms/probes/E003_differential_tier1_probe_only_inventory_path"),
        "confirmatory_output": Path("reports/mechanisms/probes/E003_differential_tier1_confirmatory_inventory_path"),
        "seed": "1",
    },
    {
        "label": "E004",
        "experiment_id": "E004_operator_binding_qkv_gauntlet",
        "candidate": "operator_valued",
        "task_file": Path("configs/mechanisms/tier1_tasks/E004_operator_valued_negation_tier1.yaml"),
        "hypothesis_doc": Path("docs/mechanisms/hypotheses/E004_operator_valued_negation_tier1.yaml"),
        "probe_only_output": Path("reports/mechanisms/probes/E004_operator_valued_tier1_probe_only_inventory_path"),
        "confirmatory_output": Path("reports/mechanisms/probes/E004_operator_valued_tier1_confirmatory_inventory_path"),
        "seed": "2",
    },
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preflight and optionally run Tier-1 E003/E004 mechanism probe suites.")
    parser.add_argument("--preflight-only", action="store_true", help="Only validate inputs and checkpoint availability.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--fdr-alpha", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    all_available = True
    print("Tier-1 mechanism probe checkpoint table")
    for row in TIER1_RUNS:
        report = _build_preflight(row)
        candidate_ok = report["candidate_checkpoint_exists"]
        control_ok = report["control_available"]
        all_available = all_available and bool(candidate_ok and control_ok)
        print(
            f"{row['label']}: candidate={report['candidate_checkpoint_path']} "
            f"exists={candidate_ok}; control={report['actual_control_checkpoint_path']} exists={control_ok}"
        )
        if not report["task_suite_validation"]["valid"]:
            raise SystemExit(f"{row['label']} preflight task validation failed: {report['task_suite_validation']['errors']}")
        if not report["hypothesis_doc_validation"]["valid"]:
            raise SystemExit(f"{row['label']} hypothesis validation failed")

    if args.preflight_only:
        print("preflight complete; execution skipped by --preflight-only")
        return
    if not all_available:
        print("checkpoint unavailable, execution skipped; no fake artifacts were created")
        raise SystemExit(2)

    for row in TIER1_RUNS:
        preset = resolve_preset(row["experiment_id"], row["candidate"])
        checkpoint = preset.expected_checkpoint_path
        _run_suite(row, checkpoint=checkpoint, exploratory=True, probe_only=True, output_dir=row["probe_only_output"], args=args)
        _run_suite(row, checkpoint=checkpoint, exploratory=False, probe_only=False, output_dir=row["confirmatory_output"], args=args)
        for output_dir in (row["probe_only_output"], row["confirmatory_output"]):
            errors = validate_suite_artifacts(output_dir)
            if errors:
                raise SystemExit(f"{output_dir} artifact validation failed: {'; '.join(errors)}")
            print(f"validated artifacts: {output_dir}")


def _build_preflight(row: dict[str, object]) -> dict[str, object]:
    preset = resolve_preset(str(row["experiment_id"]), str(row["candidate"]))
    validate_hypothesis_doc(row["hypothesis_doc"])
    suite = load_task_suite(row["task_file"])
    task_validation = validate_task_suite(
        suite,
        confirmatory=True,
        exploratory=False,
        min_n=50,
        require_decoys=True,
        require_restoration_tokens=True,
    )
    control = resolve_control(
        preset,
        control_mode="matched",
        control_config=None,
        control_checkpoint=None,
    )
    selected_sites = site_presets_for_names(preset, None, exploratory=False)
    pooling = _resolve_feature_pooling("auto", exploratory=False)
    return _preflight_report(
        preset=preset,
        candidate_config=preset.config_path,
        candidate_checkpoint=preset.expected_checkpoint_path,
        control=control,
        control_seed_mismatched=is_seed_mismatched(preset, control),
        task_validation=task_validation,
        hypothesis_valid=True,
        selected_sites=selected_sites,
        feature_pooling=pooling,
        task_aligned_pooling=True,
        exploratory=False,
        probe_only=False,
        allow_diagnostic_with_missing_control=False,
    )


def _run_suite(
    row: dict[str, object],
    *,
    checkpoint: Path,
    exploratory: bool,
    probe_only: bool,
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    command = [
        sys.executable,
        "scripts/run_mechanism_probe_suite.py",
        "--experiment-id",
        str(row["experiment_id"]),
        "--candidate",
        str(row["candidate"]),
        "--checkpoint",
        str(checkpoint),
        "--task-file",
        str(row["task_file"]),
        "--output-dir",
        str(output_dir),
        "--control-mode",
        "matched",
        "--min-n",
        "50",
        "--bootstrap-samples",
        str(args.bootstrap_samples),
        "--fdr-alpha",
        str(args.fdr_alpha),
        "--seed",
        str(row["seed"]),
        "--device",
        args.device,
        "--batch-size",
        str(args.batch_size),
    ]
    if exploratory:
        command.append("--exploratory")
    if probe_only:
        command.append("--probe-only")
    if not exploratory:
        command.extend(["--hypothesis-doc", str(row["hypothesis_doc"])])
    result = subprocess.run(command, cwd=Path.cwd(), check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(result.stderr or result.stdout)
    print(result.stdout.strip())


if __name__ == "__main__":
    main()
