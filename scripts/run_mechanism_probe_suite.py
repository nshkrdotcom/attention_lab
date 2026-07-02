#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any

from attention_lab.mechanisms.activations import (
    capture_feature_matrices,
    load_mechanism_model,
    probe_dataset_from_matrix,
)
from attention_lab.mechanisms.alignment import probe_direction_alignment
from attention_lab.mechanisms.claim_gates import evaluate_claim_gate
from attention_lab.mechanisms.controls import choose_random_site_null, resolve_control
from attention_lab.mechanisms.hypotheses import load_hypothesis_doc
from attention_lab.mechanisms.linear_probe import run_linear_probe_with_nulls
from attention_lab.mechanisms.presets import MechanismPreset, SitePreset, get_preset
from attention_lab.mechanisms.statistics import (
    bootstrap_auc_difference,
    ci_excludes_zero,
    fdr_bh,
    target_vs_decoy_specificity,
)
from attention_lab.mechanisms.summary import write_suite_artifacts
from attention_lab.mechanisms.task_schema import TaskSuite, load_task_suite


CONFIRMATORY_PAIR_FLOOR = 50


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        run_suite(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Tier-1 statistical mechanism probe suite.")
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--config", help="Candidate config. Defaults to the Tier-1 preset config.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--hypothesis-doc")
    parser.add_argument("--exploratory", action="store_true")
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--sites", help="Comma-separated candidate site bases. Defaults to preset Tier-1 sites.")
    parser.add_argument("--control-mode", choices=("matched", "none"), default="matched")
    parser.add_argument("--control-checkpoint")
    parser.add_argument("--control-config")
    parser.add_argument("--force-noncanonical-control", action="store_true")
    parser.add_argument("--min-n", type=int, default=CONFIRMATORY_PAIR_FLOOR)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--fdr-alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--training-steps", type=int, default=200)
    return parser


def run_suite(args: argparse.Namespace) -> dict[str, Any]:
    if args.bootstrap_samples <= 0:
        raise ValueError("--bootstrap-samples must be positive")
    if args.min_n <= 0:
        raise ValueError("--min-n must be positive")
    if not args.exploratory and args.hypothesis_doc is None:
        raise ValueError("--hypothesis-doc is required unless --exploratory is set")

    preset = get_preset(args.experiment_id, args.candidate)
    if not preset.executable:
        raise ValueError(f"preset {preset.candidate} has status {preset.status}; Tier-2/Tier-3 presets are not executable")
    hypothesis = load_hypothesis_doc(args.hypothesis_doc) if args.hypothesis_doc else None
    suite = load_task_suite(args.task_file)
    requested_sites = _resolve_sites(preset, args.sites, args.layer)
    capture_site_bases = _capture_site_bases(preset, requested_sites)

    control = resolve_control(
        preset,
        control_checkpoint=args.control_checkpoint,
        control_config=args.control_config,
        force_noncanonical=args.force_noncanonical_control,
    )

    candidate_config = Path(args.config) if args.config else preset.config
    if candidate_config is None:
        raise ValueError("candidate config is required")
    candidate_loaded = load_mechanism_model(
        config_path=candidate_config,
        checkpoint_path=args.checkpoint,
        device=args.device,
    )
    examples = suite.examples()
    candidate_matrices = capture_feature_matrices(
        candidate_loaded,
        examples,
        sites=capture_site_bases,
        layer=args.layer,
        batch_size=args.batch_size,
        device=args.device,
    )

    control_matrices = {}
    control_available = False
    if args.control_mode == "matched" and control.control_checkpoint and control.control_config:
        if Path(control.control_checkpoint).exists() and Path(control.control_config).exists():
            control_loaded = load_mechanism_model(
                config_path=control.control_config,
                checkpoint_path=control.control_checkpoint,
                device=args.device,
            )
            control_site_bases = sorted(
                {control_site for site in requested_sites for control_site in _site_preset(preset, site).control_sites}
            )
            if control_site_bases:
                control_matrices = capture_feature_matrices(
                    control_loaded,
                    examples,
                    sites=control_site_bases,
                    layer=args.layer,
                    batch_size=args.batch_size,
                    device=args.device,
                )
                control_available = bool(control_matrices)

    probe_metrics, gate_inputs = _compute_probe_metrics(
        preset=preset,
        suite=suite,
        candidate_matrices=candidate_matrices,
        control_matrices=control_matrices,
        requested_sites=requested_sites,
        min_n=args.min_n,
        seed=args.seed,
        bootstrap_samples=args.bootstrap_samples,
        fdr_alpha=args.fdr_alpha,
        training_steps=args.training_steps,
    )

    floor_passed, floor_reasons = suite.validate_confirmatory_floor(min_pairs_per_family=CONFIRMATORY_PAIR_FLOOR)
    min_n_below_floor = args.min_n < CONFIRMATORY_PAIR_FLOOR and not args.exploratory
    patching_restoration = (
        {"skipped": True, "reason": "probe-only mode"}
        if args.probe_only
        else {
            "skipped": False,
            "valid": False,
            "reason": (
                "full restoration requires task metadata with answer_token_id and foil_token_id for logitdiff; "
                "no fake restoration metrics were emitted"
            ),
        }
    )

    gate_metrics = {
        **gate_inputs,
        "exploratory": bool(args.exploratory),
        "probe_only": bool(args.probe_only),
        "hypothesis_doc_valid": hypothesis is not None,
        "matched_control_available": bool(control_available),
        "control_canonical": bool(control.is_canonical),
        "confirmatory_floor_passed": bool(floor_passed and not min_n_below_floor),
        "min_n_below_floor": bool(min_n_below_floor),
        "patching_valid": bool(patching_restoration.get("valid", False)),
        "restoration_valid": bool(patching_restoration.get("valid", False)),
        "mediation_fraction_valid": bool(patching_restoration.get("valid", False)),
        "raw_delta_only": False,
    }
    gate = evaluate_claim_gate(gate_metrics)

    metrics = {
        "schema_version": 1,
        "experiment_id": args.experiment_id,
        "candidate": args.candidate,
        "candidate_preset": preset.candidate,
        "checkpoint": str(args.checkpoint),
        "config": str(candidate_config),
        "canonical_control_checkpoint": str(control.canonical_control_checkpoint) if control.canonical_control_checkpoint else None,
        "actual_control_checkpoint": str(control.control_checkpoint) if control.control_checkpoint else None,
        "canonical_control_config": str(control.canonical_control_config) if control.canonical_control_config else None,
        "actual_control_config": str(control.control_config) if control.control_config else None,
        "control_is_override": control.is_override,
        "control_is_canonical": control.is_canonical,
        "control_noncanonical_reason": control.noncanonical_reason,
        "control_available": control_available,
        "task_file": str(args.task_file),
        "task_suite_provenance": {"deterministic": suite.deterministic_provenance, **suite.metadata},
        "pair_counts_per_family": suite.pair_counts_per_family,
        "confirmatory_floor_reasons": floor_reasons,
        "missing_decoys": not suite.has_decoys,
        "hypothesis_doc": str(hypothesis.path) if hypothesis else None,
        "hypothesis": hypothesis.payload if hypothesis else None,
        "exploratory": bool(args.exploratory),
        "probe_only": bool(args.probe_only),
        "sites_evaluated": requested_sites,
        "min_n": args.min_n,
        "bootstrap_samples": args.bootstrap_samples,
        "fdr_alpha": args.fdr_alpha,
        "seed": args.seed,
        "probe_metrics": probe_metrics["by_site"],
        "random_site_null": probe_metrics["random_site_null_summary"],
        "matched_control_metrics": probe_metrics["matched_control_summary"],
        "target_vs_decoy_specificity": probe_metrics["specificity_summary"],
        "bootstrap_ci_results": probe_metrics["bootstrap_summary"],
        "fdr_bh": probe_metrics["fdr_bh"],
        "fdr_scope": "every computed site x layer x task_family x metric cell in the run",
        "patching_restoration": patching_restoration,
        "alignment_to_control": probe_metrics["alignment_summary"],
        "claim_gate_inputs": gate_metrics,
        "limitations": ["single-seed", "not replicated"],
    }
    write_suite_artifacts(args.output_dir, metrics, gate)
    print(f"wrote {args.output_dir}")
    return metrics


def _resolve_sites(preset: MechanismPreset, sites_arg: str | None, layer: int) -> list[str]:
    if sites_arg:
        bases = [site.strip() for site in sites_arg.split(",") if site.strip()]
    else:
        bases = [site.site for site in preset.sites]
    if not bases:
        raise ValueError("no executable sites are declared for this preset")
    return [f"{base}[{layer}]" for base in bases]


def _capture_site_bases(preset: MechanismPreset, requested_site_keys: list[str]) -> list[str]:
    requested_bases = {site.split("[", 1)[0] for site in requested_site_keys}
    preset_bases = {site.site for site in preset.sites}
    return sorted(requested_bases | preset_bases)


def _site_preset(preset: MechanismPreset, site_key: str) -> SitePreset:
    base = site_key.split("[", 1)[0]
    for site in preset.sites:
        if site.site == base:
            return site
    raise ValueError(f"site {base!r} is not declared for preset {preset.candidate}")


def _compute_probe_metrics(
    *,
    preset: MechanismPreset,
    suite: TaskSuite,
    candidate_matrices: dict[str, Any],
    control_matrices: dict[str, Any],
    requested_sites: list[str],
    min_n: int,
    seed: int,
    bootstrap_samples: int,
    fdr_alpha: float,
    training_steps: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    examples = suite.examples()
    by_site: dict[str, dict[str, Any]] = {}
    test_cells: list[dict[str, Any]] = []
    random_nulls = []
    controls = []
    alignments = []
    specificities = []
    bootstraps = []
    grouped_ok = True
    random_all_available = True
    control_any_available = False
    primary_ci_all = True
    specificity_all = True

    for site_key in requested_sites:
        if site_key not in candidate_matrices:
            by_site[site_key] = {"unavailable": True, "reason": "site was not emitted by checkpoint forward pass"}
            random_all_available = False
            primary_ci_all = False
            specificity_all = False
            continue
        matrix = candidate_matrices[site_key]
        by_site[site_key] = {}
        random_selection = choose_random_site_null(
            candidate_site=site_key,
            candidate=matrix,
            available=candidate_matrices,
            seed=seed,
        )
        random_nulls.append(asdict(random_selection))
        random_all_available = random_all_available and random_selection.available
        site_preset = _site_preset(preset, site_key)
        control_site_key = None
        for control_base in site_preset.control_sites:
            candidate_key = f"{control_base}[{site_preset.layer}]"
            if candidate_key in control_matrices:
                control_site_key = candidate_key
                break
        for family_id in suite.pair_counts_per_family:
            dataset = probe_dataset_from_matrix(matrix, examples, family_id=family_id)
            result = run_linear_probe_with_nulls(
                dataset,
                seed=seed,
                min_n=min_n,
                training_steps=training_steps,
            )
            grouped_ok = grouped_ok and not result.grouped_split.pair_group_leakage
            family_metrics: dict[str, Any] = {
                "linear_probe_auc": result.primary.auc,
                "shuffled_label_auc": result.shuffled.auc,
                "auc_minus_shuffled_auc": result.auc_minus_shuffled_auc,
                "grouped_split": {
                    "group_field": result.grouped_split.group_field,
                    "pair_group_leakage": result.grouped_split.pair_group_leakage,
                    "template_group_leakage": result.grouped_split.template_group_leakage,
                },
            }
            by_site[site_key][family_id] = family_metrics
            shuffled_ci = bootstrap_auc_difference(
                result.primary.labels,
                result.primary.scores,
                result.shuffled.scores,
                seed=seed,
                samples=bootstrap_samples,
            )
            primary_ci_all = primary_ci_all and shuffled_ci.valid and ci_excludes_zero(shuffled_ci)
            cell = _cell(site_key, family_id, "auc_minus_shuffled_auc", shuffled_ci)
            test_cells.append(cell)
            bootstraps.append(cell)

            if random_selection.available and random_selection.selected_site:
                random_dataset = probe_dataset_from_matrix(
                    candidate_matrices[random_selection.selected_site],
                    examples,
                    family_id=family_id,
                )
                random_result = run_linear_probe_with_nulls(
                    random_dataset,
                    seed=seed,
                    min_n=min_n,
                    training_steps=training_steps,
                )
                random_ci = bootstrap_auc_difference(
                    result.primary.labels,
                    result.primary.scores,
                    random_result.primary.scores,
                    seed=seed + 1,
                    samples=bootstrap_samples,
                )
                family_metrics["random_site_auc"] = random_result.primary.auc
                family_metrics["auc_minus_random_site_auc"] = (
                    None if random_result.primary.auc is None else result.primary.auc - random_result.primary.auc
                )
                cell = _cell(site_key, family_id, "auc_minus_random_site_auc", random_ci)
                test_cells.append(cell)
                bootstraps.append(cell)

            control_result = None
            if control_site_key:
                control_dataset = probe_dataset_from_matrix(control_matrices[control_site_key], examples, family_id=family_id)
                control_result = run_linear_probe_with_nulls(
                    control_dataset,
                    seed=seed,
                    min_n=min_n,
                    training_steps=training_steps,
                )
                control_any_available = True
                control_ci = bootstrap_auc_difference(
                    result.primary.labels,
                    result.primary.scores,
                    control_result.primary.scores,
                    seed=seed + 2,
                    samples=bootstrap_samples,
                )
                family_metrics["matched_control_site"] = control_site_key
                family_metrics["matched_control_auc"] = control_result.primary.auc
                family_metrics["auc_minus_matched_control_auc"] = (
                    None if control_result.primary.auc is None else result.primary.auc - control_result.primary.auc
                )
                cell = _cell(site_key, family_id, "auc_minus_matched_control_auc", control_ci)
                test_cells.append(cell)
                bootstraps.append(cell)
                alignment = probe_direction_alignment(result.primary.weights, control_result.primary.weights)
                alignment_dict = asdict(alignment)
                alignment_dict.update({"site": site_key, "family_id": family_id, "control_site": control_site_key})
                alignments.append(alignment_dict)
            else:
                controls.append({"site": site_key, "available": False, "reason": "no compatible preset control site available"})

            test_dataset = _subset_dataset(dataset, result.primary.test_indices)
            specificity = target_vs_decoy_specificity(
                test_dataset,
                result.primary.scores,
                seed=seed + 3,
                samples=bootstrap_samples,
            )
            specificity_all = specificity_all and specificity.valid and ci_excludes_zero(specificity)
            cell = _cell(site_key, family_id, "target_vs_decoy_specificity", specificity)
            test_cells.append(cell)
            specificities.append(cell)

    corrected = fdr_bh(test_cells, alpha=fdr_alpha)
    primary_cells = [cell for cell in corrected if cell["metric"] == "auc_minus_shuffled_auc"]
    specificity_cells = [cell for cell in corrected if cell["metric"] == "target_vs_decoy_specificity"]
    fdr_primary_passed = bool(primary_cells) and all(cell["rejected"] for cell in primary_cells)
    fdr_specificity_passed = bool(specificity_cells) and all(cell["rejected"] for cell in specificity_cells)
    gate_inputs = {
        "has_real_probe_metrics": bool(by_site),
        "minimum_n_passed": all(count >= min_n for count in suite.pair_counts_per_family.values()),
        "grouped_split_passed": grouped_ok,
        "random_site_null_available": random_all_available,
        "stats_valid": bool(corrected),
        "fdr_primary_passed": fdr_primary_passed,
        "bootstrap_primary_ci_excludes_null": primary_ci_all,
        "decoy_specificity_passed": specificity_all and fdr_specificity_passed,
    }
    return (
        {
            "by_site": by_site,
            "random_site_null_summary": {
                "available": random_all_available,
                "results": random_nulls,
                "reason": None if random_all_available else "one or more sites lacked a matched-dimensional random-site null",
            },
            "matched_control_summary": {"available": control_any_available, "results": controls},
            "specificity_summary": specificities,
            "bootstrap_summary": bootstraps,
            "fdr_bh": corrected,
            "alignment_summary": alignments[0] if alignments else {"available": False, "reason": "matched control probe unavailable"},
        },
        gate_inputs,
    )


def _cell(site: str, family_id: str, metric: str, result) -> dict[str, Any]:
    return {
        "cell_id": f"{site}|layer0|{family_id}|{metric}",
        "site": site,
        "layer": 0,
        "task_family": family_id,
        "metric": metric,
        "estimate": result.estimate,
        "ci_low": result.low,
        "ci_high": result.high,
        "p_value": result.p_value,
        "valid": result.valid,
        "reason": result.reason,
        "ci_excludes_zero_positive": ci_excludes_zero(result) if result.valid else False,
    }


def _subset_dataset(dataset, indices):
    from attention_lab.mechanisms.linear_probe import LinearProbeDataset

    return LinearProbeDataset(
        X=dataset.X[indices],
        y=dataset.y[indices],
        pair_ids=dataset.pair_ids[indices],
        template_ids=dataset.template_ids[indices],
        family_ids=dataset.family_ids[indices],
        variants=dataset.variants[indices],
    )


if __name__ == "__main__":
    main()
