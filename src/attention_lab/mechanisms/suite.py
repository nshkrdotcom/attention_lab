from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from attention_lab.mechanisms.activations import collect_activation_features, encode_texts, load_mechanism_model
from attention_lab.mechanisms.alignment import probe_direction_alignment
from attention_lab.mechanisms.claim_gates import (
    CellGateInputs,
    evaluate_cell_claim_gate,
    overall_status,
)
from attention_lab.mechanisms.controls import (
    ControlResolution,
    is_seed_mismatched,
    resolve_control,
    select_random_site_null,
)
from attention_lab.mechanisms.capture import capture_activations
from attention_lab.mechanisms.hook_sites import is_discrete_hook_site
from attention_lab.mechanisms.hypotheses import exploratory_hypothesis_label, validate_hypothesis_doc
from attention_lab.mechanisms.interventions import run_with_interventions
from attention_lab.mechanisms.linear_probe import (
    GroupedSplit,
    LinearProbeResult,
    grouped_train_test_split,
    score_with_probe,
    train_linear_probe,
    train_shuffled_label_null,
)
from attention_lab.mechanisms.patching import make_cache_patch, mediation_fraction, restoration_score
from attention_lab.mechanisms.presets import MechanismProbePreset, SitePreset, resolve_preset, site_presets_for_names
from attention_lab.mechanisms.statistics import BootstrapResult, auc_score, bootstrap_metric, ci_excludes_null, fdr_bh
from attention_lab.mechanisms.summary import write_suite_artifacts
from attention_lab.mechanisms.task_schema import (
    TaskExample,
    TaskRecord,
    examples_for_probe,
    load_task_suite,
    validate_task_suite,
)


def run_probe_suite(
    *,
    experiment_id: str,
    candidate: str,
    checkpoint: str | Path,
    task_file: str | Path,
    output_dir: str | Path,
    hypothesis_doc: str | Path | None,
    exploratory: bool,
    probe_only: bool,
    sites: list[str] | None,
    control_mode: str,
    control_checkpoint: str | Path | None,
    control_config: str | Path | None,
    min_n: int,
    bootstrap_samples: int,
    fdr_alpha: float,
    seed: int,
    config: str | Path | None = None,
    device: str = "cpu",
    batch_size: int = 16,
    force_noncanonical_control: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    preset = resolve_preset(experiment_id, candidate)
    if not preset.executable:
        metrics = _stub_metrics(preset, checkpoint, task_file, exploratory, probe_only)
        claim_gates = {
            "overall_status": "insufficient_evidence",
            "preset_status": preset.status,
            "reason": preset.notes,
            "cells": {},
        }
        write_suite_artifacts(output_dir, metrics, claim_gates)
        return metrics, claim_gates

    if not exploratory and hypothesis_doc is None:
        raise ValueError("confirmatory mechanism probe suite requires --hypothesis-doc; use --exploratory otherwise")
    if exploratory and probe_only is False:
        # Full exploratory runs are allowed, but still capped later.
        pass

    hypothesis_payload: dict[str, Any]
    hypothesis_valid = False
    if hypothesis_doc is not None:
        hypothesis = validate_hypothesis_doc(hypothesis_doc)
        hypothesis_payload = {"path": str(hypothesis.path), "fields": hypothesis.fields}
        hypothesis_valid = True
    else:
        hypothesis_payload = exploratory_hypothesis_label()

    suite = load_task_suite(task_file)
    task_validation = validate_task_suite(
        suite,
        confirmatory=not exploratory,
        exploratory=exploratory,
        min_n=min_n,
        require_decoys=not exploratory,
    )
    control = resolve_control(
        preset,
        control_mode=control_mode,
        control_config=control_config,
        control_checkpoint=control_checkpoint,
        force_noncanonical=force_noncanonical_control,
    )
    control_seed_mismatched = is_seed_mismatched(preset, control)
    selected_sites = site_presets_for_names(preset, sites)

    candidate_config = Path(config) if config is not None else preset.config_path
    candidate_checkpoint = Path(checkpoint)
    if not candidate_config.exists():
        raise ValueError(f"candidate config does not exist: {candidate_config}")
    if not candidate_checkpoint.exists():
        raise ValueError(f"candidate checkpoint does not exist: {candidate_checkpoint}")

    metrics: dict[str, Any] = {
        "schema_version": 1,
        "run": {
            "experiment_id": experiment_id,
            "candidate": candidate,
            "preset_run_name": preset.run_name,
            "attention_type": preset.attention_type,
            "config": str(candidate_config),
            "checkpoint": str(candidate_checkpoint),
            "expected_checkpoint": str(preset.expected_checkpoint_path),
            "task_file": str(task_file),
            "hypothesis_doc": str(hypothesis_doc) if hypothesis_doc else None,
        },
        "mode": {"exploratory": exploratory, "probe_only": probe_only},
        "hypothesis": hypothesis_payload,
        "control": _control_metrics(control, control_seed_mismatched),
        "task_suite": {
            "metadata": suite.metadata,
            "deterministic_provenance": task_validation.deterministic_provenance,
            "pair_counts_by_family": task_validation.pair_counts_by_family,
            "confirmatory_floor_met": task_validation.confirmatory_floor_met,
            "validation_errors": list(task_validation.errors),
            "validation_warnings": list(task_validation.warnings),
            "min_n": min_n,
        },
        "sites_evaluated": [site.key for site in selected_sites],
        "cells": {},
    }

    candidate_model = load_mechanism_model(candidate_config, candidate_checkpoint, device=device)
    control_model = None
    if control.available and control.config_path is not None and control.checkpoint_path is not None:
        control_model = load_mechanism_model(control.config_path, control.checkpoint_path, device=device)

    fdr_inputs: dict[str, BootstrapResult] = {}
    cell_gate_inputs: dict[str, CellGateInputs] = {}
    split_by_family: dict[str, GroupedSplit] = {}
    for family in suite.families():
        records = suite.records_for_family(family)
        examples = examples_for_probe(records)
        texts = [example.text for example in examples]
        labels = np.asarray([example.label for example in examples], dtype=int)
        primary_indices = np.asarray(
            [index for index, example in enumerate(examples) if example.variant in {"pos", "neg"}],
            dtype=int,
        )
        decoy_indices = np.asarray(
            [index for index, example in enumerate(examples) if example.variant in {"pos", "decoy"}],
            dtype=int,
        )
        primary_examples = [examples[index] for index in primary_indices]
        primary_pair_ids = [example.pair_id for example in primary_examples]
        primary_template_ids = [example.template_id for example in primary_examples]
        primary_labels = labels[primary_indices]
        try:
            split = grouped_train_test_split(
                pair_ids=primary_pair_ids,
                template_ids=primary_template_ids,
                seed=seed,
                group_by_template=True,
            )
            split_by_family[family] = split
        except ValueError:
            split = None

        capture_sites = sorted({site.site for site in (*selected_sites, *preset.random_site_pool)})
        if control_model is not None:
            capture_sites.extend(sorted({site.control_site for site in selected_sites if site.control_site}))
        capture_sites = sorted(set(capture_sites))
        candidate_features = collect_activation_features(
            candidate_model,
            texts,
            sites=capture_sites,
            checkpoint_path=candidate_checkpoint,
            device=device,
            batch_size=batch_size,
        )
        control_features = None
        if control_model is not None and control.checkpoint_path is not None:
            control_sites = sorted({site.control_site for site in selected_sites if site.control_site})
            if control_sites:
                control_features = collect_activation_features(
                    control_model,
                    texts,
                    sites=control_sites,
                    checkpoint_path=control.checkpoint_path,
                    device=device,
                    batch_size=batch_size,
                )

        feature_shapes = {key: value.shape for key, value in candidate_features.features.items()}
        for site_index, site in enumerate(selected_sites):
            cell_id = f"{site.key}|family={family}"
            cell_metrics, gate_inputs = _evaluate_site_cell(
                site=site,
                site_index=site_index,
                family=family,
                examples=examples,
                records=records,
                primary_indices=primary_indices,
                decoy_indices=decoy_indices,
                primary_labels=primary_labels,
                primary_pair_ids=primary_pair_ids,
                primary_template_ids=primary_template_ids,
                split=split,
                candidate_features=candidate_features.features,
                candidate_model=candidate_model,
                candidate_checkpoint=candidate_checkpoint,
                control_features=control_features.features if control_features is not None else {},
                feature_shapes=feature_shapes,
                preset=preset,
                control=control,
                control_seed_mismatched=control_seed_mismatched,
                task_validation_errors=task_validation.errors,
                task_validation=task_validation,
                hypothesis_valid=hypothesis_valid,
                exploratory=exploratory,
                probe_only=probe_only,
                min_n=min_n,
                bootstrap_samples=bootstrap_samples,
                fdr_alpha=fdr_alpha,
                seed=seed,
                device=device,
                batch_size=batch_size,
            )
            metrics["cells"][cell_id] = cell_metrics
            cell_gate_inputs[cell_id] = gate_inputs
            for metric_name, result in cell_metrics.get("_bootstrap_results", {}).items():
                if result["valid"]:
                    fdr_inputs[f"{cell_id}|metric={metric_name}"] = _bootstrap_from_dict(result)
            cell_metrics.pop("_bootstrap_results", None)

    fdr_results = fdr_bh({metric_id: result.p_value for metric_id, result in fdr_inputs.items()}, alpha=fdr_alpha)
    metrics["fdr_bh"] = {
        "alpha": fdr_alpha,
        "comparison_family": "every computed (site x layer x task_family x metric) cell in the run",
        "tested_cells": sorted(fdr_results),
        "results": {
            metric_id: {
                "p_value": result.p_value,
                "q_value": result.q_value,
                "rejected": result.rejected,
            }
            for metric_id, result in sorted(fdr_results.items())
        },
    }

    cell_gate_results = {}
    for cell_id, inputs in cell_gate_inputs.items():
        updated = _inputs_with_fdr(cell_id, inputs, fdr_results)
        result = evaluate_cell_claim_gate(updated)
        cell_gate_results[cell_id] = result
    claim_gates = {
        "overall_status": overall_status(list(cell_gate_results.values())),
        "cells": {cell_id: result.to_dict() for cell_id, result in sorted(cell_gate_results.items())},
        "status_vocabulary": [
            "insufficient_evidence",
            "exploratory_probe_signal",
            "controlled_probe_signal",
            "candidate_mechanism_evidence",
        ],
        "status_vocabulary_scope": "mechanism-probe scoped; distinct from global experiment statuses",
    }
    write_suite_artifacts(output_dir, metrics, claim_gates)
    return metrics, claim_gates


def _evaluate_site_cell(
    *,
    site: SitePreset,
    site_index: int,
    family: str,
    examples: list[TaskExample],
    records: list[TaskRecord],
    primary_indices: np.ndarray,
    decoy_indices: np.ndarray,
    primary_labels: np.ndarray,
    primary_pair_ids: list[str],
    primary_template_ids: list[str],
    split: GroupedSplit | None,
    candidate_features: dict[str, np.ndarray],
    candidate_model,
    candidate_checkpoint: Path,
    control_features: dict[str, np.ndarray],
    feature_shapes: dict[str, tuple[int, int]],
    preset: MechanismProbePreset,
    control: ControlResolution,
    control_seed_mismatched: bool,
    task_validation_errors: tuple[str, ...],
    task_validation,
    hypothesis_valid: bool,
    exploratory: bool,
    probe_only: bool,
    min_n: int,
    bootstrap_samples: int,
    fdr_alpha: float,
    seed: int,
    device: str,
    batch_size: int,
) -> tuple[dict[str, Any], CellGateInputs]:
    cell_seed = seed + 1009 * site_index
    key = site.key
    base_metrics: dict[str, Any] = {
        "site": site.site,
        "layer": site.layer,
        "family_id": family,
        "linear_probe_auc": None,
        "auc_minus_shuffled_auc": None,
        "auc_minus_random_site_auc": None,
        "auc_minus_matched_control_auc": None,
        "target_vs_decoy_specificity": None,
        "bootstrap_alpha": fdr_alpha,
        "bootstrap_samples": bootstrap_samples,
        "random_site_null": {},
        "matched_control": {},
        "alignment_to_control": {},
        "patching": {"valid": False, "reason": "probe-only mode" if probe_only else "not computed by this probe cell"},
        "mediation_fraction": {"valid": False, "reason": "not computed by this probe cell"},
        "_bootstrap_results": {},
    }
    blockers = []
    if split is None:
        blockers.append("grouped split failed")
    if key not in candidate_features:
        blockers.append(f"candidate site features unavailable: {key}")
    if blockers:
        return base_metrics, _gate_inputs(
            exploratory=exploratory,
            probe_only=probe_only,
            hypothesis_valid=hypothesis_valid,
            real_probe_metrics=False,
            min_n_passed=not task_validation_errors,
            confirmatory_floor_met=task_validation.confirmatory_floor_met,
            grouped_split=False,
            control=control,
            control_seed_mismatched=control_seed_mismatched,
            random_available=False,
            extra_blockers=tuple(blockers),
        )

    features = candidate_features[key][primary_indices]
    try:
        probe = train_linear_probe(
            features,
            primary_labels,
            pair_ids=primary_pair_ids,
            template_ids=primary_template_ids,
            split=split,
            seed=cell_seed,
        )
        shuffled = train_shuffled_label_null(
            features,
            primary_labels,
            pair_ids=primary_pair_ids,
            template_ids=primary_template_ids,
            split=split,
            seed=cell_seed + 1,
        )
    except ValueError as exc:
        base_metrics["probe_error"] = str(exc)
        return base_metrics, _gate_inputs(
            exploratory=exploratory,
            probe_only=probe_only,
            hypothesis_valid=hypothesis_valid,
            real_probe_metrics=False,
            min_n_passed=not task_validation_errors,
            confirmatory_floor_met=task_validation.confirmatory_floor_met,
            grouped_split=split is not None,
            control=control,
            control_seed_mismatched=control_seed_mismatched,
            random_available=False,
            extra_blockers=(str(exc),),
        )

    base_metrics.update(probe.to_metrics())
    base_metrics["shuffled_label_auc"] = float(shuffled.auc)
    base_metrics["auc_minus_shuffled_auc"] = float(probe.auc - shuffled.auc)
    base_metrics["_bootstrap_results"]["linear_probe_auc_minus_0_5"] = asdict(
        _bootstrap_auc(
            labels=primary_labels[split.test_indices],
            scores=probe.scores[split.test_indices],
            group_ids=[primary_pair_ids[index] for index in split.test_indices],
            samples=bootstrap_samples,
            seed=cell_seed + 2,
            null=0.5,
        )
    )
    base_metrics["_bootstrap_results"]["auc_minus_shuffled_auc"] = asdict(
        _bootstrap_auc_difference(
            labels=primary_labels[split.test_indices],
            left_scores=probe.scores[split.test_indices],
            right_scores=shuffled.scores[split.test_indices],
            group_ids=[primary_pair_ids[index] for index in split.test_indices],
            samples=bootstrap_samples,
            seed=cell_seed + 3,
        )
    )

    random_selection = select_random_site_null(
        candidate=site,
        candidate_key=key,
        feature_shapes=feature_shapes,
        pool=preset.random_site_pool,
        seed=cell_seed + 4,
    )
    base_metrics["random_site_null"] = random_selection.to_dict()
    random_pass = False
    if random_selection.available and random_selection.selected_site is not None:
        random_features = candidate_features[random_selection.selected_site][primary_indices]
        random_probe = train_linear_probe(
            random_features,
            primary_labels,
            pair_ids=primary_pair_ids,
            template_ids=primary_template_ids,
            split=split,
            seed=cell_seed + 5,
        )
        base_metrics["random_site_auc"] = float(random_probe.auc)
        base_metrics["auc_minus_random_site_auc"] = float(probe.auc - random_probe.auc)
        random_ci = _bootstrap_auc_difference(
            labels=primary_labels[split.test_indices],
            left_scores=probe.scores[split.test_indices],
            right_scores=random_probe.scores[split.test_indices],
            group_ids=[primary_pair_ids[index] for index in split.test_indices],
            samples=bootstrap_samples,
            seed=cell_seed + 6,
        )
        base_metrics["_bootstrap_results"]["auc_minus_random_site_auc"] = asdict(random_ci)
        random_pass = ci_excludes_null(random_ci)

    control_pass = False
    control_probe: LinearProbeResult | None = None
    if site.control_site is None:
        base_metrics["matched_control"] = {"available": False, "reason": "site has no matched control site metadata"}
    else:
        control_key = f"{site.control_site}[{site.layer}]"
        control_matrix = control_features.get(control_key)
        if control_matrix is None:
            base_metrics["matched_control"] = {"available": False, "control_site": control_key, "reason": "control site features unavailable"}
        elif control_matrix.shape[1] != features.shape[1]:
            base_metrics["matched_control"] = {
                "available": False,
                "control_site": control_key,
                "reason": f"shape mismatch: candidate_dim={features.shape[1]}, control_dim={control_matrix.shape[1]}",
            }
        else:
            control_probe = train_linear_probe(
                control_matrix[primary_indices],
                primary_labels,
                pair_ids=primary_pair_ids,
                template_ids=primary_template_ids,
                split=split,
                seed=cell_seed + 7,
            )
            base_metrics["matched_control"] = {"available": True, "control_site": control_key, "auc": float(control_probe.auc)}
            base_metrics["matched_control_auc"] = float(control_probe.auc)
            base_metrics["auc_minus_matched_control_auc"] = float(probe.auc - control_probe.auc)
            control_ci = _bootstrap_auc_difference(
                labels=primary_labels[split.test_indices],
                left_scores=probe.scores[split.test_indices],
                right_scores=control_probe.scores[split.test_indices],
                group_ids=[primary_pair_ids[index] for index in split.test_indices],
                samples=bootstrap_samples,
                seed=cell_seed + 8,
            )
            base_metrics["_bootstrap_results"]["auc_minus_matched_control_auc"] = asdict(control_ci)
            base_metrics["alignment_to_control"] = probe_direction_alignment(probe.weight, control_probe.weight).to_dict()
            control_pass = ci_excludes_null(control_ci)
    if control_probe is None:
        base_metrics["alignment_to_control"] = {
            "available": False,
            "reason": "matched control probe unavailable",
            "probe_direction_cosine_to_control": None,
            "probe_direction_alignment_abs": None,
        }

    decoy_features = candidate_features[key][decoy_indices]
    decoy_examples = [examples[index] for index in decoy_indices]
    decoy_labels = np.asarray([example.label for example in decoy_examples], dtype=int)
    decoy_groups = [example.pair_id for example in decoy_examples]
    test_group_set = set(split.test_groups)
    decoy_test_indices = np.asarray(
        [index for index, example in enumerate(decoy_examples) if example.template_id in test_group_set],
        dtype=int,
    )
    if len(decoy_test_indices) == 0:
        decoy_test_indices = np.arange(len(decoy_examples))
    decoy_scores = score_with_probe(decoy_features, probe)
    try:
        decoy_auc = auc_score(decoy_labels[decoy_test_indices], decoy_scores[decoy_test_indices])
        base_metrics["decoy_auc"] = float(decoy_auc)
        base_metrics["target_vs_decoy_specificity"] = float(probe.auc - decoy_auc)
        base_metrics["_bootstrap_results"]["target_vs_decoy_specificity"] = asdict(
            _bootstrap_specificity(
                primary_labels=primary_labels[split.test_indices],
                primary_scores=probe.scores[split.test_indices],
                primary_groups=[primary_pair_ids[index] for index in split.test_indices],
                decoy_labels=decoy_labels[decoy_test_indices],
                decoy_scores=decoy_scores[decoy_test_indices],
                decoy_groups=[decoy_groups[index] for index in decoy_test_indices],
                samples=bootstrap_samples,
                seed=cell_seed + 9,
            )
        )
    except ValueError as exc:
        base_metrics["decoy_specificity_error"] = str(exc)

    if not probe_only:
        patching = _compute_patching_metrics(
            records=records,
            site=site,
            model=candidate_model.model,
            attention_type=candidate_model.attention_type,
            tokenizer_name=str(candidate_model.tokenizer["tokenizer"]),
            block_size=int(candidate_model.tokenizer["block_size"]),
            vocab_size=int(candidate_model.tokenizer["vocab_size"]),
            checkpoint_path=candidate_checkpoint,
            device=device,
            batch_size=batch_size,
            bootstrap_samples=bootstrap_samples,
            seed=cell_seed + 10,
        )
        base_metrics["patching"] = patching["patching"]
        base_metrics["mediation_fraction"] = patching["mediation_fraction"]
        for metric_name, result in patching.get("_bootstrap_results", {}).items():
            base_metrics["_bootstrap_results"][metric_name] = result

    primary_ci = _bootstrap_from_dict(base_metrics["_bootstrap_results"]["linear_probe_auc_minus_0_5"])
    shuffled_ci = _bootstrap_from_dict(base_metrics["_bootstrap_results"]["auc_minus_shuffled_auc"])
    specificity_ci = _bootstrap_from_dict(
        base_metrics["_bootstrap_results"].get(
            "target_vs_decoy_specificity",
            {
                "valid": False,
                "estimate": float("nan"),
                "ci_low": float("nan"),
                "ci_high": float("nan"),
                "p_value": float("nan"),
                "samples": bootstrap_samples,
                "alpha": fdr_alpha,
                "expected_direction": "positive",
                "reason": "specificity unavailable",
            },
        )
    )
    return base_metrics, _gate_inputs(
        exploratory=exploratory,
        probe_only=probe_only,
        hypothesis_valid=hypothesis_valid,
        real_probe_metrics=True,
        min_n_passed=not task_validation_errors,
        confirmatory_floor_met=task_validation.confirmatory_floor_met,
        grouped_split=True,
        control=control,
        control_seed_mismatched=control_seed_mismatched,
        shuffled_pass=ci_excludes_null(shuffled_ci),
        random_available=random_selection.available,
        random_pass=random_pass,
        control_pass=control_pass,
        primary_ci_pass=ci_excludes_null(primary_ci),
        specificity_ci_pass=ci_excludes_null(specificity_ci),
        patching_valid=bool(base_metrics["patching"].get("valid")),
        mediation_valid=bool(base_metrics["mediation_fraction"].get("valid")),
    )


def _compute_patching_metrics(
    *,
    records: list[TaskRecord],
    site: SitePreset,
    model,
    attention_type: str,
    tokenizer_name: str,
    block_size: int,
    vocab_size: int,
    checkpoint_path: Path,
    device: str,
    batch_size: int,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    _ = batch_size
    if is_discrete_hook_site(attention_type, site.site):
        return {
            "patching": {"valid": False, "reason": "discrete route/index sites are capture-only"},
            "mediation_fraction": mediation_fraction(
                component_patch_restoration=None,
                full_layer_patch_restoration=None,
            ).to_dict(),
            "_bootstrap_results": {},
        }

    token_pairs = []
    for record in records:
        target = record.metadata.get("target_token_id")
        foil = record.metadata.get("foil_token_id")
        if target is None or foil is None:
            return {
                "patching": {
                    "valid": False,
                    "reason": "task records lack explicit target_token_id/foil_token_id patching metadata",
                },
                "mediation_fraction": mediation_fraction(
                    component_patch_restoration=None,
                    full_layer_patch_restoration=None,
                ).to_dict(),
                "_bootstrap_results": {},
            }
        token_pairs.append((int(target), int(foil)))

    component_scores: list[float] = []
    full_layer_scores: list[float] = []
    pair_ids: list[str] = []
    with torch.no_grad():
        for record, (target_id, foil_id) in zip(records, token_pairs, strict=True):
            encoded = encode_texts(
                [record.x_pos, record.x_neg],
                tokenizer_name=tokenizer_name,
                block_size=block_size,
                vocab_size=vocab_size,
            )
            clean_ids = encoded.input_ids[:1].to(device)
            corrupt_ids = encoded.input_ids[1:2].to(device)
            clean_len = encoded.lengths[0]
            corrupt_len = encoded.lengths[1]
            clean_capture = capture_activations(
                model,
                clean_ids,
                sites=[site.site, "attn_out"],
                detach=True,
                cpu=True,
                checkpoint_path=checkpoint_path,
                schedule_mode="eval",
            )
            corrupt_logits, _ = model(corrupt_ids, schedule_mode="eval")
            clean_logitdiff = _logitdiff(clean_capture.logits, target_id, foil_id, clean_len)
            corrupted_logitdiff = _logitdiff(corrupt_logits, target_id, foil_id, corrupt_len)

            component = run_with_interventions(
                model,
                corrupt_ids,
                [make_cache_patch(clean_capture.cache, site=site.site, layer=site.layer)],
                capture_sites=[site.site],
                schedule_mode="eval",
            )
            component_result = restoration_score(
                clean_logitdiff=clean_logitdiff,
                corrupted_logitdiff=corrupted_logitdiff,
                patched_logitdiff=_logitdiff(component.logits, target_id, foil_id, corrupt_len),
            )
            full_layer = run_with_interventions(
                model,
                corrupt_ids,
                [make_cache_patch(clean_capture.cache, site="attn_out", layer=site.layer)],
                capture_sites=["attn_out"],
                schedule_mode="eval",
            )
            full_result = restoration_score(
                clean_logitdiff=clean_logitdiff,
                corrupted_logitdiff=corrupted_logitdiff,
                patched_logitdiff=_logitdiff(full_layer.logits, target_id, foil_id, corrupt_len),
            )
            if component_result.valid and full_result.valid:
                component_scores.append(float(component_result.restoration_score))
                full_layer_scores.append(float(full_result.restoration_score))
                pair_ids.append(record.pair_id)

    if not component_scores:
        return {
            "patching": {"valid": False, "reason": "no valid restoration denominators"},
            "mediation_fraction": mediation_fraction(
                component_patch_restoration=None,
                full_layer_patch_restoration=None,
            ).to_dict(),
            "_bootstrap_results": {},
        }

    component_array = np.asarray(component_scores, dtype=float)
    full_array = np.asarray(full_layer_scores, dtype=float)
    component_mean = float(np.mean(component_array))
    full_mean = float(np.mean(full_array))
    mediation = mediation_fraction(
        component_patch_restoration=component_mean,
        full_layer_patch_restoration=full_mean,
    )
    component_bootstrap = bootstrap_metric(
        component_array,
        pair_ids,
        lambda values: float(np.mean(values)),
        samples=bootstrap_samples,
        seed=seed,
        expected_direction="positive",
    )
    mediation_values = component_array / np.where(np.abs(full_array) < 1e-6, np.nan, full_array)
    mediation_bootstrap = bootstrap_metric(
        mediation_values,
        pair_ids,
        lambda values: float(np.nanmean(values)),
        samples=bootstrap_samples,
        seed=seed + 1,
        expected_direction="positive",
    )
    return {
        "patching": {
            "valid": True,
            "component_patch_restoration": component_mean,
            "full_layer_patch_restoration": full_mean,
            "valid_pairs": len(component_scores),
            "formula": "(patched_logitdiff - corrupted_logitdiff) / (clean_logitdiff - corrupted_logitdiff)",
        },
        "mediation_fraction": mediation.to_dict(),
        "_bootstrap_results": {
            "component_patch_restoration": asdict(component_bootstrap),
            "mediation_fraction": asdict(mediation_bootstrap),
        },
    }


def _logitdiff(logits: torch.Tensor, target_id: int, foil_id: int, length: int) -> float:
    position = max(0, min(length - 1, logits.shape[1] - 1))
    return float((logits[0, position, target_id] - logits[0, position, foil_id]).detach().cpu().item())


def _bootstrap_auc(
    *,
    labels: np.ndarray,
    scores: np.ndarray,
    group_ids: list[str],
    samples: int,
    seed: int,
    null: float,
) -> BootstrapResult:
    values = np.arange(len(labels))
    return bootstrap_metric(
        values,
        group_ids,
        lambda indices: auc_score(labels[np.asarray(indices, dtype=int)], scores[np.asarray(indices, dtype=int)]) - null,
        samples=samples,
        seed=seed,
        expected_direction="positive",
    )


def _bootstrap_auc_difference(
    *,
    labels: np.ndarray,
    left_scores: np.ndarray,
    right_scores: np.ndarray,
    group_ids: list[str],
    samples: int,
    seed: int,
) -> BootstrapResult:
    values = np.arange(len(labels))
    return bootstrap_metric(
        values,
        group_ids,
        lambda indices: (
            auc_score(labels[np.asarray(indices, dtype=int)], left_scores[np.asarray(indices, dtype=int)])
            - auc_score(labels[np.asarray(indices, dtype=int)], right_scores[np.asarray(indices, dtype=int)])
        ),
        samples=samples,
        seed=seed,
        expected_direction="positive",
    )


def _bootstrap_specificity(
    *,
    primary_labels: np.ndarray,
    primary_scores: np.ndarray,
    primary_groups: list[str],
    decoy_labels: np.ndarray,
    decoy_scores: np.ndarray,
    decoy_groups: list[str],
    samples: int,
    seed: int,
) -> BootstrapResult:
    primary_values = np.arange(len(primary_labels))

    def statistic(indices: np.ndarray) -> float:
        selected_groups = {primary_groups[int(index)] for index in indices}
        decoy_indices = np.asarray(
            [index for index, group in enumerate(decoy_groups) if group in selected_groups],
            dtype=int,
        )
        if len(decoy_indices) == 0:
            decoy_indices = np.arange(len(decoy_labels))
        primary_auc = auc_score(primary_labels[np.asarray(indices, dtype=int)], primary_scores[np.asarray(indices, dtype=int)])
        decoy_auc = auc_score(decoy_labels[decoy_indices], decoy_scores[decoy_indices])
        return primary_auc - decoy_auc

    return bootstrap_metric(
        primary_values,
        primary_groups,
        statistic,
        samples=samples,
        seed=seed,
        expected_direction="positive",
    )


def _inputs_with_fdr(cell_id: str, inputs: CellGateInputs, fdr_results: dict[str, Any]) -> CellGateInputs:
    def rejected(metric: str) -> bool:
        result = fdr_results.get(f"{cell_id}|metric={metric}")
        return bool(result and result.rejected)

    return CellGateInputs(
        **{
            **inputs.__dict__,
            "primary_fdr_passed": rejected("linear_probe_auc_minus_0_5"),
            "shuffled_null_passed": inputs.shuffled_null_passed and rejected("auc_minus_shuffled_auc"),
            "random_site_null_passed": inputs.random_site_null_passed and rejected("auc_minus_random_site_auc"),
            "matched_control_passed": inputs.matched_control_passed and rejected("auc_minus_matched_control_auc"),
            "specificity_fdr_passed": rejected("target_vs_decoy_specificity"),
        }
    )


def _gate_inputs(
    *,
    exploratory: bool,
    probe_only: bool,
    hypothesis_valid: bool,
    real_probe_metrics: bool,
    min_n_passed: bool,
    confirmatory_floor_met: bool,
    grouped_split: bool,
    control: ControlResolution,
    control_seed_mismatched: bool,
    shuffled_pass: bool = False,
    random_available: bool = False,
    random_pass: bool = False,
    control_pass: bool = False,
    primary_ci_pass: bool = False,
    specificity_ci_pass: bool = False,
    patching_valid: bool = False,
    mediation_valid: bool = False,
    extra_blockers: tuple[str, ...] = (),
) -> CellGateInputs:
    return CellGateInputs(
        exploratory=exploratory,
        probe_only=probe_only,
        hypothesis_doc_valid=hypothesis_valid,
        real_probe_metrics=real_probe_metrics,
        min_n_passed=min_n_passed,
        confirmatory_floor_met=confirmatory_floor_met,
        grouped_split=grouped_split,
        matched_control_available=control.available,
        canonical_control=control.canonical and not control_seed_mismatched,
        noncanonical_control=not control.canonical or control_seed_mismatched,
        shuffled_null_passed=shuffled_pass,
        random_site_null_available=random_available,
        random_site_null_passed=random_pass,
        matched_control_passed=control_pass,
        primary_fdr_passed=False,
        primary_ci_passed=primary_ci_pass,
        specificity_fdr_passed=False,
        specificity_ci_passed=specificity_ci_pass,
        patching_valid=patching_valid,
        mediation_valid=mediation_valid,
        force_noncanonical_control=control.force_noncanonical,
        extra_blockers=extra_blockers,
    )


def _control_metrics(control: ControlResolution, seed_mismatched: bool) -> dict[str, Any]:
    expected = control.expected_control
    return {
        "expected_control_run": expected.run_name if expected else None,
        "expected_control_config": str(expected.config_path) if expected else None,
        "expected_control_checkpoint": str(expected.checkpoint_path) if expected else None,
        "actual_control_config": str(control.config_path) if control.config_path else None,
        "actual_control_checkpoint": str(control.checkpoint_path) if control.checkpoint_path else None,
        "canonical": control.canonical and not seed_mismatched,
        "override_used": control.override_used,
        "available": control.available,
        "force_noncanonical": control.force_noncanonical,
        "seed_mismatched": seed_mismatched,
        "reason": control.reason,
    }


def _stub_metrics(
    preset: MechanismProbePreset,
    checkpoint: str | Path,
    task_file: str | Path,
    exploratory: bool,
    probe_only: bool,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run": {
            "experiment_id": preset.experiment_id,
            "candidate": preset.candidate,
            "checkpoint": str(checkpoint),
            "task_file": str(task_file),
        },
        "mode": {"exploratory": exploratory, "probe_only": probe_only},
        "preset_status": preset.status,
        "tier": preset.tier,
        "executable": preset.executable,
        "reason": preset.notes,
        "cells": {},
    }


def _bootstrap_from_dict(payload: dict[str, Any]) -> BootstrapResult:
    return BootstrapResult(
        estimate=float(payload.get("estimate", float("nan"))),
        ci_low=float(payload.get("ci_low", float("nan"))),
        ci_high=float(payload.get("ci_high", float("nan"))),
        p_value=float(payload.get("p_value", float("nan"))),
        samples=int(payload.get("samples", 0)),
        alpha=float(payload.get("alpha", 0.05)),
        expected_direction=str(payload.get("expected_direction", "positive")),
        valid=bool(payload.get("valid")),
        reason=payload.get("reason"),
    )
