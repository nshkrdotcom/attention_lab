from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import yaml

from attention_lab.mechanisms.task_schema import (
    CONFIRMATORY_MIN_PAIRS_PER_FAMILY,
    load_task_suite,
    validate_task_suite,
)


GENERATOR_NAME = "attention_lab_template_filler_contrast"
GENERATOR_VERSION = "1"
TIER1_GENERATOR_NAME = "attention_lab_tier1_negation_template_filler"
TIER1_GENERATOR_VERSION = "1"
TIER1_CANDIDATES = ("e003_differential", "e004_operator_valued")
DEFAULT_CREATED_AT = "2026-07-01T00:00:00Z"
TARGET_TOKEN_TEXT = " true"
FOIL_TOKEN_TEXT = " false"


def generate_template_filler_suite(
    *,
    template_set: str,
    filler_set: str,
    families: dict[str, dict[str, list[str]]],
    pairs_per_family: int,
    seed: int,
    created_at: str = "1970-01-01T00:00:00Z",
) -> dict[str, Any]:
    if pairs_per_family <= 0:
        raise ValueError("pairs_per_family must be positive")
    rng = random.Random(seed)
    records: list[dict[str, Any]] = []
    for family_id, spec in sorted(families.items()):
        templates = spec.get("templates") or []
        targets = spec.get("targets") or []
        negatives = spec.get("negatives") or []
        paraphrases = spec.get("paraphrases") or []
        decoys = spec.get("decoys") or []
        if not all((templates, targets, negatives, paraphrases, decoys)):
            raise ValueError(f"family {family_id} must define templates, targets, negatives, paraphrases, and decoys")
        for index in range(pairs_per_family):
            template = templates[index % len(templates)]
            target = rng.choice(targets)
            negative = rng.choice(negatives)
            paraphrase = rng.choice(paraphrases)
            decoy = rng.choice(decoys)
            template_id = f"{family_id}_template_{index % len(templates):03d}"
            pair_id = f"{family_id}_pair_{index:04d}"
            records.append(
                {
                    "pair_id": pair_id,
                    "template_id": template_id,
                    "family_id": family_id,
                    "x_pos": template.format(target=target, marker=target),
                    "x_neg": template.format(target=negative, marker=negative),
                    "x_para": template.format(target=paraphrase, marker=paraphrase),
                    "x_decoy": template.format(target=decoy, marker=decoy),
                    "metadata": {
                        "template_index": index % len(templates),
                        "target": target,
                        "negative": negative,
                        "paraphrase": paraphrase,
                        "decoy": decoy,
                    },
                }
            )
    return {
        "schema_version": 1,
        "metadata": {
            "generator_name": GENERATOR_NAME,
            "generator_version": GENERATOR_VERSION,
            "template_set": template_set,
            "filler_set": filler_set,
            "generation_seed": seed,
            "created_at": created_at,
        },
        "records": records,
    }


def write_template_filler_suite(path: str | Path, suite: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(suite, sort_keys=False), encoding="utf-8")


def generate_tier1_negation_suite(
    *,
    candidate: str,
    pairs_per_family: int,
    seed: int,
    created_at: str = DEFAULT_CREATED_AT,
) -> dict[str, Any]:
    normalized = candidate.strip().lower()
    if normalized not in TIER1_CANDIDATES:
        raise ValueError(f"candidate must be one of: {', '.join(TIER1_CANDIDATES)}")
    if pairs_per_family < CONFIRMATORY_MIN_PAIRS_PER_FAMILY:
        raise ValueError(
            f"confirmatory Tier-1 suites require at least {CONFIRMATORY_MIN_PAIRS_PER_FAMILY} pairs per family"
        )

    target_id = gpt2_single_token_id(TARGET_TOKEN_TEXT)
    foil_id = gpt2_single_token_id(FOIL_TOKEN_TEXT)
    rng = random.Random(seed)
    records: list[dict[str, Any]] = []
    family_id = "negation_scope"
    combinations = _negation_combinations()
    rng.shuffle(combinations)
    prompt_templates = _prompt_templates(normalized)
    for index in range(pairs_per_family):
        subject, verb, obj, decoy_adverb = combinations[index % len(combinations)]
        prompt_template = prompt_templates[index % len(prompt_templates)]
        template_id = f"{family_id}_template_{index % len(prompt_templates):03d}"
        pair_id = f"{normalized}_{family_id}_pair_{index:04d}"
        x_pos = prompt_template.format(sentence=f"{subject} did not {verb} {obj}.")
        x_neg = prompt_template.format(sentence=f"{subject} {verb} {obj}.")
        x_para = prompt_template.format(sentence=f"{subject} never {verb} {obj}.")
        x_decoy = prompt_template.format(sentence=f"{subject} {decoy_adverb} {verb} {obj}.")
        records.append(
            {
                "pair_id": pair_id,
                "template_id": template_id,
                "family_id": family_id,
                "x_pos": x_pos,
                "x_neg": x_neg,
                "x_para": x_para,
                "x_decoy": x_decoy,
                "metadata": {
                    "phenomenon": "explicit_negation",
                    "target_token_text": TARGET_TOKEN_TEXT,
                    "foil_token_text": FOIL_TOKEN_TEXT,
                    "target_token_id": target_id,
                    "foil_token_id": foil_id,
                    "subject": subject,
                    "verb": verb,
                    "object": obj,
                    "positive_marker": "did not",
                    "paraphrase_marker": "never",
                    "decoy_control": decoy_adverb,
                },
            }
        )

    return {
        "schema_version": 1,
        "metadata": {
            "generator_name": TIER1_GENERATOR_NAME,
            "generator_version": TIER1_GENERATOR_VERSION,
            "template_set": f"{normalized}_negation_prompts_v1",
            "filler_set": "negation_scope_fillers_v1",
            "generation_seed": seed,
            "created_at": created_at,
            "candidate": normalized,
            "restoration_token_metadata": "gpt2_single_token_true_false_v1",
        },
        "records": records,
    }


def write_tier1_negation_suite(path: str | Path, suite: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(suite, sort_keys=False), encoding="utf-8")


def validate_tier1_suite_file(path: str | Path, *, min_n: int = CONFIRMATORY_MIN_PAIRS_PER_FAMILY) -> None:
    suite = load_task_suite(path)
    result = validate_task_suite(
        suite,
        confirmatory=True,
        exploratory=False,
        min_n=min_n,
        require_decoys=True,
        require_restoration_tokens=True,
    )
    if not result.valid:
        raise ValueError("invalid Tier-1 task suite: " + "; ".join(result.errors))


def gpt2_single_token_id(token_text: str) -> int:
    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    encoded = enc.encode(token_text)
    if len(encoded) != 1:
        raise ValueError(f"{token_text!r} is not a single GPT-2 token")
    return int(encoded[0])


def _prompt_templates(candidate: str) -> list[str]:
    prefix = "E003 differential" if candidate == "e003_differential" else "E004 operator-valued"
    return [
        f"{prefix} Tier-1 probe. Sentence: {{sentence}} Does the sentence contain explicit negation? Answer:",
        f"{prefix} Tier-1 probe. Statement: {{sentence}} Is a negation marker present? Answer:",
        f"{prefix} Tier-1 probe. Text: {{sentence}} Does this text explicitly negate the action? Answer:",
        f"{prefix} Tier-1 probe. Example: {{sentence}} Is the action denied by a negation cue? Answer:",
        f"{prefix} Tier-1 probe. Input: {{sentence}} Does a negation cue reverse the action? Answer:",
        f"{prefix} Tier-1 probe. Clause: {{sentence}} Is there an explicit not-or-never style negation? Answer:",
        f"{prefix} Tier-1 probe. Premise: {{sentence}} Does the premise include negation? Answer:",
        f"{prefix} Tier-1 probe. Record: {{sentence}} Is explicit negation present in the record? Answer:",
        f"{prefix} Tier-1 probe. Utterance: {{sentence}} Does the utterance contain a negating marker? Answer:",
        f"{prefix} Tier-1 probe. Observation: {{sentence}} Is the described action negated? Answer:",
    ]


def _negation_combinations() -> list[tuple[str, str, str, str]]:
    subjects = [
        "The analyst",
        "The editor",
        "The teacher",
        "The engineer",
        "The nurse",
        "The planner",
        "The auditor",
        "The coach",
        "The curator",
        "The pilot",
    ]
    verbs = [
        "approve",
        "deliver",
        "revise",
        "publish",
        "verify",
        "accept",
        "archive",
        "inspect",
        "confirm",
        "release",
    ]
    objects = [
        "the report",
        "the schedule",
        "the invoice",
        "the proposal",
        "the dataset",
        "the memo",
        "the checklist",
        "the design",
        "the package",
        "the update",
    ]
    decoy_adverbs = ["quickly", "carefully", "usually", "quietly", "clearly", "today", "often", "boldly"]
    return [
        (subject, verb, obj, adverb)
        for subject in subjects
        for verb in verbs
        for obj in objects
        for adverb in decoy_adverbs
    ]
