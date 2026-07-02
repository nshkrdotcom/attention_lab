from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import yaml


GENERATOR_NAME = "attention_lab_template_filler_contrast"
GENERATOR_VERSION = "1"


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
