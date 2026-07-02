from __future__ import annotations

from datetime import UTC, datetime

from attention_lab.mechanisms.task_schema import TaskRecord, TaskSuite


GENERATOR_NAME = "attention_lab_template_filler_v1"
GENERATOR_VERSION = "1"


def generate_deterministic_contrast_suite(
    *,
    family_id: str = "negation_scope",
    pairs_per_family: int = 50,
    generation_seed: int = 1,
) -> TaskSuite:
    """Generate a small deterministic template/filler contrast suite.

    This helper is intentionally simple and auditable. It is not a claim that these
    prompts are sufficient for science; confirmatory runs still need hypothesis docs,
    controls, nulls, and gates.
    """

    templates = (
        "The {object} is {state}.",
        "In the note, the {object} is {state}.",
        "A report says the {object} is {state}.",
        "The label marks the {object} as {state}.",
        "The operator found the {object} {state}.",
    )
    objects = (
        "valve",
        "switch",
        "sensor",
        "module",
        "relay",
        "panel",
        "circuit",
        "filter",
        "router",
        "meter",
    )
    positive_states = ("open", "active", "enabled", "connected", "ready")
    negative_states = ("closed", "inactive", "disabled", "disconnected", "blocked")
    decoy_states = ("nearby", "metal", "labeled", "visible", "portable")
    records: list[TaskRecord] = []
    for idx in range(pairs_per_family):
        template_idx = (idx + generation_seed) % len(templates)
        object_idx = (idx * 3 + generation_seed) % len(objects)
        state_idx = (idx * 5 + generation_seed) % len(positive_states)
        template = templates[template_idx]
        obj = objects[object_idx]
        pos = positive_states[state_idx]
        neg = negative_states[state_idx]
        decoy = decoy_states[(idx * 7 + generation_seed) % len(decoy_states)]
        records.append(
            TaskRecord(
                x_pos=template.format(object=obj, state=pos),
                x_neg=template.format(object=obj, state=neg),
                x_para=f"Status: {obj} remains {pos}.",
                x_decoy=template.format(object=obj, state=decoy),
                pair_id=f"{family_id}_pair_{idx:04d}",
                template_id=f"{family_id}_template_{template_idx}",
                family_id=family_id,
                metadata={"template_index": template_idx, "filler_index": object_idx},
            )
        )
    return TaskSuite(
        records=tuple(records),
        metadata={
            "generator_name": GENERATOR_NAME,
            "generator_version": GENERATOR_VERSION,
            "template_set": "default_status_templates",
            "filler_set": "default_status_fillers",
            "generation_seed": generation_seed,
            "created_at": datetime.now(UTC).date().isoformat(),
        },
    )
