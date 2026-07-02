from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml


def test_tier1_task_generation_is_deterministic_for_same_seed(tmp_path):
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    for output in (first, second):
        result = subprocess.run(
            [
                sys.executable,
                "scripts/generate_tier1_mechanism_tasks.py",
                "--output",
                str(output),
                "--candidate",
                "e003_differential",
                "--pairs-per-family",
                "50",
                "--seed",
                "11",
            ],
            cwd=Path.cwd(),
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr

    assert first.read_text(encoding="utf-8") == second.read_text(encoding="utf-8")


def test_tier1_task_generation_changes_fillers_for_different_seed_without_schema_drift(tmp_path):
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    for seed, output in ((11, first), (12, second)):
        result = subprocess.run(
            [
                sys.executable,
                "scripts/generate_tier1_mechanism_tasks.py",
                "--output",
                str(output),
                "--candidate",
                "e004_operator_valued",
                "--pairs-per-family",
                "50",
                "--seed",
                str(seed),
            ],
            cwd=Path.cwd(),
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr

    first_payload = yaml.safe_load(first.read_text(encoding="utf-8"))
    second_payload = yaml.safe_load(second.read_text(encoding="utf-8"))
    assert first_payload["metadata"].keys() == second_payload["metadata"].keys()
    assert first_payload["records"][0].keys() == second_payload["records"][0].keys()
    assert first_payload["records"] != second_payload["records"]


def test_tier1_task_validate_only_rejects_missing_provenance_too_small_and_missing_tokens(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "metadata": {},
                "records": [
                    {
                        "pair_id": "pair_0",
                        "template_id": "template_0",
                        "family_id": "negation_scope",
                        "x_pos": "Sentence: The analyst did not approve the report. Answer:",
                        "x_neg": "Sentence: The analyst approved the report. Answer:",
                        "x_para": "Sentence: The analyst never approved the report. Answer:",
                        "x_decoy": "Sentence: The analyst carefully approved the report. Answer:",
                        "metadata": {},
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/generate_tier1_mechanism_tasks.py",
            "--output",
            str(bad),
            "--candidate",
            "e003_differential",
            "--pairs-per-family",
            "50",
            "--validate-only",
        ],
        cwd=Path.cwd(),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "deterministic generator provenance" in result.stderr
    assert "below --min-n=50" in result.stderr
    assert "lacks restoration token metadata" in result.stderr
