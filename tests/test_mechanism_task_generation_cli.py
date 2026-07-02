from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

from attention_lab.mechanisms.task_schema import load_task_suite, validate_task_suite


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


def test_tier1_task_validate_only_rejects_tampered_generated_suite(tmp_path):
    task_file = tmp_path / "generated.yaml"
    generated = subprocess.run(
        [
            sys.executable,
            "scripts/generate_tier1_mechanism_tasks.py",
            "--output",
            str(task_file),
            "--candidate",
            "e003_differential",
            "--pairs-per-family",
            "50",
            "--seed",
            "13",
        ],
        cwd=Path.cwd(),
        check=False,
        capture_output=True,
        text=True,
    )
    assert generated.returncode == 0, generated.stderr

    payload = yaml.safe_load(task_file.read_text(encoding="utf-8"))
    assert payload["metadata"].get("content_sha256")
    payload["records"][0]["x_pos"] = payload["records"][0]["x_pos"].replace("not", "definitely not")
    task_file.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/generate_tier1_mechanism_tasks.py",
            "--output",
            str(task_file),
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
    assert "content_sha256" in result.stderr or "deterministic generator output" in result.stderr


def test_committed_tier1_task_suites_validate_with_restoration_alignment_metadata():
    for path in (
        Path("configs/mechanisms/tier1_tasks/E003_differential_negation_tier1.yaml"),
        Path("configs/mechanisms/tier1_tasks/E004_operator_valued_negation_tier1.yaml"),
    ):
        suite = load_task_suite(path)
        result = validate_task_suite(
            suite,
            confirmatory=True,
            exploratory=False,
            min_n=50,
            require_restoration_tokens=True,
        )

        assert result.valid, result.errors
        assert result.deterministic_provenance
        assert result.deterministic_fingerprint_valid
        assert result.confirmatory_floor_met
        for record in suite.records:
            assert record.x_pos
            assert record.x_neg
            assert record.x_para
            assert record.x_decoy
            assert record.pair_id
            assert record.template_id
            assert record.family_id
            assert isinstance(record.metadata["target_token_id"], int)
            assert isinstance(record.metadata["foil_token_id"], int)
            assert isinstance(record.metadata["clean_answer_position"], int)
            assert isinstance(record.metadata["corrupted_answer_position"], int)
            assert isinstance(record.metadata["patch_token_indices"], list)


def test_tier1_verify_script_preflight_only_does_not_require_execution():
    result = subprocess.run(
        [
            sys.executable,
            "scripts/verify_tier1_mechanism_probe_suite.py",
            "--preflight-only",
        ],
        cwd=Path.cwd(),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "preflight complete" in result.stdout
