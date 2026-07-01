from __future__ import annotations

from attention_lab.queue.discipline import validate_hypothesis_doc


def test_queue_discipline_guide_covers_required_source_requirements(repo_root):
    guide_path = repo_root / "docs" / "guides" / "experiment_queue_discipline_checklist.md"
    assert guide_path.exists()
    guide = guide_path.read_text(encoding="utf-8")

    required_markers = [
        "Source Documents Covered",
        "0700_Experiment_Queue_System_arch.md",
        "0701_considerations.md",
        "0702_discipline_layer.md",
        "Phase 00 - Source Freeze And Evidence Boundary",
        "Phase 01 - Queue Filesystem And Git Hygiene",
        "Phase 02 - SQLite Ledger",
        "Phase 03 - Config Ingestion",
        "Phase 04 - Stage-1 Screener",
        "Phase 05 - Full Run Executor",
        "Phase 06 - Watchdog Daemon",
        "Phase 07 - Queue CLI",
        "Phase 08 - Leaderboard",
        "Phase 09 - Discipline Gate",
        "Phase 10 - Required Fast-Run Ladder",
        "Phase 11 - Mechanism Proof",
        "Phase 12 - Failure Taxonomy",
        "Phase 13 - Combinatorial Discipline",
        "Phase 14 - Morning Workflow",
        "Phase 15 - E002 And Future Config Convention",
        "Phase 16 - TDD And QC Gates",
        "Hypothesis Document Template",
        "Full Runner Pipeline Contract",
        "Manual Full-Run Freeze Policy",
        "No Full Runs In This Working Copy",
    ]
    for marker in required_markers:
        assert marker in guide, marker


def test_queue_discipline_guide_is_linked_from_operator_docs(repo_root):
    readme = (repo_root / "README.md").read_text(encoding="utf-8")
    checklist = (repo_root / "docs" / "architecture_variant_checklist.md").read_text(encoding="utf-8")
    guide_ref = "docs/guides/experiment_queue_discipline_checklist.md"
    assert guide_ref in readme
    assert guide_ref in checklist


def test_agents_md_summarizes_agent_process(repo_root):
    agents = (repo_root / "AGENTS.md").read_text(encoding="utf-8")
    required = [
        "README.md",
        "docs/architecture_experiment_contract.md",
        "docs/architecture_variant_checklist.md",
        "docs/guides/experiment_queue_discipline_checklist.md",
        "Do not rewrite the trainer",
        "Required QC",
    ]
    for marker in required:
        assert marker in agents


def test_e001_hypothesis_docs_exist_and_validate(repo_root):
    docs_dir = repo_root / "docs" / "experiments" / "E001_cp_trilinear_attention"
    names = [
        "hypothesis_standard_30m_seed1.md",
        "hypothesis_standard_refactor_control_30m_seed1.md",
        "hypothesis_cp_bilinear_r8_30m_seed1.md",
        "hypothesis_cp_trilinear_r8_30m_seed1.md",
        "hypothesis_cp_trilinear_r8_lambda0_30m_seed1.md",
    ]
    for name in names:
        path = docs_dir / name
        assert path.exists(), name
        result = validate_hypothesis_doc(path)
        assert result.ok is True, result.missing_fields
