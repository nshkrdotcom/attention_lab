# QC Report: Tier-1 Probe Schema Refinement

- Date: 2026-07-02
- Working directory: `/home/home/p/g/n/learning/attention_lab`
- Base commit before this remediation: `9bdf0f7b6e25f577688be7537da546ee11b45317`
- Scope: claim-gate schema refinement, confirmatory built-in task-suite regeneration enforcement, documentation/status updates, and Tier-1 artifact regeneration.

## Commands

| Command | Result |
| --- | --- |
| `uv run pytest tests/test_mechanism_claim_gates.py tests/test_mechanism_probe_suite_cli.py::test_confirmatory_builtin_suite_regeneration_fails_even_if_hash_recomputed tests/test_mechanism_probe_summary.py` | Passed, 19 tests. |
| `uv run pytest tests/test_mechanism_claim_gates.py tests/test_mechanism_probe_suite_cli.py tests/test_mechanism_probe_summary.py tests/test_mechanism_task_generation_cli.py` | Passed, 42 tests. |
| `uv run scripts/verify_tier1_mechanism_probe_suite.py --preflight-only` | Passed; E003/E004 candidate and matched-control checkpoints existed locally. |
| `uv run scripts/verify_tier1_mechanism_probe_suite.py --device cuda --bootstrap-samples 1000 --fdr-alpha 0.05 --batch-size 32` | Passed; regenerated and validated four Tier-1 suite artifact directories. |
| `uv run ruff check .` | Passed. |
| `uv run pytest` | Passed, 456 passed, 1 skipped. |
| `uv sync` | Passed. |
| `uv run scripts/validate_experiment.py --id E001_cp_trilinear_attention` | Passed, `ok: True`. |
| `uv run scripts/validate_experiment.py --id E002_multitrack_qkv_shift_register` | Passed, `ok: True`. |
| `uv run scripts/validate_experiment.py --id E003_qkv_architecture_gauntlet` | Passed, `ok: True`. |
| `uv run scripts/validate_experiment.py --id E004_operator_binding_qkv_gauntlet` | Passed, `ok: True`. |
| `uv run pytest tests/test_mechanism_linear_probe.py tests/test_mechanism_controls.py tests/test_mechanism_claim_gates.py tests/test_mechanism_probe_suite_cli.py tests/test_mechanism_probe_summary.py` | Passed, 53 tests. |
| `uv run scripts/verify_data.py --data_root data/fineweb_edu_100m --manifest data/fineweb_edu_100m/manifest.json --verify_hashes` | Passed; manifest verified. |
| `uv run attn-queue doctor --experiment E001_cp_trilinear_attention` | Passed. |
| `uv run attn-queue doctor --experiment E002_multitrack_qkv_shift_register` | Passed. |
| `uv run attn-queue doctor --experiment E003_qkv_architecture_gauntlet` | Passed. |
| `uv run attn-queue doctor --experiment E004_operator_binding_qkv_gauntlet` | Passed. |
| `uv run scripts/summarize_mechanism_probe_suite.py --output-dir reports/mechanisms/probes/E003_differential_tier1_probe_only_inventory_path --validate` | Passed. |
| `uv run scripts/summarize_mechanism_probe_suite.py --output-dir reports/mechanisms/probes/E003_differential_tier1_confirmatory_inventory_path --validate` | Passed. |
| `uv run scripts/summarize_mechanism_probe_suite.py --output-dir reports/mechanisms/probes/E004_operator_valued_tier1_probe_only_inventory_path --validate` | Passed. |
| `uv run scripts/summarize_mechanism_probe_suite.py --output-dir reports/mechanisms/probes/E004_operator_valued_tier1_confirmatory_inventory_path --validate` | Passed. |
| Backfill checkpoint consistency Python check from `AGENTS.md` | Passed; every available checkpoint path exists on disk. |

## Artifact Results

Regenerated paths:

```text
reports/mechanisms/probes/E003_differential_tier1_probe_only_inventory_path/
reports/mechanisms/probes/E003_differential_tier1_confirmatory_inventory_path/
reports/mechanisms/probes/E004_operator_valued_tier1_probe_only_inventory_path/
reports/mechanisms/probes/E004_operator_valued_tier1_confirmatory_inventory_path/
```

Observed gate status after regeneration:

```text
E003 confirmatory: insufficient_evidence
E003 probe-only: exploratory_probe_signal
E004 confirmatory: insufficient_evidence
E004 probe-only: exploratory_probe_signal
```

No scientific evidence claim was upgraded. No training, screen run, full run, or new long-running experiment was launched. The regenerated Tier-1 artifacts were produced by the mechanism probe suite against existing local checkpoints.
