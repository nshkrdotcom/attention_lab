# QC Report: Tier-1 Probe Gate Scope Docs Check

- Date: 2026-07-02
- Working directory: `/home/home/p/g/n/learning/attention_lab`
- Base commit before this check: `252def739487d78ed13fbdd3b1abc7c27b3ce1bf`
- Scope: root Markdown review, gate-scope documentation, machine-readable `overall_gate_aggregation`, `highest_status` semantics, and Tier-1 artifact regeneration.

## Markdown Review

Checked root Markdown files:

```text
AGENTS.md
README.md
EXPERIMENT_STATUS_AND_TECHNICAL_NOTES.md
UPSTREAM_BUILD_NANOGPT_README.md
```

`UPSTREAM_BUILD_NANOGPT_README.md` is an upstream nanoGPT reference and did not require Attention Lab mechanism-probe updates. The active project docs were updated to explain that run-level `overall_*` gate booleans are existential over evaluated cells and that `highest_status` is the highest claim ladder threshold reached for a cell.

## Commands

| Command | Result |
| --- | --- |
| `uv run pytest tests/test_mechanism_claim_gates.py tests/test_mechanism_probe_summary.py` | Passed, 18 tests. |
| `uv run scripts/verify_tier1_mechanism_probe_suite.py --device cuda --bootstrap-samples 1000 --fdr-alpha 0.05 --batch-size 32` | Passed; regenerated and validated four Tier-1 suite artifact directories. |
| `uv run ruff check .` | Passed. |
| `uv run pytest tests/test_docs_guides.py tests/test_mechanism_claim_gates.py tests/test_mechanism_probe_summary.py tests/test_mechanism_probe_suite_cli.py` | Passed, 41 tests. |
| `uv run scripts/summarize_mechanism_probe_suite.py --output-dir reports/mechanisms/probes/E003_differential_tier1_probe_only_inventory_path --validate` | Passed. |
| `uv run scripts/summarize_mechanism_probe_suite.py --output-dir reports/mechanisms/probes/E003_differential_tier1_confirmatory_inventory_path --validate` | Passed. |
| `uv run scripts/summarize_mechanism_probe_suite.py --output-dir reports/mechanisms/probes/E004_operator_valued_tier1_probe_only_inventory_path --validate` | Passed. |
| `uv run scripts/summarize_mechanism_probe_suite.py --output-dir reports/mechanisms/probes/E004_operator_valued_tier1_confirmatory_inventory_path --validate` | Passed. |

## Artifact Status

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

No scientific evidence claim was upgraded. No training, screen run, full run, or long-running experiment was launched.
