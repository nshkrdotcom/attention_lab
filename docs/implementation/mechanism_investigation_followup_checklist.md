# Mechanism Investigation Follow-Up Checklist

Status: implemented and validated.

This checklist hardens the initial mechanism investigation substrate after review. It
does not add new attention architectures and does not reinterpret historical runs.

## Checklist

- [x] Probe CLI exposes the full supported intervention contract.
  - Long-term fix: move probe argument validation and intervention construction into
    a reusable library module instead of embedding it in the script.
  - Required behavior: `zero`, `mean_ablate`, `scale`, `replace`, and
    `patch_from_cache` are all accepted by the CLI when their required inputs are
    present.
  - Required failure mode: `replace` without a replacement tensor or source cache,
    `patch_from_cache` without a source cache, and `scale` without `--scale` fail
    clearly before a model forward pass.
  - Tests: probe helper tests build real `InterventionSpec` objects from tensors and
    `ActivationCache` objects.

- [x] Probe tokenization is config-driven.
  - Long-term fix: derive the tokenizer from `config["data"]["tokenizer"]`, validate
    prompt token IDs against the configured vocabulary size, and record tokenizer
    metadata in probe outputs.
  - Required behavior: GPT-2 configs continue to work; unsupported tokenizers fail
    explicitly instead of silently producing IDs from the wrong vocabulary.
  - Tests: tokenizer encoding validates vocabulary bounds and rejects unsupported
    tokenizer names.

- [x] Positive candidate classifications require evidence.
  - Long-term fix: centralize evidence gating in report classification before
    architecture-specific positive categories.
  - Required behavior: `not_available` rows cannot be classified as promotion,
    diagnostic rescue, profiling redesign, route-specialization workbench, or CP
    diagnostic follow-up.
  - Tests: unavailable E001 CP and E002 route-specialization-shaped rows classify as
    `not_evaluated`.

- [x] Capture-all can report declared-but-unemitted hook sites.
  - Long-term fix: add explicit strict capture completeness reporting without changing
    default non-strict capture behavior.
  - Required behavior: `capture_activations(..., require_declared_sites=True)` reports
    unsupported declared sites and supported declared sites that were not emitted by
    the actual forward pass.
  - Tests: standard tiny models emit every declared standard site, CP rank components
    are reported as unsupported, and disabled Q3 pair-product sites are reported as
    declared but unemitted.

- [x] No-op capture equivalence is checked for every instrumented novel family.
  - Long-term fix: every live architecture-specific recorder path has a regression
    test proving capture-only forwards preserve logits in eval mode.
  - Required behavior: operator-valued, differential QKV, scope-gated QKV, Multi-QKV,
    CP, dynamic-value, and Q3K3V3 capture-only forwards match baseline forwards.
  - Tests: tiny real model forward-pass tests use real tensors and no mocks.

- [x] Backfill inventories include deterministic generation provenance.
  - Long-term fix: generated inventory JSON/Markdown records the git commit used as
    the source state and declares that paths are repo-root-relative. No timestamp is
    emitted, so regenerated inventories remain deterministic for a fixed tree.
  - Required behavior: committed derived reports state their provenance without
    overwriting historical experiment reports.
  - Tests: temporary realistic inventories contain provenance keys and deterministic
    repeated generation.

- [x] Documentation and generated reports are refreshed.
  - Long-term fix: README and mechanism implementation docs describe the hardened CLI,
    strict completeness mode, evidence-gated classification, and deterministic
    provenance.
  - Required behavior: `reports/mechanisms/backfill/` and
    `reports/mechanisms/cross_experiment_candidate_report.md` are regenerated from
    structured artifacts, not hand-edited.

- [x] QC is green before commit and push.
  - Required commands:
    - `uv sync`
    - `uv run ruff check .`
    - `uv run pytest`
    - `uv run pytest --run-integration`
    - `uv run scripts/validate_experiment.py --id E001_cp_trilinear_attention`
    - `uv run scripts/validate_experiment.py --id E002_multitrack_qkv_shift_register`
    - `uv run scripts/validate_experiment.py --id E003_qkv_architecture_gauntlet`
    - `uv run scripts/validate_experiment.py --id E004_operator_binding_qkv_gauntlet`
    - `uv run scripts/verify_data.py --data_root data/fineweb_edu_100m --manifest data/fineweb_edu_100m/manifest.json --verify_hashes`
    - `uv run attn-queue doctor --experiment E001_cp_trilinear_attention`
    - `uv run attn-queue doctor --experiment E002_multitrack_qkv_shift_register`
    - `uv run attn-queue doctor --experiment E003_qkv_architecture_gauntlet`
    - `uv run attn-queue doctor --experiment E004_operator_binding_qkv_gauntlet`
    - `uv run scripts/backfill_mechanism_inventory.py --experiments E001,E002,E003,E004`
    - `uv run scripts/compare_mechanism_candidates.py --backfill-root reports/mechanisms/backfill --output reports/mechanisms/cross_experiment_candidate_report.md`
