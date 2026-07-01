# Pre-Experiment Cleanup Checklist

Status: resolved before first novel attention implementation.

## Checklist

- [x] Add `verify_run.py --expect-data-manifest`.
  - Requires `data_manifest.json`.
  - Requires `data_manifest.sha256`.
  - Verifies the manifest digest.
  - Verifies local shard hashes when the configured data root is present.

- [x] Prevent silent eval-loss data drift.
  - `eval_loss.py` compares checkpoint/run manifest provenance against the selected
    data root manifest.
  - Cross-data evaluation requires `--allow-data-manifest-mismatch`.

- [x] Store manifest identity inside checkpoints.
  - Checkpoints now include `data_manifest` and `data_manifest_sha256` when the run has
    a data manifest.
  - Resume validation uses checkpoint-embedded manifest provenance first, then falls
    back to run-directory manifests for older checkpoints.

- [x] Reject config typos early.
  - Strict allowed-key validation covers `run`, `data`, `model`, `train`, and `sample`
    sections.

- [x] Clarify config ladder data requirements.
  - `baseline_70m_fineweb300m.yaml` and `baseline_125m_fineweb1b.yaml` are documented
    as templates requiring their data roots and manifests before use.

- [x] Clean up 124M vs 125M naming.
  - `baseline_125m_fineweb1b.yaml` is canonical.
  - `baseline_124m_fineweb1b.yaml` remains only as a historical alias.

- [x] Add accurate full-run script.
  - `scripts/run_full_30m_baseline.sh` uses the accurate `baseline_30m` naming.
  - `scripts/run_full_baseline.sh` remains a historical completed-run reproducer.

- [x] Add HellaSwag cache provenance.
  - Eval JSON records data path, source URL, and SHA256 of the cached JSONL file.

## Still Deferred

- HF export remains an honest stub.
- `lm-evaluation-harness` remains deferred until HF export is implemented and verified.
- `trilinear_cp` remains intentionally unimplemented.
