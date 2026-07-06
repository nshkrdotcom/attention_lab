# Experiment Status And Technical Notes

This document is the current dynamic status page for Attention Lab experiments. It is intentionally both an experiment-status ledger and a technical interpretation note. Keep `README.md` and `AGENTS.md` short by pointing to this file instead of duplicating fast-changing run state.

Last reconciled with the current local filesystem: 2026-07-02 UTC. Checkpoint availability entries are `checkpoint_available_at_last_local_reconciliation` facts for this local working copy; checkpoints are generated artifacts and may be absent in a fresh clone.

Latest incremental update: Tier-1 remediation tightened deterministic task-suite validation with `content_sha256` plus built-in generator regeneration during confirmatory suite execution, records explicit `exploratory_signal`, `controlled_probe_gate_passed`, `candidate_mechanism_gate_passed`, and `highest_status` fields so controlled-probe evidence is not confused with candidate mechanism evidence, and documents that run-level `overall_*` gate booleans are existential over evaluated cells. It also marks E004 `operator_probs` capture/probe-only for patch/restoration and records the complete preset-declared random-site null pool in suite preflight metadata. The current working copy contains the E002 canonical candidate checkpoints and E003/E004 `runs/screen` rung checkpoints listed below. The committed E003/E004 Tier-1 suite report artifacts were regenerated from current local checkpoints with this source state and remain `insufficient_evidence`, not `candidate_mechanism_evidence`.

Operational update for the E003/E004 full-depth promotion batch: the promoted full-run helper now resumes from `checkpoints/ckpt_last.pt` when present and the runnable E003/E004 full-run configs checkpoint every 100 steps rather than 1000. A separate optional stall watcher, `scripts/experiments/watchdog_restart_on_stall.sh`, can capture live diagnostics and terminate a stale `train.py` process so the foreground helper resumes from the latest checkpoint. This is recovery plumbing only. It is not evidence that any E003/E004 3000-step promotion run completed; do not mark those full runs available until the expected final checkpoint and verifier/eval artifacts exist.

Latest incremental update (2026-07-06): the E003/E004 3000-step promotion batch described above did complete. Full local checkpoints now exist for `differential_qkv_anti_value_30m_seed1`, `scope_gated_qkv_30m_seed1`, `operator_valued_attention_30m_seed2`, and a newly-promoted `standard_refactor_control_30m_seed2` (E004's seed2 control had never been run past rung500 before; it was needed so `operator_valued_attention`'s full-depth checkpoint has an equally-full-depth, not rung500, matched control). Re-running the *original* rung500 Tier-1 presets against these full checkpoints does not, on its own, produce a trustworthy verdict: `canonical` in `controls.py` is hardcoded to the rung500 control path, so any different checkpoint is flagged noncanonical and hard-capped at `insufficient_evidence` regardless of the underlying statistics. New presets (`differential_full`, `operator_valued_full` in `presets.py`) were added instead, matched to the full-depth controls, so `canonical_control` resolves `True` with no override. Both re-run at full depth still return `insufficient_evidence` (`reports/mechanisms/probes/E00{3,4}_*_tier1_confirmatory_full_run_canonical/`), but now for genuine statistical reasons, not a tooling artifact. Separately, a new general-purpose (not confirmatory-hypothesis-specific) instrumentation/visualization toolkit was added — see `docs/mechanisms/spelunking_toolkit.md` for what it is and its first-pass findings across the whole quiver, including a real attn_q/attn_k instrumentation gap found and fixed in `dynamic_value_query_conditioned_attention` (it computes real causal content attention via a fused SDPA kernel but never recorded q/k at all before this).

## Current Bottom Line

The project has moved from architecture/training setup into checkpoint-backed mechanism backfill and post-hoc probing.

The important current status is:

| Experiment | Current status | Checkpoint evidence | Mechanism probe status | Interpretation boundary |
| --- | --- | --- | --- | --- |
| E001 CP trilinear attention | Completed local full runs for standard, CP-bilinear, and CP-trilinear; lambda-zero and standard-refactor remain incomplete/unavailable. | Three local full-run checkpoints exist under runs/experiments/E001_cp_trilinear_attention. | Quick probes exist for standard, CP-bilinear, and CP-trilinear. | Local single-seed training result only; not a broad architecture claim. |
| E002 Multi-QKV shift/register | Canonical first-build standard/candidate checkpoints are present locally. | Four local full-run checkpoints exist under runs/experiments/E002_multitrack_qkv_shift_register. | Quick-probe report folders exist for standard, static, train rotation, position rotation, and position-rotation capture-only. | Route-index semantics remain important; local checkpoints make follow-up possible but do not prove route specialization. |
| E003 QKV architecture gauntlet | Configs, reports, task suites, and Tier-1 report artifacts exist; Tier-1 rung500 candidate/control checkpoints were present at last local reconciliation. | Current backfill is regenerated from restored local checkpoints. | Tier-1 differential exploratory and confirmatory suite artifacts were regenerated with this remediation; confirmatory status remains `insufficient_evidence`. | Current Tier-1 local recomputation is verifier-backed for the selected rung500 candidate/control checkpoints in this working copy. |
| E004 operator/binding QKV gauntlet | Configs, reports, task suites, and Tier-1 report artifacts exist; Tier-1 rung500 candidate/control checkpoints were present at last local reconciliation. | Current backfill is regenerated from restored local checkpoints. | Tier-1 operator-valued exploratory and confirmatory suite artifacts were regenerated with this remediation; confirmatory status remains `insufficient_evidence`; `operator_probs` patch/restoration is invalid by policy. | Current Tier-1 local recomputation is verifier-backed for the selected rung500 candidate/control checkpoints in this working copy. |

## Current Checkpoint Inventory

The local checkpoint inventory includes:

```text
runs/experiments/E001_cp_trilinear_attention/cp_bilinear_r8_30m_seed1/checkpoints/ckpt_last.pt
runs/experiments/E001_cp_trilinear_attention/cp_trilinear_r8_30m_seed1/checkpoints/ckpt_last.pt
runs/experiments/E001_cp_trilinear_attention/standard_30m_seed1/checkpoints/ckpt_last.pt

runs/experiments/E002_multitrack_qkv_shift_register/multi_qkv_position_rotation_3track_global_30m_seed1/checkpoints/ckpt_last.pt
runs/experiments/E002_multitrack_qkv_shift_register/multi_qkv_static_3track_global_30m_seed1/checkpoints/ckpt_last.pt
runs/experiments/E002_multitrack_qkv_shift_register/multi_qkv_train_rotation_3track_global_30m_seed1/checkpoints/ckpt_last.pt
runs/experiments/E002_multitrack_qkv_shift_register/standard_refactor_control_30m_seed1/checkpoints/ckpt_last.pt

runs/screen/differential_qkv_anti_value_30m_seed1_rung500_407d76f5952e/checkpoints/ckpt_last.pt
runs/screen/operator_valued_attention_30m_seed2_rung500_b6177af38f93/checkpoints/ckpt_last.pt
runs/screen/standard_refactor_control_30m_seed1_rung500_7752266a764e/checkpoints/ckpt_last.pt
runs/screen/standard_refactor_control_30m_seed2_rung500_3cc31db15c20/checkpoints/ckpt_last.pt

runs/experiments/E003_qkv_architecture_gauntlet/differential_qkv_anti_value_30m_seed1/checkpoints/ckpt_last.pt
runs/experiments/E003_qkv_architecture_gauntlet/scope_gated_qkv_30m_seed1/checkpoints/ckpt_last.pt
runs/experiments/E004_operator_binding_qkv_gauntlet/operator_valued_attention_30m_seed2/checkpoints/ckpt_last.pt
runs/experiments/E004_operator_binding_qkv_gauntlet/standard_refactor_control_30m_seed2/checkpoints/ckpt_last.pt
```

Added 2026-07-06: the four full-depth (3000-step) checkpoints above. `differential_qkv_anti_value`/`scope_gated_qkv` reuse E002's existing `standard_refactor_control_30m_seed1` as their matched control (config-identical apart from metadata); `operator_valued_attention` uses the newly-promoted `standard_refactor_control_30m_seed2` above. The rung500 `runs/screen/...` checkpoints remain the record for the original gauntlet-screening pass and are not superseded, just no longer the only depth available.

Additional E003/E004 rung020/rung150 and auxiliary candidate screen checkpoints also exist under `runs/screen/`. Use the verification commands near the end of this file for the exact current list before making evidence claims.

## Mechanism Backfill Status

Backfill inventories have been regenerated for:

```text
reports/mechanisms/backfill/E001_cp_trilinear_attention/inventory.json
reports/mechanisms/backfill/E002_multitrack_qkv_shift_register/inventory.json
reports/mechanisms/backfill/E003_qkv_architecture_gauntlet/inventory.json
reports/mechanisms/backfill/E004_operator_binding_qkv_gauntlet/inventory.json
```

The cross-experiment candidate report now classifies the strongest immediate mechanism follow-ups as:

```text
Promote full mechanism run:
- differential_qkv_anti_value_30m_seed1_rung500
- scope_gated_qkv_30m_seed1_rung500
- operator_valued_attention_30m_seed2_rung500

Diagnostic rescue:
- dynamic_value_query_conditioned_attention_30m_seed2_rung500

Profiling redesign:
- q3k3v3_role_routed_attention_30m_seed2_rung020

Route specialization workbench:
- multi_qkv_position_rotation_3track_global_30m_seed1
- multi_qkv_static_3track_global_30m_seed1

CP diagnostic follow-up:
- cp_bilinear_r8_30m_seed1
- cp_trilinear_r8_30m_seed1
```

The report also correctly preserves the evidence boundary:

```text
Checkpoint availability only means post-hoc recomputation is possible.
Survival-screen pass does not establish semantic mechanism roles.
Validation-loss differences are not architecture evidence without matched controls and diagnostics.
Missing historical activations cannot be reconstructed without saved tensors.
```

## Post-Hoc Mechanism Probe Outputs

Current probe outputs exist for:

- reports/mechanisms/probes/E001_standard_quick/
- reports/mechanisms/probes/E001_cp_bilinear_quick/
- reports/mechanisms/probes/E001_cp_trilinear_quick/
- reports/mechanisms/probes/E002_standard_refactor_control_quick/
- reports/mechanisms/probes/E002_static_global_quick/
- reports/mechanisms/probes/E002_train_rotation_quick/
- reports/mechanisms/probes/E002_position_rotation_quick/
- reports/mechanisms/probes/E002_position_rotation_capture_quick/
- reports/mechanisms/probes/E003_standard_rung500_quick/
- reports/mechanisms/probes/E003_differential_rung500_inventory_path/
- reports/mechanisms/probes/E003_scope_gated_rung500_inventory_path/
- reports/mechanisms/probes/E004_standard_rung500_quick/
- reports/mechanisms/probes/E004_operator_valued_rung500/
- reports/mechanisms/probes/E004_operator_valued_rung500_inventory_path/
- reports/mechanisms/probes/E004_dynamic_value_rung500_diagnostic_rescue/
- reports/mechanisms/probes/E004_q3k3v3_rung020_quick/
- reports/mechanisms/probes/E003_differential_tier1_probe_only_inventory_path/
- reports/mechanisms/probes/E003_differential_tier1_confirmatory_inventory_path/
- reports/mechanisms/probes/E004_operator_valued_tier1_probe_only_inventory_path/
- reports/mechanisms/probes/E004_operator_valued_tier1_confirmatory_inventory_path/

The canonical E004 operator-valued rung500 probe is:

```text
reports/mechanisms/probes/E004_operator_valued_rung500_inventory_path/
```

The older folder:

```text
reports/mechanisms/probes/E004_operator_valued_rung500/
```

is retained as historical/partial evidence. It points to the same checkpoint but captured fewer sites and only ran the older zero-intervention probe. It is not a blocker, but the `_inventory_path` version should supersede it in future analysis.

## E001 Status

E001 has completed local 3000-step checkpoint artifacts for:

```text
standard_30m_seed1
cp_bilinear_r8_30m_seed1
cp_trilinear_r8_30m_seed1
```

Earlier local summaries indicated:

| Run                                   | Status from provided artifacts    |    Final val loss |          Final ppl |  Median tokens/sec | Peak VRAM allocated | Notes                                                                 |
| ------------------------------------- | --------------------------------- | ----------------: | -----------------: | -----------------: | ------------------: | --------------------------------------------------------------------- |
| `standard_30m_seed1`                  | completed to step 3000            | 4.076830863952637 | 58.958326507329176 | 109427.64995743861 | 3240.92431640625 MB | Main standard-attention E001 baseline.                                |
| `cp_bilinear_r8_30m_seed1`            | completed to step 3000            | 4.086292743682861 |  59.51883063145288 |  33625.54559587996 | 4443.17626953125 MB | Active CP branch; worse loss than standard and much slower.           |
| `cp_trilinear_r8_30m_seed1`           | completed to step 3000            | 4.062333106994629 |  58.10972926077219 | 16761.225878491987 | 6772.76513671875 MB | Best loss among the three, but very slow and high VRAM.               |
| `cp_trilinear_r8_lambda0_30m_seed1`   | incomplete / failed / interrupted |           unknown |            unknown |            unknown |             unknown | Directory exists but required artifacts are unavailable.              |
| `standard_refactor_control_30m_seed1` | unavailable for E001              |           unknown |            unknown |            unknown |             unknown | Config exists but no checkpoint is available in the current backfill. |

### E001 Technical Interpretation

At face value, `cp_trilinear_r8_30m_seed1` had the best final validation loss in the earlier local summaries:

```text
standard final_val_loss      = 4.076830863952637
cp_bilinear final_val_loss   = 4.086292743682861
cp_trilinear final_val_loss  = 4.062333106994629
```

Relative to standard, CP-trilinear improved final validation loss by about `0.01450`, while CP-bilinear was worse than standard by about `0.00946`.

This is directionally interesting because the value-conditioned trilinear branch beat both the plain standard baseline and the CP-bilinear low-rank score-capacity control in this one seed.

This is not enough for a broad claim. The correct claim shape remains:

```text
In one local ~30M GPT / FineWeb-Edu 100M E001 run, CP-trilinear reached a lower final validation loss than both standard attention and CP-bilinear, but at severe throughput and VRAM cost. This warrants replication and mechanism follow-up, not an architecture superiority claim.
```

### Throughput and VRAM

The performance cost is large:

```text
standard median_tokens_per_sec      = 109427.65
cp_bilinear median_tokens_per_sec   = 33625.55
cp_trilinear median_tokens_per_sec  = 16761.23

standard peak_vram_allocated_mb     = 3240.92
cp_bilinear peak_vram_allocated_mb  = 4443.18
cp_trilinear peak_vram_allocated_mb = 6772.77
```

Approximate ratios:

| Run          | Speed vs standard | VRAM vs standard |
| ------------ | ----------------: | ---------------: |
| CP-bilinear  |            0.307x |           1.371x |
| CP-trilinear |            0.153x |           2.090x |

So even if the CP-trilinear loss signal survives replication, the current implementation is not an efficiency result. It is an architecture/mechanism probe result at best.

### Mechanism Activity

The CP branches are not dead. Earlier diagnostics showed nonzero `cp_gradient_norm` and nonzero CP-score statistics for both CP-bilinear and CP-trilinear across training. At step 3000, CP-bilinear had nonzero `cp_gradient_norm` in all six layers, and CP-trilinear also had nonzero `cp_gradient_norm` in all six layers.

That supports the narrow statement that the branches were active and trainable. It does not prove that the branch caused the loss difference.

### CP-Bilinear Control

The CP-bilinear run is important because it controls for extra low-rank score capacity without value-conditioned trilinear structure. Since CP-bilinear was worse than standard while CP-trilinear was better than standard in the current summaries, the first-order result is not simply “adding a CP score branch helps.”

The interesting hypothesis is narrower:

```text
The value-conditioned trilinear score branch may be doing something different from the bilinear low-rank score-capacity control.
```

That hypothesis still needs replication and the lambda-zero control.

### Lambda-Zero Control

`cp_trilinear_r8_lambda0_30m_seed1` is not interpretable yet. The directory exists, but required checkpoint/eval/sample artifacts are unavailable. Treat it as incomplete, failed, or interrupted until metrics and verifier output say otherwise.

This run matters because it distinguishes “the trilinear code path exists” from “the trilinear branch contributes nonzero score augmentation.” Without it, some implementation/wiring explanations remain open.

## E002 Status

E002 has completed local 3000-step checkpoints for the canonical first-build variants:

```text
standard_refactor_control_30m_seed1
multi_qkv_static_3track_global_30m_seed1
multi_qkv_train_rotation_3track_global_30m_seed1
multi_qkv_position_rotation_3track_global_30m_seed1
```

Mechanism probe outputs exist for standard refactor control, static, train rotation, position rotation, and position-rotation capture-only.

### Route-Index Probe Semantics

A real bug was found and fixed in the E002 position-rotation mechanism probe path.

Problem:

```text
multi_qkv_position_rotation_3track_global crashed during intervention execution because torch.gather received a non-integer route-index tensor.
```

Corrected semantics:

```text
selected_track is a discrete route/index diagnostic site.
track_q, track_k, track_v, and track_out are continuous intervention sites.
```

The fix made route indices robust at the gather boundary and made the probe CLI support separate capture sites and intervention sites. `selected_track` can be captured for diagnostics but should not be treated as a continuous activation for zero/scale intervention.

The current E002 position-rotation probe correctly captures `selected_track` while applying interventions only to continuous track sites.

### E002 Interpretation Boundary

E002 is now ready for route-specialization mechanism work, but not for a final scientific claim.

The next real E002 mechanism question is:

```text
Do the global QKV tracks specialize into distinguishable routing/content roles, and do route/track interventions produce predictable localized effects?
```

Next E002 work should focus on:

```text
route replacement
track ablation matrix
track-specific intervention locality
cross-run comparison against standard_refactor_control_30m_seed1
```

## E003 Status

E003 is now beyond `implemented_not_run`.

Implemented variants:

```text
differential_qkv_anti_value
scope_gated_qkv
standard_refactor_control
```

Rung checkpoint paths referenced by historical reports and present in this working copy include:

```text
differential_qkv_anti_value_30m_seed1_rung020
differential_qkv_anti_value_30m_seed1_rung150
differential_qkv_anti_value_30m_seed1_rung500

scope_gated_qkv_30m_seed1_rung020
scope_gated_qkv_30m_seed1_rung150
scope_gated_qkv_30m_seed1_rung500

standard_refactor_control_30m_seed1_rung020
standard_refactor_control_30m_seed1_rung150
standard_refactor_control_30m_seed1_rung500
```

Post-hoc mechanism probe report folders exist for standard control, differential, and scope-gated rung500. Current local recomputation is possible for checkpoint paths that exist on disk, subject to normal device/runtime constraints.

Tier-1 differential suite artifacts now exist:

```text
reports/mechanisms/probes/E003_differential_tier1_probe_only_inventory_path/
reports/mechanisms/probes/E003_differential_tier1_confirmatory_inventory_path/
```

The regenerated confirmatory Tier-1 report used the canonical seed1 matched control, deterministic 50-pair negation task suite with `content_sha256`, task-aligned `patch_positions_mean` pooling, and validated restoration alignment metadata. It completed as `insufficient_evidence`: random-site comparisons, matched-control comparisons, target-vs-decoy specificity, and/or corrected restoration gates did not clear the full claim ladder. This is not `candidate_mechanism_evidence`. The regenerated artifacts record explicit exploratory, controlled-probe, candidate-mechanism, and highest-status gate fields plus random-site null pool metadata.

Current best E003 candidates:

```text
differential_qkv_anti_value_30m_seed1_rung500
scope_gated_qkv_30m_seed1_rung500
```

Interpretation boundary:

```text
E003 has committed rung/probe/Tier-1 report artifacts, and the selected Tier-1 rung500 checkpoint files are present for local recomputation.
This supports follow-up planning, not a fresh local mechanism claim from the current filesystem.
It does not establish that differential or scope-gated streams form semantically clean mechanisms.
```

The next E003 question is:

```text
Do branch-delta, positive/negative streams, scope_out, gate, and content_scope_product provide more local causal handles than standard attention?
```

## E004 Status

E004 is now beyond `implemented_not_run`.

Implemented variants:

```text
operator_valued_attention
dynamic_value_query_conditioned_attention
q3k3v3_role_routed_attention
standard_refactor_control
```

Rung checkpoint paths referenced by historical reports and present in this working copy include:

```text
operator_valued_attention_30m_seed2_rung020
operator_valued_attention_30m_seed2_rung150
operator_valued_attention_30m_seed2_rung500

dynamic_value_query_conditioned_attention_30m_seed2_rung020
dynamic_value_query_conditioned_attention_30m_seed2_rung150
dynamic_value_query_conditioned_attention_30m_seed2_rung500

q3k3v3_role_routed_attention_30m_seed2_rung020

standard_refactor_control_30m_seed2_rung020
standard_refactor_control_30m_seed2_rung150
standard_refactor_control_30m_seed2_rung500
```

Current best E004 candidate:

```text
operator_valued_attention_30m_seed2_rung500
```

Canonical probe artifact:

```text
reports/mechanisms/probes/E004_operator_valued_rung500_inventory_path/
```

Tier-1 operator-valued suite artifacts:

```text
reports/mechanisms/probes/E004_operator_valued_tier1_probe_only_inventory_path/
reports/mechanisms/probes/E004_operator_valued_tier1_confirmatory_inventory_path/
```

The regenerated confirmatory Tier-1 report used the canonical seed2 matched control, deterministic 50-pair negation task suite with `content_sha256`, task-aligned `patch_positions_mean` pooling, and validated restoration alignment metadata. It completed as `insufficient_evidence`: full-width operator output sites did not clear the full probe/null/control/specificity/restoration gate family, and `operator_probs` had the expected low-dimensional random-site feasibility cap plus no matched-control site. This is not `candidate_mechanism_evidence`. E004 `operator_probs` is now explicitly capture/probe-only for patch/restoration until a validated probability-site intervention exists, and the regenerated report marks patching/mediation invalid for that site.

Historical partial probe artifact:

```text
reports/mechanisms/probes/E004_operator_valued_rung500/
```

`dynamic_value_query_conditioned_attention_30m_seed2_rung500` is classified as diagnostic rescue from historical artifact summaries. Its checkpoint is present in this working copy, so local recomputation is possible, but diagnostic-rescue status is not promotion evidence.

`q3k3v3_role_routed_attention_30m_seed2_rung020` is classified as profiling redesign from historical artifact summaries. Its checkpoint is present in this working copy, so profiling can be recomputed locally, but redesign/promotion claims still require fresh artifacts.

Interpretation boundary:

```text
E004 operator-valued has committed rung/probe/Tier-1 report artifacts, and the selected Tier-1 rung500 checkpoint files are present for local recomputation.
This supports follow-up planning, not a fresh local operator-mode mechanism claim from the current filesystem.
It does not prove operator probabilities or operator-specific outputs are semantically clean mechanisms.
```

The next E004 question is:

```text
Do operator_probs, add/suppress/gate/transform/bind outputs, and combined operator output provide more local causal handles than standard attention or E003 stream decompositions?
```

## QC Status

The latest local QC for this remediation pass shows:

```text
uv sync
Resolved 106 packages; audited 100 packages.

uv run ruff check .
All checks passed.

uv run pytest
455 passed, 1 skipped in 52.82s
```

Targeted mechanism tests passed:

```text
uv run pytest tests/test_mechanism_claim_gates.py tests/test_mechanism_controls.py \
  tests/test_mechanism_task_generation_cli.py tests/test_mechanism_probe_suite_cli.py \
  tests/test_mechanism_probe_summary.py
52 passed
```

Tier-1 preflight passed with E003/E004 candidate and matched-control checkpoints present. All four regenerated Tier-1 suite report directories validated with `scripts/summarize_mechanism_probe_suite.py --validate`. The regenerated reports still do not establish `candidate_mechanism_evidence`.

E001-E004 `validate_experiment`, FineWeb-Edu hash verification, E001-E004 `attn-queue doctor`, and the backfill checkpoint-path consistency check also passed in this reconciliation pass. Broader mechanism and queue tests were covered by the full `uv run pytest` pass above.

Previous committed status anchor before this remediation:

```text
8aa6add8b77a2c376c2194801a46d7e572dbd0ce
```

Previous anchor commit message:

```text
Update mechanism backfill inventory commit hashes
```

## Evidence Boundary

The current evidence is enough to say:

```text
E001, E002, E003, and E004 now have filesystem-grounded checkpoint inventories for the checkpoints actually present in this working copy.

E003/E004 selected Tier-1 rung500 checkpoint paths are currently available in this working copy.

Mechanism backfill reports are grounded in existing checkpoint paths.

Committed post-hoc mechanism probe and Tier-1 suite report artifacts exist for selected E001/E002/E003/E004 candidates. Current local recomputation is possible for checkpoint paths that exist on disk; generated reports should be regenerated before using them as current artifacts after source-behavior changes.

The route-index intervention bug for E002 position rotation has been fixed and covered by tests.
```

The current evidence is not enough to say:

```text
CP-trilinear is a superior architecture.

Multi-QKV tracks have proven route specialization.

Differential QKV has proven positive/negative semantic separation.

Scope-gated QKV has proven scope semantics.

Operator-valued attention has proven clean operator modes.

Dynamic-value or q3k3v3 are ready for promotion.
```

## Current Research Interpretation

The project is now in the correct phase for mechanism-first investigation.

The right claim shape is:

```text
Attention Lab is using deliberately nonstandard attention architectures as interpretability instruments. The current repository has local checkpoints for the subset listed above, plus committed reports and post-hoc probe artifacts from broader E001-E004 work. The next step is not to claim architecture superiority, but to rerun or regenerate the specific mechanism artifacts needed under the current source state and test whether these architectures produce more separable, more local, and more causally controllable mechanisms than matched standard controls.
```

## What To Do Next

The E001-E004 backfill / quick-probe artifact phase is complete. Do not rerun broad quick-probe backfill just to refresh status docs.

The Tier-1 E003/E004 mechanism-probe workflow now exists and regenerated suite report artifacts are present. The current confirmatory outcome remains `insufficient_evidence`, not `candidate_mechanism_evidence`. Future source-behavior changes should regenerate these reports again before treating them as current artifacts.

Follow-up work remains useful for scope_gated_qkv_30m_seed1_rung500, E001 CP diagnostics, E002 route specialization, dynamic-value rescue, and q3k3v3 profiling. For E003/E004 Tier-1, the next scientific investment should be improving or broadening pre-registered task contrasts only if the new task design is justified before running, then adding cross-seed replication if a future Tier-1 run clears gates.

Defer broad scientific claims until component-level intervention locality, feature purity, trained-probe evidence, statistical controls, matched controls, and cross-seed or cross-run stability are measured.

## Suggested Status Verification Commands

From the repo root:

```bash
find runs -path '*/checkpoints/ckpt_last.pt' -print | sort

jq -r '
  .candidates[]
  | select(.checkpoint_status=="available")
  | [.experiment_id, .run_name, .checkpoint_path]
  | @tsv
' reports/mechanisms/backfill/*/inventory.json | sort

uv run python - <<'PY'
import json
from pathlib import Path

root = Path(".")
bad = []

for inv_path in sorted(Path("reports/mechanisms/backfill").glob("*/inventory.json")):
    inv = json.loads(inv_path.read_text())
    for row in inv["candidates"]:
        cp = row.get("checkpoint_path")
        status = row.get("checkpoint_status")
        run_name = row.get("run_name")

        if status == "available":
            if not cp:
                bad.append((inv_path, run_name, "available_but_no_checkpoint_path", cp))
            elif not (root / cp).exists():
                bad.append((inv_path, run_name, "available_but_path_missing", cp))

        if status != "available" and cp:
            bad.append((inv_path, run_name, "unavailable_but_checkpoint_path_present", cp))

if bad:
    print("BAD CHECKPOINT STATUS:")
    for item in bad:
        print("\t".join(map(str, item)))
    raise SystemExit(1)

print("OK: every available checkpoint path exists on disk")
PY

find reports/mechanisms/probes -maxdepth 2 -type f | sort

sed -n '1,260p' reports/mechanisms/cross_experiment_candidate_report.md

uv run ruff check .
uv run pytest
```

## Status Labels Used Here

* `completed local full run`: a local experiment checkpoint exists under `runs/experiments/.../checkpoints/ckpt_last.pt` for a full configured run.
* `screen/rung checkpoint`: a gauntlet rung checkpoint exists under `runs/screen/.../checkpoints/ckpt_last.pt`.
* `checkpoint-recompute available`: a checkpoint exists and post-hoc activation capture/probing can be run from it.
* `quick probe`: a small post-hoc mechanism probe using a short prompt set and simple interventions such as zero/scale.
* `canonical probe`: the preferred current probe artifact for a candidate.
* `historical/partial probe`: older probe artifact retained for traceability but superseded by a richer current probe.
* `diagnostic rescue`: a candidate has enough artifacts to investigate but needs specific post-hoc diagnostics before promotion.
* `profiling redesign`: a candidate is interesting but blocked primarily by throughput/engineering behavior.
* `not_available`: no usable checkpoint or run artifact exists in the current working copy.
* `stale report`: a checked-in or generated report contradicts newer local artifacts.
