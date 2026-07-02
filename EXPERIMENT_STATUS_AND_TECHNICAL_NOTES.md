# Experiment Status And Technical Notes

This document is the current dynamic status page for Attention Lab experiments. It is intentionally both an experiment-status ledger and a technical interpretation note. Keep `README.md` and `AGENTS.md` short by pointing to this file instead of duplicating fast-changing run state.

Last updated from pasted local run artifacts: 2026-07-01 UTC.

Latest incremental update: E001 and E002 have completed local 3000-step checkpoints for the main completed variants; E003 and E004 have rung checkpoints under `runs/screen`; mechanism backfill inventories have been regenerated; post-hoc mechanism probes now exist for E001, E002, E003 rung500, and E004 operator-valued rung500; route-index probe semantics were fixed for E002 position rotation; QC passed with `ruff` and `pytest`; latest commit recorded by the user is `ee7a9a32f11a81de2651309323c0c24a34ff196c`.

## Current Bottom Line

The project has moved from architecture/training setup into checkpoint-backed mechanism backfill and post-hoc probing.

The important current status is:

| Experiment                         | Current status                                                                                                                          | Checkpoint evidence                                                                                                     | Mechanism probe status                                                                                                                                           | Interpretation boundary                                                                                                                   |
| ---------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| E001 CP trilinear attention        | Completed local full runs for standard, CP-bilinear, and CP-trilinear; lambda-zero and standard-refactor remain incomplete/unavailable. | `runs/experiments/E001_cp_trilinear_attention/.../checkpoints/ckpt_last.pt` exists for three runs.                      | Quick probes exist for CP-bilinear and CP-trilinear.                                                                                                             | Local single-seed training result only; not a broad architecture claim.                                                                   |
| E002 Multi-QKV shift/register      | Completed local full runs for standard refactor, static global, train-rotation global, and position-rotation global.                    | `runs/experiments/E002_multitrack_qkv_shift_register/.../checkpoints/ckpt_last.pt` exists for canonical completed runs. | Quick probes exist for static, train rotation, position rotation, and position-rotation capture-only.                                                            | Route specialization workbench is now probeable; route-index semantics require discrete/continuous separation.                            |
| E003 QKV architecture gauntlet     | Screen/rung artifacts exist for differential, scope-gated, and standard controls.                                                       | `runs/screen/.../checkpoints/ckpt_last.pt` exists for rung020/rung150/rung500 candidates and controls.                  | Inventory-path rung500 probes exist for differential and scope-gated.                                                                                            | Rung survival and post-hoc probes support mechanism follow-up, not final scientific claims.                                               |
| E004 operator/binding QKV gauntlet | Screen/rung artifacts exist for operator-valued, dynamic-value, q3k3v3 rung020, and standard controls.                                  | `runs/screen/.../checkpoints/ckpt_last.pt` exists for available E004 rung artifacts.                                    | Inventory-path rung500 probe exists for operator-valued; an older partial E004 operator-valued probe also exists and is retained as historical/partial evidence. | Operator-valued is ready for deeper mechanism probing; dynamic-value needs diagnostic rescue; q3k3v3 remains profiling/redesign-oriented. |

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

runs/screen/differential_qkv_anti_value_30m_seed1_rung020_fbd4f88169c6/checkpoints/ckpt_last.pt
runs/screen/differential_qkv_anti_value_30m_seed1_rung150_56be93756c60/checkpoints/ckpt_last.pt
runs/screen/differential_qkv_anti_value_30m_seed1_rung500_407d76f5952e/checkpoints/ckpt_last.pt

runs/screen/scope_gated_qkv_30m_seed1_rung020_6cc9c5a39321/checkpoints/ckpt_last.pt
runs/screen/scope_gated_qkv_30m_seed1_rung150_3d10fefe8676/checkpoints/ckpt_last.pt
runs/screen/scope_gated_qkv_30m_seed1_rung500_bb3de557aae8/checkpoints/ckpt_last.pt

runs/screen/operator_valued_attention_30m_seed2_rung020_e16115e99fa0/checkpoints/ckpt_last.pt
runs/screen/operator_valued_attention_30m_seed2_rung150_1b42af943403/checkpoints/ckpt_last.pt
runs/screen/operator_valued_attention_30m_seed2_rung500_b6177af38f93/checkpoints/ckpt_last.pt

runs/screen/dynamic_value_query_conditioned_attention_30m_seed2_rung020_85a48338c457/checkpoints/ckpt_last.pt
runs/screen/dynamic_value_query_conditioned_attention_30m_seed2_rung150_a243dd9ecd2d/checkpoints/ckpt_last.pt
runs/screen/dynamic_value_query_conditioned_attention_30m_seed2_rung500_99b5756e77ed/checkpoints/ckpt_last.pt

runs/screen/q3k3v3_role_routed_attention_30m_seed2_rung020_e640cc594862/checkpoints/ckpt_last.pt

runs/screen/standard_refactor_control_30m_seed1_rung020_2cf14f22fc15/checkpoints/ckpt_last.pt
runs/screen/standard_refactor_control_30m_seed1_rung150_5f9f89469e9d/checkpoints/ckpt_last.pt
runs/screen/standard_refactor_control_30m_seed1_rung500_7752266a764e/checkpoints/ckpt_last.pt

runs/screen/standard_refactor_control_30m_seed2_rung020_6c6e16c6fec7/checkpoints/ckpt_last.pt
runs/screen/standard_refactor_control_30m_seed2_rung150_a844c93881ba/checkpoints/ckpt_last.pt
runs/screen/standard_refactor_control_30m_seed2_rung500_3cc31db15c20/checkpoints/ckpt_last.pt
```

The key clarification is that E003 and E004 checkpoints live under `runs/screen/...`, not under `runs/experiments/...`. Backfill correctly reports those as checkpoint-recompute available because the checkpoint files exist in the current working copy.

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

```text
reports/mechanisms/probes/E001_cp_bilinear_quick/
reports/mechanisms/probes/E001_cp_trilinear_quick/

reports/mechanisms/probes/E002_static_global_quick/
reports/mechanisms/probes/E002_train_rotation_quick/
reports/mechanisms/probes/E002_position_rotation_quick/
reports/mechanisms/probes/E002_position_rotation_capture_quick/

reports/mechanisms/probes/E003_differential_rung500_inventory_path/
reports/mechanisms/probes/E003_scope_gated_rung500_inventory_path/

reports/mechanisms/probes/E004_operator_valued_rung500/
reports/mechanisms/probes/E004_operator_valued_rung500_inventory_path/
```

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

Mechanism probe outputs exist for:

```text
E002_static_global_quick
E002_train_rotation_quick
E002_position_rotation_quick
E002_position_rotation_capture_quick
```

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

Available rung checkpoints:

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

Post-hoc mechanism probes now exist for:

```text
reports/mechanisms/probes/E003_differential_rung500_inventory_path/
reports/mechanisms/probes/E003_scope_gated_rung500_inventory_path/
```

Current best E003 candidates:

```text
differential_qkv_anti_value_30m_seed1_rung500
scope_gated_qkv_30m_seed1_rung500
```

Interpretation boundary:

```text
E003 has checkpoint-backed rung evidence and post-hoc probe artifacts.
This supports deeper mechanism investigation.
It does not yet establish that differential or scope-gated streams form semantically clean mechanisms.
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

Available rung checkpoints:

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

Historical partial probe artifact:

```text
reports/mechanisms/probes/E004_operator_valued_rung500/
```

`dynamic_value_query_conditioned_attention_30m_seed2_rung500` is classified as diagnostic rescue. It has a checkpoint, but needs gate/delta post-hoc probe and causal ablation.

`q3k3v3_role_routed_attention_30m_seed2_rung020` is classified as profiling redesign. It has a checkpoint, but the next question is throughput/profiling and role-stream redesign rather than immediate promotion.

Interpretation boundary:

```text
E004 operator-valued has checkpoint-backed rung evidence and a richer post-hoc probe.
This supports deeper operator-mode mechanism investigation.
It does not yet prove operator probabilities or operator-specific outputs are semantically clean mechanisms.
```

The next E004 question is:

```text
Do operator_probs, add/suppress/gate/transform/bind outputs, and combined operator output provide more local causal handles than standard attention or E003 stream decompositions?
```

## QC Status

The latest pasted local QC output shows:

```text
uv run ruff check .
All checks passed.

uv run pytest
388 passed, 1 skipped in 12.81s
```

Targeted tests also passed:

```text
uv run pytest tests/test_mechanism_backfill.py
4 passed

uv run pytest tests/test_mechanism_probe_cli.py
8 passed

uv run pytest tests/test_mechanism_capture_multi_qkv.py
6 passed

uv run pytest tests/test_attention_multi_qkv_global.py
33 passed
```

Latest commit recorded by the user:

```text
ee7a9a32f11a81de2651309323c0c24a34ff196c
```

Commit message:

```text
Update experiment gauntlet results and backfill mechanism inventories
```

## Evidence Boundary

The current evidence is enough to say:

```text
E001, E002, E003, and E004 now have filesystem-grounded checkpoint inventories for their available completed/screen runs.

E003/E004 available rung checkpoints are real local files under runs/screen.

Mechanism backfill reports are grounded in existing checkpoint paths.

Initial post-hoc mechanism probes have been run for selected E001/E002/E003/E004 candidates.

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
Attention Lab is using deliberately nonstandard attention architectures as interpretability instruments. The current experiments have produced trainable checkpoints and post-hoc probe artifacts that make mechanism investigation possible. The next step is not to claim architecture superiority, but to test whether these architectures produce more separable, more local, and more causally controllable mechanisms than matched standard controls.
```

## What To Do Next

1. Treat `reports/mechanisms/probes/E004_operator_valued_rung500_inventory_path/` as the canonical E004 operator-valued quick probe.
2. Keep `reports/mechanisms/probes/E004_operator_valued_rung500/` as historical/partial evidence unless it creates report ambiguity.
3. Run matched full mechanism probes for:

   * `differential_qkv_anti_value_30m_seed1_rung500`
   * `scope_gated_qkv_30m_seed1_rung500`
   * `operator_valued_attention_30m_seed2_rung500`
4. Run diagnostic rescue for:

   * `dynamic_value_query_conditioned_attention_30m_seed2_rung500`
5. Run profiling/redesign analysis for:

   * `q3k3v3_role_routed_attention_30m_seed2_rung020`
6. Run route-specialization follow-up for:

   * `multi_qkv_position_rotation_3track_global_30m_seed1`
   * `multi_qkv_static_3track_global_30m_seed1`
7. Run CP diagnostic follow-up for:

   * `cp_bilinear_r8_30m_seed1`
   * `cp_trilinear_r8_30m_seed1`
8. Defer any broad scientific claims until component-level intervention locality, feature purity, matched controls, and cross-seed or cross-run stability are measured.

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

python - <<'PY'
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
