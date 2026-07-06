# Instrumentation/Visualization Spelunking Toolkit

This document covers a different kind of tool than `mechanism_probe_framework.md`.
Tier-1 answers a narrow, pre-registered question (does a specific candidate
show statistically controlled evidence of a specific claimed mechanism,
against a specific matched control). This toolkit is the opposite: **general
exploratory instrumentation**, for looking at what a checkpoint is actually
doing before you've committed to a specific hypothesis. Not a generic
probing/SAE platform, not a claim-gate system, not a replacement for Tier-1
— a earlier step that can motivate what a later Tier-1 suite should even
test.

## What it is

- `attn_weights[layer]` — a real hook site, now recorded directly at the
  point of computation for every architecture that computes attention
  unfused (CP family, multi-QKV family, and now `dynamic_value_query_
  conditioned_attention`'s content-attention branch). Preferred over
  external reconstruction wherever possible, since it removes an entire
  class of formula-transcription risk.
- `src/attention_lab/mechanisms/attention_reconstruction.py` — external
  reconstruction (`softmax(q @ k^T / sqrt(head_dim))`, causal) for the one
  case where a real hook isn't possible: `standard` attention's fused
  `F.scaled_dot_product_attention` kernel never exposes an intermediate.
  Safe here because there's no extra branch to reconstruct incorrectly.
- `src/attention_lab/mechanisms/synthetic_prompts.py` — `build_induction_
  probe`: a random token pattern repeated exactly once (the classic Olsson
  et al. 2022 behavioral induction-head test), plus `induction_accuracy`
  scoring. No tokenizer or real text needed.
- `src/attention_lab/mechanisms/visualize.py` — `plot_attention_heatmap`
  (per-head attention matrix as an image) and `plot_track_selection_
  histogram` (routed-track distribution for multi-QKV variants).
- `scripts/spelunk_checkpoint.py` — ties all of the above together: load a
  checkpoint, run either a real prompt or a synthetic induction probe,
  capture/reconstruct attention weights per layer, render heatmaps (+
  track histograms where applicable), and write a `spelunk_summary.json`.

```bash
uv run python scripts/spelunk_checkpoint.py \
  --config CONFIG.yaml --checkpoint CKPT.pt --output-dir OUT/ \
  --induction-probe-pattern-len 15 --device cpu
# or, for a real-text prompt instead of the synthetic probe:
  --prompt "The history of mathematics"
```

## Findings (2026-07-06, first pass)

Multi-trial induction-probe sweep (5 seeds, `pattern_len=20`, `--device
cpu`) across every checkpoint that currently loads, including the two
never-fully-promoted E004 architectures (using their best available
rung-level checkpoint, flagged as such):

| checkpoint | mean accuracy (5 seeds) | individual |
|---|---:|---|
| `standard` (E001) | 51.6% | 0.47, 0.68, 0.53, 0.63, 0.26 |
| `cp_bilinear` | 38.9% | 0.63, 0.37, 0.26, 0.47, 0.21 |
| `cp_trilinear` | 48.4% | 0.47, 0.74, 0.42, 0.47, 0.32 |
| `standard_refactor_control` (E002 seed1) | 43.2% | 0.42, 0.58, 0.37, 0.42, 0.37 |
| `multi_qkv_static` | 24.2% | 0.32, 0.42, 0.16, 0.26, 0.05 |
| `multi_qkv_train_rotation` | **0.0%** | 0, 0, 0, 0, 0 (exact, every seed) |
| `multi_qkv_position_rotation` | 23.2% | 0.26, 0.37, 0.16, 0.32, 0.05 |
| `differential_qkv_anti_value` (E003, full 3000-step) | 38.9% | 0.32, 0.42, 0.37, 0.58, 0.26 |
| `scope_gated_qkv` (E003, full 3000-step) | 42.1% | 0.21, 0.63, 0.42, 0.58, 0.26 |
| `operator_valued_attention` (E004, full 3000-step) | 33.7% | 0.11, 0.53, 0.21, 0.37, 0.47 |
| `standard_refactor_control` (E004 seed2, full 3000-step) | 48.4% | 0.53, 0.58, 0.37, 0.68, 0.26 |
| `dynamic_value_query_conditioned_attention` (rung500 only) | **0.0%** | 0, 0, 0, 0, 0 (exact) |
| `q3k3v3_role_routed_attention` (rung020 only) | **0.0%** | 0, 0, 0, 0, 0 (exact) |

**First observation, worth calibrating expectations against: nothing here
shows strong, clean induction behavior.** Even `standard` attention only
reaches ~52% mean accuracy with high seed-to-seed variance (0.26-0.68).
This is not the "textbook clean" induction-head phenomenon often described
in larger models — at 30M params / 3000 steps on FineWeb-Edu-100M,
induction formation is present but partial and noisy, not dominant. Worth
remembering before treating any single number in this table as a strong
signal.

**A real dissociation between "what a head's attention pattern looks like"
and "aggregate next-token accuracy":** visualizing `standard`'s layer 3
heads on an induction probe shows heads 0, 3, and 4 with a clear
offset-diagonal stripe at exactly the "attend to the position the same
token last appeared" location — the textbook induction-head signature.
Heads 1, 2, 5 look like ordinary local/recency attention instead. Yet the
*aggregate* logit-level accuracy for this checkpoint is only ~52%. Some
heads clearly specialize toward induction; that specialization isn't
strong enough (or isn't being read out cleanly enough downstream) to
dominate the final prediction. This is exactly the kind of thing a
single scalar accuracy number would hide and a real visualization catches.

**The most striking single number: `multi_qkv_train_rotation` scores an
exact 0.0% across all five seeds — not just weak, a total, flat failure —
markedly worse than `multi_qkv_static`'s 24.2% and `position_rotation`'s
23.2%.** This is surprising given a real, verified fact:
`multi_qkv_static.py` and `multi_qkv_train_rotation.py`'s `select_scalar_
track` both reduce to the *identical* formula at eval time
(`layer_idx % track_count` — confirmed by reading both source files
directly, not assumed). The routing decision itself is the same at
deployment; the training-time difference is that `train_rotation`'s track
assignment *rotated* across training steps (`(layer_idx + step) %
track_count`) while `static`'s was fixed for the entire run. The most
likely explanation — not yet proven, a real next question — is that
`train_rotation`'s per-track Q/K/V weights ended up less specialized
(each track was trained under multiple shifting roles rather than one
consistent one), leaving weaker induction-relevant structure even though
the final routing decision looks identical to `static`'s. Worth a deeper
look, not a settled conclusion.

**`dynamic_value_query_conditioned_attention` investigation (this was
flagged `insufficient_evidence`/"degenerate mechanism diagnostics" by the
original gauntlet screening — the E004 report itself cautioned this might
be an instrumentation problem, not a real dead end, and recommended a
"diagnostic rescue"):**
- Its content-attention branch (`F.scaled_dot_product_attention` over real
  Q/K/V) was **never actually recorded at all** before this session —
  `attn_q[layer]`/`attn_k[layer]` were declared hook sites but silently
  never emitted, a pure instrumentation gap, not an architectural one.
  Fixed the same way Phase 1 fixed CP/multi-QKV (recording q/k/v via
  `activation_recorder.record(...)`, verified not to change model output).
- With that fixed, the content-attention pattern is now visible for the
  first time, and it is **not degenerate in any obvious visual sense**:
  heads 1 and 3 show a real, structured "landmark" pattern (attending
  consistently to one or two fixed key positions once the query passes
  them), other heads show smoother local/recency attention. Not collapsed,
  not uniform, not nonsensical.
- The gate values themselves (`dynamic_gate[layer]`, the actual signal the
  gauntlet's diagnostics complained about) also don't show an obvious
  collapse: per-layer means 0.26-0.45, stds 0.15-0.21, spanning a real
  range from ~0.01 to ~0.96-0.98 — not saturated at 0 or 1.
- **What this does and doesn't establish:** this is consistent with the
  E004 report's own hedge that this checkpoint isn't a proven dead end —
  real structure exists in both the content attention and the gate values.
  It does *not* prove the original "degenerate" verdict was wrong, since
  this pass didn't replicate the exact automated check that produced that
  verdict (which may use a more specific criterion than aggregate
  mean/std). The honest state: a diagnostic rescue looks more promising
  after this look than before it, not yet confirmed as workable.
- Its induction accuracy (0.0% across all 5 seeds) is real but expected
  given the above — a checkpoint that's only reached rung500 (500 of 3000
  planned steps) shouldn't be expected to show strong induction regardless
  of architecture.

**`q3k3v3_role_routed_attention`'s 0.0% is not informative about the
architecture at all** — its only available checkpoint is rung020 (20
training steps, essentially untrained), blocked from further gauntlet
progress by a throughput gate, not a correctness one (see
`EXPERIMENT_STATUS_AND_TECHNICAL_NOTES.md` / the E004 report for the full
account). Zero induction accuracy here reflects "barely trained," not a
statement about the architecture's ceiling.

## What's deliberately not covered here

- Per-architecture diagnostic visualizations beyond attention heatmaps and
  track-selection histograms (e.g. a dedicated `operator_probs[layer]`
  distribution plot for E004's `operator_valued_attention`) — not built
  this pass, a natural extension of the same `visualize.py` module.
- A rigorous, confirmatory-grade version of any of the above (matched
  controls, nulls, FDR correction, replication across more seeds) — this
  is exploratory instrumentation, the whole point is that it doesn't
  commit to a hypothesis in advance. If something here looks worth a real
  claim, it graduates to a proper Tier-1-style design next, the same way
  the negation-scope hypothesis did for `differential_qkv_anti_value` and
  `operator_valued_attention`.
