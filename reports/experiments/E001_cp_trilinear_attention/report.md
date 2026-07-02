# Technical Report: E001 CP-Bilinear and CP-Trilinear Attention Experiment

## 1. Summary

E001 evaluates whether a GPT-style causal language model can tolerate explicit low-rank interaction structure inside the attention block during real pretraining.

The experiment changes the internal geometry of attention. Standard attention uses Q/K dot-product compatibility to produce weights over V. E001 introduces CP-structured interaction paths intended to create inspectable, rank-indexed mechanism handles inside the attention computation.

The completed E001 runs are:

```text
1. standard_30m_seed1
2. cp_bilinear_r8_30m_seed1
3. cp_trilinear_r8_30m_seed1
```

A fourth configuration exists:

```text
4. cp_trilinear_r8_lambda0_30m_seed1
```

but it did not complete in the uploaded dump. It only shows a step-0 validation event, so it should be treated as configured but not evaluated.

The completed E001 summaries show that standard, CP-bilinear, and CP-trilinear each reached 3000 steps with 301 train events, 13 validation events, and 3 checkpoints.

The main result is:

```text
A 30M-scale GPT-style model can train with CP-bilinear and CP-trilinear attention modifications without immediate collapse.
```

CP-bilinear remained close to the standard run but was substantially slower. CP-trilinear also trained stably and slightly outperformed the standard control on validation loss in this single seed, while being much more expensive in throughput and VRAM.

---

## 2. Research Objective

E001 asks whether attention-side interaction structure can be made explicit and stable enough to study.

Standard attention has the structure:

```text
Q/K compatibility -> attention weights -> weighted read from V
```

E001 modifies that by adding low-rank CP-structured interaction terms. The purpose is to create internal components that can later be inspected, ablated, compared, and tracked during training.

The target object is not just an end-to-end validation loss curve. The target object is the CP interaction path itself:

```text
Can rank-limited attention interaction components become live, causal, separable mechanism handles?
```

The experiment therefore has three practical goals:

```text
1. Confirm that CP-structured attention variants train from scratch.
2. Compare their stability, loss, speed, and VRAM against a standard control.
3. Identify whether the CP paths are worth deeper diagnostics and causal interventions.
```

---

## 3. Experimental Setup

The E001 runs use a common 30M-scale GPT-style setup.

Shared configuration shape:

```text
data_root: data/fineweb_edu_100m
tokenizer: gpt2
vocab_size: 50304
train_tokens: 100000000
val_tokens: 4000000

block_size: 1024
n_layer: 6
n_head: 6
n_embd: 384
dropout: 0.0
bias: false

device: cuda
dtype: bfloat16
compile: false
eval_at_start: true
B: 4
T: 1024
total_batch_size: 262144
max_steps: 3000
grad_clip: 1.0
weight_decay: 0.1
learning_rate: 0.0006
min_lr: 0.00006
warmup_steps: 100
val_every: 250
val_steps: 20
save_every: 1000
log_every: 10
```

The standard E001 config records the baseline model shape and training schedule.

The CP-bilinear config uses `attention_type: cp_bilinear`, `cp_rank: 8`, `cp_lambda_init: 0.0`, `cp_lambda_trainable: true`, and `cp_lambda_fixed: false`.

The completed CP-trilinear config uses `attention_type: cp_trilinear`, `cp_rank: 8`, `cp_lambda_init: 0.0`, `cp_lambda_trainable: true`, and `cp_lambda_fixed: false`.

The lambda-fixed CP-trilinear config uses the same CP-trilinear rank-8 structure, but sets `cp_lambda_trainable: false` and `cp_lambda_fixed: true`.

---

## 4. Architecture 0: Standard Attention Control

The standard control is ordinary GPT-style causal self-attention.

```text
Input residual stream
x: [batch, seq, d_model]
        |
        v
Linear projection: c_attn
        |
        v
+-----------------------------+
| packed QKV = [Q | K | V]    |
+-----------------------------+
        |          |          |
        v          v          v
        Q          K          V
        |          |          |
        +----------+----------+
                   |
                   v
        causal scaled dot-product attention
        Attention(Q, K, V)
                   |
                   v
        attention output
        [batch, seq, d_model]
                   |
                   v
        output projection: c_proj
                   |
                   v
        residual write
```

Token-level view:

```text
For receiver token i:

Q_i compares with K_j for all j <= i
        |
        v
attention weights over previous tokens
        |
        v
weighted sum of V_j
        |
        v
write back into residual stream at token i
```

Conceptual roles:

```text
Q = what the receiver token searches for
K = what source tokens advertise
V = what source tokens provide if attended to
```

This control establishes the baseline behavior for training stability, loss, throughput, and VRAM.

---

## 5. Architecture 1: CP-Bilinear Rank-8 Attention

CP-bilinear attention adds an explicit low-rank bilinear interaction structure to the attention computation.

The tested configuration:

```text
attention_type: cp_bilinear
cp_rank: 8
cp_lambda_init: 0.0
cp_lambda_trainable: true
cp_lambda_fixed: false
```

The config confirms the rank-8 CP-bilinear setup and trainable lambda.

### 5.1 High-Level Structure

Standard attention computes compatibility through Q/K interaction and then reads V.

CP-bilinear adds a rank-limited interaction path over Q and K:

```text
standard path:
    Q/K dot-product compatibility

CP path:
    rank-8 bilinear Q/K interaction

combined:
    ordinary attention score + lambda * CP-bilinear score
```

Architecture diagram:

```text
Input residual stream
x: [batch, seq, d_model]
        |
        v
Linear projection: c_attn
        |
        v
+-----------------------------+
| packed QKV = [Q | K | V]    |
+-----------------------------+
        |          |          |
        v          v          v
        Q          K          V
        |          |
        |          +-----------------------------+
        |                                        |
        v                                        v
ordinary QK compatibility             CP-bilinear interaction
Q_i · K_j                             rank-8 structured term
        |                                        |
        +----------------+-----------------------+
                         |
                         v
              combined attention scores
                         |
                         v
              causal mask + softmax
                         |
                         v
              attention weights
                         |
                         v
              weighted sum of V
                         |
                         v
              output projection: c_proj
                         |
                         v
              residual write
```

### 5.2 Token-Level View

For receiver token `i` and source token `j <= i`:

```text
ordinary_score_ij = Q_i · K_j

cp_score_ij = rank_8_bilinear_interaction(Q_i, K_j)

combined_score_ij = ordinary_score_ij + lambda * cp_score_ij

attention_weights_i = softmax(masked combined_score_i)

output_i = sum_j attention_weights_ij * V_j
```

This makes Q/K compatibility partly decomposable into explicit low-rank components.

### 5.3 Research Meaning

CP-bilinear is useful because it gives the attention block a structured compatibility basis.

Possible mechanism questions:

```text
Do individual CP rank components specialize?
Do CP factors align with token classes, position patterns, syntax, induction-like behavior, or modifier/scope relations?
Does ablating a rank component selectively damage specific behaviors?
Does lambda grow from zero, stay small, or become functionally significant?
```

The important object is the CP interaction basis, not the aggregate model score alone.

---

## 6. Architecture 2: CP-Trilinear Rank-8 Attention

CP-trilinear is the more ambitious E001 variant. It extends the low-rank interaction idea beyond Q/K compatibility and into the Q/K/V read-write pathway.

The completed tested configuration:

```text
attention_type: cp_trilinear
cp_rank: 8
cp_lambda_init: 0.0
cp_lambda_trainable: true
cp_lambda_fixed: false
```

The config confirms `cp_trilinear`, rank 8, and trainable lambda.

### 6.1 High-Level Structure

Standard attention separates compatibility and readout:

```text
Q and K determine where to read.
V determines what is read.
```

CP-trilinear introduces a structured interaction path involving the Q, K, and value/content side.

Architecture-level diagram:

```text
Input residual stream
x: [batch, seq, d_model]
        |
        v
Linear projection: c_attn
        |
        v
+-----------------------------+
| packed QKV = [Q | K | V]    |
+-----------------------------+
        |          |          |
        v          v          v
        Q          K          V
        |          |          |
        |          |          +----------------------+
        |          |                                 |
        |          +-------------------+             |
        |                              |             |
        v                              v             v
ordinary attention path        CP-trilinear interaction path
Attention(Q,K,V)               rank-8 Q/K/V structured term
        |                              |
        |                              v
        |                    lambda-scaled CP contribution
        |                              |
        +---------------+--------------+
                        |
                        v
              combined attention output
                        |
                        v
              output projection: c_proj
                        |
                        v
              residual write
```

### 6.2 Token-Level View

For receiver token `i`:

```text
ordinary path:
    Q_i compares with K_j
    attention weights select V_j
    ordinary_output_i = sum_j weights_ij * V_j

CP-trilinear path:
    Q_i, K_j, and V_j/content-side factors participate
    in a rank-8 structured interaction

combined output:
    output_i = ordinary_output_i + lambda * cp_trilinear_output_i
```

The exact tensor contraction should be documented from the implementation before formal publication, but the architectural distinction is:

```text
CP-bilinear:
    low-rank structure over Q/K compatibility

CP-trilinear:
    low-rank structure over Q/K/V interaction
```

### 6.3 Research Meaning

CP-trilinear is interesting because it gives the model a structured basis over the attention read/write pathway.

Possible mechanism questions:

```text
Do CP components become relation-like read/write operators?
Does a rank component specialize to copying, suppression, positional routing, modifier handling, or induction-like behavior?
Can individual CP rank components be ablated causally?
Does the CP path matter uniformly across layers, or only in specific depths?
Does lambda become a real learned contribution or remain near-dormant?
```

The CP-trilinear variant is therefore a candidate mechanism object for studying explicit low-rank attention-side computation.

---

## 7. Architecture 3: CP-Trilinear Lambda0 Fixed Control

The lambda0 fixed control is configured but not completed in the dump.

Configuration:

```text
attention_type: cp_trilinear
cp_rank: 8
cp_lambda_init: 0.0
cp_lambda_trainable: false
cp_lambda_fixed: true
```

The config explicitly sets lambda as fixed and non-trainable.

### 7.1 Intended Purpose

This control asks:

```text
If the CP-trilinear machinery is present but its contribution is fixed at zero, does the model behave like the standard baseline?
```

Dataflow:

```text
Input residual stream
x: [batch, seq, d_model]
        |
        v
[Q | K | V]
        |
        +-----------------------------+
        |                             |
        v                             v
ordinary attention path        CP-trilinear path
Attention(Q,K,V)               rank-8 structured term
        |                             |
        |                             v
        |                      lambda * CP term
        |                             |
        |                      lambda = 0 fixed
        |                             |
        +-------------+---------------+
                      |
                      v
            ordinary output only
                      |
                      v
            output projection: c_proj
                      |
                      v
            residual write
```

### 7.2 Status

The dump only shows a step-0 validation event for this run:

```text
step: 0
val_loss: 10.8903
val_perplexity: 53653.89
```

That is not a completed training result.

Correct status:

```text
configured but not completed in this dump
```

This run remains important, because it would isolate whether the active trainable CP path matters beyond mere parameterization.

---

## 8. Training Results

| Run                               | Final val loss | Final perplexity | Median tok/s | Peak VRAM | Checkpoints |
| --------------------------------- | -------------: | ---------------: | -----------: | --------: | ----------: |
| standard_30m_seed1                |         4.0768 |            58.96 |      109,428 |   3.24 GB |           3 |
| cp_bilinear_r8_30m_seed1          |         4.0863 |            59.52 |       33,626 |   4.44 GB |           3 |
| cp_trilinear_r8_30m_seed1         |         4.0623 |            58.11 |       16,761 |   6.77 GB |           3 |
| cp_trilinear_r8_lambda0_30m_seed1 |     incomplete |       incomplete |            — |         — |           — |

The completed run summaries report final validation loss, perplexity, throughput, VRAM, and checkpoint counts for standard, CP-bilinear, and CP-trilinear.

### 8.1 Standard Control

```text
final val loss:       4.0768
final perplexity:     58.96
median tokens/sec:    109,428
peak VRAM:            3.24 GB
checkpoint count:     3
```

The standard control provides the baseline for this experiment.

### 8.2 CP-Bilinear

```text
final val loss:       4.0863
final perplexity:     59.52
median tokens/sec:    33,626
peak VRAM:            4.44 GB
checkpoint count:     3
```

CP-bilinear completed training and stayed close to the standard control in validation loss. It was substantially slower and used more VRAM.

Approximate differences relative to standard:

```text
val loss delta:       +0.0095
throughput ratio:     ~0.31x standard
VRAM ratio:           ~1.37x standard
```

### 8.3 CP-Trilinear

```text
final val loss:       4.0623
final perplexity:     58.11
median tokens/sec:    16,761
peak VRAM:            6.77 GB
checkpoint count:     3
```

CP-trilinear completed training and slightly beat the standard control on final validation loss in this single seed. It was much slower and used much more VRAM.

Approximate differences relative to standard:

```text
val loss delta:       -0.0145
throughput ratio:     ~0.15x standard
VRAM ratio:           ~2.09x standard
```

---

## 9. HellaSwag Sanity Eval

The HellaSwag eval used only 100 validation examples, so it should be interpreted as a coarse sanity check rather than a ranking.

Results:

```text
standard:      0.32
CP-bilinear:   0.28
CP-trilinear:  0.32
```

The dump records CP-bilinear at 28/100, CP-trilinear at 32/100, and standard at 32/100.

Interpretation:

```text
The variants are not obviously broken.
The sample is too small to infer meaningful downstream capability differences.
```

The main E001 evidence is training survival plus the existence of CP-structured attention variants that remain close enough to standard learning dynamics to warrant mechanism diagnostics.

---

## 10. Interpretation by Variant

### 10.1 Standard Control

The standard run confirms the harness and baseline behavior.

It establishes:

```text
normal training loss descent
normal throughput
normal VRAM range
ordinary attention reference point
```

The E001 standard and E002 standard-refactor runs are also close enough to suggest the baseline training setup is stable across experiment families.

---

### 10.2 CP-Bilinear

CP-bilinear is a stable low-rank Q/K interaction object.

The result says:

```text
Adding rank-8 bilinear Q/K structure does not collapse training.
```

It does not yet say whether the CP rank components are semantically meaningful.

The architectural significance is that Q/K compatibility now contains an explicit low-rank factorized path. That path can be inspected directly.

Potentially relevant future measurements:

```text
1. learned lambda trajectory
2. CP contribution norm vs ordinary attention score norm
3. CP rank-component norms
4. CP rank-component cosine similarity
5. component ablation by rank index
6. per-layer CP activity
7. Q-side and K-side factor specialization
```

The key question is:

```text
Do the CP-bilinear factors become meaningful compatibility features?
```

---

### 10.3 CP-Trilinear

CP-trilinear is a stable low-rank Q/K/V interaction object.

The result says:

```text
Adding rank-8 trilinear structure over the attention read/write pathway does not collapse training.
```

It is more expensive than CP-bilinear, but it is also the more interesting E001 specimen because it exposes a richer interaction surface.

The slight validation-loss advantage over standard is not the central point. The central point is that the CP-trilinear path survived and remained compatible with real training.

Potentially relevant future measurements:

```text
1. learned lambda trajectory
2. CP output norm vs ordinary attention output norm
3. CP rank-component ablations
4. layer-wise CP activity
5. factor similarity across rank components
6. factor similarity across layers
7. CP-only vs standard-only output decomposition
8. whether CP contribution correlates with token role, position, modifier/scope, or long-range dependency behavior
```

The key question is:

```text
Do CP-trilinear rank components become interpretable read/write interaction operators?
```

---

### 10.4 CP-Trilinear Lambda0 Fixed

This run remains necessary.

Without it, E001 cannot fully separate:

```text
effect of active CP contribution
```

from:

```text
effect of adding CP scaffolding / parameterization / implementation changes
```

The lambda0 fixed control should complete the same 3000-step schedule before any stronger interpretation of CP-trilinear’s behavior.

Expected diagnostic role:

```text
If trainable CP-trilinear differs from lambda0-fixed:
    the active CP path likely matters.

If trainable CP-trilinear matches lambda0-fixed:
    the CP path may be dormant or irrelevant at this scale/schedule.
```

---

## 11. Evidence Boundaries

E001 currently supports:

```text
1. CP-bilinear rank-8 attention trains to completion at 30M scale.
2. CP-trilinear rank-8 attention trains to completion at 30M scale.
3. CP-bilinear stays close to standard validation loss but with large throughput cost.
4. CP-trilinear stays close to standard validation loss and slightly improves final validation loss in this single seed, with very large throughput and VRAM cost.
5. CP-structured attention is stable enough to justify mechanism-specific diagnostics.
```

E001 does not yet establish:

```text
1. whether CP components are mechanistically interpretable
2. whether CP-trilinear reliably improves loss across seeds
3. whether the effect persists at larger scales
4. whether the CP path is causally important
5. whether lambda actually grows into a meaningful contribution
6. whether CP-bilinear or CP-trilinear helps with any specific linguistic or mechanistic behavior
```

The evidence is sufficient for architecture survival and prioritization, not for a final architecture conclusion.

---

## 12. Recommended Next E001 Analyses

The next E001 pass should expose the CP mechanism directly.

### 12.1 Lambda Diagnostics

Required:

```text
1. lambda value over training
2. lambda by layer/head if applicable
3. lambda gradient norm
4. lambda-zero eval ablation
5. lambda scaling sweep at eval time
```

Interventions:

```text
lambda = 0
lambda = learned value
lambda = 2x learned value
lambda = -learned value, if mathematically valid in implementation
```

Purpose:

```text
Determine whether the CP path is actually used.
```

### 12.2 CP Contribution Diagnostics

Log:

```text
1. ordinary attention output norm
2. CP contribution norm
3. CP-to-standard output ratio
4. CP contribution cosine similarity with ordinary output
5. per-layer CP contribution trajectory
6. per-head CP contribution trajectory, if head-level structure exists
```

Purpose:

```text
Determine whether CP is a small perturbation, a meaningful parallel path, or a mostly dormant structure.
```

### 12.3 Rank-Component Diagnostics

For each CP rank component:

```text
1. factor norm
2. output contribution norm
3. gradient norm
4. cosine similarity with other rank components
5. ablation loss delta
6. token-position activation profile
7. layer/head specialization profile
```

Purpose:

```text
Determine whether the CP rank dimension creates separable mechanism handles.
```

### 12.4 Component Ablation Tests

Recommended perturbations:

```text
1. zero one rank component at a time
2. zero top-k rank components by norm
3. randomize one component
4. swap rank components between layers
5. zero all CP contribution
6. standard-only forward pass
7. CP-only diagnostic forward pass, if implementation permits
```

Purpose:

```text
Test whether CP components are causal, redundant, or dead.
```

### 12.5 Finish Lambda0 Fixed Control

The lambda0 fixed control should be completed before any stronger CP-trilinear interpretation.

Minimum required outputs:

```text
1. full 3000-step metrics
2. run_summary.json
3. HellaSwag sanity eval
4. CP diagnostics showing lambda remains zero
5. comparison to standard and trainable CP-trilinear
```

Purpose:

```text
Separate active CP contribution from inert CP scaffolding.
```

---

## 13. Canonical E001 Conclusion

E001 shows that CP-structured attention is viable enough to study.

The completed results establish:

```text
standard:
    stable baseline

CP-bilinear:
    stable explicit low-rank Q/K interaction object

CP-trilinear:
    stable explicit low-rank Q/K/V interaction object
```

The strongest result is not a benchmark result. The strongest result is architectural:

```text
The attention computation can be augmented with explicit CP low-rank interaction structure without immediate pretraining collapse.
```

The experiment identifies two mechanism-object candidates:

```text
CP-bilinear:
    useful for studying structured Q/K compatibility components

CP-trilinear:
    useful for studying structured Q/K/V read-write interaction components
```

The next step is to stop treating the CP variants as black-box architecture variants and start treating their rank components as internal objects:

```text
measure them
track them
ablate them
compare them
ask whether they specialize
```

The E001 result is therefore:

```text
CP low-rank interaction structure is a viable attention-side research surface at this scale.
```

The unresolved question is:

```text
Do the learned CP components become interpretable, causal, and differentiated mechanisms?
```

## Mechanism Investigation Addendum

Derived backfill artifacts now live under:

```text
reports/mechanisms/backfill/E001_cp_trilinear_attention/
```

The backfill found checkpoint-recompute availability for `standard_30m_seed1`, `cp_bilinear_r8_30m_seed1`, and `cp_trilinear_r8_30m_seed1`. `cp_trilinear_r8_lambda0_30m_seed1` is explicitly `checkpoint_unavailable`, so historical activations cannot be recovered for it.

CP capture currently supports `cp_score`, `cp_output`, and `cp_lambda`. `cp_rank_component[layer, rank]` is declared in the registry but full tensor capture is unsupported until an optimized path exists. This addendum records a new post-hoc workflow and does not alter the historical claims above.
