# Technical Report: E003 QKV Architecture Gauntlet

## 1. Executive Summary

E003 tested whether two nonstandard QKV attention variants could survive early GPT-style pretraining well enough to become mechanistic-interpretability research objects.

The experiment compared:

1. `standard_refactor_control`
2. `differential_qkv_anti_value`
3. `scope_gated_qkv`

Each was run through 20-step, 150-step, and 500-step gauntlet rungs under a matched small GPT pretraining setup on FineWeb-Edu 100M. The run environment was valid: CUDA was available on an NVIDIA RTX 5060 Ti, bf16 was supported, the FineWeb-Edu train and validation shards were present, and the data manifest was verified.

The central result is:

**Both nonstandard variants trained stably, maintained validation loss close to the standard control through 500 steps, produced nondegenerate mechanism diagnostics, and passed the gauntlet.**

This should be interpreted as a **successful research-object screen**, not as evidence that either architecture improves language-model quality.

At 500 steps:

| Variant                       | Final val loss | Loss ratio vs control | Median tok/sec | Speed ratio vs control | Peak VRAM | VRAM ratio vs control | Mechanism active |
| ----------------------------- | -------------: | --------------------: | -------------: | ---------------------: | --------: | --------------------: | ---------------- |
| `standard_refactor_control`   |        5.59294 |               1.00000 |        104,110 |                  1.000 | 3240.9 MB |                 1.000 | N/A              |
| `differential_qkv_anti_value` |        5.59823 |               1.00095 |         86,748 |                  0.833 | 3379.7 MB |                 1.043 | true             |
| `scope_gated_qkv`             |        5.59785 |               1.00088 |         78,750 |                  0.756 | 3496.1 MB |                 1.079 | true             |

The most important scientific conclusion is:

**The E003 variants are not dead-on-arrival. They are stable enough to justify causal and mechanistic follow-up.**

The strongest next candidate is `differential_qkv_anti_value`, because it is simpler, closer to the control in loss, cheaper than `scope_gated_qkv`, and more directly falsifiable as a subtractive/counterwrite branch.

---

## 2. Research Objective

The objective of E003 is not to improve perplexity, benchmark score, throughput, or model quality.

The objective is to explore whether alternative Q/K/V stream geometries can produce stable, inspectable architectural substrates for mechanistic-interpretability research.

The motivating questions are:

* Is standard static Q/K/V too entangled to expose certain mechanisms cleanly?
* Can additional Q/K/V branches create interpretable separations between ordinary writes and suppressive/counterwrites?
* Can sidecar streams expose content/operator/scope-like separation?
* Can a model train stably when attention is given explicit subtractive or gated side channels?
* Do these added streams become mechanistically meaningful under training, or are they merely active-but-boring extra capacity?

E003 only addresses the first survival question:

**Can these architectures train without collapse while producing nondegenerate internal activity?**

It does not yet answer whether the branches learn semantic negation, linguistic scope, exception handling, or causal modifier behavior.

---

## 3. Experiment Setup

### 3.1 Experiment ID

```text
E003_qkv_architecture_gauntlet
```

The validator found three canonical first-build configs and no unimplemented runnable configs.

The canonical base configs were:

```text
standard_refactor_control_30m_seed1.yaml
differential_qkv_anti_value_30m_seed1.yaml
scope_gated_qkv_30m_seed1.yaml
```

The gauntlet produced 20-step, 150-step, and 500-step rung configs for each family.

### 3.2 Hardware and Runtime

The run used CUDA with:

```text
torch: 2.11.0+cu128
cuda available: True
cuda version: 12.8
device: NVIDIA GeForce RTX 5060 Ti
bf16 supported: True
```

The terminal output confirms all of these before experiment execution.

### 3.3 Dataset

The data root was:

```text
data/fineweb_edu_100m
```

The verified shards were:

```text
edufineweb_train_000001.npy: 100,000,000 tokens
edufineweb_val_000000.npy: 4,000,000 tokens
```

The manifest was verified before the run.

### 3.4 Shared Training Contract

The rung configs used:

```text
device: cuda
dtype: bfloat16
compile: false
B: 4
T: 1024
total_batch_size: 262144
grad_clip: 1.0
weight_decay: 0.1
learning_rate: 0.0006
min_lr: 0.00006
warmup_steps: 100
dropout: 0.0
bias: false
n_layer: 6
n_head: 6
n_embd: 384
block_size: 1024
```

These settings appear consistently across the E003 rung configs.

### 3.5 Gauntlet Rungs

The gauntlet used three rung lengths:

```text
rung020: 20 steps
rung150: 150 steps
rung500: 500 steps
```

## The generated configs set matching `max_steps`, `val_every`, `save_every`, and diagnostics cadence per rung.

## 4. Architectures Tested

## 4.1 Standard Refactor Control

### Computation

```text
Q, K, V = split(c_attn(x))

Y = Attention(Q, K, V)

output = c_proj(Y)
```

The standard control is ordinary GPT-style causal self-attention: one projection produces packed Q/K/V, causal scaled-dot-product attention reads values using Q/K weights, and the result is projected back to the residual stream.

### ASCII Diagram

```text
x
|
v
c_attn
|
v
[Q | K | V]
 |   |   |
 +---+---+
     |
     v
Attention(Q,K,V)
     |
     v
c_proj
     |
     v
residual write
```

### Role in E003

This is the control object. It establishes the early-training baseline for loss descent, throughput, VRAM, and training stability.

---

## 4.2 Differential QKV Anti-Value

### Computation

The tested configuration used separate positive and negative Q/K/V branches:

```text
Q_pos, K_pos, V_pos, Q_neg, K_neg, V_neg = split(c_attn(x))

Y_pos = Attention(Q_pos, K_pos, V_pos)

Y_neg = Attention(Q_neg, K_neg, V_neg)

lambda = softplus(lambda_raw)

Y = Y_pos - lambda * Y_neg

output = c_proj(Y)
```

The implementation projects to six streams when values are not shared, runs two separate causal SDPA calls, constrains lambda positive with `softplus`, and subtracts the second branch from the first.

### ASCII Diagram

```text
x
|
v
c_attn
|
v
[Q_pos | K_pos | V_pos | Q_neg | K_neg | V_neg]
   |       |       |       |       |       |
   +-------+-------+       +-------+-------+
           |                       |
           v                       v
 Attention(Q_pos,K_pos,V_pos)  Attention(Q_neg,K_neg,V_neg)
           |                       |
           v                       v
        Y_pos                   Y_neg
           |                       |
           +----------+------------+
                      |
                      v
          Y = Y_pos - lambda * Y_neg
                      |
                      v
                   c_proj
                      |
                      v
                residual write
```

### What Is Enforced

Only the algebraic sign is enforced.

The second branch is “negative” because the model output is computed as:

```text
Y_pos - lambda * Y_neg
```

where `lambda` is positive.

This does **not** enforce semantic negation, counterfactuality, anti-evidence, or suppression. The branch may learn those roles, but E003 does not prove that.

### Diagnostics

The module logs:

```text
diff_lambda
pos_output_norm
neg_output_norm
neg_to_pos_output_norm_ratio
branch_output_delta
```

These diagnostics show branch activity and nondegeneracy, not semantic interpretation.

### Research Motivation

This variant gives the model an explicit subtractive/counterwrite path. The research question is whether that path becomes mechanistically meaningful under training.

Possible later hypotheses include:

```text
The negative branch may specialize in:
- suppressive writes
- cancellation
- negation-sensitive features
- exception handling
- contradiction handling
- modifier-conditioned erasure
- anti-evidence
```

But none of these claims are established by E003.

---

## 4.3 Scope-Gated QKV

### Computation

This variant uses one ordinary content value stream, one sidecar scope stream, a receiver-side gate, and a multiplicative content/scope interaction.

```text
Q, K, V_content, V_scope = split(c_attn(x))

Content = Attention(Q, K, V_content)

Scope = Attention(Q, K, V_scope)

Scope = scope_scale * Scope

Gate = sigmoid(c_gate(x))

Combined = concat(
    Content,
    Scope,
    Content * Scope,
    Gate * Content
)

output = c_proj(Combined)
```

The implementation projects `x` into `Q`, `K`, `V`, and `scope`; computes `content` and `scoped` through separate SDPA calls sharing Q/K; computes a sigmoid gate from the receiver token; concatenates content, scope, their elementwise product, and gated content; then projects back to `d_model`.

### ASCII Diagram

```text
                          +----------------+
                          |                |
                          v                |
x ---------------------> c_attn            |
|                         |                |
|                         v                |
|                 [Q | K | V | Scope]      |
|                   |   |   |    |         |
|                   |   |   |    |         |
|                   |   |   |    v         |
|                   |   |   |  Scope stream|
|                   |   |   |              |
|                   +---+---+              |
|                       |                  |
|                       v                  |
|              Attention(Q,K,V)            |
|                       |                  |
|                       v                  |
|                    Content               |
|                                          |
|                   Q,K,Scope              |
|                       |                  |
|                       v                  |
|              Attention(Q,K,Scope)        |
|                       |                  |
|                       v                  |
|                     Scope                |
|                                          |
+-----------------> c_gate                 |
                          |                |
                          v                |
                    sigmoid gate           |
                          |                |
                          v                |
                        Gate               |
                                          |
Content ----------------------------------+
Scope ------------------------------------+
Content * Scope --------------------------+
Gate * Content ---------------------------+
        |
        v
concat[Content, Scope, Content*Scope, Gate*Content]
        |
        v
c_proj
        |
        v
residual write
```

### What Is Enforced

The architecture enforces:

1. a separate value-like sidecar stream,
2. an explicit content/scope interaction term,
3. a receiver-side gate on the content write.

It does **not** enforce that the sidecar stream learns linguistic scope. The name `scope` is a research hypothesis, not an established semantic role.

### Diagnostics

The module logs:

```text
content_output_norm
scope_output_norm
scope_to_content_norm_ratio
gate_mean
gate_std
scope_content_interaction_norm
scope_stream_scale
```

These diagnostics show that the content, scope, gate, and interaction paths are active. They do not prove modifier-scope specialization.

### Research Motivation

This variant externalizes a possible content/operator separation:

```text
content stream: ordinary attended content
scope stream: sidecar/context/operator/modifier information
interaction: multiplicative binding between content and scope
gate: receiver-side regulation of the content write
```

Later work must test whether the sidecar stream actually learns interpretable modifier, negation, exception, conditional, or boundary-sensitive behavior.

---

## 5. Gauntlet Policy and Decision Logic

The gauntlet policy was intentionally permissive for screening. It required descending loss, checkpoint existence, mechanism activity for nonstandard variants, no NaN/Inf, loss ratio below 1.20, speed ratio above 0.20, and VRAM ratio below 2.75.

The implementation blocks nonstandard variants if `mechanism_active` is not true or diagnostics are degenerate. It then compares final validation loss, median tokens/sec, and peak VRAM against the matched control.

Therefore, “passed gauntlet policy” should be interpreted as:

```text
stable enough to continue investigating
```

not:

```text
better architecture
```

The final gauntlet report advanced every rung for every candidate. The 500-step rows for all three families require manual full-row promotion rather than automatic full-run continuation.

---

## 6. Results

## 6.1 Standard Control

### 20-step rung

The 20-step standard control reached:

```text
val loss: 9.7037
val ppl: 16378.27
```

after starting from val loss 10.9101.

### 150-step rung

The 150-step standard control reached:

```text
final val loss: 6.39920
median tokens/sec: 103,178
peak VRAM: 3240.9 MB
```

The report shows descending loss, a present checkpoint, no NaN/Inf, and promotion.

### 500-step rung

The 500-step standard control reached:

```text
final val loss: 5.59294
final train loss: 5.59894
median tokens/sec: 104,110
peak VRAM: 3240.9 MB
```

The report shows descending loss and no NaN/Inf.

---

## 6.2 Differential QKV Anti-Value

### 20-step rung

At 20 steps:

```text
final val loss: 9.71225
loss ratio vs control: 1.00088
median tokens/sec: 85,120
speed ratio vs control: 0.85082
peak VRAM: 3379.7 MB
VRAM ratio vs control: 1.04283
mechanism active: true
```

The gauntlet advanced it to the 150-step rung.

### 150-step rung

At 150 steps:

```text
final val loss: 6.40037
loss ratio vs control: 1.00018
median tokens/sec: 86,811
speed ratio vs control: 0.84137
peak VRAM: 3379.7 MB
VRAM ratio vs control: 1.04283
mechanism active: true
```

The gauntlet advanced it to the 500-step rung.

### 500-step rung

At 500 steps:

```text
final val loss: 5.59823
final train loss: 5.60933
loss ratio vs control: 1.00095
median tokens/sec: 86,748
speed ratio vs control: 0.83324
peak VRAM: 3379.7 MB
VRAM ratio vs control: 1.04283
mechanism active: true
```

The report shows descending loss, checkpoint present, nondegenerate diagnostics, no NaN/Inf, and promotion.

### Mechanism Diagnostics

At the 500-step rung, the diagnostic summary included:

```text
branch_output_delta_max: 533.01
diff_lambda_min: 0.484375
diff_lambda_max: 0.5078125
neg_output_norm_max: 476.76
pos_output_norm_max: 378.49
rows_seen: 30
```

These indicate that both branches were active and lambda remained finite and positive.

### Interpretation

The differential branch survived training and remained active.

The correct conclusion is:

```text
The explicit subtractive branch is a viable mechanistic research object.
```

The incorrect conclusion would be:

```text
The branch learned semantic negation or counterfactuality.
```

No semantic specialization has been shown yet.

---

## 6.3 Scope-Gated QKV

### 20-step rung

At 20 steps:

```text
final val loss: 9.77728
loss ratio vs control: 1.00758
median tokens/sec: 76,534
speed ratio vs control: 0.76500
peak VRAM: 3496.1 MB
VRAM ratio vs control: 1.07872
mechanism active: true
```

The gauntlet advanced it to the 150-step rung.

### 150-step rung

At 150 steps:

```text
final val loss: 6.43059
loss ratio vs control: 1.00491
median tokens/sec: 77,823
speed ratio vs control: 0.75426
peak VRAM: 3496.1 MB
VRAM ratio vs control: 1.07872
mechanism active: true
```

The gauntlet advanced it to the 500-step rung.

### 500-step rung

At 500 steps:

```text
final val loss: 5.59785
final train loss: 5.60834
loss ratio vs control: 1.00088
median tokens/sec: 78,750
speed ratio vs control: 0.75641
peak VRAM: 3496.1 MB
VRAM ratio vs control: 1.07872
mechanism active: true
```

The report shows descending loss, checkpoint present, nondegenerate diagnostics, no NaN/Inf, and promotion.

### Mechanism Diagnostics

At 500 steps, the diagnostic summary included:

```text
content_output_norm_max: 328.14
scope_output_norm_max: 344.67
scope_content_interaction_norm_max: 99.44
gate_mean_min: 0.34557
gate_mean_max: 0.52282
gate_std_max: 0.23587
rows_seen: 30
```

These indicate that the content stream, scope stream, interaction term, and gate were active and nondegenerate.

### Interpretation

The scope-gated architecture survived training and activated all intended pathways.

The correct conclusion is:

```text
The content/scope/gate decomposition is trainable and inspectable.
```

The incorrect conclusion would be:

```text
The sidecar stream learned linguistic scope.
```

No such semantic specialization has been demonstrated yet.

---

## 7. Comparative Analysis

## 7.1 Stability

All three architecture families completed 20, 150, and 500-step rungs. The final gauntlet report marked every rung as `advance` / `pass`.

Both nonstandard variants satisfied the stability criteria:

```text
loss descended
checkpoint present
no NaN/Inf
diagnostics present
diagnostics nondegenerate
mechanism active
```

This is the core positive result.

## 7.2 Validation Loss

At 500 steps, both nonstandard variants were extremely close to the control:

```text
standard:      5.59294
differential:  5.59823
scope-gated:   5.59785
```

The loss ratios were:

```text
differential: 1.00095
scope-gated:  1.00088
```

These are small differences. They are useful as evidence that the variants did not disrupt early training dynamics. They are not evidence of improvement.

## 7.3 Speed

The speed penalties were significant:

```text
differential: ~0.83x control at 500 steps
scope-gated:  ~0.76x control at 500 steps
```

The differential variant is cheaper and simpler. The scope-gated variant is more expensive, largely because it performs additional attention/value-stream computation and projects a concatenated 4x-width combined representation back to `d_model`.

## 7.4 VRAM

The VRAM penalties were modest:

```text
differential: ~1.043x control
scope-gated:  ~1.079x control
```

The VRAM overhead is not alarming for research use, especially because the goal is not deployment efficiency.

## 7.5 Parameter Count

The variants were not parameter-matched:

```text
standard:      29,938,560 parameters
differential:  32,592,774 parameters
scope-gated:   34,364,550 parameters
```

The terminal output shows the standard, differential, and scope-gated parameter counts during their respective runs.
This is acceptable for an initial survival screen, but it must be addressed before any stronger claims.

---

## 8. What E003 Supports

E003 supports the following claims:

### 8.1 Harness Validity

The experiment infrastructure can validate CUDA, verify data, validate experiment configs, generate gauntlet rungs, run screens, collect promotion reports, and produce a gauntlet report.

### 8.2 Trainability

Both nonstandard architectures train through early pretraining without collapse.

### 8.3 Nondegenerate Mechanism Activity

The novel branches and streams are active according to their built-in diagnostics.

For `differential_qkv_anti_value`, the positive branch, negative branch, branch delta, and lambda were all nondegenerate.

For `scope_gated_qkv`, the content stream, scope stream, gate, and multiplicative interaction were all nondegenerate.

### 8.4 Research-Object Viability

The variants are stable enough to become subjects of mechanistic follow-up.

This is the main result.

---

## 9. What E003 Does Not Support

E003 does not support the following claims:

### 9.1 It does not show model improvement

Neither variant outperformed the standard control in validation loss at 500 steps.

### 9.2 It does not show semantic specialization

The differential branch is not yet known to encode negation, counterfactuality, suppression, or anti-evidence.

The scope stream is not yet known to encode linguistic scope, operators, modifiers, or exception structure.

### 9.3 It does not rule out capacity explanations

Both nonstandard variants have more parameters than the standard control. This leaves open a boring explanation for any future advantage unless parameter-matched and compute-matched controls are added.

### 9.4 It does not establish scaling behavior

A 500-step, 30M-class, single-seed screen is not enough to infer behavior at larger model sizes, longer training, or different data scales.

### 9.5 It does not establish interpretability superiority

The variants are instrumented, but they are not yet shown to be easier to interpret than standard attention. That requires causal probes, representation analyses, and controlled semantic tasks.

---

## 10. Scientific Assessment by Variant

## 10.1 Differential QKV Anti-Value

### Strengths

* Clean architectural intervention.
* Explicit subtractive branch.
* Mechanism is easy to ablate.
* Stable through all rungs.
* Very close to standard validation loss.
* Moderate overhead relative to scope-gated.
* Strong fit to the research question around suppressive writes and alternative value-stream behavior.

### Weaknesses

* Not parameter-matched to standard.
* No semantic specialization shown.
* Negative branch may simply be a generic second attention branch with a sign convention.
* Lambda is global/simple; branch specialization may be hard to interpret without more diagnostics.

### Assessment

This is the primary follow-up candidate.

Its simplicity makes it experimentally valuable. The next question should be whether the subtractive branch develops task- or token-specific causal roles.

---

## 10.2 Scope-Gated QKV

### Strengths

* Explicit sidecar stream.
* Explicit content/scope interaction.
* Receiver-side gate.
* All pathways active.
* Stable through all rungs.
* Directly relevant to operator/content separation hypotheses.

### Weaknesses

* More complex than differential.
* Higher parameter count.
* Larger speed penalty.
* More possible boring explanations.
* “Scope” is only a label until semantic probes demonstrate specialization.

### Assessment

This is a viable secondary follow-up candidate.

It should continue only with ablations that isolate whether the sidecar stream and gate do anything beyond adding generic capacity.

---

## 11. Required Controls for Future Work

Before making any stronger mechanistic or architectural claim, the next experiment should add controls.

### 11.1 Parameter-Matched Standard Control

Create a standard attention or widened baseline with approximately the same parameter count as each variant.

Purpose:

```text
Rule out "it only works because it has more parameters."
```

### 11.2 Compute-Aware Control

Track FLOPs or wall-clock-normalized comparisons.

Purpose:

```text
Rule out "it only works because it spends more computation."
```

### 11.3 Differential Branch Controls

For `differential_qkv_anti_value`:

```text
Y = Y_pos + lambda * Y_neg
Y = Y_pos only
Y = Y_neg only
lambda fixed at zero
lambda fixed at initial value
lambda trainable but initialized near zero
shared-value variant
same-parameter two-branch non-subtractive variant
negative branch dropout
layerwise negative-branch ablation
```

### 11.4 Scope-Gated Controls

For `scope_gated_qkv`:

```text
content only
scope only
no gate
gate only
no Content * Scope interaction
frozen gate
random gate
gate clamped to 0.5
scope stream shuffled
scope stream swapped across minimal pairs
same-parameter expanded projection control
```

---

## 12. Recommended Mechanistic Probes

## 12.1 Minimal Pair Tasks

Use controlled text pairs:

```text
The dog is friendly.
The dog is not friendly.

The treatment is safe.
The treatment is not safe.

The key is inside the box.
The key is not inside the box.

All birds can fly.
Not all birds can fly.

The rule applies.
The rule applies except on weekends.
```

Measure branch/stream activity around:

```text
not
never
no
except
unless
without
but
failed to
no longer
not all
```

### Question

Does activity track negation/scope structure beyond token identity, sentence length, and topic?

## 12.2 Branch Ablation

For `differential_qkv_anti_value`:

```text
zero Y_neg
zero Y_pos
flip sign of Y_neg
scale lambda up/down
ablate by layer
ablate by head
```

For `scope_gated_qkv`:

```text
zero Scope
zero Content
zero Content * Scope
freeze or clamp Gate
shuffle Scope
swap Scope across prompts
```

### Question

Do targeted ablations selectively affect negation, exception, modifier, or scope-sensitive tasks?

## 12.3 Branch Swap Experiments

Swap activations between minimal pairs:

```text
A: The medicine is safe.
B: The medicine is not safe.
```

For differential:

```text
swap Y_neg(A) with Y_neg(B)
```

For scope-gated:

```text
swap Scope(A) with Scope(B)
swap Gate(A) with Gate(B)
swap Content*Scope(A) with Content*Scope(B)
```

### Question

Does swapping the novel stream induce predictable behavioral changes?

## 12.4 Representation Analysis

Collect per-layer/per-head diagnostics over curated examples.

Analyze:

```text
branch norms
branch direction cosine similarity
lambda evolution
scope/content norm ratios
gate saturation
gate entropy
interaction norms
token-conditioned diagnostic distributions
```

### Question

Do novel streams specialize by layer, token class, or linguistic role?

---

## 13. Recommended Next Experiment

The next experiment should be:

```text
E004_causal_qkv_mechanism_controls
```

Purpose:

```text
Move from "stable and active" to "causally meaningful or boring."
```

Required components:

1. longer training horizon,
2. at least three seeds for the strongest candidate,
3. parameter-matched controls,
4. branch ablation hooks,
5. minimal-pair evaluation set,
6. token-level diagnostics,
7. branch-swap experiments,
8. clear kill criteria.

### Recommended primary track

Advance `differential_qkv_anti_value` first.

Reason:

```text
It is simpler, cheaper, more directly tied to the subtractive-write hypothesis,
and easier to falsify.
```

### Recommended secondary track

Keep `scope_gated_qkv` as a secondary candidate.

Reason:

```text
It is stable and active, but more complex and more vulnerable to generic-capacity explanations.
```

---

## 14. Suggested Claim Language

Use this wording:

```text
E003 tested whether two nonstandard QKV attention variants could remain stable and diagnostically active during early small-GPT pretraining. Both differential anti-value QKV and scope-gated QKV completed 20-, 150-, and 500-step gauntlet rungs without collapse, with validation loss within approximately 0.1% of the standard control at 500 steps. Both variants produced nondegenerate internal diagnostics, indicating that their novel branches/streams were active. These results support continuing them as mechanistic-interpretability research objects, but do not establish model improvement, semantic specialization, or causal interpretability.
```

Avoid this wording:

```text
The new architectures improve attention.
The negative branch learns counterfactuals.
The scope stream learns linguistic scope.
The variants outperform standard attention.
```

---

## 15. Final Conclusion

E003 succeeded as an architecture-landscape screen.

It found that:

```text
differential_qkv_anti_value
```

and

```text
scope_gated_qkv
```

are stable, trainable, nondegenerate alternatives to standard Q/K/V attention under early 30M-class GPT pretraining.

The result is not a performance claim. It is a viability claim.

The most scientifically useful next step is not simply to scale training. It is to determine whether the added streams become **causally meaningful mechanisms** or merely **active extra capacity**.

The strongest candidate for immediate follow-up is `differential_qkv_anti_value`, because its subtractive branch gives the clearest testbed for studying counterwrite behavior, suppression, cancellation, and the limitations of standard value-stream writes.
