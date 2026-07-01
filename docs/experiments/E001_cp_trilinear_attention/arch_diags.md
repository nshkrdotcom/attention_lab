# E001 Architecture Diagrams: CP-Bilinear and CP-Trilinear Attention Variants

E001 is about changing the internal geometry of the attention computation.

The research object is not:

```text id="q6x5kw"
better benchmark score
faster attention
SOTA architecture
generic model improvement
```

The research object is:

```text id="diw6ed"
Can a GPT-style attention block tolerate explicit low-rank interaction structure inside attention, and does that structure remain active enough to become a mechanistic object?
```

The completed E001 variants in the uploaded dump are:

```text id="klj7e9"
0. Standard attention control
1. CP-bilinear rank-8 attention
2. CP-trilinear rank-8 attention
```

There is also:

```text id="v4bgxg"
3. CP-trilinear rank-8 lambda0 fixed
```

but it is not completed in the dump. It only has a step-0 validation record, so it belongs in the diagram/config section, not in the completed-result section.

The completed E001 summaries show that standard, CP-bilinear, and CP-trilinear each reached 3000 steps with 301 train events, 13 validation events, and 3 checkpoints.

---

# 0. Standard attention control

This is the ordinary GPT-style causal self-attention block.

The standard config uses:

```text id="wqpr9d"
attention_type: standard
block_size: 1024
n_layer: 6
n_head: 6
n_embd: 384
device: cuda
dtype: bfloat16
B: 4
T: 1024
total_batch_size: 262144
max_steps: 3000
```

The standard config records the baseline model shape and shared training setup.

## Standard dataflow

```text id="x9rgfx"
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

## Token-level view

```text id="hwo6bn"
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

## Conceptual role

```text id="hsqgpz"
Q = what am I looking for?
K = what do I advertise?
V = what content do I provide?
```

Standard attention is the control object:

```text id="o1oykx"
one Q stream
one K stream
one V stream
ordinary dot-product compatibility
ordinary weighted value read
ordinary residual write
```

E001 asks what happens if the attention block is given an explicit CP low-rank interaction component.

---

# 1. CP-bilinear rank-8 attention

CP-bilinear attention introduces an explicit low-rank bilinear interaction structure into attention.

The tested config uses:

```text id="4iy185"
attention_type: cp_bilinear
cp_rank: 8
cp_lambda_init: 0.0
cp_lambda_trainable: true
cp_lambda_fixed: false
```

The CP-bilinear config confirms `attention_type: cp_bilinear`, `cp_rank: 8`, and trainable lambda.

## High-level idea

Standard attention compares Q and K directly:

```text id="kh6skk"
score(i,j) = Q_i · K_j
```

CP-bilinear adds a structured low-rank interaction path:

```text id="v2osnt"
score(i,j) =
    ordinary QK compatibility
    +
    lambda * CP_bilinear(Q_i, K_j)
```

The exact implementation details should be verified from code before writing equations with tensor indices, but architecturally the point is:

```text id="fpa8nd"
Do not only let Q and K interact through the ordinary dot product.
Add an explicit rank-limited interaction basis that can become inspectable.
```

## CP-bilinear dataflow

```text id="9ad0ve"
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
        |          |
        |          +-----------------------------+
        |                                        |
        v                                        v
ordinary QK scores                    CP-bilinear interaction
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

## Token-level view

```text id="cayrj8"
For receiver token i and source token j <= i:

ordinary_score_ij = Q_i · K_j

cp_score_ij = low_rank_bilinear_interaction(Q_i, K_j)

combined_score_ij = ordinary_score_ij + lambda * cp_score_ij

attention_weights_i = softmax(combined_score_i)

output_i = sum_j attention_weights_ij * V_j
```

## What this architecture changes

Standard attention says:

```text id="k51o2h"
Q/K compatibility is a direct dot product.
```

CP-bilinear says:

```text id="ja8cz0"
Q/K compatibility can include a structured low-rank bilinear term.
```

This creates a possible mechanism object:

```text id="oxpopz"
Do the learned CP rank components become specialized compatibility features?
```

## Why CP-bilinear is a research object

The point is not that CP-bilinear should win on loss.

The point is:

```text id="f8l6nv"
A rank-limited interaction basis gives you named internal components to inspect, ablate, compare, and track during training.
```

Possible later probes:

```text id="og7aes"
- component-wise CP factor norms
- rank-component ablations
- learned lambda trajectory
- cosine similarity among CP factors
- whether certain CP components specialize by layer/head/token pattern
- whether CP components capture syntactic, positional, negation, modifier, or induction-like compatibility
```

## Completed-run behavior

CP-bilinear completed the 3000-step run:

```text id="q3je4c"
final val loss:       4.0863
final perplexity:     59.52
median tokens/sec:    33,626
peak VRAM:            4.44 GB
checkpoint count:     3
```

The run summary confirms these values.

Interpretation:

```text id="zzdcte"
CP-bilinear did not collapse.
It remained close to the standard control.
It is much slower than standard.
Its current value is as a stable structured-interaction specimen, not as an efficiency or quality improvement.
```

---

# 2. CP-trilinear rank-8 attention

CP-trilinear is the more ambitious E001 variant.

The tested completed config uses:

```text id="ei34bh"
attention_type: cp_trilinear
cp_rank: 8
cp_lambda_init: 0.0
cp_lambda_trainable: true
cp_lambda_fixed: false
```

The CP-trilinear config records `attention_type: cp_trilinear`, `cp_rank: 8`, and trainable CP lambda.

## High-level idea

Standard attention separates:

```text id="r0luw8"
1. compatibility: Q_i compares with K_j
2. content read: weights select V_j
```

CP-trilinear asks whether the model can use a structured low-rank interaction that involves the value/content side as well.

Architecturally:

```text id="n47eea"
Instead of only modifying Q/K compatibility,
introduce a CP low-rank interaction over the Q, K, and V pathway.
```

The exact code-level contraction should be checked before writing a formal tensor equation, but the canonical architectural meaning is:

```text id="rvde5t"
CP-bilinear:  structured interaction over Q and K
CP-trilinear: structured interaction over Q, K, and V/content
```

## CP-trilinear dataflow

```text id="bjmxny"
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

## Token-level view

```text id="nq9lsx"
For receiver token i:

ordinary path:
    Q_i compares with K_j
    weights select V_j
    ordinary output_i = sum_j weights_ij * V_j

CP-trilinear path:
    Q_i, K_j, and V_j/content-side factors participate
    in a rank-8 structured interaction

combined output:
    output_i = ordinary_output_i + lambda * cp_trilinear_output_i
```

Again, this is the architecture-level diagram. The implementation should be used as the source of truth for exact index notation.

## What this architecture changes

Standard attention:

```text id="n4br1y"
Q/K decide where to read.
V supplies what is read.
```

CP-trilinear:

```text id="u7e9ep"
Q, K, and value/content structure can participate in an explicit low-rank interaction.
```

This creates a richer mechanistic object than CP-bilinear:

```text id="lsofk6"
Not just "what token is relevant?"
But "what structured relation between receiver query, source key, and source value/content is being expressed?"
```

## Why CP-trilinear is a research object

CP-trilinear gives you a structured low-rank basis over the full attention read/write path.

Possible later probes:

```text id="suw5qx"
- learned lambda trajectory
- CP rank-component norms
- CP factor similarity
- layer/head component specialization
- component ablation by rank index
- whether components align with token roles or relation types
- whether any component behaves like a suppressive, modifier, scope, induction, or copy-like mechanism
```

The key research question is:

```text id="ejqjg7"
Do low-rank Q/K/V interaction components become interpretable mechanism handles?
```

## Completed-run behavior

CP-trilinear completed the 3000-step run:

```text id="zxluzl"
final val loss:       4.0623
final perplexity:     58.11
median tokens/sec:    16,761
peak VRAM:            6.77 GB
checkpoint count:     3
```

The run summary confirms the 3000-step completion, final loss, perplexity, throughput, VRAM, and checkpoint count.

Interpretation:

```text id="q4xe1z"
CP-trilinear did not collapse.
It stayed trainable at 30M scale.
It produced a slightly lower validation loss than the E001 standard run in this single seed.
It is very expensive.
Its importance is that it survived as a structured Q/K/V interaction object.
```

---

# 3. CP-trilinear rank-8 lambda0 fixed control

This config exists but is not completed in the uploaded run dump.

The config uses:

```text id="oe3x9b"
attention_type: cp_trilinear
cp_rank: 8
cp_lambda_init: 0.0
cp_lambda_trainable: false
cp_lambda_fixed: true
```

The config explicitly sets `cp_lambda_trainable: false` and `cp_lambda_fixed: true`.

## Intended purpose

The lambda0 fixed control asks:

```text id="0m2wno"
If the CP-trilinear path is present but lambda is fixed at zero,
does the model behave like the standard baseline?
```

Architecturally:

```text id="z211wy"
ordinary attention path remains active
CP-trilinear path exists structurally
lambda = 0
CP contribution is suppressed
```

## Lambda0 fixed dataflow

```text id="2ix8da"
Input residual stream
x: [batch, seq, d_model]
        |
        v
Linear projection: c_attn
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

## Completed-run status

This is not a completed run in the uploaded dump.

The only visible metric for `cp_trilinear_r8_lambda0_30m_seed1` is a step-0 validation event:

```text id="dyrwi0"
step: 0
val_loss: 10.8903
val_perplexity: 53653.89
```

That is not enough to interpret training behavior.

## Correct status label

```text id="t9ioda"
configured but not completed in this dump
```

Do not include it as evidence that the lambda-fixed control behaves like standard until the 3000-step run exists.

---

# 4. Side-by-side architectural comparison

```text id="p08uwr"
STANDARD CONTROL
================

x
|
v
[Q K V]
 | | |
 v v v
Attention(Q,K,V)
|
v
c_proj
|
v
residual write
```

```text id="33kzxb"
CP-BILINEAR RANK-8
==================

x
|
v
[Q K V]
 | | |
 | | +----------------------+
 | |                        |
 v v                        v
ordinary QK score     CP-bilinear(Q,K)
 |                         |
 +-----------+-------------+
             |
             v
combined attention scores
             |
             v
softmax + causal mask
             |
             v
weighted sum of V
             |
             v
c_proj
             |
             v
residual write
```

```text id="0k5ywk"
CP-TRILINEAR RANK-8
===================

x
|
v
[Q K V]
 | | |
 | | +----------------------+
 | |                        |
 v v v                      v
ordinary Attention(Q,K,V)   CP-trilinear(Q,K,V/content)
          |                 |
          |                 v
          |          lambda-scaled CP contribution
          |                 |
          +--------+--------+
                   |
                   v
          combined attention output
                   |
                   v
                 c_proj
                   |
                   v
             residual write
```

```text id="8yheyp"
CP-TRILINEAR LAMBDA0 FIXED CONTROL
==================================

x
|
v
[Q K V]
 | | |
 | | +----------------------+
 | |                        |
 v v v                      v
ordinary Attention(Q,K,V)   CP-trilinear(Q,K,V/content)
          |                 |
          |                 v
          |          lambda * CP contribution
          |                 |
          |             lambda = 0
          |                 |
          +--------+--------+
                   |
                   v
          ordinary attention output only
                   |
                   v
                 c_proj
                   |
                   v
             residual write
```

---

# 5. Deepest distinction

## Standard attention

```text id="svdyo8"
one ordinary Q/K/V pathway
compatibility = Q dot K
read = weighted sum of V
```

Best for:

```text id="o4x4tj"
control object
training sanity
baseline loss/speed/VRAM
ordinary attention comparison
```

## CP-bilinear attention

```text id="3ziyxb"
ordinary Q/K/V path
plus explicit low-rank Q/K interaction term
```

Best for asking:

```text id="bt0kcg"
Do structured low-rank Q/K compatibility components become interpretable?
Do specific CP rank components specialize?
Can component ablation reveal attention-compatibility mechanisms?
```

## CP-trilinear attention

```text id="zvyeb8"
ordinary Q/K/V path
plus explicit low-rank Q/K/V interaction structure
```

Best for asking:

```text id="j5zv84"
Do structured Q/K/V interaction components become live mechanism handles?
Can a low-rank basis capture relation-like read/write patterns?
Do rank components specialize into distinct attention-side operations?
```

## CP-trilinear lambda0 fixed

```text id="2dmjgp"
same CP-trilinear structure
but lambda fixed at zero
```

Best for asking, once completed:

```text id="zwese9"
Does simply adding the dormant CP machinery affect training?
Does the active CP contribution matter beyond parameterization/scaffolding?
Does trainable lambda explain the difference between CP-trilinear and control?
```

---

# 6. Canonical E001 interpretation

The important E001 result is not:

```text id="zljc4c"
CP-trilinear is better than standard attention.
```

The important E001 result is:

```text id="ox9lm8"
A GPT-style model can tolerate explicit CP low-rank interaction structure inside the attention block during real pretraining.
```

The completed runs show:

```text id="xiv3cj"
standard:
    stable control

CP-bilinear:
    stable Q/K low-rank interaction object
    close to standard loss
    slower and higher VRAM

CP-trilinear:
    stable Q/K/V low-rank interaction object
    slightly lower single-seed val loss than standard
    much slower and much higher VRAM
```

The architectural takeaway is:

```text id="e0fbdp"
Attention-side interaction structure is a viable research surface.

You can add explicit low-rank bilinear or trilinear structure without immediate collapse.

The next question is not whether it is better.

The next question is whether the learned rank components become interpretable, causal, and separable mechanism handles.
```

---

# 7. What the next E001 diagnostics should canonically target

The next E001 diagnostic layer should not be generic evals. It should expose the CP mechanism itself.

Required diagnostics:

```text id="aeeauy"
1. learned lambda over training
2. CP contribution norm vs ordinary attention output norm
3. CP-to-standard output ratio by layer/head
4. CP rank-component norms
5. CP rank-component cosine similarity
6. component ablation by rank index
7. lambda-zero ablation at eval time
8. CP-only / standard-only output decomposition
9. per-layer CP activity trajectory
10. comparison to completed lambda0 fixed control
```

Canonical intervention tests:

```text id="efc6a4"
zero CP contribution
freeze lambda at current value
force lambda to zero
double lambda
randomize CP rank components
ablate one rank component at a time
ablate top-k CP components by norm
swap CP factors across layers if implementation permits
```

Canonical questions:

```text id="y6gg8h"
Does the CP path matter causally?
Do rank components specialize?
Does CP-trilinear create richer mechanisms than CP-bilinear?
Is the slight CP-trilinear loss improvement associated with active CP contribution or just noise?
Does lambda grow from zero, stay near zero, or become layer/head-specific if allowed?
```

That is the E001 architecture story.
