# Technical Report: E002 Multi-Track QKV Shift-Register Experiment

## 1. Summary

E002 evaluates whether a GPT-style causal language model can tolerate alternative QKV stream geometries during real pretraining, and whether those alternative stream geometries create live, separable, intervention-ready internal pathways.

The central question is not whether the variants improve benchmark performance. The central question is:

```text
Can the ordinary static per-layer QKV stream be replaced with multiple globally shared QKV tracks and nonstandard routing rules without training collapse, and do those routing rules induce different kinds of track specialization?
```

The completed E002 experiment compares four 30M-scale runs:

```text
1. standard_refactor_control_30m_seed1
2. multi_qkv_static_3track_global_30m_seed1
3. multi_qkv_train_rotation_3track_global_30m_seed1
4. multi_qkv_position_rotation_3track_global_30m_seed1
```

All four completed 3000 training steps, produced 301 train events, 13 validation events, and 3 checkpoints.

The main result is that all tested Multi-QKV variants survived real pretraining, but their routing rules produced sharply different intervention signatures:

```text
static-global:
    strong route specialization

train-rotation:
    near track interchangeability while pathway remains live

position-rotation:
    intermediate route dependence with position-conditioned routing
```

The strongest evidence comes from destructive QKV-track perturbation tests. Static-global is highly sensitive to route identity, train-rotation is almost insensitive to route identity, and position-rotation lands between them while still showing a live selected pathway.

---

## 2. Research Objective

Standard GPT attention usually gives each layer one Q stream, one K stream, and one V stream. During inference, the model’s past is represented through a static cached K/V stream. That framing tends to treat K/V mostly as an efficiency object.

E002 treats QKV as an architectural research surface.

The experiment asks whether a model can learn with multiple QKV tracks and different routing rules:

```text
standard:
    one ordinary QKV stream

static-global Multi-QKV:
    multiple global QKV tracks, selected by layer

train-rotation Multi-QKV:
    multiple global QKV tracks, selected by layer plus training step

position-rotation Multi-QKV:
    multiple global QKV tracks, selected by layer plus token position
```

The practical purpose is to map which variants are stable enough to become mechanistic-interpretability research objects.

A variant is interesting if it satisfies:

```text
1. It trains without collapse.
2. It remains close enough to normal language-model learning to be a meaningful specimen.
3. It exposes an internal stream/track/route that can be inspected.
4. Interventions on that stream produce nontrivial effects.
5. Different routing rules produce distinguishable internal specialization regimes.
```

---

## 3. Experimental Setup

The E002 runs use the same model size class and training schedule:

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
B: 4
T: 1024
total_batch_size: 262144
max_steps: 3000
learning_rate: 0.0006
min_lr: 0.00006
warmup_steps: 100
val_every: 250
val_steps: 20
save_every: 1000
log_every: 10
```

The standard refactor control config records the shared model and training shape.

The position-rotation config records the same training structure while changing the attention type to `multi_qkv_position_rotation_3track_global`, setting `qkv_track_count: 3`, enabling `qkv_global_bank: true`, and using `qkv_route_formula: layer_plus_position`.

The static-global config records `attention_type: multi_qkv_static_3track_global`, `qkv_track_count: 3`, `qkv_global_bank: true`, and `qkv_route_formula: layer_mod`.

---

## 4. Architectures

### 4.1 Standard Refactor Control

The standard control is ordinary causal self-attention.

```text
Input residual stream
x: [batch, seq, d_model]
        |
        v
Linear projection: c_attn
        |
        v
[Q | K | V]
 |   |   |
 v   v   v
causal Attention(Q, K, V)
        |
        v
output projection: c_proj
        |
        v
residual write
```

Conceptually:

```text
Q = what the receiver token searches for
K = what each source token advertises
V = what each source token offers if attended to
```

This is the baseline object. It gives the experiment a direct comparison for training loss, throughput, VRAM, and stability.

---

### 4.2 Static-Global Multi-QKV

Static-global Multi-QKV replaces the single QKV pathway with a global bank of three QKV tracks:

```text
Track 0: Q0, K0, V0
Track 1: Q1, K1, V1
Track 2: Q2, K2, V2
```

Routing rule:

```text
active_track = layer_idx % 3
```

For a six-layer model:

```text
layer 0 -> track 0
layer 1 -> track 1
layer 2 -> track 2
layer 3 -> track 0
layer 4 -> track 1
layer 5 -> track 2
```

Dataflow:

```text
Input residual stream
        |
        v
Global QKV track bank
        |
        v
select track by layer_idx % 3
        |
        v
Attention(Q_active, K_active, V_active)
        |
        v
c_proj
        |
        v
residual write
```

This architecture asks:

```text
If each layer repeatedly uses a fixed global QKV track, do those tracks become specialized and causally important?
```

The diagnostics confirm the intended routing. At step 3000, static-global has exactly one active track per layer: layer 0 uses track 0, layer 1 uses track 1, and layer 2 uses track 2; inactive tracks have zero output norm, and the route formula is `layer_idx % track_count`.

---

### 4.3 Train-Rotation Multi-QKV

Train-rotation uses the same three-track global QKV bank, but changes which track a layer sees during training.

Training-time routing:

```text
active_track = (layer_idx + step) % 3
```

Evaluation/generation routing:

```text
active_track = layer_idx % 3
```

The diagnostics explicitly record the formula as `(layer_idx + step) % track_count during train; layer_idx % track_count during eval/generate`.

Dataflow:

```text
Input residual stream
        |
        v
Global QKV track bank
        |
        v
select track by layer and training step

    train: active_track = (layer_idx + step) % 3
    eval:  active_track = layer_idx % 3

        |
        v
Attention(Q_active, K_active, V_active)
        |
        v
c_proj
        |
        v
residual write
```

This architecture asks:

```text
If layer-track assignment changes during training, does the model still learn stable attention behavior, and does this suppress fixed track specialization?
```

---

### 4.4 Position-Rotation Multi-QKV

Position-rotation also uses a three-track global QKV bank, but routing depends on token position as well as layer.

Routing rule:

```text
active_track = (layer_idx + position) % 3
```

This means different token positions inside the same layer can use different QKV tracks.

Example for layer 0:

```text
position 0 -> track 0
position 1 -> track 1
position 2 -> track 2
position 3 -> track 0
position 4 -> track 1
position 5 -> track 2
...
```

Example for layer 1:

```text
position 0 -> track 1
position 1 -> track 2
position 2 -> track 0
position 3 -> track 1
position 4 -> track 2
position 5 -> track 0
...
```

Dataflow:

```text
Input residual stream
x: [B, T, d_model]
        |
        v
Global QKV track bank
        |
        v
for layer l and token position p:

    active_track = (l + p) % 3

        |
        v
position-conditioned Q/K/V selection
        |
        v
causal attention over selected streams
        |
        v
c_proj
        |
        v
residual write
```

This architecture asks:

```text
Does every token position in a layer need to use the same QKV stream, or can QKV routing vary across positions while preserving trainability?
```

The diagnostics confirm the intended balanced position routing. At step 3000, position-rotation has active track counts like `{0:342, 1:341, 2:341}`, `active_track_index: null`, `track_entropy` around `1.0986`, and route formula `(layer_idx + position) % track_count`.

---

## 5. Training Results

| Run                         | Final val loss | Final perplexity | Median tok/s | Peak VRAM | Checkpoints |
| --------------------------- | -------------: | ---------------: | -----------: | --------: | ----------: |
| standard refactor control   |         4.0816 |            59.24 |      107,619 |   3.24 GB |           3 |
| static-global Multi-QKV     |         4.1213 |            61.64 |       44,374 |   4.09 GB |           3 |
| train-rotation Multi-QKV    |         4.4028 |            81.68 |       44,152 |   4.09 GB |           3 |
| position-rotation Multi-QKV |         4.1806 |            65.41 |       40,782 |   4.29 GB |           3 |

The run summaries show the standard control reached final val loss 4.0816, static-global reached 4.1213, train-rotation reached 4.4028, and position-rotation reached 4.1806.

### 5.1 Stability

All three Multi-QKV variants trained to completion.

That means the first experimental hurdle was cleared:

```text
Multiple global QKV tracks and nonstandard routing rules do not automatically break GPT-style pretraining at this scale.
```

### 5.2 Loss ordering

Loss ordering:

```text
standard control:       4.0816
static-global:          4.1213
position-rotation:      4.1806
train-rotation:         4.4028
```

Interpretation:

```text
static-global:
    closest Multi-QKV variant to standard loss

position-rotation:
    moderate penalty but still stable

train-rotation:
    much larger penalty, but still learns
```

The train-rotation result is especially important as a contrast object: it survives, but the routing rule appears to prevent useful fixed track specialization.

### 5.3 Throughput and VRAM

The Multi-QKV variants are slower than standard:

```text
standard:          ~107.6k tok/s
static-global:      ~44.4k tok/s
train-rotation:     ~44.2k tok/s
position-rotation:  ~40.8k tok/s
```

VRAM is also higher:

```text
standard:          ~3.24 GB
static-global:      ~4.09 GB
train-rotation:     ~4.09 GB
position-rotation:  ~4.29 GB
```

This confirms that these are not efficiency variants in their current form. Their relevance is mechanistic: they create additional inspectable stream structure.

---

## 6. Destructive Track Perturbation Tests

The key E002 evidence comes from destructive QKV-track tests.

The tests apply three interventions:

```text
rotate_tracks:
    remap selected track identities

force_track_0:
    force use of track 0

zero_selected:
    zero the selected track/pathway
```

These distinguish two questions:

```text
1. Is the selected QKV pathway live?
   -> measured by zero_selected

2. Does the identity of the selected track matter?
   -> measured by rotate_tracks and force_track_0
```

### 6.1 Results

| Architecture      | rotate_tracks Δloss | force_track_0 Δloss | zero_selected Δloss | Signature                                           |
| ----------------- | ------------------: | ------------------: | ------------------: | --------------------------------------------------- |
| static-global     |             +3.1309 |             +1.8131 |             +2.1382 | strong route specialization                         |
| train-rotation    |             +0.0086 |             +0.0052 |             +1.8103 | track identity nearly interchangeable; pathway live |
| position-rotation |             +0.2065 |             +0.8394 |             +1.9672 | intermediate route dependence                       |

Static-global destructive-test values are recorded with large losses for rotate, force, and zero perturbations.

Train-rotation destructive-test values show very small rotate/force deltas but a large zero-selected delta.

Position-rotation destructive-test values show moderate route perturbation sensitivity and a large zero-selected delta.

---

## 7. Mechanistic Interpretation

### 7.1 Static-global: route-specialized

Static-global has the strongest route-identity dependence.

The relevant pattern is:

```text
rotate_tracks:   large loss increase
force_track_0:   large loss increase
zero_selected:   large loss increase
```

Interpretation:

```text
The selected track is live.
The identity of the selected track matters.
The model has specialized around fixed layer-track assignments.
```

This makes static-global the cleanest intervention target. It provides a strong route-specialized object for studying whether different QKV tracks learn different functions.

Possible follow-up questions:

```text
Do track 0, track 1, and track 2 specialize by layer role?
Do repeated layers sharing the same track learn compatible functions?
Do track identities align with early/middle/late computation?
Can track swaps reveal layer-specific reliance on QKV geometry?
```

---

### 7.2 Train-rotation: route-despecialized but pathway-live

Train-rotation has the most distinctive destructive-test signature.

The relevant pattern is:

```text
rotate_tracks:   almost no effect
force_track_0:   almost no effect
zero_selected:   large effect
```

Interpretation:

```text
The active QKV pathway matters.
But the identity of the specific track barely matters.
```

That suggests train-time route rotation pressures the tracks toward interchangeability.

This is not a failure. It is a useful contrast class.

Train-rotation asks:

```text
What happens when the architecture has multiple QKV tracks, but training discourages permanent layer-track identity?
```

The answer from this run is:

```text
The model still trains, but loses much of the fixed route specialization seen in static-global.
```

That gives E002 a controlled contrast:

```text
same global track count
same general Multi-QKV machinery
different routing schedule
different specialization regime
```

---

### 7.3 Position-rotation: position-conditioned intermediate specialization

Position-rotation is the most relevant result for questioning static per-layer QKV assumptions.

The relevant pattern is:

```text
rotate_tracks:   moderate effect
force_track_0:   substantial effect
zero_selected:   large effect
```

Interpretation:

```text
The active pathway is live.
Track identity matters.
But track identity is less rigidly specialized than static-global.
```

The architecture matters because different token positions in the same layer route through different QKV tracks.

This directly tests the assumption:

```text
Every token in a given layer should use the same QKV stream.
```

The run shows that this assumption is not required for short 30M-scale pretraining stability. The model can tolerate position-conditioned QKV routing.

Possible follow-up questions:

```text
Do tracks specialize by position modulo class?
Do tracks specialize by token role rather than literal position?
Does position-conditioned routing expose sequence-structure mechanisms hidden by standard per-layer attention?
Does track use correlate with local syntax, induction behavior, delimiter handling, or scope/modifier position?
```

---

## 8. HellaSwag Sanity Eval

The HellaSwag eval is small: 100 validation examples. It is useful only as a sanity check that the models are not obviously broken.

Results:

```text
standard control:       0.33
position-rotation:      0.31
static-global:          0.30
train-rotation:         0.28
```

The dump records static-global at 30/100, train-rotation at 28/100, standard at 33/100, and position-rotation at 31/100.
Interpretation:

```text
All variants remain in the same rough sanity range.
The sample is too small for ranking.
The eval does not drive the scientific conclusion.
```

The scientific conclusion comes from survival plus route perturbation behavior, not from this small HellaSwag sample.

---

## 9. Completed vs Unimplemented E002 Variants

The completed E002 runs are:

```text
standard_refactor_control_30m_seed1
multi_qkv_static_3track_global_30m_seed1
multi_qkv_train_rotation_3track_global_30m_seed1
multi_qkv_position_rotation_3track_global_30m_seed1
```

The config dump also contains E002 variants marked `experimental_unimplemented`, including layer-shift and non-global static variants.
Those unimplemented configs should not be treated as tested results.

---

## 10. Evidence Boundaries

The E002 evidence supports the following:

```text
1. Multi-track QKV routing can survive real GPT-style pretraining at 30M scale.
2. Static layer-based routing produces strong track specialization.
3. Train-time route rotation produces near track interchangeability while preserving pathway dependence.
4. Position-conditioned routing survives and produces intermediate track dependence.
5. Destructive route perturbations expose real causal differences between routing schemes.
```

The E002 evidence does not yet establish:

```text
1. Whether any variant improves model quality in general.
2. Whether results replicate across seeds.
3. Whether results persist at larger model sizes.
4. Whether the tracks have human-interpretable semantic roles.
5. Whether position-rotation tracks specialize by linguistic function.
6. Whether the same effects hold under longer training.
```

These boundaries matter because the next stage should not ask whether E002 has already solved a problem. It should ask which surviving architecture is most promising for deeper mechanism analysis.

---

## 11. Recommended Next Analyses

The next E002 work should deepen the architecture-specific evidence.

### 11.1 Expand destructive tests

Current destructive tests are useful, but should be expanded.

Recommended:

```text
1. run destructive tests over more validation batches
2. compute confidence intervals over loss deltas
3. run per-layer track perturbations
4. run per-position-class perturbations for position-rotation
5. separate Q-only, K-only, V-only track perturbations
6. compare rotate, force, zero, randomize, and swap interventions
```

The most important extension is to separate Q, K, and V:

```text
zero Q track only
zero K track only
zero V track only
rotate Q tracks only
rotate K tracks only
rotate V tracks only
```

This would identify whether route specialization lives primarily in query formation, key advertisement, value content, or the whole QKV bundle.

### 11.2 Add track similarity diagnostics

For each variant:

```text
1. Q track cosine similarity
2. K track cosine similarity
3. V track cosine similarity
4. output cosine similarity by track
5. gradient cosine similarity by track
6. track norm trajectories over training
```

Expected patterns:

```text
static-global:
    lower similarity between tracks if specialization is real

train-rotation:
    higher similarity between tracks if interchangeability is real

position-rotation:
    intermediate similarity, possibly structured by position class
```

### 11.3 Position-rotation-specific analysis

Position-rotation deserves its own analysis path.

Recommended:

```text
1. group tokens by position mod 3
2. compare losses under perturbing only one position class
3. analyze track activity by token type and position class
4. inspect whether delimiters, punctuation, BOS-like positions, or local syntactic roles correlate with track use
5. test whether route identity matters more in early, middle, or late layers
```

Because routing is deterministic by `(layer_idx + position) % 3`, the immediate question is:

```text
Does the model merely tolerate this routing, or does it exploit position-class structure?
```

### 11.4 Train-rotation-specific analysis

Train-rotation needs careful separation between training-time routing and eval-time routing.

Recommended:

```text
1. evaluate under frozen train-phase routes
2. evaluate under standard eval route
3. evaluate under random route assignment
4. compare track similarity to static-global
5. inspect whether all tracks converge toward similar Q/K/V norms and outputs
```

The core question:

```text
Is train-rotation producing genuinely interchangeable tracks, or are the current perturbations too coarse to reveal specialization?
```

### 11.5 Static-global-specific analysis

Static-global is the cleanest track-specialization object.

Recommended:

```text
1. per-layer track swap matrix
2. track replacement matrix: layer l using track t
3. track ablation by layer
4. track-specific Q/K/V norm and gradient trajectories
5. compare layers sharing the same track, e.g. layer 0 and layer 3
```

The core question:

```text
What does each track specialize for, and is specialization shared across layers that reuse the same global track?
```

---

## 12. Canonical E002 Conclusion

E002 demonstrates that QKV stream geometry is a viable research surface.

The core result is:

```text
A GPT-style model can train with multiple global QKV tracks and deterministic routing rules that differ from ordinary one-stream-per-layer attention.
```

The routing rule controls the specialization regime:

```text
static-global:
    layer-fixed routing
    strong track identity dependence

train-rotation:
    training-step-rotated routing
    near track identity interchangeability
    selected pathway remains live

position-rotation:
    layer-plus-position routing
    intermediate track identity dependence
    per-position QKV stream variation survives training
```

The most important experimental artifact is not a benchmark number. It is the intervention signature:

```text
Architecture          rotate Δ    force0 Δ    zero-selected Δ    interpretation
--------------------------------------------------------------------------------
static-global         +3.1309     +1.8131     +2.1382            route-specialized
train-rotation        +0.0086     +0.0052     +1.8103            track-interchangeable, pathway-live
position-rotation     +0.2065     +0.8394     +1.9672            intermediate, position-conditioned
```

This gives a small but concrete landscape:

```text
1. fixed layer routing creates specialization
2. train-time route rotation suppresses specialization
3. position-conditioned routing survives and creates a distinct intermediate regime
```

The strongest next target is position-rotation, because it most directly challenges the static per-layer QKV stream assumption while remaining stable enough to study.

The cleanest diagnostic target is static-global, because it gives the strongest route-specialized intervention surface.

The most useful contrast class is train-rotation, because it shows that multiple tracks alone are not enough; the routing schedule determines whether tracks become specialized or interchangeable.

E002 therefore succeeds as a mechanism-oriented architecture-landscape experiment:

```text
It identifies multiple nonstandard QKV-stream geometries that train,
distinguishes their route-specialization behavior,
and selects the next objects for deeper mechanistic analysis.
```

