# E002 Architecture Diagrams: Multi-Track QKV Shift-Register Variants

E002 is about changing the geometry of the Q/K/V stream.

The research object is not “better language modeling.” The research object is:

```text
Can a GPT-style attention block tolerate multiple QKV tracks and nonstandard routing rules, and do those routes become live, differentiated, intervenable streams?
```

The completed E002 variants are:

```text
0. Standard refactor control
1. Multi-QKV static 3-track global
2. Multi-QKV train-rotation 3-track global
3. Multi-QKV position-rotation 3-track global
```

The completed run summaries show that all four reached 3000 steps with 301 train events, 13 validation events, and 3 checkpoints.

---

# 0. Standard refactor control

This is the ordinary GPT-style causal self-attention block. It is the control object for E002.

The standard config uses:

```text
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

The E002 standard control config records the standard attention type and shared training schedule.

## Standard dataflow

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

## Token-level view

```text
receiver token i:

Q_i compares with K_j for j <= i
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

```text
Q = what am I looking for?
K = what do I advertise to future tokens?
V = what content do I offer if attended to?
```

Standard attention has one Q stream, one K stream, and one V stream per layer.

The E002 variants ask what happens when the attention block has multiple possible Q/K/V tracks instead of a single static Q/K/V pathway.

---

# 1. Multi-QKV static 3-track global

This is the cleanest E002 mechanism object.

The architecture creates a global bank of three separate QKV tracks:

```text
Track 0: Q0, K0, V0
Track 1: Q1, K1, V1
Track 2: Q2, K2, V2
```

The routing rule is static:

```text
active_track = layer_idx % track_count
```

So, for a 6-layer model and 3 tracks:

```text
layer 0 -> track 0
layer 1 -> track 1
layer 2 -> track 2
layer 3 -> track 0
layer 4 -> track 1
layer 5 -> track 2
```

The static-global config uses `attention_type: multi_qkv_static_3track_global`, `qkv_track_count: 3`, `qkv_global_bank: true`, and `qkv_route_formula: layer_mod`.

## Static-global dataflow

```text
Input residual stream
x: [batch, seq, d_model]
        |
        v
Global Multi-QKV projection bank
        |
        v
+---------------------------------------------------+
| Track 0: [Q0 | K0 | V0]                           |
| Track 1: [Q1 | K1 | V1]                           |
| Track 2: [Q2 | K2 | V2]                           |
+---------------------------------------------------+
        |
        v
Select track by layer:

    active_track = layer_idx % 3

        |
        v

Example:
+---------+--------------+----------------+
| Layer   | Active track | QKV used       |
+---------+--------------+----------------+
| L0      | Track 0      | Q0, K0, V0     |
| L1      | Track 1      | Q1, K1, V1     |
| L2      | Track 2      | Q2, K2, V2     |
| L3      | Track 0      | Q0, K0, V0     |
| L4      | Track 1      | Q1, K1, V1     |
| L5      | Track 2      | Q2, K2, V2     |
+---------+--------------+----------------+

        |
        v
causal scaled dot-product attention
Attention(Q_active, K_active, V_active)
        |
        v
output projection: c_proj
        |
        v
residual write
```

## Static-global token-level view

```text
For receiver token i in layer l:

track t = l % 3

Q_t,i compares with K_t,j for j <= i
        |
        v
attention weights within selected track t
        |
        v
weighted sum of V_t,j
        |
        v
write to residual stream
```

## What this architecture changes

Standard attention says:

```text
Each layer has one Q/K/V pathway.
```

Static-global Multi-QKV says:

```text
There are multiple Q/K/V pathways.
Layer identity chooses which pathway is used.
The same global track bank is reused across layers.
```

This creates a clean intervention target:

```text
What happens if we rotate, force, or zero the selected QKV track?
```

## Diagnostic signature

The static-global diagnostics show exactly one active track per layer. At step 3000, layer 0 uses track 0, layer 1 uses track 1, and layer 2 uses track 2; inactive tracks have zero output norm, and the route formula is `layer_idx % track_count`.

## Destructive-test result

Static-global is strongly route-specialized:

```text
rotate_tracks loss delta: +3.1309
force_track_0 loss delta: +1.8131
zero_selected loss delta: +2.1382
```

The destructive-test JSON records all three perturbations passing and shows large loss/logit changes.

## Interpretation

```text
The tracks are not interchangeable.
The selected track identity matters.
The architecture creates a live, differentiated routing structure.
```

This is the cleanest E002 evidence that multi-track QKV streams can become mechanistically real rather than dead or redundant.

---

# 2. Multi-QKV train-rotation 3-track global

This variant uses the same global 3-track QKV bank, but changes the routing rule during training.

The core idea:

```text
Do not let a layer always see the same track during training.
Rotate which track a layer sees as training progresses.
```

The diagnostic route formula is:

```text
train-time route:
    active_track = (layer_idx + step) % track_count

eval/generate route:
    active_track = layer_idx % track_count
```

The uploaded diagnostics explicitly show the formula as `(layer_idx + step) % track_count during train; layer_idx % track_count during eval/generate`.

## Train-rotation dataflow

```text
Input residual stream
x: [batch, seq, d_model]
        |
        v
Global Multi-QKV projection bank
        |
        v
+---------------------------------------------------+
| Track 0: [Q0 | K0 | V0]                           |
| Track 1: [Q1 | K1 | V1]                           |
| Track 2: [Q2 | K2 | V2]                           |
+---------------------------------------------------+
        |
        v
Select track by layer and training step:

    during training:
        active_track = (layer_idx + step) % 3

    during eval/generation:
        active_track = layer_idx % 3

        |
        v
causal scaled dot-product attention
Attention(Q_active, K_active, V_active)
        |
        v
output projection: c_proj
        |
        v
residual write
```

## Train-time route example

For three consecutive route phases:

```text
step phase 0:
    L0 -> T0
    L1 -> T1
    L2 -> T2
    L3 -> T0
    L4 -> T1
    L5 -> T2

step phase 1:
    L0 -> T1
    L1 -> T2
    L2 -> T0
    L3 -> T1
    L4 -> T2
    L5 -> T0

step phase 2:
    L0 -> T2
    L1 -> T0
    L2 -> T1
    L3 -> T2
    L4 -> T0
    L5 -> T1
```

## What this architecture changes

Static-global asks:

```text
Can layers specialize around fixed track identities?
```

Train-rotation asks:

```text
What happens if track identity is prevented from staying fixed during training?
```

The expected research-object difference is:

```text
Static-global should encourage track specialization.
Train-rotation should pressure tracks toward interchangeability or shared competence.
```

## Destructive-test result

Train-rotation has a very different perturbation signature:

```text
rotate_tracks loss delta: +0.0086
force_track_0 loss delta: +0.0052
zero_selected loss delta: +1.8103
```

The destructive-test output shows rotate and force-track perturbations barely change loss, while zeroing the selected pathway still causes a large loss increase.

## Interpretation

```text
The selected QKV pathway is live.
But track identity is nearly interchangeable.
```

That is the core point.

This variant is not interesting because it performs well. It is interesting because it creates a contrast class:

```text
same number of QKV tracks
same global-bank idea
different routing schedule
different specialization regime
```

Static-global produces route identity dependence. Train-rotation largely destroys route identity dependence while preserving dependence on the existence of the active pathway.

---

# 3. Multi-QKV position-rotation 3-track global

This is the most conceptually important E002 variant.

It keeps the global 3-track QKV bank, but makes routing depend on token position as well as layer.

The routing rule is:

```text
active_track = (layer_idx + position) % track_count
```

The position-rotation config uses `attention_type: multi_qkv_position_rotation_3track_global`, `qkv_track_count: 3`, `qkv_global_bank: true`, and `qkv_route_formula: layer_plus_position`.

## Position-rotation dataflow

```text
Input residual stream
x: [batch, seq, d_model]
        |
        v
Global Multi-QKV projection bank
        |
        v
+---------------------------------------------------+
| Track 0: [Q0 | K0 | V0]                           |
| Track 1: [Q1 | K1 | V1]                           |
| Track 2: [Q2 | K2 | V2]                           |
+---------------------------------------------------+
        |
        v
Select track by layer and token position:

    active_track(token position p, layer l)
        = (l + p) % 3

        |
        v
For each token position, use the selected Q/K/V track
        |
        v
causal scaled dot-product attention
Attention(Q_selected_by_position, K_selected_by_position, V_selected_by_position)
        |
        v
output projection: c_proj
        |
        v
residual write
```

## Position-level routing example

For a single layer, different token positions route to different tracks.

```text
Layer 0:

position 0 -> track 0
position 1 -> track 1
position 2 -> track 2
position 3 -> track 0
position 4 -> track 1
position 5 -> track 2
...

Layer 1:

position 0 -> track 1
position 1 -> track 2
position 2 -> track 0
position 3 -> track 1
position 4 -> track 2
position 5 -> track 0
...

Layer 2:

position 0 -> track 2
position 1 -> track 0
position 2 -> track 1
position 3 -> track 2
position 4 -> track 0
position 5 -> track 1
...
```

## Position-rotation diagram

```text
Input residual stream
x: [B, T, d_model]
        |
        v
Global QKV track bank
        |
        v
+------------------------------------------------+
| T0: Q0,K0,V0                                   |
| T1: Q1,K1,V1                                   |
| T2: Q2,K2,V2                                   |
+------------------------------------------------+
        |
        v
For layer l and position p:

        t = (l + p) % 3

        |
        v

Token positions inside one layer:

p0       p1       p2       p3       p4       p5
|        |        |        |        |        |
v        v        v        v        v        v
T0       T1       T2       T0       T1       T2     for layer 0

        |
        v
Position-conditioned Q/K/V selection
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

## What this architecture changes

Standard attention:

```text
one QKV stream per layer
```

Static-global Multi-QKV:

```text
one selected QKV track per layer
```

Position-rotation Multi-QKV:

```text
QKV track identity varies across token positions inside a layer
```

That is the important distinction.

This variant directly questions the static stream assumption:

```text
Does every token position in a layer need to use the same QKV stream?
```

## Diagnostic signature

The diagnostics show balanced active track counts across a 1024-token sequence. At step 3000, position-rotation has counts like `{0:342, 1:341, 2:341}`, `active_track_index: null`, `track_entropy` around `1.0986`, and route formula `(layer_idx + position) % track_count`.

That is the expected signature for per-position routing across three tracks.

## Destructive-test result

Position-rotation is intermediate between static-global and train-rotation:

```text
rotate_tracks loss delta: +0.2065
force_track_0 loss delta: +0.8394
zero_selected loss delta: +1.9672
```

The destructive test shows all three perturbations are active, with moderate route-swap sensitivity and strong zero-selected sensitivity.

## Interpretation

```text
The selected pathway is live.
Track identity matters more than in train-rotation.
Track identity matters less than in static-global.
The model tolerates position-conditioned QKV routing without collapsing.
```

This is the most relevant E002 result for the original research direction because it moves beyond a static per-layer stream.

---

# 4. Side-by-side architectural comparison

```text
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

```text
STATIC-GLOBAL MULTI-QKV
=======================

x
|
v
Global track bank:
[T0: Q0 K0 V0]
[T1: Q1 K1 V1]
[T2: Q2 K2 V2]
|
v
select by layer:

    t = layer_idx % 3

|
v
Attention(Q_t,K_t,V_t)
|
v
c_proj
|
v
residual write
```

```text
TRAIN-ROTATION MULTI-QKV
========================

x
|
v
Global track bank:
[T0: Q0 K0 V0]
[T1: Q1 K1 V1]
[T2: Q2 K2 V2]
|
v
select by layer and training step:

    train: t = (layer_idx + step) % 3
    eval:  t = layer_idx % 3

|
v
Attention(Q_t,K_t,V_t)
|
v
c_proj
|
v
residual write
```

```text
POSITION-ROTATION MULTI-QKV
===========================

x
|
v
Global track bank:
[T0: Q0 K0 V0]
[T1: Q1 K1 V1]
[T2: Q2 K2 V2]
|
v
select by layer and token position:

    t = (layer_idx + position) % 3

|
v
Per-position selected Q/K/V streams
|
v
Attention(Q_t(p),K_t(p),V_t(p))
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

```text
one layer-local QKV read/write stream
```

The model can represent many effects, but there is no explicit alternative stream to inspect.

## Static-global Multi-QKV

```text
multiple global QKV tracks
layer chooses track
track identity can specialize
```

Best for asking:

```text
Do different QKV tracks learn different functions?
What happens when a layer is forced to use the wrong track?
Are route identities causally meaningful?
```

## Train-rotation Multi-QKV

```text
multiple global QKV tracks
training rotates layer-track assignment
track identity is pressured toward interchangeability
```

Best for asking:

```text
Can routing schedules suppress track specialization?
Does rotating assignment create shared/redundant track competence?
What structure remains when track identity stops mattering?
```

## Position-rotation Multi-QKV

```text
multiple global QKV tracks
token position participates in route selection
different positions inside the same layer use different QKV tracks
```

Best for asking:

```text
Can attention use multiple position-conditioned QKV streams?
Do tracks specialize by position class, token role, or local sequence function?
Does per-position routing expose structure hidden by static per-layer QKV?
```

---

# 6. Canonical E002 interpretation

The important E002 result is not that a variant beats standard attention.

The important result is:

```text
A GPT-style model can tolerate multiple global QKV tracks and nonstandard routing rules during real pretraining.
```

The routing rule changes the kind of internal pathway structure that emerges:

```text
static-global:
    strong track specialization

train-rotation:
    near track interchangeability

position-rotation:
    intermediate track dependence with position-conditioned routing
```

The destructive tests give the cleanest architectural signature:

```text
Architecture          rotate Δ    force0 Δ    zero-selected Δ    interpretation
--------------------------------------------------------------------------------
static-global         +3.1309     +1.8131     +2.1382            route-specialized
train-rotation        +0.0086     +0.0052     +1.8103            track-interchangeable, pathway-live
position-rotation     +0.2065     +0.8394     +1.9672            intermediate, position-conditioned
```

The zero-selected perturbation matters because it shows the pathway is live.

The rotate/force perturbations matter because they show whether the identity of the selected track matters.

So the E002 architecture takeaway is:

```text
The QKV stream is a viable research surface.

It can be split into multiple tracks.
Those tracks can be routed by layer, training step, or token position.
The resulting systems remain trainable.
Different routing rules produce different degrees of track specialization.
```

That is the canonical E002 diagram-level story.
