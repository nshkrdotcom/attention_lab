Below are **E002 only**: the control/baseline plus the canonical implemented Multi-QKV variants from E002.

E002 is not “new attention math” in the same way E003/E004 are. It is mostly about **using a global bank of multiple QKV projection tracks** and then routing each layer, step, or position to a selected QKV track before doing ordinary causal attention.

The implemented canonical variants are:

```text id="o25x73"
multi_qkv_static_3track_global
multi_qkv_train_rotation_3track_global
multi_qkv_position_rotation_3track_global
```

The common base enforces canonical 3-track global-bank configs and recognizes exactly those three attention types. 

---

## 0. Control / baseline: `standard_refactor_control`

The control is standard GPT causal self-attention: one QKV projection per layer, then causal attention, then output projection. 

```text id="omuhvq"
STANDARD GPT CAUSAL SELF-ATTENTION
==================================

Input residual stream
x : [B, T, C]
        |
        v
+-------------------------------+
| Layer-local Linear c_attn     |
| C -> 3C                       |
| produces Q, K, V              |
+-------------------------------+
        |
        +------------------+------------------+
        |                  |                  |
        v                  v                  v
        Q                  K                  V
   [B,H,T,D]          [B,H,T,D]          [B,H,T,D]
        |                  |                  |
        +------------------+------------------+
                           |
                           v
          +-----------------------------------+
          | causal scaled dot-product attn    |
          | softmax(QK^T / sqrt(D)) V         |
          +-----------------------------------+
                           |
                           v
                    attention heads
                           |
                           v
                     merge heads
                      [B,T,C]
                           |
                           v
                 +----------------+
                 | Linear c_proj  |
                 | C -> C         |
                 +----------------+
                           |
                           v
                    dropout / output
```

Compact:

```text id="uvbczz"
x -> layer-local Q,K,V -> causal attention -> c_proj -> output
```

---

## Shared E002 mechanism: global 3-track QKV bank

All canonical E002 variants use the same base mechanism:

```text id="cf85qz"
GLOBAL MULTI-QKV BANK
=====================

                 shared across Multi-QKV layers
                 not one c_attn per layer

        +------------------------------+
        | Global QKV Track Bank        |
        | track_count = 3              |
        +------------------------------+
              |             |             |
              v             v             v
       +-------------+ +-------------+ +-------------+
       | Track 0     | | Track 1     | | Track 2     |
       | Linear      | | Linear      | | Linear      |
       | C -> 3C     | | C -> 3C     | | C -> 3C     |
       +-------------+ +-------------+ +-------------+
```

The shared bank is a `ModuleList` of QKV projections, one per track. Each track maps `C -> 3C`. 

The base forward path is:

```text id="qvbuh6"
Input x
  |
  v
compute active track(s)
  |
  v
select QKV projection from global bank
  |
  v
split selected QKV into Q,K,V
  |
  v
ordinary causal attention
  |
  v
c_proj
  |
  v
output
```

More explicitly:

```text id="lvt4vw"
MULTI-QKV BASE FORWARD
======================

Input residual stream
x : [B, T, C]
        |
        v
+------------------------------------+
| routing rule chooses active track  |
| scalar track: one track for layer  |
| vector track: one track per token  |
+------------------------------------+
        |
        v
+------------------------------------+
| project x through selected track(s)|
| from global QKV bank               |
+------------------------------------+
        |
        v
selected_qkv : [B, T, 3C]
        |
        +------------------+------------------+
        |                  |                  |
        v                  v                  v
        Q                  K                  V
   [B,H,T,D]          [B,H,T,D]          [B,H,T,D]
        |                  |                  |
        +------------------+------------------+
                           |
                           v
+------------------------------------------------+
| manual causal attention                        |
| scores = QK^T / sqrt(D)                        |
| causal mask                                    |
| softmax(scores)                                |
| y = attention @ V                              |
+------------------------------------------------+
                           |
                           v
                     merge heads
                      [B,T,C]
                           |
                           v
                    Linear c_proj
                           |
                           v
                         output
```

The base implementation computes active tracks, records `selected_track`, projects either one scalar selected track or all tracks followed by per-position selection, then performs manual causal attention and records `track_out`. 

---

## 1. E002: `multi_qkv_static_3track_global`

Static routing chooses track by layer index:

```text id="pqbyfn"
selected_track = layer_idx % track_count
```

The implementation’s route formula is exactly `layer_idx % track_count`. 

```text id="5iqxd6"
MULTI-QKV STATIC 3-TRACK GLOBAL
===============================

Global bank:
        Track 0: Wqkv_0
        Track 1: Wqkv_1
        Track 2: Wqkv_2

Routing rule:
        selected_track = layer_idx % 3

Example layer routing:
        layer 0 -> track 0
        layer 1 -> track 1
        layer 2 -> track 2
        layer 3 -> track 0
        layer 4 -> track 1
        layer 5 -> track 2

Per layer forward:
        x
        |
        v
+-------------------------------+
| selected_track = layer % 3    |
+-------------------------------+
        |
        v
+-------------------------------+
| Global QKV Bank               |
| choose one track              |
+-------------------------------+
        |
        +-------- track 0
        |        track 1
        |        track 2
        |
        v
 selected Wqkv_track(x)
        |
        v
      Q,K,V
        |
        v
 causal attention
        |
        v
     c_proj
        |
        v
     output
```

Compact:

```text id="pe8r8n"
x
 -> route by layer_idx % 3
 -> select global QKV track
 -> Q,K,V
 -> causal attention
 -> c_proj
 -> output
```

This is the cleanest E002 workbench variant: **different layers are forced to use different global QKV projection tracks.**

---

## 2. E002: `multi_qkv_train_rotation_3track_global`

Train-rotation routing changes the active track over training steps, but freezes to static layer routing during eval/generation.

During training:

```text id="2bb5ph"
selected_track = (layer_idx + step) % track_count
```

During eval/generate:

```text id="4f79h3"
selected_track = layer_idx % track_count
```

The implementation uses exactly that rule and requires `step` during training. 

```text id="rwevx6"
MULTI-QKV TRAIN-ROTATION 3-TRACK GLOBAL
=======================================

Global bank:
        Track 0: Wqkv_0
        Track 1: Wqkv_1
        Track 2: Wqkv_2

Training route:
        selected_track = (layer_idx + step) % 3

Eval/generate route:
        selected_track = layer_idx % 3

Training example, layer 1:
        step 0 -> (1 + 0) % 3 = track 1
        step 1 -> (1 + 1) % 3 = track 2
        step 2 -> (1 + 2) % 3 = track 0
        step 3 -> (1 + 3) % 3 = track 1

Forward:
        x
        |
        v
+------------------------------------------+
| if train:                                |
|   selected_track = (layer_idx + step)%3  |
| if eval/generate:                        |
|   selected_track = layer_idx % 3         |
+------------------------------------------+
        |
        v
+-------------------------------+
| Global QKV Bank               |
| choose one active track       |
+-------------------------------+
        |
        v
 selected Wqkv_track(x)
        |
        v
      Q,K,V
        |
        v
 causal attention
        |
        v
     c_proj
        |
        v
     output
```

Compact:

```text id="wg50ue"
TRAIN:
x -> route by (layer_idx + step) % 3 -> selected QKV track -> attention -> output

EVAL/GENERATE:
x -> route by layer_idx % 3          -> selected QKV track -> attention -> output
```

The idea is: **during training, each layer sees rotating QKV parameterizations; at eval, routing freezes into a deterministic layer-wise assignment.**

---

## 3. E002: `multi_qkv_position_rotation_3track_global`

Position-rotation routing chooses track per token position, not one scalar track per layer. The formula is:

```text id="rhy8q4"
selected_track[t] = (layer_idx + position[t]) % track_count
```

The implementation enables position routing and computes `(layer_idx + position_ids) % track_count`. 

```text id="z4ccas"
MULTI-QKV POSITION-ROTATION 3-TRACK GLOBAL
==========================================

Global bank:
        Track 0: Wqkv_0
        Track 1: Wqkv_1
        Track 2: Wqkv_2

Routing rule:
        selected_track[position] = (layer_idx + position) % 3

Example, layer 0:
        token position 0 -> track 0
        token position 1 -> track 1
        token position 2 -> track 2
        token position 3 -> track 0
        token position 4 -> track 1
        token position 5 -> track 2

Example, layer 1:
        token position 0 -> track 1
        token position 1 -> track 2
        token position 2 -> track 0
        token position 3 -> track 1
        token position 4 -> track 2
        token position 5 -> track 0

Forward:
        x : [B,T,C]
        |
        v
+----------------------------------------------+
| active_tracks[t] = (layer_idx + position)%3  |
| one selected track per token position        |
+----------------------------------------------+
        |
        v
+----------------------------------------------+
| project x through all global tracks          |
| Wqkv_0(x), Wqkv_1(x), Wqkv_2(x)              |
+----------------------------------------------+
        |
        v
+----------------------------------------------+
| gather per-position selected projection      |
| token t receives Wqkv_active_tracks[t](x_t)  |
+----------------------------------------------+
        |
        v
 selected_qkv : [B,T,3C]
        |
        v
      Q,K,V
        |
        v
 causal attention
        |
        v
     c_proj
        |
        v
     output
```

Compact:

```text id="wk8scr"
x
 -> compute selected_track for each position
 -> project x through all QKV tracks
 -> gather per-token selected QKV projection
 -> Q,K,V
 -> causal attention
 -> c_proj
 -> output
```

This is the most mechanistically interesting E002 variant because **routing varies within a sequence**, not just across layers or training steps.

---

## Side-by-side compression

```text id="6fu6ve"
BASELINE / CONTROL
------------------
x
 -> one layer-local QKV projection
 -> Q,K,V
 -> causal attention
 -> c_proj
 -> output


MULTI_QKV_STATIC_3TRACK_GLOBAL
------------------------------
x
 -> selected_track = layer_idx % 3
 -> global QKV bank[selected_track]
 -> Q,K,V
 -> causal attention
 -> c_proj
 -> output


MULTI_QKV_TRAIN_ROTATION_3TRACK_GLOBAL
--------------------------------------
train:
  x
   -> selected_track = (layer_idx + step) % 3
   -> global QKV bank[selected_track]
   -> Q,K,V
   -> causal attention
   -> c_proj
   -> output

eval/generate:
  x
   -> selected_track = layer_idx % 3
   -> global QKV bank[selected_track]
   -> Q,K,V
   -> causal attention
   -> c_proj
   -> output


MULTI_QKV_POSITION_ROTATION_3TRACK_GLOBAL
-----------------------------------------
x
 -> selected_track[t] = (layer_idx + position[t]) % 3
 -> project through all global tracks
 -> gather selected QKV per token position
 -> Q,K,V
 -> causal attention
 -> c_proj
 -> output
```

## One combined overview

```text id="x3u4fj"
                              E002 MULTI-QKV FAMILY
                              =====================

                                      x
                                      |
        +-----------------------------+------------------------------+
        |                             |                              |
        v                             v                              v

  BASELINE CONTROL              STATIC GLOBAL                 TRAIN ROTATION
  ----------------              -------------                 --------------

  layer-local Wqkv              global bank                   global bank
  C -> 3C                       Wqkv_0,Wqkv_1,Wqkv_2           Wqkv_0,Wqkv_1,Wqkv_2
        |                             |                              |
        v                             v                              v
      Q,K,V                  track = layer % 3          train: track=(layer+step)%3
        |                             |                  eval:  track=layer%3
        v                             v                              |
   causal attn                selected Wqkv                         v
        |                             |                       selected Wqkv
        v                             v                              |
      c_proj                        Q,K,V                            v
        |                             |                             Q,K,V
        v                             v                              |
      output                    causal attn                          v
                                      |                         causal attn
                                      v                              |
                                    c_proj                           v
                                      |                            c_proj
                                      v                              |
                                    output                           v
                                                                   output


                                      x
                                      |
                                      v
                         POSITION ROTATION GLOBAL
                         ------------------------

                         global bank Wqkv_0,Wqkv_1,Wqkv_2
                                      |
                                      v
               selected_track[t] = (layer_idx + position[t]) % 3
                                      |
                                      v
                         project x through all tracks
                                      |
                                      v
                         gather per-token selected QKV
                                      |
                                      v
                                    Q,K,V
                                      |
                                      v
                                causal attn
                                      |
                                      v
                                    c_proj
                                      |
                                      v
                                    output
```

## Mechanistic interpretation

```text id="y46fli"
STANDARD:
  "Every layer owns one QKV map."

E002 STATIC:
  "Layers select among a shared set of QKV maps."

E002 TRAIN ROTATION:
  "Layers cycle through QKV maps during training, then freeze to static routing."

E002 POSITION ROTATION:
  "Tokens at different positions select different QKV maps inside the same layer."
```

My read: **E002 is a route-specialization workbench**, not a fundamentally new attention operator. Its most useful role is to test whether QKV projection banks develop track-specific specialization and whether interventions like route replacement, forced-track, rotate-track, and zero-selected-track produce interpretable failures.
