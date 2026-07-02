Below are **E001 only**: baseline/control plus the implemented CP score-augmentation variants: `cp_bilinear` and `cp_trilinear`.

E001 is different from E002/E003/E004: it keeps the standard attention value read, but **adds an extra low-rank CP-derived term directly into the attention score matrix before softmax**. The common base class describes this as GPT attention with an additive low-rank CP score branch. 

---

## 0. Control / baseline: `standard_30m_seed1`

The E001 baseline is ordinary GPT causal self-attention: packed QKV projection, causal scaled dot-product attention, merge heads, output projection. 

```text id="sslwy9"
STANDARD GPT CAUSAL SELF-ATTENTION
==================================

Input residual stream
x : [B, T, C]
        |
        v
+-------------------------------+
| Linear c_attn: C -> 3C        |
| produces packed Q, K, V       |
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
          | scores = QK^T / sqrt(D)           |
          | softmax(scores) V                 |
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

```text id="qy059w"
x -> Q,K,V -> standard_scores -> causal softmax -> attention @ V -> c_proj -> output
```

---

## Shared E001 mechanism: CP score augmentation

Both CP variants preserve the standard QKV path and add an extra score branch:

```text id="op7fel"
scores = standard_scores + cp_lambda * cp_score
```

The common implementation computes standard scores from `QK^T / sqrt(head_size)`, computes `extra_scores`, multiplies by `cp_lambda`, adds that to the standard scores, applies the causal mask and softmax, then reads from the ordinary V stream. 

```text id="7euyp5"
COMMON CP-AUGMENTED ATTENTION
=============================

Input residual stream
x : [B,T,C]
        |
        +---------------------------------------------------+
        |                                                   |
        v                                                   v
+-------------------------------+              +----------------------------+
| Standard QKV projection       |              | Low-rank CP projections    |
| c_attn: C -> 3C               |              | q_low, k_low, maybe v_low  |
+-------------------------------+              +----------------------------+
        |                                                   |
        v                                                   v
      Q,K,V                                          low-rank factors
        |                                                   |
        v                                                   v
standard_scores = QK^T/sqrt(D)                  cp_score branch
        |                                                   |
        +-------------------------+-------------------------+
                                  |
                                  v
                 cp_output = cp_lambda * cp_score
                                  |
                                  v
              final_scores = standard_scores + cp_output
                                  |
                                  v
                           causal mask
                                  |
                                  v
                              softmax
                                  |
                                  v
                           attention @ V
                                  |
                                  v
                               c_proj
                                  |
                                  v
                              output
```

Another compact form:

```text id="k18nq3"
                 +--> Q,K,V ---------> standard_scores = QK^T/sqrt(D) --+
x --------------+                                                        +--> final_scores
                 +--> low-rank factors -> cp_score ---------------------+
                                             |
                                             v
                                      cp_lambda * cp_score

final_scores -> causal softmax -> attention @ V -> c_proj -> output
```

---

## 1. E001: `cp_bilinear`

`cp_bilinear` uses two low-rank factors, `q_low` and `k_low`, to add a bilinear low-rank score matrix. The implementation’s extra score formula is:

```text id="sjjg1f"
cp_score[i,j] = sum_r q_low[i,r] * k_low[j,r] / sqrt(cp_rank)
```

In code, it is `einsum("bhir,bhjr->bhij", q_low, k_low) * (1/sqrt(cp_rank))`. 

```text id="d1uf9h"
CP-BILINEAR ATTENTION
=====================

Input residual stream
x : [B,T,C]
        |
        +---------------------------------------------+
        |                                             |
        v                                             v
+-------------------------------+        +-----------------------------+
| Standard QKV projection       |        | Low-rank projections        |
| c_attn: C -> 3C               |        | q_low: C -> H*R             |
|                               |        | k_low: C -> H*R             |
+-------------------------------+        +-----------------------------+
        |                                             |
        v                                             v
      Q,K,V                                  q_low, k_low
        |                                  [B,H,T,R]
        |                                             |
        v                                             v
standard_scores                       cp_bilinear_score
QK^T / sqrt(D)                        q_low @ k_low^T / sqrt(R)
[B,H,T,T]                             [B,H,T,T]
        |                                             |
        |                                             v
        |                              cp_output = cp_lambda * cp_score
        |                                             |
        +-----------------------------+---------------+
                                      |
                                      v
                    final_scores = standard_scores + cp_output
                                      |
                                      v
                              causal mask + softmax
                                      |
                                      v
                                  attention @ V
                                      |
                                      v
                                    c_proj
                                      |
                                      v
                                    output
```

Compact:

```text id="m9qt01"
x
 -> Q,K,V -------------------------------> standard_scores
 -> q_low,k_low -> q_low*k_low CP scores -> cp_lambda * cp_score
 -> standard_scores + cp_output
 -> softmax
 -> attention @ V
 -> c_proj
 -> output
```

Interpretation:

```text id="18nkm0"
baseline:
  attention compatibility comes only from Q dot K

cp_bilinear:
  attention compatibility = Q dot K + lambda * low_rank(Q_low dot K_low)
```

So CP-bilinear asks whether a low-rank score-side correction can learn useful extra pairwise token compatibility without replacing the normal value read.

---

## 2. E001: `cp_trilinear`

`cp_trilinear` uses three low-rank factors: `q_low`, `k_low`, and `v_low`. But importantly, this `v_low` does **not** replace the ordinary V used after softmax. It conditions the **score branch**. The implementation computes:

```text id="xtskgv"
cp_score[i,j] = sum_r q_low[i,r] * k_low[j,r] * v_low[j,r] / sqrt(cp_rank)
```

In code, it is `einsum("bhir,bhjr,bhjr->bhij", q_low, k_low, v_low) * (1/sqrt(cp_rank))`. 

```text id="2v2348"
CP-TRILINEAR ATTENTION
======================

Input residual stream
x : [B,T,C]
        |
        +------------------------------------------------------+
        |                                                      |
        v                                                      v
+-------------------------------+              +-----------------------------+
| Standard QKV projection       |              | Low-rank projections        |
| c_attn: C -> 3C               |              | q_low: C -> H*R             |
|                               |              | k_low: C -> H*R             |
|                               |              | v_low: C -> H*R             |
+-------------------------------+              +-----------------------------+
        |                                                      |
        v                                                      v
      Q,K,V                                      q_low, k_low, v_low
        |                                            [B,H,T,R]
        |                                                      |
        v                                                      v
standard_scores                            cp_trilinear_score
QK^T / sqrt(D)                             q_low[i] * k_low[j] * v_low[j]
[B,H,T,T]                                  summed over rank r
        |                                  [B,H,T,T]
        |                                                      |
        |                                      cp_output = cp_lambda * cp_score
        |                                                      |
        +-----------------------------+------------------------+
                                      |
                                      v
                    final_scores = standard_scores + cp_output
                                      |
                                      v
                              causal mask + softmax
                                      |
                                      v
                                  attention @ V
                                      |
                                      v
                                    c_proj
                                      |
                                      v
                                    output
```

Compact:

```text id="6p5v5b"
x
 -> Q,K,V ----------------------------------------> standard_scores
 -> q_low,k_low,v_low -> trilinear CP score ------> cp_lambda * cp_score
 -> standard_scores + cp_output
 -> softmax
 -> attention @ ordinary V
 -> c_proj
 -> output
```

The key distinction from CP-bilinear:

```text id="i0gzr3"
CP-bilinear:
  score_extra(i,j) = q_low(i) * k_low(j)

CP-trilinear:
  score_extra(i,j) = q_low(i) * k_low(j) * v_low(j)
```

So CP-trilinear asks whether **the key/value-side token can contribute a low-rank value-conditioned compatibility term before softmax**.

---

## 3. E001 lambda-zero ablation: `cp_trilinear_r8_lambda0_30m_seed1`

There is a config for a fixed-lambda-zero trilinear CP run: `cp_lambda_init: 0.0`, `cp_lambda_trainable: false`, `cp_lambda_fixed: true`. 

Architecturally, that means the CP branch exists but its score contribution is multiplied by zero:

```text id="ws4opz"
CP-TRILINEAR LAMBDA0 ABLATION
=============================

x
 -> Q,K,V -------------------------------------> standard_scores
 -> q_low,k_low,v_low -> cp_trilinear_score ---+
                                               |
                                               v
                                  cp_output = 0.0 * cp_score
                                             = 0
                                               |
                                               v
                         final_scores = standard_scores + 0
                                      = standard_scores
                                               |
                                               v
                                      behaves like baseline
                                      except CP parameters/branch exist
                                      but do not affect attention scores
```

However, in the current backfill report, that specific historical run is **not evaluated** because its checkpoint is unavailable. The report keeps `cp_trilinear_r8_lambda0_30m_seed1` under `not_evaluated`, not CP diagnostic follow-up. 

So diagrammatically it exists as a design/config, but scientifically you should not treat its historical result as available evidence.

---

## Side-by-side compression

```text id="svedu3"
BASELINE / CONTROL
------------------
x
 -> Q,K,V
 -> standard_scores = QK^T / sqrt(D)
 -> causal softmax
 -> attention @ V
 -> c_proj
 -> output


CP_BILINEAR
-----------
x
 -> Q,K,V
 -> standard_scores = QK^T / sqrt(D)

x
 -> q_low,k_low
 -> cp_score = q_low(i) * k_low(j) / sqrt(R)

final_scores = standard_scores + cp_lambda * cp_score
 -> causal softmax
 -> attention @ V
 -> c_proj
 -> output


CP_TRILINEAR
------------
x
 -> Q,K,V
 -> standard_scores = QK^T / sqrt(D)

x
 -> q_low,k_low,v_low
 -> cp_score = q_low(i) * k_low(j) * v_low(j) / sqrt(R)

final_scores = standard_scores + cp_lambda * cp_score
 -> causal softmax
 -> attention @ ordinary V
 -> c_proj
 -> output


CP_TRILINEAR_LAMBDA0 ABLATION
-----------------------------
x
 -> Q,K,V
 -> standard_scores

x
 -> q_low,k_low,v_low
 -> cp_score
 -> cp_lambda = 0
 -> cp_output = 0

final_scores = standard_scores
 -> causal softmax
 -> attention @ V
 -> c_proj
 -> output
```

## One combined overview

```text id="5uzyba"
                              E001 CP ATTENTION FAMILY
                              ========================

                                      x
                                      |
        +-----------------------------+-----------------------------+
        |                             |                             |
        v                             v                             v

  BASELINE CONTROL              CP-BILINEAR                  CP-TRILINEAR
  ----------------              -----------                  ------------

  Q,K,V                         Q,K,V                        Q,K,V
    |                              |                            |
    v                              v                            v
  QK scores                      QK scores                    QK scores
    |                              |                            |
    v                              v                            v
  softmax                        + cp branch                   + cp branch
    |                              |                            |
    v                              v                            v
  attention @ V                  q_low,k_low                  q_low,k_low,v_low
    |                              |                            |
    v                              v                            v
  c_proj                         q_low*k_low                  q_low*k_low*v_low
    |                              |                            |
    v                              |                            |
  output                         v                            v
                         cp_lambda * cp_score        cp_lambda * cp_score
                                  |                            |
                                  v                            v
                         QK scores + CP scores       QK scores + CP scores
                                  |                            |
                                  v                            v
                               softmax                      softmax
                                  |                            |
                                  v                            v
                             attention @ V              attention @ V
                                  |                            |
                                  v                            v
                                c_proj                       c_proj
                                  |                            |
                                  v                            v
                                output                       output
```

## Mechanistic interpretation

```text id="n6tg2b"
STANDARD:
  "Attention score is pure query-key compatibility."

CP_BILINEAR:
  "Attention score is query-key compatibility plus a learned low-rank pairwise correction."

CP_TRILINEAR:
  "Attention score is query-key compatibility plus a learned low-rank correction conditioned by a value-side/key-position factor."

CP_TRILINEAR_LAMBDA0:
  "The CP score branch is structurally present but causally silenced by lambda=0."
```

My read: **E001 is the score-side perturbation family.** E002 perturbs QKV routing, E003 introduces competing/scoping branches, E004 adds richer operator/role/value mechanisms. E001 is the cleanest place to ask: “Can a low-rank additive score branch do anything meaningful without changing the value read path?”
