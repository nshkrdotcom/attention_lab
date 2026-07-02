Below are **E003 only**: the control/baseline plus the two E003 architecture variants: `differential_qkv_anti_value` and `scope_gated_qkv`.

## 0. Control / baseline: `standard_refactor_control`

E003 uses the same standard GPT causal self-attention baseline: one packed QKV projection, one causal SDPA call, merge heads, then output projection. 

```text id="9mvjqq"
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

```text id="msj4n8"
x -> Q,K,V -> causal attention -> merge -> c_proj -> output
```

---

## 1. E003: `differential_qkv_anti_value`

This variant creates two attention branches: a positive branch and a negative/subtractive branch. It computes both branches independently, then subtracts the negative branch scaled by a learned positive `lambda`. If `diff_qkv_share_value` is enabled, the branches share V; otherwise they have separate positive and negative V streams. 

```text id="wo1t8a"
DIFFERENTIAL QKV ANTI-VALUE ATTENTION
=====================================

Input residual stream
x : [B, T, C]
        |
        v
+--------------------------------------------+
| Linear c_attn                              |
| if share_value = false: C -> 6C            |
|   Q_pos, K_pos, V_pos, Q_neg, K_neg, V_neg |
| if share_value = true:  C -> 5C            |
|   Q_pos, K_pos, V_shared, Q_neg, K_neg     |
+--------------------------------------------+
        |
        +-------------------------------+-------------------------------+
        |                               |                               |
        v                               v                               v
  POSITIVE BRANCH                  NEGATIVE BRANCH              learned lambda
  ---------------                  ---------------              --------------
  Q_pos, K_pos, V_pos              Q_neg, K_neg, V_neg          lambda_raw
        |                               |                            |
        v                               v                            v
+----------------------+        +----------------------+        softplus()
| causal attention     |        | causal attention     |             |
| Q_pos,K_pos,V_pos    |        | Q_neg,K_neg,V_neg    |             v
+----------------------+        +----------------------+       lambda >= 0
        |                               |
        v                               v
    pos_out                         neg_out
     [B,T,C]                         [B,T,C]
        |                               |
        +---------------+---------------+
                        |
                        v
          +------------------------------------+
          | branch_delta = pos_out             |
          |              - lambda * neg_out    |
          +------------------------------------+
                        |
                        v
                  Linear c_proj
                        |
                        v
                 dropout / output
```

Compact:

```text id="fn6weu"
x
 -> [Q+,K+,V+] -> causal attention -> pos_out
 -> [Q-,K-,V-] -> causal attention -> neg_out
 -> lambda = softplus(lambda_raw)
 -> pos_out - lambda * neg_out
 -> c_proj
 -> output
```

With shared V:

```text id="w1328p"
x -> Q+,K+,V_shared,Q-,K-
       |      |          |
       |      |          +-- used as V+ and V-
       |      |
       v      v
 positive attn     negative attn
       |              |
       v              v
    pos_out       neg_out
       \            /
        \          /
         v        v
      pos_out - lambda * neg_out
              |
              v
            output
```

The implementation records the positive and negative Q/K/V streams, the learned `lambda`, `pos_out`, `neg_out`, and `branch_delta`. 

Core difference from baseline:

```text id="g3cpz7"
BASELINE:
  one attention result

DIFFERENTIAL:
  positive attention result - learned_scale * negative attention result
```

---

## 2. E003: `scope_gated_qkv`

This variant adds a fourth stream, `scope`, alongside Q/K/V. It performs two attention reads using the same Q/K addressing: one into normal V content and one into the scope stream. Then it forms a content-scope product and a receiver-side gated content stream, concatenates four components, and projects back to C. 

```text id="rvnqzx"
SCOPE-GATED QKV ATTENTION
=========================

Input residual stream
x : [B, T, C]
        |
        v
+-------------------------------+
| Linear c_attn: C -> 4C        |
| produces Q, K, V, Scope       |
+-------------------------------+
        |
        +-------------+-------------+-------------+-------------+
        |             |             |             |             |
        v             v             v             v
        Q             K             V           Scope
   [B,H,T,D]     [B,H,T,D]     [B,H,T,D]     [B,H,T,D]
        |             |             |             |
        |             |             |             |
        +------+------+             |             |
               |                    |             |
               v                    v             v
     +---------------------+   +---------------------+
     | content attention   |   | scope attention     |
     | Q,K,V               |   | Q,K,Scope           |
     +---------------------+   +---------------------+
               |                    |
               v                    v
          content_out            scope_out
           [B,T,C]               [B,T,C]
               |                    |
               |                    v
               |              scope_scale * scope_out
               |                    |
               |                    v
               |             scaled_scope_out
               |                    |
               +----------+---------+
                          |
                          v
              content_scope_product
              content_out * scaled_scope_out
                          |
                          |
Input x ------------------+--------------------+
                                               |
                                               v
                                    +-------------------+
                                    | Linear c_gate     |
                                    | sigmoid           |
                                    +-------------------+
                                               |
                                               v
                                             gate
                                           [B,T,C]
                                               |
                                               v
                                     gated_content
                                   gate * content_out
                                               |
                          +--------------------+
                          |
                          v
+-------------------------------------------------------------+
| concatenate:                                                |
| [content_out, scaled_scope_out,                             |
|  content_scope_product, gated_content]                      |
| shape [B,T,4C]                                              |
+-------------------------------------------------------------+
                          |
                          v
                   Linear c_proj: 4C -> C
                          |
                          v
                   dropout / output
```

Compact:

```text id="v0ajdu"
x -> Q,K,V,Scope

Q,K,V     -> causal attention -> content_out
Q,K,Scope -> causal attention -> scope_out -> scope_scale * scope_out

content_scope_product = content_out * scope_out
gate = sigmoid(c_gate(x))
gated_content = gate * content_out

concat(content_out,
       scope_out,
       content_scope_product,
       gated_content)
 -> c_proj
 -> output
```

The implementation records `content_out`, `scope_out`, `gate`, `content_scope_product`, and `gated_content`. 

Core difference from baseline:

```text id="okx5se"
BASELINE:
  Q,K,V -> one content read -> c_proj

SCOPE-GATED:
  Q,K,V     -> content read
  Q,K,Scope -> scope read
  x         -> receiver gate
  [content, scope, content*scope, gate*content] -> c_proj
```

---

## Side-by-side compression

```text id="y87w7p"
BASELINE / CONTROL
------------------
x
 -> Q,K,V
 -> causal attention
 -> content
 -> c_proj
 -> output


DIFFERENTIAL_QKV_ANTI_VALUE
---------------------------
x
 -> Q+,K+,V+  -> causal attention -> pos_out
 -> Q-,K-,V-  -> causal attention -> neg_out
 -> lambda = softplus(lambda_raw)
 -> pos_out - lambda * neg_out
 -> c_proj
 -> output


SCOPE_GATED_QKV
---------------
x
 -> Q,K,V,Scope
 -> Q,K,V       -> causal attention -> content_out
 -> Q,K,Scope   -> causal attention -> scope_out
 -> content_out * scope_out
 -> sigmoid(c_gate(x)) * content_out
 -> concat(content, scope, product, gated_content)
 -> c_proj
 -> output
```

## One combined overview

```text id="e9axsu"
                             E003 ATTENTION FAMILY
                             =====================

                                      x
                                      |
        +-----------------------------+-----------------------------+
        |                             |                             |
        v                             v                             v

  BASELINE CONTROL              DIFFERENTIAL QKV              SCOPE-GATED QKV
  ----------------              ----------------              ----------------

  Q,K,V                         Q+,K+,V+                      Q,K,V,Scope
    |                              |                              |
    v                              v                              +--> Q,K,V
  causal attn                   pos causal attn                 |     |
    |                              |                              |     v
    v                              v                              |   content_out
  content                       pos_out                          |
    |                                                             |
    v                          Q-,K-,V-                          +--> Q,K,Scope
  c_proj                          |                                    |
    |                              v                                    v
    v                          neg causal attn                      scope_out
  output                          |                                    |
                                  v                                    v
                               neg_out                          content * scope
                                  |                                    |
             lambda = softplus(lambda_raw)                            |
                                  |                                    v
                                  v                              gate = sigmoid(Wx)
                         pos_out - lambda*neg_out                     |
                                  |                                    v
                                  v                              gate * content
                                c_proj                                 |
                                  |                                    v
                                  v                         concat 4 streams -> c_proj
                                output                                 |
                                                                       v
                                                                     output
```

My read: **E003 is cleaner than E004 as an architectural probe set.** Differential QKV asks whether an explicit subtractive anti-value branch can remain trainable and interpretable. Scope-gated QKV asks whether a second retrieved stream can act as a scope/modulation channel rather than just another value stream.
