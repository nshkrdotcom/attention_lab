Yes. Forget “better model.” These are **research-object architectures**: ways of changing the Q/K/V stream geometry so you can later ask what the extra streams learn, whether they become suppressive, whether scope/modifier structure separates, and whether static KV is hiding limitations.

Below are the three variants actually tested: the standard control, `differential_qkv_anti_value`, and `scope_gated_qkv`.

---

# 0. Standard attention control

This is the baseline architecture being compared against. It is the ordinary GPT-style causal self-attention block.

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
        |
        +----------+----------+
        |          |          |
        v          v          v
        Q          K          V
        |          |          |
        +----------+----------+
                   |
                   v
        causal scaled dot-product attention
        attention(Q, K, V)
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

In code, the standard module uses one projection from `d_model` to `3 * d_model`, splits it into Q/K/V, runs causal SDPA, reshapes back to `[batch, seq, d_model]`, and applies the output projection. 

## Conceptual meaning

Each token produces:

```text
Q = what am I looking for?
K = what do I advertise to others?
V = what content do I offer if attended to?
```

Then each receiver token uses its query to read from previous tokens’ keys and values.

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

## Why this is the control

The control asks:

```text
Given ordinary Q/K/V attention,
what does early training loss, speed, VRAM, and stability look like
under the same data, same model size class, same optimizer, same schedule?
```

This is the thing the variants must remain comparable to so they can become interpretable research specimens.

---

# 1. Differential QKV anti-value

This is the cleaner of the two novel variants.

The core idea is:

```text
Instead of one Q/K/V branch, create two attention branches:

1. a positive/content branch
2. a negative/suppressive branch

Then subtract the second branch from the first.
```

The implementation projects to either `5 * d_model` if sharing values or `6 * d_model` if using separate values. In the tested configs, `diff_qkv_share_value: false`, so it used six streams: `q_pos, k_pos, v_pos, q_neg, k_neg, v_neg`.  

## High-level diagram

```text
Input residual stream
x: [batch, seq, d_model]
        |
        v
Linear projection: c_attn
        |
        v
+-------------------------------------------------------+
| [Q_pos | K_pos | V_pos | Q_neg | K_neg | V_neg]       |
+-------------------------------------------------------+
       |       |       |       |       |       |
       v       v       v       v       v       v

   positive branch                negative / anti-value branch
   ---------------                ----------------------------
   Q_pos K_pos V_pos              Q_neg K_neg V_neg
       \  |  /                         \  |  /
        \ | /                           \ | /
         v                               v
 attention(Q_pos,K_pos,V_pos)     attention(Q_neg,K_neg,V_neg)
         |                               |
         v                               v
      Y_pos                           Y_neg
         |                               |
         +---------------+---------------+
                         |
                         v
              Y_branch = Y_pos - lambda * Y_neg
                         |
                         v
                  reshape/merge heads
                         |
                         v
                  output projection c_proj
                         |
                         v
                  residual write
```

## Formula

Standard attention is:

```text
Y = Attention(Q, K, V)
```

Differential anti-value attention is:

```text
Y_pos = Attention(Q_pos, K_pos, V_pos)

Y_neg = Attention(Q_neg, K_neg, V_neg)

lambda = softplus(lambda_raw)

Y = Y_pos - lambda * Y_neg
```

The actual implementation does exactly that: it runs one causal SDPA call for the positive branch, another causal SDPA call for the negative branch, computes `lambda_value = softplus(lambda_raw)`, and forms `y_branch = y_pos - lambda_value * y_neg`. 

## What each stream means

The intended interpretation is not “better language model.” The intended research question is:

```text
Can a model use one attention branch to write content
and another branch to subtract, suppress, cancel, erase, or counterwrite?
```

More concretely:

```text
Q_pos  = what content should I positively retrieve?
K_pos  = what tokens advertise positive relevance?
V_pos  = what positive content gets written?

Q_neg  = what suppressive/anti-content should I retrieve?
K_neg  = what tokens advertise suppressive relevance?
V_neg  = what negative/anti-value content gets subtracted?

lambda = global learned strength of the negative branch
```

## Token-level view

For a receiver token `i`:

```text
                    previous tokens j <= i
                          |
                          v
receiver Q_pos_i  ---> compare with K_pos_j ---> weights_pos
                          |
                          v
                 weighted sum of V_pos_j
                          |
                          v
                        Y_pos_i


receiver Q_neg_i  ---> compare with K_neg_j ---> weights_neg
                          |
                          v
                 weighted sum of V_neg_j
                          |
                          v
                        Y_neg_i


final write at token i:

Y_i = Y_pos_i - lambda * Y_neg_i
```

So this architecture creates a built-in distinction between:

```text
what to add/write
```

and:

```text
what to subtract/suppress
```

That is why it is interesting for studying negation, exception handling, cancellation, modifier scope, or “anti-evidence.”

## Why it is a research object

Standard attention can already represent subtractive effects indirectly through value vectors and later MLP/residual interactions. But it does not expose a clean architectural separation between a positive read branch and a suppressive read branch.

This variant asks:

```text
If we give the model an explicit subtractive branch,
does that branch become mechanistically meaningful?
```

For example, later probes could ask:

```text
Does the negative branch activate more around:
- not
- never
- except
- unless
- without
- but not
- failed to
- no longer
- impossible
- contradiction markers
- instruction reversals
```

And causally:

```text
If we zero Y_neg, does the model become worse at negation/suppression tasks?

If we flip the sign, does behavior invert?

If we freeze lambda near zero, does the branch die?

If we force lambda larger, does the model over-suppress?

If we ablate specific layers' negative branch, where does the effect live?
```

## Diagnostics actually logged

The module records:

```text
diff_lambda
pos_output_norm
neg_output_norm
neg_to_pos_output_norm_ratio
branch_output_delta
```

Those are not semantic proofs. They only prove the mechanism is alive and nondegenerate. The implementation logs these quantities after computing the positive and negative branch outputs. 

## What E003 showed for this variant

At 500 steps, it passed the gauntlet with:

```text
final val loss:       5.59823
control val loss:     5.59294
loss ratio:           1.00095
median tokens/sec:    86,748
speed ratio:          0.833
VRAM ratio:           1.043
mechanism active:     true
```

The 500-step diagnostic summary showed positive branch/negative branch activity and finite lambda. 

Correct interpretation:

```text
This architecture did not collapse.
The subtractive branch was active.
It remained close enough to standard learning dynamics to be worth probing.
```

Not:

```text
It improved the model.
```

---

# 2. Scope-gated QKV

This variant is more elaborate.

The core idea is:

```text
Keep ordinary content attention,
add a separate "scope" value stream,
then combine content, scope, their interaction, and a gated content write.
```

The implementation projects to `4 * d_model`, split as:

```text
Q | K | V | scope
```

Then it separately computes:

```text
content = Attention(Q, K, V)
scoped  = Attention(Q, K, scope)
```

It also computes a receiver-side gate from the input residual stream.  

## High-level diagram

```text
Input residual stream
x: [batch, seq, d_model]
        |
        +-----------------------------+
        |                             |
        v                             v
Linear projection: c_attn        Linear projection: c_gate
        |                             |
        v                             v
+-----------------------+          sigmoid
| [Q | K | V | Scope]   |             |
+-----------------------+             v
    |   |   |    |                   Gate
    |   |   |    |             [batch, seq, d_model]
    |   |   |    |
    |   |   |    +-------------------------+
    |   |   |                              |
    v   v   v                              v
    Q   K   V                            Scope
    |   |   |                              |
    +---+---+                              |
        |                                  |
        v                                  v
Attention(Q,K,V)                  Attention(Q,K,Scope)
        |                                  |
        v                                  v
    Content                            Scoped
        |                                  |
        |                                  v
        |                         scope_scale * Scoped
        |                                  |
        +--------------+-------------------+
                       |
                       v
        +--------------------------------------------+
        | concatenate four streams:                  |
        |                                            |
        | 1. Content                                 |
        | 2. Scoped                                  |
        | 3. Content * Scoped                        |
        | 4. Gate * Content                          |
        +--------------------------------------------+
                       |
                       v
              output projection c_proj
                       |
                       v
                 residual write
```

## Formula

```text
Q, K, V, S = split(c_attn(x))

Content = Attention(Q, K, V)

Scoped = Attention(Q, K, S)

Scoped = scope_scale * Scoped

Gate = sigmoid(c_gate(x))

Combined = concat(
    Content,
    Scoped,
    Content * Scoped,
    Gate * Content
)

Y = c_proj(Combined)
```

The implementation follows exactly this structure: it computes `content`, computes `scoped`, applies `scope_scale`, computes `gate = sigmoid(c_gate(x))`, concatenates `content_flat`, `scoped_flat`, `content_flat * scoped_flat`, and `gate * content_flat`, then projects back down through `c_proj`. 

## What “scope” means here

Important: the architecture does **not** magically know linguistic scope.

The name “scope” means:

```text
an extra value-like stream that is read using the same Q/K attention pattern,
but is kept separate from ordinary content before being combined downstream.
```

So standard attention has:

```text
Q, K, V
```

Scope-gated attention has:

```text
Q, K, V_content, V_scope
```

The same attention weights are used for both content and scope reads, because both use the same `Q` and `K`.

```text
weights = softmax(QK^T causal mask)

Content_i = sum_j weights_ij * V_j

Scoped_i  = sum_j weights_ij * Scope_j
```

The research bet is:

```text
Maybe the model can put modifier/operator/contextual information into Scope
while keeping ordinary semantic content in V.
```

Examples of possible future specialization:

```text
content stream:
    dog, friendly, safe, harmful, allowed, present

scope stream:
    not, never, except, unless, before, after, quoted, hypothetical,
    conditional, negated, instruction boundary, modifier extent
```

But E003 did not prove this. It only showed the stream is active and trainable.

## Why there is a gate

The gate is receiver-side, because it is computed from the current token’s residual stream:

```text
Gate_i = sigmoid(W_gate x_i + b_gate)
```

Then it modulates the content write:

```text
Gate_i * Content_i
```

So token `i` can learn something like:

```text
How much should I allow ordinary content attention to write here?
```

The tested config initialized `scope_gate_bias_init: 0.0`, so the initial sigmoid gate is centered near 0.5 before training. 

## Token-level view

For receiver token `i`:

```text
                  previous tokens j <= i
                        |
                        v
Q_i compares with K_j -> attention weights
                        |
            +-----------+------------+
            |                        |
            v                        v
weighted V_content_j       weighted V_scope_j
            |                        |
            v                        v
       Content_i                Scoped_i
            |                        |
            +-----------+------------+
                        |
                        v

Gate_i = sigmoid(W_gate x_i)

Combined_i =
[
  Content_i,
  Scoped_i,
  Content_i * Scoped_i,
  Gate_i * Content_i
]

Y_i = output_projection(Combined_i)
```

## What the four concatenated pieces mean

The output projection receives:

```text
1. Content_i
```

Ordinary attended content.

```text
2. Scoped_i
```

Separate attended sidecar stream.

```text
3. Content_i * Scoped_i
```

Multiplicative interaction between content and scope. This is important because it gives the model a direct feature-crossing mechanism:

```text
content says: "friendly"
scope says:   "negated"
interaction:  "negated-friendly"
```

Again, not proven, but architecturally available.

```text
4. Gate_i * Content_i
```

Receiver-controlled content write. This lets the receiving token decide how much attended content should pass through.

## Why this is a research object

Standard attention gives you one value stream. If modifier/scope/operator information is entangled in the same V stream as ordinary content, later interpretability has to separate it after the fact.

Scope-gated QKV asks:

```text
If we give the model a separate sidecar stream and explicit interaction term,
does it learn to separate operator-like information from content-like information?
```

Possible probes:

```text
Does Scope activate more on:
- negation tokens
- exception tokens
- conditionals
- quotation boundaries
- instruction delimiters
- temporal modifiers
- quantifiers
- modal verbs
- contrastive conjunctions
```

Causal probes:

```text
Zero Scoped only.
Zero Content only.
Zero Content * Scoped interaction only.
Freeze Gate.
Randomize Gate.
Clamp Gate to 0.5.
Ablate scope stream by layer.
Swap scope stream between minimal-pair examples.
```

## Diagnostics actually logged

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

These are activity diagnostics, not semantic diagnostics. They prove the scope path, gate path, and interaction path are not dead. 

## What E003 showed for this variant

At 500 steps, it passed with:

```text
final val loss:       5.59785
control val loss:     5.59294
loss ratio:           1.00088
median tokens/sec:    78,750
speed ratio:          0.756
VRAM ratio:           1.079
mechanism active:     true
```

Its 500-step diagnostics showed nonzero content, scope, interaction, and gate activity. 

Correct interpretation:

```text
The explicit scope/gate machinery can survive early training.
It is active.
It remains close enough to standard loss to study.
```

Not:

```text
It learned linguistic scope.
```

---

# 3. Side-by-side architectural comparison

```text
STANDARD
========

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
write


DIFFERENTIAL QKV ANTI-VALUE
===========================

x
|
v
[Q+ K+ V+ Q- K- V-]
 |  |  |  |  |  |
 +--+--+  +--+--+
    |        |
    v        v
 Attn+    Attn-
    |        |
    v        v
  Y_pos    Y_neg
    |        |
    +--- -lambda* ---+
            |
            v
 Y = Y_pos - lambda Y_neg
            |
            v
          c_proj
            |
            v
          write


SCOPE-GATED QKV
===============

x
|
+-----------------------------+
|                             |
v                             v
[Q K V Scope]                 Gate = sigmoid(W_gate x)
 | | |   |
 | | |   +-------------------+
 | | |                       |
 +-+-+                       |
   |                         |
   v                         v
Attention(Q,K,V)      Attention(Q,K,Scope)
   |                         |
   v                         v
Content                  Scoped
   |                         |
   +-----------+-------------+
               |
               v
concat[
  Content,
  Scoped,
  Content * Scoped,
  Gate * Content
]
               |
               v
             c_proj
               |
               v
             write
```

---

# 4. The deepest distinction

## Standard attention

```text
one read/write channel:

attend to previous tokens -> write value mixture
```

It can represent suppression or scope, but only implicitly.

## Differential anti-value

```text
two read/write channels:

positive channel   -> write
negative channel   -> subtract
```

This is interesting because it makes suppressive/counterfactual/canceling behavior architecturally explicit.

## Scope-gated QKV

```text
one attention pattern, two value-like streams:

content stream -> what is being said
scope stream   -> contextual/operator sidecar

then explicit interaction and receiver-side gating
```

This is interesting because it gives the model a place where modifier/scope/operator information could become more separable from content.

---

# 5. What each architecture is good for studying

## Differential anti-value is good for questions like:

```text
Does the model learn a subtractive stream?

Does the negative branch specialize by layer?

Does it carry "anti-evidence"?

Does it become active around negation, exceptions, contradiction, refusal, reversal, or suppression?

Can causal ablation of Y_neg selectively damage tasks involving negation/suppression?

Does lambda grow, shrink, stabilize, or specialize?
```

## Scope-gated QKV is good for questions like:

```text
Does a sidecar scope stream become separable from content?

Does the multiplicative Content * Scope term encode modifier interactions?

Does the receiver-side gate learn to regulate writes around boundary/modifier/operator tokens?

Can scope stream swapping between minimal pairs change interpretation?

Does the scope stream become more interpretable than standard V?
```

---

# 6. Why these passed E003 in the research sense

The correct “pass” criterion is not:

```text
lower loss than standard
```

The correct criterion is:

```text
still trains
does not NaN
loss descends
checkpoint exists
mechanism path is active
diagnostics are nondegenerate
loss is close enough to standard that the object is not broken
```

The gauntlet logic enforces exactly that: checkpoint, descending loss, active/nondegenerate mechanism diagnostics for nonstandard variants, no bad loss/speed/VRAM ratio beyond configured thresholds. 

So the architectural takeaway is:

```text
STANDARD:
    control object

DIFFERENTIAL:
    stable explicit suppressive/anti-value branch object

SCOPE-GATED:
    stable explicit content/scope/gate interaction object
```

That is the result.
