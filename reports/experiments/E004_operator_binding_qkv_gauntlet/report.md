# Scientific Report: E004 Operator/Binding QKV Gauntlet and Cross-Experiment Advisement Across E001–E004

## Executive summary

E004 is a useful screen, but it is not yet a full scientific result. The strongest result is that `operator_valued_attention` survives the full 20/150/500-step gauntlet with nondegenerate mechanism diagnostics, loss essentially matched to the standard control, and acceptable VRAM overhead, albeit at roughly half baseline throughput. It is the clear E004 promotion candidate.

`dynamic_value_query_conditioned_attention` is scientifically interesting because it tracks the standard control closely in loss and speed and shows live gate/delta activity at early rungs, but it fails the 500-step mechanism gate: the run reaches good loss, yet the promotion system marks mechanism activity false and kills it for degenerate/missing diagnostics. That should not be interpreted as “the architecture cannot train.” It should be interpreted as “the current dynamic-value mechanism is not stable enough, or the activity criterion/instrumentation is not yet discriminating enough, to treat the 500-step checkpoint as a reliable object.”

`q3k3v3_role_routed_attention` is active but too slow at the first rung. It should not be promoted as-is. The evidence suggests the architecture has live role streams and pair interactions, but the implementation or design is currently too expensive for the screening pipeline.

Across all four experiments, the best near-term priority is no longer “invent more variants.” The right next step is to build a reproducible investigation substrate: stable promoted checkpoints, hooks/adapters, activation capture, causal patch/ablation tools, and per-architecture diagnostic suites. The highest-priority model families to carry forward are:

1. `operator_valued_attention` from E004.
2. `differential_qkv_anti_value` from E003.
3. `scope_gated_qkv` from E003.
4. `multi_qkv_static_3track_global` and `multi_qkv_position_rotation_3track_global` from E002, but for route-specialization studies rather than general promotion.
5. `cp_trilinear_r8` from E001 only if you can afford its speed/VRAM cost and only after CP-specific diagnostics are added.

## Evidence base and experiment validity

The E004 run was executed on CUDA with bf16 support on an NVIDIA RTX 5060 Ti, and the FineWeb-Edu 100M train shard and 4M validation shard were verified before running. The experiment validator found four runnable canonical configs and no unimplemented configs, covering the standard control, operator-valued attention, Q3K3V3 role-routed attention, and dynamic-value query-conditioned attention.

The E004 gauntlet policy used three rungs: 20 steps, 150 steps, and 500 steps. It required loss descent, checkpoints, mechanism activity, no NaN/Inf, loss ratio no worse than 1.25× control, speed at least 0.15× control, and VRAM no worse than 3.25× control.

That policy is reasonable for an architecture screen, but it is intentionally permissive. A model can pass while being slower or only barely different from the control. Passing means “stable and instrumentable enough to study,” not “scientifically understood.”

## E004 architecture-level findings

### 1. Standard control

The E004 standard control passed all rungs. At rung500, it reached final validation loss 5.6105, median throughput 106,882 tokens/sec, and peak VRAM 3,240.9 MB. The full-run auto-approval was blocked only because the base full-run row was not present in the queue ledger, not because the run failed.

This matters because the E004 control is slightly different from E003: it uses seed2 rather than seed1. The E004 standard rung500 final loss of 5.6105 is close to the E003 standard rung500 final loss of 5.5929, so the gauntlet behavior is broadly stable across these two seeds.

### 2. Operator-valued attention

This is the best E004 result.

The architecture retrieves content using ordinary causal Q/K/V attention, then routes the retrieved message through fixed update operators: add, suppress, gate, transform, and bind. The hypothesis document defines activity as positive combined output norm, at least two active operator norms, nonzero finite router entropy, finite probabilities, positive suppress scale, and no collapse to a single operator.

The implementation matches that description. It computes standard causal content, forms a router input from `[x, content]`, softmaxes over operator modes, and combines add, negative suppress, gate, transform, and bind outputs. The suppress path is explicitly negative-signed via `-suppress_scale * op_suppress(content)`, and bind uses `x * content`.

At rung500, operator-valued attention passed with final validation loss 5.5967 versus the standard control’s 5.6105, a loss ratio of 0.9975. Mechanism activity was true, diagnostics were nondegenerate, and the promotion recommendation was `promote`. The tradeoff is speed: median throughput was 58,571 tokens/sec, about 0.548× control, with VRAM ratio 1.156× control.

The router did not collapse. At rung500, the mean operator probabilities were approximately add 0.169, bind 0.247, gate 0.239, suppress 0.159, transform 0.187, with router entropy mean 1.277. That distribution is not uniform, but it remains distributed across multiple operator modes.

Scientific interpretation: this is a strong survival result. The model is using the operator machinery enough to pass activity gates, and it does so while staying loss-matched to the control. The bind and gate probabilities rise by rung500 relative to rung020/rung150, while suppress and transform decline somewhat. That is not yet an interpretation of function, but it is a concrete developmental signal worth tracking.

Recommendation: promote `operator_valued_attention` to a longer run and instrument it heavily. It is the cleanest E004 candidate because its internal objects are named, typed, causal, and relatively easy to ablate: add, suppress, gate, transform, bind.

### 3. Dynamic-value query-conditioned attention

This architecture keeps standard causal Q/K routing but gates retrieved value content with a receiver-conditioned read-mode gate before the output projection. The hypothesis is that the same source content may be read differently depending on receiver-side context. Activity requires nonzero static/gated/delta norms, finite gate mean/std, nonzero gate std, and no exact saturation.

At rung020 and rung150, dynamic-value attention passed. At rung150, it reached final validation loss 6.3970 versus control 6.4053, loss ratio 0.9987, mechanism activity true, median throughput 87,462 tokens/sec, and VRAM ratio 1.023× control. Its dynamic value delta/static ratio was substantial, around 0.579, and the gate had mean 0.474 and std 0.131.

At rung500, it reached final validation loss 5.5920, better than the control’s 5.6105, with speed ratio 0.824 and VRAM ratio 1.023. However, it was killed because mechanism activity was false and diagnostics were marked degenerate/missing. The decision reason explicitly says the mechanism diagnostics were missing or degenerate and the promotion recommendation was `kill`.

This is a subtle result. The loss curve says the architecture can train. The mechanism gate says the current dynamic-value object is not reliable as an investigated mechanism at the rung500 checkpoint. Those are different claims.

The most likely interpretations are:

1. The gate becomes less meaningfully active or fails the current activity predicate by rung500.
2. The diagnostic predicate is too brittle and reports false death despite nontrivial values.
3. The mechanism remains numerically active but becomes functionally boring, for example a near-static rescaling of values.
4. The architecture learns around the dynamic path, preserving loss while making the intended handle uninformative.

Recommendation: do not promote this exact run automatically. Run a targeted diagnostic rescue before discarding the family. Specifically, inspect per-layer gate mean/std over time, distribution histograms rather than only max/mean summaries, gate saturation by layer, delta-to-static ratio by layer, and causal ablations of the dynamic delta. If the causal ablation is small despite nonzero deltas, kill or redesign. If the ablation is large and the current gate failed for a brittle threshold reason, revise the mechanism check.

### 4. Q3K3V3 role-routed attention

The hypothesis is conceptually attractive: create three typed Q/K/V role streams—content-like, operator-like, and binding-like—and project content, operator, binding, and pair-product streams back to the residual dimension.

The implementation is much heavier than standard attention. It projects to 9× embedding dimension for role-specific Q/K/V, performs separate attention for content, operator, and binding roles, optionally includes pair products, and projects a concatenated role representation back down.
At rung020, Q3K3V3 showed mechanism activity and nondegenerate diagnostics, but the screen failed with failure class `SLOW`. It reached final validation loss 9.7348, loss ratio 1.0005, and VRAM ratio 1.133, but median throughput was only 25,566 tokens/sec, a speed ratio of 0.239× control. The decision was `needs_investigation`, not promotion.

The role streams are alive: content, operator, and binding output norms are all large, and all pair interaction norms are nonzero. But the implementation is too expensive to enter the standard gauntlet path as-is.

Recommendation: do not kill the idea, but do not run it forward unchanged. First profile it. The likely bottleneck is three SDPA calls plus a wide role projection and pair-product projection. The next version should reduce width and test whether the same role-separation signal survives with lower cost: shared K, shared V, low-rank role projections, fewer pair products, or a single fused role attention formulation. Only re-enter the gauntlet after throughput is at least in the same band as scope-gated QKV, roughly ≥0.70× control, unless there is a deliberately separate “expensive specimen” track.

## Cross-experiment synthesis

### E001: CP-bilinear and CP-trilinear attention

E001 established that explicit low-rank CP interaction structure can be inserted into attention without immediate collapse. Standard, CP-bilinear, and CP-trilinear completed 3000 steps; the lambda-fixed CP-trilinear configuration was present but not evaluated.

The final metrics show:

| E001 run        | Final val loss |   PPL | Median tok/s | Peak VRAM |
| --------------- | -------------: | ----: | -----------: | --------: |
| standard        |         4.0768 | 58.96 |      109,428 |   3.24 GB |
| CP-bilinear r8  |         4.0863 | 59.52 |       33,626 |   4.44 GB |
| CP-trilinear r8 |         4.0623 | 58.11 |       16,761 |   6.77 GB |

The scientific value of E001 is that CP paths expose rank-indexed handles. The cost is high, especially for CP-trilinear. The E001 report itself says CP-structured attention is stable enough to justify mechanism-specific diagnostics, but does not establish interpretability, causal importance, larger-scale persistence, or whether lambda becomes meaningful.

Advisement: keep E001 as a mechanism-family, but do not prioritize it ahead of E003/E004 unless you add CP-specific diagnostics. CP-trilinear is interesting but extremely expensive. CP-bilinear is cheaper but less rich. The missing lambda-fixed/null contribution control is important before making strong claims.

### E002: Multi-track QKV shift-register family

E002 is the strongest evidence so far that QKV stream geometry can be manipulated while preserving trainability. All completed Multi-QKV variants finished 3000 steps. Static-global, train-rotation, and position-rotation all trained, but with different losses and throughput: static-global was closest to standard, train-rotation was worst in loss, and position-rotation was intermediate.

The most important result is not the loss table. It is the destructive perturbation signature. Static-global is highly route-identity-sensitive, train-rotation is almost route-identity-interchangeable while the selected pathway remains live, and position-rotation is intermediate.

Advisement: E002 should become the route-specialization workbench. It is less immediately attractive as a single promoted architecture than E003/E004, but it is excellent for studying how architectural routing rules induce or suppress specialization. The recommended next analyses from the existing report—Q-only/K-only/V-only perturbations, per-layer track swaps, per-position-class perturbations, and route replacement matrices—are exactly the right direction.

### E003: Differential QKV and scope-gated QKV

E003 is the cleanest successful gauntlet before E004. Both `differential_qkv_anti_value` and `scope_gated_qkv` passed 20/150/500-step rungs, maintained validation loss close to the control, and produced nondegenerate mechanism diagnostics.

At rung500:

| E003 variant                | Final val loss | Loss ratio | Median tok/s | Speed ratio | Peak VRAM | Mechanism active |
| --------------------------- | -------------: | ---------: | -----------: | ----------: | --------: | ---------------- |
| standard                    |        5.59294 |    1.00000 |      104,110 |       1.000 | 3240.9 MB | N/A              |
| differential QKV anti-value |        5.59823 |    1.00095 |       86,748 |       0.833 | 3379.7 MB | true             |
| scope-gated QKV             |        5.59785 |    1.00088 |       78,750 |       0.756 | 3496.1 MB | true             |

The report correctly prioritizes `differential_qkv_anti_value` as the strongest next E003 candidate because it is simpler, cheaper, close to control in loss, and directly falsifiable as a subtractive branch.

Advisement: differential QKV and operator-valued attention should now be compared directly. They are conceptually adjacent: both provide an explicit nonstandard write/counterwrite surface, but differential QKV separates branches before attention, while operator-valued attention routes after retrieval. That distinction is scientifically valuable.

## Overall ranking for next work

### Tier 1: promote now

**1. Operator-valued attention**

Rationale: passed all E004 rungs, active nondegenerate operator diagnostics, loss slightly better than control at rung500, modest VRAM overhead, and clear named internal operators. The speed hit is real but acceptable for a high-value specimen.

Next: run 3000-step full training, then run operator ablation and routing analyses.

**2. Differential QKV anti-value**

Rationale: passed all E003 rungs, cheapest and simplest of the successful side-stream variants, and directly testable. It is the strongest E003 continuation candidate.

Next: run full training if not already done, then measure branch causal effect, branch specialization, and lambda dynamics.

### Tier 2: promote with targeted question

**3. Scope-gated QKV**

Rationale: passed E003, but is slower and more complex than differential QKV. It remains valuable because its gate/scope/content interaction is different enough to test whether multiplicative side streams produce separable mechanisms.

Next: full run only if paired with gate/scope ablation analyses.

**4. Static-global and position-rotation Multi-QKV**

Rationale: E002 gives real causal route-perturbation evidence. These are not merely active; they create distinguishable route-specialization regimes.

Next: do route/track causal matrices, not generic full promotion.

### Tier 3: hold, repair, or redesign

**5. Dynamic-value query-conditioned attention**

Rationale: good loss and speed, but failed the rung500 mechanism gate. It might be valuable, but it needs diagnostic rescue before promotion.

Next: inspect whether the death is real, brittle, or just a poor activity predicate.

**6. Q3K3V3 role-routed attention**

Rationale: conceptually rich and active, but too slow immediately. Needs profiling/redesign.

Next: reduce compute or create a separate expensive-specimen policy.

**7. CP-trilinear / CP-bilinear**

Rationale: E001 shows survival, but the cost is high and the mechanism evidence is underdeveloped. CP-trilinear is richer; CP-bilinear is cheaper. Neither should outrank the cleaner E003/E004 candidates until CP diagnostics are implemented.

Next: lambda/null controls and rank-component causal tests.

## What to do next: concrete research process

The next phase should be a transition from “architecture survival screening” to “mechanism-readout infrastructure.”

### Step 1: freeze the promoted checkpoint set

Create a promoted-candidate manifest with:

| Candidate                                                     | Source experiment | Status                         |
| ------------------------------------------------------------- | ----------------- | ------------------------------ |
| `operator_valued_attention_30m_seed2_rung500`                 | E004              | promote to full                |
| `differential_qkv_anti_value_30m_seed1_rung500`               | E003              | promote to full                |
| `scope_gated_qkv_30m_seed1_rung500`                           | E003              | promote with targeted analysis |
| `multi_qkv_static_3track_global_30m_seed1`                    | E002              | route-specialization workbench |
| `multi_qkv_position_rotation_3track_global_30m_seed1`         | E002              | position/route workbench       |
| `dynamic_value_query_conditioned_attention_30m_seed2_rung500` | E004              | diagnostic rescue only         |
| `q3k3v3_role_routed_attention_30m_seed2_rung020`              | E004              | performance investigation only |

Do not add more new architectures until this manifest has at least one full causal analysis path.

### Step 2: build hooks/adapters rather than waiting for TransformerLens compatibility

The current codebase already has custom attention modules and a custom diagnostics collection path: `collect_attention_diagnostics` walks model blocks, calls `attention.attention_diagnostics(step, layer)`, and appends rows when modules expose that method.

That is useful, but it is not enough for serious investigation. The missing abstraction is a hook/activation interface that exposes common and architecture-specific sites.

For standard-like models, define canonical hook points:

```text
resid_pre[layer]
attn_q[layer]
attn_k[layer]
attn_v[layer]
attn_pattern[layer]
attn_out[layer]
resid_mid[layer]
mlp_out[layer]
resid_post[layer]
logits
```

For novel architectures, add architecture-specific sites:

```text
operator_valued:
    operator_probs[layer]
    operator_add_out[layer]
    operator_suppress_out[layer]
    operator_gate_out[layer]
    operator_transform_out[layer]
    operator_bind_out[layer]
    operator_combined_out[layer]

differential_qkv:
    pos_q/k/v[layer]
    neg_q/k/v[layer]
    pos_out[layer]
    neg_out[layer]
    lambda[layer]
    branch_delta[layer]

scope_gated:
    content_out[layer]
    scope_out[layer]
    gate[layer]
    content_scope_product[layer]
    gated_content[layer]

multi_qkv:
    track_q/k/v[layer, track]
    selected_track[layer, token]
    track_out[layer]
    route_id[layer, token]

cp:
    cp_score[layer, rank]
    cp_output[layer, rank]
    lambda[layer]
    cp_total[layer]
```

Do not force everything into TransformerLens first. Build an internal `ActivationCache`-like object with these sites, then later write a thin compatibility adapter for TL-style analysis where shapes match.

### Step 3: implement causal tests before semantic claims

For every promoted candidate, require three tests:

1. **Component ablation:** zero the candidate mechanism and measure loss/logit changes.
2. **Component replacement:** replace mechanism activations with control or shuffled activations.
3. **Counterfactual patching:** patch mechanism activations between paired prompts and measure targeted logit effects.

Minimum viable analyses:

```text
operator_valued:
    ablate one operator at a time
    force router to one operator
    swap router probabilities across tokens/layers
    patch suppress/bind/gate outputs between paired contexts

differential_qkv:
    zero negative branch
    zero positive branch
    vary lambda
    swap negative branch between contexts
    measure whether branch_delta predicts logit shifts

scope_gated:
    set gate to mean
    zero scope stream
    zero content*scope product
    patch gate/scope separately

multi_qkv:
    Q-only/K-only/V-only track swaps
    per-layer route replacement matrix
    per-position-class perturbations

cp:
    rank-wise ablation
    lambda sweep
    CP path zeroing
    rank activation clustering
```

### Step 4: train full versions only after instrumentation is ready

A 3000-step full run is useful, but without hooks, it only gives better loss curves. The next full run should produce:

```text
checkpoints:
    step 500
    step 1000
    step 2000
    step 3000

logs:
    ordinary train/val metrics
    architecture-specific diagnostics
    activation-stat snapshots
    causal-ablation evals
    generation samples
    small downstream evals if already available
```

The process should not be “run longer, then wonder what happened.” It should be “run longer while collecting the evidence needed to decide whether the mechanism becomes clearer, collapses, or becomes irrelevant.”

### Step 5: compare matched mechanisms across architectures

The most important cross-experiment comparisons are:

```text
counterwrite-like behavior:
    differential_qkv negative branch
    operator_valued suppress operator
    scope_gated scope/content interaction

binding-like behavior:
    operator_valued bind operator
    q3k3v3 binding stream if rescued
    cp_trilinear rank components

routing specialization:
    E002 static/global tracks
    E002 position-rotation tracks
    operator router modes
```

This is where the project becomes scientifically coherent. Each architecture is not just a standalone variant; it is a probe into a family of possible internal decompositions.

## Interpretation boundaries

The evidence supports these claims:

```text
1. Several nonstandard attention/QKV architectures train without collapse at 30M scale.
2. E003 differential QKV and scope-gated QKV pass early gauntlet screens.
3. E004 operator-valued attention passes the gauntlet and is the strongest new candidate.
4. E002 routing variants produce distinguishable causal perturbation signatures.
5. E001 CP variants can complete full 3000-step training but are expensive.
```

The evidence does not yet support these claims:

```text
1. Any variant is better as a language model.
2. Any internal branch has a known semantic role.
3. Any result replicates across seeds.
4. Any mechanism persists at larger model sizes.
5. Any architecture reveals clean superposition structure yet.
6. Any architecture is ready for strong claims without causal intervention data.
```

## Final recommendation

Promote `operator_valued_attention` and `differential_qkv_anti_value` as the two primary specimens for the next phase. Run them through matched full training with a shared hook/activation-cache interface and causal tests. Keep `scope_gated_qkv` as the third candidate if compute allows. Use E002 as a route-specialization workbench. Put `dynamic_value_query_conditioned_attention` into diagnostic rescue, not full promotion. Put Q3K3V3 into profiling/redesign. Do not create E005 until the project can answer causal questions about at least one promoted checkpoint.

The immediate next implementation task should be:

```text
Build the architecture-aware activation/hook and causal-intervention layer, then run matched causal readouts on:
1. operator_valued_attention_30m_seed2_rung500
2. differential_qkv_anti_value_30m_seed1_rung500
3. scope_gated_qkv_30m_seed1_rung500
```

That process solves the current bottleneck: the models are not TransformerLens-compatible by default, but they do not need to be. They need stable named hook points, activation capture, component ablation/replacement, paired-prompt patching, and exportable artifacts. TransformerLens compatibility can come afterward as an adapter, not as a prerequisite.
