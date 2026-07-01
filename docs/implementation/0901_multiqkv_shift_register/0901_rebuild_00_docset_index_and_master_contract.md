# 0901 Rebuild 00: Docset Index and Master Contract

## Purpose

This docset turns the Multi-QKV Shift Register idea into a concrete Attention Lab implementation plan. The first build is limited to:

```text
bundled-triple
globally-shared
deterministic-modular
full-dense-matrix
hard-switch
```

The implementation must support only the initial A/B/C experiment family:

```text
A: depth-only static cycle
B: train-time step rotation, frozen depth-only deployment
C: position rotation, active at train and inference
```

It must not drift into learned routing, softmix, LoRA-delta rotation, stochastic routing, coprime decoupled Q/K/V clocks, or typed content/control/binding streams.

## Source Idea Contract

Structural change:

```text
- Keep a pool of 3 Q matrices, 3 K matrices, and 3 V matrices.
- The 9 matrices are the same shape as ordinary Q/K/V projections.
- At a given layer/token/step, exactly one bundled Q/K/V triple is active.
- Inactive triples do not participate in that forward path.
```

Scheduling rule:

```text
- The active triple is chosen by a fixed deterministic formula.
- The formula is not learned.
- The formula is not content-dependent.
- The formula may depend on depth, training step, and/or sequence position.
```

The core hypothesis is not "more parameters." It is that deterministic rotation pressures the shared banked Q/K/V matrices toward depth- or phase-portable subroutines.

## Docset Index

The docset contains exactly these files:

```text
0901_rebuild_00_docset_index_and_master_contract.md
0901_rebuild_01_experiment_math_and_variant_contract.md
0901_rebuild_02_model_and_registry_implementation.md
0901_rebuild_03_tdd_and_qc_plan.md
0901_rebuild_04_diagnostics_and_mechanism_checks.md
0901_rebuild_05_configs_scripts_and_manual_run_loop.md
0901_rebuild_06_agent_execution_prompt_and_final_acceptance.md
```

## Canonical First-Build Names

```text
standard_refactor_control_30m_seed1
multi_qkv_static_3track_global_30m_seed1
multi_qkv_train_rotation_3track_global_30m_seed1
multi_qkv_position_rotation_3track_global_30m_seed1
```

`standard_30m_seed1.yaml` remains runnable only as a legacy/noncanonical comparison config. It is not one of the four
canonical first-build configs. The old E002 skeleton names remain compatibility/planning artifacts only and stay
`status: experimental_unimplemented`.

## Manual Run Boundary

The implementation agent may run unit tests, integration tests, config validation, model inspection, and tiny/sanity training only when explicitly marked as tests. The implementation agent must not run full 3000-step E002 experiments, approve queue full runs, fake result artifacts, or handwrite metrics.

## Acceptance Gates

Architecture acceptance requires one globally shared 3-track Q/K/V bank, all blocks using the same bank object, bundled Q/K/V track selection, A/B/C formulas exactly as specified, B frozen to static depth routing during eval/generation, and no learned or stochastic routing.

Code-quality acceptance requires standard attention invariance under step/position threading, CP attention regression coverage, strict config validation for new fields, direct `GPT` construction rejecting invalid canonical Multi-QKV settings, registered attention types, training step passed to model, and eval/generation schedule context handled explicitly.

Testing acceptance requires formula tests, global sharing tests, one-track identity with standard attention, causal masking tests, inactive-track gradient tests, train-loop step threading tests, eval/generation schedule tests, and diagnostics tests.

Documentation/script acceptance requires updated E002 plan, hypothesis docs, report README, manual run loop, destructive test script, full-run scripts, and initial comparison script. No report may claim full-run evidence until manual operator runs produce verified artifacts.

## Required QC

```bash
uv run ruff check .
uv run pytest
uv run pytest --run-integration
uv run scripts/validate_experiment.py --id E002_multitrack_qkv_shift_register
uv run scripts/inspect_model_config.py --config configs/experiments/E002_multitrack_qkv_shift_register/standard_refactor_control_30m_seed1.yaml
uv run scripts/inspect_model_config.py --config configs/experiments/E002_multitrack_qkv_shift_register/multi_qkv_static_3track_global_30m_seed1.yaml
uv run scripts/inspect_model_config.py --config configs/experiments/E002_multitrack_qkv_shift_register/multi_qkv_train_rotation_3track_global_30m_seed1.yaml
uv run scripts/inspect_model_config.py --config configs/experiments/E002_multitrack_qkv_shift_register/multi_qkv_position_rotation_3track_global_30m_seed1.yaml
```

## Commit Rule

Commit and push only after QC is green:

```bash
git add .
git commit -m "Implement initial E002 global multi-QKV rotation experiments"
git push
```
