# E002 Implementation Notes

## Scope

Implemented first-build global Multi-QKV variants:

```text
multi_qkv_static_3track_global
multi_qkv_train_rotation_3track_global
multi_qkv_position_rotation_3track_global
```

Not implemented in this pass:

```text
softmix
warmup routing
LoRA deltas
learned routing
stochastic routing
coprime decoupled Q/K/V clocks
typed content/control/binding streams
```

## Parameter Counts

From `scripts/inspect_model_config.py`:

| run | attention type | excluding positional | including positional | trainable | attention projection | non-attention | delta vs standard |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| standard_refactor_control_30m_seed1 | standard | 29938560 | 30331776 | 30331776 | 3538944 | 26792832 | 0 |
| multi_qkv_static_3track_global_30m_seed1 | multi_qkv_static_3track_global | 28611456 | 29004672 | 29004672 | 2211840 | 26792832 | -1327104 |
| multi_qkv_train_rotation_3track_global_30m_seed1 | multi_qkv_train_rotation_3track_global | 28611456 | 29004672 | 29004672 | 2211840 | 26792832 | -1327104 |
| multi_qkv_position_rotation_3track_global_30m_seed1 | multi_qkv_position_rotation_3track_global | 28611456 | 29004672 | 29004672 | 2211840 | 26792832 | -1327104 |

The negative delta is expected because Q/K/V banks are globally shared across layers. B and C should be interpreted primarily against A/static-global.

`inspect_model_config.py` reports trainable parameters, attention projection
parameters, and non-attention parameters for the canonical configs.

## Full-Run Status

Full 3000-step E002 runs were not executed during implementation. Manual execution is required.

## Config Classification

E002 currently has four canonical first-build configs and one legacy/noncanonical runnable comparison config:

```text
canonical_first_build_config_count = 4
legacy_or_auxiliary_runnable_config_count = 1
runnable_config_count = 5
unimplemented_config_count = 6
legacy_or_auxiliary_runnable_configs = ["standard_30m_seed1.yaml"]
```

The six old skeleton configs remain `status: experimental_unimplemented`.

## Closure QC

Implementation/code readiness: passed the commands below on 2026-06-30.

Scientific result status: no full 3000-step E002 runs were executed or claimed. No checkpoint, validation, HellaSwag,
run-summary, destructive-test, or comparison metrics were fabricated.

```text
uv sync
PASSED: Resolved 106 packages; audited 100 packages.

uv run ruff check .
PASSED: All checks passed.

uv run pytest
PASSED: 235 passed, 1 skipped.

uv run pytest --run-integration
PASSED: 236 passed.

uv run scripts/validate_experiment.py --id E002_multitrack_qkv_shift_register
PASSED: ok=True, config_count=11, runnable_config_count=5, unimplemented_config_count=6,
canonical_first_build_config_count=4, legacy_or_auxiliary_runnable_config_count=1.

uv run scripts/inspect_model_config.py --config configs/experiments/E002_multitrack_qkv_shift_register/standard_refactor_control_30m_seed1.yaml
PASSED: attention_type=standard, parameters_including_positional=30331776, global_qkv_bank_parameters=0.

uv run scripts/inspect_model_config.py --config configs/experiments/E002_multitrack_qkv_shift_register/multi_qkv_static_3track_global_30m_seed1.yaml
PASSED: attention_type=multi_qkv_static_3track_global, qkv_track_count=3, qkv_global_bank=True,
qkv_route_formula=layer_mod, global_qkv_bank_parameters=1327104.

uv run scripts/inspect_model_config.py --config configs/experiments/E002_multitrack_qkv_shift_register/multi_qkv_train_rotation_3track_global_30m_seed1.yaml
PASSED: attention_type=multi_qkv_train_rotation_3track_global, qkv_track_count=3, qkv_global_bank=True,
qkv_route_formula=layer_plus_step_train_layer_eval, global_qkv_bank_parameters=1327104.

uv run scripts/inspect_model_config.py --config configs/experiments/E002_multitrack_qkv_shift_register/multi_qkv_position_rotation_3track_global_30m_seed1.yaml
PASSED: attention_type=multi_qkv_position_rotation_3track_global, qkv_track_count=3, qkv_global_bank=True,
qkv_route_formula=layer_plus_position, global_qkv_bank_parameters=1327104.

uv run scripts/validate_experiment.py --id E001_cp_trilinear_attention
PASSED: ok=True, config_count=5, runnable_config_count=5, unimplemented_config_count=0.

uv run scripts/verify_data.py --data_root data/fineweb_edu_100m --manifest data/fineweb_edu_100m/manifest.json --verify_hashes
PASSED: manifest verified.

uv run scripts/verify_data.py --config configs/experiments/E002_multitrack_qkv_shift_register/standard_refactor_control_30m_seed1.yaml --verify_hashes
PASSED: manifest verified.

uv run attn-queue doctor --experiment E001_cp_trilinear_attention
PASSED.

uv run attn-queue doctor --experiment E002_multitrack_qkv_shift_register
PASSED.
```
