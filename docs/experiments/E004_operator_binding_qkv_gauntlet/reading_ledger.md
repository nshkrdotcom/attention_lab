# E004 Reading Ledger

Status: prepared before code edits.

## Required Repo Context

- `README.md`
- `AGENTS.md`
- `EXPERIMENT_STATUS_AND_TECHNICAL_NOTES.md`
- `docs/architecture_experiment_contract.md`
- `docs/architecture_variant_checklist.md`
- `docs/pre_experiment_cleanup_checklist.md`
- `docs/guides/experiment_queue_discipline_checklist.md`
- `docs/experiments/experiments.yaml`

## E003 Context

- `docs/experiments/E003_qkv_architecture_gauntlet_plan.md`
- `docs/experiments/E003_qkv_architecture_gauntlet/`
- `reports/experiments/E003_qkv_architecture_gauntlet/`
- `configs/experiments/E003_qkv_architecture_gauntlet/`
- `scripts/experiments/E003_qkv_architecture_gauntlet/`

E003 established the reusable screen-first gauntlet shape: standard control, candidate configs, generated rung configs, promotion reports, mechanism checks, and machine decisions. E004 reuses this infrastructure and does not treat its own generated configs or reports as run evidence.

## Queue, Gauntlet, And Reporting Code

- `src/attention_lab/queue/cli.py`
- `src/attention_lab/queue/gauntlet.py`
- `src/attention_lab/queue/ledger.py`
- `src/attention_lab/queue/mechanism_checks.py`
- `src/attention_lab/queue/promotion.py`
- `src/attention_lab/queue/runner.py`
- `src/attention_lab/queue/screener.py`
- `src/attention_lab/queue/reporting.py`
- `src/attention_lab/training/config.py`
- `src/attention_lab/training/validate_experiment.py`

Phase use:

- Mechanism checks: extend `mechanism_checks.py` without weakening E001/E002/E003.
- Reports: keep promotion reports as the source for gauntlet decisions and lift E004 diagnostic summaries into `gauntlet_report`.
- CLI: reuse generic `gauntlet-plan`, `gauntlet-run`, and `gauntlet-report`.
- Validation: add strict E004 config keys and mechanism-check names.

## Model And Attention Code

- `src/attention_lab/models/gpt.py`
- `src/attention_lab/models/attention/registry.py`
- `src/attention_lab/models/attention/__init__.py`
- `src/attention_lab/models/attention/standard.py`
- `src/attention_lab/models/attention/differential_qkv.py`
- `src/attention_lab/models/attention/scope_gated_qkv.py`
- `src/attention_lab/models/attention/cp_bilinear.py`
- `src/attention_lab/models/attention/cp_trilinear.py`
- `src/attention_lab/models/attention/multi_qkv_common.py`
- `src/attention_lab/models/attention/multi_qkv_static.py`
- `src/attention_lab/models/attention/multi_qkv_train_rotation.py`
- `src/attention_lab/models/attention/multi_qkv_position_rotation.py`

Phase use:

- Phase 1: implement `operator_valued_attention` as a registry module with SDPA, fixed small operator set, negative suppressive contribution, diagnostics, and tests.
- Phase 2: implement `q3k3v3_role_routed_attention` with diagonal role interactions first and optional full grid behind config validation.
- Phase 3: implement `dynamic_value_query_conditioned_attention` with receiver-only value gate and clear rejection for pairwise mode.
- Phase 4: register all attention types and extend `GPTConfig`.

## Design Notes

Primary 2026-06-30 QKV docset read:

- `/home/home/p/g/j/jido_brainstorm/nshkrdotcom/docs/20260630/qkv/0001_gpt.md`
- `/home/home/p/g/j/jido_brainstorm/nshkrdotcom/docs/20260630/qkv/0002.md`
- `/home/home/p/g/j/jido_brainstorm/nshkrdotcom/docs/20260630/qkv/0003.md`
- `/home/home/p/g/j/jido_brainstorm/nshkrdotcom/docs/20260630/qkv/0004.md`
- `/home/home/p/g/j/jido_brainstorm/nshkrdotcom/docs/20260630/qkv/0010_claude_sonnet_5_first_prompt.md`
- `/home/home/p/g/j/jido_brainstorm/nshkrdotcom/docs/20260630/qkv/0011.md`
- `/home/home/p/g/j/jido_brainstorm/nshkrdotcom/docs/20260630/qkv/0020_hydrahead_check.md`
- `/home/home/p/g/j/jido_brainstorm/nshkrdotcom/docs/20260630/qkv/0021_explaining_attention_with_program_synthesis_check.md`
- `/home/home/p/g/j/jido_brainstorm/nshkrdotcom/docs/20260630/qkv/0030_overcomplete_training_qkv.md`

Design takeaways for E004:

- Operator-valued attention tests whether a retrieved message needs multiple write/update modes rather than one fixed OV map.
- Q3K3V3 tests whether content-like, operator-like, and binding-like streams can remain active and separable.
- Dynamic value query-conditioned attention tests receiver-conditioned read modes while holding routing fixed.
- E004 is a survival and diagnostics gauntlet, not an efficiency or model-quality claim.

## Implementation Checklist

- [x] Add failing attention tests for `operator_valued_attention`.
- [x] Implement `operator_valued_attention`.
- [x] Add `operator_valued_activity`.
- [x] Add failing attention tests for `q3k3v3_role_routed_attention`.
- [x] Implement `q3k3v3_role_routed_attention`.
- [x] Add `q3k3v3_role_activity`.
- [x] Add failing attention tests for `dynamic_value_query_conditioned_attention`.
- [x] Implement `dynamic_value_query_conditioned_attention`.
- [x] Add `dynamic_value_activity`.
- [x] Register attention types and config validation.
- [x] Add E004 configs and gauntlet policy.
- [x] Add E004 docs, hypotheses, report templates, and experiment registration.
- [x] Add script wrapper.
- [ ] Update README, AGENTS, status, and queue guide.
- [ ] Run targeted tests and full QC.
- [ ] Commit and push.
