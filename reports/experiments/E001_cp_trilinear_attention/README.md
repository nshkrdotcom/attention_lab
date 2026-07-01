# E001 CP Trilinear Attention

This report directory tracks the first CP-bilinear / CP-trilinear attention
experiment. The implementation and tiny integration checks are complete; full
3000-step comparison runs are prepared for manual execution and are not claimed here.

## Paths

- Plan: `docs/experiments/E001_cp_trilinear_attention_plan.md`
- Configs: `configs/experiments/E001_cp_trilinear_attention/`
- Runs: `runs/experiments/E001_cp_trilinear_attention/`
- Full-run scripts: `scripts/experiments/E001_cp_trilinear_attention/`
- Implementation notes: `implementation_notes.md`
- Results report: `results.md`
- Run index: `run_index.md`, `run_index.json`
- Comparison template: `comparison_template.json`
- Results template: `results_template.md`

## Required Completion Gate

Before any full-run result is reported, every completed run must pass:

```text
verify_run.py --expect-complete-training --expect-sample --expect-eval-loss --expect-hellaswag --expect-data-manifest
```

After all required full-run summaries exist, run:

```bash
scripts/experiments/E001_cp_trilinear_attention/compare_full_runs.sh
```
