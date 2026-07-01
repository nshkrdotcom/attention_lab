# Baseline 30M FineWeb-Edu 100M Summary

Date: 2026-06-30

This is the accurate-size alias for the historical
`baseline_15m_fineweb100m_seed1` run. The completed run used the same model, data, and
training recipe now exposed as `configs/baseline_30m_fineweb100m.yaml`.

Git commit at historical run time: `0760b275d46a5c920d79761609b59600d602f6f8`

## Model

- Config for new runs: `configs/baseline_30m_fineweb100m.yaml`
- Historical completed config: `configs/baseline_15m_fineweb100m.yaml`
- Attention: `standard`
- Block size: `1024`
- Layers: `6`
- Heads: `6`
- Embedding size: `384`
- Vocabulary size: `50304`
- Parameters, excluding positional embeddings: `29938560`
- Parameters, including positional embeddings: `30331776`

## Data

- Dataset: `HuggingFaceFW/fineweb-edu`
- Dataset config: `sample-10BT`
- Tokenizer: `gpt2`
- Train tokens: `100000000`
- Validation tokens: `4000000`
- Manifest: `data/fineweb_edu_100m/manifest.json`
- Manifest SHA256: `3302a779a89ee9f77a0c5717a963dd2744b5ee89dfef56b8c0d098cb61718f17`
- Train shard SHA256: `7bc89b5e75a6eba3e471c5434b03e98dd3be6aaa8ce043a9aae564bf51e25893`
- Validation shard SHA256: `efb01e4b8dad9ce4aa906ca8afbb36bd0329d4135e00741556eb4a70689f784c`

## Training

- Device: `cuda`
- Dtype: `bfloat16`
- Compile: `false`
- Micro batch: `B=4`, `T=1024`
- Total batch size: `262144`
- Steps: `3000`
- Training token positions: `786432000`
- Data-order note: this is multiple passes over the 100M-token train shard, not a
  unique 786M-token corpus.

## Result

- Initial validation loss: `10.910149574279785`
- Final validation loss: `4.081209182739258`
- Best validation loss: `4.081209182739258`
- Initial validation perplexity: `54729.03074804456`
- Final validation perplexity: `59.2170307875361`
- Median tokens/sec: `107022.7422894312`
- Peak allocated VRAM: `3240.92431640625` MB
- Peak reserved VRAM: unavailable in the historical metrics; new runs log it as
  `peak_vram_reserved_mb`.
- Checkpoints written: `3`
- Historical checkpoint: `runs/baseline_15m_fineweb100m_seed1/checkpoints/ckpt_last.pt`

## Bounded HellaSwag

- Split: `val`
- Examples: `100`
- Normalized correct: `34`
- Normalized accuracy: `0.34`
- Data URL: `https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl`
- Data SHA256: `0aa3b88843990f3f10a97b9575c94d7b71fb2205240ba04ae4884d9e9c992588`

## Verification

`scripts/verify_run.py` passed with:

```text
--expect-complete-training --expect-sample --expect-eval-loss --expect-hellaswag
```

After the pre-experiment cleanup pass, new 30M runs should verify with:

```text
--expect-complete-training --expect-sample --expect-eval-loss --expect-hellaswag --expect-data-manifest
```
