# Mechanism Probe Report

- config: `configs/experiments/E001_cp_trilinear_attention/standard_30m_seed1.yaml`
- checkpoint: `runs/experiments/E001_cp_trilinear_attention/standard_30m_seed1/checkpoints/ckpt_last.pt`
- attention_type: `standard`
- tokenizer: `gpt2`
- vocab_size: `50304`
- captured_sites: 24
- missing_sites: 0
- declared_but_unemitted_sites: 0
- intervention_sites: attn_q, attn_k, attn_v, attn_out
- invalid_interventions: 0
- interventions: zero, scale
