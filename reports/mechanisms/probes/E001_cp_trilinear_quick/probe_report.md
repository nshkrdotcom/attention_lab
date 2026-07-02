# Mechanism Probe Report

- config: `configs/experiments/E001_cp_trilinear_attention/cp_trilinear_r8_30m_seed1.yaml`
- checkpoint: `runs/experiments/E001_cp_trilinear_attention/cp_trilinear_r8_30m_seed1/checkpoints/ckpt_last.pt`
- attention_type: `cp_trilinear`
- tokenizer: `gpt2`
- vocab_size: `50304`
- captured_sites: 36
- missing_sites: 0
- declared_but_unemitted_sites: 0
- intervention_sites: attn_q, attn_k, attn_v, cp_score, cp_lambda, cp_output
- invalid_interventions: 0
- interventions: zero, scale
