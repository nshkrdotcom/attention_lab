# Mechanism Probe Report

- config: `configs/experiments/E004_operator_binding_qkv_gauntlet/standard_refactor_control_30m_seed2_rung500.yaml`
- checkpoint: `runs/screen/standard_refactor_control_30m_seed2_rung500_3cc31db15c20/checkpoints/ckpt_last.pt`
- attention_type: `standard`
- tokenizer: `gpt2`
- vocab_size: `50304`
- captured_sites: 24
- missing_sites: 0
- declared_but_unemitted_sites: 0
- intervention_sites: attn_q, attn_k, attn_v, attn_out
- invalid_interventions: 0
- interventions: zero, scale
