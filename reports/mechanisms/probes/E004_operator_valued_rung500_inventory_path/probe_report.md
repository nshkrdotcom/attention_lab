# Mechanism Probe Report

- config: `configs/experiments/E004_operator_binding_qkv_gauntlet/operator_valued_attention_30m_seed2_rung500.yaml`
- checkpoint: `runs/screen/operator_valued_attention_30m_seed2_rung500_b6177af38f93/checkpoints/ckpt_last.pt`
- attention_type: `operator_valued_attention`
- tokenizer: `gpt2`
- vocab_size: `50304`
- captured_sites: 42
- missing_sites: 0
- declared_but_unemitted_sites: 0
- intervention_sites: operator_probs, operator_add_out, operator_suppress_out, operator_gate_out, operator_transform_out, operator_bind_out, operator_combined_out
- invalid_interventions: 0
- interventions: zero, scale
