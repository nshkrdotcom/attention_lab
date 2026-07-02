# Mechanism Probe Report

- config: `configs/experiments/E004_operator_binding_qkv_gauntlet/dynamic_value_query_conditioned_attention_30m_seed2_rung500.yaml`
- checkpoint: `runs/screen/dynamic_value_query_conditioned_attention_30m_seed2_rung500_99b5756e77ed/checkpoints/ckpt_last.pt`
- attention_type: `dynamic_value_query_conditioned_attention`
- tokenizer: `gpt2`
- vocab_size: `50304`
- captured_sites: 24
- missing_sites: 0
- declared_but_unemitted_sites: 0
- intervention_sites: static_value_content, dynamic_gate, dynamic_delta, dynamic_value_output
- invalid_interventions: 0
- interventions: zero, scale
