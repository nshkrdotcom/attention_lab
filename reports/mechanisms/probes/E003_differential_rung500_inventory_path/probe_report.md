# Mechanism Probe Report

- config: `configs/experiments/E003_qkv_architecture_gauntlet/differential_qkv_anti_value_30m_seed1_rung500.yaml`
- checkpoint: `runs/screen/differential_qkv_anti_value_30m_seed1_rung500_407d76f5952e/checkpoints/ckpt_last.pt`
- attention_type: `differential_qkv_anti_value`
- tokenizer: `gpt2`
- vocab_size: `50304`
- captured_sites: 60
- missing_sites: 0
- declared_but_unemitted_sites: 0
- intervention_sites: pos_q, pos_k, pos_v, neg_q, neg_k, neg_v, pos_out, neg_out, branch_delta, lambda
- invalid_interventions: 0
- interventions: zero, scale
