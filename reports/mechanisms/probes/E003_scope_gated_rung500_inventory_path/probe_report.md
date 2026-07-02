# Mechanism Probe Report

- config: `configs/experiments/E003_qkv_architecture_gauntlet/scope_gated_qkv_30m_seed1_rung500.yaml`
- checkpoint: `runs/screen/scope_gated_qkv_30m_seed1_rung500_bb3de557aae8/checkpoints/ckpt_last.pt`
- attention_type: `scope_gated_qkv`
- tokenizer: `gpt2`
- vocab_size: `50304`
- captured_sites: 30
- missing_sites: 0
- declared_but_unemitted_sites: 0
- intervention_sites: content_out, scope_out, gate, content_scope_product, gated_content
- invalid_interventions: 0
- interventions: zero, scale
