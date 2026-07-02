# Mechanism Probe Report

- config: `configs/experiments/E002_multitrack_qkv_shift_register/multi_qkv_train_rotation_3track_global_30m_seed1.yaml`
- checkpoint: `runs/experiments/E002_multitrack_qkv_shift_register/multi_qkv_train_rotation_3track_global_30m_seed1/checkpoints/ckpt_last.pt`
- attention_type: `multi_qkv_train_rotation_3track_global`
- tokenizer: `gpt2`
- vocab_size: `50304`
- captured_sites: 30
- missing_sites: 0
- declared_but_unemitted_sites: 0
- intervention_sites: track_q, track_k, track_v, track_out
- invalid_interventions: 2
- interventions: zero, scale

## Invalid Interventions
- `selected_track` / `zero`: selected_track is a discrete route-index hook site. It is captured for diagnostics, but continuous activation interventions are only applied to continuous sites.
- `selected_track` / `scale`: selected_track is a discrete route-index hook site. It is captured for diagnostics, but continuous activation interventions are only applied to continuous sites.
