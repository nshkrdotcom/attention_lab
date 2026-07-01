import inspect

import torch


def build_optimizer(
    model: torch.nn.Module,
    weight_decay: float,
    learning_rate: float,
    device_type: str,
    master_process: bool = True,
) -> torch.optim.Optimizer:
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for _, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]

    if master_process:
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == "cuda"
    if master_process:
        print(f"using fused AdamW: {use_fused}")
    return torch.optim.AdamW(
        optim_groups,
        lr=learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=use_fused,
    )

