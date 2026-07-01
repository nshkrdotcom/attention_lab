import torch.nn as nn


class TrilinearCPCausalSelfAttention(nn.Module):
    """Reserved implementation point for the first novel attention variant."""

    def __init__(self, config):
        super().__init__()
        rank = getattr(config, "cp_rank", None)
        raise NotImplementedError(
            "attention_type='trilinear_cp' is intentionally not implemented in "
            f"the baseline refactor yet. Requested cp_rank={rank!r}."
        )
