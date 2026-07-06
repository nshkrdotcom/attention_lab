from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display needed, just PNG output
import matplotlib.pyplot as plt
import torch


def plot_attention_heatmap(weights: torch.Tensor, out_path: str | Path, title: str = "") -> Path:
    """weights: (heads, query_token, key_token) or (batch, heads, query_token,
    key_token) (batch index 0 is used). One subplot per head.
    """
    if weights.dim() == 4:
        weights = weights[0]
    n_heads = weights.shape[0]

    fig, axes = plt.subplots(1, n_heads, figsize=(4 * n_heads, 4), squeeze=False)
    axes = axes[0]
    for head_idx, ax in enumerate(axes):
        ax.imshow(weights[head_idx].detach().cpu().numpy(), cmap="viridis", vmin=0, vmax=1)
        ax.set_title(f"head {head_idx}")
        ax.set_xlabel("key position")
        ax.set_ylabel("query position")
    if title:
        fig.suptitle(title)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    return out_path


def plot_track_selection_histogram(
    selected_track: torch.Tensor, track_count: int, out_path: str | Path, title: str = ""
) -> Path:
    """selected_track: a scalar tensor (multi_qkv_static/train_rotation
    record one route decision per layer) or a (seq_len,) tensor
    (multi_qkv_position_rotation records one per position).
    """
    values = selected_track.detach().cpu().reshape(-1).numpy()

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(values, bins=range(track_count + 1), align="left", rwidth=0.8)
    ax.set_xlabel("selected track")
    ax.set_ylabel("count")
    ax.set_xticks(range(track_count))
    if title:
        ax.set_title(title)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    return out_path
