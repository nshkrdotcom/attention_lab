from __future__ import annotations

import torch

from attention_lab.mechanisms.visualize import plot_attention_heatmap, plot_track_selection_histogram


def test_plot_attention_heatmap_writes_a_real_png(tmp_path):
    weights = torch.rand(2, 6, 6)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    out_path = tmp_path / "heatmap.png"

    result_path = plot_attention_heatmap(weights, out_path, title="test")

    assert result_path == out_path
    assert out_path.exists()
    assert out_path.stat().st_size > 1000  # a real rendered image, not an empty/near-empty file


def test_plot_attention_heatmap_handles_batch_dimension(tmp_path):
    weights = torch.rand(1, 2, 6, 6)
    out_path = tmp_path / "heatmap_batched.png"

    plot_attention_heatmap(weights, out_path)

    assert out_path.exists()
    assert out_path.stat().st_size > 1000


def test_plot_attention_heatmap_handles_a_single_head(tmp_path):
    weights = torch.rand(1, 4, 4)
    out_path = tmp_path / "single_head.png"

    plot_attention_heatmap(weights, out_path)

    assert out_path.exists()


def test_plot_track_selection_histogram_writes_a_real_png(tmp_path):
    selected_track = torch.tensor([0, 1, 2, 1, 0, 2, 2])
    out_path = tmp_path / "tracks.png"

    result_path = plot_track_selection_histogram(selected_track, track_count=3, out_path=out_path, title="test")

    assert result_path == out_path
    assert out_path.exists()
    assert out_path.stat().st_size > 500


def test_plot_track_selection_histogram_handles_a_scalar_track():
    # multi_qkv_static/train_rotation record a single scalar track (not
    # per-position) -- must not crash on a 0-dim tensor.
    from pathlib import Path
    import tempfile

    selected_track = torch.tensor(1)
    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "scalar_track.png"
        plot_track_selection_histogram(selected_track, track_count=3, out_path=out_path)
        assert out_path.exists()
