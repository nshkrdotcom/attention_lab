from __future__ import annotations

import torch

from attention_lab.mechanisms.cache import ActivationCache, ActivationRecord


def test_activation_cache_metadata_and_tensor_roundtrip(tmp_path):
    tensor = torch.randn(2, 3, requires_grad=True)
    cache = ActivationCache(
        records={
            "resid_pre[0]": ActivationRecord(
                site="resid_pre[0]",
                layer=0,
                tensor=tensor,
                metadata={"shape_semantics": ("batch", "token", "embed")},
            )
        },
        model_name="tiny",
        attention_type="standard",
        checkpoint_path=None,
        config_hash="abc123",
        batch_metadata={"prompt_count": 1},
    )

    metadata = cache.metadata_summary()
    assert metadata["records"]["resid_pre[0]"]["shape"] == [2, 3]
    assert metadata["records"]["resid_pre[0]"]["requires_grad"] is True
    assert "mean" in metadata["records"]["resid_pre[0]"]

    out = tmp_path / "cache.pt"
    cache.save(out, include_tensors=True)
    loaded = ActivationCache.load(out)

    assert loaded.model_name == "tiny"
    assert loaded.attention_type == "standard"
    assert torch.allclose(loaded.records["resid_pre[0]"].tensor, tensor.detach())
    assert loaded.records["resid_pre[0]"].metadata["shape_semantics"] == ("batch", "token", "embed")


def test_activation_cache_summary_save_without_tensors(tmp_path):
    tensor = torch.randn(1, 2, 3)
    cache = ActivationCache(
        records={
            "logits": ActivationRecord(site="logits", layer=None, tensor=tensor, metadata={}),
        },
        model_name="tiny",
        attention_type="standard",
        checkpoint_path=None,
        config_hash=None,
        batch_metadata={},
    )

    out = tmp_path / "activation_summary.json"
    cache.save_metadata_summary(out)

    text = out.read_text(encoding="utf-8")
    assert '"logits"' in text
    assert '"tensor_saved": false' in text
