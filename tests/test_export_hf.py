from __future__ import annotations

import pytest

from attention_lab.export_hf import EXPORT_NOT_IMPLEMENTED_MESSAGE, main


def test_export_hf_stub_fails_clearly(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["export_hf.py", "--checkpoint", "ckpt.pt", "--out_dir", "exported"],
    )
    with pytest.raises(SystemExit, match=EXPORT_NOT_IMPLEMENTED_MESSAGE):
        main()
