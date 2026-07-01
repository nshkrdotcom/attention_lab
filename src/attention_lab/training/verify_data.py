from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from attention_lab.training.config import load_config
from attention_lab.training.data_manifest import DataManifestError, verify_data_manifest


def verify_data_root(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root)
    shards = sorted(data_root.glob("*.npy"))
    if not shards:
        raise SystemExit(f"No .npy shards found in {data_root}")
    for path in shards:
        x = np.load(path)
        if x.size == 0:
            raise SystemExit(f"Empty shard: {path}")
        min_token = int(x.min())
        max_token = int(x.max())
        print(path, x.shape, x.dtype, min_token, max_token)
        if min_token < 0:
            raise SystemExit(f"Negative token id found in shard: {path}")
        if max_token >= 2**16:
            raise SystemExit(f"Token id too large for uint16 shard: {path}")
    manifest = getattr(args, "manifest", None)
    if manifest:
        try:
            verify_data_manifest(data_root, manifest, verify_hashes=bool(getattr(args, "verify_hashes", False)))
        except DataManifestError as exc:
            raise SystemExit(str(exc)) from exc
        print(f"manifest verified: {manifest}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--config", default=None, help="Optional config path to derive data_root and manifest.")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--verify_hashes", action="store_true")
    args = parser.parse_args()
    if args.config:
        config = load_config(args.config)
        args.data_root = args.data_root or config["data"]["data_root"]
        default_manifest = Path(args.data_root) / "manifest.json"
        if args.manifest is None and default_manifest.exists():
            args.manifest = str(default_manifest)
    if args.data_root is None:
        raise SystemExit("--data_root is required unless --config is provided")
    verify_data_root(args)


if __name__ == "__main__":
    main()
