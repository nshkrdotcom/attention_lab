from __future__ import annotations

import argparse


EXPORT_NOT_IMPLEMENTED_MESSAGE = (
    "HF export is not implemented yet. This is required before lm-evaluation-harness integration."
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.parse_args()
    raise SystemExit(EXPORT_NOT_IMPLEMENTED_MESSAGE)


if __name__ == "__main__":
    main()
