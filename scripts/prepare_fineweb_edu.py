import argparse
import os
from pathlib import Path
import sys

import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--train_tokens", type=int, required=True)
    parser.add_argument("--val_tokens", type=int, required=True)
    parser.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--name", default="sample-10BT")
    parser.add_argument("--split", default="train")
    parser.add_argument("--tokenizer", default="gpt2")
    parser.add_argument("--shard_tokens", type=int, default=100_000_000)
    return parser.parse_args()


def write_shard(path: Path, tokens: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, tokens.astype(np.uint16, copy=False))
    print(f"wrote {path} ({len(tokens):,} tokens)")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    total_tokens = args.val_tokens + args.train_tokens
    enc = tiktoken.get_encoding(args.tokenizer)
    eot_token = "<|" + "endoftext" + "|>"
    eot = enc._special_tokens[eot_token]
    dataset = load_dataset(args.dataset, name=args.name, split=args.split, streaming=True)

    val_buffer = np.empty(args.val_tokens, dtype=np.uint16)
    train_buffer = np.empty(min(args.shard_tokens, args.train_tokens), dtype=np.uint16)
    val_pos = 0
    train_pos = 0
    train_written = 0
    shard_index = 1

    progress = tqdm(total=total_tokens, unit="tokens", desc="FineWeb-Edu")
    for doc in dataset:
        tokens = np.array([eot] + enc.encode_ordinary(doc["text"]), dtype=np.uint32)
        if (tokens >= 2**16).any():
            raise ValueError("Tokenizer produced an id that does not fit in uint16.")

        offset = 0
        while offset < len(tokens) and val_pos < args.val_tokens:
            n = min(args.val_tokens - val_pos, len(tokens) - offset)
            val_buffer[val_pos : val_pos + n] = tokens[offset : offset + n]
            val_pos += n
            offset += n
            progress.update(n)
            if val_pos == args.val_tokens:
                write_shard(out_dir / "edufineweb_val_000000.npy", val_buffer)

        while offset < len(tokens) and train_written < args.train_tokens:
            remaining_train = args.train_tokens - train_written
            remaining_shard = len(train_buffer) - train_pos
            n = min(remaining_train, remaining_shard, len(tokens) - offset)
            train_buffer[train_pos : train_pos + n] = tokens[offset : offset + n]
            train_pos += n
            train_written += n
            offset += n
            progress.update(n)

            if train_pos == len(train_buffer) or train_written == args.train_tokens:
                shard = train_buffer[:train_pos]
                write_shard(out_dir / f"edufineweb_train_{shard_index:06d}.npy", shard)
                shard_index += 1
                next_size = min(args.shard_tokens, args.train_tokens - train_written)
                if next_size > 0:
                    train_buffer = np.empty(next_size, dtype=np.uint16)
                train_pos = 0

        if train_written >= args.train_tokens:
            break

    progress.close()
    if val_pos < args.val_tokens or train_written < args.train_tokens:
        raise RuntimeError(
            f"Dataset ended before enough tokens were collected: "
            f"val={val_pos:,}/{args.val_tokens:,}, train={train_written:,}/{args.train_tokens:,}"
        )


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    # Avoid a datasets/pyarrow native-extension finalization crash after successful writes.
    os._exit(0)
