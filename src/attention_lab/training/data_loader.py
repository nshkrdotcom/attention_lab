from pathlib import Path
from typing import Any

import numpy as np
import torch


def load_tokens(filename: str | Path) -> torch.Tensor:
    tokens = np.load(filename)
    tokens = tokens.astype(np.int32, copy=False)
    return torch.tensor(tokens, dtype=torch.long)


class TokenShardLoader:
    def __init__(
        self,
        data_root: str | Path,
        B: int,
        T: int,
        process_rank: int,
        num_processes: int,
        split: str,
        master_process: bool = True,
    ):
        if split not in {"train", "val"}:
            raise ValueError("split must be 'train' or 'val'")

        self.data_root = Path(data_root)
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.split = split

        self.shards = sorted(self.data_root.glob(f"*{split}*.npy"))
        if not self.shards:
            raise FileNotFoundError(f"No {split} shards found in {self.data_root}")
        if master_process:
            print(f"found {len(self.shards)} {split} shard(s) in {self.data_root}")
        self.reset()

    def reset(self) -> None:
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank
        self._validate_current_shard()

    def _validate_current_shard(self) -> None:
        min_tokens = self.B * self.T * self.num_processes + 1
        if len(self.tokens) < min_tokens:
            raise ValueError(
                f"Shard {self.shards[self.current_shard]} has {len(self.tokens)} tokens, "
                f"but at least {min_tokens} are needed for B={self.B}, T={self.T}, "
                f"num_processes={self.num_processes}."
            )

    def _advance_shard(self) -> None:
        self.current_shard = (self.current_shard + 1) % len(self.shards)
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank
        self._validate_current_shard()

    def state_dict(self) -> dict[str, Any]:
        return {
            "data_root": str(self.data_root),
            "B": self.B,
            "T": self.T,
            "process_rank": self.process_rank,
            "num_processes": self.num_processes,
            "split": self.split,
            "current_shard": self.current_shard,
            "current_position": self.current_position,
            "shards": [str(path) for path in self.shards],
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        for key in ("B", "T", "process_rank", "num_processes", "split"):
            if state.get(key) != getattr(self, key):
                raise ValueError(f"Cannot load loader state with mismatched {key}: {state.get(key)!r}")
        state_shards = [Path(path).name for path in state.get("shards", [])]
        current_shards = [path.name for path in self.shards]
        if state_shards and state_shards != current_shards:
            raise ValueError("Cannot load loader state with different shard ordering")
        current_shard = int(state["current_shard"])
        if current_shard < 0 or current_shard >= len(self.shards):
            raise ValueError(f"Invalid loader current_shard: {current_shard}")
        current_position = int(state["current_position"])
        if current_position < 0:
            raise ValueError(f"Invalid loader current_position: {current_position}")

        self.current_shard = current_shard
        self.tokens = load_tokens(self.shards[self.current_shard])
        self._validate_current_shard()
        if current_position + self.B * self.T + 1 > len(self.tokens):
            raise ValueError("Loader state current_position cannot produce a full next batch")
        self.current_position = current_position

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        B, T = self.B, self.T
        if self.current_position + B * T + 1 > len(self.tokens):
            self._advance_shard()

        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)

        self.current_position += B * T * self.num_processes
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self._advance_shard()
        return x, y
