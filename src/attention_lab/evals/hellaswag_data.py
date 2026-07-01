from __future__ import annotations

import json
from pathlib import Path

import requests
import tiktoken
import torch
from tqdm import tqdm


DATA_CACHE_DIR = Path("hellaswag")
HELLASWAG_URLS = {
    "train": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl",
    "val": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl",
    "test": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_test.jsonl",
}
ENC = tiktoken.get_encoding("gpt2")


def download_file(url: str, path: Path, chunk_size: int = 1024) -> None:
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))
    with path.open("wb") as file, tqdm(
        desc=str(path),
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)


def download(split: str) -> Path:
    if split not in HELLASWAG_URLS:
        raise ValueError(f"Unknown HellaSwag split: {split}")
    DATA_CACHE_DIR.mkdir(exist_ok=True)
    data_path = DATA_CACHE_DIR / f"hellaswag_{split}.jsonl"
    if not data_path.exists():
        print(f"Downloading {HELLASWAG_URLS[split]} to {data_path}...")
        download_file(HELLASWAG_URLS[split], data_path)
    return data_path


def render_example(example: dict):
    ctx = example["ctx"]
    label = example["label"]
    endings = example["endings"]

    data = {
        "label": label,
        "ctx_tokens": None,
        "ending_tokens": [],
    }

    ctx_tokens = ENC.encode(ctx)
    data["ctx_tokens"] = ctx_tokens
    token_rows = []
    mask_rows = []
    for ending in endings:
        ending_tokens = ENC.encode(" " + ending)
        token_rows.append(ctx_tokens + ending_tokens)
        mask_rows.append([0] * len(ctx_tokens) + [1] * len(ending_tokens))
        data["ending_tokens"].append(ending_tokens)

    max_len = max(len(row) for row in token_rows)
    tokens = torch.zeros((4, max_len), dtype=torch.long)
    mask = torch.zeros((4, max_len), dtype=torch.long)
    for i, (token_row, mask_row) in enumerate(zip(token_rows, mask_rows, strict=True)):
        tokens[i, : len(token_row)] = torch.tensor(token_row)
        mask[i, : len(mask_row)] = torch.tensor(mask_row)

    return data, tokens, mask, label


def iterate_examples(split: str):
    data_path = download(split)
    yield from iterate_examples_from_path(data_path)


def iterate_examples_from_path(data_path: Path):
    with data_path.open("r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)
