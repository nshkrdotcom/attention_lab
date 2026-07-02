from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F

from attention_lab.mechanisms.statistics import auc_score


@dataclass(frozen=True)
class GroupedSplit:
    train_indices: np.ndarray
    test_indices: np.ndarray
    group_field: str
    train_groups: tuple[str, ...]
    test_groups: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "group_field": self.group_field,
            "train_size": int(len(self.train_indices)),
            "test_size": int(len(self.test_indices)),
            "train_groups": list(self.train_groups),
            "test_groups": list(self.test_groups),
            "grouped_split_discipline": True,
        }


@dataclass(frozen=True)
class LinearProbeResult:
    auc: float
    scores: np.ndarray
    labels: np.ndarray
    split: GroupedSplit
    weight: np.ndarray
    bias: float
    mean: np.ndarray
    std: np.ndarray
    train_loss: float
    feature_dim: int

    def to_metrics(self) -> dict[str, object]:
        return {
            "linear_probe_auc": float(self.auc),
            "feature_dim": int(self.feature_dim),
            "train_loss": float(self.train_loss),
            "split": self.split.to_dict(),
        }


def grouped_train_test_split(
    *,
    pair_ids: list[str],
    template_ids: list[str],
    seed: int,
    test_fraction: float = 0.3,
    group_by_template: bool = True,
) -> GroupedSplit:
    if len(pair_ids) != len(template_ids):
        raise ValueError("pair_ids and template_ids must have matching length")
    group_field = "template_id" if group_by_template else "pair_id"
    group_ids = np.asarray(template_ids if group_by_template else pair_ids)
    unique_groups = np.unique(group_ids)
    if len(unique_groups) < 2:
        raise ValueError(f"grouped split requires at least two {group_field} groups")
    rng = np.random.default_rng(seed)
    shuffled = unique_groups.copy()
    rng.shuffle(shuffled)
    test_group_count = max(1, int(round(len(shuffled) * test_fraction)))
    if test_group_count >= len(shuffled):
        test_group_count = len(shuffled) - 1
    test_groups = set(shuffled[:test_group_count])
    train_groups = set(shuffled[test_group_count:])
    train_indices = np.flatnonzero(np.asarray([group not in test_groups for group in group_ids]))
    test_indices = np.flatnonzero(np.asarray([group in test_groups for group in group_ids]))
    if len(train_indices) == 0 or len(test_indices) == 0:
        raise ValueError("grouped split produced an empty train or test partition")
    return GroupedSplit(
        train_indices=train_indices,
        test_indices=test_indices,
        group_field=group_field,
        train_groups=tuple(sorted(train_groups)),
        test_groups=tuple(sorted(test_groups)),
    )


def train_linear_probe(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    pair_ids: list[str],
    template_ids: list[str],
    seed: int,
    split: GroupedSplit | None = None,
    group_by_template: bool = True,
    steps: int = 250,
    lr: float = 0.05,
    weight_decay: float = 1e-3,
) -> LinearProbeResult:
    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.float32)
    if features.ndim != 2:
        raise ValueError("linear probe features must be a 2D array")
    if len(features) != len(labels):
        raise ValueError("features and labels must have matching length")
    if len(np.unique(labels)) < 2:
        raise ValueError("linear probe requires both positive and negative labels")
    if split is None:
        split = grouped_train_test_split(
            pair_ids=pair_ids,
            template_ids=template_ids,
            seed=seed,
            group_by_template=group_by_template,
        )
    _validate_split_labels(labels, split)

    torch.manual_seed(seed)
    train_x = torch.tensor(features[split.train_indices], dtype=torch.float32)
    train_y = torch.tensor(labels[split.train_indices], dtype=torch.float32)
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
    train_x = (train_x - mean) / std

    weight = torch.zeros(features.shape[1], dtype=torch.float32, requires_grad=True)
    bias = torch.zeros((), dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([weight, bias], lr=lr, weight_decay=weight_decay)
    loss = torch.tensor(float("nan"))
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = train_x @ weight + bias
        loss = F.binary_cross_entropy_with_logits(logits, train_y)
        loss.backward()
        optimizer.step()

    all_x = (torch.tensor(features, dtype=torch.float32) - mean) / std
    scores = (all_x @ weight.detach() + bias.detach()).numpy()
    auc = auc_score(labels[split.test_indices].astype(int), scores[split.test_indices])
    return LinearProbeResult(
        auc=auc,
        scores=scores,
        labels=labels.astype(int),
        split=split,
        weight=weight.detach().numpy(),
        bias=float(bias.detach().item()),
        mean=mean.squeeze(0).numpy(),
        std=std.squeeze(0).numpy(),
        train_loss=float(loss.detach().item()),
        feature_dim=int(features.shape[1]),
    )


def train_shuffled_label_null(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    pair_ids: list[str],
    template_ids: list[str],
    split: GroupedSplit,
    seed: int,
) -> LinearProbeResult:
    rng = np.random.default_rng(seed)
    shuffled_labels = np.asarray(labels).copy()
    train_labels = shuffled_labels[split.train_indices].copy()
    rng.shuffle(train_labels)
    shuffled_labels[split.train_indices] = train_labels
    return train_linear_probe(
        features,
        shuffled_labels,
        pair_ids=pair_ids,
        template_ids=template_ids,
        split=split,
        seed=seed,
    )


def score_with_probe(features: np.ndarray, result: LinearProbeResult) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2:
        raise ValueError("features must be 2D")
    if features.shape[1] != result.feature_dim:
        raise ValueError(f"feature dimension {features.shape[1]} does not match probe dimension {result.feature_dim}")
    standardized = (features - result.mean) / result.std
    return standardized @ result.weight + result.bias


def _validate_split_labels(labels: np.ndarray, split: GroupedSplit) -> None:
    for name, indices in (("train", split.train_indices), ("test", split.test_indices)):
        if len(np.unique(labels[indices])) < 2:
            raise ValueError(f"linear probe {name} split must contain both labels")
