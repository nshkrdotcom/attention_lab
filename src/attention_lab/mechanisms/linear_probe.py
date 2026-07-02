from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class LinearProbeDataset:
    X: np.ndarray
    y: np.ndarray
    pair_ids: np.ndarray
    template_ids: np.ndarray
    family_ids: np.ndarray
    variants: np.ndarray

    def validate(self) -> None:
        n = int(self.X.shape[0])
        if self.X.ndim != 2:
            raise ValueError("linear probe features must be a 2D array")
        for name in ("y", "pair_ids", "template_ids", "family_ids", "variants"):
            value = getattr(self, name)
            if len(value) != n:
                raise ValueError(f"{name} length {len(value)} does not match feature rows {n}")
        unique = set(np.asarray(self.y).astype(int).tolist())
        if unique != {0, 1}:
            raise ValueError("linear probe labels must contain both 0 and 1")


@dataclass(frozen=True)
class GroupedSplit:
    train_indices: np.ndarray
    test_indices: np.ndarray
    group_field: str
    pair_group_leakage: bool
    template_group_leakage: bool


@dataclass(frozen=True)
class ProbeFit:
    auc: float | None
    weights: np.ndarray
    bias: float
    scores: np.ndarray
    labels: np.ndarray
    test_indices: np.ndarray
    unavailable_reason: str | None = None


@dataclass(frozen=True)
class ProbeWithNullsResult:
    primary: ProbeFit
    shuffled: ProbeFit
    grouped_split: GroupedSplit
    auc_minus_shuffled_auc: float | None
    metadata: dict[str, Any]


def grouped_train_test_split(
    dataset: LinearProbeDataset,
    *,
    seed: int,
    test_fraction: float = 0.25,
    group_by_template: bool = False,
) -> GroupedSplit:
    dataset.validate()
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between 0 and 1")
    group_values = dataset.template_ids if group_by_template else dataset.pair_ids
    group_field = "template_id" if group_by_template else "pair_id"
    unique_groups = np.asarray(sorted(set(group_values.tolist())))
    if len(unique_groups) < 2:
        raise ValueError(f"need at least two {group_field} groups for a grouped split")
    rng = np.random.default_rng(seed)
    shuffled = unique_groups.copy()
    rng.shuffle(shuffled)
    n_test = max(1, int(round(len(shuffled) * test_fraction)))
    if n_test >= len(shuffled):
        n_test = len(shuffled) - 1
    test_groups = set(shuffled[:n_test].tolist())
    test_mask = np.asarray([value in test_groups for value in group_values])
    train_indices = np.flatnonzero(~test_mask)
    test_indices = np.flatnonzero(test_mask)

    _validate_split_classes(dataset.y, train_indices, test_indices)
    train_pairs = set(dataset.pair_ids[train_indices].tolist())
    test_pairs = set(dataset.pair_ids[test_indices].tolist())
    train_templates = set(dataset.template_ids[train_indices].tolist())
    test_templates = set(dataset.template_ids[test_indices].tolist())
    return GroupedSplit(
        train_indices=train_indices,
        test_indices=test_indices,
        group_field=group_field,
        pair_group_leakage=bool(train_pairs & test_pairs),
        template_group_leakage=bool(train_templates & test_templates),
    )


def run_linear_probe_with_nulls(
    dataset: LinearProbeDataset,
    *,
    seed: int,
    min_n: int,
    group_by_template: bool = False,
    test_fraction: float = 0.25,
    training_steps: int = 200,
) -> ProbeWithNullsResult:
    dataset.validate()
    pair_count = len(set(dataset.pair_ids.tolist()))
    if pair_count < min_n:
        raise ValueError(f"minimum N failed: {pair_count} pairs < min_n={min_n}")
    split = grouped_train_test_split(
        dataset,
        seed=seed,
        test_fraction=test_fraction,
        group_by_template=group_by_template,
    )
    primary = fit_linear_probe(dataset, split=split, seed=seed, training_steps=training_steps)
    shuffled_dataset = _shuffle_labels(dataset, seed=seed + 17)
    shuffled = fit_linear_probe(shuffled_dataset, split=split, seed=seed + 23, training_steps=training_steps)
    auc_delta = None
    if primary.auc is not None and shuffled.auc is not None:
        auc_delta = float(primary.auc - shuffled.auc)
    return ProbeWithNullsResult(
        primary=primary,
        shuffled=shuffled,
        grouped_split=split,
        auc_minus_shuffled_auc=auc_delta,
        metadata={
            "min_n": int(min_n),
            "pair_count": int(pair_count),
            "group_by_template": bool(group_by_template),
            "training_steps": int(training_steps),
        },
    )


def fit_linear_probe(
    dataset: LinearProbeDataset,
    *,
    split: GroupedSplit,
    seed: int,
    training_steps: int = 200,
    learning_rate: float = 0.05,
    weight_decay: float = 1e-4,
) -> ProbeFit:
    dataset.validate()
    torch.manual_seed(seed)
    X = np.asarray(dataset.X, dtype=np.float32)
    y = np.asarray(dataset.y, dtype=np.float32)
    train_X = X[split.train_indices]
    test_X = X[split.test_indices]
    train_y = y[split.train_indices]
    test_y = y[split.test_indices]
    _validate_split_classes(y, split.train_indices, split.test_indices)

    mean = train_X.mean(axis=0, keepdims=True)
    std = train_X.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    train_X = (train_X - mean) / std
    test_X = (test_X - mean) / std

    x_tensor = torch.tensor(train_X, dtype=torch.float32)
    y_tensor = torch.tensor(train_y, dtype=torch.float32)
    weights = torch.zeros(x_tensor.shape[1], dtype=torch.float32, requires_grad=True)
    bias = torch.zeros((), dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([weights, bias], lr=learning_rate, weight_decay=weight_decay)
    for _ in range(training_steps):
        optimizer.zero_grad(set_to_none=True)
        logits = x_tensor @ weights + bias
        loss = F.binary_cross_entropy_with_logits(logits, y_tensor)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        test_scores = torch.sigmoid(torch.tensor(test_X, dtype=torch.float32) @ weights + bias).cpu().numpy()
    auc = roc_auc(test_y.astype(int), test_scores)
    return ProbeFit(
        auc=auc,
        weights=weights.detach().cpu().numpy().astype(float),
        bias=float(bias.detach().cpu().item()),
        scores=test_scores.astype(float),
        labels=test_y.astype(int),
        test_indices=split.test_indices.copy(),
    )


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores).astype(float)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    comparisons = 0.0
    total = 0
    for p_score in pos:
        comparisons += float(np.sum(p_score > neg))
        comparisons += 0.5 * float(np.sum(p_score == neg))
        total += len(neg)
    return float(comparisons / total)


def _shuffle_labels(dataset: LinearProbeDataset, *, seed: int) -> LinearProbeDataset:
    rng = np.random.default_rng(seed)
    shuffled = np.asarray(dataset.y).copy()
    rng.shuffle(shuffled)
    return LinearProbeDataset(
        X=dataset.X,
        y=shuffled,
        pair_ids=dataset.pair_ids,
        template_ids=dataset.template_ids,
        family_ids=dataset.family_ids,
        variants=dataset.variants,
    )


def _validate_split_classes(labels: np.ndarray, train_indices: np.ndarray, test_indices: np.ndarray) -> None:
    train_classes = set(np.asarray(labels)[train_indices].astype(int).tolist())
    test_classes = set(np.asarray(labels)[test_indices].astype(int).tolist())
    if train_classes != {0, 1}:
        raise ValueError("grouped train split must contain both classes")
    if test_classes != {0, 1}:
        raise ValueError("grouped test split must contain both classes")
