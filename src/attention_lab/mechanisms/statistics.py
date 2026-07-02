from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from attention_lab.mechanisms.linear_probe import LinearProbeDataset, roc_auc


@dataclass(frozen=True)
class BootstrapResult:
    estimate: float
    low: float
    high: float
    p_value: float
    samples: int
    valid: bool = True
    reason: str | None = None


def bootstrap_ci(
    values: np.ndarray,
    *,
    seed: int,
    samples: int,
    alpha: float = 0.05,
    expected_direction: str = "positive",
) -> BootstrapResult:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return BootstrapResult(0.0, 0.0, 0.0, 1.0, samples, valid=False, reason="no finite values")
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=float)
    for idx in range(samples):
        sample = rng.choice(values, size=values.size, replace=True)
        estimates[idx] = float(np.mean(sample))
    low, high = np.quantile(estimates, [alpha / 2.0, 1.0 - alpha / 2.0])
    estimate = float(np.mean(values))
    p_value = _directional_p_value(estimates, expected_direction=expected_direction)
    return BootstrapResult(estimate=estimate, low=float(low), high=float(high), p_value=p_value, samples=samples)


def bootstrap_auc_difference(
    labels: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    *,
    seed: int,
    samples: int,
    expected_direction: str = "positive",
) -> BootstrapResult:
    labels = np.asarray(labels).astype(int)
    scores_a = np.asarray(scores_a, dtype=float)
    scores_b = np.asarray(scores_b, dtype=float)
    if not (len(labels) == len(scores_a) == len(scores_b)):
        raise ValueError("labels and score arrays must have the same length")
    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    indices = np.arange(len(labels))
    for _ in range(samples):
        sample_idx = rng.choice(indices, size=len(indices), replace=True)
        auc_a = roc_auc(labels[sample_idx], scores_a[sample_idx])
        auc_b = roc_auc(labels[sample_idx], scores_b[sample_idx])
        if auc_a is not None and auc_b is not None:
            estimates.append(float(auc_a - auc_b))
    if not estimates:
        return BootstrapResult(0.0, 0.0, 0.0, 1.0, samples, valid=False, reason="bootstrap samples lacked both classes")
    estimates_array = np.asarray(estimates, dtype=float)
    low, high = np.quantile(estimates_array, [0.025, 0.975])
    auc_a = roc_auc(labels, scores_a)
    auc_b = roc_auc(labels, scores_b)
    if auc_a is None or auc_b is None:
        return BootstrapResult(0.0, 0.0, 0.0, 1.0, samples, valid=False, reason="test split lacked both classes")
    return BootstrapResult(
        estimate=float(auc_a - auc_b),
        low=float(low),
        high=float(high),
        p_value=_directional_p_value(estimates_array, expected_direction=expected_direction),
        samples=samples,
    )


def target_vs_decoy_specificity(
    dataset: LinearProbeDataset,
    scores: np.ndarray,
    *,
    seed: int,
    samples: int,
) -> BootstrapResult:
    scores = np.asarray(scores, dtype=float)
    if len(scores) != len(dataset.test_indices if hasattr(dataset, "test_indices") else dataset.y):
        # The suite passes a test-split dataset here; direct callers may pass a full dataset.
        pass
    pair_effects = []
    for pair_id in sorted(set(dataset.pair_ids.tolist())):
        mask = dataset.pair_ids == pair_id
        variants = dataset.variants[mask]
        pair_scores = scores[mask]
        by_variant = {variant: pair_scores[variants == variant] for variant in set(variants.tolist())}
        required = {"x_pos", "x_neg", "x_para", "x_decoy"}
        if not required.issubset(by_variant):
            continue
        target_score = float(np.mean(np.concatenate([by_variant["x_pos"], by_variant["x_para"]])))
        neg_score = float(np.mean(by_variant["x_neg"]))
        decoy_score = float(np.mean(by_variant["x_decoy"]))
        pair_effects.append((target_score - neg_score) - (decoy_score - neg_score))
    return bootstrap_ci(np.asarray(pair_effects, dtype=float), seed=seed, samples=samples)


def fdr_bh(cells: list[dict[str, Any]], *, alpha: float) -> list[dict[str, Any]]:
    if not cells:
        return []
    indexed = []
    for idx, cell in enumerate(cells):
        p_value = float(cell["p_value"])
        if not 0.0 <= p_value <= 1.0:
            raise ValueError(f"p_value must be in [0,1], got {p_value}")
        indexed.append((idx, p_value, cell))
    ordered = sorted(indexed, key=lambda item: item[1])
    m = len(ordered)
    max_reject_rank = -1
    for rank, (_, p_value, _) in enumerate(ordered, start=1):
        if p_value <= (rank / m) * alpha:
            max_reject_rank = rank
    adjusted_by_idx: dict[int, dict[str, Any]] = {}
    running_q = 1.0
    for reverse_rank, (original_idx, p_value, cell) in enumerate(reversed(ordered), start=1):
        rank = m - reverse_rank + 1
        running_q = min(running_q, p_value * m / rank)
        corrected = dict(cell)
        corrected["q_value"] = float(min(1.0, running_q))
        corrected["rejected"] = bool(rank <= max_reject_rank)
        adjusted_by_idx[original_idx] = corrected
    return [adjusted_by_idx[idx] for idx in range(len(cells))]


def ci_excludes_zero(result: BootstrapResult, *, expected_direction: str = "positive") -> bool:
    if not result.valid:
        return False
    if expected_direction == "positive":
        return result.low > 0.0
    if expected_direction == "negative":
        return result.high < 0.0
    raise ValueError("expected_direction must be 'positive' or 'negative'")


def _directional_p_value(estimates: np.ndarray, *, expected_direction: str) -> float:
    if expected_direction == "positive":
        return float((np.sum(estimates <= 0.0) + 1.0) / (len(estimates) + 1.0))
    if expected_direction == "negative":
        return float((np.sum(estimates >= 0.0) + 1.0) / (len(estimates) + 1.0))
    raise ValueError("expected_direction must be 'positive' or 'negative'")
