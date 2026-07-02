from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class BootstrapResult:
    estimate: float
    ci_low: float
    ci_high: float
    p_value: float
    samples: int
    alpha: float
    expected_direction: str
    valid: bool
    reason: str | None = None


@dataclass(frozen=True)
class FDRResult:
    metric_id: str
    p_value: float
    q_value: float
    rejected: bool


def auc_score(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores).astype(float)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError("AUC requires at least one positive and one negative example")
    combined = np.concatenate([pos, neg])
    order = np.argsort(combined, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(combined) + 1, dtype=float)
    sorted_scores = combined[order]
    start = 0
    while start < len(sorted_scores):
        end = start + 1
        while end < len(sorted_scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        if end - start > 1:
            ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    pos_ranks = ranks[: len(pos)]
    return float((pos_ranks.sum() - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def bootstrap_metric(
    values: np.ndarray,
    group_ids: list[str],
    statistic,
    *,
    samples: int,
    seed: int,
    alpha: float = 0.05,
    expected_direction: str = "positive",
    null_value: float = 0.0,
) -> BootstrapResult:
    if samples <= 0:
        return BootstrapResult(
            estimate=float(statistic(values)),
            ci_low=float("nan"),
            ci_high=float("nan"),
            p_value=float("nan"),
            samples=samples,
            alpha=alpha,
            expected_direction=expected_direction,
            valid=False,
            reason="bootstrap samples must be positive",
        )
    values = np.asarray(values)
    groups = np.asarray(group_ids)
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        return BootstrapResult(
            estimate=float(statistic(values)),
            ci_low=float("nan"),
            ci_high=float("nan"),
            p_value=float("nan"),
            samples=samples,
            alpha=alpha,
            expected_direction=expected_direction,
            valid=False,
            reason="bootstrap requires at least two groups",
        )
    rng = np.random.default_rng(seed)
    group_to_indices = {group: np.flatnonzero(groups == group) for group in unique_groups}
    estimates = []
    for _ in range(samples):
        drawn = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        indices = np.concatenate([group_to_indices[group] for group in drawn])
        estimates.append(float(statistic(values[indices])))
    estimates_array = np.asarray(estimates, dtype=float)
    estimate = float(statistic(values))
    ci_low = float(np.quantile(estimates_array, alpha / 2.0))
    ci_high = float(np.quantile(estimates_array, 1.0 - alpha / 2.0))
    if expected_direction == "positive":
        p_value = float((np.sum(estimates_array <= null_value) + 1) / (len(estimates_array) + 1))
    elif expected_direction == "negative":
        p_value = float((np.sum(estimates_array >= null_value) + 1) / (len(estimates_array) + 1))
    else:
        p_value = float((np.sum(np.abs(estimates_array - null_value) <= abs(estimate - null_value)) + 1) / (len(estimates_array) + 1))
    return BootstrapResult(
        estimate=estimate,
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=p_value,
        samples=samples,
        alpha=alpha,
        expected_direction=expected_direction,
        valid=True,
    )


def bootstrap_mean_difference(
    left: np.ndarray,
    right: np.ndarray,
    group_ids: list[str],
    *,
    samples: int,
    seed: int,
    alpha: float = 0.05,
    expected_direction: str = "positive",
) -> BootstrapResult:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.shape != right.shape:
        raise ValueError("bootstrap paired mean difference requires matching shapes")
    values = left - right
    return bootstrap_metric(
        values,
        group_ids,
        lambda value: float(np.mean(value)),
        samples=samples,
        seed=seed,
        alpha=alpha,
        expected_direction=expected_direction,
        null_value=0.0,
    )


def fdr_bh(p_values: dict[str, float], *, alpha: float) -> dict[str, FDRResult]:
    usable = [(metric_id, float(p)) for metric_id, p in p_values.items() if np.isfinite(p)]
    unusable = [metric_id for metric_id, p in p_values.items() if not np.isfinite(p)]
    if not usable:
        return {
            metric_id: FDRResult(metric_id=metric_id, p_value=float("nan"), q_value=float("nan"), rejected=False)
            for metric_id in p_values
        }
    usable.sort(key=lambda item: item[1])
    m = len(usable)
    raw_q: dict[str, float] = {}
    for rank, (metric_id, p_value) in enumerate(usable, start=1):
        raw_q[metric_id] = min(1.0, p_value * m / rank)
    adjusted: dict[str, float] = {}
    running = 1.0
    for metric_id, _ in reversed(usable):
        running = min(running, raw_q[metric_id])
        adjusted[metric_id] = running
    result = {
        metric_id: FDRResult(
            metric_id=metric_id,
            p_value=p_value,
            q_value=adjusted[metric_id],
            rejected=adjusted[metric_id] <= alpha,
        )
        for metric_id, p_value in usable
    }
    for metric_id in unusable:
        result[metric_id] = FDRResult(metric_id=metric_id, p_value=float("nan"), q_value=float("nan"), rejected=False)
    return result


def ci_excludes_null(result: BootstrapResult, *, null_value: float = 0.0) -> bool:
    if not result.valid:
        return False
    if result.expected_direction == "positive":
        return result.ci_low > null_value
    if result.expected_direction == "negative":
        return result.ci_high < null_value
    return result.ci_low > null_value or result.ci_high < null_value


def finite_p_values(results: Iterable[tuple[str, BootstrapResult]]) -> dict[str, float]:
    return {metric_id: result.p_value for metric_id, result in results if result.valid}
