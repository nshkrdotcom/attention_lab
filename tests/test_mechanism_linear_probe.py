from __future__ import annotations

import numpy as np
import pytest
import torch

from attention_lab.mechanisms.activations import tensor_to_feature_matrix
from attention_lab.mechanisms.linear_probe import (
    grouped_train_test_split,
    train_linear_probe,
    train_shuffled_label_null,
)
from attention_lab.mechanisms.statistics import auc_score, bootstrap_mean_difference, fdr_bh
from attention_lab.mechanisms.task_schema import TaskExample


def _toy_probe_data(n_pairs: int = 24) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    rng = np.random.default_rng(0)
    features = []
    labels = []
    pair_ids = []
    template_ids = []
    for pair in range(n_pairs):
        template = f"template_{pair % 6}"
        for label in (1, 0):
            center = 2.0 if label else -2.0
            features.append(rng.normal(center, 0.15, size=4))
            labels.append(label)
            pair_ids.append(f"pair_{pair}")
            template_ids.append(template)
    return np.asarray(features, dtype=np.float32), np.asarray(labels), pair_ids, template_ids


def test_real_linear_probe_training_emits_auc_and_shuffled_null():
    features, labels, pair_ids, template_ids = _toy_probe_data()
    split = grouped_train_test_split(pair_ids=pair_ids, template_ids=template_ids, seed=1)
    result = train_linear_probe(features, labels, pair_ids=pair_ids, template_ids=template_ids, split=split, seed=1)
    shuffled = train_shuffled_label_null(
        features,
        labels,
        pair_ids=pair_ids,
        template_ids=template_ids,
        split=split,
        seed=2,
    )

    assert result.auc > 0.95
    assert "linear_probe_auc" in result.to_metrics()
    assert shuffled.auc <= result.auc


def test_grouped_split_keeps_pairs_and_templates_together():
    _, _, pair_ids, template_ids = _toy_probe_data()
    pair_split = grouped_train_test_split(
        pair_ids=pair_ids,
        template_ids=template_ids,
        seed=3,
        group_by_template=False,
    )
    for pair_id in set(pair_ids):
        indices = [index for index, value in enumerate(pair_ids) if value == pair_id]
        in_train = [index in set(pair_split.train_indices) for index in indices]
        assert all(in_train) or not any(in_train)

    template_split = grouped_train_test_split(
        pair_ids=pair_ids,
        template_ids=template_ids,
        seed=3,
        group_by_template=True,
    )
    assert set(template_split.train_groups).isdisjoint(set(template_split.test_groups))


def test_auc_bootstrap_and_fdr_bh_behavior():
    labels = np.asarray([1, 1, 0, 0])
    assert auc_score(labels, np.asarray([0.9, 0.8, 0.1, 0.2])) == 1.0
    ci = bootstrap_mean_difference(
        np.asarray([1.0, 1.2, 0.9, 1.1]),
        np.asarray([0.0, 0.1, 0.2, 0.0]),
        ["a", "b", "c", "d"],
        samples=100,
        seed=0,
    )
    assert ci.valid
    assert ci.ci_low > 0

    fdr = fdr_bh({"primary": 0.001, "secondary": 0.04, "weak": 0.5}, alpha=0.05)
    assert fdr["primary"].rejected
    assert not fdr["weak"].rejected


def test_answer_position_pooling_uses_declared_position():
    tensor = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    examples = [
        TaskExample("a", 1, "pos", "p0", "t0", "f", {"clean_answer_position": 1}),
        TaskExample("b", 0, "neg", "p0", "t0", "f", {"corrupted_answer_position": 2}),
    ]

    pooled = tensor_to_feature_matrix(
        tensor,
        expected_batch=2,
        lengths=[4, 4],
        examples=examples,
        feature_pooling="answer_position",
    )

    assert np.allclose(pooled[0], tensor[0, 1].numpy())
    assert np.allclose(pooled[1], tensor[1, 2].numpy())


def test_patch_position_pooling_uses_declared_aligned_positions_only():
    tensor = torch.arange(2 * 5 * 2, dtype=torch.float32).reshape(2, 5, 2)
    examples = [
        TaskExample("a", 1, "pos", "p0", "t0", "f", {"clean_patch_token_indices": [1, 3]}),
        TaskExample("b", 0, "neg", "p0", "t0", "f", {"corrupted_patch_token_indices": [2, 4]}),
    ]

    pooled = tensor_to_feature_matrix(
        tensor,
        expected_batch=2,
        lengths=[5, 5],
        examples=examples,
        feature_pooling="patch_positions_mean",
    )

    assert np.allclose(pooled[0], tensor[0, [1, 3]].mean(dim=0).numpy())
    assert np.allclose(pooled[1], tensor[1, [2, 4]].mean(dim=0).numpy())


def test_invalid_pooling_metadata_blocks_task_aligned_pooling():
    tensor = torch.zeros(1, 3, 2)
    examples = [TaskExample("a", 1, "pos", "p0", "t0", "f", {"clean_answer_position": 99})]

    with pytest.raises(ValueError, match="out of range"):
        tensor_to_feature_matrix(
            tensor,
            expected_batch=1,
            lengths=[3],
            examples=examples,
            feature_pooling="answer_position",
        )
