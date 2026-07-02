from __future__ import annotations

import numpy as np

from attention_lab.mechanisms.linear_probe import (
    LinearProbeDataset,
    grouped_train_test_split,
    run_linear_probe_with_nulls,
)
from attention_lab.mechanisms.statistics import bootstrap_ci, fdr_bh, target_vs_decoy_specificity


def _toy_probe_dataset(n_pairs: int = 40) -> LinearProbeDataset:
    rows = []
    labels = []
    pair_ids = []
    template_ids = []
    family_ids = []
    variants = []
    rng = np.random.default_rng(7)
    for pair in range(n_pairs):
        template = f"template_{pair // 4}"
        for variant, label, shift in (
            ("x_pos", 1, 1.5),
            ("x_para", 1, 1.2),
            ("x_neg", 0, -1.2),
            ("x_decoy", 0, -0.2),
        ):
            rows.append(rng.normal(size=6) + np.array([shift, shift, 0, 0, 0, 0]))
            labels.append(label)
            pair_ids.append(f"pair_{pair}")
            template_ids.append(template)
            family_ids.append("negation")
            variants.append(variant)
    return LinearProbeDataset(
        X=np.asarray(rows, dtype=np.float32),
        y=np.asarray(labels, dtype=np.int64),
        pair_ids=np.asarray(pair_ids),
        template_ids=np.asarray(template_ids),
        family_ids=np.asarray(family_ids),
        variants=np.asarray(variants),
    )


def test_grouped_split_keeps_pair_variants_together():
    dataset = _toy_probe_dataset()
    split = grouped_train_test_split(dataset, seed=3, test_fraction=0.3)

    train_pairs = set(dataset.pair_ids[split.train_indices])
    test_pairs = set(dataset.pair_ids[split.test_indices])

    assert train_pairs.isdisjoint(test_pairs)
    assert len(train_pairs) > 0
    assert len(test_pairs) > 0


def test_grouped_split_can_prevent_template_leakage():
    dataset = _toy_probe_dataset()
    split = grouped_train_test_split(dataset, seed=5, test_fraction=0.3, group_by_template=True)

    train_templates = set(dataset.template_ids[split.train_indices])
    test_templates = set(dataset.template_ids[split.test_indices])

    assert train_templates.isdisjoint(test_templates)


def test_real_linear_probe_training_emits_auc_and_shuffled_null():
    dataset = _toy_probe_dataset()
    result = run_linear_probe_with_nulls(dataset, seed=11, min_n=20, training_steps=120)

    assert result.primary.auc > 0.9
    assert result.shuffled.auc is not None
    assert result.auc_minus_shuffled_auc is not None
    assert result.auc_minus_shuffled_auc > 0.2
    assert result.primary.weights.shape == (dataset.X.shape[1],)
    assert result.grouped_split.pair_group_leakage is False


def test_bootstrap_ci_and_fdr_bh_cover_all_test_cells():
    values = np.array([0.4, 0.3, 0.2, 0.5, 0.6])
    ci = bootstrap_ci(values, seed=1, samples=200)
    assert ci.estimate > 0
    assert ci.low > 0

    cells = [
        {"cell_id": "site_a|primary_auc", "p_value": 0.001},
        {"cell_id": "site_a|matched_control", "p_value": 0.02},
        {"cell_id": "site_b|target_vs_decoy", "p_value": 0.20},
    ]
    corrected = fdr_bh(cells, alpha=0.05)
    assert {row["cell_id"] for row in corrected} == {row["cell_id"] for row in cells}
    assert corrected[0]["rejected"] is True
    assert corrected[-1]["rejected"] is False


def test_target_vs_decoy_specificity_uses_pair_bootstrap():
    dataset = _toy_probe_dataset(n_pairs=30)
    scores = np.where(dataset.variants == "x_decoy", 0.15, dataset.y.astype(float))
    result = target_vs_decoy_specificity(dataset, scores, seed=2, samples=200)

    assert result.estimate > 0
    assert result.low > 0
    assert result.p_value < 0.05
