from __future__ import annotations

import torch

from attention_lab.mechanisms.synthetic_prompts import build_induction_probe, induction_accuracy


def test_build_induction_probe_repeats_the_pattern_exactly():
    probe = build_induction_probe(vocab_size=64, pattern_len=5, generator=torch.Generator().manual_seed(0))

    assert probe.input_ids.shape == (1, 10)
    first_half = probe.input_ids[0, :5]
    second_half = probe.input_ids[0, 5:]
    assert torch.equal(first_half, second_half)


def test_induction_positions_are_the_second_half_minus_the_last_position():
    probe = build_induction_probe(vocab_size=64, pattern_len=5, generator=torch.Generator().manual_seed(0))

    # second half occupies indices [5, 9]; position 9 has no "next token"
    # within the sequence, so only [5, 6, 7, 8] are scorable induction positions.
    assert probe.induction_positions == [5, 6, 7, 8]


def test_induction_accuracy_with_a_perfect_oracle_is_one():
    torch.manual_seed(0)
    probe = build_induction_probe(vocab_size=64, pattern_len=5, generator=torch.Generator().manual_seed(1))

    seq_len = probe.input_ids.shape[1]
    vocab_size = 64
    logits = torch.full((1, seq_len, vocab_size), -100.0)
    for position in probe.induction_positions:
        correct_next_token = probe.input_ids[0, position + 1]
        logits[0, position, correct_next_token] = 100.0

    accuracy = induction_accuracy(logits, probe)

    assert accuracy == 1.0


def test_induction_accuracy_with_random_predictions_is_near_chance():
    torch.manual_seed(0)
    probe = build_induction_probe(vocab_size=64, pattern_len=20, generator=torch.Generator().manual_seed(2))

    seq_len = probe.input_ids.shape[1]
    vocab_size = 64
    logits = torch.randn(1, seq_len, vocab_size)

    accuracy = induction_accuracy(logits, probe)

    assert 0.0 <= accuracy <= 1.0


def test_different_generators_produce_different_patterns():
    probe_a = build_induction_probe(vocab_size=64, pattern_len=8, generator=torch.Generator().manual_seed(0))
    probe_b = build_induction_probe(vocab_size=64, pattern_len=8, generator=torch.Generator().manual_seed(1))

    assert not torch.equal(probe_a.input_ids, probe_b.input_ids)


def test_same_generator_seed_is_deterministic():
    probe_a = build_induction_probe(vocab_size=64, pattern_len=8, generator=torch.Generator().manual_seed(42))
    probe_b = build_induction_probe(vocab_size=64, pattern_len=8, generator=torch.Generator().manual_seed(42))

    assert torch.equal(probe_a.input_ids, probe_b.input_ids)
