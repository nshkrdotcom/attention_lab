from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class InductionProbe:
    """A synthetic repeated-random-token sequence for the classic induction-head
    behavioral test (Olsson et al. 2022): random tokens, then the exact same
    tokens repeated. No tokenizer or real text involved -- the model's own
    vocab_size is the only real-world quantity that matters.
    """

    input_ids: torch.Tensor
    induction_positions: list[int]


def build_induction_probe(vocab_size: int, pattern_len: int, generator: torch.Generator) -> InductionProbe:
    if pattern_len < 2:
        raise ValueError("pattern_len must be at least 2 to have any scorable induction positions")

    pattern = torch.randint(0, vocab_size, (pattern_len,), generator=generator)
    input_ids = torch.cat([pattern, pattern]).unsqueeze(0)

    # Second half occupies indices [pattern_len, 2*pattern_len - 1]. The last
    # position has no "next token" within this sequence to score against, so
    # only [pattern_len, 2*pattern_len - 2] are scorable.
    induction_positions = list(range(pattern_len, 2 * pattern_len - 1))

    return InductionProbe(input_ids=input_ids, induction_positions=induction_positions)


def induction_accuracy(logits: torch.Tensor, probe: InductionProbe) -> float:
    predictions = logits[0].argmax(dim=-1)
    correct = 0
    for position in probe.induction_positions:
        predicted_token = predictions[position]
        actual_next_token = probe.input_ids[0, position + 1]
        if predicted_token == actual_next_token:
            correct += 1
    return correct / len(probe.induction_positions)
