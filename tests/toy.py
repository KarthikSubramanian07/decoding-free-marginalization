"""Shared toy fixtures for the correctness tests.

A tiny in-memory vocabulary plus a deterministic bigram "model" over token ids.
This lets the tests verify the lattice, sampler, estimator, and baselines
exactly, with no model download and no floating-point drift beyond the model's
own arithmetic.
"""

from __future__ import annotations

import math
import random

# Surface vocabulary over the alphabet {a, b}. Ids are arbitrary but distinct.
TOY_VOCAB: dict[str, int] = {
    "a": 0,
    "b": 1,
    "aa": 2,
    "ab": 3,
    "ba": 4,
    "bb": 5,
    "aaa": 6,
    "bbb": 7,
}


class ToyBigramModel:
    """A fixed autoregressive model over token ids.

    Probability of a tokenization is prod_i P(t_i | t_{i-1}) with a dedicated
    start state. Transition rows are seeded so the model is deterministic across
    runs. This is a genuine autoregressive distribution over id sequences, which
    is what makes it a fair stand-in for an LLM when checking that summed
    tokenization probabilities behave correctly.
    """

    def __init__(self, vocab_size: int, seed: int = 0):
        self.vocab_size = vocab_size
        rng = random.Random(seed)
        # rows indexed by previous id (or vocab_size for the start state).
        self._log_trans: list[list[float]] = []
        for _ in range(vocab_size + 1):
            logits = [rng.gauss(0.0, 1.0) for _ in range(vocab_size)]
            m = max(logits)
            denom = math.log(sum(math.exp(x - m) for x in logits)) + m
            self._log_trans.append([x - denom for x in logits])

    def log_prob(self, tokenization: tuple[int, ...]) -> float:
        prev = self.vocab_size  # start state
        total = 0.0
        for tid in tokenization:
            total += self._log_trans[prev][tid]
            prev = tid
        return total

    def score_fn(self):
        def _score(toks):
            return [self.log_prob(t) for t in toks]

        return _score

    def cond_logprob(self, prefix: tuple[int, ...], candidates: list[int]) -> list[float]:
        """log P(candidate | prefix) for each candidate, matching ``log_prob``."""
        prev = prefix[-1] if prefix else self.vocab_size
        return [self._log_trans[prev][c] for c in candidates]
