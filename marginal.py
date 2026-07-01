"""Decoding-free marginal estimator.

The marginal probability of a string is the sum of the probabilities of all its
tokenizations:

    p(s) = sum over t in tokenizations(s) of  prod_i P_theta(t_i | t_<i)

Computing it exactly is intractable because the number of tokenizations grows
exponentially. This module estimates it by scoring a chosen subset of unique
tokenizations and summing their probabilities. Because every tokenization has
non-negative probability and we only ever sum *distinct* ones, the estimate is a
strict lower bound on the true marginal that increases monotonically as more
tokenizations are added. That is the property the paper relies on.

The estimator is deliberately independent of any particular model: it takes a
``score_fn`` mapping a list of tokenizations to their log-probabilities. The
Hugging Face implementation lives in ``scoring.py``; the tests pass a toy
``score_fn`` so they can verify correctness without loading a model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

from lattice import Lattice
from sampling import off_by_one, uniform_sample
import random

Tokenization = tuple[int, ...]
ScoreFn = Callable[[Sequence[Tokenization]], list[float]]


def logsumexp(values: Sequence[float]) -> float:
    """Numerically stable log(sum(exp(v))). Returns -inf for an empty input."""
    finite = [v for v in values if v != -math.inf]
    if not finite:
        return -math.inf
    m = max(finite)
    return m + math.log(sum(math.exp(v - m) for v in finite))


@dataclass
class MarginalEstimate:
    """Result of a marginal estimation run.

    ``log_marginal`` is the log of the estimated (lower-bound) marginal.
    ``curve`` is the cumulative log-marginal after each unique tokenization is
    added, in the order they were collected (near-canonical first, then uniform
    samples). It is non-decreasing by construction.
    """

    log_marginal: float
    tokenizations: list[Tokenization]
    log_probs: list[float]
    curve: list[float]
    num_near_canonical: int
    canonical_log_prob: float

    @property
    def marginal(self) -> float:
        return math.exp(self.log_marginal)

    @property
    def num_unique(self) -> int:
        return len(self.tokenizations)


def _cumulative_curve(log_probs: Sequence[float]) -> list[float]:
    curve: list[float] = []
    running = -math.inf
    for lp in log_probs:
        # running = logsumexp(running, lp)
        if running == -math.inf:
            running = lp
        elif lp == -math.inf:
            pass
        else:
            m = max(running, lp)
            running = m + math.log(math.exp(running - m) + math.exp(lp - m))
        curve.append(running)
    return curve


def estimate_marginal(
    lattice: Lattice,
    id_to_piece: dict[int, str],
    score_fn: ScoreFn,
    k: int,
    max_len: int | None = None,
    rng: random.Random | None = None,
    use_off_by_one: bool = True,
) -> MarginalEstimate:
    """Estimate the marginal of the string described by ``lattice``.

    Collects up to ``k`` unique tokenizations: first the near-canonical
    (off-by-one) set when ``use_off_by_one`` is set, then uniform samples to
    fill the remaining budget, then scores them all and sums.
    """
    tokenizations: list[Tokenization] = []
    seen: set[Tokenization] = set()

    if use_off_by_one:
        near = off_by_one(lattice, id_to_piece, include_canonical=True)
    else:
        near = [lattice.canonical]
    for tok in near:
        if tok not in seen:
            seen.add(tok)
            tokenizations.append(tok)
    num_near = len(tokenizations)

    remaining = k - len(tokenizations)
    if remaining > 0:
        sampled = uniform_sample(
            lattice, remaining, max_len=max_len, rng=rng, exclude=seen
        )
        for tok in sampled:
            if tok not in seen:
                seen.add(tok)
                tokenizations.append(tok)

    log_probs = list(score_fn(tokenizations))
    if len(log_probs) != len(tokenizations):
        raise ValueError("score_fn returned the wrong number of log-probs")

    curve = _cumulative_curve(log_probs)
    log_marginal = curve[-1] if curve else -math.inf

    canonical_lp = -math.inf
    for tok, lp in zip(tokenizations, log_probs):
        if tok == lattice.canonical:
            canonical_lp = lp
            break

    return MarginalEstimate(
        log_marginal=log_marginal,
        tokenizations=tokenizations,
        log_probs=log_probs,
        curve=curve,
        num_near_canonical=num_near,
        canonical_log_prob=canonical_lp,
    )


def exact_marginal(lattice: Lattice, score_fn: ScoreFn, max_len: int | None = None) -> float:
    """Exact log-marginal by enumerating every tokenization.

    Only tractable for short strings; used by the correctness tests as ground
    truth for the estimator's lower bound to converge toward.
    """
    all_paths = list(lattice.iter_paths(max_len=max_len))
    log_probs = score_fn(all_paths)
    return logsumexp(log_probs)
