"""Baselines to compare the decoding-free estimator against.

Two baselines, matching the paper:

* ``canonical_log_prob`` scores only the canonical tokenization. It is fast and
  is a (very loose) lower bound on the marginal.

* Importance sampling with a constrained proxy. The proxy walks the lattice
  autoregressively: at each step it masks the model's distribution down to the
  tokens that keep the prefix a valid tokenization of the string, renormalizes,
  and samples. This *requires generation* (one forward pass per step), which is
  the cost the decoding-free method avoids.

A useful simplification: for the constrained proxy q, the importance weight of a
sample collapses to the product of the per-step renormalizers. If Z_i is the
model's probability mass on the allowed tokens at step i, then

    p(t)/q(t) = prod_i P(t_i | t_<i) / prod_i ( P(t_i | t_<i) / Z_i ) = prod_i Z_i

so the marginal estimate is the sample mean of prod_i Z_i. This is exact (an
unbiased estimator of the marginal), which the tests confirm, and it is also
what lets us reproduce the paper's finding that this estimator, at small sample
counts, systematically lands below the lattice estimate.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Callable

from lattice import Lattice

Tokenization = tuple[int, ...]

# cond_logprob(prefix, candidate_ids) -> log P_theta(candidate | prefix) for each candidate.
CondLogProb = Callable[[tuple[int, ...], list[int]], list[float]]


def canonical_log_prob(lattice: Lattice, score_fn) -> float:
    """Log-probability of the canonical tokenization only."""
    return score_fn([lattice.canonical])[0]


@dataclass
class ISResult:
    """Importance-sampling estimate and the bookkeeping to compare cost."""

    log_marginal: float
    log_weights: list[float]
    n_samples: int
    n_forward_passes: int

    @property
    def marginal(self) -> float:
        return math.exp(self.log_marginal)


def importance_sample(
    lattice: Lattice,
    cond_logprob: CondLogProb,
    n_samples: int,
    rng: random.Random | None = None,
) -> ISResult:
    """Estimate the marginal by constrained-proxy importance sampling.

    ``cond_logprob(prefix, candidates)`` must return the model's log-probability
    of each candidate token given the prefix. The number of forward passes is
    tracked to make the runtime comparison against the lattice method honest:
    every step is one call, i.e. generation cost.
    """
    rng = rng or random.Random()
    log_weights: list[float] = []
    n_forward = 0

    for _ in range(n_samples):
        prefix: list[int] = []
        pos = 0
        log_w = 0.0
        while pos != lattice.n:
            edges = lattice.out[pos]
            candidates = [e.token_id for e in edges]
            cand_lp = cond_logprob(tuple(prefix), candidates)
            n_forward += 1

            # Z_i = mass on allowed tokens; renormalize to sample from the proxy.
            log_z = _logsumexp(cand_lp)
            probs = [math.exp(lp - log_z) for lp in cand_lp]
            choice = _sample_index(probs, rng)

            log_w += log_z  # accumulate the importance weight prod_i Z_i
            edge = edges[choice]
            prefix.append(edge.token_id)
            pos = edge.end

        log_weights.append(log_w)

    # Estimate = mean of weights; in log space: logsumexp(log_w) - log(N).
    log_marginal = _logsumexp(log_weights) - math.log(n_samples)
    return ISResult(
        log_marginal=log_marginal,
        log_weights=log_weights,
        n_samples=n_samples,
        n_forward_passes=n_forward,
    )


class HFConditional:
    """``cond_logprob`` backed by a Hugging Face model, with a KV cache.

    The lattice walk always extends the prefix by exactly one token, so this
    keeps the past key/values around and does a single-token incremental forward
    each step, giving O(n) forward passes per sample. It resets automatically
    whenever a fresh sample starts the prefix over at length zero.
    """

    def __init__(self, model, tokenizer, add_bos: bool = True, context_ids=None):
        import torch

        self.torch = torch
        self.model = model
        self.device = next(model.parameters()).device
        bos = tokenizer.bos_token_id
        bos_id = bos if (add_bos and bos is not None) else None
        # Fixed tokens seen before the first answer token: BOS then the prompt.
        self.prefix_base: list[int] = []
        if bos_id is not None:
            self.prefix_base.append(bos_id)
        if context_ids:
            self.prefix_base.extend(context_ids)
        self._cached_prefix: tuple[int, ...] | None = None
        self._past = None
        self._last_logprobs = None

    def __call__(self, prefix: tuple[int, ...], candidates: list[int]) -> list[float]:
        torch = self.torch
        with torch.no_grad():
            if prefix != self._cached_prefix:
                self._advance(prefix)
        lp = self._last_logprobs
        return [float(lp[c]) for c in candidates]

    def _advance(self, prefix: tuple[int, ...]):
        torch = self.torch
        # Fresh sample, or a prefix that does not extend the cache: recompute.
        extends = (
            self._cached_prefix is not None
            and len(prefix) == len(self._cached_prefix) + 1
            and prefix[:-1] == self._cached_prefix
        )
        if not extends:
            self._past = None
            # Fresh sample: prime the cache with BOS + prompt + any answer prefix.
            step_ids = self.prefix_base + list(prefix)
        else:
            step_ids = [prefix[-1]]

        input_ids = torch.tensor([step_ids], dtype=torch.long, device=self.device)
        out = self.model(input_ids=input_ids, past_key_values=self._past, use_cache=True)
        self._past = out.past_key_values
        logits = out.logits[0, -1, :].float()
        self._last_logprobs = torch.log_softmax(logits, dim=-1)
        self._cached_prefix = prefix


def _logsumexp(values: Sequence[float]) -> float:
    finite = [v for v in values if v != -math.inf]
    if not finite:
        return -math.inf
    m = max(finite)
    return m + math.log(sum(math.exp(v - m) for v in finite))


def _sample_index(probs: Sequence[float], rng: random.Random) -> int:
    r = rng.random()
    acc = 0.0
    for i, p in enumerate(probs):
        acc += p
        if r <= acc:
            return i
    return len(probs) - 1
