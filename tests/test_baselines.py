"""Tests for the importance-sampling baseline.

Two things matter here:

1. The constrained proxy is a proper distribution over the lattice: its
   probabilities over all tokenizations sum to one.
2. The importance-sampling estimator is unbiased: the proxy-weighted mean of the
   importance weights equals the true marginal. This is what makes it a fair
   baseline, and its high variance at small sample counts is what produces the
   paper's underestimation finding.
"""

from __future__ import annotations

import math
import random

from adapters import ToyTokenizerAdapter
from baselines import _logsumexp, importance_sample
from lattice import Lattice
from marginal import exact_marginal
from tests.toy import TOY_VOCAB, ToyBigramModel


def build(text: str) -> Lattice:
    return Lattice.build(text, ToyTokenizerAdapter(TOY_VOCAB))


MODEL = ToyBigramModel(vocab_size=len(TOY_VOCAB), seed=11)


def proxy_stats(lattice: Lattice):
    """Replay the proxy over every path: return (sum of q, proxy-weighted marginal)."""
    q_total = 0.0
    est_total = 0.0
    for path in lattice.iter_paths():
        prefix: list[int] = []
        pos = 0
        log_q = 0.0
        log_w = 0.0
        idx = 0
        while pos != lattice.n:
            edges = lattice.out[pos]
            cands = [e.token_id for e in edges]
            cand_lp = MODEL.cond_logprob(tuple(prefix), cands)
            log_z = _logsumexp(cand_lp)
            chosen_id = path[idx]
            k = cands.index(chosen_id)
            log_q += cand_lp[k] - log_z
            log_w += log_z
            prefix.append(chosen_id)
            pos = edges[k].end
            idx += 1
        q_total += math.exp(log_q)
        est_total += math.exp(log_q) * math.exp(log_w)  # q(t) * weight(t) = p(t)
    return q_total, est_total


def test_proxy_is_a_proper_distribution_over_the_lattice():
    lat = build("aaaa")
    q_total, _ = proxy_stats(lat)
    assert math.isclose(q_total, 1.0, rel_tol=1e-9)


def test_importance_sampling_is_unbiased_in_expectation():
    lat = build("aaaa")
    truth = math.exp(exact_marginal(lat, MODEL.score_fn()))
    _, est = proxy_stats(lat)
    assert math.isclose(est, truth, rel_tol=1e-9)


def test_monte_carlo_estimate_is_close_with_many_samples():
    lat = build("aaaa")
    truth = exact_marginal(lat, MODEL.score_fn())
    res = importance_sample(lat, MODEL.cond_logprob, n_samples=40000, rng=random.Random(0))
    assert math.isclose(res.log_marginal, truth, rel_tol=0.03)
    # Every step is a forward pass: cost scales with samples times path length.
    assert res.n_forward_passes >= res.n_samples


def test_forward_pass_count_reflects_generation_cost():
    lat = build("aaaa")
    res = importance_sample(lat, MODEL.cond_logprob, n_samples=10, rng=random.Random(1))
    # Each sample walks from position 0 to the end taking >= 1 step per token.
    assert res.n_forward_passes >= 10
