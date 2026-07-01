"""Convergence tests for the marginal estimator.

The key claim: the estimator is a strict lower bound on the true marginal that
increases monotonically toward it as more tokenizations are scored. We verify
that against exact enumeration on short strings.
"""

from __future__ import annotations

import math
import random

from adapters import ToyTokenizerAdapter
from lattice import Lattice
from marginal import estimate_marginal, exact_marginal, logsumexp
from sampling import off_by_one, uniform_sample
from tests.toy import TOY_VOCAB, ToyBigramModel


def build(text: str) -> Lattice:
    return Lattice.build(text, ToyTokenizerAdapter(TOY_VOCAB))


MODEL = ToyBigramModel(vocab_size=len(TOY_VOCAB), seed=7)
ID_TO_PIECE = {i: s for s, i in TOY_VOCAB.items()}


def test_exact_marginal_matches_manual_sum():
    lat = build("aaaa")
    score = MODEL.score_fn()
    manual = logsumexp([MODEL.log_prob(p) for p in lat.iter_paths()])
    assert math.isclose(exact_marginal(lat, score), manual, rel_tol=1e-12)


def test_estimate_with_full_budget_equals_exact():
    for text in ["aaaa", "abab", "aabb"]:
        lat = build(text)
        score = MODEL.score_fn()
        k = lat.num_tokenizations()  # score everything
        est = estimate_marginal(lat, ID_TO_PIECE, score, k=k, rng=random.Random(1))
        assert math.isclose(est.log_marginal, exact_marginal(lat, score), rel_tol=1e-9)
        assert est.num_unique == k


def test_lower_bound_and_monotone_convergence():
    lat = build("aaaa")
    score = MODEL.score_fn()
    truth = exact_marginal(lat, score)

    prev = -math.inf
    total = lat.num_tokenizations()
    for k in range(1, total + 1):
        est = estimate_marginal(lat, ID_TO_PIECE, score, k=k, rng=random.Random(0))
        # Never exceeds the truth (it is a lower bound).
        assert est.log_marginal <= truth + 1e-12, k
        # The internal curve is non-decreasing.
        for a, b in zip(est.curve, est.curve[1:]):
            assert b >= a - 1e-12
        prev = est.log_marginal
    # With the full budget the bound is tight.
    assert math.isclose(prev, truth, rel_tol=1e-9)


def test_off_by_one_are_valid_and_include_canonical():
    lat = build("aaaa")
    valid = set(lat.iter_paths())
    near = off_by_one(lat, ID_TO_PIECE)
    assert near[0] == lat.canonical
    assert len(near) == len(set(near))  # unique
    for tok in near:
        assert tok in valid


def test_uniform_sample_is_unique_and_within_lattice():
    lat = build("aaaa")
    valid = set(lat.iter_paths())
    samples = uniform_sample(lat, k=100, rng=random.Random(3))
    assert len(samples) == len(set(samples))
    assert all(s in valid for s in samples)
    # Small lattice: without replacement returns every tokenization.
    assert set(samples) == valid


def test_length_constrained_sampling_respects_budget():
    lat = build("aaaa")
    samples = uniform_sample(lat, k=100, max_len=2, rng=random.Random(4))
    assert all(len(s) <= 2 for s in samples)
