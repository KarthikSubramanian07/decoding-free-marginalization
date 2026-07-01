"""Standalone correctness gate.

Runs the two checks the whole project rests on, using a small in-memory
vocabulary and a deterministic scoring function, so it needs no model download
and finishes instantly:

1. The lattice enumerates every tokenization of a string exactly (verified
   against brute-force enumeration).
2. The marginal estimator is a lower bound that converges to the exact marginal
   as the sample budget grows, and its internal curve is monotone.

This is the same guarantee the test suite checks, packaged as one command with a
human-readable report and a nonzero exit code on failure, so it doubles as a CI
smoke test. Run it with ``python validate.py``.
"""

from __future__ import annotations

import math
import random
import sys

from adapters import ToyTokenizerAdapter
from lattice import Lattice
from marginal import estimate_marginal, exact_marginal

VOCAB = {"a": 0, "b": 1, "aa": 2, "ab": 3, "ba": 4, "bb": 5, "aaa": 6, "bbb": 7}
STRINGS = ["aaaa", "abba", "bbbb", "abab", "aabb", "baab", "aaaaa"]
ID_TO_PIECE = {i: s for s, i in VOCAB.items()}


def deterministic_score(tokenizations):
    """A fixed, model-free log-probability for each tokenization.

    Any deterministic assignment works for the convergence check: summing a
    unique subset of probabilities always approaches the sum over all of them.
    """
    scores = []
    for tok in tokenizations:
        s = -sum((tid + 1) * 0.3 for tid in tok) - 0.5 * len(tok)
        scores.append(s)
    return scores


def brute_force_count(text: str) -> int:
    n = len(text)
    ways = [0] * (n + 1)
    ways[0] = 1
    for i in range(n):
        if ways[i] == 0:
            continue
        for j in range(i + 1, n + 1):
            if text[i:j] in VOCAB:
                ways[j] += ways[i]
    return ways[n]


def main() -> int:
    adapter = ToyTokenizerAdapter(VOCAB)
    failures = 0

    print("== check 1: lattice enumerates every tokenization exactly ==")
    for text in STRINGS:
        lat = Lattice.build(text, adapter)
        got = lat.num_tokenizations()
        want = brute_force_count(text)
        enumerated = len(set(lat.iter_paths()))
        ok = got == want == enumerated
        failures += not ok
        print(f"  {text:<7} counted={got:<4} brute_force={want:<4} enumerated={enumerated:<4} "
              f"{'ok' if ok else 'FAIL'}")

    print("\n== check 2: estimator is a monotone lower bound converging to exact ==")
    for text in STRINGS:
        lat = Lattice.build(text, adapter)
        truth = exact_marginal(lat, deterministic_score)
        total = lat.num_tokenizations()

        monotone = True
        below = True
        for k in range(1, total + 1):
            est = estimate_marginal(
                lat, ID_TO_PIECE, deterministic_score, k=k, rng=random.Random(0)
            )
            if any(b < a - 1e-12 for a, b in zip(est.curve, est.curve[1:])):
                monotone = False
            if est.log_marginal > truth + 1e-9:
                below = False
        full = estimate_marginal(
            lat, ID_TO_PIECE, deterministic_score, k=total, rng=random.Random(0)
        )
        converged = math.isclose(full.log_marginal, truth, rel_tol=1e-9)
        ok = monotone and below and converged
        failures += not ok
        e_exact, e_full = math.exp(truth), math.exp(full.log_marginal)
        print(f"  {text:<7} exact={e_exact:.3e} full_budget={e_full:.3e} "
              f"monotone={monotone} lower_bound={below} converged={converged} "
              f"{'ok' if ok else 'FAIL'}")

    print()
    if failures:
        print(f"VALIDATION FAILED: {failures} check(s) did not pass")
        return 1
    print("VALIDATION PASSED: lattice enumeration exact, estimator converges to the marginal")
    return 0


if __name__ == "__main__":
    sys.exit(main())
