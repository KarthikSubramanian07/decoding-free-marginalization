"""Decoding-free samplers over the tokenization lattice.

None of these functions call a language model. They walk the lattice cheaply to
propose tokenizations, which are then scored elsewhere. Two strategies are
implemented, matching the paper:

* ``off_by_one`` enumerates the near-canonical tokenizations: those that differ
  from the canonical one by splitting exactly one token into two. The paper's
  central empirical observation is that most of the non-canonical probability
  mass lives here, so this small set captures the bulk of the marginal cheaply.

* ``uniform_sample`` draws tokenizations uniformly at random from the lattice,
  optionally length-constrained and optionally without replacement. Uniform
  sampling covers the long tail that off-by-one misses.

Both return tokenizations as tuples of integer token ids, which is what the
scorer consumes and what makes deduplication trivial.
"""

from __future__ import annotations

import random
from collections.abc import Iterable

from lattice import Lattice

Tokenization = tuple[int, ...]


def off_by_one(
    lattice: Lattice,
    id_to_piece: dict[int, str],
    include_canonical: bool = True,
) -> list[Tokenization]:
    """Enumerate near-canonical tokenizations.

    Starting from the canonical tokenization ``t_1 ... t_n``, for each position
    ``i`` we split ``t_i`` at every internal character into ``(a, b)`` and keep
    the split when both halves are in the vocabulary. Each surviving split
    yields one tokenization identical to the canonical one except that ``t_i``
    is replaced by two tokens.

    Runtime is O(n * m) where m is the longest token length, and every result is
    a valid path in the lattice because the split halves tile the same span.

    Returns unique tokenizations. The canonical tokenization is included first
    when ``include_canonical`` is set.
    """
    vocab = {p: i for i, p in id_to_piece.items()}
    canonical = lattice.canonical

    seen: set[Tokenization] = set()
    out: list[Tokenization] = []

    def add(tok: Tokenization) -> None:
        if tok not in seen:
            seen.add(tok)
            out.append(tok)

    if include_canonical:
        add(canonical)

    for i, tid in enumerate(canonical):
        piece = id_to_piece[tid]
        for k in range(1, len(piece)):
            a, b = piece[:k], piece[k:]
            ia, ib = vocab.get(a), vocab.get(b)
            if ia is not None and ib is not None:
                add(canonical[:i] + (ia, ib) + canonical[i + 1 :])

    return out


def uniform_sample(
    lattice: Lattice,
    k: int,
    max_len: int | None = None,
    rng: random.Random | None = None,
    exclude: Iterable[Tokenization] | None = None,
    enumerate_threshold: int = 4096,
    max_attempts_factor: int = 64,
) -> list[Tokenization]:
    """Draw up to ``k`` distinct tokenizations uniformly from the lattice.

    Sampling is without replacement: the returned tokenizations are unique and
    disjoint from ``exclude``. Uniformity is over lattice *paths*, using the
    path-count DP to map a uniform integer to the corresponding path.

    Two regimes:

    * When the lattice has few enough tokenizations
      (``num_tokenizations <= enumerate_threshold``), all paths are materialized,
      shuffled with ``rng``, and the first ``k`` not in ``exclude`` are returned.
      This is exact and, for tiny strings, returns every tokenization.
    * Otherwise, integers are drawn uniformly in ``[1, N]`` and mapped to paths
      by rejection, deduplicating against previously drawn paths and
      ``exclude``. Attempts are capped so an over-large ``k`` cannot loop
      forever.

    ``max_len`` restricts to tokenizations of at most that many tokens, which
    the paper uses to avoid wasting samples on very long, low-probability
    tokenizations.
    """
    if k <= 0:
        return []
    rng = rng or random.Random()
    exclude_set: set[Tokenization] = set(exclude or ())

    counts = lattice.path_counts(max_len)
    r_max = lattice.n if max_len is None else min(max_len, lattice.n)
    total: int = counts[0][r_max]
    if total == 0:
        return []

    if total <= enumerate_threshold:
        all_paths = [lattice.nth_path(z, counts, max_len) for z in range(1, total + 1)]
        rng.shuffle(all_paths)
        result: list[Tokenization] = []
        for tok in all_paths:
            if tok in exclude_set:
                continue
            result.append(tok)
            if len(result) >= k:
                break
        return result

    # Large lattice: rejection sample distinct integers.
    result = []
    seen: set[Tokenization] = set(exclude_set)
    attempts = 0
    attempt_cap = max_attempts_factor * k
    while len(result) < k and attempts < attempt_cap:
        attempts += 1
        z = rng.randint(1, total)  # supports arbitrary-precision ints
        tok = lattice.nth_path(z, counts, max_len)
        if tok in seen:
            continue
        seen.add(tok)
        result.append(tok)
    return result
