"""Tokenization lattice.

A single piece of text can be tokenized many ways by a subword vocabulary. The
set of all valid tokenizations forms a directed acyclic graph (a lattice): nodes
are character positions in the (normalized) string, and an edge i -> j exists
whenever the substring between those positions is a token in the vocabulary. The
paper builds this via finite-state transducer composition; the equivalent and
much simpler construction used here is dynamic programming over positions.

Everything in this module is pure Python. It has no model or heavy dependency,
which is what lets the correctness tests enumerate lattices exactly.

Surface space
-------------
Subword vocabularies do not store raw text. SentencePiece marks spaces with the
meta symbol U+2581, byte-level BPE remaps bytes to printable code points, and so
on. Rather than reimplement each convention, we operate in whatever "surface"
space the tokenizer's own vocabulary lives in. A `TokenizerAdapter` is
responsible for two things that must agree with each other:

  * `canonical_pieces(text)` returns the tokenizer's default tokenization as a
    list of surface strings, and
  * `vocab()` maps every surface string to its integer id.

We then define the normalized string as the concatenation of the canonical
pieces. Because each canonical piece is a vocabulary entry and they tile the
normalized string exactly, the canonical tokenization is always a path in the
lattice, and every edge we add corresponds to a real token id.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Iterator, Protocol, Sequence


class TokenizerAdapter(Protocol):
    """Minimal surface a tokenizer must expose to build a lattice."""

    def canonical_pieces(self, text: str) -> list[str]:
        """The tokenizer's default tokenization, as surface strings."""
        ...

    def vocab(self) -> dict[str, int]:
        """Map from surface string to integer token id."""
        ...


@dataclass(frozen=True)
class Edge:
    """A lattice edge covering ``normalized[start:end]`` with one token."""

    start: int
    end: int
    piece: str
    token_id: int


@dataclass
class Lattice:
    """All tokenizations of a single string as a DAG over character positions.

    Attributes
    ----------
    normalized:
        The surface string the lattice spans (concatenation of canonical pieces).
    out:
        ``out[i]`` is the list of edges leaving position ``i``.
    canonical:
        The canonical tokenization as a tuple of token ids. Always a valid path.
    """

    normalized: str
    out: list[list[Edge]]
    canonical: tuple[int, ...]

    @property
    def n(self) -> int:
        """Number of characters in the normalized string (the sink node index)."""
        return len(self.normalized)

    # -- construction ---------------------------------------------------------

    @classmethod
    def build(cls, text: str, adapter: TokenizerAdapter, max_token_len: int | None = None) -> "Lattice":
        """Construct the lattice for ``text`` under ``adapter``.

        ``max_token_len`` caps the surface length of tokens we try to match at
        each position; it defaults to the longest token in the vocabulary. This
        is a pure performance knob and does not change the lattice, since every
        vocabulary token is shorter than that bound by definition.
        """
        pieces = adapter.canonical_pieces(text)
        vocab = adapter.vocab()
        normalized = "".join(pieces)

        canonical = tuple(vocab[p] for p in pieces)

        if max_token_len is None:
            max_token_len = max((len(tok) for tok in vocab), default=1)

        # Group vocab entries by first character so each position only probes
        # substrings that could plausibly be a token.
        n = len(normalized)
        out: list[list[Edge]] = [[] for _ in range(n + 1)]
        for i in range(n):
            upper = min(n, i + max_token_len)
            for j in range(i + 1, upper + 1):
                piece = normalized[i:j]
                tok = vocab.get(piece)
                if tok is not None:
                    out[i].append(Edge(i, j, piece, tok))

        return cls(normalized=normalized, out=out, canonical=canonical)

    # -- counting -------------------------------------------------------------

    def path_counts(self, max_len: int | None = None) -> list[list[int]]:
        """DP table of path counts respecting a maximum tokenization length.

        Returns ``D`` where ``D[i][r]`` is the number of paths from position
        ``i`` to the sink using at most ``r`` edges. Big integers are used
        throughout because the number of tokenizations grows exponentially.

        With ``max_len=None`` the bound is ``n`` (no path can have more than
        ``n`` edges), which recovers the unconstrained count ``D[0][n]``.
        """
        n = self.n
        r_max = n if max_len is None else min(max_len, n)

        # D[i][r]: paths from i to sink using at most r edges.
        D = [[0] * (r_max + 1) for _ in range(n + 1)]
        for r in range(r_max + 1):
            D[n][r] = 1  # empty path at the sink
        for i in range(n - 1, -1, -1):
            for r in range(1, r_max + 1):
                total = 0
                for e in self.out[i]:
                    total += D[e.end][r - 1]
                D[i][r] = total
        return D

    def num_tokenizations(self, max_len: int | None = None) -> int:
        """Total number of tokenizations, optionally length-constrained."""
        D = self.path_counts(max_len)
        r_max = self.n if max_len is None else min(max_len, self.n)
        return D[0][r_max]

    # -- indexing / enumeration ----------------------------------------------

    def nth_path(self, z: int, counts: list[list[int]], max_len: int | None = None) -> tuple[int, ...]:
        """Return the ``z``-th tokenization (1-indexed) under a length budget.

        ``counts`` must be the table returned by ``path_counts(max_len)``. The
        ordering is deterministic (edge insertion order), which is exactly what
        makes without-replacement sampling possible: distinct ``z`` give
        distinct tokenizations.
        """
        n = self.n
        r_max = n if max_len is None else min(max_len, n)
        if z < 1 or z > counts[0][r_max]:
            raise IndexError(f"z={z} out of range 1..{counts[0][r_max]}")

        path: list[int] = []
        i = 0
        r = r_max
        while i != n:
            for e in self.out[i]:
                sub = counts[e.end][r - 1]
                if z <= sub:
                    path.append(e.token_id)
                    i = e.end
                    r -= 1
                    break
                z -= sub
            else:  # pragma: no cover - guarded by the range check above
                raise RuntimeError("path indexing fell through; counts inconsistent")
        return tuple(path)

    def iter_paths(self, max_len: int | None = None) -> Iterator[tuple[int, ...]]:
        """Enumerate every tokenization. Only tractable for short strings.

        This is the workhorse of the exact-enumeration correctness tests.
        """
        n = self.n
        r_max = n if max_len is None else min(max_len, n)

        def walk(i: int, budget: int, acc: list[int]) -> Iterator[tuple[int, ...]]:
            if i == n:
                yield tuple(acc)
                return
            if budget == 0:
                return
            for e in self.out[i]:
                acc.append(e.token_id)
                yield from walk(e.end, budget - 1, acc)
                acc.pop()

        yield from walk(0, r_max, [])
