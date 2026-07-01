"""Tokenizer adapters.

The lattice and samplers only need a small, tokenizer-agnostic surface: the
canonical tokenization of a string and the surface-string-to-id vocabulary. Two
adapters implement it.

``ToyTokenizerAdapter`` is a hand-built vocabulary used by the correctness
tests. It has no dependencies, so tests run without downloading a model.

``HFTokenizerAdapter`` wraps any Hugging Face tokenizer. It deliberately does
not reimplement SentencePiece or byte-level conventions: it reads canonical
pieces straight from the tokenizer and the vocabulary straight from
``get_vocab()``, so both live in the same surface space by construction.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ToyTokenizerAdapter:
    """A fixed, in-memory vocabulary for testing.

    Parameters
    ----------
    surface_vocab:
        Map from surface string to integer id. Ids must be distinct.
    canonical:
        Optional explicit canonical tokenization, as a list of surface pieces,
        used when a test wants to pin the canonical path. If omitted, the
        canonical tokenization is computed as the tiling with the fewest tokens
        (ties broken toward longer earlier tokens), which always exists when the
        string is tileable by the vocabulary.
    """

    surface_vocab: dict[str, int]
    canonical: list[str] | None = None

    def vocab(self) -> dict[str, int]:
        return self.surface_vocab

    def id_to_piece(self) -> dict[int, str]:
        return {i: s for s, i in self.surface_vocab.items()}

    def canonical_pieces(self, text: str) -> list[str]:
        if self.canonical is not None:
            return list(self.canonical)
        return _min_token_tiling(text, self.surface_vocab)


def _min_token_tiling(text: str, vocab: dict[str, int]) -> list[str]:
    """Fewest-token tiling of ``text`` using ``vocab`` surface strings.

    A deterministic stand-in for a tokenizer's canonical tokenization, good
    enough for tests. Raises if the string cannot be tiled at all.
    """
    n = len(text)
    max_len = max((len(t) for t in vocab), default=1)
    # best[i] = (num_tokens_to_cover_suffix_from_i, first_piece)
    INF = float("inf")
    best: list[tuple[float, str | None]] = [(INF, None)] * (n + 1)
    best[n] = (0, None)
    for i in range(n - 1, -1, -1):
        # Prefer longer pieces first so ties favor longer earlier tokens.
        for j in range(min(n, i + max_len), i, -1):
            piece = text[i:j]
            if piece in vocab and best[j][0] + 1 < best[i][0]:
                best[i] = (best[j][0] + 1, piece)
    if best[0][1] is None:
        raise ValueError(f"cannot tile {text!r} with the given vocabulary")
    pieces: list[str] = []
    i = 0
    while i < n:
        piece = best[i][1]
        assert piece is not None
        pieces.append(piece)
        i += len(piece)
    return pieces


class HFTokenizerAdapter:
    """Adapter over a Hugging Face tokenizer (fast or slow).

    Notes
    -----
    * ``canonical_pieces`` uses ``tokenizer.tokenize`` which does not add
      special tokens, so the normalized string is the pure content surface.
    * ``vocab`` is ``tokenizer.get_vocab()``; its keys are in the same surface
      space as the pieces returned by ``tokenize``.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self._vocab = tokenizer.get_vocab()
        self._id_to_piece = {i: s for s, i in self._vocab.items()}

    def canonical_pieces(self, text: str) -> list[str]:
        return self.tokenizer.tokenize(text)

    def vocab(self) -> dict[str, int]:
        return self._vocab

    def id_to_piece(self) -> dict[int, str]:
        return self._id_to_piece
