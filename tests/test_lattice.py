"""Exact-enumeration tests for the lattice.

These are the correctness gate for the whole project: if the lattice does not
enumerate tokenizations exactly, nothing downstream can be trusted.
"""

from __future__ import annotations

from adapters import ToyTokenizerAdapter
from lattice import Lattice
from tests.toy import TOY_VOCAB


def build(text: str) -> Lattice:
    return Lattice.build(text, ToyTokenizerAdapter(TOY_VOCAB))


def brute_force_tokenizations(text: str, vocab: dict[str, int]) -> set[tuple[int, ...]]:
    """Every tiling of ``text`` by vocab surface strings, as id tuples."""
    n = len(text)
    results: set[tuple[int, ...]] = set()

    def walk(i: int, acc: list[int]):
        if i == n:
            results.add(tuple(acc))
            return
        for j in range(i + 1, n + 1):
            piece = text[i:j]
            if piece in vocab:
                acc.append(vocab[piece])
                walk(j, acc)
                acc.pop()

    walk(0, [])
    return results


def test_canonical_path_is_in_lattice():
    lat = build("aaaa")
    paths = set(lat.iter_paths())
    assert lat.canonical in paths


def test_enumeration_matches_brute_force():
    for text in ["aaaa", "abba", "bbbb", "abab", "aabb", "baab"]:
        lat = build(text)
        got = set(lat.iter_paths())
        expected = brute_force_tokenizations(text, TOY_VOCAB)
        assert got == expected, text
        assert lat.num_tokenizations() == len(expected), text


def test_nth_path_covers_every_path_exactly_once():
    lat = build("aaaa")
    counts = lat.path_counts()
    total = lat.num_tokenizations()
    seen = [lat.nth_path(z, counts) for z in range(1, total + 1)]
    # Distinct and complete.
    assert len(set(seen)) == total
    assert set(seen) == set(lat.iter_paths())


def test_length_constraint_counts_match_brute_force():
    text = "aaaa"
    lat = build(text)
    all_paths = list(lat.iter_paths())
    for max_len in range(1, 6):
        expected = sum(1 for p in all_paths if len(p) <= max_len)
        assert lat.num_tokenizations(max_len=max_len) == expected, max_len
        # And enumerating with the budget yields exactly those paths.
        got = {p for p in lat.iter_paths(max_len=max_len)}
        assert got == {p for p in all_paths if len(p) <= max_len}, max_len


def test_known_composition_count():
    # "aaaa" tiled by {a, aa, aaa}: compositions of 4 using parts 1,2,3 => 7.
    lat = build("aaaa")
    assert lat.num_tokenizations() == 7
