"""Plots and analysis for the headline artifacts.

Three figures, matching the paper's story:

* ``plot_convergence`` - the marginal estimate as a function of how many unique
  tokenizations have been scored. The curve is non-decreasing and, on short
  strings, converges to the exact marginal (drawn as a reference line).
* ``plot_runtime`` - wall-clock per method, showing the lattice method's speedup
  over importance sampling.
* ``plot_underestimation`` - how often importance sampling lands below the
  lattice estimate, per dataset.

All plotting imports live inside the functions so the rest of the repo stays
import-light. Figures are written to ``results/``.
"""

from __future__ import annotations

import math
import os
import random

from adapters import HFTokenizerAdapter
from lattice import Lattice
from marginal import estimate_marginal, exact_marginal


def convergence_data(
    model,
    tokenizer,
    text: str,
    k: int = 128,
    max_len: int | None = None,
    add_bos: bool = True,
    seed: int = 0,
    context: str | None = None,
    exact_threshold: int = 5000,
):
    """Return estimate-vs-samples data for a single string.

    ``curve`` is the estimated log-marginal after each unique tokenization is
    added. ``exact`` is the true log-marginal when the lattice is small enough to
    enumerate, else ``None``.
    """
    from scoring import LMScorer

    adapter = HFTokenizerAdapter(tokenizer)
    id_to_piece = adapter.id_to_piece()
    scorer = LMScorer(model, tokenizer, add_bos=add_bos)
    context_ids = (
        tokenizer.encode(context, add_special_tokens=False) if context else None
    )
    score_fn = scorer.score_fn(context_ids=context_ids)

    lat = Lattice.build(text, adapter)
    est = estimate_marginal(
        lat, id_to_piece, score_fn, k=k, max_len=max_len, rng=random.Random(seed)
    )

    exact = None
    if lat.num_tokenizations(max_len=max_len) <= exact_threshold:
        exact = exact_marginal(lat, score_fn, max_len=max_len)

    return {
        "text": text,
        "n_tokenizations": lat.num_tokenizations(),
        "num_near_canonical": est.num_near_canonical,
        "curve": est.curve,
        "canonical_log_prob": est.canonical_log_prob,
        "exact": exact,
    }


def plot_convergence(data_list, out_path: str = "results/convergence.png") -> str:
    """Plot one or more convergence curves (estimate vs. samples)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for d in data_list:
        xs = range(1, len(d["curve"]) + 1)
        ys = [math.exp(v) for v in d["curve"]]
        label = d["text"] if len(d["text"]) <= 24 else d["text"][:21] + "..."
        (line,) = ax.plot(xs, ys, marker="o", ms=3, label=label)
        if d.get("exact") is not None:
            ax.axhline(
                math.exp(d["exact"]), color=line.get_color(), ls="--", lw=1, alpha=0.6
            )
    ax.set_xlabel("unique tokenizations scored")
    ax.set_ylabel("marginal estimate p(s)")
    ax.set_title("Decoding-free marginal estimate vs. samples\n(dashed = exact marginal)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def plot_runtime(results, out_path: str = "results/runtime.png") -> str:
    """Grouped bar chart of wall-clock per method per dataset."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    datasets = [r.dataset for r in results]
    methods = list(results[0].tallies.keys()) if results else []

    fig, ax = plt.subplots(figsize=(7, 4.5))
    width = 0.8 / max(len(methods), 1)
    for mi, method in enumerate(methods):
        xs = [i + mi * width for i in range(len(datasets))]
        ys = [r.tallies[method].seconds for r in results]
        ax.bar(xs, ys, width=width, label=method)
    ax.set_xticks([i + width * (len(methods) - 1) / 2 for i in range(len(datasets))])
    ax.set_xticklabels(datasets)
    ax.set_ylabel("wall-clock seconds")
    ax.set_title("Runtime by method")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def plot_underestimation(results, out_path: str = "results/underestimation.png") -> str:
    """Bar chart of the fraction of items where importance sampling < lattice."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    datasets = [r.dataset for r in results]
    rates = [r.underestimation_rate for r in results]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(datasets, rates, color="#c0392b")
    ax.axhline(0.5, color="k", ls="--", lw=1, label="50% (no bias)")
    ax.set_ylim(0, 1)
    ax.set_ylabel("fraction of items IS < lattice")
    ax.set_title("Importance sampling underestimates the marginal")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path
