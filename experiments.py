"""Q&A evaluation driver.

For each multiple-choice question we estimate the marginal probability of every
answer choice as a continuation of the prompt, then pick the choice with the
highest estimate. Three methods are compared:

* ``canonical``  - score only the canonical tokenization of the answer.
* ``lattice``    - the decoding-free marginal estimator (this repo's method).
* ``importance`` - constrained-proxy importance sampling (needs generation).

The driver records accuracy, wall-clock time, and forward-pass counts per
method, plus how often importance sampling lands below the lattice estimate
(the paper's underestimation finding). Results are written to ``results/``.

Datasets: OpenBookQA, ARC (Challenge), MedMCQA, loaded via ``datasets`` and
subsampled to a fixed number of questions with a fixed seed for reproducibility.

Run ``python experiments.py --help`` for options, or ``--smoke`` to exercise the
full pipeline on CPU with a tiny model and no downloads beyond a few megabytes.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import time
from dataclasses import dataclass, field

from adapters import HFTokenizerAdapter
from baselines import HFConditional, canonical_log_prob, importance_sample
from lattice import Lattice
from marginal import estimate_marginal


@dataclass
class QAItem:
    question: str
    choices: list[str]
    answer_idx: int


# --------------------------------------------------------------------------- #
# Dataset loading
# --------------------------------------------------------------------------- #

def load_qa_dataset(name: str, n: int | None = 250, seed: int = 0) -> list[QAItem]:
    """Load and subsample a multiple-choice dataset into a common shape."""
    from datasets import load_dataset

    name = name.lower()
    if name in ("openbookqa", "obqa"):
        # Namespaced repo ids are required by modern datasets / huggingface_hub.
        ds = load_dataset("allenai/openbookqa", "main", split="validation")
        items = [
            QAItem(
                question=r["question_stem"],
                choices=r["choices"]["text"],
                answer_idx=r["choices"]["label"].index(r["answerKey"]),
            )
            for r in ds
        ]
    elif name in ("arc", "arc-challenge", "arc_challenge"):
        ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="validation")
        items = []
        for r in ds:
            labels = r["choices"]["label"]
            if r["answerKey"] not in labels:
                continue
            items.append(
                QAItem(
                    question=r["question"],
                    choices=r["choices"]["text"],
                    answer_idx=labels.index(r["answerKey"]),
                )
            )
    elif name == "medmcqa":
        ds = load_dataset("openlifescienceai/medmcqa", split="validation")
        items = [
            QAItem(
                question=r["question"],
                choices=[r["opa"], r["opb"], r["opc"], r["opd"]],
                answer_idx=r["cop"],
            )
            for r in ds
        ]
    else:
        raise ValueError(f"unknown dataset {name!r}")

    rng = random.Random(seed)
    rng.shuffle(items)
    if n is not None:
        items = items[:n]
    return items


def build_prompt(question: str) -> str:
    """Prompt template shared across datasets and models."""
    return f"Question: {question.strip()}\nAnswer:"


def answer_string(choice: str) -> str:
    """The scored continuation. Leading space so it reads as a natural answer."""
    return " " + choice.strip()


# --------------------------------------------------------------------------- #
# Per-item evaluation
# --------------------------------------------------------------------------- #

@dataclass
class MethodTally:
    correct: int = 0
    total: int = 0
    seconds: float = 0.0
    forward_passes: int = 0

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0


@dataclass
class RunResult:
    dataset: str
    model_name: str
    tallies: dict[str, MethodTally]
    is_below_lattice: int = 0
    is_lattice_pairs: int = 0
    records: list[dict] = field(default_factory=list)

    @property
    def underestimation_rate(self) -> float:
        return self.is_below_lattice / self.is_lattice_pairs if self.is_lattice_pairs else 0.0


def evaluate(
    model,
    tokenizer,
    dataset_name: str,
    items: list[QAItem],
    methods: tuple[str, ...] = ("canonical", "lattice", "importance"),
    k: int = 64,
    max_len: int | None = None,
    n_is_samples: int = 16,
    add_bos: bool = True,
    seed: int = 0,
    verbose: bool = True,
) -> RunResult:
    """Run the selected methods over ``items`` and tally accuracy and cost."""
    from scoring import LMScorer

    adapter = HFTokenizerAdapter(tokenizer)
    id_to_piece = adapter.id_to_piece()
    scorer = LMScorer(model, tokenizer, add_bos=add_bos)
    rng = random.Random(seed)
    # Compute the longest token length once; reused for every lattice build.
    max_token_len = max((len(t) for t in adapter.vocab()), default=1)

    tallies = {m: MethodTally() for m in methods}
    result = RunResult(dataset=dataset_name, model_name=model.name_or_path, tallies=tallies)

    for qi, item in enumerate(items):
        prompt = build_prompt(item.question)
        context_ids = tokenizer.encode(prompt, add_special_tokens=False)

        # Build one lattice per answer choice.
        lattices = [
            Lattice.build(answer_string(c), adapter, max_token_len=max_token_len)
            for c in item.choices
        ]

        per_choice_lattice_lp: list[float] = []
        for method in methods:
            t0 = time.perf_counter()
            scores: list[float] = []

            if method == "canonical":
                score_fn = scorer.score_fn(context_ids=context_ids)
                scores = [canonical_log_prob(lat, score_fn) for lat in lattices]

            elif method == "lattice":
                score_fn = scorer.score_fn(context_ids=context_ids)
                for lat in lattices:
                    est = estimate_marginal(
                        lat, id_to_piece, score_fn, k=k, max_len=max_len, rng=rng
                    )
                    scores.append(est.log_marginal)
                per_choice_lattice_lp = scores

            elif method == "importance":
                for lat in lattices:
                    cond = HFConditional(model, tokenizer, add_bos=add_bos, context_ids=context_ids)
                    res = importance_sample(lat, cond, n_samples=n_is_samples, rng=rng)
                    scores.append(res.log_marginal)
                    tallies["importance"].forward_passes += res.n_forward_passes
            else:
                raise ValueError(f"unknown method {method!r}")

            tallies[method].seconds += time.perf_counter() - t0
            pred = max(range(len(scores)), key=lambda i: scores[i])
            tallies[method].correct += int(pred == item.answer_idx)
            tallies[method].total += 1

            result.records.append(
                {
                    "dataset": dataset_name,
                    "model": model.name_or_path,
                    "question_idx": qi,
                    "method": method,
                    "pred": pred,
                    "gold": item.answer_idx,
                    "correct": int(pred == item.answer_idx),
                }
            )

            # Underestimation check: importance vs lattice on the gold choice.
            if method == "importance" and per_choice_lattice_lp:
                gold = item.answer_idx
                result.is_lattice_pairs += 1
                result.is_below_lattice += int(scores[gold] < per_choice_lattice_lp[gold])

        if verbose and (qi + 1) % 25 == 0:
            line = " | ".join(f"{m}: {tallies[m].accuracy:.3f}" for m in methods)
            print(f"[{dataset_name}] {qi + 1}/{len(items)}  {line}")

    return result


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #

def write_summary(results: list[RunResult], out_dir: str = "results") -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "qa_summary.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "dataset",
                "model",
                "method",
                "accuracy",
                "n_questions",
                "seconds",
                "forward_passes",
                "underestimation_rate",
            ]
        )
        for r in results:
            for method, t in r.tallies.items():
                w.writerow(
                    [
                        r.dataset,
                        r.model_name,
                        method,
                        f"{t.accuracy:.4f}",
                        t.total,
                        f"{t.seconds:.2f}",
                        t.forward_passes,
                        f"{r.underestimation_rate:.4f}" if method == "importance" else "",
                    ]
                )
    return path


def print_table(results: list[RunResult]) -> None:
    print("\n=== Q&A results ===")
    header = f"{'dataset':<12} {'method':<11} {'acc':>7} {'sec':>9} {'fwd':>10} {'IS<lat':>7}"
    print(header)
    print("-" * len(header))
    for r in results:
        lat_t = r.tallies.get("lattice")
        for method, t in r.tallies.items():
            speedup = ""
            if method == "lattice" and "importance" in r.tallies:
                imp = r.tallies["importance"].seconds
                if lat_t and lat_t.seconds > 0:
                    speedup = f"  ({imp / lat_t.seconds:.1f}x vs IS)"
            und = f"{r.underestimation_rate:.2f}" if method == "importance" else ""
            print(
                f"{r.dataset:<12} {method:<11} {t.accuracy:>7.3f} {t.seconds:>9.2f} "
                f"{t.forward_passes:>10} {und:>7}{speedup}"
            )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="google/gemma-3-1b-it")
    p.add_argument(
        "--datasets",
        nargs="+",
        default=["arc", "openbookqa", "medmcqa"],
    )
    p.add_argument("--n-questions", type=int, default=250)
    p.add_argument("--k", type=int, default=64, help="tokenizations scored by the lattice method")
    p.add_argument("--max-len", type=int, default=None, help="max tokens per tokenization")
    p.add_argument("--is-samples", type=int, default=16, help="importance-sampling samples")
    p.add_argument("--methods", nargs="+", default=["canonical", "lattice", "importance"])
    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="results")
    p.add_argument(
        "--smoke",
        action="store_true",
        help="tiny CPU model + few questions to verify the pipeline end to end",
    )
    args = p.parse_args()

    from scoring import load_model

    if args.smoke:
        args.model = "hf-internal-testing/tiny-random-gpt2"
        args.n_questions = 8
        args.k = 16
        args.is_samples = 4

    print(f"Loading {args.model} ...")
    model, tokenizer = load_model(args.model, load_in_4bit=args.load_in_4bit)

    results: list[RunResult] = []
    for name in args.datasets:
        items = load_qa_dataset(name, n=args.n_questions, seed=args.seed)
        print(f"Loaded {len(items)} questions from {name}")
        res = evaluate(
            model,
            tokenizer,
            dataset_name=name,
            items=items,
            methods=tuple(args.methods),
            k=args.k,
            max_len=args.max_len,
            n_is_samples=args.is_samples,
            seed=args.seed,
        )
        results.append(res)

    print_table(results)
    path = write_summary(results, out_dir=args.out_dir)
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
