# Decoding-Free Marginalization

[![tests](https://github.com/KarthikSubramanian07/decoding-free-marginalization/actions/workflows/tests.yml/badge.svg)](https://github.com/KarthikSubramanian07/decoding-free-marginalization/actions/workflows/tests.yml)
[![python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![license](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2510.20208-b31b1b.svg)](https://arxiv.org/abs/2510.20208)

An open reproduction of **"Decoding-Free Sampling Strategies for LLM
Marginalization"** ([arXiv:2510.20208](https://arxiv.org/abs/2510.20208), Pohl,
Cognetta, Lee, Okazaki, 2025). When I went looking, the paper had no public code
and no third-party reproduction, so I wrote one.

The idea in one line: a language model only ever scores *one* tokenization of a
string, but the same text can be tokenized many ways, and the "true"
probability of the text is the sum over all of them. This repo estimates that
sum without ever running generation, using cheap sampling over a tokenization
lattice plus a single scoring pass per candidate.

## Why this exists

The probability a subword model assigns to a piece of text depends on how that
text is split into tokens. The quantity you usually want is the marginal, the
total probability mass across every valid tokenization:

```
p(s) = sum over all tokenizations t of s of  prod_i P(t_i | t_<i)
```

Computing it exactly is intractable because the number of tokenizations grows
exponentially in the length of the string. The common workaround is importance
sampling, which forces the model to generate many token sequences and is slow.
The paper's contribution is to skip generation entirely: sample tokenizations by
walking a lattice (no model involved), then score each with one parallel forward
pass. That is the entire speed advantage, since scoring is one forward pass while
generation is one serial pass per token.

## How it works

The pipeline is five small pieces:

1. **Lattice** ([lattice.py](lattice.py)). Build the graph of every valid
   tokenization of a string as dynamic programming over character positions: an
   edge from position `i` to `j` exists whenever the substring between them is a
   token in the vocabulary. The paper uses finite-state transducer composition;
   this is the same object built the simple way.
2. **Samplers** ([sampling.py](sampling.py)). Two decoding-free strategies over
   the lattice. *Off-by-one* enumerates the near-canonical tokenizations, those
   that differ from the canonical one by splitting a single token in two, which
   is where most of the non-canonical probability mass lives. *Uniform sampling*
   draws tokenizations uniformly at random (optionally length-constrained and
   without replacement) to cover the tail.
3. **Scorer** ([scoring.py](scoring.py)). Batched teacher-forced scoring: feed a
   tokenization to an open-weight model, read the per-token log-probabilities,
   sum them. One forward pass each.
4. **Estimator** ([marginal.py](marginal.py)). Sample, score, and sum the unique
   tokenizations. Because it only ever sums distinct non-negative probabilities,
   the estimate is a strict lower bound on the true marginal that increases
   monotonically as you score more tokenizations.
5. **Baselines** ([baselines.py](baselines.py)). Canonical-only (fast, loose),
   and constrained-proxy importance sampling (accurate in expectation but needs
   generation, and biased low at small sample counts).

For a multiple-choice question, the marginal of each answer choice is estimated
as a continuation of the prompt, and the highest one is selected.

## Install

```bash
git clone https://github.com/KarthikSubramanian07/decoding-free-marginalization.git
cd decoding-free-marginalization
pip install -r requirements.txt
```

Gemma and Llama-2 weights need a one-time free license click on Hugging Face and
a `huggingface-cli login`. No payment.

## Quickstart

Verify the math with no model download:

```bash
pytest -q
```

Run the pipeline end to end on CPU with a tiny model (a few megabytes, no GPU):

```bash
python experiments.py --smoke
```

Run the real Q&A evaluation on a free GPU:

```bash
python experiments.py \
  --model google/gemma-3-1b-it \
  --datasets arc openbookqa medmcqa \
  --n-questions 250 --k 64 --is-samples 16
```

Add `--load-in-4bit` to fit `meta-llama/Llama-2-7b-hf` on a single T4. Or open
[notebook.ipynb](notebook.ipynb) on Colab or Kaggle and run all cells.

## Validation

Correctness comes before any claim. The test suite ([tests/](tests/)) checks,
against exact enumeration on short strings:

- the lattice enumerates every tokenization exactly and counts paths correctly,
  including under a length constraint;
- the estimator's lower bound is monotone and converges to the exact marginal as
  the sample budget grows;
- the importance-sampling proxy is a proper distribution over the lattice and its
  estimator is unbiased in expectation.

All of these pass (`pytest -q`), and the tiny-model smoke test confirms the
estimator matches the exact marginal to floating-point precision on a real model.

## Results

I am filling this table in as I run the full evaluation on free-tier GPUs. The
paper's reported figures are shown for reference; the "ours" columns stay blank
until I have run the corresponding setting, so that nothing here is a number I
did not measure.

**Q&A accuracy** (250 questions each, higher is better)

| Model | Dataset | Canonical (paper) | Lattice (paper) | Importance (paper) | Ours |
|-------|---------|------------------:|----------------:|-------------------:|:----:|
| Llama-2-7B | OpenBookQA | 49.8 | 47.3 | 48.1 | _tbd_ |
| Gemma-3-1B | ARC | _see paper_ | _see paper_ | _see paper_ | _tbd_ |
| Gemma-3-1B | MedMCQA | _see paper_ | _see paper_ | _see paper_ | _tbd_ |

**Speed and bias** (paper's reported findings I am reproducing)

- Lattice sampling estimates the marginal at **3.3x to 36.5x** less runtime than
  generation-based importance sampling on the Q&A tasks.
- Importance sampling under the constrained proxy **underestimates** the marginal
  more than half the time (up to 82% on ARC with Gemma-3-1B), because the proxy
  is only moderately rank-correlated with the base distribution (Spearman around
  0.58 to 0.75).

The headline plots (`results/convergence.png`, `results/runtime.png`,
`results/underestimation.png`) are generated by the notebook.

Each accuracy row is produced by the quickstart command with the matching
`--model` and `--datasets`; the plots and CSV land in `results/`. There is no
training stage to reproduce, since the method is scoring-only: it estimates the
marginal from a fixed open-weight model without any fine-tuning.

## Scope

In scope for this version: the Llama-2 BPE and Gemma tokenizer families, small
models that fit on a free GPU, the three Q&A datasets at 250 questions each, both
baselines, and the plots.

Out of scope for now, noted as future work: the paper's translation experiments,
the 12B/13B/27B models at scale, and alternative stochastic tokenizers such as
BPE-dropout and UnigramLM.

One honest limitation worth stating: closed models (Claude, Gemini, GPT) cannot
be the scored model, because you need to control the exact tokenization and read
per-token input probabilities, which their APIs do not expose. The paper uses
open weights for the same reason.

## Repository layout

```
lattice.py      tokenization lattice (DP over character positions)
adapters.py     tokenizer adapters (toy for tests, Hugging Face for real runs)
sampling.py     decoding-free samplers: off-by-one + uniform
scoring.py      batched teacher-forced scoring of tokenizations
marginal.py     the estimator: sample, score, sum, track the lower bound
baselines.py    canonical-only + importance sampling
experiments.py  Q&A evaluation driver for OpenBookQA / ARC / MedMCQA
analysis.py     convergence, runtime, and underestimation plots
notebook.ipynb  one-click Colab/Kaggle demo
tests/          exact-enumeration correctness tests
results/        generated CSVs and plots
```

## Citation

```bibtex
@article{pohl2025decodingfree,
  title   = {Decoding-Free Sampling Strategies for LLM Marginalization},
  author  = {Pohl, David and Cognetta, Marco and Lee, Junyoung and Okazaki, Naoaki},
  journal = {arXiv preprint arXiv:2510.20208},
  year    = {2025}
}
```

This repository is an independent reproduction and is not affiliated with the
authors. If you find a place where my implementation diverges from the paper, an
issue or PR is welcome.
