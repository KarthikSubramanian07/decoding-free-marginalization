# Running the full evaluation

The correctness tests need nothing but Python. The Q&A evaluation needs a model
and a GPU for anything larger than the smoke test. Here is the whole path.

## 1. Accept the model licenses (one time, free)

Only gated weights need this. Log in at huggingface.co and click through:

- Gemma 3 1B (base): https://huggingface.co/google/gemma-3-1b-pt
- Llama 2 7B (optional): https://huggingface.co/meta-llama/Llama-2-7b-hf

Then make a read token at https://huggingface.co/settings/tokens.

Use the `-pt` (base) Gemma, not `-it`. Marginalization measures raw next-token
probabilities, so the base model is the faithful choice.

## 2. Pick a runtime with a GPU

- **Colab**: colab.research.google.com, open `notebook.ipynb` from GitHub,
  Runtime > Change runtime type > T4 GPU.
- **Kaggle**: import the notebook from GitHub, set Accelerator to GPU T4.

## 3. Log in and run

In the notebook, run the setup cell, then:

```python
from huggingface_hub import notebook_login
notebook_login()   # paste your token
```

Verify the math (should print `15 passed`):

```python
!pytest -q
```

Run the evaluation. For the full paper-scale run, use 250 questions:

```bash
python experiments.py \
  --model google/gemma-3-1b-pt \
  --datasets arc openbookqa medmcqa \
  --n-questions 250 --k 64 --is-samples 16
```

For the Llama-2 7B row, add 4-bit so it fits on a T4:

```bash
python experiments.py --model meta-llama/Llama-2-7b-hf --load-in-4bit \
  --datasets arc openbookqa medmcqa --n-questions 250
```

## 4. Read the outputs

`results/` will hold:

- `qa_summary.csv` - accuracy, seconds, forward passes, and the underestimation
  rate per method per dataset.
- `runtime.png`, `underestimation.png`, `convergence.png`.

Expected, matching the paper: lattice accuracy close to importance sampling with
canonical often highest, lattice several times faster than importance sampling,
and an underestimation rate above 0.5.

## 5. Fill in the README

Copy the accuracy numbers into the results table in `README.md` (multiply by
100), add a line on where they matched the paper and where they did not, then
commit `README.md` and `results/`.

## Cost

Gemma-3-1B over 250 questions on three datasets is roughly 20 to 40 minutes on a
free T4, dominated by the importance-sampling baseline. Download the CSV and
plots from Colab before the runtime disconnects; Kaggle keeps them in the output.
