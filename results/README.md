# results/

This folder holds generated outputs. It ships empty on purpose: the numbers you
get should be numbers you produced, not numbers I pasted in.

Running the experiments or the notebook writes:

- `qa_summary.csv` - accuracy, wall-clock, and forward-pass counts per method
  per dataset, plus the importance-sampling underestimation rate.
- `convergence.png` - marginal estimate vs. number of unique tokenizations
  scored, with the exact marginal drawn as a dashed reference on short strings.
- `runtime.png` - wall-clock per method per dataset.
- `underestimation.png` - how often importance sampling lands below the lattice
  estimate.

Regenerate everything with:

```bash
python experiments.py --model google/gemma-3-1b-it --datasets arc openbookqa medmcqa
```

or open `notebook.ipynb` on Colab or Kaggle and run all cells.
