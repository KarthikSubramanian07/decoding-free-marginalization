# Contributing

Thanks for taking a look. This is a reproduction repo, so the most useful
contributions are ones that make the reproduction more faithful or more
trustworthy.

## Good things to send

- A place where the implementation diverges from the paper, with a pointer to
  the relevant section or equation.
- A new correctness test, especially one that pins down a subtle case.
- Results from running the evaluation on a model or dataset I have not filled in
  yet, with the exact command you used so I can reproduce it.
- Support for another tokenizer family or one of the out-of-scope items in the
  README.

## Ground rules

- Run the tests before opening a PR: `pytest -q`. They are pure Python and need
  no model download, so there is no excuse to skip them.
- Keep the core modules (`lattice.py`, `sampling.py`, `marginal.py`) free of
  heavy dependencies. Only `scoring.py` and things downstream of it should touch
  `torch` or `transformers`.
- If you add a result to the README, add the command that produced it. No number
  goes in that has not been measured.

## Development setup

```bash
pip install -r requirements.txt
pytest -q
python experiments.py --smoke
```
