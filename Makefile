.PHONY: install test validate lint smoke experiment plots clean

install:
	pip install -r requirements.txt

# Pure-Python correctness suite, no model download.
test:
	pytest -q

# The correctness gate as one command, with a readable report.
validate:
	python validate.py

# Style and static checks.
lint:
	ruff check .

# End-to-end pipeline on CPU with a tiny model.
smoke:
	python experiments.py --smoke

# Full Q&A reproduction. Override MODEL / DATASETS / N as needed, e.g.
#   make experiment MODEL=meta-llama/Llama-2-7b-hf EXTRA=--load-in-4bit
MODEL ?= google/gemma-3-1b-pt
DATASETS ?= arc openbookqa medmcqa
N ?= 250
experiment:
	python experiments.py --model $(MODEL) --datasets $(DATASETS) --n-questions $(N) $(EXTRA)

clean:
	rm -rf __pycache__ */__pycache__ .pytest_cache
	find . -name '*.pyc' -delete
