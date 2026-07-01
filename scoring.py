"""Teacher-forced scoring of tokenizations with an open-weight model.

Scoring a fixed tokenization needs a single forward pass: feed the token ids,
read the per-position next-token log-probabilities, and sum the ones for the
tokens that actually appear. This is the whole efficiency argument of the paper.
Generation, by contrast, needs one serial forward pass per token.

The scorer batches many tokenizations into padded forward passes. It is the
only module that depends on ``torch`` and ``transformers``; everything upstream
(lattice, sampling, marginal) is pure Python and model-free.
"""

from __future__ import annotations

from typing import Sequence

import torch

Tokenization = tuple[int, ...]


def load_model(
    model_name: str,
    load_in_4bit: bool = False,
    device: str | None = None,
    dtype: str = "auto",
):
    """Load a Hugging Face causal LM and its tokenizer.

    ``load_in_4bit`` uses bitsandbytes NF4 quantization so 7B-class models fit on
    a free T4. Returns ``(model, tokenizer)``.
    """
    import inspect

    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # transformers renamed ``torch_dtype`` to ``dtype``; support both.
    sig = inspect.signature(AutoModelForCausalLM.from_pretrained)
    dtype_key = "dtype" if "dtype" in sig.parameters else "torch_dtype"

    # bfloat16 on GPU, not float16. Models like Gemma are trained in bf16 and
    # overflow fp16, producing NaN logits (and, downstream, blank/garbage
    # marginals). bf16 has the same exponent range as fp32, so it is safe.
    gpu_dtype = torch.bfloat16

    kwargs: dict = {}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=gpu_dtype,
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = {"": 0} if device == "cuda" else device
    else:
        if dtype == "auto":
            kwargs[dtype_key] = gpu_dtype if device == "cuda" else torch.float32
        else:
            kwargs[dtype_key] = getattr(torch, dtype)

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    if not load_in_4bit:
        model = model.to(device)
    model.eval()
    return model, tokenizer


class LMScorer:
    """Batched teacher-forced scorer.

    Parameters
    ----------
    model, tokenizer:
        A loaded causal LM and its tokenizer.
    add_bos:
        Prepend the beginning-of-sequence token before scoring. Most models
        (Llama2, Gemma) expect it and their canonical probabilities assume it.
    batch_size:
        Number of tokenizations per forward pass.
    """

    def __init__(self, model, tokenizer, add_bos: bool = True, batch_size: int = 16):
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.device = next(model.parameters()).device

        bos = tokenizer.bos_token_id
        self.bos_id = bos if (add_bos and bos is not None) else None
        pad = tokenizer.pad_token_id
        if pad is None:
            pad = tokenizer.eos_token_id
        self.pad_id = pad if pad is not None else 0

    def _prefix(self, context_ids: Sequence[int] | None) -> list[int]:
        prefix: list[int] = []
        if self.bos_id is not None:
            prefix.append(self.bos_id)
        if context_ids:
            prefix.extend(context_ids)
        return prefix

    @torch.no_grad()
    def score(
        self,
        tokenizations: Sequence[Tokenization],
        context_ids: Sequence[int] | None = None,
    ) -> list[float]:
        """Return the log-probability of each tokenization.

        When ``context_ids`` is given (a fixed, canonically tokenized prompt),
        each tokenization is scored as a continuation of that context and only
        the continuation's tokens contribute to the sum. This is what a Q&A item
        needs: marginalize the answer given the question. Padding positions are
        masked out, so mixed-length sequences can share a batch safely.
        """
        if not tokenizations:
            return []

        prefix = self._prefix(context_ids)
        out: list[float] = [0.0] * len(tokenizations)
        order = sorted(range(len(tokenizations)), key=lambda i: len(tokenizations[i]))

        for start in range(0, len(order), self.batch_size):
            batch_idx = order[start : start + self.batch_size]
            rows = [(prefix, list(tokenizations[i])) for i in batch_idx]
            out_vals = self._score_batch(rows)
            for i, v in zip(batch_idx, out_vals):
                out[i] = v
        return out

    def _score_batch(self, rows: list[tuple[list[int], list[int]]]) -> list[float]:
        seqs = [prefix + answer for prefix, answer in rows]
        plens = [len(prefix) for prefix, _ in rows]
        max_len = max(len(s) for s in seqs)

        input_ids = torch.full((len(seqs), max_len), self.pad_id, dtype=torch.long)
        attn = torch.zeros((len(seqs), max_len), dtype=torch.long)
        for r, s in enumerate(seqs):
            input_ids[r, : len(s)] = torch.tensor(s, dtype=torch.long)
            attn[r, : len(s)] = 1

        input_ids = input_ids.to(self.device)
        attn = attn.to(self.device)

        logits = self.model(input_ids=input_ids, attention_mask=attn).logits
        log_probs = torch.log_softmax(logits.float(), dim=-1)

        # Predict position t from position t-1: align logits[:-1] with labels[1:].
        shift_logp = log_probs[:, :-1, :]
        labels = input_ids[:, 1:]

        # A label at shifted index j sits at original position j+1; it belongs to
        # the answer (not the prompt) when j+1 >= prefix length.
        positions = torch.arange(1, max_len, device=self.device).unsqueeze(0)
        plen_t = torch.tensor(plens, device=self.device).unsqueeze(1)
        answer_mask = (positions >= plen_t) & attn[:, 1:].bool()

        gathered = shift_logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        gathered = gathered.masked_fill(~answer_mask, 0.0)
        totals = gathered.sum(dim=-1)

        if torch.isnan(totals).any():
            raise RuntimeError(
                "NaN log-probabilities from the model. This usually means the "
                "model was loaded in float16 but needs bfloat16 (Gemma is a "
                "common case). Reload with dtype='bfloat16' or on a GPU, where "
                "load_model now defaults to bfloat16."
            )
        return [float(x) for x in totals.cpu()]

    def score_fn(self, context_ids: Sequence[int] | None = None):
        """Return a ``score_fn`` closure for the marginal estimator.

        Bind ``context_ids`` to score answer tokenizations given a fixed prompt.
        """

        def _score(toks):
            return self.score(list(toks), context_ids=context_ids)

        return _score
