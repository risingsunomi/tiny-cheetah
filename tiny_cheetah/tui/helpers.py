from __future__ import annotations

import time
from typing import Any

import tinygrad as tg
from transformers import AutoTokenizer

from tiny_cheetah.models.llm.model import Model
from tiny_cheetah.models.llm.helpers import sample


def streaming_generate(
    model: Model,
    input_ids: tg.Tensor,
    attention_mask: tg.Tensor,
    tokenizer: AutoTokenizer,
    max_new_tokens: int = 512,
    temp: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.8,
    alpha_f: float = 0.0,
    alpha_p: float = 0.0,
    verbose: bool = False,
) -> tuple[list[int], float]:
    device = input_ids.device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    out_tokens: list[int] = []
    curr_pos = attention_mask.shape[1]
    generated = 0
    start_time = time.time()

    # initial prefill
    position_ids = ((attention_mask.cumsum(axis=1) - 1) * attention_mask).to(device)
    logits = model(input_ids, attention_mask=attention_mask, position_ids=position_ids)
    next_logit = logits[:, -1, :].flatten()
    tok = sample(next_logit, temp=temp, k=top_k, p=top_p, af=alpha_f, ap=alpha_p).item()
    out_tokens.append(tok)
    generated += 1
    curr_pos += 1

    eos_hit = tok == tokenizer.eos_token_id
    limit = max_new_tokens - 1 if max_new_tokens > 0 else None

    while not eos_hit:
        if limit is not None and generated >= limit:
            break

        next_tok = tg.Tensor([[tok]], device=device)
        attention_mask = attention_mask.cat(
            tg.Tensor.ones((attention_mask.shape[0], 1), device=device), dim=1
        )
        position_ids = tg.Tensor([curr_pos], device=device)

        logits = model(
            next_tok,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        next_logit = logits[:, -1, :].flatten()
        tok = sample(next_logit, temp=temp, k=top_k, p=top_p, af=alpha_f, ap=alpha_p).item()
        out_tokens.append(tok)
        generated += 1
        curr_pos += 1

        if tok == tokenizer.eos_token_id:
            eos_hit = True

    elapsed = time.time() - start_time

    if verbose:
        tok_s = generated / elapsed if elapsed > 0 else float("inf")
        print(f"[stream] {generated} tokens in {elapsed:.3f}s -> {tok_s:.2f} tok/s")

    return out_tokens, elapsed
