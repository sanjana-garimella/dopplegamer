"""Preble benchmark scaffold.

Preble (WukLab/UCSD) optimizes serving for long shared prefixes — exactly the
pattern that emerges as RPS+ game history grows. This module simulates the
Preble shared-prefix advantage by caching the prefix token IDs and only
re-encoding the per-turn delta. With a real Preble cluster, point
`PrebleClient` at the deployed endpoint instead.
"""

from __future__ import annotations

import time

from serving.base import InferenceEngine, InferenceResult, _Timer


class PrebleEngine(InferenceEngine):
    name = "preble"

    def __init__(self, model_name: str, prefix_cache: bool = True) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device).eval()
        self.prefix_cache = prefix_cache
        self._cached_prefix: tuple[str, dict] | None = None  # (prefix_text, past_key_values)

    def warmup(self) -> None:
        self.generate("warmup", max_new_tokens=1)

    def generate(self, prompt: str, max_new_tokens: int = 8) -> InferenceResult:
        import torch

        # Identify shared prefix with the cached prompt.
        if self.prefix_cache and self._cached_prefix is not None:
            cached_text, cached_kv = self._cached_prefix
            shared = _shared_prefix(cached_text, prompt)
        else:
            shared = ""
            cached_kv = None

        with _Timer() as t_prefill, torch.no_grad():
            new_text = prompt[len(shared):]
            inputs = self.tokenizer(new_text, return_tensors="pt").to(self.device)
            past = cached_kv if cached_kv is not None else None
            output = self.model(**inputs, past_key_values=past, use_cache=True)
            kv = output.past_key_values
        ttft_ms = t_prefill.elapsed_ms

        with _Timer() as t_decode, torch.no_grad():
            input_ids = inputs["input_ids"]
            for _ in range(max_new_tokens):
                logits = output.logits[:, -1, :]
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
                input_ids = torch.cat([input_ids, next_id], dim=-1)
                output = self.model(input_ids=next_id, past_key_values=kv, use_cache=True)
                kv = output.past_key_values
        decode_ms = t_decode.elapsed_ms

        text = self.tokenizer.decode(input_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        if self.prefix_cache:
            self._cached_prefix = (prompt, kv)

        return InferenceResult(
            text=text,
            prompt_tokens=int(inputs["input_ids"].shape[1]),
            output_tokens=max_new_tokens,
            ttft_ms=ttft_ms,
            tpot_ms=decode_ms / max(1, max_new_tokens),
            total_latency_ms=ttft_ms + decode_ms,
            extra={"shared_prefix_chars": len(shared)},
        )


def _shared_prefix(a: str, b: str) -> str:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return a[:i]
