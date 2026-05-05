"""HuggingFace baseline serving — single-stream autoregressive generation.

This is the slowest engine but useful as a control. Token timings split into
prefill (TTFT) and per-decode-token (TPOT). KV cache size is approximated from
the model config and number of cached tokens.
"""

from __future__ import annotations

import time

from serving.base import InferenceEngine, InferenceResult, _Timer


class HFBaselineEngine(InferenceEngine):
    name = "huggingface"

    def __init__(self, model_name: str, device: str = "auto", dtype: str = "auto") -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        kwargs = {}
        if dtype != "auto":
            import torch as _torch
            kwargs["torch_dtype"] = getattr(_torch, dtype)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else "cpu"
        self.model.to(self.device).eval()

    def generate(self, prompt: str, max_new_tokens: int = 8) -> InferenceResult:
        import torch

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_tokens = int(inputs["input_ids"].shape[1])

        # Prefill — generate exactly one token to measure TTFT.
        with _Timer() as t_prefill, torch.no_grad():
            first = self.model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True,
            )
        ttft_ms = t_prefill.elapsed_ms

        # Decode — generate fully and subtract prefill estimate.
        with _Timer() as t_full, torch.no_grad():
            full = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True,
            )
        decode_ms = max(0.0, t_full.elapsed_ms - ttft_ms)
        output_tokens = int(full.shape[1] - prompt_tokens)
        tpot_ms = decode_ms / max(1, output_tokens - 1) if output_tokens > 1 else decode_ms
        text = self.tokenizer.decode(full[0][prompt_tokens:], skip_special_tokens=True)

        kv_mb = self._estimate_kv_mb(prompt_tokens + output_tokens)
        return InferenceResult(
            text=text,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            ttft_ms=ttft_ms,
            tpot_ms=tpot_ms,
            total_latency_ms=t_full.elapsed_ms,
            kv_cache_mb=kv_mb,
        )

    def _estimate_kv_mb(self, total_tokens: int) -> float:
        cfg = self.model.config
        n_layers = getattr(cfg, "num_hidden_layers", 0)
        n_kv = getattr(cfg, "num_key_value_heads", getattr(cfg, "num_attention_heads", 1))
        head_dim = getattr(cfg, "hidden_size", 0) // max(1, getattr(cfg, "num_attention_heads", 1))
        bytes_per = 2  # fp16
        bytes_total = 2 * n_layers * n_kv * head_dim * total_tokens * bytes_per  # K and V
        return bytes_total / (1024 * 1024)
