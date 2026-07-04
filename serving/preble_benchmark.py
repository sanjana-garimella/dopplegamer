"""HF prefix-cache ablation (not a Preble cluster client).

Simulates shared-prefix reuse by caching past_key_values at prompt length only
(token-aligned). Use engine name `hf_prefix_cache` (alias: `preble`).
"""

from __future__ import annotations

from serving.base import InferenceEngine, InferenceResult, _Timer


class HFPrefixCacheEngine(InferenceEngine):
    name = "hf_prefix_cache"

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
        # (prompt_token_ids, past_key_values at prompt length only)
        self._cached: tuple[list[int], object] | None = None

    def warmup(self) -> None:
        self.generate("warmup", max_new_tokens=1)

    def generate(self, prompt: str, max_new_tokens: int = 8) -> InferenceResult:
        import torch

        full = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        full_ids = full["input_ids"][0].tolist()
        prompt_tokens = len(full_ids)

        shared_len = 0
        past = None
        if self.prefix_cache and self._cached is not None:
            cached_ids, cached_kv = self._cached
            # Only reuse when the new prompt extends the entire previous prompt.
            if len(full_ids) >= len(cached_ids) and full_ids[: len(cached_ids)] == cached_ids:
                shared_len = len(cached_ids)
                past = cached_kv

        # Count hits only when we actually skip prefill work (strict prefix extension).
        reused_tokens = 0
        with _Timer() as t_prefill, torch.no_grad():
            if past is not None and 0 < shared_len < prompt_tokens:
                new_ids = torch.tensor([full_ids[shared_len:]], device=self.device)
                out = self.model(input_ids=new_ids, past_key_values=past, use_cache=True)
                reused_tokens = shared_len
            else:
                # Full prefill (no cache, or exact prompt repeat where incremental reuse
                # would double-count the last cached token).
                out = self.model(**full, use_cache=True)
                reused_tokens = 0
            kv = out.past_key_values
            next_id = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        ttft_ms = t_prefill.elapsed_ms

        # Cache prompt-only KV (before decode tokens are appended).
        if self.prefix_cache:
            self._cached = (full_ids, kv)

        generated: list[int] = [int(next_id.item())]
        with _Timer() as t_decode, torch.no_grad():
            for _ in range(max(0, max_new_tokens - 1)):
                out = self.model(input_ids=next_id, past_key_values=kv, use_cache=True)
                next_id = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
                kv = out.past_key_values
                generated.append(int(next_id.item()))
        decode_ms = t_decode.elapsed_ms

        output_tokens = len(generated)
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        hit = reused_tokens
        miss = prompt_tokens - reused_tokens
        return InferenceResult(
            text=text,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            ttft_ms=ttft_ms,
            tpot_ms=decode_ms / max(1, output_tokens - 1) if output_tokens > 1 else 0.0,
            total_latency_ms=ttft_ms + decode_ms,
            kv_cache_mb=self._estimate_kv_mb(prompt_tokens + output_tokens),
            prefix_cache_hit_tokens=hit,
            prefix_cache_miss_tokens=miss,
            extra={
                "actual_backend": "hf_prefix_cache",
                "shared_prefix_tokens": shared_len,
                "note": "HF ablation, not a Preble cluster",
            },
        )

    def _estimate_kv_mb(self, total_tokens: int) -> float:
        cfg = self.model.config
        n_layers = getattr(cfg, "num_hidden_layers", 0)
        n_kv = getattr(cfg, "num_key_value_heads", getattr(cfg, "num_attention_heads", 1))
        head_dim = getattr(cfg, "hidden_size", 0) // max(1, getattr(cfg, "num_attention_heads", 1))
        return (2 * n_layers * n_kv * head_dim * total_tokens * 2) / (1024 * 1024)


# Backward-compatible alias
PrebleEngine = HFPrefixCacheEngine
