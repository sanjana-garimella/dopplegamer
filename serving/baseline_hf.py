"""HuggingFace baseline serving — single-stream autoregressive generation.

Token timings use one prefill + incremental decode pass (no double generate).
KV cache size is approximated from the model config and number of cached tokens.
"""

from __future__ import annotations

from serving.base import InferenceEngine, InferenceResult, _Timer


class HFBaselineEngine(InferenceEngine):
    name = "baseline"
    supports_concurrent_clients = False

    def __init__(self, model_name: str, device: str = "auto", dtype: str = "auto") -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        kwargs = {}
        if dtype != "auto":
            kwargs["torch_dtype"] = getattr(torch, dtype)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else "cpu"
        self.model.to(self.device).eval()

    def warmup(self) -> None:
        self.generate("warmup", max_new_tokens=1)

    def generate(self, prompt: str, max_new_tokens: int = 8) -> InferenceResult:
        import torch

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_tokens = int(inputs["input_ids"].shape[1])
        generated_ids: list[int] = []

        with _Timer() as t_total, torch.no_grad():
            # Prefill: one forward over the full prompt, first token.
            with _Timer() as t_prefill:
                out = self.model(**inputs, use_cache=True)
                next_id = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
                kv = out.past_key_values
            ttft_ms = t_prefill.elapsed_ms
            generated_ids.append(int(next_id.item()))

            decode_ms = 0.0
            for _ in range(max(0, max_new_tokens - 1)):
                with _Timer() as t_tok:
                    out = self.model(input_ids=next_id, past_key_values=kv, use_cache=True)
                    next_id = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
                    kv = out.past_key_values
                decode_ms += t_tok.elapsed_ms
                generated_ids.append(int(next_id.item()))

        output_tokens = len(generated_ids)
        tpot_ms = decode_ms / max(1, output_tokens - 1) if output_tokens > 1 else 0.0
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        total_latency_ms = t_total.elapsed_ms

        return InferenceResult(
            text=text,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            ttft_ms=ttft_ms,
            tpot_ms=tpot_ms,
            total_latency_ms=total_latency_ms,
            kv_cache_mb=self._estimate_kv_mb(prompt_tokens + output_tokens),
            prefix_cache_hit_tokens=0,
            prefix_cache_miss_tokens=prompt_tokens,
            extra={"actual_backend": "huggingface", "device": self.device},
        )

    def _estimate_kv_mb(self, total_tokens: int) -> float:
        cfg = self.model.config
        n_layers = getattr(cfg, "num_hidden_layers", 0)
        n_kv = getattr(cfg, "num_key_value_heads", getattr(cfg, "num_attention_heads", 1))
        head_dim = getattr(cfg, "hidden_size", 0) // max(1, getattr(cfg, "num_attention_heads", 1))
        bytes_per = 2  # fp16
        bytes_total = 2 * n_layers * n_kv * head_dim * total_tokens * bytes_per
        return bytes_total / (1024 * 1024)
