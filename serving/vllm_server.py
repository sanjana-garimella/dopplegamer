"""vLLM serving wrapper. GPU required to actually run.

Two modes:
  1. Library mode: instantiate `vllm.LLM` directly and time `.generate()`.
  2. Server mode: launch the OpenAI-compatible server on `--port`.

The benchmark runner uses library mode for tight timing.
"""

from __future__ import annotations

import argparse

from serving.base import InferenceEngine, InferenceResult, _Timer


class VLLMEngine(InferenceEngine):
    name = "vllm"
    supports_concurrent_clients = False  # use generate_batch for continuous batching
    supports_engine_batch = True

    def __init__(
        self,
        model_name: str,
        dtype: str = "auto",
        gpu_memory_utilization: float = 0.85,
        enable_prefix_caching: bool = True,
    ) -> None:
        from vllm import LLM, SamplingParams

        self.SamplingParams = SamplingParams
        self.llm = LLM(
            model=model_name,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_prefix_caching=enable_prefix_caching,
        )
        self.model_name = model_name
        self.enable_prefix_caching = enable_prefix_caching
        self._prev_prompt_token_ids: list[int] = []

    def warmup(self) -> None:
        self.generate("warmup", max_new_tokens=1)

    def generate(self, prompt: str, max_new_tokens: int = 8) -> InferenceResult:
        return self.generate_batch([prompt], max_new_tokens=max_new_tokens)[0]

    def generate_batch(self, prompts: list[str], max_new_tokens: int = 8) -> list[InferenceResult]:
        params = self.SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
        with _Timer() as t:
            outputs = self.llm.generate(list(prompts), params)
        # Wall time is shared across the batch (continuous batching).
        per_wall = t.elapsed_ms / max(1, len(outputs))
        results: list[InferenceResult] = []
        for out in outputs:
            prompt_token_ids = list(out.prompt_token_ids)
            prompt_tokens = len(prompt_token_ids)
            output_tokens = len(out.outputs[0].token_ids)
            text = out.outputs[0].text
            ttft_ms = self._ttft_ms(out, per_wall, output_tokens)
            decode_ms = max(0.0, per_wall - ttft_ms)
            tpot_ms = decode_ms / max(1, output_tokens - 1) if output_tokens > 1 else 0.0
            hit, miss = self._prefix_hit_miss(prompt_token_ids)
            self._prev_prompt_token_ids = prompt_token_ids
            metrics = getattr(out, "metrics", None)
            results.append(
                InferenceResult(
                    text=text,
                    prompt_tokens=prompt_tokens,
                    output_tokens=output_tokens,
                    ttft_ms=ttft_ms,
                    tpot_ms=tpot_ms,
                    total_latency_ms=per_wall,
                    kv_cache_mb=self._estimate_kv_mb(prompt_tokens + output_tokens),
                    prefix_cache_hit_tokens=hit,
                    prefix_cache_miss_tokens=miss,
                    extra={
                        "actual_backend": "vllm",
                        "enable_prefix_caching": self.enable_prefix_caching,
                        "batch_size": len(prompts),
                        "batch_wall_ms": t.elapsed_ms,
                        "engine_metrics": metrics.__dict__ if metrics is not None else {},
                    },
                )
            )
        return results

    def _ttft_ms(self, out, elapsed_ms: float, output_tokens: int) -> float:
        metrics = getattr(out, "metrics", None)
        if metrics is None:
            return elapsed_ms / max(1, output_tokens)

        first = getattr(metrics, "first_token_time", None)
        arrival = getattr(metrics, "arrival_time", None)
        scheduled = getattr(metrics, "first_scheduled_time", None)
        # RequestMetrics fields are absolute timestamps (seconds); TTFT is a duration.
        if first is not None and arrival is not None:
            return max(0.0, (first - arrival) * 1000.0)
        if first is not None and scheduled is not None:
            return max(0.0, (first - scheduled) * 1000.0)
        # Some builds expose first_token_time as a duration already (small values).
        if first is not None and 0.0 <= float(first) < elapsed_ms / 1000.0 + 1.0:
            # Ambiguous: if first looks like a duration in seconds (< wall clock), use it.
            if float(first) < 60.0 and arrival is None:
                return float(first) * 1000.0
        return elapsed_ms / max(1, output_tokens)

    def _prefix_hit_miss(self, prompt_token_ids: list[int]) -> tuple[int, int]:
        if not self.enable_prefix_caching or not self._prev_prompt_token_ids:
            return 0, len(prompt_token_ids)
        shared = 0
        for a, b in zip(self._prev_prompt_token_ids, prompt_token_ids):
            if a != b:
                break
            shared += 1
        return shared, len(prompt_token_ids) - shared

    def _estimate_kv_mb(self, total_tokens: int) -> float:
        # Best-effort from HF-style config if vLLM exposes it; else 0.
        try:
            cfg = self.llm.llm_engine.model_config.hf_config
        except Exception:
            return 0.0
        n_layers = getattr(cfg, "num_hidden_layers", 0)
        n_heads = getattr(cfg, "num_attention_heads", 1)
        n_kv = getattr(cfg, "num_key_value_heads", n_heads)
        head_dim = getattr(cfg, "hidden_size", 0) // max(1, n_heads)
        bytes_per = 2
        return (2 * n_layers * n_kv * head_dim * total_tokens * bytes_per) / (1024 * 1024)


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch vLLM OpenAI-compatible server.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    import subprocess
    import sys

    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        args.model,
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
