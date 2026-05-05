"""vLLM serving wrapper. GPU required to actually run.

Two modes:
  1. Library mode: instantiate `vllm.LLM` directly and time `.generate()`.
  2. Server mode: launch the OpenAI-compatible server on `--port`.

The benchmark runner uses library mode for tight timing.
"""

from __future__ import annotations

import argparse
import time

from serving.base import InferenceEngine, InferenceResult, _Timer


class VLLMEngine(InferenceEngine):
    name = "vllm"

    def __init__(self, model_name: str, dtype: str = "auto", gpu_memory_utilization: float = 0.85) -> None:
        from vllm import LLM, SamplingParams

        self.SamplingParams = SamplingParams
        self.llm = LLM(model=model_name, dtype=dtype, gpu_memory_utilization=gpu_memory_utilization)
        self.model_name = model_name

    def generate(self, prompt: str, max_new_tokens: int = 8) -> InferenceResult:
        params = self.SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
        with _Timer() as t:
            outputs = self.llm.generate([prompt], params)
        out = outputs[0]
        prompt_tokens = len(out.prompt_token_ids)
        output_tokens = len(out.outputs[0].token_ids)
        text = out.outputs[0].text
        ttft_ms = getattr(out.metrics, "first_token_time", None)
        if ttft_ms is not None:
            ttft_ms = ttft_ms * 1000.0
        else:
            ttft_ms = t.elapsed_ms / max(1, output_tokens)
        return InferenceResult(
            text=text,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            ttft_ms=ttft_ms,
            tpot_ms=(t.elapsed_ms - ttft_ms) / max(1, output_tokens - 1) if output_tokens > 1 else 0.0,
            total_latency_ms=t.elapsed_ms,
            kv_cache_mb=0.0,
            extra={"engine_metrics": getattr(out, "metrics", None).__dict__ if getattr(out, "metrics", None) else {}},
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch vLLM OpenAI-compatible server.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    import subprocess
    import sys

    cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
           "--model", args.model, "--host", args.host, "--port", str(args.port)]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
