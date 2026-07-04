"""HF tool-interrupt ablation (not an InferCept cluster client).

Measures decode interrupted by a tool callback, then resumed. Engine name:
`hf_tool_interrupt`. For a real InferCept cluster set INFERCEPT_BASE_URL.
"""

from __future__ import annotations

from typing import Callable

from serving.base import InferenceEngine, InferenceResult, _Timer


ToolCallback = Callable[[str], str]


class HFToolInterruptEngine(InferenceEngine):
    name = "hf_tool_interrupt"
    supports_concurrent_clients = False
    supports_engine_batch = False

    def __init__(self, model_name: str, tool_callback: ToolCallback | None = None) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device).eval()
        self.tool_callback = tool_callback or (lambda s: "")

    def warmup(self) -> None:
        self.generate("warmup", max_new_tokens=2)

    def generate(self, prompt: str, max_new_tokens: int = 8) -> InferenceResult:
        import torch

        if max_new_tokens < 1:
            max_new_tokens = 1

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_tokens = int(inputs["input_ids"].shape[1])
        # Honor budget: for n=1 do a single phase only.
        if max_new_tokens == 1:
            phase1_tokens, phase2_tokens = 1, 0
        else:
            phase1_tokens = max(1, max_new_tokens // 2)
            phase2_tokens = max_new_tokens - phase1_tokens

        generated: list[int] = []
        with _Timer() as t_phase1, torch.no_grad():
            with _Timer() as t_prefill:
                out = self.model(**inputs, use_cache=True)
                next_id = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
                kv = out.past_key_values
            ttft_ms = t_prefill.elapsed_ms
            generated.append(int(next_id.item()))
            for _ in range(phase1_tokens - 1):
                out = self.model(input_ids=next_id, past_key_values=kv, use_cache=True)
                next_id = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
                kv = out.past_key_values
                generated.append(int(next_id.item()))
        phase1_text = self.tokenizer.decode(generated, skip_special_tokens=True)

        tool_ms = 0.0
        tool_output = ""
        phase2_ids: list[int] = []
        phase2_ms = 0.0
        resumed_prompt_tokens = 0
        if phase2_tokens > 0:
            with _Timer() as t_tool:
                tool_output = self.tool_callback(phase1_text)
            tool_ms = t_tool.elapsed_ms

            resumed_prompt = prompt + phase1_text + f"\nTOOL: {tool_output}\n"
            resumed = self.tokenizer(resumed_prompt, return_tensors="pt").to(self.device)
            resumed_prompt_tokens = int(resumed["input_ids"].shape[1])
            with _Timer() as t_phase2, torch.no_grad():
                out = self.model(**resumed, use_cache=True)
                next_id = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
                kv = out.past_key_values
                phase2_ids.append(int(next_id.item()))
                for _ in range(phase2_tokens - 1):
                    out = self.model(input_ids=next_id, past_key_values=kv, use_cache=True)
                    next_id = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
                    kv = out.past_key_values
                    phase2_ids.append(int(next_id.item()))
            phase2_ms = t_phase2.elapsed_ms

        output_tokens = len(generated) + len(phase2_ids)
        total = t_phase1.elapsed_ms + tool_ms + phase2_ms
        decode_ms = max(0.0, total - ttft_ms - tool_ms)
        tpot_ms = decode_ms / max(1, output_tokens - 1) if output_tokens > 1 else 0.0
        phase2_text = self.tokenizer.decode(phase2_ids, skip_special_tokens=True)
        text = phase1_text if not phase2_ids else f"{phase1_text}{phase2_text}"
        miss_tokens = prompt_tokens + resumed_prompt_tokens
        return InferenceResult(
            text=text,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            ttft_ms=ttft_ms,
            tpot_ms=tpot_ms,
            total_latency_ms=total,
            scheduling_overhead_ms=tool_ms if tool_ms > 0 else None,
            prefix_cache_hit_tokens=0,
            prefix_cache_miss_tokens=miss_tokens,
            extra={
                "actual_backend": "hf_tool_interrupt",
                "tool_output": tool_output,
                "tool_overhead_ms": tool_ms,
                "note": "HF ablation, not an InferCept cluster",
            },
        )


# Backward-compatible alias
InferCeptEngine = HFToolInterruptEngine
