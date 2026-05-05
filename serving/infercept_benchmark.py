"""InferCept benchmark scaffold.

InferCept (WukLab/UCSD, ICML 2024) optimizes serving for augmented LLMs —
specifically the pattern of decoding being interrupted by tool calls / RAG
retrieval. This module measures the overhead of the interrupt-resume cycle.

Real InferCept integration requires the published artifact. This stub uses
HF generation with a programmatic interrupt to capture the same metric:
total latency under tool-call interruptions vs. uninterrupted generation.
"""

from __future__ import annotations

import time
from typing import Callable

from serving.base import InferenceEngine, InferenceResult, _Timer


ToolCallback = Callable[[str], str]


class InferCeptEngine(InferenceEngine):
    name = "infercept"

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

    def generate(self, prompt: str, max_new_tokens: int = 8) -> InferenceResult:
        import torch

        # Phase 1: generate up to a pseudo "tool" marker.
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_tokens = int(inputs["input_ids"].shape[1])

        with _Timer() as t_phase1, torch.no_grad():
            phase1 = self.model.generate(
                **inputs,
                max_new_tokens=max(1, max_new_tokens // 2),
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True,
            )
        phase1_text = self.tokenizer.decode(phase1[0][prompt_tokens:], skip_special_tokens=True)

        # Tool call (the "intercept").
        with _Timer() as t_tool:
            tool_output = self.tool_callback(phase1_text)

        # Phase 2: resume generation with tool output appended.
        resumed_prompt = prompt + phase1_text + f"\nTOOL: {tool_output}\n"
        resumed_inputs = self.tokenizer(resumed_prompt, return_tensors="pt").to(self.device)
        with _Timer() as t_phase2, torch.no_grad():
            phase2 = self.model.generate(
                **resumed_inputs,
                max_new_tokens=max_new_tokens // 2,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True,
            )

        full_tokens = int(phase1.shape[1] - prompt_tokens) + int(phase2.shape[1] - resumed_inputs["input_ids"].shape[1])
        total = t_phase1.elapsed_ms + t_tool.elapsed_ms + t_phase2.elapsed_ms
        # TTFT is the time until the first token is produced (end of phase1 prefill),
        # not divided by token count (that would be TPOT).
        ttft = t_phase1.elapsed_ms
        decode_tokens = max(1, full_tokens - 1)
        tpot = (t_phase2.elapsed_ms) / decode_tokens
        text = self.tokenizer.decode(phase2[0][resumed_inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return InferenceResult(
            text=text,
            prompt_tokens=prompt_tokens,
            output_tokens=full_tokens,
            ttft_ms=ttft,
            tpot_ms=tpot,
            total_latency_ms=total,
            scheduling_overhead_ms=t_tool.elapsed_ms,
            extra={"tool_output": tool_output},
        )
