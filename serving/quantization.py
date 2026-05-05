"""Quantization experiments: FP16 vs INT4 (BitsAndBytes) vs FP8 where supported.

Measures latency, memory, and downstream agent win-rate degradation. Run via
the evaluation runner with `--quantization int4` or call `load_quantized` directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


Precision = Literal["fp16", "fp32", "bf16", "int8", "int4"]


@dataclass
class QuantConfig:
    precision: Precision = "fp16"


def load_quantized(model_name: str, cfg: QuantConfig):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    kwargs: dict = {}
    if cfg.precision in ("int4", "int8"):
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=cfg.precision == "int4",
            load_in_8bit=cfg.precision == "int8",
            bnb_4bit_compute_dtype="float16",
            bnb_4bit_quant_type="nf4",
        )
    elif cfg.precision == "fp16":
        import torch

        kwargs["torch_dtype"] = torch.float16
    elif cfg.precision == "bf16":
        import torch

        kwargs["torch_dtype"] = torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    return model, tokenizer


def model_memory_mb(model) -> float:
    total = 0
    for p in model.parameters():
        total += p.numel() * p.element_size()
    return total / (1024 * 1024)
