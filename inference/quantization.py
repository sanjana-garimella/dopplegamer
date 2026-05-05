"""Compatibility wrapper around serving quantization utilities."""

from __future__ import annotations

from serving.quantization import QuantConfig, load_quantized


def apply_quantization(model_name: str, mode: str = "fp16"):
    precision = mode.lower()
    cfg = QuantConfig(precision=precision)  # type: ignore[arg-type]
    return load_quantized(model_name, cfg)
