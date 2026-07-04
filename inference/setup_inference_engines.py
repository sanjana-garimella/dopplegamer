"""Inference engine registry/factory used by evaluation and API layers."""

from __future__ import annotations

import os
from dataclasses import dataclass

from serving.base import InferenceEngine
from serving.base import InferenceResult

# Primary engines for real-model runs.
PRIMARY_ENGINES = ("baseline", "vllm")
# HF ablations (not paper systems).
HF_ABLATION_ENGINES = ("hf_prefix_cache", "hf_tool_interrupt")
# Remote OpenAI-compatible endpoints only (require *_BASE_URL env vars).
REMOTE_ENGINES = ("preble", "infercept")

# Models allowed via HTTP / default allowlist. Extend with DOPPELGAMER_ALLOWED_MODELS.
_DEFAULT_ALLOWED_MODELS = frozenset({"mock", "distilgpt2", "gpt2"})


def allowed_models() -> frozenset[str]:
    extra = os.getenv("DOPPELGAMER_ALLOWED_MODELS", "")
    names = {m.strip() for m in extra.split(",") if m.strip()}
    return _DEFAULT_ALLOWED_MODELS | names


def validate_model_name(model_name: str) -> str:
    """Reject unknown models (blocks arbitrary hub IDs / local paths via API)."""
    name = (model_name or "").strip()
    if not name:
        raise ValueError("model_name must be non-empty")
    if name != "mock" and (".." in name or name.startswith("/") or name.startswith("~")):
        raise ValueError("model_name must not be a filesystem path")
    allowed = allowed_models()
    if name not in allowed:
        raise ValueError(
            f"model_name {name!r} is not allowlisted. "
            f"Allowed: {sorted(allowed)}. "
            f"Set DOPPELGAMER_ALLOWED_MODELS to extend."
        )
    return name


@dataclass
class EngineConfig:
    model_name: str = "distilgpt2"
    allow_fallback: bool = False
    enable_prefix_caching: bool = True


class EngineLoadError(RuntimeError):
    """Raised when a requested real engine cannot be constructed."""


def _known_engine_names() -> set[str]:
    return set(PRIMARY_ENGINES) | set(HF_ABLATION_ENGINES) | set(REMOTE_ENGINES)


def _remote_base_url(engine: str) -> str | None:
    keys = {
        "preble": ("PREBLE_BASE_URL", "DOPPELGAMER_PREBLE_URL"),
        "infercept": ("INFERCEPT_BASE_URL", "DOPPELGAMER_INFERCEPT_URL"),
    }
    for key in keys.get(engine, ()):
        val = os.getenv(key, "").strip()
        if val:
            return val
    return None


class _FallbackEngine(InferenceEngine):
    """Wraps a substitute backend and labels metrics so rows are not misread."""

    supports_concurrent_clients = False

    def __init__(self, requested: str, inner: InferenceEngine, reason: str) -> None:
        self.name = requested
        self._inner = inner
        self._reason = reason
        self._actual = getattr(inner, "name", type(inner).__name__)
        self.supports_concurrent_clients = bool(
            getattr(inner, "supports_concurrent_clients", False)
        )

    def warmup(self) -> None:
        warmup = getattr(self._inner, "warmup", None)
        if callable(warmup):
            warmup()

    def generate(self, prompt: str, max_new_tokens: int = 8) -> InferenceResult:
        result = self._inner.generate(prompt, max_new_tokens=max_new_tokens)
        extra = dict(result.extra or {})
        actual = extra.get("actual_backend") or self._actual
        extra.update(
            {
                "actual_backend": actual,
                "requested_engine": self.name,
                "fallback": True,
                "fallback_reason": self._reason,
            }
        )
        result.extra = extra
        return result

    def generate_batch(self, prompts: list[str], max_new_tokens: int = 8) -> list[InferenceResult]:
        return [self.generate(p, max_new_tokens=max_new_tokens) for p in prompts]


def _configure_resolved(name: str, cfg: EngineConfig) -> InferenceEngine:
    if name == "baseline":
        return configure_baseline(cfg)
    if name == "vllm":
        return configure_vllm(cfg)
    if name == "hf_prefix_cache":
        return configure_hf_prefix_cache(cfg)
    if name == "hf_tool_interrupt":
        return configure_hf_tool_interrupt(cfg)
    if name in REMOTE_ENGINES:
        return configure_remote(name, cfg)
    raise EngineLoadError(f"unknown engine `{name}`")


def setup_inference_engines(
    cfg: EngineConfig | None = None,
    engines: list[str] | None = None,
) -> dict[str, InferenceEngine]:
    """Build only the requested engines (lazy). Default: primary engines only."""
    cfg = cfg or EngineConfig()
    if engines is None:
        requested = (
            list(PRIMARY_ENGINES) + list(HF_ABLATION_ENGINES) + list(REMOTE_ENGINES)
            if cfg.model_name == "mock"
            else list(PRIMARY_ENGINES)
        )
    else:
        requested = list(engines)
    if not requested:
        return {}

    known = _known_engine_names()
    for n in requested:
        if n not in known:
            raise EngineLoadError(
                f"unknown engine `{n}`. Known: {sorted(known)}. "
                f"HF ablations: {HF_ABLATION_ENGINES}. "
                f"Remote (need BASE_URL): {REMOTE_ENGINES}."
            )

    if cfg.model_name == "mock":
        return {name: _MockEngine(name) for name in requested}

    return {name: _configure_resolved(name, cfg) for name in requested}


def configure_baseline(cfg: EngineConfig | str) -> InferenceEngine:
    if isinstance(cfg, str):
        cfg = EngineConfig(model_name=cfg)
    try:
        from serving.baseline_hf import HFBaselineEngine

        return HFBaselineEngine(model_name=cfg.model_name)
    except Exception as exc:
        if cfg.allow_fallback:
            eng = _MockEngine("baseline")
            eng._fallback_reason = str(exc)
            return _FallbackEngine("baseline", eng, str(exc))
        raise EngineLoadError(f"baseline engine failed to load for model={cfg.model_name!r}: {exc}") from exc


def configure_vllm(cfg: EngineConfig | str) -> InferenceEngine:
    if isinstance(cfg, str):
        cfg = EngineConfig(model_name=cfg)
    try:
        from serving.vllm_server import VLLMEngine

        return VLLMEngine(
            model_name=cfg.model_name,
            enable_prefix_caching=cfg.enable_prefix_caching,
        )
    except Exception as exc:
        if cfg.allow_fallback:
            inner = configure_baseline(cfg)
            return _FallbackEngine("vllm", inner, str(exc))
        raise EngineLoadError(f"vllm engine failed to load for model={cfg.model_name!r}: {exc}") from exc


def configure_hf_prefix_cache(cfg: EngineConfig | str) -> InferenceEngine:
    if isinstance(cfg, str):
        cfg = EngineConfig(model_name=cfg)
    try:
        from serving.preble_benchmark import HFPrefixCacheEngine

        return HFPrefixCacheEngine(model_name=cfg.model_name)
    except Exception as exc:
        if cfg.allow_fallback:
            eng = _MockEngine("hf_prefix_cache")
            eng._fallback_reason = str(exc)
            return _FallbackEngine("hf_prefix_cache", eng, str(exc))
        raise EngineLoadError(
            f"hf_prefix_cache engine failed to load for model={cfg.model_name!r}: {exc}"
        ) from exc


def configure_hf_tool_interrupt(cfg: EngineConfig | str) -> InferenceEngine:
    if isinstance(cfg, str):
        cfg = EngineConfig(model_name=cfg)
    try:
        from serving.infercept_benchmark import HFToolInterruptEngine

        return HFToolInterruptEngine(model_name=cfg.model_name)
    except Exception as exc:
        if cfg.allow_fallback:
            eng = _MockEngine("hf_tool_interrupt")
            eng._fallback_reason = str(exc)
            return _FallbackEngine("hf_tool_interrupt", eng, str(exc))
        raise EngineLoadError(
            f"hf_tool_interrupt engine failed to load for model={cfg.model_name!r}: {exc}"
        ) from exc


def configure_remote(name: str, cfg: EngineConfig | str) -> InferenceEngine:
    """Preble / InferCept only via remote OpenAI-compatible endpoints."""
    if isinstance(cfg, str):
        cfg = EngineConfig(model_name=cfg)
    base_url = _remote_base_url(name)
    if not base_url:
        env_hint = (
            "PREBLE_BASE_URL or DOPPELGAMER_PREBLE_URL"
            if name == "preble"
            else "INFERCEPT_BASE_URL or DOPPELGAMER_INFERCEPT_URL"
        )
        ablation = "hf_prefix_cache" if name == "preble" else "hf_tool_interrupt"
        raise EngineLoadError(
            f"`{name}` requires a remote cluster URL ({env_hint}). "
            f"For a local HF ablation use `{ablation}` instead; "
            f"do not report `{ablation}` as {name}."
        )
    try:
        from serving.openai_compat import OpenAICompatEngine

        api_key = os.getenv("DOPPELGAMER_REMOTE_API_KEY") or os.getenv("OPENAI_API_KEY")
        return OpenAICompatEngine(
            name=name,
            base_url=base_url,
            model_name=cfg.model_name,
            api_key=api_key,
        )
    except Exception as exc:
        if cfg.allow_fallback:
            eng = _MockEngine(name)
            eng._fallback_reason = str(exc)
            return _FallbackEngine(name, eng, str(exc))
        raise EngineLoadError(f"{name} remote engine failed: {exc}") from exc


# Explicit names (no silent alias to HF ablations).
configure_preble = lambda cfg: configure_remote("preble", cfg)  # noqa: E731
configure_infercept = lambda cfg: configure_remote("infercept", cfg)  # noqa: E731


class _MockEngine(InferenceEngine):
    # Sequential only: prefix set is not thread-safe and batch is not real.
    supports_concurrent_clients = False
    supports_engine_batch = False

    def __init__(self, name: str) -> None:
        self.name = name
        self._seen_prefixes: set[str] = set()
        self._fallback_reason: str | None = None

    def generate(self, prompt: str, max_new_tokens: int = 8) -> InferenceResult:
        prompt_tokens = max(1, len(prompt.split()))
        output_tokens = max_new_tokens
        ttft = 8.0
        tpot = 1.5
        prefix_key = prompt[:50]
        if prefix_key in self._seen_prefixes:
            hit = int(prompt_tokens * 0.6)
            miss = prompt_tokens - hit
        else:
            hit = 0
            miss = prompt_tokens
            self._seen_prefixes.add(prefix_key)
        total = ttft + tpot * max(1, output_tokens - 1)
        extra: dict = {"actual_backend": "mock", "requested_engine": self.name}
        if self._fallback_reason:
            extra["fallback"] = True
            extra["fallback_reason"] = self._fallback_reason
        return InferenceResult(
            text="ROCK",
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            ttft_ms=ttft,
            tpot_ms=tpot,
            total_latency_ms=total,
            kv_cache_mb=0.25 * prompt_tokens,
            scheduling_overhead_ms=None,
            prefix_cache_hit_tokens=hit,
            prefix_cache_miss_tokens=miss,
            extra=extra,
        )

    def generate_batch(self, prompts: list[str], max_new_tokens: int = 8) -> list[InferenceResult]:
        return [self.generate(p, max_new_tokens=max_new_tokens) for p in prompts]
