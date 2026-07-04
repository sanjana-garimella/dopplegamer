"""OpenAI-compatible HTTP client for remote serving endpoints.

Used when PREBLE_BASE_URL / INFERCEPT_BASE_URL (or DOPPELGAMER_* variants) point
at a real cluster. This is not an HF ablation.
"""

from __future__ import annotations

import time
from typing import Any

from serving.base import InferenceEngine, InferenceResult, _Timer


class OpenAICompatEngine(InferenceEngine):
    """POST /v1/completions (or /v1/chat/completions) against a remote server."""

    supports_concurrent_clients = True

    def __init__(
        self,
        *,
        name: str,
        base_url: str,
        model_name: str,
        api_key: str | None = None,
        use_chat: bool = False,
        timeout_s: float = 120.0,
    ) -> None:
        import urllib.parse

        self.name = name
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.api_key = api_key or "EMPTY"
        self.use_chat = use_chat
        self.timeout_s = timeout_s
        # Validate URL shape early.
        parsed = urllib.parse.urlparse(self.base_url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError(f"invalid base_url for {name}: {base_url!r}")

    def warmup(self) -> None:
        self.generate("warmup", max_new_tokens=1)

    def generate(self, prompt: str, max_new_tokens: int = 8) -> InferenceResult:
        import json
        import urllib.error
        import urllib.request

        if self.use_chat:
            path = "/v1/chat/completions"
            body: dict[str, Any] = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_new_tokens,
                "temperature": 0.0,
            }
        else:
            path = "/v1/completions"
            body = {
                "model": self.model_name,
                "prompt": prompt,
                "max_tokens": max_new_tokens,
                "temperature": 0.0,
            }

        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            self.base_url + path,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with _Timer() as t:
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"{self.name} HTTP {exc.code}: {detail[:200]}") from exc
        elapsed_ms = t.elapsed_ms

        text, prompt_tokens, output_tokens = self._parse_payload(payload, prompt)
        # Remote APIs rarely expose true TTFT; use total/n as a conservative bound.
        ttft_ms = elapsed_ms / max(1, output_tokens)
        tpot_ms = (elapsed_ms - ttft_ms) / max(1, output_tokens - 1) if output_tokens > 1 else 0.0
        return InferenceResult(
            text=text,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            ttft_ms=ttft_ms,
            tpot_ms=tpot_ms,
            total_latency_ms=elapsed_ms,
            scheduling_overhead_ms=None,
            prefix_cache_hit_tokens=0,
            prefix_cache_miss_tokens=prompt_tokens,
            extra={
                "actual_backend": self.name,
                "base_url": self.base_url,
                "remote": True,
                "ttft_estimated": True,
                "latency_estimated": True,
            },
        )

    def _parse_payload(self, payload: dict[str, Any], prompt: str) -> tuple[str, int, int]:
        usage = payload.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens") or max(1, len(prompt.split())))
        output_tokens = int(usage.get("completion_tokens") or 0)
        choices = payload.get("choices") or []
        if not choices:
            return "", prompt_tokens, max(1, output_tokens)
        choice = choices[0]
        if "text" in choice:
            text = str(choice.get("text") or "")
        else:
            msg = choice.get("message") or {}
            text = str(msg.get("content") or "")
        if output_tokens <= 0:
            output_tokens = max(1, len(text.split()))
        return text, prompt_tokens, output_tokens
