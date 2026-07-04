"""FastAPI entrypoint for Doppelgamer orchestration."""

from __future__ import annotations

import os
import secrets

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field, field_validator

from data.features import CANONICAL_BASELINE_BATTERY
from evaluation.runner import run_benchmark
from inference.setup_inference_engines import validate_model_name

app = FastAPI(title="Doppelgamer API", version="0.1.0")

_MAX_LIST = 32
_API_KEY = os.getenv("DOPPELGAMER_API_KEY", "").strip()


class BenchmarkRequest(BaseModel):
    rounds: int = Field(default=100, ge=1, le=5000)
    engines: list[str] = Field(default_factory=lambda: ["baseline", "vllm"], max_length=_MAX_LIST)
    agents: list[str] = Field(
        default_factory=lambda: list(CANONICAL_BASELINE_BATTERY),
        max_length=_MAX_LIST,
    )
    games: list[str] = Field(default_factory=lambda: ["RPS+"], max_length=_MAX_LIST)
    n_seeds: int = Field(default=1, ge=1, le=50)
    # Default mock only; real models must be allowlisted (see validate_model_name).
    model_name: str = "mock"
    allow_fallback: bool = False
    db_path: str = os.getenv("DOPPELGAMER_DB_PATH", "data/game_data.db")

    @field_validator("db_path")
    @classmethod
    def _safe_db_path(cls, v: str) -> str:
        if ".." in v or v.startswith("/"):
            raise ValueError("db_path must be a relative path without '..'")
        return v

    @field_validator("model_name")
    @classmethod
    def _allowed_model(cls, v: str) -> str:
        return validate_model_name(v)

    @field_validator("engines", "agents", "games")
    @classmethod
    def _nonempty_items(cls, v: list[str]) -> list[str]:
        for item in v:
            if not isinstance(item, str) or not item.strip():
                raise ValueError("list entries must be non-empty strings")
        return v


def _check_api_key(x_api_key: str | None) -> None:
    """When DOPPELGAMER_API_KEY is set, require a matching X-API-Key header."""
    if not _API_KEY:
        return
    provided = (x_api_key or "").strip()
    if not provided or not secrets.compare_digest(provided, _API_KEY):
        raise HTTPException(status_code=401, detail="invalid or missing API key")


@app.get("/")
async def root() -> dict[str, str]:
    return {"message": "Doppelgamer API is running."}


@app.post("/benchmark")
async def benchmark(
    req: BenchmarkRequest,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> dict:
    _check_api_key(x_api_key)
    # Real models via HTTP require an API key even if allowlist includes them.
    if req.model_name != "mock" and not _API_KEY:
        raise HTTPException(
            status_code=403,
            detail="real model_name requires DOPPELGAMER_API_KEY to be configured",
        )
    try:
        result = run_benchmark(
            rounds=req.rounds,
            engines=req.engines,
            agents=req.agents,
            db_path=req.db_path,
            model_name=req.model_name,
            n_seeds=req.n_seeds,
            games=req.games,
            allow_fallback=req.allow_fallback,
        )
        return {"status": "ok", "result": result}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception:
        # Do not echo loader paths, stack traces, or hub errors to clients.
        raise HTTPException(status_code=500, detail="benchmark failed") from None
