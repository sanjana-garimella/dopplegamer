"""FastAPI entrypoint for AgentBench orchestration."""

from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from evaluation.runner import run_benchmark

app = FastAPI(title="AgentBench API", version="0.1.0")


class BenchmarkRequest(BaseModel):
    rounds: int = Field(default=100, ge=1, le=5000)
    engines: list[str] = Field(default_factory=lambda: ["baseline", "vllm", "preble", "infercept"])
    agents: list[str] = Field(default_factory=lambda: ["sft", "rl", "bcrl", "agentic"])
    db_path: str = os.getenv("DOPPELGAMER_DB_PATH", "data/game_data.db")

    @field_validator("db_path")
    @classmethod
    def _safe_db_path(cls, v: str) -> str:
        from pathlib import PurePosixPath
        if ".." in v or v.startswith("/"):
            raise ValueError("db_path must be a relative path without '..'")
        return v


@app.get("/")
async def root() -> dict[str, str]:
    return {"message": "AgentBench API is running."}


@app.post("/benchmark")
async def benchmark(req: BenchmarkRequest) -> dict:
    try:
        result = run_benchmark(
            rounds=req.rounds,
            engines=req.engines,
            agents=req.agents,
            db_path=req.db_path,
        )
        return {"status": "ok", "result": result}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"benchmark failed: {exc}") from exc
