"""Simple data manager for per-step trajectory logging."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
from typing import Any


@dataclass
class StepRecord:
    agent: str
    episode_id: str
    step: int
    action: int
    reward: float
    done: bool
    info: dict[str, Any]


class DataManager:
    def __init__(self) -> None:
        self.records: list[StepRecord] = []

    def collect_data(
        self,
        *,
        agent: str,
        episode_id: str,
        step: int,
        action: int,
        reward: float,
        done: bool,
        info: dict[str, Any] | None = None,
    ) -> None:
        self.records.append(
            StepRecord(
                agent=agent,
                episode_id=episode_id,
                step=step,
                action=int(action),
                reward=float(reward),
                done=bool(done),
                info=info or {},
            )
        )

    def process_data(self) -> dict[str, Any]:
        if not self.records:
            return {"steps": 0, "episodes": 0, "avg_reward": 0.0}
        total_reward = sum(r.reward for r in self.records)
        episodes = len({r.episode_id for r in self.records})
        return {
            "steps": len(self.records),
            "episodes": episodes,
            "avg_reward": total_reward / len(self.records),
        }

    def save_to_storage(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps([asdict(r) for r in self.records], indent=2))
        return out