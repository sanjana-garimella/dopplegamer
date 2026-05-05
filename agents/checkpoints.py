from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS_DIR = ROOT / "checkpoints"


def _model_path_exists(path: Path) -> bool:
    if path.is_dir():
        return True
    if path.exists():
        return True
    return path.with_suffix(".zip").exists()


def resolve_checkpoint(agent_name: str) -> Path | None:
    """Return the best local checkpoint path for an agent, if one exists."""
    aliases = {
        "rl": "ppo",
        "ppo": "ppo",
        "bc_rl": "bc_rl",
        "bcrl": "bc_rl",
        "sft": "sft",
    }
    key = aliases.get(agent_name, agent_name)

    candidates: dict[str, list[Path]] = {
        "ppo": [
            CHECKPOINTS_DIR / "ppo_real",
            CHECKPOINTS_DIR / "ppo_best",
            CHECKPOINTS_DIR / "ppo_best_best" / "best_model",
        ],
        "bc_rl": [
            CHECKPOINTS_DIR / "bc_rl_real",
            CHECKPOINTS_DIR / "bc_rl",
        ],
        "sft": [
            CHECKPOINTS_DIR / "sft_real",
            CHECKPOINTS_DIR / "sft_best",
        ],
    }

    for candidate in candidates.get(key, []):
        if _model_path_exists(candidate):
            return candidate
    return None


def checkpoint_status(agent_name: str) -> tuple[bool, str | None]:
    path = resolve_checkpoint(agent_name)
    if path is None:
        return False, None
    return True, str(path)
