"""Shared dashboard/runtime configuration."""

from __future__ import annotations

import os
from pathlib import Path


def db_path(default: str = "data/game_data.db") -> Path:
    value = os.getenv("DOPPELGAMER_DB_PATH", default).strip() or default
    return Path(value)


def public_base_url(default: str = "") -> str:
    return os.getenv("DOPPELGAMER_PUBLIC_BASE_URL", default).strip()
