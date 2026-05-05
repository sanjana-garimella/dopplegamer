"""Utility launcher for the Streamlit dashboard."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def launch_dashboard(db_path: str = "data/game_data.db") -> None:
    app = Path("dashboard/app.py")
    cmd = [sys.executable, "-m", "streamlit", "run", str(app), "--", "--db-path", db_path]
    subprocess.run(cmd, check=True)