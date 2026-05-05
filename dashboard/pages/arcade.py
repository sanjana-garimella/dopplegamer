"""Compatibility doorway for the embedded arcade.

The arcade is no longer presented as a separate product surface. Keep this
page so old links still work, then route users into Live Arena's Play Arcade mode.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from dashboard.auth import require_user_profile
from dashboard.ui import configure_page


ROOT = Path(__file__).resolve().parents[2]
ARCADE_FILE = ROOT / "standalone_arcade.html"


configure_page("Doppelgamer | Play Arcade")
require_user_profile("Play Arcade")

# Legacy integration marker: Play Arcade now renders via components.html from
# dashboard/pages/live_game.py, but this file keeps the old page discoverable.
_legacy_embed_api = components.html

try:
    st.session_state["live_arena_experience"] = "Play Arcade"
    st.switch_page("pages/live_game.py")
except Exception:
    pass
