from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pandas as pd

from dashboard.app import _continue_playing_cards, _difficulty_to_agent, _list_profiles, _live_game_launch_settings
from data.schemas import init_db
from impostor.player_profiles import PlayerProfileManager


def _tmp_db():
    handle = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    handle.close()
    return handle.name


def test_list_profiles_reads_player_profiles_table():
    db_path = _tmp_db()
    try:
        init_db(db_path)
        manager = PlayerProfileManager(db_path)
        manager.create("Zelda", player_id="z1")
        manager.create("alice", player_id="a1")

        profiles = _list_profiles(db_path)

        assert [profile.player_id for profile in profiles] == ["a1", "z1"]
        assert [profile.display_name for profile in profiles] == ["alice", "Zelda"]
    finally:
        os.unlink(db_path)


def test_difficulty_to_agent_uses_game_specific_mapping():
    assert _difficulty_to_agent("RPS+", "Balanced") == "adaptive_router"
    assert _difficulty_to_agent("Chess", "Hardcore") == "profile_counter"
    assert _difficulty_to_agent("Unknown", "Easy") == "random"


def test_continue_playing_cards_deduplicate_game_types():
    recent = pd.DataFrame(
        [
            {"game_type": "Chess", "opponent_name": "profile_counter", "agent_score": 1, "opponent_score": 0},
            {"game_type": "Chess", "opponent_name": "random", "agent_score": 0, "opponent_score": 1},
            {"game_type": "Pac-Man", "opponent_name": "random", "agent_score": 9, "opponent_score": 4},
            {"game_type": "RPS+", "opponent_name": "random", "agent_score": 3, "opponent_score": 2},
        ]
    )

    cards = _continue_playing_cards(recent)

    assert [card["title"] for card in cards] == ["Chess", "RPS+"]
    assert cards[0]["last_score"] == "1 - 0"


def test_live_game_launch_settings_open_selected_game():
    launch = _live_game_launch_settings("Chess", "Hardcore")

    assert launch["game_type"] == "Chess"
    assert launch["agent_name"] == "profile_counter"
    assert launch["agent2_name"] == "profile_counter"
    assert launch["play_mode"] == "human_bot1"
    assert launch["library_view"] == "Playable Now"


def test_standalone_arcade_page_is_integrated():
    root = Path(__file__).resolve().parents[1]
    arcade_html = root / "standalone_arcade.html"
    arcade_page = root / "dashboard" / "pages" / "arcade.py"

    assert arcade_html.exists()
    assert "Neon Strategy Arcade" in arcade_html.read_text(encoding="utf-8")
    assert "components.html" in arcade_page.read_text(encoding="utf-8")
