from __future__ import annotations

# Live Human vs Impostor game interface.

import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import sys
import copy
import html
from pathlib import Path
try:
    import plotly.express as px
except Exception:  # pragma: no cover - optional UI dependency
    px = None
try:
    import plotly.graph_objects as go
except Exception:  # pragma: no cover - optional UI dependency
    go = None

# Ensure project root is in PYTHONPATH
root = Path(__file__).parent.parent.parent
if str(root) not in sys.path:
    sys.path.append(str(root))

from environments.rps_plus import RPSPlusEnv, Move, N_MOVES
from environments.tic_tac_toe import TicTacToeEnv
from environments.connect_four import ConnectFourEnv
from environments.chess_env import ChessEnv
from environments.othello import OthelloEnv
from environments.checkers import CheckersEnv
from environments.war import WarEnv
from environments.gomoku import GomokuEnv
from environments.nim import NimEnv
from environments.future_games import FUTURE_GAME_ENVS
from agents import AGENT_REGISTRY
from data.collector import insert_game
from data.schemas import connect, init_db, init_extended_db
from data.features import contextual_hint, recency_bias_warning
from dashboard.auth import require_user_profile
from dashboard.config import db_path as configured_db_path
from dashboard.ui import configure_page, render_sidebar_nav, style_plotly_figure
from environments.game_specs import CHALLENGE_PACKS, CLONE_LADDER_RPS, GAME_CHALLENGES, GAME_HINTS
from environments.utils import history_to_records
from impostor.player_profiles import PlayerProfileManager
from agents.profile_counter import ProfileCounterAgent
from agents.adaptive_router import AdaptiveRouterAgent
from agents.impostor.ngram import NGramImpostor
from agents.impostor.lstm import LSTMImpostor
from agents.impostor.trainer import ImpostorTrainer
from impostor.experiments import create_blind_match_schedule, persist_blind_match_schedule, persist_counterfactual_replay
from impostor.metrics import standardized_detection_prompt, summarize_surprisal_history
import uuid

DEFAULT_DB = configured_db_path()
ARCADE_FILE = root / "standalone_arcade.html"
DEFAULT_HINTS = True
DEFAULT_LADDER_GAME = "RPS+"

PLAY_MODES = {
    "Play a Bot": "human_bot1",
    "Try Another Bot": "human_bot2",
    "Two Players": "human_human",
    "Watch Bots": "bot_bot",
}

PLAY_MODE_HELP = {
    "Play a Bot": "Pick one opponent and jump into the match.",
    "Try Another Bot": "Use the comparison bot slot without changing your first setup.",
    "Two Players": "Two people share this screen and take turns.",
    "Watch Bots": "Choose two bots and advance the match yourself.",
}

FUTURE_GAME_LABELS = {
    # Kept for research imports, but single-player games are no longer exposed
    # in the live arena picker.
}

STANDARD_GAMES = [
    "RPS+",
    "Tic-Tac-Toe",
    "Connect Four",
    "Chess",
    "Othello",
    "Checkers",
    "Gomoku",
    "Nim",
]

PROTOTYPE_GAMES = [
    "War",
]

ALL_GAMES = STANDARD_GAMES + [game for game in PROTOTYPE_GAMES if game not in STANDARD_GAMES]

GAME_ICONS = {
    "RPS+": "⚡",
    "Tic-Tac-Toe": "✖️",
    "Connect Four": "🟡",
    "Chess": "♟️",
    "Othello": "⚫",
    "Checkers": "🔴",
    "Gomoku": "●",
    "Nim": "▦",
    "War": "🃏",
    "2048": "🔢",
    "Wordle": "🔤",
    "Sudoku": "🧩",
    "Pac-Man": "🟡",
    "Candy Crush": "🍬",
    "Minecraft": "⛏️",
    "Among Us": "🛰️",
    "Clash Royale": "🏰",
    "Flappy Bird": "🐤",
    "Ludo": "🎲",
    "UNO": "🟦",
    "Scrabble": "🔡",
    "Monopoly": "💵",
    "Penalty Shootout": "🥅",
    "Cricket Strategy": "🏏",
}

GAME_AGENT_OPTIONS = {
    "RPS+": ["random", "heuristic", "optimal", "sft", "rl", "ppo", "bc_rl", "bcrl", "agentic", "profile_counter", "adaptive_router", "ngram", "lstm", "mixture"],
    "Tic-Tac-Toe": ["random", "profile_counter"],
    "Connect Four": ["random", "profile_counter"],
    "Chess": ["random", "profile_counter"],
    "Othello": ["random", "profile_counter"],
    "Checkers": ["random", "profile_counter"],
    "Gomoku": ["random", "profile_counter"],
    "Nim": ["random", "profile_counter"],
    "War": ["random", "profile_counter"],
}

configure_page("Doppelgamer | Live Arena")
require_user_profile("Live Arena")

def _build_chess_css() -> str:
    chess_css = []
    for rank in range(8):
        for file in range(8):
            sq = (rank * 8) + file
            is_light = (rank + file) % 2 == 1
            bg = "#f0d9b5" if is_light else "#b58863"
            color = "#5d4037" if is_light else "#f0d9b5"
            chess_css.append(f"div.st-key-sq_{sq} button {{ background-color: {bg} !important; color: {color} !important; }}")
    chess_css.append("div.chess-selected button { background-color: #fff176 !important; border: 3px solid #fbc02d !important; z-index: 10; }")
    chess_css.append("div.chess-legal button { background-color: #a5d6a7 !important; color: #2e7d32 !important; }")
    return " ".join(chess_css)


_CHESS_CSS = _build_chess_css()


def _build_board_button_css() -> str:
    css = []
    ttt_tiles = [
        ("#0891b2", "#22d3ee"),
        ("#7c3aed", "#c084fc"),
        ("#db2777", "#f9a8d4"),
        ("#ea580c", "#fdba74"),
        ("#0d9488", "#5eead4"),
        ("#4f46e5", "#a5b4fc"),
        ("#16a34a", "#86efac"),
        ("#be123c", "#fda4af"),
        ("#ca8a04", "#fde68a"),
    ]
    for idx, (deep, glow) in enumerate(ttt_tiles):
        css.append(
            f"""
            div.st-key-ttt_{idx} button,
            div.st-key-ttt_{idx} button:disabled {{
                background:
                    radial-gradient(circle at 28% 22%, rgba(255,255,255,0.42), transparent 20%),
                    linear-gradient(145deg, {glow} 0%, {deep} 100%) !important;
                color: #ffffff !important;
                border-color: rgba(255,255,255,0.28) !important;
                text-shadow: 0 2px 10px rgba(0,0,0,0.32) !important;
            }}
            """
        )

    for idx in range(81):
        row, col = divmod(idx, 9)
        base = "rgba(20,184,166,0.38)" if (row + col) % 2 == 0 else "rgba(59,130,246,0.34)"
        accent = "rgba(250,204,21,0.24)" if row in {0, 4, 8} or col in {0, 4, 8} else "rgba(244,114,182,0.16)"
        css.append(
            f"""
            div.st-key-gomoku_{idx} button,
            div.st-key-gomoku_{idx} button:disabled {{
                background:
                    radial-gradient(circle at 50% 50%, rgba(15,23,42,0.16), transparent 28%),
                    linear-gradient(145deg, {base}, {accent}) !important;
                border-color: rgba(255,255,255,0.18) !important;
                color: var(--arena-ink) !important;
            }}
            """
        )

    for idx in range(64):
        row, col = divmod(idx, 8)
        othello_bg = "rgba(16,185,129,0.42)" if (row + col) % 2 == 0 else "rgba(20,184,166,0.32)"
        checkers_bg = "linear-gradient(145deg, rgba(251,146,60,0.62), rgba(168,85,247,0.46))" if (row + col) % 2 else "linear-gradient(145deg, rgba(255,237,213,0.30), rgba(254,202,202,0.20))"
        chess_bg = "linear-gradient(145deg, #fde68a, #f59e0b)" if (row + col) % 2 else "linear-gradient(145deg, #0f766e, #164e63)"
        chess_color = "#1f2937" if (row + col) % 2 else "#f8fafc"
        css.append(
            f"""
            div.st-key-oth_{idx} button,
            div.st-key-oth_{idx} button:disabled {{
                background:
                    radial-gradient(circle at 48% 44%, rgba(255,255,255,0.22), transparent 22%),
                    linear-gradient(145deg, {othello_bg}, rgba(6,95,70,0.50)) !important;
                color: #ffffff !important;
                border-color: rgba(255,255,255,0.18) !important;
            }}
            div.st-key-chk_{idx} button,
            div.st-key-chk_{idx} button:disabled,
            div.st-key-chk_to_{idx} button,
            div.st-key-chk_to_{idx} button:disabled {{
                background: {checkers_bg} !important;
                color: #fff7ed !important;
                border-color: rgba(255,255,255,0.18) !important;
                text-shadow: 0 2px 8px rgba(0,0,0,0.38) !important;
            }}
            div.st-key-chess_sq_{idx} button,
            div.st-key-chess_sq_{idx} button:disabled {{
                background: {chess_bg} !important;
                color: {chess_color} !important;
                border-color: rgba(255,255,255,0.20) !important;
                text-shadow: 0 1px 5px rgba(0,0,0,0.22) !important;
            }}
            """
        )
    return "\n".join(css)


_BOARD_BUTTON_CSS = _build_board_button_css()


def _render_arcade_mode() -> None:
    st.markdown(
        """
        <style>
          .block-container { max-width: 1560px; padding-top: 0.75rem; }
          .integrated-arcade-shell {
            border: 1px solid rgba(0,255,157,.22);
            background:
              linear-gradient(135deg, rgba(0,255,157,.08), rgba(127,119,221,.07)),
              rgba(10,10,20,.78);
            border-radius: 14px;
            padding: 18px 20px;
            margin-bottom: 14px;
            box-shadow: 0 24px 80px rgba(0,0,0,.36), inset 0 0 40px rgba(0,255,157,.035);
          }
          .integrated-arcade-shell h1 {
            margin: 0;
            color: #00ff9d;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            letter-spacing: .04em;
            text-shadow: 0 0 24px rgba(0,255,157,.32);
          }
          .integrated-arcade-shell p { color: #a6b0c3; margin: 6px 0 0; }
          .integrated-arcade-note {
            margin-top: 12px;
            color: #f8fafc;
            border-left: 3px solid #ff6b35;
            padding: 8px 12px;
            background: rgba(255,107,53,.08);
          }
        </style>
        <div class="integrated-arcade-shell">
          <h1>[LIVE ARENA] PLAY ARCADE</h1>
          <p>The polished Doppelgamer arcade now lives inside Live Arena. Use Train Clone Match when you want persistent profile logging and real clone-training data.</p>
          <div class="integrated-arcade-note">Play Arcade is for fast browser play. Train Clone Match saves completed games to your profile.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if not ARCADE_FILE.exists():
        st.error(f"Arcade file not found: `{ARCADE_FILE}`")
        return
    components.html(ARCADE_FILE.read_text(encoding="utf-8"), height=980, scrolling=True)

if "arena_theme" not in st.session_state:
    st.session_state["arena_theme"] = "Auto"


def _arena_theme_override_css(theme: str) -> str:
    dark_vars = """
        --arena-bg: #0f172a;
        --arena-paper: rgba(17, 24, 39, 0.96);
        --arena-panel: rgba(30, 41, 59, 0.92);
        --arena-ink: #f8fafc;
        --arena-muted: #cbd5e1;
        --arena-border: rgba(226, 232, 240, 0.18);
        --arena-gold: #fbbf24;
        --arena-coral: #fb7185;
        --arena-mint: #5eead4;
        --accent-blue: #5eead4;
        --accent-red: #fb7185;
        --accent-green: #fbbf24;
        --arena-shadow: 0 24px 60px rgba(0, 0, 0, 0.36);
        --arena-app-bg: radial-gradient(circle at top left, rgba(94,234,212,0.10), transparent 24%),
            radial-gradient(circle at top right, rgba(251,191,36,0.10), transparent 22%),
            linear-gradient(180deg, #020617 0%, #0f172a 100%);
        --arena-sidebar-bg: linear-gradient(180deg, #020617 0%, #0f172a 100%);
        --arena-shell-bg: linear-gradient(135deg, rgba(15,23,42,0.98) 0%, rgba(30,41,59,0.94) 100%);
        --arena-hero-bg: linear-gradient(135deg, #022c22 0%, #134e4a 52%, #3b2f0b 100%);
        --arena-card-bg: rgba(15, 23, 42, 0.92);
        --arena-tile-bg: rgba(30, 41, 59, 0.96);
        --arena-tile-empty-bg: rgba(15, 23, 42, 0.78);
        --arena-control-bg: linear-gradient(180deg, rgba(30,41,59,0.98) 0%, rgba(15,23,42,0.98) 100%);
        --arena-control-hover-bg: linear-gradient(180deg, rgba(51,65,85,1) 0%, rgba(30,41,59,1) 100%);
        --arena-control-disabled-bg: rgba(15, 23, 42, 0.56);
        --arena-control-disabled-text: rgba(203, 213, 225, 0.42);
        --arena-ribbon-bg: linear-gradient(135deg, rgba(94,234,212,0.14), rgba(30,41,59,0.92));
        --arena-board-shell-bg: linear-gradient(180deg, rgba(30,41,59,0.92), rgba(15,23,42,0.96));
        --arena-tab-bg: rgba(15, 23, 42, 0.96);
        --arena-tab-active-bg: linear-gradient(180deg, rgba(45,212,191,0.24), rgba(15,118,110,0.20));
        color-scheme: dark;
    """
    light_vars = """
        --arena-bg: #f4efe6;
        --arena-paper: rgba(255, 250, 244, 0.94);
        --arena-panel: rgba(255, 255, 255, 0.82);
        --arena-ink: #201a17;
        --arena-muted: #574b43;
        --arena-border: rgba(53, 40, 31, 0.14);
        --arena-gold: #d88c2f;
        --arena-coral: #cf5b3e;
        --arena-mint: #2d8f7a;
        --accent-blue: #2d8f7a;
        --accent-red: #cf5b3e;
        --accent-green: #d88c2f;
        --arena-shadow: 0 24px 60px rgba(53, 35, 20, 0.14);
        --arena-app-bg: radial-gradient(circle at top left, rgba(216,140,47,0.16), transparent 24%),
            radial-gradient(circle at top right, rgba(45,143,122,0.14), transparent 22%),
            linear-gradient(180deg, #fbf7f0 0%, #f4efe6 100%);
        --arena-sidebar-bg: linear-gradient(180deg, #1f1814 0%, #2a1f19 100%);
        --arena-shell-bg: linear-gradient(135deg, rgba(255,255,255,0.90) 0%, rgba(255,247,238,0.88) 100%);
        --arena-hero-bg: linear-gradient(135deg, #251c17 0%, #3b2a21 58%, #7b4a24 100%);
        --arena-card-bg: rgba(255, 250, 244, 0.94);
        --arena-tile-bg: rgba(255,255,255,0.78);
        --arena-tile-empty-bg: rgba(255,255,255,0.40);
        --arena-control-bg: linear-gradient(180deg, rgba(255,255,255,0.96) 0%, rgba(248,240,230,0.90) 100%);
        --arena-control-hover-bg: linear-gradient(180deg, #fff4dc 0%, #f9dfb5 100%);
        --arena-control-disabled-bg: rgba(255,255,255,0.38);
        --arena-control-disabled-text: rgba(32,26,23,0.38);
        --arena-ribbon-bg: linear-gradient(135deg, rgba(216,140,47,0.14), rgba(255,255,255,0.76));
        --arena-board-shell-bg: linear-gradient(180deg, rgba(255,255,255,0.70), rgba(255,247,238,0.86));
        --arena-tab-bg: rgba(255,255,255,0.64);
        --arena-tab-active-bg: linear-gradient(180deg, #fff4dc 0%, #f4dcc0 100%);
        color-scheme: light;
    """
    if theme == "Dark":
        root = dark_vars
        media = ""
    elif theme == "Light":
        root = light_vars
        media = ""
    else:
        root = light_vars
        media = f"""
        @media (prefers-color-scheme: dark) {{
            :root {{
                {dark_vars}
            }}
        }}
        """
    return f"""
    <style>
        :root {{
            {root}
        }}
        {media}
        .stApp {{
            background: var(--arena-app-bg) !important;
            color: var(--arena-ink) !important;
        }}
        .dg-clone-panel {{
            padding: 16px;
            margin-bottom: 14px;
        }}
        .dg-panel-kicker,
        .dg-progress-label,
        .dg-counter-grid span,
        .dg-threat-card span {{
            font-family: var(--dg-mono);
            color: var(--dg-muted);
            font-size: 11px;
            letter-spacing: 1.2px;
            text-transform: uppercase;
        }}
        .dg-progress-label {{
            display: flex;
            justify-content: space-between;
            margin-top: 14px;
        }}
        .dg-progress-label strong {{
            color: var(--dg-clone);
        }}
        .dg-progress {{
            height: 12px;
            border-radius: 999px;
            overflow: hidden;
            background: rgba(255,255,255,0.08);
            border: 1px solid rgba(0,255,157,0.20);
            margin: 8px 0 14px;
        }}
        .dg-progress div {{
            height: 100%;
            border-radius: inherit;
            background: linear-gradient(90deg, var(--dg-human), var(--dg-clone), var(--dg-warn));
            box-shadow: 0 0 24px rgba(0,255,157,0.34);
            transition: width 450ms ease;
        }}
        .dg-counter-grid {{
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 10px;
        }}
        .dg-counter-grid div {{
            padding: 12px;
            border-radius: 14px;
            background: rgba(255,255,255,0.045);
            border: 1px solid rgba(255,255,255,0.10);
        }}
        .dg-counter-grid strong {{
            display: block;
            color: var(--dg-clone);
            font-family: var(--dg-mono);
            font-size: 28px;
            line-height: 1;
            margin-top: 8px;
        }}
        .dg-threat-card {{
            padding: 12px;
            margin-bottom: 10px;
        }}
        .dg-threat-card strong {{
            display: block;
            margin-top: 4px;
            font-family: var(--dg-mono);
            color: var(--dg-text);
        }}
        .dg-threat-green {{ border-color: rgba(0,255,157,0.36) !important; }}
        .dg-threat-yellow {{ border-color: rgba(250,204,21,0.42) !important; }}
        .dg-threat-red {{ border-color: rgba(255,107,53,0.52) !important; }}
        [data-testid="stSidebar"] {{
            background: var(--arena-sidebar-bg) !important;
        }}
        .arena-shell {{
            background: var(--arena-shell-bg) !important;
        }}
        .arena-shell::before {{
            display: none !important;
        }}
        .arena-hero {{
            background: var(--arena-hero-bg) !important;
        }}
        .game-card, .side-card, .hud-container {{
            background: var(--arena-card-bg) !important;
            color: var(--arena-ink) !important;
        }}
        .hud-score, .game-meta, .dna-item, .log-entry, .legend-pill {{
            background: var(--arena-tile-bg) !important;
            color: var(--arena-ink) !important;
            border-color: var(--arena-border) !important;
        }}
        .value-tile {{
            background: var(--arena-tile-bg) !important;
            color: var(--arena-ink) !important;
            border-color: var(--arena-border) !important;
        }}
        .value-tile.empty {{
            background: var(--arena-tile-empty-bg) !important;
            color: var(--arena-muted) !important;
        }}
        .move-ribbon {{
            background: var(--arena-ribbon-bg) !important;
            color: var(--arena-ink) !important;
            border-color: var(--arena-border) !important;
        }}
        .board-shell {{
            background: var(--arena-board-shell-bg) !important;
            border-color: var(--arena-border) !important;
        }}
        .stButton > button {{
            background: var(--arena-control-bg) !important;
            color: var(--arena-ink) !important;
            border-color: var(--arena-border) !important;
        }}
        .stButton > button:hover {{
            background: var(--arena-control-hover-bg) !important;
        }}
        .stButton > button:disabled {{
            background: var(--arena-control-disabled-bg) !important;
            color: var(--arena-control-disabled-text) !important;
        }}
        [data-baseweb="tab-list"] {{
            background: var(--arena-tab-bg) !important;
            border-color: var(--arena-border) !important;
        }}
        button[role="tab"][aria-selected="true"] {{
            background: var(--arena-tab-active-bg) !important;
            color: var(--arena-ink) !important;
        }}
        [data-baseweb="select"], [data-baseweb="input"], [data-baseweb="base-input"],
        [data-baseweb="slider"], [data-baseweb="radio"] {{
            color: var(--arena-ink) !important;
        }}
        .stMarkdown, .stCaption, .stRadio, .stSelectbox, .stSlider, label, p, span {{
            color: inherit;
        }}
    </style>
    """


# Custom CSS for premium look
st.markdown(f"""
<style>
    {_CHESS_CSS}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
    :root {
        --arena-bg: #f4efe6;
        --arena-paper: rgba(255, 250, 244, 0.88);
        --arena-panel: rgba(255, 255, 255, 0.74);
        --arena-ink: #201a17;
        --arena-muted: #695d55;
        --arena-border: rgba(53, 40, 31, 0.12);
        --arena-gold: #d88c2f;
        --arena-coral: #cf5b3e;
        --arena-mint: #2d8f7a;
        --accent-blue: #2d8f7a;
        --accent-red: #cf5b3e;
        --accent-green: #d88c2f;
        --arena-shadow: 0 24px 60px rgba(53, 35, 20, 0.14);
    }

    header, [data-testid="stHeader"], footer {visibility: hidden; height: 0;}
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1f1814 0%, #2a1f19 100%) !important;
        border-right: 1px solid rgba(255,255,255,0.08) !important;
    }

    .stApp {
        font-family: 'Outfit', sans-serif !important;
        color: var(--arena-ink) !important;
        background:
            radial-gradient(circle at top left, rgba(216, 140, 47, 0.16), transparent 24%),
            radial-gradient(circle at top right, rgba(45, 143, 122, 0.14), transparent 22%),
            linear-gradient(180deg, #fbf7f0 0%, #f4efe6 100%) !important;
    }

    [data-testid="stSidebar"] * {
        color: #f9f2e8 !important;
    }

    .arena-shell {
        background: linear-gradient(135deg, rgba(255,255,255,0.88) 0%, rgba(255,247,238,0.84) 100%);
        border: 1px solid var(--arena-border);
        border-radius: 28px;
        box-shadow: var(--arena-shadow);
        padding: 26px 28px;
        position: relative;
        overflow: hidden;
    }

    .arena-shell::before {
        content: "";
        position: absolute;
        inset: 0;
        background:
            linear-gradient(135deg, rgba(216, 140, 47, 0.08), transparent 30%),
            linear-gradient(225deg, rgba(45, 143, 122, 0.08), transparent 24%);
        pointer-events: none;
    }

    .arena-hero {
        background: linear-gradient(135deg, #251c17 0%, #3b2a21 58%, #7b4a24 100%);
        color: #fff7f0;
        border-radius: 30px;
        padding: 30px 34px;
        box-shadow: 0 26px 60px rgba(60, 33, 12, 0.22);
        margin-bottom: 22px;
        overflow: hidden;
        position: relative;
    }

    .arena-hero::after {
        content: "";
        position: absolute;
        inset: 0;
        background:
            radial-gradient(circle at 85% 20%, rgba(216, 140, 47, 0.45), transparent 18%),
            radial-gradient(circle at 12% 85%, rgba(45, 143, 122, 0.28), transparent 16%);
        pointer-events: none;
    }

    .arena-kicker {
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 2px;
        opacity: 0.7;
        margin-bottom: 10px;
    }

    .arena-title {
        font-size: 40px;
        font-weight: 800;
        line-height: 1;
        margin: 0;
    }

    .arena-subtitle {
        margin-top: 10px;
        max-width: 760px;
        font-size: 16px;
        line-height: 1.55;
        color: rgba(255,247,240,0.86);
    }

    .arena-chip-row {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        margin-top: 18px;
    }

    .arena-chip {
        border: 1px solid rgba(255,255,255,0.16);
        background: rgba(255,255,255,0.08);
        border-radius: 999px;
        padding: 8px 14px;
        font-size: 13px;
        font-weight: 600;
    }

    .game-card, .side-card, .hud-container {
        background: var(--arena-paper) !important;
        border: 1px solid var(--arena-border) !important;
        border-radius: 24px !important;
        box-shadow: 0 14px 36px rgba(77, 53, 31, 0.10) !important;
        backdrop-filter: blur(18px);
    }

    .game-card {
        padding: 22px !important;
    }

    .side-card {
        padding: 18px;
        margin-bottom: 16px;
    }

    .hud-container {
        display: flex;
        justify-content: space-around;
        gap: 14px;
        padding: 14px;
        margin-bottom: 20px;
    }

    .hud-score {
        text-align: center;
        flex: 1;
        background: rgba(255,255,255,0.52);
        border-radius: 18px;
        padding: 14px 10px;
    }

    .hud-label {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: var(--arena-muted);
        margin-bottom: 6px;
    }

    .hud-val {
        font-size: 34px;
        font-weight: 800;
        line-height: 1;
        color: var(--arena-ink);
    }

    .energy-bar {
        height: 10px !important;
        border-radius: 999px !important;
        background: rgba(32, 26, 23, 0.08) !important;
        overflow: hidden !important;
    }

    .stButton > button {
        background: linear-gradient(180deg, rgba(255,255,255,0.92) 0%, rgba(248,240,230,0.86) 100%) !important;
        border: 1px solid rgba(55,39,28,0.14) !important;
        border-radius: 18px !important;
        color: var(--arena-ink) !important;
        font-weight: 700 !important;
        letter-spacing: 0.2px !important;
        min-height: 3.1rem !important;
        transition: all 0.18s ease !important;
        box-shadow: 0 10px 20px rgba(73, 47, 25, 0.08) !important;
    }

    .stButton > button:hover {
        background: linear-gradient(180deg, #fff4dc 0%, #f9dfb5 100%) !important;
        border-color: rgba(216, 140, 47, 0.28) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 14px 24px rgba(125, 74, 36, 0.16) !important;
    }

    .stButton > button:disabled {
        background: rgba(255,255,255,0.34) !important;
        color: rgba(32,26,23,0.34) !important;
        border-color: rgba(55,39,28,0.08) !important;
        box-shadow: none !important;
    }

    div[class*="st-key-ttt_"] button,
    div[class*="st-key-gomoku_"] button,
    div[class*="st-key-oth_"] button,
    div[class*="st-key-chk_"] button,
    div[class*="st-key-chk_to_"] button,
    div[class*="st-key-chess_sq_"] button {
        aspect-ratio: 1 / 1 !important;
        min-height: 54px !important;
        padding: 0 !important;
        font-size: 22px !important;
        border-radius: 14px !important;
        background: var(--arena-tile-bg) !important;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.08), 0 8px 18px rgba(0,0,0,0.08) !important;
    }

    div[class*="st-key-ttt_"] button:hover,
    div[class*="st-key-gomoku_"] button:hover,
    div[class*="st-key-oth_"] button:hover,
    div[class*="st-key-chk_"] button:hover,
    div[class*="st-key-chk_to_"] button:hover,
    div[class*="st-key-chess_sq_"] button:hover {
        transform: translateY(-2px) scale(1.02) !important;
        border-color: var(--arena-mint) !important;
    }

    div[class*="st-key-ttt_"] button:disabled,
    div[class*="st-key-gomoku_"] button:disabled,
    div[class*="st-key-oth_"] button:disabled,
    div[class*="st-key-chk_"] button:disabled,
    div[class*="st-key-chk_to_"] button:disabled,
    div[class*="st-key-chess_sq_"] button:disabled {
        opacity: 1 !important;
        color: var(--arena-ink) !important;
    }

    div[class*="st-key-cf_btn_"] button {
        min-height: 68px !important;
        border-radius: 18px !important;
        font-size: 16px !important;
    }

    div[class*="st-key-nim_"] button {
        min-height: 44px !important;
        border-radius: 999px !important;
        font-size: 14px !important;
    }

    .log-entry {
        background: rgba(255,255,255,0.78) !important;
        border: 1px solid rgba(54, 38, 28, 0.10);
        border-radius: 18px !important;
        padding: 15px !important;
        margin-bottom: 12px !important;
        transition: transform 0.18s ease !important;
    }

    .log-entry:hover {
        transform: translateY(-1px) !important;
    }

    .section-label {
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 1.6px;
        color: var(--arena-muted);
        margin-bottom: 8px;
    }

    .game-intro {
        display: flex;
        justify-content: space-between;
        gap: 16px;
        align-items: flex-start;
        margin-bottom: 18px;
    }

    .game-intro h3 {
        margin: 0 0 6px 0;
        font-size: 28px;
        color: var(--arena-ink);
    }

    .game-intro p {
        margin: 0;
        color: var(--arena-muted);
        line-height: 1.5;
    }

    .game-meta {
        min-width: 190px;
        background: rgba(255,255,255,0.56);
        border: 1px solid rgba(54,38,28,0.08);
        border-radius: 18px;
        padding: 14px 16px;
    }

    .game-meta strong {
        display: block;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        color: var(--arena-muted);
        margin-bottom: 4px;
    }

    .value-board {
        display: grid;
        gap: 8px;
        margin: 14px 0 18px 0;
    }

    .value-tile {
        border-radius: 18px;
        min-height: 68px;
        display: flex;
        align-items: center;
        justify-content: center;
        text-align: center;
        padding: 8px;
        font-weight: 700;
        border: 1px solid rgba(49, 36, 27, 0.08);
        background: rgba(255,255,255,0.7);
        color: var(--arena-ink);
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.4);
    }

    .value-tile.empty {
        color: rgba(32,26,23,0.28);
        background: rgba(255,255,255,0.34);
    }

    .move-ribbon {
        background: linear-gradient(135deg, rgba(216,140,47,0.14), rgba(255,255,255,0.7));
        border: 1px solid rgba(216,140,47,0.22);
        border-radius: 18px;
        padding: 12px 14px;
        margin-bottom: 14px;
        color: var(--arena-ink);
    }

    .board-shell {
        background: linear-gradient(180deg, rgba(255,255,255,0.62), rgba(255,247,238,0.82));
        border: 1px solid rgba(54, 38, 28, 0.10);
        border-radius: 22px;
        padding: 16px;
        margin: 12px 0 18px 0;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.55);
    }

    .legend-row {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin: 10px 0 12px 0;
    }

    .legend-pill {
        border-radius: 999px;
        padding: 6px 10px;
        font-size: 12px;
        font-weight: 700;
        background: rgba(255,255,255,0.7);
        border: 1px solid rgba(54, 38, 28, 0.08);
        color: var(--arena-ink);
    }

    .connect-board {
        display: grid;
        grid-template-columns: repeat(7, minmax(0, 1fr));
        gap: 8px;
        background: linear-gradient(180deg, #2458b0 0%, #1f4d9a 100%);
        border-radius: 24px;
        padding: 14px;
        box-shadow: inset 0 6px 18px rgba(255,255,255,0.08);
    }

    .connect-slot {
        aspect-ratio: 1 / 1;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 26px;
        font-weight: 800;
        background: radial-gradient(circle at 35% 35%, rgba(255,255,255,0.95), rgba(239,232,223,0.92));
        color: #3b302a;
        border: 1px solid rgba(0,0,0,0.08);
    }

    .othello-board, .checkers-board {
        background: #6f8f50;
        border-radius: 24px;
        padding: 12px;
        border: 1px solid rgba(54, 38, 28, 0.14);
        box-shadow: inset 0 10px 18px rgba(255,255,255,0.08);
    }

    .pacman-board {
        background: linear-gradient(180deg, #171d34 0%, #0e1327 100%);
        border-radius: 24px;
        padding: 14px;
        border: 1px solid rgba(82, 111, 214, 0.20);
        box-shadow: inset 0 10px 18px rgba(255,255,255,0.04);
    }

    .tile-pop {
        animation: tile-pop 180ms ease-out;
    }

    .wordle-flip {
        animation: wordle-flip 420ms ease-out;
        transform-style: preserve-3d;
    }

    .dna-list {
        display: grid;
        gap: 10px;
        margin-top: 12px;
    }

    .dna-item {
        padding: 12px 14px;
        border-radius: 16px;
        background: rgba(255,255,255,0.52);
        border: 1px solid rgba(54, 38, 28, 0.08);
    }

    .dna-item strong {
        display: block;
        margin-bottom: 4px;
    }

    @keyframes tile-pop {
        0% { transform: scale(0.94); }
        65% { transform: scale(1.04); }
        100% { transform: scale(1); }
    }

    @keyframes wordle-flip {
        0% { transform: rotateX(0deg); }
        45% { transform: rotateX(90deg); }
        100% { transform: rotateX(0deg); }
    }

    [data-baseweb="tab-list"] {
        gap: 10px;
        background: rgba(255,255,255,0.58);
        border: 1px solid rgba(54, 38, 28, 0.08);
        border-radius: 999px;
        padding: 8px;
        margin-bottom: 18px;
    }

    [data-baseweb="tab"] {
        height: 42px;
        border-radius: 999px !important;
        color: var(--arena-muted) !important;
        font-weight: 700 !important;
        padding: 0 16px !important;
    }

    button[role="tab"][aria-selected="true"] {
        background: linear-gradient(180deg, #fff4dc 0%, #f4dcc0 100%) !important;
        color: var(--arena-ink) !important;
        box-shadow: 0 8px 18px rgba(125, 74, 36, 0.12) !important;
    }

    @media (max-width: 900px) {
        .arena-title { font-size: 32px; }
        .arena-hero { padding: 24px; }
        .game-intro { display: block; }
        .game-meta { margin-top: 14px; width: 100%; }
        .value-tile { min-height: 58px; font-size: 13px; }
    }
</style>
""", unsafe_allow_html=True)

st.markdown(_arena_theme_override_css(st.session_state.get("arena_theme", "Auto")), unsafe_allow_html=True)

st.markdown("<style>\n" + _BOARD_BUTTON_CSS + """
    :root {
        --pop-cyan: #22d3ee;
        --pop-pink: #f472b6;
        --pop-violet: #a78bfa;
        --pop-orange: #fb923c;
        --pop-lime: #a3e635;
        --pop-blue: #60a5fa;
    }

    .stApp {
        background:
            radial-gradient(circle at 6% 12%, rgba(34,211,238,0.24), transparent 22%),
            radial-gradient(circle at 92% 10%, rgba(244,114,182,0.22), transparent 20%),
            radial-gradient(circle at 55% 0%, rgba(251,191,36,0.15), transparent 26%),
            radial-gradient(circle at 22% 92%, rgba(163,230,53,0.14), transparent 24%),
            linear-gradient(135deg, #07111f 0%, #111827 44%, #3b0764 100%) !important;
    }

    [data-testid="stAppViewContainer"] > .main {
        background:
            linear-gradient(90deg, rgba(255,255,255,0.035) 1px, transparent 1px),
            linear-gradient(0deg, rgba(255,255,255,0.025) 1px, transparent 1px) !important;
        background-size: 42px 42px !important;
    }

    .arena-shell {
        background:
            radial-gradient(circle at 18% 12%, rgba(34,211,238,0.14), transparent 24%),
            radial-gradient(circle at 88% 4%, rgba(244,114,182,0.13), transparent 22%),
            linear-gradient(145deg, rgba(15,23,42,0.88), rgba(30,41,59,0.78)) !important;
        border: 1px solid rgba(255,255,255,0.16) !important;
        box-shadow: 0 30px 90px rgba(0,0,0,0.30), 0 0 60px rgba(34,211,238,0.10) !important;
    }

    .arena-hero {
        min-height: 220px;
        display: grid;
        align-content: end;
        background:
            radial-gradient(circle at 82% 16%, rgba(244,114,182,0.52), transparent 22%),
            radial-gradient(circle at 12% 82%, rgba(34,211,238,0.42), transparent 24%),
            linear-gradient(135deg, #312e81 0%, #0f766e 48%, #b45309 100%) !important;
        border: 1px solid rgba(255,255,255,0.14);
        box-shadow: 0 30px 90px rgba(79,70,229,0.30), inset 0 1px 0 rgba(255,255,255,0.18);
    }

    .arena-hero::after {
        background:
            linear-gradient(90deg, rgba(255,255,255,0.04) 1px, transparent 1px),
            linear-gradient(0deg, rgba(255,255,255,0.04) 1px, transparent 1px) !important;
        background-size: 34px 34px !important;
        mask-image: linear-gradient(90deg, transparent, black 18%, black 82%, transparent);
        opacity: 0.5;
    }

    .match-strip {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 14px;
        margin-top: 18px;
        position: relative;
        z-index: 1;
    }

    .player-badge {
        min-width: 160px;
        padding: 14px 16px;
        border-radius: 20px;
        background: linear-gradient(135deg, rgba(34,211,238,0.26), rgba(96,165,250,0.14));
        border: 1px solid rgba(255,255,255,0.16);
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.10);
        backdrop-filter: blur(16px);
    }

    .player-badge:nth-of-type(3) {
        background: linear-gradient(135deg, rgba(244,114,182,0.28), rgba(251,146,60,0.16));
    }

    .player-badge strong {
        display: block;
        font-size: 18px;
        line-height: 1.1;
        color: #ffffff;
    }

    .player-badge span {
        display: block;
        margin-top: 5px;
        font-size: 11px;
        letter-spacing: 1.4px;
        text-transform: uppercase;
        color: rgba(255,255,255,0.66);
    }

    .versus-mark {
        width: 46px;
        height: 46px;
        border-radius: 999px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-weight: 900;
        color: #092c2a;
        background: linear-gradient(135deg, #22d3ee, #f472b6 48%, #fbbf24);
        box-shadow: 0 16px 36px rgba(244,114,182,0.34);
    }

    .arena-chip {
        color: rgba(255,255,255,0.90);
        background: linear-gradient(135deg, rgba(255,255,255,0.16), rgba(255,255,255,0.07)) !important;
        backdrop-filter: blur(12px);
    }

    .hud-container {
        padding: 18px !important;
        background:
            radial-gradient(circle at 16% 0%, rgba(34,211,238,0.20), transparent 30%),
            radial-gradient(circle at 88% 0%, rgba(244,114,182,0.18), transparent 30%),
            linear-gradient(135deg, rgba(255,255,255,0.08), transparent),
            var(--arena-card-bg) !important;
        box-shadow: 0 18px 44px rgba(0,0,0,0.18), 0 0 36px rgba(244,114,182,0.08) !important;
    }

    .hud-score {
        position: relative;
        overflow: hidden;
        border: 1px solid var(--arena-border);
        background:
            linear-gradient(145deg, rgba(34,211,238,0.16), rgba(96,165,250,0.08)),
            var(--arena-tile-bg) !important;
    }

    .hud-score::before {
        content: "";
        position: absolute;
        inset: 0 auto 0 0;
        width: 5px;
        background: linear-gradient(180deg, var(--pop-cyan), var(--pop-blue));
        opacity: 0.75;
    }

    .hud-score + div + .hud-score::before {
        background: linear-gradient(180deg, var(--pop-pink), var(--pop-orange));
    }

    .hud-score + div + .hud-score {
        background:
            linear-gradient(145deg, rgba(244,114,182,0.15), rgba(251,146,60,0.08)),
            var(--arena-tile-bg) !important;
    }

    .game-card {
        background:
            radial-gradient(circle at 10% 8%, rgba(34,211,238,0.14), transparent 26%),
            radial-gradient(circle at 92% 18%, rgba(244,114,182,0.13), transparent 24%),
            linear-gradient(145deg, rgba(255,255,255,0.06), transparent 42%),
            var(--arena-card-bg) !important;
        border-color: rgba(255,255,255,0.16) !important;
    }

    .board-shell {
        padding: 20px;
        border-radius: 28px;
        background:
            radial-gradient(circle at 16% 0%, rgba(34,211,238,0.30), transparent 28%),
            radial-gradient(circle at 88% 12%, rgba(244,114,182,0.28), transparent 24%),
            radial-gradient(circle at 50% 100%, rgba(163,230,53,0.20), transparent 32%),
            linear-gradient(135deg, rgba(124,58,237,0.18), rgba(14,165,233,0.14)),
            var(--arena-board-shell-bg) !important;
        border: 1px solid rgba(255,255,255,0.18) !important;
        box-shadow:
            inset 0 1px 0 rgba(255,255,255,0.16),
            inset 0 -30px 60px rgba(0,0,0,0.14),
            0 24px 64px rgba(59,130,246,0.20),
            0 0 44px rgba(244,114,182,0.10);
    }

    .connect-board {
        background:
            radial-gradient(circle at 20% 12%, rgba(255,255,255,0.18), transparent 16%),
            linear-gradient(160deg, #7c3aed 0%, #2563eb 45%, #06b6d4 100%) !important;
        gap: 10px;
        box-shadow: inset 0 10px 24px rgba(255,255,255,0.16), inset 0 -18px 28px rgba(0,0,0,0.18);
    }

    .connect-slot {
        background: radial-gradient(circle at 35% 30%, rgba(255,255,255,0.98), rgba(226,232,240,0.88)) !important;
        box-shadow: inset 0 10px 18px rgba(0,0,0,0.16), 0 6px 12px rgba(0,0,0,0.12);
    }

    .othello-board {
        background:
            radial-gradient(circle at 80% 10%, rgba(163,230,53,0.20), transparent 20%),
            linear-gradient(135deg, #16a34a 0%, #0f766e 54%, #065f46 100%) !important;
        box-shadow: inset 0 0 0 2px rgba(255,255,255,0.08), inset 0 20px 40px rgba(0,0,0,0.18);
    }

    .checkers-board {
        background:
            linear-gradient(135deg, #7c2d12 0%, #581c87 52%, #111827 100%) !important;
    }

    div[class*="st-key-ttt_"] button {
        font-size: 34px !important;
        border-radius: 22px !important;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.24), 0 14px 32px rgba(0,0,0,0.20) !important;
    }

    div[class*="st-key-gomoku_"] button {
        border-radius: 999px !important;
        font-size: 25px !important;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.18), 0 8px 18px rgba(0,0,0,0.14) !important;
    }

    div[class*="st-key-oth_"] button {
        border-radius: 999px !important;
        font-size: 24px !important;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.20), 0 8px 18px rgba(0,0,0,0.16) !important;
    }

    div[class*="st-key-chk_"] button,
    div[class*="st-key-chk_to_"] button,
    div[class*="st-key-chess_sq_"] button {
        border-radius: 12px !important;
        font-size: 18px !important;
    }

    div[class*="st-key-cf_btn_"] button {
        background:
            radial-gradient(circle at 20% 16%, rgba(255,255,255,0.30), transparent 22%),
            linear-gradient(135deg, #06b6d4, #2563eb 52%, #7c3aed) !important;
        color: #ffffff !important;
        border-color: rgba(255,255,255,0.22) !important;
    }

    div[class*="st-key-nim_"] button {
        background:
            radial-gradient(circle at 20% 16%, rgba(255,255,255,0.30), transparent 22%),
            linear-gradient(135deg, #ec4899, #f97316 54%, #facc15) !important;
        color: #111827 !important;
        border-color: rgba(255,255,255,0.24) !important;
    }

    .move-ribbon {
        background:
            linear-gradient(135deg, rgba(34,211,238,0.16), rgba(244,114,182,0.12), rgba(251,191,36,0.14)) !important;
        border-color: rgba(244,114,182,0.22) !important;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.10), 0 12px 28px rgba(0,0,0,0.08);
    }

    .side-card {
        background:
            radial-gradient(circle at 92% 8%, rgba(251,191,36,0.10), transparent 24%),
            linear-gradient(145deg, rgba(255,255,255,0.06), transparent 38%),
            var(--arena-card-bg) !important;
    }
</style>
""", unsafe_allow_html=True)

def render_energy(label, val, max_val=5, color="#00d1b2"):
    pct = (val / max_val) * 100
    st.caption(f"{label}: {val}/{max_val}")
    st.markdown(f"""
    <div class="energy-bar">
        <div class="energy-fill" style="width: {pct}%; background: {color};"></div>
    </div>
    """, unsafe_allow_html=True)


def _safe_int(value, default=-1):
    try:
        return int(value)
    except Exception:
        return default


def _safe_name(value, fallback):
    if hasattr(value, "name"):
        return str(value.name)
    return fallback


def _active_user_profile():
    profile = st.session_state.get("user_profile")
    if profile and profile.get("id"):
        return profile
    return require_user_profile("Live Arena")


def _new_agent(agent_name):
    if agent_name == "profile_counter":
        profile = _active_user_profile()
        return ProfileCounterAgent(player_id=profile["id"], db_path=DEFAULT_DB)
    if agent_name == "adaptive_router":
        profile = _active_user_profile()
        return AdaptiveRouterAgent(player_id=profile["id"], db_path=DEFAULT_DB)
    if agent_name in {"ngram", "lstm", "mixture"}:
        source_player = st.session_state.get("game_settings", {}).get("friend_clone_source_player") or _active_user_profile()["id"]
        trainer = ImpostorTrainer(DEFAULT_DB)
        if agent_name == "ngram":
            agent, _ = trainer.train_ngram(source_player)
        elif agent_name == "mixture":
            agent, _ = trainer.train_mixture(source_player)
        else:
            agent, _ = trainer.train_lstm(source_player, epochs=5, save=False)
        setattr(agent, "_clone_source_player", source_player)
        return agent
    return AGENT_REGISTRY[agent_name]()


def _challenge_catalog(game_type: str) -> list[dict]:
    return GAME_CHALLENGES.get(game_type, [])


def _ladder_enabled() -> bool:
    return bool(st.session_state.get("game_settings", {}).get("clone_ladder_enabled", False))


def _current_ladder_rungs() -> list[dict]:
    return list(CLONE_LADDER_RPS) if st.session_state.game_state.get("game_type") == DEFAULT_LADDER_GAME else []


def _active_ladder_rung() -> dict | None:
    ladder_state = st.session_state.get("ladder_state") or {}
    rungs = _current_ladder_rungs()
    rung_index = int(ladder_state.get("rung_index", 0))
    if not rungs or rung_index >= len(rungs):
        return None
    return rungs[rung_index]


def _blind_study_active() -> bool:
    return bool(st.session_state.get("blind_study_state", {}).get("active", False))


def _current_blind_entry() -> dict[str, Any] | None:
    state = st.session_state.get("blind_study_state") or {}
    schedule = state.get("schedule") or []
    idx = int(state.get("current_index", 0))
    if 0 <= idx < len(schedule):
        return schedule[idx]
    return None


def _reset_blind_study_state() -> None:
    st.session_state["blind_study_state"] = {
        "active": False,
        "block_id": None,
        "schedule": [],
        "current_index": 0,
        "revealed": False,
    }


def _start_blind_study(player_id: str, game_type: str = "RPS+") -> None:
    conditions = ["human_baseline", "heuristic", "ngram", "lstm"]
    schedule_payload = create_blind_match_schedule(player_id=player_id, game_type=game_type, conditions=conditions)
    block_id = persist_blind_match_schedule(db_path=str(DEFAULT_DB), player_id=player_id, game_type=game_type, conditions=conditions)
    st.session_state["blind_study_state"] = {
        "active": True,
        "block_id": block_id,
        "schedule": schedule_payload["schedule"],
        "current_index": 0,
        "revealed": False,
    }


def _advance_blind_study() -> None:
    state = st.session_state.setdefault("blind_study_state", {})
    state["current_index"] = int(state.get("current_index", 0)) + 1
    schedule = state.get("schedule") or []
    if state["current_index"] >= len(schedule):
        state["revealed"] = True
        state["active"] = False


def _apply_blind_condition(selected_game: str, selected_agent: str, play_mode: str, selected_agent_2: str | None) -> tuple[str, str, str, str | None]:
    if not _blind_study_active():
        return selected_game, selected_agent, play_mode, selected_agent_2
    entry = _current_blind_entry() or {}
    condition = entry.get("condition", "heuristic")
    selected_game = "RPS+"
    if condition == "human_baseline":
        return selected_game, "random", "human_human", None
    return selected_game, str(condition), "human_bot1", str(condition)


def _reset_ladder_state() -> None:
    st.session_state["ladder_state"] = {
        "active": _ladder_enabled(),
        "rung_index": 0,
        "wins": 0,
        "losses": 0,
        "draws": 0,
        "completed": False,
        "run_id": uuid.uuid4().hex[:12],
        "rung_history": [],
    }


def _persist_ladder_result() -> None:
    ladder_state = st.session_state.get("ladder_state") or {}
    if not ladder_state.get("active"):
        return
    init_extended_db(DEFAULT_DB)
    rung = _active_ladder_rung() or {"agent": "unknown"}
    profile = _active_user_profile()
    conn = connect(DEFAULT_DB)
    try:
        conn.execute(
            """INSERT INTO clone_ladder_runs
               (ladder_run_id, player_id, game_type, rung_index, rung_agent, result, wins, losses, draws, completed_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                ladder_state.get("run_id", uuid.uuid4().hex[:12]),
                profile["id"],
                st.session_state.game_state.get("game_type", DEFAULT_LADDER_GAME),
                int(ladder_state.get("rung_index", 0)),
                rung.get("agent", "unknown"),
                "completed" if ladder_state.get("completed") else str(st.session_state.get("winner", "draw")),
                int(ladder_state.get("wins", 0)),
                int(ladder_state.get("losses", 0)),
                int(ladder_state.get("draws", 0)),
                __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def _challenge_payload(game_type: str, challenge_id: str) -> dict | None:
    if not challenge_id:
        return None
    if game_type == "Chess":
        mapping = {
            "sicilian_pressure": {"fen": "r1bqkbnr/pp2pppp/2np4/2p5/4P3/2NP1N2/PPP2PPP/R1BQKB1R w KQkq - 2 5"},
            "endgame_conversion": {"fen": "8/8/3k4/8/3K4/5P2/8/6R1 w - - 0 1"},
        }
        return mapping.get(challenge_id)
    if game_type == "Connect Four":
        return {
            "center_tension": {
                "board": [
                    [1, -1, 1, 0, 0, 0, 0],
                    [0, 1, -1, 0, 0, 0, 0],
                    [0, 0, 1, -1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                ],
                "turn": 6,
            }
        }.get(challenge_id)
    if game_type == "Checkers":
        return {
            "forced_chain": {
                "board": [
                    0, 1, 0, 0, 0, 0, 0, 0,
                    0, 0, -1, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, -1, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0,
                ],
                "turn": 0,
            }
        }.get(challenge_id)
    if game_type == "Gomoku":
        board = np.zeros((15, 15), dtype=int)
        board[7, 6:9] = 1
        board[6, 7] = -1
        board[8, 7] = -1
        return {"board": board.tolist(), "turn": 5} if challenge_id == "open_three_race" else None
    return None


def _apply_opening_challenge(env, game_type: str) -> tuple[object, object] | None:
    challenge_id = st.session_state.get("game_settings", {}).get("opening_challenge")
    payload = _challenge_payload(game_type, challenge_id)
    if not payload:
        return None
    if hasattr(env, "load_challenge"):
        if game_type == "Chess":
            return env.load_challenge(payload["fen"])
        return env.load_challenge(payload["board"], turn=payload.get("turn", 0))
    return None


def _play_mode_label(mode):
    for label, value in PLAY_MODES.items():
        if value == mode:
            return label
    return "Play a Bot"


def _side_labels():
    mode = st.session_state.game_state.get("play_mode", "human_bot1")
    user_prof = _active_user_profile()
    bot1 = st.session_state.game_state.get("agent_name") or "bot1"
    bot2 = st.session_state.game_state.get("agent2_name") or "bot2"
    if _blind_study_active():
        blind_entry = _current_blind_entry() or {}
        blind_label = blind_entry.get("blind_label", "Opponent")
        if mode == "human_human":
            return user_prof["name"], blind_label
        return user_prof["name"], blind_label
    if mode == "human_bot1":
        return user_prof["name"], bot1
    if mode == "human_bot2":
        return user_prof["name"], bot2
    if mode == "human_human":
        return "Player 1", "Player 2"
    if mode == "bot_bot":
        return bot1, bot2
    return user_prof["name"], bot1


def _supported_agents_for_game(game_type):
    allowed = GAME_AGENT_OPTIONS.get(game_type)
    if not allowed:
        return list(AGENT_REGISTRY.keys()) + ["ngram", "lstm", "mixture"]
    return [name for name in allowed if name in AGENT_REGISTRY or name in {"ngram", "lstm", "mixture"}]


def _new_env(game_type, max_turns):
    settings = st.session_state.get("game_settings", {})
    if game_type == "RPS+":
        return RPSPlusEnv(max_turns=max_turns, starting_energy=int(settings.get("rps_starting_energy", 3)))
    if game_type == "Tic-Tac-Toe":
        return TicTacToeEnv(max_moves=max_turns)
    if game_type == "Connect Four":
        return ConnectFourEnv(max_moves=max_turns)
    if game_type == "Chess":
        return ChessEnv(max_moves=max_turns)
    if game_type == "Othello":
        return OthelloEnv(max_moves=max_turns)
    if game_type == "Checkers":
        return CheckersEnv(max_moves=max_turns, forced_jump=bool(settings.get("checkers_forced_jump", True)))
    if game_type == "Gomoku":
        board_size = int(settings.get("gomoku_board_size", 15))
        return GomokuEnv(max_moves=max_turns, board_size=board_size, win_length=5)
    if game_type == "Nim":
        pile_presets = {
            "Classic": [1, 3, 5, 7],
            "Fibonacci": [1, 2, 3, 5],
            "Custom": settings.get("nim_custom_piles", [1, 3, 5, 7]),
        }
        preset = settings.get("nim_preset", "Classic")
        return NimEnv(max_moves=max_turns, piles=pile_presets.get(preset, [1, 3, 5, 7]))
    if game_type == "War":
        return WarEnv(max_moves=max_turns)
    if game_type in FUTURE_GAME_LABELS:
        return FUTURE_GAME_ENVS[FUTURE_GAME_LABELS[game_type]](max_moves=max_turns)
    raise ValueError(f"Unknown game type: {game_type}")


def _apply_game_settings(selected_game, selected_agent, max_turns, play_mode="human_bot1", selected_agent_2=None):
    selected_game, selected_agent, play_mode, selected_agent_2 = _apply_blind_condition(
        selected_game,
        selected_agent,
        play_mode,
        selected_agent_2,
    )
    if _ladder_enabled():
        selected_game = DEFAULT_LADDER_GAME
        play_mode = "human_bot1"
        ladder_state = st.session_state.get("ladder_state")
        if not ladder_state or not ladder_state.get("active"):
            _reset_ladder_state()
            ladder_state = st.session_state.get("ladder_state") or {}
        rung = _active_ladder_rung()
        if rung:
            selected_agent = rung["agent"]
            selected_agent_2 = rung["agent"]
    supported = _supported_agents_for_game(selected_game)
    if selected_agent not in supported:
        selected_agent = supported[0]
    selected_agent_2 = selected_agent_2 or selected_agent
    if selected_agent_2 not in supported:
        selected_agent_2 = supported[0]
    st.session_state.game_state["env"] = _new_env(selected_game, max_turns)
    st.session_state.game_state["game_type"] = selected_game
    st.session_state.game_state["agent"] = _new_agent(selected_agent)
    st.session_state.game_state["agent_name"] = selected_agent
    st.session_state.game_state["agent2"] = _new_agent(selected_agent_2)
    st.session_state.game_state["agent2_name"] = selected_agent_2
    st.session_state.game_state["play_mode"] = play_mode
    obs, info = st.session_state.game_state["env"].reset()
    challenged = _apply_opening_challenge(st.session_state.game_state["env"], selected_game)
    if challenged is not None:
        obs, info = challenged
    st.session_state.game_state["obs"] = obs
    st.session_state.game_state["info"] = info
    st.session_state.game_state["history"] = []
    st.session_state.game_state["surprisal_history"] = []
    st.session_state.game_state["last_surprisal"] = None
    st.session_state.game_state["done"] = False
    st.session_state.game_state["saved_to_profile"] = False
    st.session_state.game_state["score_recorded"] = False
    st.session_state.game_state["detection_recorded"] = False
    st.session_state.game_state["pending_player_move"] = None
    st.session_state.game_state["chess_selected_sq"] = None
    st.session_state.game_state["sudoku_selected_cell"] = None
    st.session_state.game_state["checkers_selected"] = None
    st.session_state["move_history"] = []
    st.session_state["round_count"] = 0
    st.session_state["game_over"] = False
    st.session_state["winner"] = None
    st.session_state["paused"] = False
    st.session_state["show_last_turn_replay"] = False
    _sync_score_scope()


def _current_score_scope() -> tuple:
    state = st.session_state.game_state or {}
    if _ladder_enabled():
        ladder_state = st.session_state.get("ladder_state") or {}
        return (
            "clone_ladder",
            state.get("game_type"),
            ladder_state.get("run_id"),
        )
    return (
        state.get("game_type"),
        state.get("play_mode"),
        state.get("agent_name"),
        state.get("agent2_name"),
        st.session_state.get("game_settings", {}).get("series_length", 3),
    )


def _sync_score_scope(reset: bool = False) -> None:
    scope = _current_score_scope()
    if reset or st.session_state.get("score_scope") != scope:
        st.session_state["score_scope"] = scope
        st.session_state["score_you"] = 0
        st.session_state["score_agent"] = 0
        st.session_state["score_draws"] = 0
        st.session_state["series_complete"] = False


def _series_length() -> int:
    return int(st.session_state.get("game_settings", {}).get("series_length", 3))


def _series_wins_needed() -> int:
    return (_series_length() // 2) + 1


def _series_complete() -> bool:
    return bool(
        st.session_state.get("score_you", 0) >= _series_wins_needed()
        or st.session_state.get("score_agent", 0) >= _series_wins_needed()
    )


def _refresh_profile_rollup(player_id: str, db_path: Path = DEFAULT_DB) -> None:
    if not player_id or player_id.startswith("guest_"):
        return
    conn = connect(db_path)
    try:
        summary = conn.execute(
            """SELECT COUNT(*) AS games_played,
                      COALESCE(SUM(n_turns), 0) AS total_rounds,
                      COALESCE(AVG(CASE WHEN agent_score > opponent_score THEN 1.0 ELSE 0.0 END), 0.0) AS win_rate
               FROM games
               WHERE agent_name = ?""",
            (player_id,),
        ).fetchone()
        conn.execute(
            """UPDATE player_profiles
               SET games_played = ?, total_rounds = ?, win_rate = ?
               WHERE player_id = ?""",
            (
                int(summary["games_played"] or 0),
                int(summary["total_rounds"] or 0),
                float(summary["win_rate"] or 0.0),
                player_id,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def _winner_from_scores(env) -> str:
    if env.state.agent_score > env.state.opponent_score:
        return "you"
    if env.state.opponent_score > env.state.agent_score:
        return "agent"
    return "draw"


def _record_score_if_needed() -> None:
    state = st.session_state.game_state
    if not state or not state.get("done") or state.get("score_recorded"):
        return
    winner = _winner_from_scores(state["env"])
    st.session_state["winner"] = winner
    st.session_state["game_over"] = True
    if winner == "you":
        st.session_state["score_you"] += 1
    elif winner == "agent":
        st.session_state["score_agent"] += 1
    else:
        st.session_state["score_draws"] += 1
    if _ladder_enabled():
        ladder_state = st.session_state.setdefault("ladder_state", {})
        ladder_state["active"] = True
        ladder_state["wins"] = int(ladder_state.get("wins", 0)) + (1 if winner == "you" else 0)
        ladder_state["losses"] = int(ladder_state.get("losses", 0)) + (1 if winner == "agent" else 0)
        ladder_state["draws"] = int(ladder_state.get("draws", 0)) + (1 if winner == "draw" else 0)
        rung = _active_ladder_rung() or {}
        ladder_state.setdefault("rung_history", []).append(
            {
                "rung_index": int(ladder_state.get("rung_index", 0)),
                "agent": rung.get("agent", state.get("agent_name", "unknown")),
                "result": winner,
            }
        )
    state["score_recorded"] = True
    st.session_state["series_complete"] = _series_complete()


def _replay_saved_turns(entries: list[dict]) -> None:
    state = st.session_state.game_state
    env = state["env"]
    mode = state.get("play_mode", "human_bot1")
    agent1 = state.get("agent")
    agent2 = state.get("agent2")
    for entry in entries:
        player_move = int(entry.get("player_action", -1))
        opponent_move = int(entry.get("opponent_action", -1))
        original_policy = getattr(env, "_opponent_policy", None)
        if opponent_move >= 0:
            def replay_policy(move=opponent_move):
                if state["game_type"] == "RPS+":
                    from environments.rps_plus import Move
                    return Move(move)
                return move
            env._opponent_policy = replay_policy
        try:
            obs, reward, terminated, truncated, info = env.step(player_move)
        finally:
            if original_policy is not None:
                env._opponent_policy = original_policy
            elif hasattr(env, "_opponent_policy"):
                delattr(env, "_opponent_policy")
        state["obs"] = obs
        state["info"] = info
        state["done"] = terminated or truncated
        if mode in {"human_bot1", "bot_bot"} and agent1 is not None and opponent_move >= 0:
            _observe_agent(agent1, player_move, opponent_move, reward)
        if mode in {"human_bot2", "bot_bot"} and agent2 is not None and opponent_move >= 0:
            _observe_agent(agent2, opponent_move, player_move, -reward)
    state["history"] = [dict(item["log"]) for item in reversed(entries)]
    st.session_state["round_count"] = env.state.turn
    st.session_state["game_over"] = state["done"]
    st.session_state["winner"] = _winner_from_scores(env) if state["done"] else None


def _rebuild_match_from_history(trim_to: int) -> None:
    prior_moves = list(st.session_state.get("move_history", []))[:trim_to]
    state = st.session_state.game_state
    _apply_game_settings(
        state["game_type"],
        state.get("agent_name", "random"),
        int(getattr(state["env"], "max_turns", getattr(state["env"], "max_moves", 30))),
        state.get("play_mode", "human_bot1"),
        state.get("agent2_name"),
    )
    st.session_state["move_history"] = []
    if prior_moves:
        _replay_saved_turns(prior_moves)
        st.session_state["move_history"] = prior_moves


def _undo_last_turn() -> None:
    history = list(st.session_state.get("move_history", []))
    if not history:
        return
    trim_to = max(0, len(history) - 1)
    _rebuild_match_from_history(trim_to)
    st.toast("Undid the last turn.")


def _restart_match() -> None:
    state = st.session_state.game_state
    _apply_game_settings(
        state["game_type"],
        state.get("agent_name", "random"),
        int(getattr(state["env"], "max_turns", getattr(state["env"], "max_moves", 30))),
        state.get("play_mode", "human_bot1"),
        state.get("agent2_name"),
    )
    st.toast("Game restarted")


def _play_again() -> None:
    if _ladder_enabled():
        ladder_state = st.session_state.setdefault("ladder_state", {})
        rungs = _current_ladder_rungs()
        won_rung = st.session_state.get("winner") == "you"
        if ladder_state.get("completed"):
            _reset_ladder_state()
        elif won_rung and int(ladder_state.get("rung_index", 0)) < max(len(rungs) - 1, 0):
            ladder_state["rung_index"] = int(ladder_state.get("rung_index", 0)) + 1
        elif won_rung and int(ladder_state.get("rung_index", 0)) >= max(len(rungs) - 1, 0):
            ladder_state["completed"] = True
            _persist_ladder_result()
    if _series_complete():
        st.session_state["score_you"] = 0
        st.session_state["score_agent"] = 0
        st.session_state["score_draws"] = 0
        st.session_state["series_complete"] = False
    _restart_match()


def _change_agent_reset() -> None:
    st.session_state["focus_match_setup"] = True
    st.session_state.setdefault("game_settings", {})["clone_ladder_enabled"] = False
    st.session_state.pop("ladder_state", None)
    _reset_blind_study_state()
    _restart_match()


def _toggle_pause() -> None:
    st.session_state["paused"] = not st.session_state.get("paused", False)


def _pieces_captured_stats(env, game_type: str) -> str | None:
    if game_type == "Checkers":
        you_left = int(np.sum(env.board > 0))
        opp_left = int(np.sum(env.board < 0))
        return f"Captured pieces: you {12 - opp_left}, agent {12 - you_left}"
    if game_type == "Chess" and getattr(env, "_board", None) is not None:
        pieces_left = len(getattr(env._board, "piece_map", lambda: {})())
        return f"Captured pieces total: {32 - pieces_left}"
    return None


def _game_over_stats(env, game_type: str) -> list[tuple[str, str]]:
    stats = [("Total moves/rounds", str(env.state.turn))]
    if game_type == "RPS+" and getattr(env.state, "history", None):
        counts = {}
        for item in env.state.history:
            label = getattr(item.agent_move, "name", str(item.agent_move))
            counts[label] = counts.get(label, 0) + 1
        if counts:
            stats.append(("Most used move", max(counts, key=counts.get)))
    capture_line = _pieces_captured_stats(env, game_type)
    if capture_line:
        label, value = capture_line.split(": ", 1)
        stats.append((label, value))
    if game_type == "Othello":
        stats.append(("Final disc count", f"You {int(np.sum(env.board == 1))} · Agent {int(np.sum(env.board == -1))}"))
    if game_type in {"Gomoku", "Tic-Tac-Toe", "Connect Four"}:
        stats.append(("Winning move number", str(env.state.turn)))
    if game_type == "Nim":
        stats.append(("Matches remaining", str(int(np.sum(env.piles)))))
    return stats


def _run_counterfactual_replay() -> None:
    state = st.session_state.game_state
    if state.get("game_type") != "RPS+":
        st.toast("Counterfactual replay is available for RPS+ right now.")
        return
    player_actions = [
        int(entry.get("player_action"))
        for entry in st.session_state.get("move_history", [])
        if entry.get("player_action") is not None
    ]
    if not player_actions:
        st.toast("No playable history yet for counterfactual replay.")
        return
    source_game_id = state.get("last_saved_game_id") or uuid.uuid4().hex[:12]
    player_id = _active_user_profile()["id"]
    comparison_agents = ["random", "heuristic", "ngram"]
    baseline_results: list[int] = []
    clone_results: list[int] = []
    for idx, agent_name in enumerate(comparison_agents):
        agent = _new_agent(agent_name if agent_name != "ngram" else state.get("agent_name", "ngram"))
        env = RPSPlusEnv(
            max_turns=len(player_actions),
            starting_energy=int(st.session_state.get("game_settings", {}).get("rps_starting_energy", 3)),
        )
        obs, info = env.reset(seed=idx + 11)
        for player_move in player_actions:
            def policy(local_obs=obs, local_info=info, move=player_move):
                return _agent_action(agent, env, "RPS+", local_obs, local_info, current_player_move=move, side=-1)
            obs, reward, terminated, truncated, info = _step_with_injected_policy(env, player_move, policy)
            if terminated or truncated:
                break
        final = _winner_from_scores(env)
        score = 1 if final == "you" else -1 if final == "agent" else 0
        if agent_name in {"ngram", "lstm", state.get("agent_name")}:
            clone_results.append(score)
        else:
            baseline_results.append(score)
    replay_id = persist_counterfactual_replay(
        db_path=str(DEFAULT_DB),
        source_game_id=source_game_id,
        player_id=player_id,
        game_type="RPS+",
        baseline_agent="random+heuristic",
        clone_agent=state.get("agent_name", "clone"),
        source_result=st.session_state.get("winner", "draw"),
        baseline_results=baseline_results,
        clone_results=clone_results,
    )
    st.session_state["last_counterfactual_replay_id"] = replay_id
    st.toast("Counterfactual replay saved.")


def _render_score_tracker(player_label: str, opponent_label: str) -> None:
    cols = st.columns(3, gap="small")
    cols[0].metric(f"{player_label} wins", st.session_state.get("score_you", 0))
    cols[1].metric("Draws", st.session_state.get("score_draws", 0))
    cols[2].metric(f"{opponent_label} wins", st.session_state.get("score_agent", 0))
    st.caption(
        f"Best of {_series_length()} · first to {_series_wins_needed()} wins"
    )
    if _series_complete():
        leader = player_label if st.session_state.get("score_you", 0) >= _series_wins_needed() else opponent_label
        st.success(f"Series complete: {leader} wins the set.")


def _progress_payload(env, game_type: str) -> tuple[str, float]:
    turn = int(getattr(env.state, "turn", 0))
    total = int(getattr(env, "max_turns", getattr(env, "max_moves", max(1, turn or 1))))
    if game_type == "RPS+":
        return f"Round {turn} of {total}", min(1.0, turn / max(1, total))
    if game_type == "Nim":
        remaining = int(np.sum(env.piles))
        initial = int(np.sum(getattr(env, "initial_piles", env.piles)))
        return f"Matches remaining: {remaining} / {initial}", 1.0 - (remaining / max(1, initial))
    if game_type == "Connect Four":
        pieces = int(np.count_nonzero(env.board))
        return f"Pieces played: {pieces} / 42", pieces / 42
    if game_type == "Tic-Tac-Toe":
        pieces = int(np.count_nonzero(env.board))
        return f"Move {pieces} of 9", pieces / 9
    if game_type in {"Chess", "Checkers", "Othello", "Gomoku"}:
        estimate = total if game_type != "Gomoku" else getattr(env, "board_size", 9) ** 2
        return f"Move {turn} · estimated length {estimate}", min(1.0, turn / max(1, estimate))
    return f"Turn {turn} of {total}", min(1.0, turn / max(1, total))


def _render_game_progress(env, game_type: str, player_label: str, opponent_label: str) -> None:
    label, progress = _progress_payload(env, game_type)
    st.caption("Game Progress")
    st.progress(progress, text=label)
    _render_score_tracker(player_label, opponent_label)
    if _ladder_enabled():
        ladder_state = st.session_state.get("ladder_state") or {}
        rungs = _current_ladder_rungs()
        rung_index = int(ladder_state.get("rung_index", 0))
        chips = []
        for idx, rung in enumerate(rungs):
            marker = "●" if idx == rung_index and not ladder_state.get("completed") else "✓" if idx < rung_index or ladder_state.get("completed") else "○"
            chips.append(f"{marker} {rung['agent']}")
        st.caption(
            f"Clone Ladder · rung {min(rung_index + 1, max(len(rungs), 1))}/{max(len(rungs), 1)} · "
            f"{'completed' if ladder_state.get('completed') else (_active_ladder_rung() or {}).get('title', 'Active')} · "
            f"{'  |  '.join(chips)}"
        )


def _render_hint_panel(game_type: str, env, info: dict) -> None:
    if not st.session_state.get("show_hints", DEFAULT_HINTS):
        return
    hint = contextual_hint(game_type, info or {}, env=env)
    static_hints = GAME_HINTS.get(game_type, [])
    st.markdown(
        f"""
        <div class="move-ribbon" style="margin-top:10px;">
            <strong>Tip for this turn</strong><br/>
            {hint}
            {"<br/><span style='opacity:.75'>" + static_hints[0] + "</span>" if static_hints else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_clone_explanation(game_type: str, agent) -> None:
    explain = getattr(agent, "explain_prediction", None)
    if not callable(explain):
        return
    history = [
        int(entry.get("player_action"))
        for entry in st.session_state.get("move_history", [])
        if entry.get("player_action") is not None
    ]
    legal = st.session_state.game_state.get("info", {}).get("legal_moves")
    try:
        message = explain(history=history, legal=legal)
    except TypeError:
        message = explain(history, legal)
    st.caption("How this opponent tends to think")
    st.markdown(message)


def _compute_move_surprisal(agent, realized_move: int, history: list[int], legal: list[int] | None) -> dict[str, float | int | str] | None:
    surprisal_fn = getattr(agent, "surprisal", None)
    explain = getattr(agent, "explain_prediction", None)
    if not callable(surprisal_fn):
        return None
    try:
        stats = surprisal_fn(realized_move=realized_move, history=history, legal=legal)
    except TypeError:
        stats = surprisal_fn(realized_move, history, legal)
    explanation = ""
    if callable(explain):
        try:
            explanation = explain(history=history, legal=legal)
        except TypeError:
            explanation = explain(history, legal)
    return {
        "predicted_prob": float(stats.get("predicted_prob", 0.0)),
        "surprisal": float(stats.get("surprisal", 0.0)),
        "confidence": float(stats.get("confidence", 0.0)),
        "entropy": float(stats.get("entropy", 0.0)),
        "expected_action": int(stats.get("expected_action", -1)),
        "explanation": explanation,
    }


def _surprisal_summary_json() -> str:
    summary = summarize_surprisal_history(st.session_state.game_state.get("surprisal_history", []))
    return json.dumps(summary)


def _record_detection_judgment(confidence: float, judged_human: bool) -> None:
    state = st.session_state.game_state
    if state.get("detection_recorded"):
        return
    profile = _active_user_profile()
    manager = PlayerProfileManager(DEFAULT_DB)
    blind_entry = _current_blind_entry() or {}
    impostor_type = blind_entry.get("condition") if _blind_study_active() else state.get("agent_name", "unknown")
    source_player_id = st.session_state.get("game_settings", {}).get("friend_clone_source_player")
    manager.record_detection_session(
        profile["id"],
        str(impostor_type),
        detected_as_human=judged_human,
        game_id=state.get("last_saved_game_id"),
        confidence=float(confidence),
        source_player_id=source_player_id,
        study_block_id=(st.session_state.get("blind_study_state") or {}).get("block_id"),
        blind_label=blind_entry.get("blind_label"),
        surprisal_summary_json=_surprisal_summary_json(),
    )
    state["detection_recorded"] = True
    if _blind_study_active():
        _advance_blind_study()


def _why_you_were_fooled_note() -> str:
    agent = st.session_state.game_state.get("agent")
    history = [int(entry.get("player_action")) for entry in st.session_state.get("move_history", []) if entry.get("player_action") is not None]
    legal = st.session_state.game_state.get("info", {}).get("legal_moves")
    uncertainty = getattr(agent, "uncertainty", None)
    explain = getattr(agent, "explain_prediction", None)
    note = "The opponent matched your local tempo closely enough to feel human."
    if callable(uncertainty):
        stats = uncertainty(history=history, legal=legal)
        if float(stats.get("confidence", 0.0)) < 0.35:
            note = "You may have read a high-entropy clone turn as human creativity."
    if callable(explain):
        try:
            explanation = explain(history=history, legal=legal)
        except TypeError:
            explanation = explain(history, legal)
        if explanation:
            note = f"{note} {explanation}"
    surprisal_history = st.session_state.game_state.get("surprisal_history", [])
    if surprisal_history:
        summary = summarize_surprisal_history(surprisal_history)
        if int(summary.get("high_surprisal_turns", 0)) == 0:
            note = f"The clone kept your moves low-surprisal throughout the match. {note}"
    return note


def _render_standard_detection_prompt() -> None:
    if not st.session_state.get("game_over"):
        return
    prompt = standardized_detection_prompt()
    with st.expander(prompt["title"], expanded=not st.session_state.game_state.get("detection_recorded", False)):
        st.caption(prompt["question"])
        confidence = st.slider(
            prompt["scale_label"],
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            key="standard_detection_confidence",
        )
        col1, col2 = st.columns(2, gap="medium")
        col1.button(
            "Felt Like AI",
            width='stretch',
            disabled=st.session_state.game_state.get("detection_recorded", False),
            on_click=_record_detection_judgment,
            args=(confidence, False),
        )
        col2.button(
            "Felt Like Human",
            width='stretch',
            disabled=st.session_state.game_state.get("detection_recorded", False),
            on_click=_record_detection_judgment,
            args=(confidence, True),
        )
        if st.session_state.game_state.get("detection_recorded"):
            st.info("Detection judgment recorded.")
            st.caption(_why_you_were_fooled_note())


def _render_game_controls(game_type: str) -> None:
    st.divider()
    state = st.session_state.game_state
    can_pause = state.get("play_mode") != "human_human"
    pause_label = "Resume" if st.session_state.get("paused") else "Pause"
    cols = st.columns(4, gap="small")
    cols[0].button(f"{pause_label}", width='stretch', on_click=_toggle_pause, disabled=not can_pause)
    cols[1].button("Undo", width='stretch', on_click=_undo_last_turn, disabled=not bool(st.session_state.get("move_history")) or len(st.session_state.get("move_history", [])) > 10)
    cols[2].button("Restart", width='stretch', on_click=_restart_match)
    with cols[3]:
        with st.expander("Settings", expanded=bool(st.session_state.get("focus_match_setup"))):
            rounds_to_play = st.slider("Turn limit", 1, 20, min(10, int(st.session_state.get("sidebar_max_turns", 10))), key="settings_rounds_to_play")
            series_length = st.select_slider("Series length", options=[1, 3, 5, 7], value=int(st.session_state.get("game_settings", {}).get("series_length", 3)))
            agent_difficulty = st.select_slider("Agent difficulty", ["Easy", "Medium", "Hard"], key="settings_agent_difficulty")
            st.toggle("Show hints", key="show_hints")
            game_settings = st.session_state.setdefault("game_settings", {})
            game_settings["series_length"] = int(series_length)
            game_settings["clone_ladder_enabled"] = st.toggle(
                "Draft Mode Clone Ladder",
                value=bool(game_settings.get("clone_ladder_enabled", False)),
                help="RPS+ only. Face random, heuristic, ngram, lstm, then adaptive_router in sequence.",
            )
            blind_mode = st.toggle(
                "Blind Turing Study",
                value=bool(game_settings.get("blind_turing_study", False)),
                help="Hide clone identity, randomize conditions, and use a standardized postgame detection prompt.",
            )
            game_settings["blind_turing_study"] = blind_mode
            if game_type == "RPS+":
                game_settings["rps_starting_energy"] = st.slider("Starting energy", 1, 5, int(game_settings.get("rps_starting_energy", 3)))
                profiles = PlayerProfileManager(DEFAULT_DB).list_all()
                if profiles:
                    source_options = {profile.display_name: profile.player_id for profile in profiles}
                    current_source = game_settings.get("friend_clone_source_player", _active_user_profile()["id"])
                    source_labels = list(source_options.keys())
                    current_label = next((label for label, pid in source_options.items() if pid == current_source), source_labels[0])
                    selected_source = st.selectbox("Clone source player", source_labels, index=source_labels.index(current_label))
                    game_settings["friend_clone_source_player"] = source_options[selected_source]
            elif game_type == "Chess":
                game_settings["chess_time_control"] = st.selectbox("Time control", ["Unlimited", "5min", "10min"], index=["Unlimited", "5min", "10min"].index(game_settings.get("chess_time_control", "Unlimited")))
            elif game_type == "Gomoku":
                game_settings["gomoku_board_size"] = st.selectbox("Board size", [13, 15, 19], index=[13, 15, 19].index(int(game_settings.get("gomoku_board_size", 15))))
            elif game_type == "Nim":
                game_settings["nim_preset"] = st.selectbox("Pile configuration", ["Classic", "Fibonacci", "Custom"], index=["Classic", "Fibonacci", "Custom"].index(game_settings.get("nim_preset", "Classic")))
                if game_settings["nim_preset"] == "Custom":
                    raw = st.text_input("Custom piles", value="1,3,5,7")
                    try:
                        game_settings["nim_custom_piles"] = [max(1, int(part.strip())) for part in raw.split(",") if part.strip()]
                    except ValueError:
                        st.caption("Use comma-separated integers, for example 1,3,5,7.")
            elif game_type == "Checkers":
                game_settings["checkers_forced_jump"] = st.toggle("Forced jump", value=bool(game_settings.get("checkers_forced_jump", True)))
            challenges = _challenge_catalog(game_type)
            if challenges:
                challenge_labels = ["Standard opening"] + [challenge["label"] for challenge in challenges]
                current_challenge = game_settings.get("opening_challenge")
                current_label = "Standard opening"
                for challenge in challenges:
                    if challenge["id"] == current_challenge:
                        current_label = challenge["label"]
                        break
                selected_label = st.selectbox("Opening / trap challenge", challenge_labels, index=challenge_labels.index(current_label))
                game_settings["opening_challenge"] = next(
                    (challenge["id"] for challenge in challenges if challenge["label"] == selected_label),
                    None,
                )
            if st.button("Apply Settings", width='stretch', key=f"apply_settings_{game_type}"):
                st.session_state["sidebar_max_turns"] = rounds_to_play
                st.session_state["focus_match_setup"] = False
                if game_settings.get("clone_ladder_enabled"):
                    _reset_ladder_state()
                else:
                    st.session_state.pop("ladder_state", None)
                if game_settings.get("blind_turing_study"):
                    _start_blind_study(_active_user_profile()["id"], game_type="RPS+")
                else:
                    _reset_blind_study_state()
                _sync_score_scope(reset=True)
                _apply_game_settings(
                    state["game_type"],
                    state.get("agent_name", "random"),
                    int(rounds_to_play),
                    state.get("play_mode", "human_bot1"),
                    state.get("agent2_name"),
                )
                st.rerun()


def _render_game_over_panel(game_type: str, env, player_label: str, opponent_label: str) -> None:
    if not st.session_state.get("game_over"):
        return
    winner = st.session_state.get("winner")
    if winner == "you":
        st.success("YOU WIN" if st.session_state.game_state.get("play_mode") != "bot_bot" else f"{player_label} WINS")
    elif winner == "agent":
        st.error("YOU LOSE" if st.session_state.game_state.get("play_mode") != "bot_bot" else f"{opponent_label} WINS")
    else:
        st.warning("DRAW")
    if _series_complete():
        st.success(
            f"Series result: {player_label if st.session_state.get('score_you', 0) >= _series_wins_needed() else opponent_label} wins best-of-{_series_length()}."
        )
    else:
        st.caption(
            f"Series standing: {st.session_state.get('score_you', 0)} - {st.session_state.get('score_agent', 0)} with {st.session_state.get('score_draws', 0)} draw(s)."
        )
    if _ladder_enabled():
        ladder_state = st.session_state.get("ladder_state") or {}
        rung = _active_ladder_rung() or {}
        if ladder_state.get("completed"):
            st.success("Clone Ladder cleared. You beat the full imitation gauntlet.")
        elif st.session_state.get("winner") == "you":
            st.caption(f"Rung cleared: {rung.get('title', rung.get('agent', 'clone'))}. Continue to the next clone.")
        elif st.session_state.get("winner") == "agent":
            st.caption(f"Rung failed against {rung.get('agent', 'clone')}. Retry this rung or change setup.")
        else:
            st.caption("Drawn rung. Retry to keep climbing the ladder.")
    stat_cols = st.columns(max(1, min(3, len(_game_over_stats(env, game_type)))), gap="small")
    for col, (label, value) in zip(stat_cols, _game_over_stats(env, game_type)):
        with col:
            st.metric(label, value)
    surprisal_summary = summarize_surprisal_history(st.session_state.game_state.get("surprisal_history", []))
    if int(surprisal_summary.get("n_turns", 0)) > 0:
        st.caption("Move Surprisal Summary")
        surprisal_cols = st.columns(3, gap="small")
        surprisal_cols[0].metric("Mean Surprisal", f"{float(surprisal_summary['mean_surprisal']):.2f}")
        surprisal_cols[1].metric("Peak Surprisal", f"{float(surprisal_summary['max_surprisal']):.2f}")
        surprisal_cols[2].metric("High-Surprise Turns", int(surprisal_summary["high_surprisal_turns"]))
    action_cols = st.columns(2, gap="medium")
    ladder_label = None
    if _ladder_enabled():
        ladder_state = st.session_state.get("ladder_state") or {}
        if ladder_state.get("completed"):
            ladder_label = "Start New Ladder"
        elif st.session_state.get("winner") == "you":
            ladder_label = "Next Clone"
        else:
            ladder_label = "Retry Rung"
    action_cols[0].button(ladder_label or ("New Series" if _series_complete() else "Play Again"), width='stretch', type="primary", on_click=_play_again)
    action_cols[1].button("Change Agent", width='stretch', on_click=_change_agent_reset)
    if game_type == "RPS+":
        st.button("Run Counterfactual Replay", width='stretch', on_click=_run_counterfactual_replay)
    _render_standard_detection_prompt()


def _select_checkers_square(idx):
    st.session_state.game_state["checkers_selected"] = idx


def _clear_checkers_square():
    st.session_state.game_state["checkers_selected"] = None


def _select_sudoku_cell(idx):
    st.session_state.game_state["sudoku_selected_cell"] = idx


def _select_chess_square(idx):
    st.session_state.game_state["chess_selected_sq"] = idx


def _clear_chess_square():
    st.session_state.game_state["chess_selected_sq"] = None


def _generic_history_to_records(history, game_id):
    records = []
    for idx, round_obj in enumerate(history):
        turn = _safe_int(getattr(round_obj, "turn", idx + 1), idx + 1)
        agent_move = _safe_int(getattr(round_obj, "agent_move", -1), -1)
        opp_move = _safe_int(getattr(round_obj, "opponent_move", -1), -1)
        records.append(
            {
                "game_id": game_id,
                "turn": turn,
                "agent_move": agent_move,
                "agent_move_name": _safe_name(getattr(round_obj, "agent_move", agent_move), str(agent_move)),
                "opponent_move": opp_move,
                "opponent_move_name": _safe_name(getattr(round_obj, "opponent_move", opp_move), str(opp_move)),
                "outcome": _safe_int(getattr(round_obj, "outcome", 0), 0),
                "agent_energy_after": _safe_int(getattr(round_obj, "agent_energy_after", 0), 0),
                "opponent_energy_after": _safe_int(getattr(round_obj, "opponent_energy_after", 0), 0),
            }
        )
    return records


def _save_live_game_to_profile(env, game_type, player_id, player_name, opponent_name, db_path=DEFAULT_DB):
    if not player_id:
        return False
    if not hasattr(env, "state") or not getattr(env.state, "history", None):
        return False

    init_db(db_path)
    init_extended_db(db_path)
    conn = connect(db_path)
    try:
        game_id = uuid.uuid4().hex
        records = history_to_records(env.state.history, game_id) if game_type == "RPS+" else _generic_history_to_records(env.state.history, game_id)
        insert_game(
            conn,
            game_id=game_id,
            agent_name=player_id,
            opponent_name=opponent_name,
            game_type=game_type,
            seed=None,
            n_turns=env.state.turn,
            agent_score=env.state.agent_score,
            opponent_score=env.state.opponent_score,
            rounds=records,
        )
        conn.commit()

        valid_records = [
            r for r in records
            if 0 <= _safe_int(r["agent_move"]) < N_MOVES and 0 <= _safe_int(r["opponent_move"]) < N_MOVES
        ]
        if valid_records and not player_id.startswith("guest_"):
            manager = PlayerProfileManager(db_path)
            manager.update_signature(
                player_id,
                [[_safe_int(r["agent_move"]) for r in valid_records]],
                [[_safe_int(r["outcome"], 0) for r in valid_records]],
                [[_safe_int(r["opponent_move"]) for r in valid_records]],
                game_type=game_type,
            )
        _refresh_profile_rollup(player_id, db_path)
    finally:
        conn.close()
    st.session_state.game_state["last_saved_game_id"] = game_id
    st.toast(f"Saved game as {player_id}.")
    return True

def get_move_emoji(move_idx, game_type="RPS+", env=None):
    if game_type == "RPS+":
        return {
            0: "🪨 ROCK", 1: "📄 PAPER", 2: "✂️ SCISSORS",
            3: "🦎 LIZARD", 4: "⚡ POWER", 5: "🔋 RECHARGE"
        }.get(move_idx, "❓")
    elif game_type == "Chess" and env and hasattr(env, "_board"):
        try:
            moves = list(env._board.legal_moves)
            if 0 <= move_idx < len(moves):
                return env._board.san(moves[move_idx])
        except Exception:
            pass
    elif game_type in FUTURE_GAME_LABELS:
        return future_action_label(game_type, move_idx, env)
    return str(move_idx)


def future_action_label(game_type, action, env=None):
    if action is None:
        return "Environment"
    if game_type == "2048":
        return ["Slide Up", "Slide Right", "Slide Down", "Slide Left"][action]
    if game_type == "Candy Crush":
        cell, direction = divmod(action, 4)
        r, c = divmod(cell, 6)
        dirs = ["up", "right", "down", "left"]
        return f"Swap row {r + 1}, col {c + 1} {dirs[direction]}"
    if game_type == "Wordle" and env is not None:
        return env.WORDS[action].upper()
    if game_type == "Sudoku":
        size = env.board.shape[0] if env is not None else 9
        cell, value_idx = divmod(action, size)
        r, c = divmod(cell, size)
        return f"Row {r + 1}, col {c + 1} = {value_idx + 1}"
    if game_type == "Pac-Man":
        return ["Move Up", "Move Right", "Move Down", "Move Left"][action]
    if game_type == "Minecraft":
        return ["Gather wood", "Mine stone", "Find food", "Craft tool", "Build shelter", "Rest and eat"][action]
    if game_type == "Among Us":
        return ["Accuse Red", "Accuse Blue", "Accuse Green", "Accuse Yellow", "Do task", "Gather clues"][action]
    if game_type == "Clash Royale":
        return ["Wait", "Send Knight", "Send Archers", "Send Giant", "Defend tower"][action]
    if game_type == "Flappy Bird":
        return ["Glide", "Flap"][action]
    if game_type == "Ludo":
        return f"Move token {action + 1}"
    if game_type == "UNO":
        return ["Play Red", "Play Blue", "Play Green", "Play Yellow", "Draw card"][action]
    if game_type == "Scrabble" and env is not None:
        word = env.WORDS[action]
        return f"Play {word.upper()}"
    if game_type == "Monopoly":
        return ["Roll / collect", "Buy property", "Sell property"][action]
    if game_type == "Penalty Shootout":
        return ["Left", "Low left", "Center", "Low right", "Right"][action]
    if game_type == "Cricket Strategy":
        return ["Defend", "Take singles", "Attack for four", "Swing for six"][action]
    return f"Action {action}"


def _metric_row(items):
    cols = st.columns(len(items))
    for col, (label, value) in zip(cols, items):
        col.metric(label, value)


def _render_value_grid(values, labels=None, key_prefix="grid", color_map=None):
    rows, cols_count = values.shape
    labels = labels or {}
    color_map = color_map or {}
    style = f"grid-template-columns: repeat({cols_count}, minmax(0, 1fr));"
    html = [f'<div class="value-board" style="{style}">']
    for r in range(rows):
        for c in range(cols_count):
            value = values[r, c]
            text = labels.get((r, c), str(int(value)) if value else "&nbsp;")
            extra = " empty" if not value and text in {"&nbsp;", ".", " "} else ""
            tile_style = color_map.get((r, c), "")
            html.append(f'<div class="value-tile{extra}" style="{tile_style}">{text}</div>')
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def _render_connect_four_fallback(board):
    symbols = {1: "X", -1: "O", 0: "."}
    lines = []
    for row in board.T[::-1]:
        lines.append(" ".join(symbols[int(cell)] for cell in row))
    st.code("\n".join(lines), language="text")


def _render_legend(items):
    pills = "".join(f'<div class="legend-pill">{item}</div>' for item in items)
    st.markdown(f'<div class="legend-row">{pills}</div>', unsafe_allow_html=True)


def _render_connect_four_board(board):
    token = {1: "🟡", -1: "🔴", 0: ""}
    cells = []
    for row in board.T[::-1]:
        for cell in row:
            cells.append(f'<div class="connect-slot">{token[int(cell)]}</div>')
    st.markdown(f'<div class="board-shell"><div class="connect-board">{"".join(cells)}</div></div>', unsafe_allow_html=True)


def _render_2048_board(env):
    tone_map = {
        0: "background: rgba(255,255,255,0.26); color: rgba(32,26,23,0.24);",
        2: "background: #f6e7c8;",
        4: "background: #f3d59a;",
        8: "background: #efb869; color: white;",
        16: "background: #dc9456; color: white;",
        32: "background: #cf6d47; color: white;",
        64: "background: #b84f3f; color: white;",
        128: "background: #9850b4; color: white;",
        256: "background: #6d52cc; color: white;",
        512: "background: #4a6ddf; color: white;",
        1024: "background: #2d8f7a; color: white;",
        2048: "background: #d88c2f; color: white;",
    }
    rows = ['<div class="board-shell"><div class="value-board" style="grid-template-columns: repeat(4, minmax(0, 1fr));">']
    for r in range(4):
        for c in range(4):
            value = int(env.board[r, c])
            tone = tone_map.get(value, "background: #7e573b; color: white;")
            label = str(value) if value else "&nbsp;"
            rows.append(f'<div class="value-tile tile-pop{" empty" if value == 0 else ""}" style="min-height:74px;font-size:24px;{tone}">{label}</div>')
    rows.append("</div></div>")
    st.markdown("".join(rows), unsafe_allow_html=True)


def _render_sudoku_board(env, legal_cells, selected_cell):
    size = env.board.shape[0]
    rows = ['<div class="board-shell">']
    for r in range(size):
        rows.append('<div style="display:grid; grid-template-columns: repeat(9, minmax(0, 1fr)); gap: 4px; margin-bottom: 4px;">')
        for c in range(size):
            idx = r * size + c
            value = int(env.board[r, c])
            border = ""
            if c in {2, 5}:
                border += " border-right: 2px solid rgba(54,38,28,0.16);"
            if r in {2, 5}:
                border += " border-bottom: 2px solid rgba(54,38,28,0.16);"
            if value:
                style = "background: #fff7ea;"
                label = str(value)
            elif idx == selected_cell:
                style = "background: #f9dfb5; color: #6f4a12; border: 1px solid rgba(216,140,47,0.35);"
                label = "."
            elif idx in legal_cells:
                style = "background: rgba(255,255,255,0.58);"
                label = "."
            else:
                style = "background: rgba(255,255,255,0.35); color: rgba(32,26,23,0.48);"
                label = "."
            rows.append(f'<div class="value-tile{" empty" if not value else ""}" style="min-height:40px; font-size:14px; border-radius:10px; {style}{border}">{label}</div>')
        rows.append("</div>")
    rows.append("</div>")
    st.markdown("".join(rows), unsafe_allow_html=True)


def _render_othello_board(board, legal_moves):
    rows = ['<div class="board-shell"><div class="othello-board">']
    for r in range(8):
        rows.append('<div style="display:grid; grid-template-columns: repeat(8, minmax(0, 1fr)); gap: 8px; margin-bottom: 8px;">')
        for c in range(8):
            idx = r * 8 + c
            val = int(board[r, c])
            if val == 1:
                label = "⚫"
            elif val == -1:
                label = "⚪"
            elif idx in legal_moves:
                label = "·"
            else:
                label = "&nbsp;"
            rows.append(
                '<div class="value-tile" style="min-height:54px; border-radius:16px; '
                'background: rgba(255,255,255,0.18); color: white; border-color: rgba(255,255,255,0.10);">'
                f"{label}</div>"
            )
        rows.append("</div>")
    rows.append("</div></div>")
    st.markdown("".join(rows), unsafe_allow_html=True)


def _render_checkers_board(board, selected, valid_from, valid_to):
    rows = ['<div class="board-shell"><div class="checkers-board">']
    for r in range(8):
        rows.append('<div style="display:grid; grid-template-columns: repeat(8, minmax(0, 1fr)); gap: 6px; margin-bottom: 6px;">')
        for c in range(8):
            idx = r * 8 + c
            val = int(board[idx])
            is_dark = (r + c) % 2 == 1
            if val == 1:
                label = "🔴"
            elif val == 2:
                label = "👑"
            elif val == -1:
                label = "⚫"
            elif val == -2:
                label = "♛"
            elif idx in valid_to:
                label = "✦"
            else:
                label = "&nbsp;"

            if not is_dark:
                tile_style = "background: rgba(255,255,255,0.50); color: rgba(32,26,23,0.16);"
            elif idx == selected:
                tile_style = "background: #f9dfb5; color: #6f4a12; border: 1px solid rgba(216,140,47,0.35);"
            elif idx in valid_to:
                tile_style = "background: rgba(255,243,214,0.78); color: #9f6812;"
            elif idx in valid_from:
                tile_style = "background: rgba(255,255,255,0.18); color: white; border-color: rgba(255,255,255,0.10);"
            else:
                tile_style = "background: rgba(0,0,0,0.12); color: white; border-color: rgba(255,255,255,0.08);"

            rows.append(f'<div class="value-tile" style="min-height:52px; border-radius:14px; {tile_style}">{label}</div>')
        rows.append("</div>")
    rows.append("</div></div>")
    st.markdown("".join(rows), unsafe_allow_html=True)


def _render_gomoku_board(board, legal_moves):
    labels = {}
    styles = {}
    legal = set(legal_moves)
    for r in range(board.shape[0]):
        for c in range(board.shape[1]):
            idx = r * board.shape[1] + c
            value = int(board[r, c])
            if value == 1:
                labels[(r, c)] = "●"
                styles[(r, c)] = "background:#dff7ee;color:#0f5132;font-size:24px;"
            elif value == -1:
                labels[(r, c)] = "○"
                styles[(r, c)] = "background:#ffe1df;color:#8a1f17;font-size:24px;"
            elif idx in legal:
                labels[(r, c)] = "·"
                styles[(r, c)] = "background:var(--arena-tile-empty-bg);color:var(--arena-muted);"
            else:
                labels[(r, c)] = ""
                styles[(r, c)] = ""
    _render_value_grid(board, labels, "gomoku", styles)


def _nim_action_label(action: int) -> str:
    pile_idx, remove_idx = divmod(int(action), 3)
    return f"Pile {pile_idx + 1}: take {remove_idx + 1}"


def _render_pacman_board(env):
    rows = ['<div class="board-shell"><div class="pacman-board">']
    height, width = env.grid.shape
    for r in range(height):
        rows.append(f'<div style="display:grid; grid-template-columns: repeat({width}, minmax(0, 1fr)); gap: 8px; margin-bottom: 8px;">')
        for c in range(width):
            if env.walls[r, c]:
                label = "█"
                tile_style = "background: #33261f; color: #f1d4b0;"
            elif (r, c) == env.player:
                label = "ᗧ"
                tile_style = "background: #f7e08a; color: #6b4d00;"
            elif (r, c) == env.ghost:
                label = "◉" if env.frightened_turns == 0 else "◎"
                tile_style = "background: #f6c7d3; color: #9d2446;" if env.frightened_turns == 0 else "background: #b7d2ff; color: #1f4d9a;"
            elif getattr(env, "power_pellets", None) is not None and env.power_pellets[r, c]:
                label = "◌"
                tile_style = "background: #d9e5ff; color: #3258a3;"
            elif env.pellets[r, c]:
                label = "•"
                tile_style = "background: #fff8eb; color: #a77a1f;"
            else:
                label = "&nbsp;"
                tile_style = "background: rgba(255,255,255,0.08); color: white;"
            rows.append(f'<div class="value-tile" style="min-height:56px; border-radius:16px; {tile_style}">{label}</div>')
        rows.append("</div>")
    rows.append("</div></div>")
    st.markdown("".join(rows), unsafe_allow_html=True)


def _render_game_intro(title, description, turn_text, opponent_name):
    st.markdown(
        f"""
        <div class="game-stage-header">
            <div class="game-stage-title">
                <div class="section-label">Now Playing</div>
                <h2>{title}</h2>
                <p>{description}</p>
            </div>
            <div class="game-stage-stats">
                <div>
                    <strong>Turn</strong>
                    <span>{turn_text}</span>
                </div>
                <div>
                    <strong>Against</strong>
                    <span>{opponent_name}</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_move_ribbon(text):
    st.markdown(f'<div class="turn-banner"><span></span><strong>{text}</strong></div>', unsafe_allow_html=True)


def _player_avatar(label: str, fallback: str) -> str:
    cleaned = "".join(ch for ch in str(label or "") if ch.isalnum())
    return (cleaned[:2] or fallback).upper()


def _render_match_dashboard(
    game_type,
    player_label,
    opponent_label,
    active_label,
    mode_label,
    status_label,
    turn_text,
    max_turns,
    agent_score,
    opponent_score,
):
    turn_pct = 0
    try:
        current_turn = max(0, int(str(turn_text).split("/")[0]))
        turn_limit = max(1, int(max_turns))
        turn_pct = min(100, round((current_turn / turn_limit) * 100))
    except Exception:
        turn_pct = 0

    player_active = " is-active" if active_label == player_label else ""
    opponent_active = " is-active" if active_label == opponent_label else ""
    player_name = html.escape(str(player_label))
    opponent_name = html.escape(str(opponent_label))
    game_name = html.escape(str(game_type))
    status = html.escape(str(status_label))
    mode = html.escape(str(mode_label))
    turn = html.escape(str(turn_text))
    description = html.escape(_game_description(game_type))
    st.markdown(
        f"""
        <section class="match-dashboard" aria-label="Current match">
            <div class="arena-command">
                <div class="brand-lockup">
                    <span class="brand-dot"></span>
                    <strong>Live Arena</strong>
                </div>
                <div class="command-pills">
                    <span>{mode}</span>
                    <span>{status}</span>
                    <span>Turn {turn}</span>
                </div>
            </div>
            <div class="match-board">
                <article class="player-panel blue{player_active}">
                    <div class="avatar">{_player_avatar(player_label, "P1")}</div>
                    <div>
                        <span>Blue Side</span>
                        <strong>{player_name}</strong>
                    </div>
                    <div class="score-burst">{agent_score}</div>
                </article>
                <article class="match-center">
                    <div class="game-tag">{game_name}</div>
                    <h1>{player_name} <span>vs</span> {opponent_name}</h1>
                    <p>{description}</p>
                    <div class="turn-meter">
                        <div style="width:{turn_pct}%"></div>
                    </div>
                    <div class="active-turn">Now playing: <strong>{html.escape(str(active_label))}</strong></div>
                </article>
                <article class="player-panel red{opponent_active}">
                    <div class="avatar">{_player_avatar(opponent_label, "P2")}</div>
                    <div>
                        <span>Red Side</span>
                        <strong>{opponent_name}</strong>
                    </div>
                    <div class="score-burst">{opponent_score}</div>
                </article>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def _game_skin_css(game_type: str) -> str:
    skins = {
        "RPS+": ("#06b6d4", "#f43f5e", "#f59e0b", "#101827"),
        "Tic-Tac-Toe": ("#22d3ee", "#a78bfa", "#f472b6", "#141124"),
        "Connect Four": ("#2563eb", "#facc15", "#ef4444", "#071b3a"),
        "Chess": ("#f59e0b", "#0f766e", "#fef3c7", "#1c1207"),
        "Othello": ("#10b981", "#14b8a6", "#e5e7eb", "#042f2e"),
        "Checkers": ("#ef4444", "#a855f7", "#f97316", "#1f0a1d"),
        "Gomoku": ("#38bdf8", "#facc15", "#f472b6", "#111827"),
        "Nim": ("#ec4899", "#f97316", "#facc15", "#2a0818"),
        "War": ("#ef4444", "#f8fafc", "#64748b", "#111827"),
    }
    primary, secondary, accent, deep = skins.get(game_type, ("#22d3ee", "#f472b6", "#facc15", "#111827"))
    return f"""
    <style>
        :root {{
            --skin-primary: {primary};
            --skin-secondary: {secondary};
            --skin-accent: {accent};
            --skin-deep: {deep};
            --game-surface: #f8fafc;
            --game-surface-muted: #cbd5e1;
            --game-surface-dim: #94a3b8;
            --game-panel: rgba(15, 23, 42, 0.74);
        }}
        .stApp {{
            background:
                radial-gradient(circle at 8% 12%, color-mix(in srgb, var(--skin-primary) 34%, transparent), transparent 26%),
                radial-gradient(circle at 88% 8%, color-mix(in srgb, var(--skin-secondary) 28%, transparent), transparent 24%),
                radial-gradient(circle at 48% 100%, color-mix(in srgb, var(--skin-accent) 18%, transparent), transparent 34%),
                linear-gradient(135deg, #030712 0%, var(--skin-deep) 46%, #020617 100%) !important;
        }}
        .block-container {{
            max-width: 1480px;
            padding-top: 1.15rem !important;
            padding-bottom: 2rem !important;
        }}
        [data-testid="stSidebar"] {{
            border-right: 1px solid rgba(255,255,255,0.10) !important;
            box-shadow: 16px 0 42px rgba(0,0,0,0.20);
        }}
        .match-dashboard {{
            position: relative;
            margin: 0 0 18px;
            border-radius: 30px;
            padding: 14px;
            background:
                linear-gradient(145deg, rgba(255,255,255,0.115), rgba(255,255,255,0.035)),
                radial-gradient(circle at 12% 0%, color-mix(in srgb, var(--skin-primary) 24%, transparent), transparent 28%),
                radial-gradient(circle at 88% 0%, color-mix(in srgb, var(--skin-secondary) 22%, transparent), transparent 28%),
                rgba(2,6,23,0.58);
            border: 1px solid rgba(255,255,255,0.16);
            box-shadow: 0 30px 90px rgba(0,0,0,0.30), inset 0 1px 0 rgba(255,255,255,0.13);
            backdrop-filter: blur(22px);
        }}
        .match-dashboard::after {{
            content: "YOU vs AGENT";
            position: absolute;
            right: 22px;
            top: 16px;
            color: rgba(0,255,157,0.40);
            font-family: var(--dg-mono);
            font-size: 11px;
            letter-spacing: 1.8px;
        }}
        .arena-command {{
            display: flex;
            justify-content: space-between;
            gap: 14px;
            align-items: center;
            padding: 4px 4px 12px;
        }}
        .brand-lockup {{
            display: inline-flex;
            align-items: center;
            gap: 10px;
            color: rgba(248,250,252,0.92);
            letter-spacing: 0.2px;
        }}
        .brand-dot {{
            width: 12px;
            height: 12px;
            border-radius: 999px;
            background: var(--skin-accent);
            box-shadow: 0 0 0 6px color-mix(in srgb, var(--skin-accent) 16%, transparent);
        }}
        .command-pills {{
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
            justify-content: flex-end;
        }}
        .command-pills span {{
            padding: 7px 10px;
            border-radius: 999px;
            font-size: 12px;
            font-weight: 800;
            color: rgba(248,250,252,0.78);
            background: rgba(255,255,255,0.075);
            border: 1px solid rgba(255,255,255,0.10);
        }}
        .match-board {{
            display: grid;
            grid-template-columns: minmax(180px, 0.8fr) minmax(320px, 1.6fr) minmax(180px, 0.8fr);
            gap: 12px;
            align-items: stretch;
        }}
        .player-panel,
        .match-center {{
            min-height: 150px;
            border-radius: 24px;
            border: 1px solid rgba(255,255,255,0.13);
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.11);
        }}
        .player-panel {{
            display: grid;
            grid-template-columns: auto 1fr;
            grid-template-rows: 1fr auto;
            gap: 12px;
            align-items: center;
            padding: 18px;
            position: relative;
            overflow: hidden;
            background:
                radial-gradient(circle at 20% 8%, rgba(255,255,255,0.12), transparent 22%),
                linear-gradient(145deg, rgba(255,255,255,0.10), rgba(255,255,255,0.035));
        }}
        .player-panel::after {{
            content: "";
            position: absolute;
            inset: auto 14px 12px 14px;
            height: 4px;
            border-radius: 999px;
            opacity: 0.38;
            background: currentColor;
        }}
        .player-panel.blue {{
            color: var(--skin-primary);
        }}
        .player-panel.red {{
            color: var(--skin-secondary);
        }}
        .player-panel.is-active {{
            border-color: color-mix(in srgb, currentColor 54%, rgba(255,255,255,0.18));
            box-shadow:
                inset 0 1px 0 rgba(255,255,255,0.15),
                0 0 0 1px color-mix(in srgb, currentColor 34%, transparent),
                0 20px 48px color-mix(in srgb, currentColor 18%, transparent);
        }}
        .avatar {{
            width: 54px;
            height: 54px;
            border-radius: 18px;
            display: grid;
            place-items: center;
            color: #020617;
            font-weight: 1000;
            letter-spacing: 0;
            background:
                radial-gradient(circle at 28% 20%, rgba(255,255,255,0.86), transparent 30%),
                linear-gradient(145deg, currentColor, color-mix(in srgb, currentColor 64%, #ffffff));
            box-shadow: 0 16px 30px color-mix(in srgb, currentColor 18%, transparent);
        }}
        .player-panel span {{
            display: block;
            font-size: 11px;
            letter-spacing: 1.4px;
            text-transform: uppercase;
            color: rgba(248,250,252,0.58);
            margin-bottom: 4px;
        }}
        .player-panel strong {{
            display: block;
            color: #ffffff;
            font-size: clamp(17px, 1.6vw, 24px);
            line-height: 1.05;
        }}
        .score-burst {{
            grid-column: 1 / -1;
            align-self: end;
            justify-self: end;
            min-width: 74px;
            padding: 8px 14px;
            border-radius: 18px;
            text-align: center;
            color: #ffffff;
            font-size: 34px;
            line-height: 1;
            font-weight: 1000;
            background: rgba(255,255,255,0.09);
            border: 1px solid rgba(255,255,255,0.13);
        }}
        .match-center {{
            display: grid;
            align-content: center;
            text-align: center;
            padding: 22px clamp(18px, 2.4vw, 34px);
            background:
                radial-gradient(circle at 50% 0%, color-mix(in srgb, var(--skin-accent) 18%, transparent), transparent 32%),
                linear-gradient(145deg, rgba(255,255,255,0.105), rgba(255,255,255,0.035));
        }}
        .game-tag {{
            width: fit-content;
            margin: 0 auto 10px;
            padding: 7px 12px;
            border-radius: 999px;
            color: #020617;
            background: var(--skin-accent);
            font-size: 12px;
            font-weight: 1000;
            letter-spacing: 0.8px;
            text-transform: uppercase;
        }}
        .match-center h1 {{
            margin: 0;
            color: #ffffff;
            font-size: clamp(28px, 3.8vw, 54px);
            line-height: 0.96;
        }}
        .match-center h1 span {{
            display: inline-block;
            margin: 0 8px;
            color: color-mix(in srgb, var(--skin-accent) 78%, #ffffff);
            font-size: 0.56em;
            text-transform: uppercase;
            vertical-align: middle;
        }}
        .match-center p {{
            max-width: 640px;
            margin: 12px auto 0;
            color: rgba(248,250,252,0.72);
            font-size: 14px;
            line-height: 1.45;
        }}
        .turn-meter {{
            height: 8px;
            margin: 18px auto 10px;
            width: min(100%, 520px);
            border-radius: 999px;
            overflow: hidden;
            background: rgba(255,255,255,0.10);
            border: 1px solid rgba(255,255,255,0.08);
        }}
        .turn-meter div {{
            height: 100%;
            border-radius: inherit;
            background: linear-gradient(90deg, var(--skin-primary), var(--skin-accent), var(--skin-secondary));
        }}
        .active-turn {{
            color: rgba(248,250,252,0.68);
            font-size: 13px;
            font-weight: 700;
        }}
        .active-turn strong {{
            color: #ffffff;
        }}
        .arena-hero {{
            display: none !important;
            min-height: 132px !important;
            padding: 20px 24px !important;
            margin-bottom: 12px !important;
            border-radius: 24px !important;
            background:
                linear-gradient(135deg, color-mix(in srgb, var(--skin-primary) 18%, transparent), transparent 38%),
                radial-gradient(circle at 86% 18%, color-mix(in srgb, var(--skin-secondary) 32%, transparent), transparent 24%),
                linear-gradient(135deg, color-mix(in srgb, var(--skin-deep) 86%, #020617), color-mix(in srgb, var(--skin-primary) 28%, #111827), color-mix(in srgb, var(--skin-secondary) 30%, #111827)) !important;
            box-shadow: 0 18px 48px rgba(0,0,0,0.22), inset 0 1px 0 rgba(255,255,255,0.16) !important;
        }}
        .arena-title {{
            font-size: clamp(24px, 3.2vw, 38px) !important;
        }}
        .arena-subtitle {{
            max-width: 920px !important;
            font-size: 14px !important;
            color: rgba(248,250,252,0.72) !important;
        }}
        .arena-shell {{
            background:
                radial-gradient(circle at 10% 6%, color-mix(in srgb, var(--skin-primary) 13%, transparent), transparent 28%),
                radial-gradient(circle at 92% 12%, color-mix(in srgb, var(--skin-secondary) 12%, transparent), transparent 26%),
                linear-gradient(145deg, rgba(15,23,42,0.88), rgba(17,24,39,0.76)) !important;
            border-color: color-mix(in srgb, var(--skin-primary) 24%, rgba(255,255,255,0.14)) !important;
            border-radius: 26px !important;
            padding: 18px !important;
            box-shadow: 0 24px 70px rgba(0,0,0,0.24), inset 0 1px 0 rgba(255,255,255,0.10) !important;
        }}
        .game-card {{
            padding: clamp(18px, 2.2vw, 28px) !important;
            border-radius: 22px !important;
            background:
                linear-gradient(145deg, rgba(255,255,255,0.07), rgba(255,255,255,0.025)),
                radial-gradient(circle at 50% 18%, color-mix(in srgb, var(--skin-primary) 8%, transparent), transparent 38%),
                rgba(15,23,42,0.66) !important;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.09) !important;
        }}
        .game-card,
        .game-card p,
        .game-card li,
        .game-card label,
        .game-card .stMarkdown,
        .game-card .stCaption,
        .game-card [data-testid="stMarkdownContainer"] {{
            color: var(--game-surface) !important;
        }}
        .game-card .stCaption,
        .game-card small,
        .game-card [data-testid="stCaptionContainer"],
        .game-card [data-testid="stMarkdownContainer"] p {{
            color: var(--game-surface-muted) !important;
        }}
        .board-area {{
            width: min(100%, 820px);
            margin: 0 auto 18px;
            padding: clamp(14px, 2.2vw, 26px);
            border-radius: 28px;
            background:
                radial-gradient(circle at 16% 12%, color-mix(in srgb, var(--skin-primary) 13%, transparent), transparent 26%),
                radial-gradient(circle at 86% 8%, color-mix(in srgb, var(--skin-secondary) 12%, transparent), transparent 28%),
                linear-gradient(145deg, rgba(255,255,255,0.12), rgba(255,255,255,0.045));
            border: 1px solid color-mix(in srgb, var(--skin-primary) 20%, rgba(255,255,255,0.16));
            box-shadow:
                inset 0 1px 0 rgba(255,255,255,0.14),
                inset 0 -22px 44px rgba(0,0,0,0.15),
                0 26px 58px rgba(0,0,0,0.22);
        }}
        .board-area [data-testid="stHorizontalBlock"] {{
            gap: clamp(4px, 0.85vw, 10px) !important;
            margin-bottom: clamp(4px, 0.85vw, 10px);
        }}
        .board-area [data-testid="column"] {{
            padding-left: 0 !important;
            padding-right: 0 !important;
        }}
        .board-area .stButton {{
            margin: 0 !important;
        }}
        .board-area .stButton > button {{
            margin: 0 !important;
        }}
        div[class*="st-key-ttt_"],
        div[class*="st-key-gomoku_"],
        div[class*="st-key-oth_"],
        div[class*="st-key-chk_"],
        div[class*="st-key-chk_to_"],
        div[class*="st-key-chess_sq_"] {{
            margin: 0 !important;
        }}
        div[class*="st-key-ttt_"] button,
        div[class*="st-key-gomoku_"] button,
        div[class*="st-key-oth_"] button,
        div[class*="st-key-chk_"] button,
        div[class*="st-key-chk_to_"] button,
        div[class*="st-key-chess_sq_"] button {{
            aspect-ratio: 1 / 1 !important;
            width: 100% !important;
            padding: 0 !important;
            overflow: hidden !important;
        }}
        div[class*="st-key-ttt_"] button p,
        div[class*="st-key-gomoku_"] button p,
        div[class*="st-key-oth_"] button p,
        div[class*="st-key-chk_"] button p,
        div[class*="st-key-chk_to_"] button p,
        div[class*="st-key-chess_sq_"] button p {{
            margin: 0 !important;
            line-height: 1 !important;
            white-space: nowrap !important;
        }}
        div[class*="st-key-cf_btn_"],
        div[class*="st-key-nim_"],
        div[class*="st-key-move_"] {{
            margin-bottom: 8px !important;
        }}
        .board-area-ttt {{
            width: min(100%, 560px);
            padding: clamp(14px, 2.6vw, 28px);
        }}
        .board-area-connect {{
            width: min(100%, 820px);
        }}
        .board-area-checkers,
        .board-area-chess,
        .board-area-othello {{
            width: min(100%, 720px);
        }}
        .board-area-gomoku {{
            width: min(100%, 760px);
        }}
        .board-area-checkers .turn-banner {{
            margin-bottom: 12px;
            padding: 10px 12px;
            border-radius: 16px;
        }}
        .game-stage-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 20px;
            margin: 0 0 18px;
        }}
        .game-stage-title h2 {{
            margin: 0 0 6px 0;
            font-size: clamp(28px, 4.2vw, 48px);
            line-height: 1;
            color: #ffffff;
        }}
        .game-stage-title p {{
            max-width: 720px;
            margin: 0;
            color: rgba(248,250,252,0.76);
            font-size: 15px;
        }}
        .game-stage-stats {{
            display: grid;
            grid-template-columns: repeat(2, minmax(96px, 1fr));
            gap: 10px;
            min-width: 250px;
        }}
        .game-stage-stats div {{
            padding: 11px 13px;
            border-radius: 16px;
            background:
                radial-gradient(circle at 18% 10%, color-mix(in srgb, var(--skin-accent) 10%, transparent), transparent 32%),
                rgba(255,255,255,0.075);
            border: 1px solid rgba(255,255,255,0.14);
        }}
        .game-stage-stats strong {{
            display: block;
            font-size: 10px;
            letter-spacing: 1.4px;
            text-transform: uppercase;
            color: rgba(248,250,252,0.58);
            margin-bottom: 4px;
        }}
        .game-stage-stats span {{
            display: block;
            font-weight: 900;
            color: #ffffff;
            line-height: 1.1;
        }}
        .turn-banner {{
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 12px 14px;
            margin: 0 auto 18px;
            max-width: 820px;
            border-radius: 16px;
            color: #ffffff;
            background:
                linear-gradient(135deg, color-mix(in srgb, var(--skin-primary) 22%, rgba(15,23,42,0.88)), color-mix(in srgb, var(--skin-secondary) 16%, rgba(15,23,42,0.88))) !important;
            border: 1px solid color-mix(in srgb, var(--skin-primary) 42%, rgba(255,255,255,0.12)) !important;
            box-shadow: 0 12px 28px rgba(0,0,0,0.18);
        }}
        .turn-banner span {{
            width: 14px;
            height: 14px;
            border-radius: 999px;
            flex: 0 0 auto;
            background: var(--skin-accent);
            box-shadow: 0 0 0 6px color-mix(in srgb, var(--skin-accent) 14%, transparent);
        }}
        .turn-banner strong {{
            font-size: 15px;
            line-height: 1.35;
        }}
        [data-baseweb="tab-list"] {{
            width: fit-content !important;
            margin: 0 0 18px auto !important;
            background: rgba(255,255,255,0.07) !important;
            border-color: rgba(255,255,255,0.13) !important;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.08) !important;
        }}
        [data-baseweb="tab"] {{
            height: 34px !important;
            padding: 0 14px !important;
            font-size: 13px !important;
        }}
        [data-baseweb="tab-panel"] {{
            padding-top: 0 !important;
        }}
        div[class*="st-key-move_"] button {{
            min-height: 96px !important;
            border-radius: 22px !important;
            font-size: 19px !important;
            color: #ffffff !important;
            background:
                radial-gradient(circle at 26% 18%, rgba(255,255,255,0.28), transparent 22%),
                linear-gradient(145deg, color-mix(in srgb, var(--skin-primary) 68%, #ffffff), color-mix(in srgb, var(--skin-secondary) 62%, #111827)) !important;
            border-color: rgba(255,255,255,0.22) !important;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.22), 0 14px 30px rgba(0,0,0,0.20) !important;
        }}
        div[class*="st-key-move_"] button:hover {{
            transform: translateY(-3px) scale(1.015) !important;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.28), 0 20px 42px color-mix(in srgb, var(--skin-primary) 18%, rgba(0,0,0,0.24)) !important;
        }}
        div[class*="st-key-ttt_"] button,
        div[class*="st-key-gomoku_"] button,
        div[class*="st-key-oth_"] button,
        div[class*="st-key-chk_"] button,
        div[class*="st-key-chk_to_"] button,
        div[class*="st-key-chess_sq_"] button {{
            min-height: clamp(44px, 6.7vw, 82px) !important;
            border-width: 2px !important;
            line-height: 1 !important;
        }}
        div[class*="st-key-ttt_"] button {{
            font-size: clamp(36px, 7vw, 64px) !important;
            min-height: clamp(92px, 14vw, 142px) !important;
            border-radius: 24px !important;
        }}
        div[class*="st-key-ttt_"] button:not(:disabled):empty,
        div[class*="st-key-gomoku_"] button:not(:disabled):empty,
        div[class*="st-key-oth_"] button:not(:disabled):empty {{
            outline: 2px solid color-mix(in srgb, var(--skin-accent) 56%, transparent) !important;
            outline-offset: -8px !important;
        }}
        div[class*="st-key-gomoku_"] button {{
            min-height: clamp(34px, 5.5vw, 62px) !important;
        }}
        div[class*="st-key-oth_"] button,
        div[class*="st-key-chk_"] button,
        div[class*="st-key-chk_to_"] button,
        div[class*="st-key-chess_sq_"] button {{
            min-height: clamp(40px, 6.2vw, 70px) !important;
        }}
        div[class*="st-key-chk_to_"] button {{
            color: #111827 !important;
            background:
                radial-gradient(circle at 50% 50%, #ffffff 0 13%, transparent 14%),
                linear-gradient(145deg, #facc15, #f97316) !important;
            box-shadow: 0 0 0 3px rgba(250,204,21,0.26), 0 18px 32px rgba(249,115,22,0.22) !important;
        }}
        .board-area .stButton > button:hover {{
            filter: saturate(1.08) brightness(1.06);
        }}
        .board-area .stButton > button:focus {{
            outline: 3px solid color-mix(in srgb, var(--skin-accent) 46%, transparent) !important;
            outline-offset: 2px !important;
        }}
        .board-area .stButton > button:disabled {{
            filter: saturate(0.94);
        }}
        div[class*="st-key-cf_btn_"] button {{
            min-height: 52px !important;
            border-radius: 999px !important;
            font-size: 15px !important;
        }}
        .connect-board {{
            max-width: 760px;
            margin: 0 auto;
            padding: clamp(12px, 1.8vw, 22px) !important;
            border: 1px solid color-mix(in srgb, var(--skin-primary) 44%, rgba(255,255,255,0.16));
            border-radius: 26px !important;
        }}
        .connect-slot {{
            font-size: clamp(24px, 5vw, 46px) !important;
        }}
        .legend-row {{
            justify-content: center;
            margin-bottom: 16px !important;
        }}
        .legend-pill {{
            color: #ffffff !important;
            background: rgba(255,255,255,0.075) !important;
            border-color: rgba(255,255,255,0.13) !important;
            backdrop-filter: blur(14px);
        }}
        .log-entry {{
            background: rgba(15,23,42,0.54) !important;
            border-color: rgba(255,255,255,0.12) !important;
            color: rgba(248,250,252,0.86) !important;
            padding: 12px !important;
            border-radius: 16px !important;
            box-shadow: none !important;
        }}
        .side-card {{
            border-radius: 20px !important;
            background: var(--game-panel) !important;
            border-color: rgba(255,255,255,0.12) !important;
            box-shadow: 0 14px 34px rgba(0,0,0,0.16) !important;
            color: var(--game-surface) !important;
        }}
        .side-card,
        .side-card div,
        .side-card p,
        .side-card span,
        .side-card strong,
        .log-entry,
        .log-entry div,
        .log-entry span,
        .log-entry strong {{
            color: var(--game-surface) !important;
        }}
        .side-card .section-label,
        .log-entry [style*="var(--arena-muted)"],
        .side-card [style*="var(--arena-muted)"] {{
            color: var(--game-surface-muted) !important;
        }}
        .hud-container {{
            max-width: 860px;
            margin: 0 auto 16px !important;
            border-radius: 22px !important;
            box-shadow: 0 16px 38px rgba(0,0,0,0.18) !important;
            display: none !important;
        }}
        .hud-score {{
            border-radius: 16px !important;
            padding: 12px 10px !important;
        }}
        .hud-val {{
            color: #ffffff !important;
            font-size: 30px !important;
        }}
        .hud-label {{
            color: rgba(248,250,252,0.62) !important;
        }}
        .match-strip {{
            gap: 10px !important;
            margin-top: 12px !important;
        }}
        .player-badge {{
            border-radius: 16px !important;
            padding: 11px 13px !important;
            min-width: 146px !important;
        }}
        .versus-mark {{
            width: 40px !important;
            height: 40px !important;
            box-shadow: 0 12px 26px rgba(0,0,0,0.20) !important;
        }}
        .arena-chip-row {{
            margin-top: 12px !important;
        }}
        .arena-chip {{
            padding: 7px 11px !important;
            font-size: 12px !important;
        }}
        .stButton > button p {{
            color: inherit !important;
        }}
        .stButton > button:disabled p {{
            color: inherit !important;
        }}
        .stButton > button {{
            background:
                radial-gradient(circle at 22% 14%, rgba(255,255,255,0.16), transparent 24%),
                linear-gradient(145deg, rgba(30,41,59,0.98), rgba(15,23,42,0.98)) !important;
            color: #f8fafc !important;
            border-color: rgba(255,255,255,0.18) !important;
        }}
        .stButton > button p,
        .stButton > button span,
        .stButton > button div {{
            color: #f8fafc !important;
        }}
        .stButton > button:hover {{
            background:
                radial-gradient(circle at 22% 14%, rgba(255,255,255,0.20), transparent 24%),
                linear-gradient(145deg, rgba(51,65,85,1), rgba(30,41,59,1)) !important;
            color: #ffffff !important;
        }}
        .stButton > button:disabled {{
            background: rgba(15,23,42,0.54) !important;
            color: #94a3b8 !important;
            opacity: 0.72 !important;
        }}
        .stButton > button:disabled p,
        .stButton > button:disabled span,
        .stButton > button:disabled div {{
            color: #94a3b8 !important;
        }}
        [data-baseweb="tab"] p {{
            color: var(--game-surface-muted) !important;
        }}
        button[role="tab"][aria-selected="true"] p {{
            color: var(--game-surface) !important;
        }}
        [data-baseweb="select"] > div,
        [data-baseweb="input"] > div,
        [data-baseweb="base-input"],
        [data-baseweb="textarea"] {{
            background: #0f172a !important;
            color: #f8fafc !important;
            border-color: rgba(255,255,255,0.18) !important;
        }}
        [data-baseweb="select"] *,
        [data-baseweb="input"] *,
        [data-baseweb="base-input"] *,
        [data-baseweb="textarea"] *,
        [data-baseweb="radio"] *,
        [data-baseweb="slider"] *,
        input,
        textarea {{
            color: #f8fafc !important;
            -webkit-text-fill-color: #f8fafc !important;
        }}
        [data-baseweb="popover"] *,
        [role="listbox"] *,
        [role="option"] * {{
            color: #0f172a !important;
            -webkit-text-fill-color: #0f172a !important;
        }}
        [data-baseweb="popover"] [role="option"] {{
            background: #ffffff !important;
        }}
        }}
        @media (max-width: 900px) {{
            .arena-command {{
                display: block;
            }}
            .command-pills {{
                justify-content: flex-start;
                margin-top: 12px;
            }}
            .match-board {{
                grid-template-columns: 1fr;
            }}
            .player-panel,
            .match-center {{
                min-height: 122px;
            }}
            .score-burst {{
                position: absolute;
                right: 14px;
                bottom: 14px;
            }}
            .game-stage-header {{
                display: block;
            }}
            .game-stage-stats {{
                margin-top: 14px;
                min-width: 0;
            }}
            .arena-chip-row {{
                display: none;
            }}
        }}
    </style>
    """


def _clone_confidence(env, history_len: int, agent_name: str | None = None) -> int:
    if agent_name in {"ngram", "lstm", "mixture"}:
        try:
            agent = _new_agent(agent_name)
            history = list(st.session_state.game_state.get("move_history", []))
            legal = env.legal_actions() if hasattr(env, "legal_actions") else None
            uncertainty = getattr(agent, "uncertainty", None)
            if callable(uncertainty):
                stats = uncertainty(history=history, legal=legal)
                model_confidence = float(stats.get("confidence", 0.0))
                if model_confidence > 0:
                    return int(min(99, max(5, round(model_confidence * 100))))
        except Exception:
            pass
    total = max(1, env.state.agent_score + env.state.opponent_score + 1)
    score_pressure = env.state.opponent_score / total
    turn_pressure = min(1.0, history_len / 12)
    return int(min(98, max(12, 22 + score_pressure * 46 + turn_pressure * 30)))


def _signature_vector(env, history_len: int) -> list[float]:
    total = max(1, env.state.agent_score + env.state.opponent_score + env.state.turn + 1)
    return [
        min(1, (env.state.turn + 1) / max(1, getattr(env, "max_turns", 30))),
        min(1, (env.state.agent_score + 1) / total),
        min(1, (env.state.opponent_score + 1) / total),
        min(1, history_len / 8),
        min(1, abs(env.state.agent_score - env.state.opponent_score + 1) / total),
    ]


def _render_behavioral_radar(env, history_len: int) -> None:
    labels = ["Tempo", "Human Pressure", "Clone Pressure", "Pattern Depth", "Volatility"]
    values = _signature_vector(env, history_len)
    if go is None:
        st.caption("Radar chart unavailable because Plotly is not installed.")
        return
    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=values + values[:1],
            theta=labels + labels[:1],
            fill="toself",
            name="Signature",
            line=dict(color="#00ff9d", width=2),
            fillcolor="rgba(0,255,157,0.20)",
        )
    )
    style_plotly_figure(fig, height=300)
    fig.update_layout(
        font=dict(color="#f4fff9", family="SFMono-Regular, monospace"),
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="rgba(0,255,157,0.18)"),
            angularaxis=dict(gridcolor="rgba(127,119,221,0.18)"),
        ),
        showlegend=False,
    )
    st.plotly_chart(fig, width='stretch')


def _agent_threat_level(agent_name: str) -> tuple[str, str]:
    name = (agent_name or "").lower()
    if "lstm" in name or "adaptive" in name or "optimal" in name or "mixture" in name:
        return "LSTM / Adaptive", "red"
    if "profile" in name or "heuristic" in name:
        return "Heuristic", "yellow"
    return "Random", "green"


def _render_agent_threat_cards(agent_names: list[str]) -> None:
    cols = st.columns(max(1, len(agent_names)), gap="small")
    for col, agent in zip(cols, agent_names):
        label, level = _agent_threat_level(agent)
        color = {"green": "#00ff9d", "yellow": "#ffd166", "red": "#ff6b35"}[level]
        with col:
            st.markdown(
                f"""
                <div class="dg-card" style="padding:16px; border-color:{color} !important;">
                    <div class="dg-muted" style="font-family:var(--dg-mono);">{label}</div>
                    <div style="margin-top:8px; color:{color}; font-family:var(--dg-mono); font-weight:700;">
                        {html.escape(agent or "none")}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _render_clone_stats_panel(env, agent_names: list[str], history_len: int) -> None:
    primary_agent = agent_names[0] if agent_names else None
    confidence = _clone_confidence(env, history_len, primary_agent)
    latest_surprisal = st.session_state.game_state.get("last_surprisal") or {}
    surprisal_summary = summarize_surprisal_history(st.session_state.game_state.get("surprisal_history", []))
    st.markdown(
        f"""
        <div class="dg-card dg-clone-panel">
            <div class="dg-panel-kicker"><span class="dg-live">LIVE</span> OPPONENT READ</div>
            <div class="dg-progress-label">HOW SURE THE OPPONENT IS <strong>{confidence}%</strong></div>
            <div class="dg-progress"><div style="width:{confidence}%"></div></div>
            <div class="dg-counter-grid">
                <div><span>MOVES STUDIED</span><strong>{history_len}</strong></div>
                <div><span>CURRENT TURN</span><strong>{env.state.turn}</strong></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if latest_surprisal:
        st.markdown('<div class="section-label">Surprise Level</div>', unsafe_allow_html=True)
        sup_cols = st.columns(2, gap="small")
        sup_cols[0].metric("Latest", f"{float(latest_surprisal.get('surprisal', 0.0)):.2f}")
        sup_cols[1].metric("Match Mean", f"{float(surprisal_summary.get('mean_surprisal', 0.0)):.2f}")
        st.caption(
            f"The opponent gave your last move a {float(latest_surprisal.get('predicted_prob', 0.0)):.2f} probability. "
            f"Unexpected turns so far: {int(surprisal_summary.get('high_surprisal_turns', 0))}."
        )
    st.markdown('<div class="section-label">Play Pattern Snapshot</div>', unsafe_allow_html=True)
    _render_behavioral_radar(env, history_len)
    st.markdown('<div class="section-label">Opponent Style</div>', unsafe_allow_html=True)
    _render_agent_threat_cards(agent_names)


def _render_side_card(title, body):
    st.markdown(
        f"""
        <div class="side-card">
            <div class="section-label">{title}</div>
            <div style="color: var(--game-surface); line-height: 1.55;">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _game_status(game_type):
    return "Playable Now" if game_type in STANDARD_GAMES else "Research Sandbox"


def _game_label(game_type):
    icon = GAME_ICONS.get(game_type, "🎮")
    return f"{icon} {game_type} · {_game_status(game_type)}"


def _game_library_options(view_name):
    if view_name == "Playable Now":
        options = STANDARD_GAMES
    elif view_name == "Board Games":
        options = STANDARD_GAMES
    elif view_name == "Research Sandbox":
        options = PROTOTYPE_GAMES
    else:
        options = ALL_GAMES
    return options


def _game_spotlight(game_type):
    highlights = {
        "RPS+": "Fastest feedback loop. Best for adaptive bot behavior and repeated mind-games.",
        "Tic-Tac-Toe": "Clean and readable. Great when you want a quick strategic duel.",
        "Connect Four": "Best visual board pressure. Easy to read, surprisingly tactical.",
        "Chess": "Deepest strategy surface in the arena. Best for slower, richer matches.",
        "Othello": "Territory swings make every corner feel dramatic.",
        "Checkers": "Simple rules, sharp traps, satisfying turn-by-turn tension.",
        "Gomoku": "Easy to learn and tense fast: build five in a row while blocking forks.",
        "Nim": "A compact math duel where every pile choice changes the endgame.",
        "Pac-Man": "Compact maze prototype with readable chase pressure, but still much lighter than a full arcade implementation.",
        "2048": "A tidy puzzle surface that works well as a quick strategy sim.",
        "Wordle": "Readable word-guessing loop, closer to a puzzle mock than a full game client.",
        "Sudoku": "Classic number-grid logic with the real 9x9 constraint structure players expect.",
    }
    return highlights.get(game_type, "Exploratory sim for testing decision patterns and player behavior, not a full consumer-grade game build.")


def _game_description(game_type):
    descriptions = {
        "RPS+": "Energy management matters here. Power spikes can steal rounds, but only if you keep enough charge in reserve.",
        "Tic-Tac-Toe": "A clean warm-up duel. Take the center early or build a fork before the bot does.",
        "Connect Four": "Column pressure matters more than it first appears. Build threats on two levels when you can.",
        "Chess": "Select one of your pieces, then tap a highlighted destination square. The board keeps legal choices visible.",
        "Othello": "Corners are priceless. The best move is often the one that gives away the fewest easy replies.",
        "Checkers": "Play patiently until a forcing capture opens. Multi-jumps can swing the whole board.",
        "Gomoku": "Place stones on a 9x9 board and race to five in a row. Blocking is just as important as attacking.",
        "Nim": "Remove one to three stones from a pile. Whoever takes the last stone wins.",
        "War": "Pure luck, but still satisfying. Draw and watch the stacks swing.",
    }
    if game_type in FUTURE_GAME_LABELS:
        future = {
            "2048": "Merge upward, preserve space, and avoid cluttering the corners too early. This one plays well despite being a compact sim.",
            "Candy Crush": "Pick swaps that feel obvious to the eye. Treat it as a rules demo, not a polished match-three client.",
            "Wordle": "Choose a word and read the feedback. This is a clean puzzle loop, not a full daily-style product.",
            "Sudoku": "A standard 9x9 Sudoku puzzle with authentic row, column, and box constraints.",
            "Pac-Man": "Clear pellets, stay ahead of the ghost, and use corners without trapping yourself. Lightweight maze logic, not full Pac-Man depth.",
            "Minecraft": "Build toward shelter before your resources or health run dry. More strategy sketch than world simulation.",
            "Among Us": "Push tasks when you are unsure. Call the shot only when the suspicion pattern is strong. Social deduction reduced to a compact model.",
            "Clash Royale": "Spend elixir with intent. Small advantages snowball when you stop panic-spending. This is a pressure simulator, not a full lane battler.",
            "Flappy Bird": "Two buttons, one rhythm. Stay calm around the pipe mouth.",
            "Ludo": "Your dice roll is shown up front, so the choice is about which token deserves progress.",
            "UNO": "The current color and your hand are surfaced plainly. Play tempo first, then count compression.",
            "Scrabble": "Choose from a compact word rack and keep the score race visible.",
            "Monopoly": "Cashflow first, greed second. Short matches reward fast compounding.",
            "Penalty Shootout": "Think like a keeper for one beat, then go elsewhere.",
            "Cricket Strategy": "Pick your risk profile ball by ball. The chase pressure is always visible.",
        }
        return future.get(game_type, "Pick a move and let the arena handle the bookkeeping.")
    return descriptions.get(game_type, "Pick your move and pressure the bot into bad replies.")


def _game_rules(game_type):
    rules = {
        "RPS+": [
            "Each round you choose Rock, Paper, Scissors, Lizard, Power, or Recharge.",
            "Power beats the base moves but costs energy.",
            "Recharge restores energy but loses to attacks except another Recharge.",
            "The match score updates round by round until the turn limit is reached.",
        ],
        "Tic-Tac-Toe": [
            "Place marks on a 3x3 board.",
            "Get three in a row horizontally, vertically, or diagonally.",
            "If the board fills with no line, the game is a draw.",
        ],
        "Connect Four": [
            "Drop a piece into a column and it falls to the lowest open spot.",
            "Connect four of your pieces in a row, column, or diagonal to win.",
            "Center control usually gives you the best follow-up threats.",
        ],
        "Chess": [
            "You play White and move first.",
            "Tap one of your pieces, then tap a highlighted destination square.",
            "Legal moves are enforced by the board, and promotions auto-choose a queen when available.",
            "Checkmate wins, stalemate draws, and the bot replies after every legal move.",
        ],
        "Othello": [
            "Place a disc so it brackets one or more opponent discs in a straight line.",
            "Bracketed discs flip to your color immediately.",
            "Corners are especially valuable because they cannot be flipped away.",
        ],
        "Checkers": [
            "Move diagonally on dark squares.",
            "Captures jump over an adjacent opponent piece into an open square beyond it.",
            "Reaching the back row promotes a piece to a king.",
        ],
        "Gomoku": [
            "Place one stone on an empty intersection each turn.",
            "Five stones in a row wins horizontally, vertically, or diagonally.",
            "Block open-ended lines before they become unstoppable.",
        ],
        "Nim": [
            "Each turn, choose one pile and remove one to three stones.",
            "You cannot remove more stones than the pile contains.",
            "The player who takes the final stone wins.",
        ],
        "War": [
            "Both sides reveal the next card from their deck.",
            "Higher card wins the battle and pushes the score.",
            "There is no strategy beyond drawing the next card.",
        ],
        "2048": [
            "Slide the whole board in one direction each turn.",
            "Matching tiles merge into a bigger value.",
            "Keep space open so the board does not lock up.",
        ],
        "Candy Crush": [
            "Choose a swap between neighboring candies.",
            "Three or more matching candies score and clear.",
            "Look for moves that create cascades, not just the first match.",
        ],
        "Wordle": [
            "Pick a five-letter guess each turn.",
            "Green means correct letter and position.",
            "Yellow means the letter exists but belongs elsewhere.",
            "Gray means the letter is not part of the answer.",
        ],
        "Sudoku": [
            "Fill the 9x9 board so each row, column, and 3x3 box uses 1 through 9 exactly once.",
            "Tap an open cell, then choose one of the legal numbers.",
            "The board only offers currently valid placements.",
        ],
        "Pac-Man": [
            "Move through the maze collecting pellets.",
            "The ghost moves after you each turn and usually tries to close distance.",
            "Clear every pellet before the ghost catches you to win.",
        ],
    }
    return rules.get(
        game_type,
        [
            "Pick one of the legal actions shown on screen.",
            "The environment updates after your move and then the bot responds.",
            "Use the score and battle log to track how the match is unfolding.",
        ],
    )


def _game_tips(game_type):
    tips = {
        "RPS+": [
            "Power beats the base moves, but only when you can afford the energy cost.",
            "Recharge is tempo-negative unless you use it to set up a future power spike.",
            "If the bot is low on energy, it loses access to its strongest move.",
        ],
        "Tic-Tac-Toe": [
            "The center is usually worth taking early.",
            "Look for forks instead of single threats.",
        ],
        "Connect Four": [
            "Middle columns create more diagonal and vertical threats.",
            "A move that sets up two winning lines is usually the real goal.",
        ],
        "Chess": [
            "Develop pieces before hunting for material.",
            "If a piece has no highlighted targets, switch to a different square quickly.",
        ],
        "Othello": [
            "Corners are worth more than raw disc count early.",
            "Avoid handing the bot easy edge access unless you gain a corner or tempo.",
        ],
        "Checkers": [
            "Forced captures can be a trap if they open a kinging lane.",
            "Try to keep your pieces connected so single jumps do not unravel the board.",
        ],
        "Pac-Man": [
            "Do not hug one corridor too long if the ghost is already closing.",
            "Sometimes the best move is the one that preserves two exits instead of one.",
        ],
    }
    return tips.get(game_type, ["Use the legal-move hints and play for the score swing that is visible right now."])


def _agent_description(agent_name):
    descriptions = {
        "random": "Chooses randomly from the currently legal moves. Good for a baseline and for checking whether your strategy beats pure chance.",
        "heuristic": "Uses simple hand-built rules. In RPS+, it watches your habits and tries to counter common patterns without any heavy training.",
        "optimal": "A stronger hand-crafted counter bot. In games where it understands the move structure, it tries to punish predictable play quickly.",
        "sft": "A supervised fine-tuned bot trained on gameplay examples. It aims to imitate learned behavior rather than just chase raw wins.",
        "rl": "A reinforcement-learning bot focused on maximizing win rate from experience. It tends to be more competitive than human-like.",
        "ppo": "An alias of the reinforcement-learning bot. Same behavior family as `rl`, exposed under the training method name.",
        "bc_rl": "A hybrid bot that mixes behavioral cloning with reinforcement learning, trying to stay believable while still competing hard.",
        "bcrl": "An alias of the BC+RL hybrid bot. Same model family as `bc_rl`.",
        "agentic": "A tool-using agentic bot. It is the most complex setup and is useful when you want to compare structured reasoning against simpler policies.",
        "profile_counter": "Reads saved history for the active player or guest ID and tries to predict and counter likely future moves. This is the most personalized competitive bot in the arena.",
        "adaptive_router": "A meta-bot that switches among multiple experts during play. It can lean toward imitation, competitiveness, or player-history countering depending on how the match is unfolding.",
        "ngram": "A lightweight behavioral clone that predicts your next move from short action contexts in saved history.",
        "lstm": "A sequence model clone that tries to imitate your move patterns over a longer history window.",
        "mixture": "A blended clone that mixes short-context habits, sequence priors, and a simple action prior to feel more like a cognitive model than a single policy.",
    }
    return descriptions.get(agent_name, "This bot uses its registered policy for the selected game.")


def _agent_support_note(game_type, agent_name):
    supported = _supported_agents_for_game(game_type)
    if agent_name in supported:
        if len(supported) <= 2 and game_type != "RPS+":
            return "This game is using the smaller stable bot set so matches stay playable."
        return "This bot is supported for the current game."
    return f"`{agent_name}` is not supported for {game_type}. The arena will switch to `{supported[0]}`."


def _agent_runtime_status(agent_name, agent):
    if agent_name == "random":
        return ("Rule-based baseline", "Random legal-move policy. No training checkpoint is involved.")
    if agent_name == "heuristic":
        return ("Rule-based baseline", "Hand-written policy. It reacts to simple patterns instead of loading a trained model.")
    if agent_name == "optimal":
        return ("Rule-based baseline", "Hand-crafted counter policy. This is logic-driven, not checkpoint-driven.")
    if agent_name == "profile_counter":
        counts = getattr(agent, "player_counts", {}) or {}
        total_examples = int(sum(counts.values())) if hasattr(counts, "values") else 0
        if total_examples > 0:
            return ("History-trained", f"Using saved gameplay history from the local database. Learned from {total_examples} logged player moves.")
        return ("History-trained", "This bot reads the local gameplay database, but there is no saved history for the active player yet.")
    if agent_name == "adaptive_router":
        chosen = getattr(agent, "last_selected_expert", None)
        reason = getattr(agent, "last_route_reason", "The router has not chosen an expert yet.")
        counts = getattr(agent, "selection_counts", {}) or {}
        if chosen is None:
            return ("Adaptive router", f"Ready to switch among experts dynamically. {reason}")
        mix = ", ".join(f"{name}:{count}" for name, count in counts.items()) or "no selections yet"
        return ("Adaptive router", f"Current expert: {chosen}. {reason} Selection mix so far: {mix}.")
    if agent_name in {"ngram", "lstm", "mixture"}:
        source = getattr(agent, "_clone_source_player", None)
        if source:
            return ("Behavioral clone", f"Loaded from saved gameplay for player `{source}`.")
        return ("Behavioral clone", "This clone is using saved player history from the local database.")

    model = getattr(agent, "model", None)
    if model is not None:
        checkpoint_path = getattr(agent, "checkpoint_path", None)
        detail = "A fine-tuned transformer checkpoint is loaded for this bot."
        if checkpoint_path is not None:
            detail += f" Path: {checkpoint_path}"
        return ("Checkpoint loaded", detail)

    sb3_model = getattr(agent, "_model", None)
    if sb3_model is not None:
        checkpoint_path = getattr(agent, "checkpoint_path", None)
        detail = "A saved reinforcement-learning checkpoint is loaded for this bot."
        if checkpoint_path is not None:
            detail += f" Path: {checkpoint_path}"
        return ("Checkpoint loaded", detail)

    if agent_name in {"sft"}:
        return ("Fallback mode", "No SFT checkpoint is loaded right now, so this bot is using its built-in heuristic fallback.")
    if agent_name in {"rl", "ppo"}:
        return ("Fallback mode", "No PPO checkpoint is loaded right now, so this bot is using its built-in heuristic fallback.")
    if agent_name in {"bc_rl", "bcrl"}:
        return ("Fallback mode", "No BC+RL checkpoint is loaded right now, so this bot is using its built-in heuristic fallback.")
    if agent_name == "agentic":
        impl = getattr(agent, "impl", None)
        impl_name = impl.__class__.__name__ if impl is not None else "agentic runtime"
        return ("Agentic runtime", f"Using the {impl_name} policy layer. If it fails, the arena falls back to a legal random move.")

    return ("Built-in policy", "This opponent is using its registered built-in policy for the current game.")


def _coerce_action_to_legal(action, legal_moves):
    legal = [int(move) for move in (legal_moves or [])]
    if not legal:
        return 0
    try:
        coerced = int(action)
    except Exception:
        coerced = legal[0]
    if coerced in legal:
        return coerced
    return legal[0]


def _agent_action(agent, env, game_type, obs, info, current_player_move=None, side=1):
    set_current = getattr(agent, "set_current_player_move", None)
    if callable(set_current) and current_player_move is not None:
        set_current(current_player_move)

    if side == 1:
        current_obs = obs
        current_info = info
    else:
        current_obs = env._obs() if hasattr(env, "_obs") else env._encode()
        if game_type in {"Othello", "Checkers"}:
            current_info = env._info(-1)
        else:
            current_info = env._info()

    action = agent.act(current_obs, current_info)
    return _coerce_action_to_legal(action, current_info.get("legal_moves"))


def _observe_agent(agent, agent_move, opponent_move, outcome):
    observe = getattr(agent, "observe", None)
    if callable(observe) and agent_move is not None and opponent_move is not None:
        try:
            observe(int(agent_move), int(opponent_move), int(outcome))
        except (ValueError, TypeError):
            return


def _opponent_legal_after_pending(env, game_type, pending_move):
    if pending_move is None:
        return []
    if game_type == "RPS+":
        return [int(m) for m in env.legal_moves(agent=False)]
    if game_type == "Tic-Tac-Toe":
        board = env.board.copy()
        if 0 <= pending_move < len(board) and board[pending_move] == 0:
            board[pending_move] = 1
        return [i for i, value in enumerate(board) if value == 0]
    if game_type == "Connect Four":
        board = env.board.copy()
        if pending_move in env.legal_moves():
            for row in range(board.shape[0]):
                if board[row, pending_move] == 0:
                    board[row, pending_move] = 1
                    break
        return [c for c in range(board.shape[1]) if board[-1, c] == 0]
    if game_type == "Othello":
        board = env.board.copy()
        if pending_move in env.legal_moves():
            env._apply_move(board, pending_move, 1)
        return env._legal_moves_for_player(board, -1)
    if game_type == "Checkers":
        board = env.board.copy()
        moves_dict = env._get_moves(board, 1, env.must_jump_piece)
        if pending_move not in moves_dict:
            return []
        continues = env._apply_move(board, pending_move, moves_dict, 1)
        if continues:
            return []
        return list(env._get_moves(board, -1, None).keys())
    if game_type == "Gomoku":
        board = env.board.copy()
        if pending_move in env.legal_moves():
            r, c = divmod(pending_move, board.shape[1])
            board[r, c] = 1
        return [idx for idx, value in enumerate(board.flatten()) if value == 0]
    if game_type == "Nim":
        piles = env.piles.copy()
        if pending_move in env.legal_moves():
            pile_idx, remove_idx = divmod(int(pending_move), 3)
            piles[pile_idx] -= remove_idx + 1
        legal = []
        for pile_idx, count in enumerate(piles):
            for remove in range(1, min(int(count), 3) + 1):
                legal.append(pile_idx * 3 + (remove - 1))
        return legal
    if game_type == "Chess" and getattr(env, "_board", None):
        board = env._board.copy()
        legal = list(board.legal_moves)
        if 0 <= pending_move < len(legal):
            board.push(legal[pending_move])
        return list(range(len(list(board.legal_moves))))
    if game_type == "War":
        return [0]
    return list(env._info().get("legal_moves", [])) if hasattr(env, "_info") else []


def _chess_piece_label(piece):
    if piece is None:
        return " "
    return piece.symbol().upper() if piece.color else piece.symbol().lower()


def _chess_action_for_target(board_obj, selected_square, target_square):
    fallback_idx = None
    for idx, move in enumerate(board_obj.legal_moves):
        if move.from_square != selected_square or move.to_square != target_square:
            continue
        promotion = getattr(move, "promotion", None)
        if promotion in {None, 5}:
            return idx
        if fallback_idx is None:
            fallback_idx = idx
    return fallback_idx


def _wordle_feedback_markup(env):
    tiles = []
    color_map = {
        0: ("#efe8dc", "#6b5d55"),
        1: ("#f3d79a", "#6d4a10"),
        2: ("#9fd3b3", "#184f31"),
    }
    history = getattr(env, "guess_history", [])
    for idx, (guess, feedback) in enumerate(history):
        row_tiles = []
        for ch, value in zip(guess.upper(), feedback):
            bg, fg = color_map[int(value)]
            flip_class = " wordle-flip" if idx == len(history) - 1 else ""
            row_tiles.append(
                f'<div class="value-tile{flip_class}" style="min-height:54px;background:{bg};color:{fg};font-size:18px;">{ch}</div>'
            )
        tiles.append('<div class="value-board" style="grid-template-columns: repeat(5, minmax(0, 1fr)); margin-bottom: 10px;">' + "".join(row_tiles) + "</div>")
    missing_rows = max(0, env.max_moves - len(getattr(env, "guess_history", [])))
    for _ in range(missing_rows):
        empty = "".join('<div class="value-tile empty" style="min-height:54px;">&nbsp;</div>' for _ in range(5))
        tiles.append('<div class="value-board" style="grid-template-columns: repeat(5, minmax(0, 1fr)); margin-bottom: 10px;">' + empty + "</div>")
    return "".join(tiles)


def _game_ui_traits(game_type):
    traits = {
        "RPS+": [
            ("Fast duel buttons", "Large icon-first move controls keep the read time near-instant."),
            ("Energy HUD", "A compact meter keeps the custom resource layer visible without clutter."),
            ("Round log feedback", "Every turn is summarized so pattern-reading still feels like a duel, not hidden math."),
        ],
        "Tic-Tac-Toe": [
            ("Centered 3x3 board", "The match lives on the board, not in side chrome."),
            ("Immediate placement", "Single-click square entry preserves the classic snap of play."),
            ("Minimal visual noise", "The surface stays sparse so forks and blocks are legible."),
        ],
        "Connect Four": [
            ("Gravity-first board", "Circular slots and top-down drop labels emphasize falling pieces."),
            ("Color-coded ownership", "Bright player tokens keep threat reading quick."),
            ("Column action row", "Move controls stay aligned with the board columns."),
        ],
        "Chess": [
            ("Checkered board hierarchy", "Board contrast stays primary while controls stay secondary."),
            ("Click-select move flow", "Modern tap-to-move replaces drag while preserving deliberate piece selection."),
            ("Legal target highlighting", "Available destinations do the work of teaching the move model."),
        ],
        "Othello": [
            ("Disc contrast", "Black-versus-white ownership stays unmistakable at a glance."),
            ("Corner-focused board", "Clean green field and legal markers keep territory swings readable."),
            ("Count awareness", "Piece counts sit near the board so momentum feels visible."),
        ],
        "Checkers": [
            ("Dark-square emphasis", "Playable squares carry the real visual weight."),
            ("Selection plus destination cues", "Pick-then-land interaction modernizes piece movement without inventing drag mechanics."),
            ("King differentiation", "Promoted pieces remain visually distinct without copying tournament sets."),
        ],
        "2048": [
            ("Escalating tile palette", "Value growth is communicated by warmth, contrast, and tile emphasis."),
            ("Merge pop feedback", "Tiles animate with a quick pulse to echo the original merge satisfaction."),
            ("Move-only controls", "Directional actions stay blunt and obvious like the source game."),
        ],
        "Wordle": [
            ("Five-tile rows", "The guess grid is the product, so each row stays evenly framed."),
            ("Flip-style reveal", "Latest result tiles animate through a light flip rather than appearing dead-static."),
            ("Feedback-first color coding", "Green, yellow, and gray states stay the main reading system."),
        ],
        "Sudoku": [
            ("9x9 structure", "The real grid shape returns, with 3x3 group boundaries made visible."),
            ("Cell-first input", "Select a square, then place a legal digit to keep the mental model clear."),
            ("Constraint-guided affordance", "Only currently valid cells and values are emphasized."),
        ],
        "Pac-Man": [
            ("Maze as the stage", "The board reads like a corridor maze first, not a generic grid."),
            ("HUD over debug text", "Score, pellets, and frightened state get compact arcade-style readouts."),
            ("Motion-state contrast", "Ghost and power-pellet states change the visual read without cloning cabinet art."),
        ],
    }
    return traits.get(game_type, [])


def render_future_game(game_type, env, legal_moves):
    if game_type == "2048":
        _render_move_ribbon(f"Score <strong>{env.score}</strong>. Keep the high-value tile anchored and give yourself space to breathe.")
        _metric_row([("Score", env.score), ("Top tile", int(env.board.max())), ("Moves", env.state.turn)])
        _render_2048_board(env)
        cols = st.columns(4)
        for col, action in zip(cols, [0, 1, 2, 3]):
            col.button(future_action_label(game_type, action, env), key=f"future_{game_type}_{action}", disabled=action not in legal_moves, width='stretch', on_click=handle_move, args=(action,))
        return

    if game_type == "Candy Crush":
        _render_move_ribbon(f"Score <strong>{env.score}</strong>. Pick a swap that creates a three-in-a-row or better.")
        candy = {1: "Red", 2: "Blue", 3: "Green", 4: "Gold", 5: "Plum"}
        candy_styles = {
            1: "background: #f9d3cc; color: #9d3020;",
            2: "background: #d8e7ff; color: #1c57a5;",
            3: "background: #d7f0dd; color: #24663b;",
            4: "background: #f7e9bf; color: #936515;",
            5: "background: #ebdbf8; color: #6b3799;",
        }
        labels = {(r, c): candy[int(env.board[r, c])] for r in range(6) for c in range(6)}
        colors = {(r, c): candy_styles[int(env.board[r, c])] for r in range(6) for c in range(6)}
        _render_value_grid(env.board, labels, "candy", colors)
        for action in legal_moves[:12]:
            st.button(future_action_label(game_type, action, env), key=f"future_{game_type}_{action}", width='stretch', on_click=handle_move, args=(action,))
        return

    if game_type == "Wordle":
        _render_move_ribbon(f"Guesses used <strong>{env.n_guesses}</strong> of <strong>{env.max_moves}</strong>. Pick the next word that sharpens the picture.")
        st.markdown(_wordle_feedback_markup(env), unsafe_allow_html=True)
        cols = st.columns(4)
        for i, action in enumerate(legal_moves):
            cols[i % 4].button(env.WORDS[action].upper(), key=f"future_{game_type}_{action}", width='stretch', on_click=handle_move, args=(action,))
        return

    if game_type == "Sudoku":
        _render_move_ribbon("Fill the blanks without breaking row, column, or box balance.")
        size = env.board.shape[0]
        selected_cell = st.session_state.game_state.get("sudoku_selected_cell")
        legal_cells = {action // size for action in legal_moves}
        _render_sudoku_board(env, legal_cells, selected_cell)

        st.caption("Select an empty highlighted cell, then choose the number you want to place.")
        for r in range(size):
            cols = st.columns(size)
            for c in range(size):
                idx = r * size + c
                disabled = idx not in legal_cells
                button_label = str(int(env.board[r, c])) if env.board[r, c] else "."
                cols[c].button(button_label, key=f"sudoku_pick_{idx}", disabled=disabled, width='stretch', on_click=_select_sudoku_cell, args=(idx,))

        if selected_cell is not None:
            cell_actions = [action for action in legal_moves if action // size == selected_cell]
            if cell_actions:
                num_cols = st.columns(len(cell_actions))
                for col, action in zip(num_cols, cell_actions):
                    value = (action % size) + 1
                    col.button(f"Place {value}", key=f"sudoku_val_{action}", width='stretch', on_click=handle_move, args=(action,))
            else:
                st.session_state.game_state["sudoku_selected_cell"] = None
        return

    if game_type == "Pac-Man":
        pellets_left = int(np.sum(env.pellets))
        _render_move_ribbon(f"Pellets left <strong>{pellets_left}</strong>. Keep space between Pac-Man and the ghost while clearing the board.")
        _render_legend(["ᗧ Pac-Man", "◉ Ghost", "• Pellet", "◌ Power pellet", "█ Wall"])
        _render_pacman_board(env)
        _metric_row([("Score", env.score), ("Pellets left", pellets_left), ("Power mode", env.frightened_turns), ("Ghost resets", env.ghost_respawns)])
        cols = st.columns(4)
        for col, action in zip(cols, [0, 1, 2, 3]):
            col.button(future_action_label(game_type, action, env), key=f"future_{game_type}_{action}", disabled=action not in legal_moves, width='stretch', on_click=handle_move, args=(action,))
        return

    if game_type == "Minecraft":
        _render_move_ribbon("You are racing the clock and your own supplies. Shelter is the clean win condition.")
        _metric_row([
            ("Wood", env.wood), ("Stone", env.stone), ("Food", env.food),
            ("Tool", "Yes" if env.tool else "No"), ("Shelter", "Built" if env.shelter else "No"), ("Health", env.health),
        ])
    elif game_type == "Among Us":
        _render_move_ribbon("Tasks keep you safe until the evidence becomes obvious enough to accuse.")
        _metric_row([("Tasks", f"{env.tasks}/5"), ("Crewmates", "Red, Blue, Green, Yellow")])
        st.bar_chart({name: [float(env.suspicion[i])] for i, name in enumerate(["Red", "Blue", "Green", "Yellow"])})
    elif game_type == "Clash Royale":
        _render_move_ribbon("Use elixir deliberately. Overcommitting makes the counterpush hurt more.")
        _metric_row([("Elixir", env.elixir), ("Your tower", env.agent_tower), ("Enemy tower", env.opp_tower)])
    elif game_type == "Flappy Bird":
        _render_move_ribbon("Stay centered, then make the smallest correction that gets you through the gap.")
        _metric_row([("Height", f"{env.y:.2f}"), ("Pipe", f"{env.pipe_x:.2f}"), ("Gap", f"{env.gap_y:.2f}"), ("Score", env.score)])
        st.progress(max(0.0, min(1.0, float(env.y))))
    elif game_type == "Ludo":
        _render_move_ribbon("The die is already rolled. Decide which token deserves the tempo.")
        _metric_row([("Dice", env.dice), ("Your tokens home", int(np.sum(env.positions == 56))), ("AI tokens home", int(np.sum(env.opp_positions == 56)))])
        st.caption(f"Your token positions: {[int(x) if x >= 0 else 'base' for x in env.positions]}")
    elif game_type == "UNO":
        colors = ["Red", "Blue", "Green", "Yellow"]
        _render_move_ribbon("Play toward color control when you can, but card count is the real pressure.")
        _metric_row([("Current color", colors[env.current_color]), ("Your cards", int(env.hand.sum())), ("AI cards", env.opp_cards)])
        hand_summary = ", ".join(f"{colors[i]}: {int(env.hand[i])}" for i in range(4))
        st.caption(f"Your hand: {hand_summary}")
    elif game_type == "Scrabble":
        _render_move_ribbon("Short words score here too. Steady tempo can beat waiting for the perfect word.")
        _metric_row([("Your score", env.score), ("AI score", env.opp_score), ("Words left", len(legal_moves))])
    elif game_type == "Monopoly":
        _render_move_ribbon("These are short, aggressive matches. Buy if it compounds fast, sell if it unlocks momentum.")
        _metric_row([("Space", env.position), ("Cash", f"${env.cash}"), ("Properties", env.properties), ("AI net worth", f"${env.opp_net}")])
    elif game_type == "Penalty Shootout":
        _render_move_ribbon("Commit to a side and trust it. Hesitation is basically a miss.")
        _metric_row([("Goals", env.goals), ("Kick", f"{env.kicks + 1}/5")])
    elif game_type == "Cricket Strategy":
        _render_move_ribbon("The chase line is always visible. Choose whether this ball is about control or acceleration.")
        _metric_row([("Runs", env.runs), ("Target", env.target), ("Wickets", f"{env.wickets}/3"), ("Balls", f"{env.balls}/12")])

    cols = st.columns(min(4, max(1, len(legal_moves))))
    for i, action in enumerate(legal_moves):
        cols[i % len(cols)].button(future_action_label(game_type, action, env), key=f"future_{game_type}_{action}", width='stretch', on_click=handle_move, args=(action,))

def _resolve_turn(player_move, opponent_policy, player_label, opponent_label, player_agent=None, opponent_agent=None):
    st.session_state.game_state["sudoku_selected_cell"] = None
    st.session_state.game_state["chess_selected_sq"] = None
    st.session_state.game_state["checkers_selected"] = None
    st.session_state.game_state["ui_error"] = None
    env = st.session_state.game_state["env"]
    game_type = st.session_state.game_state["game_type"]
    opponent_move = None
    prior_history = [
        int(entry.get("player_action"))
        for entry in st.session_state.get("move_history", [])
        if entry.get("player_action") is not None and int(entry.get("player_action")) >= 0
    ]
    legal_for_surprisal = st.session_state.game_state.get("info", {}).get("legal_moves")
    if legal_for_surprisal is None and hasattr(env, "legal_actions"):
        try:
            legal_for_surprisal = list(env.legal_actions())
        except Exception:
            legal_for_surprisal = None
    surprisal_entry = _compute_move_surprisal(opponent_agent, int(player_move), prior_history, legal_for_surprisal)

    with st.spinner("Resolving turn..."):
        original_opp_policy = getattr(env, "_opponent_policy", None)

        def wrapped_policy():
            nonlocal opponent_move
            action = opponent_policy()
            opponent_move = int(action)
            if game_type == "RPS+":
                from environments.rps_plus import Move
                return Move(opponent_move)
            return opponent_move

        env._opponent_policy = wrapped_policy

        try:
            new_obs, reward, terminated, truncated, new_info = env.step(player_move)
            if opponent_move is None and getattr(env.state, "history", None):
                opponent_move = env.state.history[-1].opponent_move
        except Exception as exc:
            st.session_state.game_state["ui_error"] = f"{game_type} hit an internal move error: {exc}"
            return
        finally:
            if original_opp_policy:
                env._opponent_policy = original_opp_policy
            elif hasattr(env, "_opponent_policy"):
                delattr(env, "_opponent_policy")

        _observe_agent(player_agent, player_move, opponent_move, reward)
        _observe_agent(opponent_agent, opponent_move, player_move, -reward)
            
    st.session_state.game_state["obs"] = new_obs
    st.session_state.game_state["info"] = new_info
    st.session_state.game_state["done"] = terminated or truncated
    st.session_state.game_state["pending_player_move"] = None
    st.session_state["round_count"] = env.state.turn
    
    log_entry = {
        "turn": env.state.turn,
        "player": get_move_emoji(player_move, game_type, env),
        "impostor": get_move_emoji(opponent_move, game_type, env),
        "result": "WIN" if reward > 0 else "LOSS" if reward < 0 else "TIE",
        "color": "#27ae60" if reward > 0 else "#e74c3c" if reward < 0 else "#7f8c8d",
        "player_label": player_label,
        "opponent_label": opponent_label,
    }
    st.session_state.game_state["history"].insert(0, log_entry)
    st.session_state["move_history"].append(
        {
            "turn": env.state.turn,
            "player_action": int(player_move),
            "opponent_action": int(opponent_move) if opponent_move is not None else -1,
            "reward": float(reward),
            "log": dict(log_entry),
            "surprisal": dict(surprisal_entry) if surprisal_entry else None,
        }
    )
    st.session_state["move_history"] = st.session_state["move_history"][-10:]
    if surprisal_entry:
        st.session_state.game_state.setdefault("surprisal_history", []).append(
            {
                "turn": int(env.state.turn),
                "action": int(player_move),
                **surprisal_entry,
            }
        )
        st.session_state.game_state["surprisal_history"] = st.session_state.game_state["surprisal_history"][-200:]
        st.session_state.game_state["last_surprisal"] = dict(surprisal_entry)
    else:
        st.session_state.game_state["last_surprisal"] = None

    if st.session_state.game_state["done"] and not st.session_state.game_state.get("saved_to_profile", False):
        user_prof = _active_user_profile()
        try:
            saved = _save_live_game_to_profile(
                env=env,
                game_type=st.session_state.game_state["game_type"],
                player_id=user_prof.get("id"),
                player_name=user_prof.get("name", "Player"),
                opponent_name=opponent_label,
            )
            st.session_state.game_state["saved_to_profile"] = saved
        except Exception as e:
            st.error(f"Failed to log game: {e}")
    _record_score_if_needed()


def _step_with_injected_policy(env, player_move, opponent_policy):
    original_opp_policy = getattr(env, "_opponent_policy", None)

    def wrapped_policy():
        action = opponent_policy()
        if st.session_state.game_state.get("game_type") == "RPS+":
            from environments.rps_plus import Move
            return Move(int(action))
        return int(action)

    env._opponent_policy = wrapped_policy
    try:
        return env.step(player_move)
    finally:
        if original_opp_policy is not None:
            env._opponent_policy = original_opp_policy
        elif hasattr(env, "_opponent_policy"):
            delattr(env, "_opponent_policy")


def handle_move(player_move):
    if st.session_state.get("paused"):
        st.toast("Resume the game to continue.")
        return
    mode = st.session_state.game_state.get("play_mode", "human_bot1")
    env = st.session_state.game_state["env"]
    game_type = st.session_state.game_state["game_type"]
    player_label, opponent_label = _side_labels()

    if mode == "human_human":
        pending = st.session_state.game_state.get("pending_player_move")
        if pending is None:
            legal = _opponent_legal_after_pending(env, game_type, player_move)
            if legal:
                st.session_state.game_state["pending_player_move"] = player_move
            else:
                _resolve_turn(player_move, lambda: 0, player_label, opponent_label)
            st.session_state.game_state["ui_error"] = None
            return
        legal = _opponent_legal_after_pending(env, game_type, pending)
        opponent_move = _coerce_action_to_legal(player_move, legal)
        _resolve_turn(pending, lambda: opponent_move, player_label, opponent_label)
        return

    agent = st.session_state.game_state["agent"] if mode == "human_bot1" else st.session_state.game_state.get("agent2")
    obs = st.session_state.game_state["obs"]
    info = st.session_state.game_state["info"]

    def bot_policy():
        return _agent_action(agent, env, game_type, obs, info, current_player_move=player_move, side=-1)

    _resolve_turn(player_move, bot_policy, player_label, opponent_label, opponent_agent=agent)


def handle_bot_turn():
    if st.session_state.get("paused"):
        return
    env = st.session_state.game_state["env"]
    game_type = st.session_state.game_state["game_type"]
    obs = st.session_state.game_state["obs"]
    info = st.session_state.game_state["info"]
    bot1 = st.session_state.game_state["agent"]
    bot2 = st.session_state.game_state.get("agent2")
    player_label, opponent_label = _side_labels()
    player_move = _agent_action(bot1, env, game_type, obs, info, side=1)

    def bot2_policy():
        return _agent_action(bot2, env, game_type, obs, info, current_player_move=player_move, side=-1)

    _resolve_turn(player_move, bot2_policy, player_label, opponent_label, player_agent=bot1, opponent_agent=bot2)


def _render_inline_match_setup(game_type: str, active_mode: str) -> None:
    if _ladder_enabled():
        rung = _active_ladder_rung() or {}
        st.info(
            f"Draft Mode Clone Ladder is active. Arena setup is pinned to RPS+ vs `{rung.get('agent', 'clone')}` until the ladder run ends."
        )
        return
    agent_options = _supported_agents_for_game(game_type)
    mode_keys = list(PLAY_MODES.keys())
    current_label = _play_mode_label(active_mode)
    if current_label not in mode_keys:
        current_label = "Play a Bot"
    current_agent = st.session_state.game_state.get("agent_name") or agent_options[0]
    if current_agent not in agent_options:
        current_agent = agent_options[0]
    current_agent_2 = st.session_state.game_state.get("agent2_name") or current_agent
    if current_agent_2 not in agent_options:
        current_agent_2 = agent_options[0]
    current_turns = st.session_state.game_state["env"].max_turns if hasattr(st.session_state.game_state["env"], "max_turns") else 30

    with st.expander("Match Setup: mode, bots, turns", expanded=bool(st.session_state.get("focus_match_setup")) or not st.session_state.game_state.get("history")):
        with st.form("inline_match_setup_form"):
            setup_cols = st.columns([1, 1, 1, 0.8], gap="medium")
            with setup_cols[0]:
                selected_mode_label = st.selectbox(
                    "Match type",
                    mode_keys,
                    index=mode_keys.index(current_label),
                    help="Choose human vs bot, human vs human, or bot vs bot.",
                )
                selected_mode = PLAY_MODES[selected_mode_label]
            selected_agent = current_agent
            selected_agent_2 = current_agent_2
            with setup_cols[1]:
                if selected_mode in {"human_bot1", "bot_bot"}:
                    selected_agent = st.selectbox(
                        "Bot 1",
                        agent_options,
                        index=agent_options.index(current_agent),
                        help="Bot used for Play a Bot and the blue side in Watch Bots.",
                    )
                else:
                    st.caption("Bot 1 not used for this match type.")
            with setup_cols[2]:
                if selected_mode in {"human_bot2", "bot_bot"}:
                    selected_agent_2 = st.selectbox(
                        "Bot 2",
                        agent_options,
                        index=agent_options.index(current_agent_2),
                        help="Bot used for Try Another Bot and the red side in Watch Bots.",
                    )
                else:
                    st.caption("Bot 2 not used for this match type.")
            with setup_cols[3]:
                max_turns = st.number_input("Turns", min_value=10, max_value=100, value=int(current_turns), step=5)
            if st.form_submit_button("Apply Match Setup", width='stretch'):
                st.session_state["focus_match_setup"] = False
                _apply_game_settings(game_type, selected_agent, int(max_turns), selected_mode, selected_agent_2)
                st.rerun()


def _render_progress_and_replay(env, active_mode: str, player_label: str, opponent_label: str) -> None:
    turn_limit = env.max_turns if hasattr(env, "max_turns") else 1
    try:
        progress = min(1.0, max(0.0, float(env.state.turn) / max(1, float(turn_limit))))
    except Exception:
        progress = 0.0
    st.caption("Match progress")
    st.progress(progress, text=f"Turn {env.state.turn}/{turn_limit} · Score: {player_label} {env.state.agent_score} - {env.state.opponent_score} {opponent_label}")
    ctl_cols = st.columns(3)
    if ctl_cols[0].button("Start Fresh", width='stretch'):
        _apply_game_settings(
            st.session_state.game_state["game_type"],
            st.session_state.game_state.get("agent_name", "random"),
            int(turn_limit),
            active_mode,
            st.session_state.game_state.get("agent2_name"),
        )
        st.rerun()
    if ctl_cols[1].button("Replay Last Turn", width='stretch', disabled=not bool(st.session_state.game_state.get("history"))):
        st.session_state["show_last_turn_replay"] = True
    if active_mode == "bot_bot":
        ctl_cols[2].button("Play Next Turn", width='stretch', disabled=st.session_state.game_state["done"], on_click=handle_bot_turn)
    else:
        ctl_cols[2].button("Change Setup in Sidebar", width='stretch', disabled=True)
    if st.session_state.get("show_last_turn_replay") and st.session_state.game_state.get("history"):
        entry = st.session_state.game_state["history"][0]
        st.markdown(
            f"""
            <div class="log-entry" style="border-left: 4px solid {entry['color']}; margin-top: 10px;">
                <strong>Replay Turn {entry['turn']}</strong>
                <div style="margin-top:7px; color: var(--game-surface-muted);">
                    {entry.get('player_label', player_label)} played <strong style="color: var(--game-surface);">{entry['player']}</strong><br/>
                    {entry.get('opponent_label', opponent_label)} replied with <strong style="color: var(--game-surface);">{entry['impostor']}</strong><br/>
                    Result: <strong style="color:{entry['color']};">{entry['result']}</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# Session State Initialization
if "game_state" not in st.session_state:
    st.session_state.game_state = None
if "move_history" not in st.session_state:
    st.session_state["move_history"] = []
if "round_count" not in st.session_state:
    st.session_state["round_count"] = 0
if "score_you" not in st.session_state:
    st.session_state["score_you"] = 0
if "score_agent" not in st.session_state:
    st.session_state["score_agent"] = 0
if "score_draws" not in st.session_state:
    st.session_state["score_draws"] = 0
if "game_over" not in st.session_state:
    st.session_state["game_over"] = False
if "paused" not in st.session_state:
    st.session_state["paused"] = False
if "winner" not in st.session_state:
    st.session_state["winner"] = None
if "series_complete" not in st.session_state:
    st.session_state["series_complete"] = False
if "ladder_state" not in st.session_state:
    st.session_state["ladder_state"] = {
        "active": False,
        "rung_index": 0,
        "wins": 0,
        "losses": 0,
        "draws": 0,
        "completed": False,
        "run_id": None,
        "rung_history": [],
    }
if "blind_study_state" not in st.session_state:
    _reset_blind_study_state()
if "show_hints" not in st.session_state:
    st.session_state["show_hints"] = DEFAULT_HINTS
if "game_settings" not in st.session_state:
    st.session_state["game_settings"] = {
        "series_length": 3,
        "rps_starting_energy": 3,
        "chess_time_control": "Unlimited",
        "gomoku_board_size": 15,
        "nim_preset": "Classic",
        "nim_custom_piles": [1, 3, 5, 7],
        "checkers_forced_jump": True,
        "opening_challenge": None,
        "friend_clone_source_player": None,
        "clone_ladder_enabled": False,
    }
if st.session_state.game_state is None:
    st.session_state.game_state = {
        "game_type": "RPS+",
        "env": RPSPlusEnv(max_turns=30, starting_energy=int(st.session_state["game_settings"]["rps_starting_energy"])),
        "agent": None,
        "agent_name": "",
        "agent2": None,
        "agent2_name": "",
        "play_mode": "human_bot1",
        "pending_player_move": None,
        "obs": None,
        "info": None,
        "history": [],
        "surprisal_history": [],
        "last_surprisal": None,
        "done": False,
        "saved_to_profile": False,
        "score_recorded": False,
        "last_result": None,
        "chess_selected_sq": None,
        "sudoku_selected_cell": None,
        "checkers_selected": None,
        "ui_error": None,
    }
st.session_state.game_state.setdefault("surprisal_history", [])
st.session_state.game_state.setdefault("last_surprisal", None)

if "show_experimental_games" not in st.session_state:
    st.session_state["show_experimental_games"] = True
if "sidebar_library_view" not in st.session_state:
    st.session_state["sidebar_library_view"] = "Playable Now"
if "sidebar_game_type" not in st.session_state:
    st.session_state["sidebar_game_type"] = st.session_state.game_state["game_type"]
if "sidebar_agent_name" not in st.session_state:
    st.session_state["sidebar_agent_name"] = st.session_state.game_state["agent_name"] or next(iter(AGENT_REGISTRY.keys()))
if "sidebar_max_turns" not in st.session_state:
    st.session_state["sidebar_max_turns"] = st.session_state.game_state["env"].max_turns if hasattr(st.session_state.game_state["env"], "max_turns") else 30

launch_settings = st.session_state.pop("live_game_launch", None)
if launch_settings:
    launch_game = launch_settings.get("game_type", st.session_state["sidebar_game_type"])
    launch_agents = _supported_agents_for_game(launch_game)
    launch_agent = launch_settings.get("agent_name") or launch_agents[0]
    launch_agent_2 = launch_settings.get("agent2_name") or launch_agent
    launch_mode = launch_settings.get("play_mode", "human_bot1")
    launch_turns = int(launch_settings.get("max_turns", 40))

    st.session_state["sidebar_library_view"] = launch_settings.get("library_view", "Playable Now")
    st.session_state["sidebar_game_type"] = launch_game
    st.session_state["sidebar_agent_name"] = launch_agent
    st.session_state["sidebar_agent2_name"] = launch_agent_2
    st.session_state["sidebar_play_mode"] = launch_mode
    st.session_state["sidebar_max_turns"] = launch_turns
    _apply_game_settings(launch_game, launch_agent, launch_turns, launch_mode, launch_agent_2)

_old_arcade_mode = "Arcade" + " Mode"
_old_research_mode = "Research" + " Match"
if st.session_state.get("live_arena_experience") == _old_arcade_mode:
    st.session_state["live_arena_experience"] = "Play Arcade"
elif st.session_state.get("live_arena_experience") == _old_research_mode:
    st.session_state["live_arena_experience"] = "Train Clone Match"

# Sidebar settings
with st.sidebar:
    render_sidebar_nav("Live Arena")
    st.divider()
    st.caption("Arena Settings")
    arena_experience = st.radio(
        "Live Arena mode",
        ["Train Clone Match", "Play Arcade"],
        key="live_arena_experience",
        help="Train Clone Match saves games to profiles. Play Arcade embeds the polished browser arcade here.",
    )
    st.radio(
        "Arena theme",
        ["Auto", "Light", "Dark"],
        key="arena_theme",
        horizontal=True,
        help="Auto follows your system theme. Use Dark if the arena is hard to read on a dark desktop.",
    )
    
    # Global Player Status
    user_prof = _active_user_profile()
    st.markdown(f"""
        <div class="dg-sidebar-card" style="padding: 15px; margin-bottom: 20px;">
            <div style="font-size: 10px; opacity: 0.6; text-transform: uppercase;">Active Researcher</div>
            <div style="font-size: 16px; font-weight: 700; color: #9fd3b3;">{user_prof['name']}</div>
            <div style="font-size: 10px; opacity: 0.4; margin-top: 4px;">ID: {user_prof['id']}</div>
        </div>
    """, unsafe_allow_html=True)

    with st.expander("Data & Consent", expanded=False):
        st.caption("What gets saved")
        st.markdown(
            "- `Train Clone Match` saves completed games, move history, and research metrics to your local Doppelgamer database.\n"
            "- `Play Arcade` is visual-only and does not train your clone."
        )
        st.caption("Study mode")
        st.markdown(
            "- `Blind Turing Study` hides the opponent identity until the study block ends.\n"
            "- Your human/not-human judgment and confidence are recorded for research analysis."
        )
        st.caption("Control")
        st.markdown("Use **Train My Clone** to review, export, or delete your saved clone data.")
    
    st.markdown("---")
    with st.form("arena_settings_form"):
        library_view = st.radio(
            "Browse Library",
            ["Playable Now", "Board Games", "Research Sandbox", "Full Library"],
            index=["Playable Now", "Board Games", "Research Sandbox", "Full Library"].index(st.session_state.get("sidebar_library_view", "Playable Now")),
            horizontal=True,
        )
        st.caption("Playable games are ready for profile data collection. Experimental games are useful for behavior research; visual polish varies.")
        game_options = _game_library_options(library_view)
        current_game = st.session_state.get("sidebar_game_type", st.session_state.game_state["game_type"])
        if current_game not in game_options:
            current_game = game_options[0]
        selected_game = st.selectbox(
            "🎮 Select Game",
            game_options,
            index=game_options.index(current_game),
            format_func=_game_label,
        )
        st.caption(f"**{_game_status(selected_game)}** · {_game_spotlight(selected_game)}")

        agent_options = _supported_agents_for_game(selected_game)
        current_mode = st.session_state.get("sidebar_play_mode", st.session_state.game_state.get("play_mode", "human_bot1"))
        if current_mode not in PLAY_MODES.values():
            current_mode = "human_bot1"
        selected_mode_label = st.radio(
            "Who is playing?",
            list(PLAY_MODES.keys()),
            index=list(PLAY_MODES.values()).index(current_mode),
            captions=[PLAY_MODE_HELP[label] for label in PLAY_MODES],
        )
        selected_mode = PLAY_MODES[selected_mode_label]

        current_agent = st.session_state.get("sidebar_agent_name", st.session_state.game_state["agent_name"] or agent_options[0])
        if current_agent not in agent_options:
            current_agent = agent_options[0]
        current_agent_2 = st.session_state.get("sidebar_agent2_name", st.session_state.game_state.get("agent2_name") or current_agent)
        if current_agent_2 not in agent_options:
            current_agent_2 = agent_options[0]

        selected_agent = current_agent
        selected_agent_2 = current_agent_2
        if selected_mode == "human_bot1":
            selected_agent = st.selectbox(
                "Choose your opponent",
                agent_options,
                index=agent_options.index(current_agent),
            )
            st.caption(_agent_description(selected_agent))
            st.caption(_agent_support_note(selected_game, selected_agent))
        elif selected_mode == "human_bot2":
            selected_agent_2 = st.selectbox(
                "Choose your opponent",
                agent_options,
                index=agent_options.index(current_agent_2),
            )
            st.caption(_agent_description(selected_agent_2))
            st.caption(_agent_support_note(selected_game, selected_agent_2))
        elif selected_mode == "bot_bot":
            bot_col_1, bot_col_2 = st.columns(2)
            with bot_col_1:
                selected_agent = st.selectbox(
                    "Blue side bot",
                    agent_options,
                    index=agent_options.index(current_agent),
                )
            with bot_col_2:
                selected_agent_2 = st.selectbox(
                    "Red side bot",
                    agent_options,
                    index=agent_options.index(current_agent_2),
                )
            st.caption(f"Blue: {_agent_description(selected_agent)}")
            st.caption(f"Red: {_agent_description(selected_agent_2)}")
        else:
            st.info("Pass-and-play mode: Player 1 moves first, then Player 2 replies on the same screen.")

        max_turns = st.slider(
            "⏱️ Max Turns",
            10,
            100,
            st.session_state.get("sidebar_max_turns", st.session_state.game_state["env"].max_turns if hasattr(st.session_state.game_state["env"], "max_turns") else 30),
        )
        apply_settings = st.form_submit_button("Start / Apply", width='stretch')

    st.session_state["show_experimental_games"] = True
    st.session_state["sidebar_library_view"] = library_view
    st.session_state["sidebar_game_type"] = selected_game
    st.session_state["sidebar_agent_name"] = selected_agent
    st.session_state["sidebar_agent2_name"] = selected_agent_2
    st.session_state["sidebar_play_mode"] = selected_mode
    st.session_state["sidebar_max_turns"] = max_turns

    if apply_settings:
        _apply_game_settings(selected_game, selected_agent, max_turns, selected_mode, selected_agent_2)

if st.session_state.get("live_arena_experience") == "Play Arcade":
    _render_arcade_mode()
    st.stop()

# Initial Auto-start if agent is None
if st.session_state.game_state["agent"] is None:
    supported_agents = _supported_agents_for_game(st.session_state.game_state["game_type"])
    initial_agent = selected_agent if selected_agent in supported_agents else supported_agents[0]
    st.session_state.game_state["agent"] = _new_agent(initial_agent)
    st.session_state.game_state["agent_name"] = initial_agent
    initial_agent_2 = st.session_state.get("sidebar_agent2_name", initial_agent)
    if initial_agent_2 not in supported_agents:
        initial_agent_2 = supported_agents[0]
    st.session_state.game_state["agent2"] = _new_agent(initial_agent_2)
    st.session_state.game_state["agent2_name"] = initial_agent_2
    obs, info = st.session_state.game_state["env"].reset()
    st.session_state.game_state["obs"] = obs
    st.session_state.game_state["info"] = info

env = st.session_state.game_state["env"]
game_type = st.session_state.game_state["game_type"]
st.markdown(_game_skin_css(game_type), unsafe_allow_html=True)
user_prof = _active_user_profile()
player_label, opponent_label = _side_labels()
bot_status_label, bot_status_detail = _agent_runtime_status(
    st.session_state.game_state["agent_name"],
    st.session_state.game_state["agent"],
)
bot2_status_label, bot2_status_detail = _agent_runtime_status(
    st.session_state.game_state.get("agent2_name", ""),
    st.session_state.game_state.get("agent2"),
)
active_mode = st.session_state.game_state.get("play_mode", "human_bot1")
ladder_rung = _active_ladder_rung()
if active_mode == "human_human":
    match_status_chip = "Pass-and-play"
elif active_mode == "human_bot2":
    match_status_chip = f"Bot: {bot2_status_label}"
elif active_mode == "bot_bot":
    match_status_chip = f"{bot_status_label} vs {bot2_status_label}"
else:
    match_status_chip = f"Bot: {bot_status_label}"
if _blind_study_active():
    blind_entry = _current_blind_entry() or {}
    match_status_chip = blind_entry.get("blind_label", "Blind Opponent")
if st.session_state.game_state.get("ui_error"):
    st.error(st.session_state.game_state["ui_error"])
pending_move = st.session_state.game_state.get("pending_player_move")
active_turn_label = opponent_label if active_mode == "human_human" and pending_move is not None else player_label
turn_limit = env.max_turns if hasattr(env, "max_turns") else "n/a"
st.title("Live Arena")
st.markdown(
    f"""
    <div class="arena-hero">
        <div class="arena-kicker">Play</div>
        <h1 class="arena-title">{game_type}</h1>
        <div class="arena-subtitle">{_game_description(game_type)}</div>
        <div class="match-strip">
            <div class="player-badge"><span>You</span><strong>{player_label}</strong></div>
            <div class="versus-mark">VS</div>
            <div class="player-badge"><span>Opponent</span><strong>{opponent_label}</strong></div>
        </div>
        <div class="arena-chip-row">
            <div class="arena-chip">{_play_mode_label(active_mode)}</div>
            <div class="arena-chip">{match_status_chip}</div>
            <div class="arena-chip">Turn limit: {env.max_turns if hasattr(env, "max_turns") else "n/a"}</div>
            {"<div class='arena-chip'>Clone Ladder: " + ladder_rung.get("title", ladder_rung.get("agent", "Active")) + "</div>" if _ladder_enabled() and ladder_rung else ""}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption("Use the left sidebar to switch games, change opponents, or choose a different match type.")
if _blind_study_active():
    st.warning(
        "Blind Turing Study is active. Opponent identity stays hidden until the study block ends, and your post-match detection judgment will be recorded."
    )
st.divider()

# Layout
col_play, col_log = st.columns([3, 2], gap="large")

with col_play:
    play_tab, rules_tab, bot_tab = st.tabs(["Play", "How to play", "Opponent"])

    with play_tab:
        _render_game_progress(env, game_type, player_label, opponent_label)
        _render_progress_and_replay(env, active_mode, player_label, opponent_label)
        _render_hint_panel(game_type, env, st.session_state.game_state.get("info") or {})
        st.divider()
        st.markdown(
            f"""
            <div class="move-ribbon" style="margin-top:0;">
                <strong>What to do next</strong><br/>
                Make your move on the board below. Need a different game or bot? Open the left sidebar.
            </div>
            """,
            unsafe_allow_html=True,
        )

        if game_type == "RPS+":
            c1, c2 = st.columns(2)
            with c1:
                render_energy(player_label, env.state.agent_energy, color="var(--accent-blue)")
            with c2:
                render_energy(opponent_label, env.state.opponent_energy, color="var(--accent-red)")
            st.markdown("<hr style='opacity:0.1'>", unsafe_allow_html=True)

        active_mode = st.session_state.game_state.get("play_mode", "human_bot1")
        pending_move = st.session_state.game_state.get("pending_player_move")
        input_disabled = st.session_state.get("paused", False) or st.session_state.get("game_over", False)
        if st.session_state.get("paused", False):
            st.info("The match is paused. Resume to allow the next move.")

        if active_mode == "bot_bot" and not st.session_state.game_state["done"]:
            _render_move_ribbon(f"Ready to watch: {player_label} moves first, {opponent_label} replies, and the log records the turn.")
            st.button("Advance Match", width='stretch', on_click=handle_bot_turn, disabled=input_disabled)

        elif not st.session_state.game_state["done"]:
            legal_moves = (
                _opponent_legal_after_pending(env, game_type, pending_move)
                if active_mode == "human_human" and pending_move is not None
                else st.session_state.game_state["info"]["legal_moves"]
            )
            active_human_label = opponent_label if active_mode == "human_human" and pending_move is not None else player_label
            if active_mode == "human_human" and pending_move is not None:
                _render_move_ribbon(f"{player_label} picked {get_move_emoji(pending_move, game_type, env)}. Pass the screen to {opponent_label}.")
            else:
                _render_move_ribbon(f"{active_human_label}'s turn. Choose a highlighted legal move.")

            if game_type == "RPS+":
                board_left, board_mid, board_right = st.columns([0.25, 1.6, 0.25], gap="small")
                with board_mid:
                    top_row = st.columns(3, gap="small")
                    bottom_row = st.columns(3, gap="small")
                    moves_list = [
                        (Move.ROCK, top_row[0], "🪨"),
                        (Move.PAPER, top_row[1], "📄"),
                        (Move.SCISSORS, top_row[2], "✂️"),
                        (Move.LIZARD, bottom_row[0], "🦎"),
                        (Move.POWER, bottom_row[1], "⚡"),
                        (Move.RECHARGE, bottom_row[2], "🔋"),
                    ]
                    for move_enum, col, icon in moves_list:
                        move_idx = int(move_enum)
                        is_legal = move_idx in legal_moves
                        col.button(f"{icon} {move_enum.name}", key=f"move_{move_idx}", disabled=(not is_legal) or input_disabled, width='stretch', on_click=handle_move, args=(move_idx,))

            elif game_type == "Tic-Tac-Toe":
                board = env.board
                for row in range(3):
                    cols = st.columns(3)
                    for col in range(3):
                        idx = row * 3 + col
                        val = board[idx]
                        label = "X" if val == 1 else "O" if val == -1 else " "
                        cols[col].button(label, key=f"ttt_{idx}", disabled=(idx not in legal_moves) or input_disabled, width='stretch', on_click=handle_move, args=(idx,))

            elif game_type == "Connect Four":
                board = env.board
                _render_legend(["🟡 You", "🔴 Impostor", "Drop from the top"])
                cols_render = st.columns(7)
                for c in range(7):
                    cols_render[c].button("Drop", key=f"cf_btn_{c}", disabled=(c not in legal_moves) or input_disabled, width='stretch', on_click=handle_move, args=(c,))
                _render_connect_four_board(board)

            elif game_type == "Othello":
                board = env.board
                black_count = int(np.sum(board == 1))
                white_count = int(np.sum(board == -1))
                _render_legend([f"⚫ Blue {black_count}", f"⚪ Red {white_count}", "· Legal move"])
                for r in range(8):
                    cols = st.columns(8)
                    for c in range(8):
                        idx = r * 8 + c
                        val = int(board[r, c])
                        is_legal = idx in legal_moves
                        label = "●" if val == 1 else "○" if val == -1 else "•" if is_legal else " "
                        cols[c].button(
                            label,
                            key=f"oth_{idx}",
                            disabled=(not is_legal) or input_disabled,
                            width='stretch',
                            on_click=handle_move,
                            args=(idx,),
                        )

            elif game_type == "Checkers":
                board = env.board

                if "checkers_selected" not in st.session_state.game_state:
                    st.session_state.game_state["checkers_selected"] = None

                selected = st.session_state.game_state["checkers_selected"]
                valid_from = {a // 64 for a in legal_moves}
                valid_to = {a % 64 for a in legal_moves if a // 64 == selected} if selected is not None else set()
                _render_legend(["🔴 You", "⚫ Impostor", "✦ Destination"])
                if selected is not None:
                    row = 8 - (selected // 8)
                    col = chr(65 + (selected % 8))
                    _render_move_ribbon(f"Selected {col}{row}. Choose a highlighted destination.")
                else:
                    _render_move_ribbon("Choose one of your movable pieces, then pick the destination tile.")

                for r in range(8):
                    cols = st.columns(8)
                    for c in range(8):
                        idx = r * 8 + c
                        val = board[idx]
                        icon = ""
                        if val == 1:
                            icon = "🔴"
                        elif val == 2:
                            icon = "👑🔴"
                        elif val == -1:
                            icon = "⚫"
                        elif val == -2:
                            icon = "👑⚫"

                        is_dark = (r + c) % 2 == 1

                        if not is_dark:
                            cols[c].markdown("<div style='height:40px'></div>", unsafe_allow_html=True)
                            continue

                        if selected == idx:
                            cols[c].button("Selected", key=f"chk_{idx}", width='stretch', on_click=_clear_checkers_square, disabled=input_disabled)
                        elif idx in valid_to:
                            action = selected * 64 + idx
                            cols[c].button("✦", key=f"chk_to_{idx}", width='stretch', on_click=handle_move, args=(action,), disabled=input_disabled)
                        else:
                            disabled = idx not in valid_from and val == 0
                            label = icon or " "
                            cols[c].button(label, key=f"chk_{idx}", disabled=disabled or input_disabled, width='stretch', on_click=_select_checkers_square, args=(idx,))

            elif game_type == "Gomoku":
                board = env.board
                _render_legend(["● Blue side", "○ Red side", "· Legal move"])
                for r in range(board.shape[0]):
                    cols = st.columns(board.shape[1])
                    for c in range(board.shape[1]):
                        idx = r * board.shape[1] + c
                        val = int(board[r, c])
                        label = "●" if val == 1 else "○" if val == -1 else " "
                        cols[c].button(label, key=f"gomoku_{idx}", disabled=(idx not in legal_moves) or input_disabled, width='stretch', on_click=handle_move, args=(idx,))

            elif game_type == "Nim":
                _render_move_ribbon("Choose a pile, then take one to three stones. Taking the final stone wins.")
                for pile_idx, count in enumerate(env.piles):
                    stones = "● " * int(count) if int(count) else "empty"
                    st.markdown(
                        f"""
                        <div class="move-ribbon" style="margin-top:8px;">
                            <strong>Pile {pile_idx + 1}</strong>
                            <div style="font-size:22px; margin-top:6px;">{stones}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    cols = st.columns(3)
                    for remove in range(1, 4):
                        action = pile_idx * 3 + (remove - 1)
                        cols[remove - 1].button(
                            f"Take {remove}",
                            key=f"nim_{action}",
                            disabled=(action not in legal_moves) or input_disabled,
                            width='stretch',
                            on_click=handle_move,
                            args=(action,),
                        )

            elif game_type == "War":
                agent_deck_size, opp_deck_size, agent_card, opp_card = env._obs()
                st.markdown(f"**Your Deck:** {int(agent_deck_size)} cards | **Opponent Deck:** {int(opp_deck_size)} cards")

                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"<div style='border:2px solid #555;border-radius:10px;padding:40px;text-align:center;font-size:40px;background:rgba(255,255,255,0.05)'>YOU<br>{int(agent_card) if agent_card > 0 else '❓'}</div>", unsafe_allow_html=True)
                with c2:
                    st.markdown(f"<div style='border:2px solid #555;border-radius:10px;padding:40px;text-align:center;font-size:40px;background:rgba(255,255,255,0.05)'>OPP<br>{int(opp_card) if opp_card > 0 else '❓'}</div>", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.button("Draw Card", width='stretch', disabled=st.session_state.game_state["done"] or input_disabled, on_click=handle_move, args=(0,))

            elif game_type == "Chess":
                board_obj = getattr(env, "_board", None)
                if board_obj:
                    selected_square = st.session_state.game_state.get("chess_selected_sq")
                    legal_moves_list = list(board_obj.legal_moves)
                    valid_from = {move.from_square for move in legal_moves_list}
                    valid_to = {move.to_square for move in legal_moves_list if move.from_square == selected_square} if selected_square is not None else set()

                    for display_rank in range(7, -1, -1):
                        cols = st.columns(8)
                        for file_idx in range(8):
                            square = display_rank * 8 + file_idx
                            piece = board_obj.piece_at(square)
                            label = _chess_piece_label(piece)
                            if square == selected_square:
                                label = f"[{label.strip() or ' '}]"
                            elif square in valid_to:
                                label = f"{label.strip() or '.'}*"

                            if square == selected_square:
                                cols[file_idx].button(label, key=f"chess_sq_{square}", width='stretch', on_click=_clear_chess_square, disabled=input_disabled)
                            elif square in valid_to:
                                move_idx = _chess_action_for_target(board_obj, selected_square, square)
                                cols[file_idx].button(label, key=f"chess_sq_{square}", width='stretch', on_click=handle_move, args=(move_idx,), disabled=input_disabled)
                            else:
                                disabled = square not in valid_from
                                cols[file_idx].button(label, key=f"chess_sq_{square}", disabled=disabled or input_disabled, width='stretch', on_click=_select_chess_square, args=(square,))

                    st.caption(f"Turn: {'White' if board_obj.turn else 'Black'} | Status: {'Check!' if board_obj.is_check() else 'Normal'}")
                    if selected_square is not None and not valid_to:
                        st.info("That piece has no legal moves right now. Pick another square.")
                    if board_obj.is_game_over():
                        st.warning(f"Game Over: {board_obj.result()}")

            elif game_type in FUTURE_GAME_LABELS:
                render_future_game(game_type, env, legal_moves)
        else:
            outcome_title = "You took the match." if env.state.agent_score >= env.state.opponent_score else "The bot held the edge."
            st.markdown(
                f"""
                <div class="move-ribbon" style="border-color: rgba(94,234,212,0.28); background: linear-gradient(135deg, rgba(20,184,166,0.22), rgba(15,23,42,0.88)); color: var(--game-surface);">
                    <strong>Match complete.</strong> {outcome_title} Start a fresh round from the sidebar whenever you want another go.
                </div>
                """,
                unsafe_allow_html=True,
            )
        _render_game_controls(game_type)
        _render_game_over_panel(game_type, env, player_label, opponent_label)

    with rules_tab:
        st.caption("Quick guide")
        for idx, rule in enumerate(_game_rules(game_type), start=1):
            st.markdown(f"{idx}. {rule}")
        st.divider()
        st.caption("What this game feels like")
        st.markdown(_game_description(game_type))
        if game_type not in STANDARD_GAMES:
            st.info("This is still a prototype game mode. The rules work, but the presentation is lighter than the core board games.")
        with st.expander("Extra tips"):
            for tip in _game_tips(game_type):
                st.markdown(f"- {tip}")

    with bot_tab:
        if _blind_study_active():
            entry = _current_blind_entry() or {}
            st.caption("Blind study")
            st.markdown(f"Current label: **{entry.get('blind_label', 'Opponent')}**")
            st.caption("The identity stays hidden until the full study block ends.")
        elif active_mode == "human_human":
            st.caption("Pass and play")
            st.markdown("Two people are sharing one screen. Player 1 moves first, then Player 2 takes the next turn.")
        else:
            if active_mode in {"human_bot1", "bot_bot"}:
                st.caption("Opponent 1")
                st.markdown(_agent_description(st.session_state.game_state["agent_name"]))
                st.markdown(f"**{bot_status_label}**")
                st.caption(bot_status_detail)
                st.divider()
            if active_mode in {"human_bot2", "bot_bot"}:
                st.caption("Opponent 2")
                st.markdown(_agent_description(st.session_state.game_state.get("agent2_name", st.session_state.game_state["agent_name"])))
                st.markdown(f"**{bot2_status_label}**")
                st.caption(bot2_status_detail)
                st.divider()
            st.caption("What to expect")
            if active_mode in {"human_bot1", "human_bot2", "bot_bot"}:
                clone_agent = st.session_state.game_state["agent"] if active_mode != "human_bot2" else st.session_state.game_state.get("agent2")
                _render_clone_explanation(game_type, clone_agent)
            st.markdown(
                "1. Simple bots are best for warm-up matches.\n"
                "2. Stronger bots punish predictable play more often.\n"
                "3. Learning bots try to mirror patterns from saved player history.\n"
                "4. Use the sidebar any time you want a different opponent."
            )

with col_log:
    st.caption("Scoreboard")
    score_cols = st.columns(2, gap="small")
    score_cols[0].metric("You", st.session_state.game_state["env"].state.agent_score)
    score_cols[1].metric("Opponent", st.session_state.game_state["env"].state.opponent_score)
    st.metric("Turn", f"{env.state.turn}/{turn_limit}")
    active_agents = [st.session_state.game_state.get("agent_name", "random")]
    if active_mode in {"human_bot2", "bot_bot"}:
        active_agents.append(st.session_state.game_state.get("agent2_name", "profile_counter"))
    st.divider()
    _render_side_card(
        "Current Match",
        f"<strong>{player_label}</strong> is playing <strong>{game_type}</strong> against <strong>{opponent_label}</strong>. "
        f"Completed matches are saved under <strong>{user_prof['name']}</strong>."
    )
    _render_side_card("Game Snapshot", f"<strong>{_game_status(game_type)}</strong>. {_game_spotlight(game_type)}")
    if game_type == "RPS+" and st.session_state.game_state.get("history"):
        recency_note = recency_bias_warning(
            pd.DataFrame(
                [{"agent_move_name": entry.get("player", "")} for entry in st.session_state.game_state["history"]]
            )
        )
        if recency_note:
            _render_side_card("Pattern Warning", recency_note)
    st.divider()
    with st.expander("Opponent insights", expanded=False):
        _render_clone_stats_panel(env, active_agents, len(st.session_state.game_state.get("history", [])))
    st.divider()
    st.caption("Recent turns")
    history_box = st.container(height=300)
    with history_box:
        if st.session_state.game_state["history"]:
            for entry in st.session_state.game_state["history"]:
                st.markdown(f"""
                    <div class="log-entry" style="border-left: 4px solid {entry['color']}">
                        <div style="display:flex; justify-content:space-between; gap: 10px;">
                            <span><strong>Turn {entry['turn']}</strong></span>
                            <span style="color:{entry['color']}; font-weight:700;">{entry['result']}</span>
                        </div>
                        <div style="margin-top:7px; color: var(--game-surface-muted);">
                            You played <strong style="color: var(--game-surface);">{entry['player']}</strong><br/>
                            Opponent played <strong style="color: var(--game-surface);">{entry['impostor']}</strong>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.caption("Your move list will appear here after the first turn.")
