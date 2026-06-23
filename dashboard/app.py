"""Streamlit home hub for the AgentBench multi-game platform."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

try:
    import plotly.express as px
except Exception:  # pragma: no cover - optional UI dependency
    px = None

# Ensure project root is in PYTHONPATH
root = Path(__file__).parent.parent
if str(root) not in sys.path:
    sys.path.append(str(root))

from data.schemas import connect, init_db
from dashboard.config import db_path as configured_db_path
from dashboard.navigation import switch_page_compat
from dashboard.ui import configure_page, render_sidebar_nav, style_plotly_figure
from impostor.player_profiles import PlayerProfileManager

DEFAULT_DB = configured_db_path()
charts_available = px is not None

GAME_LIBRARY = {
    "Strategy Classics": [
        {
            "title": "RPS+",
            "icon": "⚡",
            "badge": "Strategy",
            "accent": "strategy",
            "description": "High-tempo mind games with energy management and counterplay.",
        },
        {
            "title": "Tic-Tac-Toe",
            "icon": "✖️",
            "badge": "Strategy",
            "accent": "strategy",
            "description": "Short, readable duels where every move matters immediately.",
        },
        {
            "title": "Connect Four",
            "icon": "🟡",
            "badge": "Strategy",
            "accent": "strategy",
            "description": "Clean board pressure with fast turns and satisfying traps.",
        },
        {
            "title": "Chess",
            "icon": "♟️",
            "badge": "Strategy",
            "accent": "strategy",
            "description": "The deepest ruleset in the hub, with click-to-move play.",
        },
        {
            "title": "Othello",
            "icon": "⚫",
            "badge": "Strategy",
            "accent": "strategy",
            "description": "Territory swings, corner tension, and strong momentum reversals.",
        },
        {
            "title": "Checkers",
            "icon": "🔴",
            "badge": "Strategy",
            "accent": "strategy",
            "description": "Simple entry, sharp tactics, and satisfying capture chains.",
        },
        {
            "title": "Gomoku",
            "icon": "●",
            "badge": "Strategy",
            "accent": "strategy",
            "description": "A five-in-a-row duel with quick threats and clean blocking.",
        },
        {
            "title": "Nim",
            "icon": "▦",
            "badge": "Strategy",
            "accent": "strategy",
            "description": "A compact pile-taking math duel with sharp endgames.",
        },
    ],
}

DIFFICULTY_OPTIONS = ["Easy", "Balanced", "Hardcore"]
DIFFICULTY_TO_AGENT = {
    "RPS+": {"Easy": "random", "Balanced": "adaptive_router", "Hardcore": "optimal"},
    "Tic-Tac-Toe": {"Easy": "random", "Balanced": "profile_counter", "Hardcore": "profile_counter"},
    "Connect Four": {"Easy": "random", "Balanced": "profile_counter", "Hardcore": "profile_counter"},
    "Chess": {"Easy": "random", "Balanced": "profile_counter", "Hardcore": "profile_counter"},
    "Othello": {"Easy": "random", "Balanced": "profile_counter", "Hardcore": "profile_counter"},
    "Checkers": {"Easy": "random", "Balanced": "profile_counter", "Hardcore": "profile_counter"},
    "Gomoku": {"Easy": "random", "Balanced": "profile_counter", "Hardcore": "profile_counter"},
    "Nim": {"Easy": "random", "Balanced": "profile_counter", "Hardcore": "profile_counter"},
}
GAME_META = {item["title"]: item for section in GAME_LIBRARY.values() for item in section}


_ALLOWED_TABLES = {
    "agent_results",
    "inference_benchmarks",
    "games",
    "rounds",
    "player_profiles",
    "impostor_results",
    "detection_sessions",
    "clone_ab_runs",
    "shareable_reports",
    "clone_ladder_runs",
}


def _ensure_db_initialized(db_path: Path) -> None:
    key = f"_db_initialized_{Path(db_path).resolve()}"
    if not st.session_state.get(key):
        init_db(db_path)
        st.session_state[key] = True


def _read_table(conn: sqlite3.Connection, table: str) -> pd.DataFrame:
    if table not in _ALLOWED_TABLES:
        return pd.DataFrame()
    try:
        return pd.read_sql_query(f"SELECT * FROM {table}", conn)
    except Exception:
        return pd.DataFrame()


def _list_profiles(db_path: Path):
    manager = PlayerProfileManager(db_path)
    return sorted(manager.list_all(), key=lambda profile: profile.display_name.lower())


def _set_user_profile(player_id: str, display_name: str) -> None:
    st.session_state["user_profile"] = {"id": player_id, "name": display_name}


def _clear_user_profile() -> None:
    st.session_state.pop("user_profile", None)


def _create_user_profile(db_path: Path, display_name: str) -> None:
    name = display_name.strip()
    if not name:
        st.session_state["profile_create_error"] = "Display name cannot be empty."
        return
    profile = PlayerProfileManager(db_path).create(name)
    st.session_state.pop("profile_create_error", None)
    st.session_state["user_profile"] = {"id": profile.player_id, "name": profile.display_name}


def _login_user_profile(db_path: Path, identifier: str) -> None:
    value = identifier.strip()
    if not value:
        st.session_state["profile_login_error"] = "Enter your profile name or subject ID."
        return
    profiles = _list_profiles(db_path)
    exact_id = next((profile for profile in profiles if str(profile.player_id) == value), None)
    exact_name = next((profile for profile in profiles if str(profile.display_name).casefold() == value.casefold()), None)
    partial_name = next((profile for profile in profiles if value.casefold() in str(profile.display_name).casefold()), None)
    match = exact_id or exact_name or partial_name
    if match is None:
        st.session_state["profile_login_error"] = "No matching profile found. Check the name or create a new profile."
        return
    st.session_state.pop("profile_login_error", None)
    _set_user_profile(match.player_id, match.display_name)


def _safe_charts_warning():
    if not charts_available:
        st.info("Plotly is unavailable here, so research charts fall back to tables.")


def _game_gradient(accent: str) -> str:
    if accent == "arcade":
        return "linear-gradient(135deg, rgba(251,191,36,0.90), rgba(249,115,22,0.88))"
    return "linear-gradient(135deg, rgba(59,130,246,0.92), rgba(139,92,246,0.90))"


def _difficulty_to_agent(game_title: str, difficulty: str) -> str:
    return DIFFICULTY_TO_AGENT.get(game_title, {}).get(difficulty, "random")


def _live_game_launch_settings(game_title: str, difficulty: str = "Balanced") -> dict:
    game = GAME_META[game_title]
    return {
        "game_type": game_title,
        "library_view": "Playable Now" if game["badge"] == "Strategy" else "Research Sandbox",
        "agent_name": _difficulty_to_agent(game_title, difficulty),
        "agent2_name": _difficulty_to_agent(game_title, difficulty),
        "play_mode": "human_bot1",
        "max_turns": 40,
    }


def _queue_live_game_launch(game_title: str, difficulty: str = "Balanced") -> None:
    launch = _live_game_launch_settings(game_title, difficulty)
    st.session_state["live_game_launch"] = launch
    st.session_state["live_arena_experience"] = "Train Clone Match"
    st.session_state["sidebar_game_type"] = launch["game_type"]
    st.session_state["sidebar_library_view"] = launch["library_view"]
    st.session_state["sidebar_agent_name"] = launch["agent_name"]
    st.session_state["sidebar_agent2_name"] = launch["agent2_name"]
    st.session_state["sidebar_play_mode"] = launch["play_mode"]
    st.session_state["sidebar_max_turns"] = launch["max_turns"]


def _switch_to_live_arena() -> None:
    st.session_state["live_arena_experience"] = "Train Clone Match"
    switch_page_compat("pages/live_game.py")


def _switch_to_player_profiles() -> None:
    switch_page_compat("pages/player_profile.py")


def _switch_to_arcade() -> None:
    st.session_state["live_arena_experience"] = "Play Arcade"
    switch_page_compat("pages/live_game.py")


def _user_tracking_id() -> str | None:
    profile = st.session_state.get("user_profile")
    if not profile:
        return None
    return profile.get("id")


def _fetch_recent_games(db_path: Path, player_id: str | None) -> pd.DataFrame:
    if not player_id or not db_path.exists():
        return pd.DataFrame()
    conn = connect(db_path)
    try:
        return pd.read_sql_query(
            """
            SELECT game_type, opponent_name, agent_score, opponent_score, n_turns, started_at
            FROM games
            WHERE agent_name = ?
            ORDER BY started_at DESC
            LIMIT 12
            """,
            conn,
            params=(player_id,),
        )
    except Exception:
        return pd.DataFrame()
    finally:
        conn.close()


def _continue_playing_cards(recent_games: pd.DataFrame) -> list[dict]:
    if recent_games.empty:
        return []
    cards: list[dict] = []
    seen: set[str] = set()
    for row in recent_games.itertuples(index=False):
        game_type = row.game_type or "Unknown"
        if game_type in seen or game_type not in GAME_META:
            continue
        seen.add(game_type)
        cards.append(
            {
                "title": game_type,
                "last_score": f"{row.agent_score} - {row.opponent_score}",
                "difficulty": "Balanced",
                "subtitle": f"Last vs {row.opponent_name}",
            }
        )
    return cards[:4]


def _hub_stat_values(db_path: Path, recent_games: pd.DataFrame) -> dict:
    total_games = int(len(recent_games)) if not recent_games.empty else 0
    win_rate = 0.0
    avg_fidelity = 0.0
    active_agents = 0
    has_clone_results = False
    if db_path.exists():
        conn = connect(db_path)
        try:
            games = _read_table(conn, "games")
            total_games = int(len(games)) if not games.empty else total_games
            if not games.empty and {"agent_score", "opponent_score"}.issubset(games.columns):
                win_rate = float((games["agent_score"] > games["opponent_score"]).mean())
            agents = _read_table(conn, "agent_results")
            if not agents.empty and "agent_name" in agents.columns:
                active_agents = int(agents["agent_name"].nunique())
            impostors = _read_table(conn, "impostor_results")
            if not impostors.empty:
                has_clone_results = True
                if "fidelity_score" in impostors:
                    avg_fidelity = float(impostors["fidelity_score"].fillna(0).mean())
        finally:
            conn.close()
    return {
        "total_games": total_games,
        "win_rate": win_rate,
        "avg_fidelity": avg_fidelity,
        "active_agents": active_agents,
        "has_clone_results": has_clone_results,
    }


def _load_ladder_summary(db_path: Path, player_id: str | None) -> dict | None:
    if not player_id or not db_path.exists():
        return None
    conn = connect(db_path)
    try:
        runs = pd.read_sql_query(
            """
            SELECT run_id, final_rung, cleared, completed_at
            FROM clone_ladder_runs
            WHERE player_id = ?
            ORDER BY completed_at DESC
            """,
            conn,
            params=(player_id,),
        )
    except Exception:
        return None
    finally:
        conn.close()
    if runs.empty:
        return None
    latest = runs.iloc[0]
    best_rung = int(runs["final_rung"].fillna(0).max())
    clears = int(runs["cleared"].fillna(0).sum())
    return {
        "latest_rung": int(latest.get("final_rung", 0) or 0),
        "best_rung": best_rung,
        "clears": clears,
    }


def _render_dg_stat_cards(stats: dict) -> None:
    clone_ready = bool(stats.get("has_clone_results"))
    metric_cols = st.columns(4, gap="medium")
    values = [
        ("Total Games", f"{stats['total_games']:,}", "Recorded matches"),
        ("Win Rate", f"{stats['win_rate']:.0%}" if stats["total_games"] else "No games", "Human-side archive"),
        ("Avg Fidelity", f"{stats['avg_fidelity']:.0%}" if clone_ready else "Not trained", "Clone evaluation"),
        ("Active Agents", f"{stats['active_agents']:,}" if stats["active_agents"] else "No benchmarks", "Benchmarked agents"),
    ]
    for col, (label, value, help_text) in zip(metric_cols, values):
        with col:
            st.metric(label, value, help=help_text)
    if not clone_ready:
        st.caption("Clone metrics appear after you save Train Clone Match data and run Train My Clone.")


def _render_live_feed(recent_games: pd.DataFrame) -> None:
    if recent_games.empty:
        rows = [
            ("NO TRACE", "Awaiting first captured match."),
            ("SYSTEM", "Clone lab standing by."),
            ("LIVE", "Behavioral stream initialized."),
        ]
    else:
        rows = []
        for row in recent_games.head(5).itertuples(index=False):
            result = "WIN" if row.agent_score > row.opponent_score else "LOSS" if row.agent_score < row.opponent_score else "DRAW"
            rows.append((result, f"{row.game_type} vs {row.opponent_name} · {row.agent_score}-{row.opponent_score}"))
    items = "".join(f"<div class='dg-feed-row'><strong>{tag}</strong><span>{text}</span></div>" for tag, text in rows[:5])
    st.markdown(
        f"""
        <div class="dg-feed">
            <div class="dg-panel-title"><span class="dg-live">LIVE FEED</span></div>
            {items}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_ladder_summary_card(summary: dict | None) -> None:
    if not summary:
        st.markdown(
            """
            <div class="dg-card">
                <div class="dg-card-title">
                    <div><strong>🪜 Clone Ladder</strong></div>
                    <div class="dg-badge">NO RUN</div>
                </div>
                <div class="dg-card-copy">Start a Draft Mode Clone Ladder in Live Arena to record your best rung and clears.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return
    st.markdown(
        f"""
        <div class="dg-card">
            <div class="dg-card-title">
                <div><strong>🪜 Clone Ladder</strong></div>
                <div class="dg-badge">LIVE</div>
            </div>
            <div class="dg-card-copy">Best rung: <strong>{summary['best_rung']}</strong> · Last rung: <strong>{summary['latest_rung']}</strong></div>
            <div class="dg-muted" style="margin-top:12px;">Full clears recorded: {summary['clears']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_login_screen(db_path: Path):
    st.markdown(
        """
        <div class="dg-hero">
            <div>
                <div class="dg-kicker">Doppelgamer Hub</div>
                <h1>Pick up your next match.</h1>
                <p>Select or create a profile first. Doppelgamer will not open games or research pages until your activity has somewhere to save.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        st.markdown('<div class="surface-card auth-shell">', unsafe_allow_html=True)
        tab_login, tab_signup = st.tabs(["Login", "Create Profile"])

        _ensure_db_initialized(db_path)
        manager = PlayerProfileManager(db_path)

        with tab_login:
            profiles = _list_profiles(db_path)
            if profiles:
                st.caption("Enter your profile name or subject ID")
                login_identifier = st.text_input(
                    "Profile name or ID",
                    key="login_identifier",
                    placeholder="e.g. abc or 6555db3e",
                    label_visibility="collapsed",
                )
                st.button(
                    "Enter Hub",
                    key="login_btn",
                    width='stretch',
                    on_click=_login_user_profile,
                    args=(db_path, login_identifier),
                )
                if st.session_state.get("profile_login_error"):
                    st.error(st.session_state["profile_login_error"])
                sample_names = ", ".join(profile.display_name for profile in profiles[:3])
                st.caption(f"Existing profiles: {sample_names}")
            else:
                st.info("No profiles yet. Create one to save history and personalize the bots.")

        with tab_signup:
            new_name = st.text_input("Display name")
            st.button(
                "Create Profile",
                width='stretch',
                on_click=_create_user_profile,
                args=(db_path, new_name),
            )
            if st.session_state.get("profile_create_error"):
                st.error(st.session_state["profile_create_error"])

        st.markdown("</div>", unsafe_allow_html=True)


def _preview_dialog(game_title: str, recent_games: pd.DataFrame):
    game = GAME_META[game_title]
    diff_key = f"preview_diff_{game_title}"
    default_difficulty = st.session_state.get(diff_key, "Balanced")
    st.markdown(
        f"""
        <div class="preview-hero" style="background:{_game_gradient(game['accent'])}">
            <div class="preview-badge">{game['badge']}</div>
            <div class="preview-icon">{game['icon']}</div>
            <h2>{game_title}</h2>
            <p>{game['description']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    last_score = "No previous result"
    if not recent_games.empty:
        matches = recent_games[recent_games["game_type"] == game_title]
        if not matches.empty:
            row = matches.iloc[0]
            last_score = f"{int(row['agent_score'])} - {int(row['opponent_score'])}"

    left, right = st.columns([1.2, 1], gap="large")
    with left:
        st.caption("Last score")
        st.markdown(last_score)
        st.caption("Category")
        st.markdown(game["badge"])
    with right:
        difficulty = st.segmented_control(
            "Difficulty",
            DIFFICULTY_OPTIONS,
            selection_mode="single",
            default=default_difficulty,
            key=diff_key,
        )
        difficulty = difficulty or default_difficulty
        st.caption(f"Bot routing: `{_difficulty_to_agent(game_title, difficulty)}`")

    if st.button("Play", width='stretch', type="primary", key=f"play_{game_title}"):
        _queue_live_game_launch(game_title, difficulty)
        switch_page_compat("pages/live_game.py")


if hasattr(st, "dialog"):
    _game_preview_dialog = st.dialog("Game Preview")(_preview_dialog)
else:  # pragma: no cover - fallback for older Streamlit
    def _game_preview_dialog(game_title: str, recent_games: pd.DataFrame):
        st.warning("Preview dialog is unavailable in this Streamlit version.")
        _preview_dialog(game_title, recent_games)


def _open_preview(game_title: str):
    st.session_state["hub_preview_game"] = game_title


def _render_section_header(title: str, subtitle: str):
    st.caption(subtitle)
    st.markdown(f"### {title}")


def _render_game_card(game: dict, prefix: str):
    st.markdown(
        f"""
        <div class="dg-card">
            <div class="dg-card-title">
                <div><strong>{game['icon']} {game['title']}</strong></div>
                <div class="dg-badge">CLONE TEST</div>
            </div>
            <div class="dg-card-copy">{game['description']}</div>
            <div class="dg-muted" style="margin-top:12px;">Surveillance protocol ready · capture move signature</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.button("PLAY", key=f"{prefix}_{game['title']}", width='stretch', on_click=_open_preview, args=(game["title"],))


def _render_game_row(category: str, games: list[dict]):
    _render_section_header(category, "Curated rail")
    for start in range(0, len(games), 3):
        cols = st.columns(3, gap="medium")
        for col, game in zip(cols, games[start : start + 3]):
            with col:
                _render_game_card(game, f"card_{category}")


def _render_continue_playing(cards: list[dict]):
    _render_section_header("Continue Playing", "Jump back in")
    if not cards:
        st.info("No recent games yet. Start with RPS+, Chess, or Connect Four and this rail will remember where you left off.")
        return
    cols = st.columns(min(len(cards), 3), gap="medium")
    for col, item in zip(cols, cards):
        game = GAME_META[item["title"]]
        with col:
            st.markdown(
                f"""
                <div class="dg-card">
                    <div class="dg-card-title">
                        <div><strong>{game['icon']} {item['title']}</strong></div>
                        <div class="dg-badge">{game['badge']}</div>
                    </div>
                    <div class="dg-card-copy">{item['subtitle']}</div>
                    <div class="dg-muted" style="margin-top:12px;">Last score: {item['last_score']}</div>
                    <div class="dg-pill-note" style="margin-top:12px;">Ready to resume</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("Resume", key=f"resume_{item['title']}", width='stretch'):
                _queue_live_game_launch(item["title"])
                switch_page_compat("pages/live_game.py")


def _render_research_snapshot(db_path: Path):
    if not db_path.exists():
        st.warning(f"Database not found at `{db_path}`.")
        return
    conn = connect(db_path)
    try:
        agents = _read_table(conn, "agent_results")
        inf = _read_table(conn, "inference_benchmarks")
    finally:
        conn.close()

    if agents.empty and inf.empty:
        st.info("No benchmark results yet.")
        return

    _safe_charts_warning()
    metric_cols = st.columns(4)
    total_games = int(agents["games_played"].sum()) if not agents.empty else 0
    best_win = float(agents["win_rate"].max()) if not agents.empty else 0.0
    best_name = str(agents.loc[agents["win_rate"].idxmax(), "agent_name"]) if not agents.empty else "n/a"
    avg_latency = float(inf["total_latency_ms"].mean()) if not inf.empty else 0.0
    engines = int(inf["engine"].nunique()) if not inf.empty else 0
    metrics = [
        ("Total Games", total_games),
        ("Best Win Rate", f"{best_win:.1%}" if not agents.empty else "n/a"),
        ("Avg Latency", f"{avg_latency:.1f} ms" if not inf.empty else "n/a"),
        ("Engines Tested", engines),
    ]
    for col, (label, value) in zip(metric_cols, metrics):
        with col:
            st.markdown(f"<div class='stat-card'><div class='stat-label'>{label}</div><div class='stat-value'>{value}</div></div>", unsafe_allow_html=True)
    if not agents.empty:
        st.caption(f"Current top competitive agent: {best_name}")

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("<div class='surface-card'>", unsafe_allow_html=True)
        st.subheader("Agent Win Rate")
        if charts_available:
            fig = px.bar(agents, x="agent_name", y="win_rate", color="agent_name")
            style_plotly_figure(fig, height=320)
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, width='stretch')
        else:
            st.dataframe(agents[["agent_name", "win_rate", "games_played"]], width='stretch', hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown("<div class='surface-card'>", unsafe_allow_html=True)
        st.subheader("Latency by Engine")
        if not inf.empty:
            agg = inf.groupby("engine", as_index=False)["total_latency_ms"].mean()
            if charts_available:
                fig = px.bar(agg, x="engine", y="total_latency_ms", color="engine")
                style_plotly_figure(fig, height=320)
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, width='stretch')
            else:
                st.dataframe(agg, width='stretch', hide_index=True)
        else:
            st.info("No inference benchmarks yet.")
        st.markdown("</div>", unsafe_allow_html=True)


def _render_home_hub(db_path: Path):
    user_name = st.session_state["user_profile"]["name"]
    tracking_id = _user_tracking_id()
    recent_games = _fetch_recent_games(db_path, tracking_id)
    continue_cards = _continue_playing_cards(recent_games)
    stats = _hub_stat_values(db_path, recent_games)
    ladder_summary = _load_ladder_summary(db_path, tracking_id)

    st.markdown(
        f"""
        <div class="dg-hero">
            <div>
                <div class="dg-kicker"><span class="dg-live">Surveillance Lab Online</span></div>
                <h1>👾 DOPPELGAMER</h1>
                <p>Can you detect your own clone? Welcome back, {user_name}. Every move becomes evidence in the behavioral impersonation lab.</p>
                <div style="margin-top:16px;" class="dg-pill-note">Subject {tracking_id}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _render_dg_stat_cards(stats)
    st.divider()

    st.caption("Research Snapshot")
    ladder_col, note_col = st.columns([1.1, 2.2], gap="medium")
    with ladder_col:
        _render_ladder_summary_card(ladder_summary)
    with note_col:
        st.markdown(
            """
            <div class="dg-card">
                <div class="dg-card-title">
                    <div><strong>Paper Battery Status</strong></div>
                    <div class="dg-badge">READY</div>
                </div>
                <div class="dg-card-copy">Use Train My Clone to run the canonical baseline battery, session-ordered generalization sweep, and clone-vs-baseline evaluation flow from one profile.</div>
                <div class="dg-muted" style="margin-top:12px;">The hub now tracks ladder progress, clone metrics, and Turing-test ops as one research loop.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.divider()

    quick_cols = st.columns(4, gap="medium")
    if continue_cards:
        with quick_cols[0]:
            if st.button(
                f"Resume {continue_cards[0]['title']}",
                width='stretch',
                type="primary",
                key="hero_resume_latest",
            ):
                _queue_live_game_launch(continue_cards[0]["title"])
                switch_page_compat("pages/live_game.py")
            st.caption("Continue collecting clone-training data.")
    else:
        with quick_cols[0]:
            if st.button(
                "Train Clone Match",
                width='stretch',
                type="primary",
                key="hero_start_connect_four",
            ):
                _queue_live_game_launch("RPS+")
                switch_page_compat("pages/live_game.py")
            st.caption("Saved matches for clone training.")
    with quick_cols[1]:
        if st.button(
            "Play Arcade",
            width='stretch',
            key="hero_arcade",
        ):
            _switch_to_arcade()
        st.caption("Polished browser play. No training data saved.")
    with quick_cols[2]:
        if st.button(
            "Train My Clone",
            width='stretch',
            key="hero_profiles",
        ):
            _switch_to_player_profiles()
        st.caption("Open your profile to train/evaluate clone models.")
    with quick_cols[3]:
        if st.button(
            "Clone Ops",
            width='stretch',
            key="hero_clone_ops",
        ):
            switch_page_compat("pages/impostor_leaderboard.py")
        st.caption("Inspect fool rate, A/B blocks, and Turing-test ops.")

    st.divider()
    main_col, feed_col = st.columns([3, 1], gap="large")
    with main_col:
        _render_continue_playing(continue_cards)
        st.divider()
        _render_game_row("Strategy Classics", GAME_LIBRARY["Strategy Classics"])
    with feed_col:
        _render_live_feed(recent_games)

    preview_game = st.session_state.pop("hub_preview_game", None)
    if preview_game:
        _game_preview_dialog(preview_game, recent_games)


def run_dashboard(db_path: str = str(DEFAULT_DB)) -> None:
    configure_page("Doppelgamer | Hub")

    db_file = Path(db_path)
    _ensure_db_initialized(db_file)
    if "user_profile" not in st.session_state:
        _render_login_screen(db_file)
        return

    render_sidebar_nav("Hub")
    st.sidebar.divider()
    if st.sidebar.button("Logout / Switch Profile", width='stretch'):
        _clear_user_profile()
        st.rerun()

    st.title("Doppelgamer")
    top_left, top_right = st.columns([1.2, 1], gap="medium")
    with top_left:
        st.markdown(
            """
            <div class="dg-topbar">
                <div class="dg-brand">
                    <div class="dg-brand-mark">👾</div>
                    <div>
                        <div><strong>DOPPELGAMER</strong></div>
                        <div class="dg-muted">Behavioral cloning research hub</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with top_right:
        st.markdown("<div class='dg-topbar'>", unsafe_allow_html=True)
        nav_cols = st.columns(3, gap="small")
        nav_cols[0].button("Hub", width='stretch', disabled=True)
        if nav_cols[1].button("Live Arena", width='stretch', key="topnav_live"):
            _switch_to_live_arena()
        if nav_cols[2].button("Profiles", width='stretch', key="topnav_profile"):
            _switch_to_player_profiles()
        st.markdown("</div>", unsafe_allow_html=True)

    st.divider()
    _render_home_hub(db_file)
    st.divider()
    _render_research_snapshot(db_file)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", default=str(DEFAULT_DB))
    args = parser.parse_args()
    run_dashboard(args.db_path)
