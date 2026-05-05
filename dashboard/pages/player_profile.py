import streamlit as st
import sys
import datetime as dt
from pathlib import Path
import sqlite3
import json
import uuid
import numpy as np
import pandas as pd
try:
    import plotly.express as px
except ImportError:  # Optional in test/minimal environments
    px = None
try:
    import plotly.graph_objects as go
except ImportError:  # Optional in test/minimal environments
    go = None

# Ensure project root is in PYTHONPATH
root = Path(__file__).parent.parent.parent
if str(root) not in sys.path:
    sys.path.append(str(root))

# Imports already moved to top

from data.backfill import canonical_game_type
from data.collector import insert_game
from data.features import (
    behavioral_timeline,
    canonical_figure_frames,
    data_sufficiency_status,
    failure_case_gallery,
    load_clone_results,
    player_dataset_card,
    per_game_fingerprint,
    slice_registry_frame,
    session_generalization_frame,
    top_move_name,
)
from data.schemas import connect, init_db, init_extended_db
from dashboard.auth import require_user_profile
from dashboard.config import db_path as configured_db_path
from dashboard.ui import configure_page, render_empty_state, render_sidebar_nav, style_plotly_figure
from environments.rps_plus import Move, RPSPlusEnv
from environments.utils import history_to_records
from impostor.player_profiles import PlayerProfileManager
from environments.rps_plus import N_MOVES
from impostor.experiments import (
    experiment_1_fidelity_comparison,
    experiment_bias_interventions,
    experiment_controlled_rearing,
    experiment_session_ordered_generalization,
    run_paper_battery,
)
from agents.impostor.trainer import ImpostorTrainer
from evaluation.runner import benchmark_clone_variants, run_clone_human_ab_evaluation
from impostor.metrics import behavioral_drift, build_clone_report, narrative_summary, paired_significance, retraining_trigger, susceptibility_score

DEFAULT_DB = configured_db_path()


def _reset_legacy_game_session() -> None:
    for key in [
        "game_env", "game_saved", "game_done",
        "game_max_turns", "game_profile_id", "game_game_type"
    ]:
        st.session_state.pop(key, None)
MOVE_NAMES = [m.name for m in Move]


def _style_figure(fig, *, show_legend: bool = True):
    if fig is None:
        return fig
    style_plotly_figure(fig, height=300)
    fig.update_layout(
        font=dict(color="#F8FAFC"),
        title_font=dict(color="#F8FAFC"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#F8FAFC")),
        xaxis=dict(color="#F8FAFC", gridcolor="rgba(255,255,255,0.08)", zerolinecolor="rgba(255,255,255,0.08)"),
        yaxis=dict(color="#F8FAFC", gridcolor="rgba(255,255,255,0.08)", zerolinecolor="rgba(255,255,255,0.08)"),
        showlegend=show_legend,
    )
    return fig


def _initials(name: str) -> str:
    parts = [part for part in str(name).replace("_", " ").split() if part]
    if not parts:
        return "??"
    return "".join(part[0] for part in parts[:2]).upper()


def _render_signature_header(display_name: str, player_id: str) -> None:
    st.markdown(
        f"""
        <div class="dg-profile-header">
            <div class="dg-avatar">{_initials(display_name)}</div>
            <div>
                <div class="dg-live">BEHAVIORAL SIGNATURE REPORT</div>
                <div style="margin:4px 0; font-size:30px; font-weight:700; color:var(--dg-text);">{display_name}</div>
                <p>Subject ID: <code>{player_id}</code> · clone fidelity monitoring enabled</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_sufficiency_badge(sufficiency: dict) -> None:
    color = {
        "Too little data": "#ff6b35",
        "Trainable": "#ffd166",
        "Paper-ready": "#00ff9d",
    }.get(sufficiency.get("label"), "#7f77dd")
    st.markdown(
        f"""
        <div style="
            display:inline-flex;
            align-items:center;
            gap:10px;
            margin-top:12px;
            padding:10px 14px;
            border-radius:999px;
            border:1px solid {color}55;
            background: {color}14;
            color:{color};
            font-family:var(--dg-mono);
            font-size:13px;
            font-weight:700;
        ">
            <span>{sufficiency.get('label', 'Unknown')}</span>
            <span style="color:var(--dg-text-soft); font-weight:500;">
                {sufficiency.get('sessions', 0)} sessions · {sufficiency.get('rounds', 0)} rounds
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _plot_kl_gauge(value: float):
    if go is None:
        return None
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"font": {"color": "#00ff9d", "family": "SFMono-Regular, monospace"}},
            title={"text": "KL Divergence", "font": {"color": "#f4fff9"}},
            gauge={
                "axis": {"range": [0, 1], "tickcolor": "#8fa7a0"},
                "bar": {"color": "#00ff9d"},
                "bgcolor": "rgba(255,255,255,0.04)",
                "bordercolor": "rgba(0,255,157,0.2)",
                "steps": [
                    {"range": [0, 0.3], "color": "rgba(0,255,157,0.18)"},
                    {"range": [0.3, 1], "color": "rgba(255,107,53,0.28)"},
                ],
                "threshold": {"line": {"color": "#ff6b35", "width": 4}, "value": 0.3},
            },
        )
    )
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", height=260, margin=dict(l=20, r=20, t=40, b=20))
    return fig


def _plot_win_sparkline(sessions_df: pd.DataFrame):
    if go is None or sessions_df.empty:
        return None
    wins = (sessions_df.sort_values("started_at")["result"] == "Win").astype(int)
    rolling = wins.expanding().mean()
    fig = go.Figure(go.Scatter(y=rolling, mode="lines+markers", line=dict(color="#7f77dd", width=3), marker=dict(color="#00ff9d", size=6)))
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=220, margin=dict(l=20, r=20, t=24, b=20), yaxis=dict(range=[0, 1], tickformat=".0%"))
    return fig


def _plot_clone_learning_narrative(
    timeline_df: pd.DataFrame,
    drift_points,
    fooled_count: int,
):
    if go is None or timeline_df.empty:
        return None
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=timeline_df["started_at"],
            y=timeline_df["cum_win_rate"],
            mode="lines+markers",
            name="Win rate",
            line=dict(color="#7f77dd", width=3),
        )
    )
    if "fidelity_score" in timeline_df.columns and not timeline_df["fidelity_score"].dropna().empty:
        fig.add_trace(
            go.Scatter(
                x=timeline_df["started_at"],
                y=timeline_df["fidelity_score"],
                mode="lines+markers",
                name="Clone fidelity",
                line=dict(color="#00ff9d", width=3),
            )
        )
    drift_df = pd.DataFrame([point.__dict__ for point in drift_points])
    if not drift_df.empty:
        fig.add_trace(
            go.Scatter(
                x=drift_df["timestamp"],
                y=drift_df["drift_score"],
                mode="lines+markers",
                name="Drift",
                yaxis="y2",
                line=dict(color="#ff6b35", width=2, dash="dot"),
            )
        )
    fig.update_layout(
        yaxis2=dict(
            title="Drift",
            overlaying="y",
            side="right",
            range=[0, max(1.0, float(drift_df["drift_score"].max()) if not drift_df.empty else 1.0)],
            showgrid=False,
            color="#ff6b35",
        ),
        annotations=[
            dict(
                xref="paper",
                yref="paper",
                x=0.99,
                y=1.14,
                showarrow=False,
                text=f"Foiled detections: {fooled_count}",
                font=dict(color="#F8FAFC"),
            )
        ],
        title="The Clone Learns You",
    )
    return _style_figure(fig)


def _render_bias_scan(moves_df: pd.DataFrame, sessions_df: pd.DataFrame) -> None:
    recency = min(1.0, len(moves_df.tail(20)) / 20) if not moves_df.empty else 0.0
    win_streak = 0
    if not sessions_df.empty:
        for result in sessions_df.sort_values("started_at", ascending=False)["result"]:
            if result == "Win":
                win_streak += 1
            else:
                break
    aggression = min(1.0, win_streak / 5)
    loss_rate = float((sessions_df["result"] == "Loss").mean()) if not sessions_df.empty else 0.0
    items = [
        ("Recency Bias", recency),
        ("Win-Streak Aggression", aggression),
        ("Loss Aversion", loss_rate),
    ]
    bars = "".join(
        f"""
        <div class="dg-bias-row">
            <div><strong>{name}</strong><span>{value:.0%}</span></div>
            <i><b style="width:{value * 100:.0f}%"></b></i>
        </div>
        """
        for name, value in items
    )
    st.markdown(f"<div class='dg-card dg-bias'><h3>COGNITIVE BIAS SCAN</h3>{bars}</div>", unsafe_allow_html=True)


def _get_game_winner(env, game_type: str) -> str:
    if game_type == "RPS+":
        if env.state.agent_score > env.state.opponent_score:
            return "player"
        elif env.state.agent_score < env.state.opponent_score:
            return "ai"
        else:
            return "tie"
    elif game_type in ("Tic-Tac-Toe", "Connect Four"):
        if env.state.agent_score > env.state.opponent_score:
            return "player"
        elif env.state.agent_score < env.state.opponent_score:
            return "ai"
        else:
            return "tie"
    elif game_type == "Chess":
        if env._board.is_checkmate():
            return "ai" if env._board.turn else "player"  # AI is black
        elif env._board.is_stalemate():
            return "tie"
        else:
            return "tie"  # For now
    return "tie"


def _format_chess_move(move, board) -> str:
    try:
        import chess
        return board.san(move)
    except Exception:
        return move.uci()


def _load_player_moves(conn: sqlite3.Connection, player_id: str) -> pd.DataFrame:
    try:
        return pd.read_sql_query(
            """SELECT r.agent_move, r.agent_move_name, r.outcome, r.turn, g.game_id
               FROM rounds r JOIN games g USING (game_id)
               WHERE g.agent_name = ?
               ORDER BY g.started_at, r.turn""",
            conn,
            params=(player_id,),
        )
    except Exception:
        return pd.DataFrame()


def _load_profiles(conn: sqlite3.Connection) -> pd.DataFrame:
    try:
        return pd.read_sql_query(
            "SELECT player_id, display_name, games_played, total_rounds, win_rate FROM player_profiles",
            conn,
        )
    except Exception:
        return pd.DataFrame()


def _load_impostor_results(conn: sqlite3.Connection, player_id: str) -> pd.DataFrame:
    try:
        return pd.read_sql_query(
            "SELECT impostor_type, game_type, fidelity_score, kl_divergence, fool_rate, explanation_sample, n_training_rounds "
            "FROM impostor_results WHERE player_id = ? ORDER BY fidelity_score DESC",
            conn,
            params=(player_id,),
        )
    except Exception:
        return pd.DataFrame()


def _load_ab_runs(conn: sqlite3.Connection, player_id: str) -> pd.DataFrame:
    try:
        return pd.read_sql_query(
            """SELECT run_id, block_id, game_type, clone_type, baseline_type, n_games,
                      clone_wins, baseline_wins, draws, clone_fidelity, detection_rate, created_at
               FROM clone_ab_runs
               WHERE player_id = ?
               ORDER BY created_at DESC""",
            conn,
            params=(player_id,),
        )
    except Exception:
        return pd.DataFrame()


def _persist_shareable_report(conn: sqlite3.Connection, player_id: str, report: dict) -> str:
    report_id = uuid.uuid4().hex[:12]
    conn.execute(
        """INSERT INTO shareable_reports (report_id, player_id, report_json, created_at)
           VALUES (?, ?, ?, ?)""",
        (
            report_id,
            player_id,
            json.dumps(report),
            dt.datetime.now(dt.timezone.utc).isoformat(),
        ),
    )
    conn.commit()
    return report_id


def _load_rps_training_sequences(conn: sqlite3.Connection, player_id: str) -> list[list[int]]:
    try:
        rows = pd.read_sql_query(
            """SELECT g.game_id, r.turn, r.agent_move
               FROM rounds r JOIN games g USING (game_id)
               WHERE g.agent_name = ? AND g.game_type = 'RPS+'
               ORDER BY g.started_at, g.game_id, r.turn""",
            conn,
            params=(player_id,),
        )
    except Exception:
        return []
    if rows.empty:
        return []
    sequences: list[list[int]] = []
    for _, group in rows.groupby("game_id", sort=False):
        seq = [
            int(move)
            for move in group["agent_move"].tolist()
            if pd.notna(move) and 0 <= int(move) < N_MOVES
        ]
        if seq:
            sequences.append(seq)
    return sequences


def _train_clone_for_player(db_path: Path, player_id: str) -> tuple[int, int]:
    init_db(db_path)
    init_extended_db(db_path)
    conn = connect(db_path)
    try:
        sequences = _load_rps_training_sequences(conn, player_id)
        training_rounds = sum(len(seq) for seq in sequences)
        if len(sequences) < 2 or training_rounds < 10:
            return 0, training_rounds

        report = experiment_1_fidelity_comparison(sequences, player_id=player_id, rounds=min(50, max(20, training_rounds)))
        run_id = uuid.uuid4().hex
        trained_at = dt.datetime.now(dt.timezone.utc).isoformat()
        conn.executemany(
            """INSERT OR REPLACE INTO impostor_results
               (run_id, player_id, impostor_type, game_type, n_training_rounds, fidelity_score,
                kl_divergence, tvd, fool_rate, explanation_sample, embedding_json, trained_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    run_id,
                    player_id,
                    row["agent_type"],
                    "RPS+",
                    training_rounds,
                    float(row["fidelity_score"]),
                    float(row["kl_divergence"]),
                    float(row["tvd"]),
                    None,
                    None,
                    None,
                    trained_at,
                )
                for row in report.results
            ],
        )
        conn.commit()
        return len(report.results), training_rounds
    finally:
        conn.close()


def _load_game_sessions(conn: sqlite3.Connection) -> pd.DataFrame:
    try:
        sessions = pd.read_sql_query(
            """SELECT
                   g.game_id,
                   g.agent_name AS player_id,
                   COALESCE(pp.display_name, g.agent_name) AS player_name,
                   g.opponent_name AS bot_name,
                   COALESCE(NULLIF(g.game_type, ''), 'Unknown / Legacy') AS game_type,
                   g.started_at,
                   g.n_turns,
                   g.agent_score,
                   g.opponent_score
               FROM games g
               LEFT JOIN player_profiles pp
                 ON pp.player_id = g.agent_name
               ORDER BY g.started_at DESC""",
            conn,
        )
    except Exception:
        return pd.DataFrame()

    if sessions.empty:
        return sessions

    sessions = sessions.copy()
    sessions["game_type"] = sessions["game_type"].apply(canonical_game_type)
    sessions["result"] = np.where(
        sessions["agent_score"] > sessions["opponent_score"],
        "Win",
        np.where(sessions["agent_score"] < sessions["opponent_score"], "Loss", "Tie"),
    )
    return sessions


def _summarize_session_groups(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    summary = (
        df.groupby(group_cols, dropna=False)
        .agg(
            games=("game_id", "nunique"),
            wins=("result", lambda s: int((s == "Win").sum())),
            losses=("result", lambda s: int((s == "Loss").sum())),
            ties=("result", lambda s: int((s == "Tie").sum())),
            avg_turns=("n_turns", "mean"),
            avg_player_score=("agent_score", "mean"),
            avg_bot_score=("opponent_score", "mean"),
        )
        .reset_index()
    )
    summary["win_rate"] = summary["wins"] / summary["games"].clip(lower=1)
    return summary.sort_values(["games", "win_rate"], ascending=[False, False])


def _filter_sessions(
    sessions_df: pd.DataFrame,
    players: list[str] | None = None,
    bots: list[str] | None = None,
    games: list[str] | None = None,
) -> pd.DataFrame:
    if sessions_df.empty:
        return sessions_df

    filtered = sessions_df.copy()
    if players:
        filtered = filtered[filtered["player_id"].astype(str).isin(players)]
    if bots:
        filtered = filtered[filtered["bot_name"].astype(str).isin(bots)]
    if games:
        filtered = filtered[filtered["game_type"].astype(str).isin(games)]
    return filtered


def _save_played_game(db_path: Path, player_id: str, env: RPSPlusEnv, manager: PlayerProfileManager) -> None:
    if not hasattr(env, "state") or not hasattr(env.state, "history"):
        return

    def _safe_int(value, default: int = -1) -> int:
        try:
            return int(value)
        except Exception:
            return default

    def _safe_name(value: object, fallback: str) -> str:
        if hasattr(value, "name"):
            return str(value.name)
        return fallback

    def _history_to_generic_records(history, game_id: str) -> list[dict]:
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

    conn = connect(str(db_path))
    try:
        game_id = uuid.uuid4().hex
        if isinstance(env, RPSPlusEnv):
            records = history_to_records(env.state.history, game_id)
        else:
            records = _history_to_generic_records(env.state.history, game_id)

        insert_game(
            conn,
            game_id=game_id,
            agent_name=player_id,
            opponent_name="scripted",
            game_type=canonical_game_type(getattr(env, "name", env.__class__.__name__)),
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
        if valid_records:
            manager.update_signature(
                player_id,
                [[_safe_int(r["agent_move"]) for r in valid_records]],
                [[_safe_int(r["outcome"], 0) for r in valid_records]],
                [[_safe_int(r["opponent_move"]) for r in valid_records]],
            )
    finally:
        conn.close()


def _render_profile_tab(conn: sqlite3.Connection, manager: PlayerProfileManager) -> tuple[str, str]:
    profiles_df = _load_profiles(conn)
    sessions_df = _load_game_sessions(conn)

    if profiles_df.empty:
        render_empty_state(
            "🧬",
            "No player profiles yet",
            "Create a profile first, then collect gameplay data so the dashboard has a behavioral history to analyze.",
        )
        with st.form("create_profile_form"):
            display_name = st.text_input("Player display name", "")
            submitted = st.form_submit_button("Create Profile")

        if submitted:
            display_name = display_name.strip()
            if not display_name:
                st.error("Please enter a display name to create a profile.")
            else:
                profile = manager.create(display_name)
                st.success(f"Created profile **{profile.display_name}** with ID `{profile.player_id}`.")
                st.info("Next: open Live Arena, play a Train Clone Match, and then return here to review the profile.")
        return "", ""

    player_options = [f"{name} ({pid})" for name, pid in zip(profiles_df["display_name"], profiles_df["player_id"])]
    player_map = dict(zip(player_options, profiles_df["player_id"]))
    selected_option = st.selectbox("Select player", player_options)
    player_id = player_map[selected_option]
    row = profiles_df[profiles_df["player_id"] == player_id].iloc[0]
    display_name = row["display_name"]
    player_sessions = sessions_df[sessions_df["player_id"].astype(str) == str(player_id)].copy() if not sessions_df.empty else pd.DataFrame()
    imp_df = _load_impostor_results(conn, player_id)
    sufficiency = data_sufficiency_status(DEFAULT_DB, player_id)
    dataset_card = player_dataset_card(DEFAULT_DB, player_id)
    moves_df = _load_player_moves(conn, player_id)
    kl_value = float(imp_df["kl_divergence"].dropna().iloc[0]) if not imp_df.empty and "kl_divergence" in imp_df and not imp_df["kl_divergence"].dropna().empty else 0.0
    fidelity = float(imp_df["fidelity_score"].dropna().mean()) if not imp_df.empty and "fidelity_score" in imp_df and not imp_df["fidelity_score"].dropna().empty else 0.0

    head_left, head_right = st.columns([1, 3], gap="medium")
    with head_left:
        st.markdown(f'<div class="dg-avatar-circle">{_initials(display_name)}</div>', unsafe_allow_html=True)
    with head_right:
        _render_signature_header(display_name, player_id)
        stat_cols = st.columns(3, gap="medium")
        stat_cols[0].metric("Games Played", int(row["games_played"]))
        stat_cols[1].metric("Win Rate", f"{row['win_rate']:.1%}")
        stat_cols[2].metric("Saved Rounds", int(row.get("total_rounds", 0)))
        _render_sufficiency_badge(sufficiency)

    dataset_cols = st.columns(5, gap="medium")
    dataset_cols[0].metric("Rounds", int(dataset_card.get("rounds", 0)))
    dataset_cols[1].metric("Sessions", int(dataset_card.get("sessions", 0)))
    dataset_cols[2].metric("Game Coverage", int(dataset_card.get("game_coverage", 0)))
    dataset_cols[3].metric("Drift Coverage", f"{float(dataset_card.get('drift_coverage', 0.0)):.0%}")
    dataset_cols[4].metric("Trainability", str(dataset_card.get("trainability", "Unknown")))

    st.divider()
    st.caption("Clone Training")
    if imp_df.empty:
        st.info("No trained clone yet. Train one from your saved RPS+ profile data to populate fidelity, KL divergence, and active clone metrics.")
    else:
        best_fidelity = float(imp_df["fidelity_score"].max()) if "fidelity_score" in imp_df else 0.0
        st.success(f"Clone evaluations available. Best fidelity: {best_fidelity:.0%}.")
    train_col, note_col = st.columns([0.32, 0.68])
    with train_col:
        if st.button("Train My Clone", width='stretch', type="primary"):
            with st.spinner("Training and evaluating NGram/LSTM clone candidates..."):
                n_models, n_rounds = _train_clone_for_player(DEFAULT_DB, player_id)
            if n_models:
                st.success(f"Trained/evaluated {n_models} clone models on {n_rounds} RPS+ rounds.")
                st.rerun()
            else:
                st.warning(f"Need more saved RPS+ data before training. Current usable rounds: {n_rounds}.")
    with note_col:
        st.caption("Train Clone uses saved Live Arena RPS+ games. Play Arcade is visual-only and does not write training data.")
    action_cols = st.columns(3, gap="medium")
    with action_cols[0]:
        if st.button("Online Adapt Clone", width='stretch'):
            trainer = ImpostorTrainer(DEFAULT_DB)
            result = trainer.online_update(player_id)
            if result.get("updated"):
                st.success(f"Applied online clone adaptation from the latest {result['game_type']} session.")
            else:
                st.warning("No recent data available for online adaptation yet.")
    with action_cols[1]:
        if st.button("Run Clone vs Baseline A/B", width='stretch'):
            if imp_df.empty:
                st.warning("Train at least one clone first so the A/B block has a clone to evaluate.")
            else:
                clone_row = imp_df.sort_values("fidelity_score", ascending=False).iloc[0]
                run_clone_human_ab_evaluation(
                    db_path=DEFAULT_DB,
                    player_id=player_id,
                    game_type=str(clone_row.get("game_type", "RPS+")),
                    clone_type=str(clone_row["impostor_type"]),
                    baseline_type="heuristic",
                    clone_scores=[1 if float(clone_row.get("fidelity_score", 0)) >= 0.5 else 0],
                    baseline_scores=[0],
                    clone_fidelity=float(clone_row.get("fidelity_score", 0) or 0.0),
                    detection_rate=float(clone_row.get("fool_rate", 0) or 0.0),
                )
                st.success("Logged a clone-vs-baseline A/B evaluation block.")
    with action_cols[2]:
        if st.button("Generate Clone Report", width='stretch'):
            clone_rows = imp_df.to_dict("records") if not imp_df.empty else []
            report = build_clone_report(
                player_id,
                clone_rows,
                [move for move in [top_move_name(moves_df)] if move],
            )
            report_id = _persist_shareable_report(conn, player_id, report.__dict__)
            st.success(f"Saved shareable clone report `{report_id}`.")
    utility_cols = st.columns(2, gap="medium")
    with utility_cols[0]:
        if st.button("Run Paper Battery", width='stretch'):
            with st.spinner("Running the paper experiment battery..."):
                reports = run_paper_battery(str(DEFAULT_DB), player_id)
            experiment_names = ", ".join(report.experiment for report in reports.values())
            st.success(f"Paper battery completed with {len(reports)} experiment outputs.")
            st.caption(f"Runs completed: {experiment_names}")
    with utility_cols[1]:
        if st.button("Recency-Weighted Adapt", width='stretch'):
            trainer = ImpostorTrainer(DEFAULT_DB)
            result = trainer.recency_weighted_online_update(player_id)
            if result.get("updated"):
                st.success(f"Applied recency-weighted adaptation with decay {result['decay']}.")
            else:
                st.warning("Need more session history for recency-weighted adaptation.")

    st.divider()
    st.caption("Data & Consent")
    with st.expander("What Doppelgamer stores and how to reset it", expanded=False):
        st.markdown(
            """
            - **Train Clone Match** saves completed matches, turn-by-turn moves, behavioral summaries, and clone-study results to your local research database.
            - **Play Arcade** is for visual play only and does not feed clone training.
            - **Blind Turing Study** hides opponent identity until the block ends and records your detection judgment as research data.
            - You can clear either clone outputs only or all saved gameplay for this profile below.
            """
        )
        consent_cols = st.columns(2, gap="medium")
        with consent_cols[0]:
            reset_clone_ok = st.checkbox(
                "I understand this will delete clone results, reports, A/B blocks, and detection sessions for this profile.",
                key=f"confirm_reset_clone_{player_id}",
            )
            if st.button("Reset Clone Outputs", width='stretch', disabled=not reset_clone_ok, key=f"reset_clone_outputs_{player_id}"):
                deleted = manager.clear_clone_artifacts(player_id)
                total_deleted = sum(int(value) for value in deleted.values())
                st.success(f"Cleared clone artifacts for {display_name}. Removed {total_deleted} stored research rows.")
                st.rerun()
        with consent_cols[1]:
            delete_games_ok = st.checkbox(
                "I understand this will delete saved gameplay, move history, ladder runs, and reset this profile's behavioral signature.",
                key=f"confirm_reset_games_{player_id}",
            )
            if st.button("Delete Saved Gameplay", width='stretch', disabled=not delete_games_ok, key=f"reset_saved_games_{player_id}"):
                deleted = manager.clear_gameplay_data(player_id)
                total_deleted = sum(int(value) for value in deleted.values())
                st.success(f"Deleted saved gameplay for {display_name}. Removed {total_deleted} stored rows and reset the profile summary.")
                st.rerun()

    ab_df = _load_ab_runs(conn, player_id)
    timeline_df = behavioral_timeline(DEFAULT_DB, player_id)
    fingerprint_df = per_game_fingerprint(DEFAULT_DB, player_id)
    generalization_df = session_generalization_frame(DEFAULT_DB, player_id)
    failure_df = failure_case_gallery(DEFAULT_DB, player_id)
    figure_frames = canonical_figure_frames(DEFAULT_DB, player_id)
    snapshots = manager.load_behavioral_snapshots(player_id)
    drift_points = behavioral_drift(snapshots)
    retrain = retraining_trigger(snapshots)
    fooled_count = int((imp_df["fool_rate"].fillna(0.0) > 0.5).sum()) if not imp_df.empty and "fool_rate" in imp_df else 0
    if retrain.needs_retraining:
        st.warning(f"Retraining recommended: {retrain.rationale}")
    else:
        st.caption(f"Retraining status: {retrain.rationale}")
    st.divider()
    chart_l, chart_r = st.columns(2, gap="large")
    with chart_l:
        st.caption("KL Divergence Gauge")
        gauge = _plot_kl_gauge(kl_value)
        if gauge is not None:
            st.plotly_chart(gauge, width='stretch')
        else:
            render_empty_state("📈", "No divergence chart", "Plotly is unavailable in this environment.")
    with chart_r:
        st.caption("Win Rate Over Time")
        spark = _plot_win_sparkline(player_sessions)
        if spark is not None:
            st.plotly_chart(spark, width='stretch')
        else:
            render_empty_state("📉", "No data yet", "Play saved Live Arena matches to build a win-rate timeline.")
    if not timeline_df.empty and px is not None:
        st.divider()
        time_l, time_r = st.columns(2, gap="large")
        with time_l:
            st.caption("Behavioral Timeline")
            fig_time = px.line(
                timeline_df,
                x="started_at",
                y=["cum_win_rate", "fidelity_score"],
                title="Win rate and clone fidelity over time",
            )
            _style_figure(fig_time)
            st.plotly_chart(fig_time, width='stretch')
        with time_r:
            st.caption("Behavioral Drift")
            drift_df = pd.DataFrame([point.__dict__ for point in drift_points])
            if not drift_df.empty:
                fig_drift = px.line(
                    drift_df,
                    x="timestamp",
                    y="drift_score",
                    color="game_type",
                    title="Signature drift by session snapshot",
                )
                _style_figure(fig_drift)
                st.plotly_chart(fig_drift, width='stretch')
            else:
                render_empty_state("🧭", "No drift snapshots yet", "Save a few profile updates to start seeing behavioral drift.")
    if generalization_df is not None and not generalization_df.empty:
        st.caption("Session-Ordered Generalization Sweep")
        st.dataframe(generalization_df, width='stretch', hide_index=True)
        gen_report = experiment_session_ordered_generalization(str(DEFAULT_DB), player_id)
        if gen_report.results:
            st.caption("Held-out later-session evaluation")
            st.dataframe(pd.DataFrame(gen_report.results), width='stretch', hide_index=True)

    st.divider()
    st.caption("Controlled Rearing and Slice Registry")
    slice_df = slice_registry_frame(DEFAULT_DB, player_id)
    if not slice_df.empty:
        st.dataframe(slice_df, width='stretch', hide_index=True)
    rearing_cols = st.columns(3, gap="medium")
    with rearing_cols[0]:
        if st.button("Run Controlled Rearing", width='stretch'):
            report = experiment_controlled_rearing(str(DEFAULT_DB), player_id)
            st.session_state["last_controlled_rearing"] = pd.DataFrame(report.results)
            st.success(report.summary)
    with rearing_cols[1]:
        if st.button("Run Bias Interventions", width='stretch'):
            report = experiment_bias_interventions(str(DEFAULT_DB), player_id)
            st.session_state["last_bias_interventions"] = pd.DataFrame(report.results)
            st.success(report.summary)
    with rearing_cols[2]:
        if st.button("Benchmark Clone Variants", width='stretch'):
            rows = benchmark_clone_variants(player_id=player_id, db_path=str(DEFAULT_DB))
            st.session_state["last_clone_benchmark"] = pd.DataFrame(rows)
            st.success(f"Benchmarked {len(rows)} clone variants for latency/fidelity tradeoffs.")
    if isinstance(st.session_state.get("last_controlled_rearing"), pd.DataFrame) and not st.session_state["last_controlled_rearing"].empty:
        st.caption("Controlled Rearing Results")
        st.dataframe(st.session_state["last_controlled_rearing"], width='stretch', hide_index=True)
    if isinstance(st.session_state.get("last_bias_interventions"), pd.DataFrame) and not st.session_state["last_bias_interventions"].empty:
        st.caption("Bias Intervention Results")
        st.dataframe(st.session_state["last_bias_interventions"], width='stretch', hide_index=True)
    if isinstance(st.session_state.get("last_clone_benchmark"), pd.DataFrame) and not st.session_state["last_clone_benchmark"].empty:
        st.caption("Clone Systems Benchmark")
        st.dataframe(st.session_state["last_clone_benchmark"], width='stretch', hide_index=True)

    st.divider()
    st.caption("Cognitive Bias Scan")
    bias_cols = st.columns(3, gap="medium")
    recency = min(1.0, len(moves_df.tail(20)) / 20) if not moves_df.empty else 0.0
    win_streak = 0
    if not player_sessions.empty:
        for result in player_sessions.sort_values("started_at", ascending=False)["result"]:
            if result == "Win":
                win_streak += 1
            else:
                break
    aggression = min(1.0, win_streak / 5)
    loss_rate = float((player_sessions["result"] == "Loss").mean()) if not player_sessions.empty else 0.0
    for col, (label, value) in zip(
        bias_cols,
        [("Recency Bias", recency), ("Win-Streak Aggression", aggression), ("Loss Aversion", loss_rate)],
    ):
        with col:
            st.metric(label, f"{value:.0%}")
            st.progress(int(value * 100))

    if not moves_df.empty:
        moves_df = moves_df.copy()
        if "agent_move_name" in moves_df.columns:
            moves_df["move_name"] = moves_df["agent_move_name"].astype(str)
        else:
            moves_df["move_name"] = moves_df["agent_move"].apply(
                lambda m: MOVE_NAMES[int(m)] if int(m) in range(len(MOVE_NAMES)) else str(m)
            )
        dist = moves_df["move_name"].value_counts().reindex(MOVE_NAMES, fill_value=0)
        dist_df = pd.DataFrame({"move": MOVE_NAMES, "count": dist.values})
        fig = px.bar(dist_df, x="move", y="count", color="move",
                     title=f"{display_name}'s Move Frequency")
        _style_figure(fig)
        st.plotly_chart(fig, width='stretch')

        st.caption("Outcome Distribution")
        outcome_map = {1: "Win", -1: "Loss", 0: "Tie"}
        moves_df["outcome_label"] = moves_df["outcome"].map(outcome_map)
        fig2 = px.pie(moves_df, names="outcome_label",
                      title="Win / Loss / Tie breakdown")
        fig2.update_traces(textfont_color="#F8FAFC")
        _style_figure(fig2)
        st.plotly_chart(fig2, width='stretch')

        st.caption("Move Sequence Over Time")
        fig3 = px.scatter(
            moves_df.head(200), x="turn", y="move_name",
            color="move_name",
            labels={"move_name": "Move", "turn": "Round"},
            title="Move choices across game turns",
        )
        fig3.update_layout(
            yaxis=dict(categoryorder="array", categoryarray=MOVE_NAMES)
        )
        _style_figure(fig3)
        st.plotly_chart(fig3, width='stretch')
    else:
        render_empty_state(
            "🎮",
            f"No gameplay data for {display_name}",
            "Once gameplay is collected for this profile, return here to see move distributions, outcomes, and clone fidelity.",
        )
        st.caption("To collect gameplay data, open Live Arena and play Train Clone Match.")

    st.divider()
    st.caption("Clone Fidelity Scores")
    if not imp_df.empty:
        fig4 = px.bar(imp_df, x="impostor_type", y="fidelity_score",
                      color="impostor_type",
                      title="How closely each Impostor type replicates this player")
        _style_figure(fig4)
        st.plotly_chart(fig4, width='stretch')
        st.dataframe(imp_df, width='stretch', hide_index=True)
        if not fingerprint_df.empty:
            st.caption("Per-Game Strategy Fingerprint")
            st.dataframe(fingerprint_df, width='stretch', hide_index=True)
        st.caption("Clone Ablation Table")
        ablation_cols = [col for col in ["impostor_type", "game_type", "fidelity_score", "kl_divergence", "fool_rate", "n_training_rounds"] if col in imp_df.columns]
        st.dataframe(imp_df[ablation_cols], width='stretch', hide_index=True)
    else:
        render_empty_state("🛰️", "No clone evaluations yet", "Use Train My Clone above after saving enough RPS+ rounds.")

    if not ab_df.empty:
        st.divider()
        st.caption("Clone-vs-Human A/B Evaluation Blocks")
        st.dataframe(ab_df, width='stretch', hide_index=True)
        paired = paired_significance(
            ab_df["baseline_wins"].fillna(0).astype(float).tolist(),
            ab_df["clone_wins"].fillna(0).astype(float).tolist(),
            metric="clone_vs_baseline_wins",
        )
        st.caption(f"Paired significance: delta {paired.mean_delta:.2f}, t={paired.t_stat:.2f}, p={paired.p_value:.3f}")
        if px is not None:
            fig_ab = px.bar(
                ab_df,
                x="clone_type",
                y=["clone_wins", "baseline_wins", "draws"],
                barmode="group",
                title="Clone vs baseline controlled blocks",
            )
            _style_figure(fig_ab)
            st.plotly_chart(fig_ab, width='stretch')
    if failure_df is not None and not failure_df.empty:
        st.divider()
        st.caption("Failure-Case Gallery")
        st.dataframe(failure_df, width='stretch', hide_index=True)

    if not timeline_df.empty:
        st.divider()
        st.caption("Narrative Figure")
        narrative_fig = _plot_clone_learning_narrative(timeline_df, drift_points, fooled_count)
        if narrative_fig is not None:
            st.plotly_chart(narrative_fig, width='stretch')

    clone_footer = (
        f"Clone model trained/evaluated. Fidelity: <strong>{fidelity:.0%}</strong>"
        if not imp_df.empty
        else "Clone model not trained yet. Use <strong>Train My Clone</strong> after saving enough RPS+ matches."
    )
    st.markdown(
        f"""
        <div class="dg-profile-footer">
            Your behavioral profile has <strong>{int(row['games_played'])}</strong> saved games.
            {clone_footer}
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        narrative_summary(
            games_played=int(row["games_played"]),
            fidelity_scores=imp_df["fidelity_score"].fillna(0.0).tolist() if not imp_df.empty else [],
            drift_scores=[point.drift_score for point in drift_points],
            fooled_count=fooled_count,
        )
    )
    if st.button("Play Against This Clone in Live Arena", width='stretch'):
        best_clone = "ngram"
        if not imp_df.empty:
            best_row = imp_df.sort_values("fidelity_score", ascending=False).iloc[0]
            best_clone = "lstm" if "lstm" in str(best_row["impostor_type"]).lower() else "ngram"
        st.session_state["game_settings"] = {
            **st.session_state.get("game_settings", {}),
            "friend_clone_source_player": player_id,
        }
        st.session_state["live_game_launch"] = {
            "game_type": "RPS+",
            "agent_name": best_clone,
            "agent2_name": best_clone,
            "play_mode": "human_bot1",
            "max_turns": 20,
        }
        st.switch_page("pages/live_game.py")

    st.divider()
    st.caption("Match Analytics")
    if sessions_df.empty:
        st.info("No saved matches yet, so grouped player/bot/game stats are not available.")
        return player_id, display_name

    default_player_filter = [player_id] if player_id in sessions_df["player_id"].astype(str).tolist() else []
    player_options = sorted(sessions_df["player_id"].astype(str).unique().tolist())
    bot_options = sorted(sessions_df["bot_name"].astype(str).unique().tolist())
    game_options = sorted(sessions_df["game_type"].astype(str).unique().tolist())

    filt1, filt2, filt3 = st.columns(3)
    with filt1:
        selected_players = st.multiselect(
            "Filter players",
            player_options,
            default=default_player_filter,
            help="Filter the grouped stats to one or more tracked players or guest IDs.",
        )
    with filt2:
        selected_bots = st.multiselect(
            "Filter bots",
            bot_options,
            default=[],
            help="Filter the grouped stats by opponent bot name.",
        )
    with filt3:
        selected_games = st.multiselect(
            "Filter games",
            game_options,
            default=[],
            help="Filter the grouped stats by game type.",
        )

    filtered_sessions = _filter_sessions(
        sessions_df,
        players=selected_players,
        bots=selected_bots,
        games=selected_games,
    )

    if filtered_sessions.empty:
        st.warning("No recorded matches match the current player, bot, and game filters.")
        return player_id, display_name

    m1, m2, m3, m4 = st.columns(4, gap="medium")
    m1.metric("Filtered matches", int(filtered_sessions["game_id"].nunique()))
    m2.metric("Players", int(filtered_sessions["player_id"].nunique()))
    m3.metric("Bots", int(filtered_sessions["bot_name"].nunique()))
    m4.metric("Games", int(filtered_sessions["game_type"].nunique()))

    by_player = _summarize_session_groups(filtered_sessions, ["player_name", "player_id"])
    by_bot = _summarize_session_groups(filtered_sessions, ["bot_name"])
    by_player_game = _summarize_session_groups(filtered_sessions, ["player_name", "game_type"])
    by_bot_game = _summarize_session_groups(filtered_sessions, ["bot_name", "game_type"])

    st.caption("Grouped by player")
    st.dataframe(by_player, width='stretch', hide_index=True)

    st.caption("Grouped by bot")
    st.dataframe(by_bot, width='stretch', hide_index=True)

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        st.caption("Player by game")
        st.dataframe(by_player_game, width='stretch', hide_index=True)
    with c2:
        st.caption("Bot by game")
        st.dataframe(by_bot_game, width='stretch', hide_index=True)

    if px is not None:
        chart1, chart2 = st.columns(2, gap="medium")
        with chart1:
            fig5 = px.bar(
                by_player,
                x="player_name",
                y="win_rate",
                color="games",
                title="Win rate by player",
                hover_data=["games", "wins", "losses", "ties"],
            )
            _style_figure(fig5)
            st.plotly_chart(fig5, width='stretch')
        with chart2:
            fig6 = px.bar(
                by_bot,
                x="bot_name",
                y="win_rate",
                color="games",
                title="Win rate by bot matchup",
                hover_data=["games", "wins", "losses", "ties"],
            )
            _style_figure(fig6)
            st.plotly_chart(fig6, width='stretch')

    st.caption("Filtered match log")
    st.dataframe(
        filtered_sessions[
            [
                "started_at",
                "player_name",
                "player_id",
                "bot_name",
                "game_type",
                "n_turns",
                "agent_score",
                "opponent_score",
                "result",
            ]
        ],
        width='stretch',
        hide_index=True,
    )

    return player_id, display_name


def _render_play_tab(player_id: str, display_name: str, db_file: Path, manager: PlayerProfileManager) -> None:
    st.caption("Play In Live Arena")
    st.markdown("Use Train Clone Match for saved gameplay. Finished matches are saved back to the selected profile.")

    if not player_id:
        st.info("Select a player profile in the Profiles tab before playing.")
        return

    st.session_state["user_profile"] = {"id": player_id, "name": display_name}
    st.success(f"Active profile: {display_name} ({player_id})")
    st.caption("Live Arena will show this profile as the active researcher and auto-save completed Train Clone Matches.")

    if st.button("Start Train Clone Match", width='stretch'):
        st.session_state["live_arena_experience"] = "Train Clone Match"
        try:
            st.switch_page("pages/live_game.py")
        except Exception:
            st.info("Use Train Clone Match from the home sidebar. Your selected profile is already active.")


def _render_admin_tab(conn: sqlite3.Connection, manager: PlayerProfileManager) -> None:
    st.caption("Admin Console")
    st.markdown("View player profile metadata and raw game counts from the database.")

    profiles_df = _load_profiles(conn)
    st.caption("Player Profiles")
    if profiles_df.empty:
        st.info("No player profiles available yet.")
    else:
        st.dataframe(profiles_df, width='stretch', hide_index=True)

    try:
        game_counts = pd.read_sql_query(
            "SELECT agent_name AS player_id, COUNT(*) AS games_played FROM games GROUP BY agent_name",
            conn,
        )
        st.caption("Game Counts by Profile")
        st.dataframe(game_counts, width='stretch', hide_index=True)
    except Exception:
        st.info("No games have been recorded yet.")

    st.markdown("---")
    st.caption("Admin Game Actions")
    st.info("Gameplay now happens in Live Arena. Finished Train Clone Matches save directly to the active profile.")
    active_profile = st.session_state.get("game_profile_id")
    active_game_type = st.session_state.get("game_game_type")
    game_done = st.session_state.get("game_done", False)
    if active_profile and active_game_type:
        st.markdown(f"**Current session:** {active_profile} / {active_game_type}")
        if game_done:
            st.markdown("This game has finished. Admins can save, inspect, or reset the session.")
            if not st.session_state.get("game_saved", False):
                if st.button("Save finished game to profile", key="admin_save_game"):
                    _save_played_game(DEFAULT_DB, active_profile, st.session_state["game_env"], manager)
                    st.session_state["game_saved"] = True
                    st.success("Saved finished game to profile.")
            else:
                st.info("This game has already been saved.")
        else:
            st.info("No finished older embedded game yet. Finish games in Live Arena.")
        st.button("Reset current game session", key="admin_reset_game", on_click=_reset_legacy_game_session)
    else:
        st.info("No older embedded play session found. Use the Play tab to launch Live Arena.")


def run() -> None:
    configure_page("Doppelgamer | Train My Clone")
    require_user_profile("Train My Clone")
    st.markdown(
        """
        <style>
            .dg-profile-header {
                display: flex;
                align-items: center;
                gap: 18px;
                padding: 22px;
                margin-bottom: 18px;
                border-radius: 22px;
                border: 1px solid rgba(0,255,157,0.24);
                background: linear-gradient(135deg, rgba(0,255,157,0.08), rgba(127,119,221,0.06)), #0a0a14;
            }
            .dg-avatar {
                width: 72px;
                height: 72px;
                border-radius: 999px;
                display: grid;
                place-items: center;
                color: var(--dg-clone);
                border: 2px solid var(--dg-clone);
                box-shadow: 0 0 28px rgba(0,255,157,0.22);
                font-family: var(--dg-mono);
                font-weight: 1000;
                font-size: 24px;
            }
            .dg-profile-header h1 {
                margin: 4px 0;
                font-family: var(--dg-mono);
                color: var(--dg-text);
            }
            .dg-bias {
                padding: 18px;
                margin: 12px 0 22px;
            }
            .dg-bias h3 {
                margin-top: 0;
                font-family: var(--dg-mono);
                color: var(--dg-clone) !important;
            }
            .dg-bias-row {
                margin-top: 14px;
            }
            .dg-bias-row div {
                display: flex;
                justify-content: space-between;
                color: var(--dg-text);
                font-family: var(--dg-mono);
                font-size: 13px;
            }
            .dg-bias-row i {
                display: block;
                height: 10px;
                margin-top: 6px;
                border-radius: 999px;
                background: rgba(255,255,255,0.08);
                overflow: hidden;
            }
            .dg-bias-row b {
                display: block;
                height: 100%;
                background: linear-gradient(90deg, var(--dg-human), var(--dg-clone), var(--dg-warn));
                border-radius: inherit;
            }
            .dg-profile-footer {
                margin-top: 24px;
                padding: 18px;
                border: 1px solid rgba(0,255,157,0.24);
                border-radius: 18px;
                background: rgba(0,255,157,0.055);
                color: var(--dg-text);
                font-family: var(--dg-mono);
            }
            .dg-profile-footer strong {
                color: var(--dg-clone);
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
    render_sidebar_nav("Train My Clone")
    st.title("Train My Clone")

    db_file = DEFAULT_DB
    init_db(db_file)
    init_extended_db(db_file)
    manager = PlayerProfileManager(db_file)
    conn = connect(str(db_file))

    try:
        tabs = st.tabs(["Profiles", "Arena", "Admin"])

        with tabs[0]:
            player_id, display_name = _render_profile_tab(conn, manager)

        with tabs[1]:
            _render_play_tab(player_id, display_name, db_file, manager)

        with tabs[2]:
            _render_admin_tab(conn, manager)
    finally:
        conn.close()


if __name__ == "__main__":
    run()
