from __future__ import annotations

"""Impostor leaderboard dashboard page."""

import sys
from pathlib import Path

# Ensure project root is in PYTHONPATH
root = Path(__file__).parent.parent.parent
if str(root) not in sys.path:
    sys.path.append(str(root))

import json
import sqlite3
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.auth import require_user_profile
from dashboard.config import db_path as configured_db_path
from dashboard.ui import configure_page, render_sidebar_nav, style_plotly_figure
from data.features import canonical_figure_frames
from impostor.metrics import estimate_required_sample_size, paired_significance, susceptibility_score
from data.schemas import init_db, init_extended_db

DEFAULT_DB = configured_db_path()


def _load_impostor_results(conn: sqlite3.Connection) -> pd.DataFrame:
    try:
        return pd.read_sql_query(
            """SELECT ir.player_id, pp.display_name, ir.impostor_type,
                      ir.game_type, ir.fidelity_score, ir.kl_divergence, ir.tvd, ir.fool_rate,
                      ir.n_training_rounds, ir.trained_at
               FROM impostor_results ir
               LEFT JOIN player_profiles pp USING (player_id)
               ORDER BY ir.fidelity_score DESC""",
            conn,
        )
    except Exception:
        return pd.DataFrame()


def _load_detection_stats(conn: sqlite3.Connection) -> pd.DataFrame:
    try:
        return pd.read_sql_query(
            """SELECT player_id, impostor_type,
                      COUNT(*) as n_trials,
                      AVG(detected_as_human) as detection_rate,
                      AVG(confidence) as mean_confidence
               FROM detection_sessions
               GROUP BY player_id, impostor_type""",
            conn,
        )
    except Exception:
        return pd.DataFrame()


def _load_profiles(conn: sqlite3.Connection) -> pd.DataFrame:
    try:
        return pd.read_sql_query(
            "SELECT player_id, display_name, games_played, win_rate FROM player_profiles",
            conn,
        )
    except Exception:
        return pd.DataFrame()


def _load_ab_runs(conn: sqlite3.Connection) -> pd.DataFrame:
    try:
        return pd.read_sql_query(
            """SELECT player_id, game_type, clone_type, baseline_type, n_games,
                      clone_wins, baseline_wins, draws, clone_fidelity, detection_rate, created_at
               FROM clone_ab_runs
               ORDER BY created_at DESC""",
            conn,
        )
    except Exception:
        return pd.DataFrame()


def run() -> None:
    configure_page("Doppelgamer | Clone Leaderboard")
    require_user_profile("Clone Leaderboard")
    render_sidebar_nav("Clone Leaderboard")
    st.title("Clone Leaderboard")
    st.caption(
        "All trained clone agents ranked by behavioral fidelity and detection resistance."
    )
    st.divider()

    db_file = DEFAULT_DB
    if not db_file.exists():
        st.warning("Database not found. Play Train Clone Match and train a clone first.")
        return
    init_db(db_file)
    init_extended_db(db_file)

    conn = sqlite3.connect(str(db_file))
    try:
        imp_df = _load_impostor_results(conn)
        det_df = _load_detection_stats(conn)
        prof_df = _load_profiles(conn)
        ab_df = _load_ab_runs(conn)

        # ── summary metrics
        col1, col2, col3 = st.columns(3, gap="medium")
        col1.metric("Trained Clones", len(imp_df))
        col2.metric("Unique Players", imp_df["player_id"].nunique() if not imp_df.empty else 0)
        col3.metric(
            "Avg Fidelity",
            f"{imp_df['fidelity_score'].mean():.2f}" if not imp_df.empty else "—",
        )

        # ── fidelity ranking
        st.caption("Fidelity Ranking")
        if not imp_df.empty:
            label = imp_df.get("display_name", imp_df["player_id"]).fillna(imp_df["player_id"])
            imp_df["label"] = label + " (" + imp_df["impostor_type"] + ")"
            fig = px.bar(
                imp_df.head(20),
                x="fidelity_score",
                y="label",
                orientation="h",
                color="impostor_type",
                title="Top Clones by Behavioral Fidelity (1 = perfect clone)",
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            style_plotly_figure(fig, height=340)
            st.plotly_chart(fig, width='stretch')

            st.caption("Fidelity vs KL Divergence")
            fig2 = px.scatter(
                imp_df,
                x="kl_divergence",
                y="fidelity_score",
                color="impostor_type",
                hover_data=["player_id", "n_training_rounds"],
                title="Lower KL → closer clone (fidelity ≥ 0.7 is publication threshold)",
            )
            fig2.add_hline(y=0.7, line_dash="dash", annotation_text="0.70 threshold")
            style_plotly_figure(fig2, height=320)
            st.plotly_chart(fig2, width='stretch')

            st.caption("Training Data vs Fidelity")
            fig3 = px.scatter(
                imp_df,
                x="n_training_rounds",
                y="fidelity_score",
                color="impostor_type",
                trendline="lowess",
                title="More training data → higher fidelity?",
            )
            style_plotly_figure(fig3, height=320)
            st.plotly_chart(fig3, width='stretch')
        else:
            st.info("No clone training results found yet. Open Player Profile and choose Train My Clone.")

        # ── detection stats (Turing test)
        st.divider()
        st.caption("Human Detection Rates (Turing Test)")
        if not det_df.empty:
            fig4 = px.bar(
                det_df,
                x="impostor_type",
                y="detection_rate",
                color="impostor_type",
                title="Detection Rate by Impostor Type (lower = more convincing)",
            )
            fig4.add_hline(y=0.5, line_dash="dash", annotation_text="random chance")
            style_plotly_figure(fig4, height=320)
            st.plotly_chart(fig4, width='stretch')
            st.caption("Live Turing Test Ops Console")
            ops1, ops2 = st.columns(2, gap="medium")
            with ops1:
                fig_hist = px.histogram(
                    det_df,
                    x="mean_confidence",
                    color="impostor_type",
                    nbins=10,
                    title="Confidence histogram by clone type",
                )
                style_plotly_figure(fig_hist, height=280)
                st.plotly_chart(fig_hist, width='stretch')
            with ops2:
                susceptibility = (
                    det_df.groupby("player_id")["detection_rate"]
                    .apply(lambda s: susceptibility_score(s.tolist()))
                    .reset_index(name="susceptibility")
                )
                if not susceptibility.empty:
                    fig_sus = px.bar(
                        susceptibility,
                        x="player_id",
                        y="susceptibility",
                        title="Per-player susceptibility to clone deception",
                    )
                    style_plotly_figure(fig_sus, height=280)
                    st.plotly_chart(fig_sus, width='stretch')
            human_baseline = det_df.loc[det_df["impostor_type"] == "human_baseline"].copy()
            if not human_baseline.empty:
                st.caption("Human-vs-Human Confusion Baseline")
                st.dataframe(human_baseline, width='stretch', hide_index=True)
        else:
            st.info("No Turing test data recorded yet.")

        st.divider()
        st.caption("Clone Leaderboards by Fool Rate")
        if not imp_df.empty:
            fool_df = imp_df.copy()
            fool_df["fool_rate"] = fool_df["fool_rate"].fillna(0.0)
            leader_cols = st.columns(3, gap="medium")
            hardest = fool_df.sort_values("fool_rate", ascending=False).head(10)
            most_faithful = fool_df.sort_values("fidelity_score", ascending=False).head(10)
            if not det_df.empty:
                fooled = det_df.sort_values("detection_rate").head(10)
            else:
                fooled = pd.DataFrame()
            with leader_cols[0]:
                st.caption("Most convincing clones")
                st.dataframe(
                    hardest[["display_name", "impostor_type", "game_type", "fool_rate"]],
                    width='stretch',
                    hide_index=True,
                )
            with leader_cols[1]:
                st.caption("Hardest humans to imitate")
                st.dataframe(
                    most_faithful[["display_name", "impostor_type", "game_type", "fidelity_score"]],
                    width='stretch',
                    hide_index=True,
                )
            with leader_cols[2]:
                st.caption("Most fooled by a clone")
                if fooled.empty:
                    st.info("No detection sessions yet.")
                else:
                    st.dataframe(fooled, width='stretch', hide_index=True)

        if not ab_df.empty:
            st.divider()
            st.caption("Controlled A/B Blocks")
            st.dataframe(ab_df, width='stretch', hide_index=True)
            if {"clone_fidelity", "detection_rate"}.issubset(ab_df.columns):
                power_cols = st.columns(2, gap="medium")
                with power_cols[0]:
                    fidelity_effect = float(ab_df["clone_fidelity"].fillna(0).mean())
                    power = estimate_required_sample_size(effect_size=max(fidelity_effect, 0.1))
                    st.metric("Estimated N / group", power.estimated_n_per_group)
                    st.caption(f"Power analysis from current fidelity effect size ({power.effect_size:.2f}).")
                with power_cols[1]:
                    paired = paired_significance(
                        ab_df["baseline_wins"].fillna(0).astype(float).tolist(),
                        ab_df["clone_wins"].fillna(0).astype(float).tolist(),
                        metric="clone_vs_baseline_wins",
                    )
                    st.metric("Paired delta", f"{paired.mean_delta:.2f}")
                    st.caption(f"Paired t-stat {paired.t_stat:.2f}, p={paired.p_value:.3f} across {paired.n_players} player blocks.")

        if not imp_df.empty and "player_id" in imp_df.columns:
            st.divider()
            st.caption("Ablation / Failure / Figure Preview")
            selected_player = st.selectbox(
                "Figure subject",
                sorted(imp_df["player_id"].dropna().astype(str).unique().tolist()),
                key="leaderboard_selected_player",
            )
            frames = canonical_figure_frames(DEFAULT_DB, selected_player)
            preview_cols = st.columns(3, gap="medium")
            with preview_cols[0]:
                st.caption("Generalization split")
                generalization = frames.get("generalization", pd.DataFrame())
                if generalization is not None and not generalization.empty:
                    st.dataframe(generalization, width='stretch', hide_index=True)
            with preview_cols[1]:
                st.caption("Failure cases")
                failures = frames.get("failures", pd.DataFrame())
                if failures is not None and not failures.empty:
                    st.dataframe(failures.head(5), width='stretch', hide_index=True)
            with preview_cols[2]:
                st.caption("Fingerprint")
                fingerprint = frames.get("fingerprint", pd.DataFrame())
                if fingerprint is not None and not fingerprint.empty:
                    st.dataframe(fingerprint.head(5), width='stretch', hide_index=True)

        # ── player profiles
        st.divider()
        st.caption("Player Profiles")
        if not prof_df.empty:
            st.dataframe(prof_df, width='stretch', hide_index=True)

        # ── raw table
        if not imp_df.empty:
            st.caption("Full Results Table")
            cols = [c for c in ["display_name", "impostor_type", "fidelity_score",
                                "kl_divergence", "n_training_rounds", "trained_at"]
                    if c in imp_df.columns]
            st.dataframe(imp_df[cols], width='stretch', hide_index=True)
            csv = imp_df[cols].to_csv(index=False)
            st.download_button("Download CSV", csv, "impostor_results.csv", "text/csv")
    finally:
        conn.close()


if __name__ == "__main__":
    run()
