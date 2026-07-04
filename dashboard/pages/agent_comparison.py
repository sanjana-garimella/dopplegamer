"""Agent comparison dashboard page: win rate, fidelity, action KL divergence."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

import sys
from pathlib import Path

# Ensure project root is in PYTHONPATH
root = Path(__file__).parent.parent.parent
if str(root) not in sys.path:
    sys.path.append(str(root))

from dashboard.auth import require_user_profile
from dashboard.config import db_path as configured_db_path
from dashboard.ui import configure_page, render_sidebar_nav, style_plotly_figure
from data.features import filter_aggregated_agent_results
from data.schemas import connect


def load_agent_results(db_path: str | Path) -> pd.DataFrame:
    """Load aggregated agent benchmark results (drop per-seed duplicates)."""
    try:
        conn = connect(str(db_path))
        df = pd.read_sql_query(
            "SELECT * FROM agent_results ORDER BY run_id, agent_name",
            conn
        )
        conn.close()
        return filter_aggregated_agent_results(df)
    except Exception as e:
        st.error(f"Failed to load agent results: {e}")
        return pd.DataFrame()


def render_agent_comparison() -> None:
    configure_page("Doppelgamer | Agent Comparison")
    require_user_profile("Agent Comparison")
    render_sidebar_nav("Agent Comparison")
    st.title("Agent Comparison")
    st.caption("Compare win rate, behavioral fidelity, action divergence, and latency across agent architectures.")
    st.divider()
    
    db_path = configured_db_path()
    if not db_path.exists():
        st.warning("Database not found. Run evaluation first: `python -m evaluation.runner`")
        return
    
    df = load_agent_results(db_path)
    if df.empty:
        st.info("No agent results in database yet.")
        return

    required_columns = {"agent_name", "win_rate", "behavioral_fidelity", "action_kl", "avg_decision_ms"}
    if not required_columns.issubset(df.columns):
        st.error("Agent results are missing expected columns. Please regenerate the benchmark data.")
        return

    df = df.copy().fillna(0)
    
    # Overview metrics
    st.caption("Summary Statistics")
    col1, col2, col3, col4 = st.columns(4, gap="medium")
    
    with col1:
        avg_wr = df["win_rate"].mean()
        st.metric("Avg Win Rate", f"{avg_wr:.2%}")
    
    with col2:
        avg_fidelity = df["behavioral_fidelity"].mean()
        st.metric("Avg Fidelity", f"{avg_fidelity:.2%}")
    
    with col3:
        avg_kl = df["action_kl"].mean()
        st.metric("Avg Action KL", f"{avg_kl:.3f}")
    
    with col4:
        avg_latency = df["avg_decision_ms"].mean()
        st.metric("Avg Latency", f"{avg_latency:.1f}ms")
    
    # Win Rate Comparison
    st.divider()
    st.caption("Win Rate by Agent")
    fig_wr = px.bar(
        df.groupby("agent_name")["win_rate"].mean().reset_index(),
        x="agent_name",
        y="win_rate",
        title="Average Win Rate",
        labels={"agent_name": "Agent", "win_rate": "Win Rate"},
        color="win_rate",
        color_continuous_scale="Viridis"
    )
    style_plotly_figure(fig_wr, height=320)
    fig_wr.update_layout(showlegend=False)
    st.plotly_chart(fig_wr, width='stretch')
    
    # Behavioral Fidelity Comparison
    st.caption("Behavioral Fidelity")
    fig_fidelity = px.bar(
        df.groupby("agent_name")["behavioral_fidelity"].mean().reset_index(),
        x="agent_name",
        y="behavioral_fidelity",
        title="Average Behavioral Fidelity (Move Prediction Accuracy)",
        labels={"agent_name": "Agent", "behavioral_fidelity": "Fidelity"},
        color="behavioral_fidelity",
        color_continuous_scale="Blues"
    )
    style_plotly_figure(fig_fidelity, height=320)
    fig_fidelity.update_layout(showlegend=False)
    st.plotly_chart(fig_fidelity, width='stretch')
    
    # Win Rate vs Fidelity Scatter
    st.caption("Win Rate vs Behavioral Fidelity")
    fig_scatter = px.scatter(
        df,
        x="behavioral_fidelity",
        y="win_rate",
        color="agent_name",
        size="avg_decision_ms",
        hover_data=["games_played", "action_kl"],
        title="Fidelity-Performance Tradeoff (size = latency)",
        labels={"behavioral_fidelity": "Fidelity", "win_rate": "Win Rate", "agent_name": "Agent"}
    )
    style_plotly_figure(fig_scatter, height=320)
    st.plotly_chart(fig_scatter, width='stretch')
    
    # Action KL Divergence
    st.caption("Action Distribution KL Divergence")
    fig_kl = px.box(
        df,
        x="agent_name",
        y="action_kl",
        title="KL Divergence from Human Move Distribution",
        labels={"agent_name": "Agent", "action_kl": "KL(agent || human)"},
        color="agent_name"
    )
    style_plotly_figure(fig_kl, height=320)
    st.plotly_chart(fig_kl, width='stretch')
    
    # Detailed table
    st.divider()
    st.caption("Detailed Results")
    display_df = df[["agent_name", "games_played", "wins", "losses", "ties", 
                      "win_rate", "behavioral_fidelity", "action_kl", "avg_decision_ms"]].copy()
    display_df["win_rate"] = display_df["win_rate"].apply(lambda x: f"{x:.2%}")
    display_df["behavioral_fidelity"] = display_df["behavioral_fidelity"].apply(lambda x: f"{x:.2%}")
    display_df["action_kl"] = display_df["action_kl"].apply(lambda x: f"{x:.3f}")
    display_df["avg_decision_ms"] = display_df["avg_decision_ms"].apply(lambda x: f"{x:.1f}")
    st.dataframe(display_df, width='stretch', hide_index=True)
    
    # Raw data download
    st.divider()
    st.caption("Export Data")
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="agent_comparison.csv",
        mime="text/csv"
    )


if __name__ == "__main__":
    render_agent_comparison()
