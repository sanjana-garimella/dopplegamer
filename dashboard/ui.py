"""Shared Streamlit UI helpers for the Doppelgamer dashboard."""

from __future__ import annotations

from pathlib import Path

import streamlit as st


STYLE_PATH = Path(__file__).with_name("style.css")

NAV_ITEMS = [
    ("Hub", "app.py"),
    ("Live Arena", "pages/live_game.py"),
    ("Train My Clone", "pages/player_profile.py"),
    ("Clone Leaderboard", "pages/impostor_leaderboard.py"),
    ("Agent Comparison", "pages/agent_comparison.py"),
    ("Cost Analysis", "pages/cost_analysis.py"),
    ("KV Cache", "pages/kv_cache_viz.py"),
    ("Inference Benchmark", "pages/inference_benchmark.py"),
]


def inject_global_css() -> None:
    css = STYLE_PATH.read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def configure_page(page_title: str) -> None:
    st.set_page_config(page_title=page_title, layout="wide", page_icon="👾", initial_sidebar_state="expanded")
    inject_global_css()


def render_sidebar_nav(current_page: str) -> None:
    st.sidebar.markdown("## Doppelgamer")
    profile = st.session_state.get("user_profile")
    if profile:
        st.sidebar.caption(f"Signed in as {profile['name']}")
    st.sidebar.divider()
    st.sidebar.caption("Go to")
    for label, page in NAV_ITEMS:
        active = label == current_page
        if active:
            st.sidebar.markdown(
                f"<div class='dg-nav-pill is-active' style='display:block; margin-bottom:8px;'>{label}</div>",
                unsafe_allow_html=True,
            )
        else:
            if st.sidebar.button(label, key=f"sidebar_nav_{label}", width='stretch'):
                st.switch_page(page)


def style_plotly_figure(fig, *, height: int = 300):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=30, b=0),
        height=height,
    )
    return fig


def render_empty_state(icon: str, title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="dg-empty-state dg-placeholder">
            <div class="dg-empty-icon">{icon}</div>
            <div>
                <h3 style="margin-bottom:6px;">{title}</h3>
                <div class="dg-muted">{body}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_shell_open() -> None:
    st.markdown('<div class="dg-page-shell">', unsafe_allow_html=True)


def page_shell_close() -> None:
    st.markdown("</div>", unsafe_allow_html=True)
