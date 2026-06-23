"""Shared Streamlit profile gate for secondary pages."""

from __future__ import annotations

import streamlit as st

from dashboard.navigation import switch_page_compat

try:  # pragma: no cover - depends on Streamlit runtime internals
    from streamlit.runtime.scriptrunner import get_script_run_ctx
except Exception:  # pragma: no cover
    get_script_run_ctx = None


def require_user_profile(page_name: str) -> dict:
    """Stop page rendering until the user selects or creates a profile."""
    profile = st.session_state.get("user_profile")
    if profile:
        return profile
    if get_script_run_ctx is None or get_script_run_ctx() is None:
        return {"id": "test_profile", "name": "Test Profile"}

    st.markdown(
        f"""
        <div style="
            border: 1px solid rgba(0,255,157,.24);
            background: linear-gradient(135deg, rgba(0,255,157,.08), rgba(127,119,221,.06)), #0a0a14;
            border-radius: 16px;
            padding: 28px;
            margin: 32px auto;
            max-width: 760px;
            box-shadow: 0 24px 80px rgba(0,0,0,.35);
        ">
            <div style="color:#00ff9d; font-family:monospace; letter-spacing:.08em; font-size:13px;">
                PROFILE REQUIRED
            </div>
            <h1 style="margin:8px 0 8px; color:#f8fafc;">Sign in before opening {page_name}</h1>
            <p style="color:#a6b0c3; margin:0;">
                Doppelgamer needs an active profile so matches, clone training, and research metrics stay attached to the right player.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Go To Login", type="primary", width='content'):
        switch_page_compat("app.py")
    st.stop()
