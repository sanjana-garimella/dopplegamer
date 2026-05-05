"""Shared Doppelgamer surveillance-lab Streamlit theme."""

from __future__ import annotations

import streamlit as st


def inject_doppelgamer_theme() -> None:
    st.markdown(
        """
        <style>
            :root {
                --dg-bg: #0a0a14;
                --dg-panel: rgba(11, 16, 32, 0.92);
                --dg-panel-soft: rgba(16, 24, 44, 0.82);
                --dg-clone: #00ff9d;
                --dg-human: #7f77dd;
                --dg-warn: #ff6b35;
                --dg-text: #f4fff9;
                --dg-muted: #8fa7a0;
                --dg-line: rgba(0,255,157,0.20);
                --dg-grid: rgba(0,255,157,0.055);
                --dg-mono: "SFMono-Regular", "Cascadia Code", "Roboto Mono", Consolas, monospace;
                --dg-sans: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            }
            header, [data-testid="stHeader"], footer, [data-testid="stToolbar"] {
                visibility: hidden !important;
                height: 0 !important;
            }
            html, body, .stApp {
                background:
                    radial-gradient(circle at 14% 8%, rgba(0,255,157,0.12), transparent 26rem),
                    radial-gradient(circle at 84% 10%, rgba(127,119,221,0.14), transparent 24rem),
                    linear-gradient(180deg, #0a0a14 0%, #080812 100%) !important;
                color: var(--dg-text) !important;
                font-family: var(--dg-sans) !important;
            }
            .stApp::before {
                content: "";
                position: fixed;
                inset: 0;
                pointer-events: none;
                background-image:
                    repeating-linear-gradient(0deg, transparent 0 23px, var(--dg-grid) 24px),
                    repeating-linear-gradient(90deg, transparent 0 23px, var(--dg-grid) 24px);
                mask-image: radial-gradient(circle at 50% 20%, black, transparent 78%);
                opacity: 0.9;
                z-index: 0;
            }
            .block-container { position: relative; z-index: 1; }
            [data-testid="stSidebar"] {
                background:
                    repeating-linear-gradient(0deg, rgba(0,255,157,0.055) 0 1px, transparent 1px 18px),
                    repeating-linear-gradient(90deg, rgba(127,119,221,0.045) 0 1px, transparent 1px 18px),
                    #070710 !important;
                border-right: 1px solid var(--dg-line) !important;
                box-shadow: 20px 0 60px rgba(0,0,0,0.38) !important;
            }
            [data-testid="stSidebar"] * { color: var(--dg-text) !important; }
            ::-webkit-scrollbar { width: 8px; height: 8px; }
            ::-webkit-scrollbar-track { background: #070710; }
            ::-webkit-scrollbar-thumb {
                background: linear-gradient(180deg, var(--dg-clone), var(--dg-human));
                border-radius: 999px;
            }
            [data-testid="stMetric"],
            .dg-stat-card,
            .dg-card,
            .dg-feed,
            .dg-threat-card {
                background:
                    linear-gradient(145deg, rgba(0,255,157,0.055), rgba(127,119,221,0.045)),
                    var(--dg-panel) !important;
                border: 1px solid var(--dg-line) !important;
                border-radius: 18px !important;
                box-shadow: 0 18px 50px rgba(0,0,0,0.34), inset 0 1px 0 rgba(255,255,255,0.05) !important;
            }
            [data-testid="stMetric"] {
                padding: 16px !important;
            }
            [data-testid="stMetricValue"] {
                color: var(--dg-clone) !important;
                font-family: var(--dg-mono) !important;
                text-shadow: 0 0 18px rgba(0,255,157,0.34);
            }
            [data-testid="stMetricLabel"],
            [data-testid="stMetricDelta"] {
                color: var(--dg-muted) !important;
                font-family: var(--dg-mono) !important;
            }
            .stButton > button {
                background: rgba(0,255,157,0.035) !important;
                color: var(--dg-clone) !important;
                border: 1px solid rgba(0,255,157,0.44) !important;
                border-radius: 12px !important;
                font-family: var(--dg-mono) !important;
                font-weight: 800 !important;
                letter-spacing: 0.02em !important;
                transition: transform 160ms ease, box-shadow 160ms ease, background 160ms ease !important;
            }
            .stButton > button:hover {
                background: rgba(0,255,157,0.18) !important;
                box-shadow: 0 0 28px rgba(0,255,157,0.22) !important;
                transform: translateY(-1px);
            }
            .stButton > button[kind="primary"] {
                background: linear-gradient(135deg, rgba(0,255,157,0.24), rgba(127,119,221,0.20)) !important;
                color: #f4fff9 !important;
            }
            h1, h2, h3 { color: var(--dg-text) !important; }
            p, label, .stCaption, [data-testid="stCaptionContainer"] {
                color: var(--dg-muted) !important;
            }
            code, pre {
                color: var(--dg-clone) !important;
                font-family: var(--dg-mono) !important;
            }
            .dg-live {
                display: inline-flex;
                align-items: center;
                gap: 8px;
                color: var(--dg-clone);
                font-family: var(--dg-mono);
                font-weight: 900;
            }
            .dg-live::before {
                content: "";
                width: 10px;
                height: 10px;
                border-radius: 999px;
                background: var(--dg-clone);
                box-shadow: 0 0 0 0 rgba(0,255,157,0.55);
                animation: dg-pulse 1.2s infinite;
            }
            @keyframes dg-pulse {
                70% { box-shadow: 0 0 0 10px rgba(0,255,157,0); }
                100% { box-shadow: 0 0 0 0 rgba(0,255,157,0); }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
