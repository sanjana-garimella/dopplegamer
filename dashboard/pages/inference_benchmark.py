from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is in PYTHONPATH
root = Path(__file__).parent.parent.parent
if str(root) not in sys.path:
    sys.path.append(str(root))

"""Inference benchmark dashboard page: latency, throughput, KV cache by serving engine."""

import sqlite3
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from dashboard.auth import require_user_profile
from dashboard.config import db_path as configured_db_path
from dashboard.ui import configure_page, render_sidebar_nav, style_plotly_figure
from data.schemas import connect
from data.features import clone_latency_fidelity_frame


def load_inference_benchmarks(db_path: str | Path) -> pd.DataFrame:
    """Load inference benchmark results from database."""
    try:
        conn = connect(str(db_path))
        df = pd.read_sql_query(
            """SELECT * FROM inference_benchmarks 
               ORDER BY run_id, engine, prompt_tokens DESC""",
            conn
        )
        conn.close()
        return df
    except Exception as e:
        st.error(f"Failed to load inference benchmarks: {e}")
        return pd.DataFrame()


def render_inference_benchmark() -> None:
    configure_page("Doppelgamer | Inference Benchmark")
    require_user_profile("Inference Benchmark")
    render_sidebar_nav("Inference Benchmark")
    st.title("Inference Benchmark")
    st.caption("Benchmark latency, throughput, and memory behavior across inference engines.")
    st.divider()
    
    db_path = configured_db_path()
    if not db_path.exists():
        st.warning("Database not found. Run evaluation first: `python -m evaluation.runner`")
        return
    
    df = load_inference_benchmarks(db_path)
    if df.empty:
        st.info("No inference benchmark results in database yet.")
        return
    
    # Overview metrics
    st.caption("Summary Statistics")
    col1, col2, col3, col4 = st.columns(4, gap="medium")
    
    with col1:
        avg_ttft = df["ttft_ms"].mean()
        st.metric("Avg TTFT", f"{avg_ttft:.2f}ms", help="Time to First Token (Prefill)")
    
    with col2:
        avg_tpot = df["tpot_ms"].mean()
        st.metric("Avg TPOT", f"{avg_tpot:.2f}ms", help="Time Per Output Token (Decode)")
    
    with col3:
        avg_latency = df["total_latency_ms"].mean()
        st.metric("Avg Total Latency", f"{avg_latency:.2f}ms")
    
    with col4:
        avg_kv = df["kv_cache_mb"].mean()
        st.metric("Avg KV Cache", f"{avg_kv:.1f}MB")
    
    # Time to First Token (TTFT)
    st.divider()
    st.caption("Time to First Token (Prefill Latency)")
    fig_ttft = px.box(
        df,
        x="engine",
        y="ttft_ms",
        title="TTFT Distribution by Serving Engine",
        labels={"engine": "Engine", "ttft_ms": "TTFT (ms)"},
        color="engine"
    )
    style_plotly_figure(fig_ttft, height=320)
    st.plotly_chart(fig_ttft, width='stretch')
    
    # Time Per Output Token (TPOT)
    st.caption("Time Per Output Token (Decode Latency)")
    fig_tpot = px.box(
        df,
        x="engine",
        y="tpot_ms",
        title="TPOT Distribution by Serving Engine",
        labels={"engine": "Engine", "tpot_ms": "TPOT (ms)"},
        color="engine"
    )
    style_plotly_figure(fig_tpot, height=320)
    st.plotly_chart(fig_tpot, width='stretch')
    
    # Total Latency Comparison
    st.caption("Total Latency by Engine")
    if "total_latency_ms" in df.columns:
        latency_by_engine = df.groupby("engine", as_index=False)["total_latency_ms"].mean()
        latency_by_engine = latency_by_engine.rename(columns={"total_latency_ms": "total"})
    else:
        latency_by_engine = df.groupby("engine")[["ttft_ms", "tpot_ms", "output_tokens"]].mean()
        latency_by_engine["total"] = latency_by_engine["ttft_ms"] + latency_by_engine["tpot_ms"] * (
            latency_by_engine.get("output_tokens", 1) - 1
        ).clip(lower=0)
        latency_by_engine = latency_by_engine.reset_index()
    
    fig_total = px.bar(
        latency_by_engine,
        x="engine",
        y="total",
        title="Average Total Latency",
        labels={"engine": "Engine", "total": "Latency (ms)"},
        color="total",
        color_continuous_scale="Reds"
    )
    style_plotly_figure(fig_total, height=320)
    st.plotly_chart(fig_total, width='stretch')
    
    # Prefill vs Decode breakdown
    st.caption("Prefill vs Decode Latency Breakdown")
    breakdown_data = []
    for engine in df["engine"].unique():
        engine_df = df[df["engine"] == engine]
        breakdown_data.append({
            "Engine": engine,
            "Prefill (TTFT)": engine_df["ttft_ms"].mean(),
            "Decode (TPOT)": engine_df["tpot_ms"].mean()
        })
    breakdown_df = pd.DataFrame(breakdown_data)
    
    fig_breakdown = px.bar(
        breakdown_df,
        x="Engine",
        y=["Prefill (TTFT)", "Decode (TPOT)"],
        title="Prefill vs Decode Latency",
        barmode="stack",
        labels={"value": "Latency (ms)"}
    )
    style_plotly_figure(fig_breakdown, height=320)
    st.plotly_chart(fig_breakdown, width='stretch')
    
    # KV Cache Memory
    st.caption("KV Cache Memory Usage")
    fig_kv = px.box(
        df,
        x="engine",
        y="kv_cache_mb",
        title="KV Cache Memory by Engine",
        labels={"engine": "Engine", "kv_cache_mb": "KV Cache (MB)"},
        color="engine"
    )
    style_plotly_figure(fig_kv, height=320)
    st.plotly_chart(fig_kv, width='stretch')
    
    # Throughput (decisions per second)
    st.caption("Throughput Analysis")
    df["decisions_per_sec"] = 1000.0 / (df["total_latency_ms"] + 0.001)
    
    fig_throughput = px.bar(
        df.groupby("engine")["decisions_per_sec"].mean().reset_index(),
        x="engine",
        y="decisions_per_sec",
        title="Average Throughput (Decisions per Second)",
        labels={"engine": "Engine", "decisions_per_sec": "Decisions/sec"},
        color="decisions_per_sec",
        color_continuous_scale="Greens"
    )
    style_plotly_figure(fig_throughput, height=320)
    st.plotly_chart(fig_throughput, width='stretch')
    
    # Latency vs Prompt Size
    st.caption("Latency Scaling with Context Length")
    fig_scaling = px.scatter(
        df,
        x="prompt_tokens",
        y="total_latency_ms",
        color="engine",
        title="Total Latency vs Prompt Tokens",
        labels={"prompt_tokens": "Prompt Tokens", "total_latency_ms": "Latency (ms)", "engine": "Engine"}
    )
    style_plotly_figure(fig_scaling, height=320)
    st.plotly_chart(fig_scaling, width='stretch')
    
    # Detailed table
    st.divider()
    clone_df = clone_latency_fidelity_frame(db_path)
    if not clone_df.empty:
        st.caption("Clone Fidelity vs Latency")
        fig_clone = px.scatter(
            clone_df,
            x="avg_decision_ms",
            y="fidelity_score",
            color="impostor_type",
            size="n_training_rounds",
            hover_data=["player_id", "fool_rate", "game_type"],
            title="Clone fidelity / latency frontier",
            labels={"avg_decision_ms": "Average decision latency (ms)", "fidelity_score": "Fidelity score"},
        )
        style_plotly_figure(fig_clone, height=340)
        st.plotly_chart(fig_clone, width='stretch')

        if "fool_rate" in clone_df.columns:
            fig_fool = px.scatter(
                clone_df,
                x="avg_decision_ms",
                y="fool_rate",
                color="impostor_type",
                size="fidelity_score",
                hover_data=["player_id", "n_training_rounds"],
                title="Fool rate vs latency",
                labels={"avg_decision_ms": "Average decision latency (ms)", "fool_rate": "Fool rate"},
            )
            style_plotly_figure(fig_fool, height=320)
            st.plotly_chart(fig_fool, width='stretch')

    st.divider()
    st.caption("Detailed Benchmark Results")
    display_df = df[["engine", "quantization", "prompt_tokens", "output_tokens",
                     "ttft_ms", "tpot_ms", "total_latency_ms", "kv_cache_mb"]].copy()
    display_df["ttft_ms"] = display_df["ttft_ms"].apply(lambda x: f"{x:.2f}")
    display_df["tpot_ms"] = display_df["tpot_ms"].apply(lambda x: f"{x:.2f}")
    display_df["total_latency_ms"] = display_df["total_latency_ms"].apply(lambda x: f"{x:.2f}")
    display_df["kv_cache_mb"] = display_df["kv_cache_mb"].apply(lambda x: f"{x:.1f}")
    st.dataframe(display_df, width='stretch', hide_index=True)
    
    # Raw data download
    st.divider()
    st.caption("Export Data")
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="inference_benchmark.csv",
        mime="text/csv"
    )


if __name__ == "__main__":
    render_inference_benchmark()
