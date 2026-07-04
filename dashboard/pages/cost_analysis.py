"""Cost analysis dashboard page: cost per 1000 decisions, cost-quality ratios."""

from __future__ import annotations

from pathlib import Path

import numpy as np
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
from data.features import clone_latency_fidelity_frame, filter_aggregated_agent_results
from data.schemas import connect


def load_inference_benchmarks(db_path: str | Path) -> pd.DataFrame:
    """Load inference benchmark results from database."""
    try:
        conn = connect(str(db_path))
        df = pd.read_sql_query(
            "SELECT * FROM inference_benchmarks ORDER BY engine",
            conn
        )
        conn.close()
        return df
    except Exception as e:
        st.error(f"Failed to load benchmarks: {e}")
        return pd.DataFrame()


def load_agent_results(db_path: str | Path) -> pd.DataFrame:
    """Load aggregated agent results from database."""
    try:
        conn = connect(str(db_path))
        df = pd.read_sql_query(
            "SELECT * FROM agent_results ORDER BY agent_name",
            conn
        )
        conn.close()
        return filter_aggregated_agent_results(df)
    except Exception as e:
        st.error(f"Failed to load agent results: {e}")
        return pd.DataFrame()


def render_cost_analysis() -> None:
    configure_page("Doppelgamer | Cost Analysis")
    require_user_profile("Cost Analysis")
    render_sidebar_nav("Cost Analysis")
    st.title("Cost Analysis")
    st.caption("Study serving cost, throughput, and quality tradeoffs for the Doppelgamer stack.")
    st.divider()
    
    db_path = configured_db_path()
    if not db_path.exists():
        st.warning("Database not found. Run evaluation first: `python -m evaluation.runner`")
        return
    
    # Load data
    inference_df = load_inference_benchmarks(db_path)
    agent_df = load_agent_results(db_path)
    
    if inference_df.empty:
        st.info("No benchmark data available. Run inference benchmarks first.")
        return

    if "total_latency_ms" not in inference_df.columns:
        st.error("Inference benchmark data is missing latency columns. Re-run the benchmark to populate inference_benchmarks.")
        return
    
    # Cost configuration
    st.sidebar.caption("Cost Configuration")
    
    gpu_cost_per_hour = st.sidebar.number_input(
        "GPU Cost ($/hour)",
        min_value=0.0,
        value=2.50,
        step=0.10,
        help="H100 costs ~$2.50/hour on cloud. RTX 4090 ~$0.20/hour self-hosted."
    )
    
    utilization = st.sidebar.slider(
        "GPU Utilization %",
        min_value=10,
        max_value=100,
        value=80,
        step=5,
        help="Actual % of GPU time spent running inference"
    )
    
    # Calculate costs
    inference_df = inference_df.copy()
    inference_df["latency_sec"] = inference_df["total_latency_ms"] / 1000.0
    inference_df["decisions_per_hour"] = 3600.0 / (inference_df["latency_sec"] + 0.001)
    inference_df["cost_per_1k_decisions"] = (gpu_cost_per_hour * (utilization / 100.0)) / (inference_df["decisions_per_hour"] / 1000.0)
    inference_df["cost_per_decision_usd"] = inference_df["cost_per_1k_decisions"] / 1000.0
    
    # Overview metrics
    st.caption("Summary")
    col1, col2, col3, col4 = st.columns(4, gap="medium")
    
    with col1:
        avg_cost = inference_df["cost_per_1k_decisions"].mean()
        st.metric("Avg Cost / 1K Decisions", f"${avg_cost:.4f}")
    
    with col2:
        min_cost = inference_df["cost_per_1k_decisions"].min()
        min_engine = inference_df[inference_df["cost_per_1k_decisions"] == min_cost]["engine"].iloc[0]
        st.metric("Cheapest Engine", min_engine)
    
    with col3:
        max_cost = inference_df["cost_per_1k_decisions"].max()
        max_engine = inference_df[inference_df["cost_per_1k_decisions"] == max_cost]["engine"].iloc[0]
        st.metric("Most Expensive", max_engine)
    
    with col4:
        st.metric("GPU Cost (hourly)", f"${gpu_cost_per_hour:.2f}")
    
    # Cost per 1K decisions by engine
    st.divider()
    st.caption("Cost Per 1,000 Decisions by Engine")
    
    cost_by_engine = inference_df.groupby("engine")[
        "cost_per_1k_decisions"
    ].mean().sort_values()
    
    fig_cost = px.bar(
        cost_by_engine.reset_index(),
        x="engine",
        y="cost_per_1k_decisions",
        title="Average Cost per 1,000 Decisions",
        labels={"engine": "Engine", "cost_per_1k_decisions": "Cost ($)"},
        color="cost_per_1k_decisions",
        color_continuous_scale="Reds"
    )
    style_plotly_figure(fig_cost, height=320)
    st.plotly_chart(fig_cost, width='stretch')
    
    # Throughput vs Cost
    st.caption("Throughput vs Cost Tradeoff")
    
    fig_throughput_vs_cost = px.scatter(
        inference_df,
        x="cost_per_1k_decisions",
        y="decisions_per_hour",
        color="engine",
        size="kv_cache_mb",
        hover_data=["total_latency_ms", "quantization"],
        title="Cost vs Throughput (size = KV cache)",
        labels={
            "cost_per_1k_decisions": "Cost per 1K Decisions ($)",
            "decisions_per_hour": "Throughput (decisions/hour)"
        }
    )
    style_plotly_figure(fig_throughput_vs_cost, height=320)
    st.plotly_chart(fig_throughput_vs_cost, width='stretch')
    
    # Cost by Quantization
    st.caption("Cost Impact of Quantization")
    
    cost_by_quant = inference_df.groupby("quantization")[
        "cost_per_1k_decisions"
    ].mean().sort_values()
    
    if not cost_by_quant.empty:
        fig_quant = px.bar(
            cost_by_quant.reset_index(),
            x="quantization",
            y="cost_per_1k_decisions",
            title="Cost per 1K Decisions by Quantization",
            labels={"quantization": "Quantization", "cost_per_1k_decisions": "Cost ($)"},
            color="cost_per_1k_decisions",
            color_continuous_scale="Viridis"
        )
        style_plotly_figure(fig_quant, height=320)
        st.plotly_chart(fig_quant, width='stretch')
    
    # Cost breakdown for different scales
    st.caption("Cost Projection: Scale Analysis")
    
    scales = [100, 1000, 10000, 100000, 1000000]
    scale_data = []
    
    for engine in inference_df["engine"].unique():
        cost_per_1k = inference_df[inference_df["engine"] == engine]["cost_per_1k_decisions"].mean()
        for scale in scales:
            total_cost = (scale / 1000.0) * cost_per_1k
            scale_data.append({
                "Engine": engine,
                "Scale (decisions)": scale,
                "Total Cost ($)": total_cost
            })
    
    scale_df = pd.DataFrame(scale_data)
    
    fig_scale = px.line(
        scale_df,
        x="Scale (decisions)",
        y="Total Cost ($)",
        color="Engine",
        title="Cost Scaling to Different Decision Volumes",
        markers=True,
        log_x=True
    )
    style_plotly_figure(fig_scale, height=320)
    st.plotly_chart(fig_scale, width='stretch')
    
    # Quality vs Cost (if agent results available)
    if not agent_df.empty:
        st.caption("Quality vs Cost: Agent Performance")
        
        # Merge agent quality with serving cost (estimate via serving cost per agent)
        cost_by_engine_dict = inference_df.groupby("engine")[
            "cost_per_1k_decisions"
        ].mean().to_dict()
        
        # Assume serving cost is averaged across engines for simplicity
        avg_serving_cost = inference_df["cost_per_1k_decisions"].mean()
        
        agent_df = agent_df.copy()
        agent_df["estimated_cost_per_1k"] = avg_serving_cost  # Simplified assumption
        
        fig_quality_vs_cost = px.scatter(
            agent_df,
            x="estimated_cost_per_1k",
            y="win_rate",
            color="agent_name",
            size="avg_decision_ms",
            hover_data=["behavioral_fidelity", "games_played"],
            title="Agent Quality vs Serving Cost (size = latency)",
            labels={
                "estimated_cost_per_1k": "Serving Cost per 1K Decisions ($)",
                "win_rate": "Win Rate",
                "agent_name": "Agent"
            }
        )
        style_plotly_figure(fig_quality_vs_cost, height=320)
        st.plotly_chart(fig_quality_vs_cost, width='stretch')

    clone_df = clone_latency_fidelity_frame(db_path)
    if not clone_df.empty:
        st.divider()
        st.caption("Runtime Condition Study")
        runtime_conditions = pd.DataFrame(
            [
                {"condition": "interactive_live", "latency_budget_ms": 5.0, "min_fidelity": 0.60},
                {"condition": "arena_research", "latency_budget_ms": 20.0, "min_fidelity": 0.70},
                {"condition": "paper_quality", "latency_budget_ms": 60.0, "min_fidelity": 0.78},
            ]
        )
        condition_rows = []
        for cond in runtime_conditions.itertuples(index=False):
            feasible = clone_df[
                (clone_df["avg_decision_ms"].fillna(np.inf) <= cond.latency_budget_ms)
                & (clone_df["fidelity_score"].fillna(0.0) >= cond.min_fidelity)
            ]
            best = feasible.sort_values("fidelity_score", ascending=False).head(1)
            condition_rows.append(
                {
                    "condition": cond.condition,
                    "latency_budget_ms": cond.latency_budget_ms,
                    "min_fidelity": cond.min_fidelity,
                    "feasible_variants": int(len(feasible)),
                    "recommended_clone": best["impostor_type"].iloc[0] if not best.empty else "none",
                }
            )
        runtime_df = pd.DataFrame(condition_rows)
        st.dataframe(runtime_df, width='stretch', hide_index=True)

        fig_runtime = px.scatter(
            clone_df,
            x="avg_decision_ms",
            y="fidelity_score",
            color="impostor_type",
            hover_data=["player_id", "fool_rate"],
            title="Runtime feasibility map for clone variants",
            labels={"avg_decision_ms": "Decision latency (ms)", "fidelity_score": "Fidelity"},
        )
        fig_runtime.add_vline(x=5.0, line_dash="dot", line_color="#ff6b35")
        fig_runtime.add_vline(x=20.0, line_dash="dot", line_color="#ffd166")
        fig_runtime.add_hline(y=0.70, line_dash="dash", line_color="#00ff9d")
        style_plotly_figure(fig_runtime, height=340)
        st.plotly_chart(fig_runtime, width='stretch')
    
    # Economic recommendations
    st.divider()
    st.caption("Economic Recommendations")
    
    col1, col2 = st.columns(2, gap="medium")
    
    with col1:
        best_cost = inference_df.loc[inference_df["cost_per_1k_decisions"].idxmin()]
        st.markdown(f"""
        **Most Cost-Effective Setup:**
        - **Engine**: {best_cost["engine"]}
        - **Quantization**: {best_cost["quantization"]}
        - **Cost/1K**: ${best_cost["cost_per_1k_decisions"]:.4f}
        - **Throughput**: {best_cost["decisions_per_hour"]:.0f} decisions/hour
        """)
    
    with col2:
        best_throughput = inference_df.loc[inference_df["decisions_per_hour"].idxmax()]
        st.markdown(f"""
        **Highest Throughput:**
        - **Engine**: {best_throughput["engine"]}
        - **Quantization**: {best_throughput["quantization"]}
        - **Throughput**: {best_throughput["decisions_per_hour"]:.0f} decisions/hour
        - **Cost/1K**: ${best_throughput["cost_per_1k_decisions"]:.4f}
        """)
    
    # Detailed cost table
    st.divider()
    st.caption("Detailed Cost Breakdown")
    
    display_df = inference_df[[
        "engine", "quantization", "total_latency_ms", 
        "decisions_per_hour", "cost_per_1k_decisions", "cost_per_decision_usd"
    ]].drop_duplicates().sort_values("cost_per_1k_decisions")
    
    display_df["total_latency_ms"] = display_df["total_latency_ms"].apply(lambda x: f"{x:.2f}")
    display_df["decisions_per_hour"] = display_df["decisions_per_hour"].apply(lambda x: f"{x:.0f}")
    display_df["cost_per_1k_decisions"] = display_df["cost_per_1k_decisions"].apply(lambda x: f"${x:.6f}")
    display_df["cost_per_decision_usd"] = display_df["cost_per_decision_usd"].apply(lambda x: f"${x:.8f}")
    
    st.dataframe(display_df, width='stretch', hide_index=True)
    
    # Download results
    st.divider()
    st.caption("Export Analysis")
    csv = inference_df.to_csv(index=False)
    st.download_button(
        label="Download Cost Analysis CSV",
        data=csv,
        file_name="cost_analysis.csv",
        mime="text/csv"
    )


if __name__ == "__main__":
    render_cost_analysis()
