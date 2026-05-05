from __future__ import annotations

"""KV cache growth visualization dashboard page."""

import sys
from pathlib import Path

# Ensure project root is in PYTHONPATH
root = Path(__file__).parent.parent.parent
if str(root) not in sys.path:
    sys.path.append(str(root))

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analysis.kv_cache_profiler import track_kv_cache_growth
from dashboard.auth import require_user_profile
from dashboard.ui import configure_page, render_sidebar_nav, style_plotly_figure


def render_kv_cache_visualization() -> None:
    configure_page("Doppelgamer | KV Cache Analysis")
    require_user_profile("KV Cache Analysis")
    render_sidebar_nav("KV Cache")
    st.title("KV Cache Analysis")
    st.caption("Visualize memory pressure as context grows during sustained agent play.")
    st.divider()
    
    st.markdown("""
    **What is KV Cache?**
    
    The Key-Value (KV) cache stores computed attention matrices from previous tokens, 
    preventing recomputation during decoding. As game context grows, KV cache grows linearly 
    with tokens. This visualization shows memory pressure under sustained agent operations.
    """)
    
    # Configuration
    st.sidebar.caption("KV Cache Configuration")
    
    model = st.sidebar.selectbox(
        "Model Architecture",
        ["Llama-3.2 1B", "Llama-3.2 8B", "Llama-2 7B", "GPT-2"],
        index=0
    )
    
    model_configs = {
        "Llama-3.2 1B": {"n_layers": 16, "n_kv_heads": 8, "head_dim": 64},
        "Llama-3.2 8B": {"n_layers": 32, "n_kv_heads": 8, "head_dim": 128},
        "Llama-2 7B": {"n_layers": 32, "n_kv_heads": 32, "head_dim": 128},
        "GPT-2": {"n_layers": 12, "n_kv_heads": 12, "head_dim": 64},
    }
    
    config = model_configs.get(model, model_configs["Llama-3.2 1B"])
    
    max_turns = st.sidebar.slider("Max Game Turns", 10, 1000, 500, step=10)
    tokens_per_turn = st.sidebar.slider("Tokens per Turn", 1, 100, 20, step=1)
    quantization = st.sidebar.selectbox(
        "Quantization",
        ["fp32 (4B/value)", "fp16 (2B/value)", "int8 (1B/value)", "int4 (0.5B/value)"],
        index=1
    )
    
    quant_bytes = {
        "fp32 (4B/value)": 4,
        "fp16 (2B/value)": 2,
        "int8 (1B/value)": 1,
        "int4 (0.5B/value)": 0.5,
    }.get(quantization, 2)
    
    # Calculate KV cache growth
    samples = track_kv_cache_growth(
        turns=max_turns,
        tokens_per_turn=tokens_per_turn,
        n_layers=config["n_layers"],
        n_kv_heads=config["n_kv_heads"],
        head_dim=config["head_dim"],
    )
    
    # Base values are computed assuming FP16 bytes-per-value (2 bytes).
    base_samples_df = pd.DataFrame([
        {
            "turn": s.turn,
            "tokens": s.tokens,
            "kv_cache_mb": s.kv_cache_mb,
        }
        for s in samples
    ])

    # Adjust for quantization
    samples_df = base_samples_df.copy()
    samples_df["kv_cache_mb"] = samples_df["kv_cache_mb"] * (quant_bytes / 2.0)
    
    # Overview metrics
    st.caption("Summary")
    col1, col2, col3, col4 = st.columns(4, gap="medium")
    
    with col1:
        final_cache = samples_df["kv_cache_mb"].iloc[-1]
        st.metric("Final KV Cache", f"{final_cache:.1f}MB")
    
    with col2:
        tokens_final = samples_df["tokens"].iloc[-1]
        st.metric("Total Tokens", f"{tokens_final:,}")
    
    with col3:
        growth_rate = (samples_df["kv_cache_mb"].iloc[-1] - samples_df["kv_cache_mb"].iloc[0]) / max_turns
        st.metric("Growth Rate", f"{growth_rate:.3f}MB/turn")
    
    with col4:
        st.metric("Model", model)
    
    # KV Cache Growth Curve
    st.divider()
    st.caption("KV Cache Memory Growth Over Game Turns")
    fig_growth = px.line(
        samples_df,
        x="turn",
        y="kv_cache_mb",
        title="KV Cache Memory Accumulation",
        labels={"turn": "Game Turn", "kv_cache_mb": "KV Cache (MB)"},
        markers=True
    )
    fig_growth.add_hline(
        y=16.0, line_dash="dash", line_color="red",
        annotation_text="16GB GPU Limit",
        annotation_position="right"
    )
    style_plotly_figure(fig_growth, height=320)
    st.plotly_chart(fig_growth, width='stretch')
    
    # Memory vs Turn relationship (log scale)
    st.caption("Memory Growth Rate (Linear Scale)")
    fig_linear = go.Figure()
    fig_linear.add_trace(go.Scatter(
        x=samples_df["turn"],
        y=samples_df["kv_cache_mb"],
        mode="lines+markers",
        name="KV Cache",
        line=dict(color="blue", width=3),
    ))
    fig_linear.update_layout(
        title="KV Cache: Linear Growth Pattern",
        xaxis_title="Turn",
        yaxis_title="Memory (MB)",
        hovermode="x unified"
    )
    style_plotly_figure(fig_linear, height=320)
    st.plotly_chart(fig_linear, width='stretch')
    
    # Practical implications
    st.divider()
    st.caption("Practical Implications")
    
    col1, col2 = st.columns(2, gap="medium")
    
    with col1:
        st.markdown("""
        **Memory Pressure Points:**
        - **100 turns**: ~{:.1f}MB KV cache
        - **250 turns**: ~{:.1f}MB KV cache  
        - **500 turns**: ~{:.1f}MB KV cache
        
        **GPU Memory Requirements:**
        - Model weights: ~{:.1f}GB
        - Activations (batch=1): ~{:.1f}GB
        - KV Cache (500 turns): ~{:.1f}MB
        """.format(
            samples_df[samples_df["turn"] == 100]["kv_cache_mb"].iloc[0] if 100 in samples_df["turn"].values else 0,
            samples_df[samples_df["turn"] == 250]["kv_cache_mb"].iloc[0] if 250 in samples_df["turn"].values else 0,
            samples_df[samples_df["turn"] == 500]["kv_cache_mb"].iloc[0] if 500 in samples_df["turn"].values else 0,
            2.0 if "1B" in model else 8.0,
            1.0 if "1B" in model else 4.0,
            samples_df["kv_cache_mb"].iloc[-1]
        ))
    
    with col2:
        st.markdown("""
        **Optimization Strategies:**
        - **PagedAttention (vLLM)**: ~2-4x memory savings
        - **Shared Prefix (Preble)**: Reuse cached history across requests
        - **Quantization (INT4)**: 4x memory reduction
        - **Sliding Window**: Drop oldest tokens beyond threshold
        
        **Recommendation:**
        For sustained agent operation (100+ turns), use:
        - vLLM + PagedAttention + INT4 quantization
        """)
    
    # Comparison with different quantizations
    st.divider()
    st.caption("Quantization Impact on KV Cache")
    
    quant_comparison = []
    final_base_mb = base_samples_df["kv_cache_mb"].iloc[-1]
    for quant_name, quant_bytes in [("FP32", 4), ("FP16", 2), ("INT8", 1), ("INT4", 0.5)]:
        final_mb = final_base_mb * (quant_bytes / 2.0)
        quant_comparison.append({
            "Quantization": quant_name,
            "Final Cache (MB)": final_mb,
            "Speedup": 2.0 / quant_bytes,
            "Memory Savings": f"{(1 - quant_bytes/4)*100:.0f}%"
        })
    
    quant_df = pd.DataFrame(quant_comparison)
    fig_quant = px.bar(
        quant_df,
        x="Quantization",
        y="Final Cache (MB)",
        title="KV Cache Size by Quantization",
        color="Quantization",
        hover_data=["Speedup", "Memory Savings"]
    )
    style_plotly_figure(fig_quant, height=320)
    st.plotly_chart(fig_quant, width='stretch')
    
    # Detailed table
    st.divider()
    st.caption("Detailed KV Cache Samples")
    display_df = samples_df.copy()
    display_df["kv_cache_mb"] = display_df["kv_cache_mb"].apply(lambda x: f"{x:.3f}")
    st.dataframe(display_df, width='stretch', hide_index=True)


if __name__ == "__main__":
    render_kv_cache_visualization()
