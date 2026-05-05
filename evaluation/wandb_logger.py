"""Weights & Biases integration for experiment tracking.

Logs agent performance, inference benchmarks, and system metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
import os

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


@dataclass
class WBConfig:
    """Weights & Biases configuration."""
    enabled: bool = False
    project: str = "agentbench"
    entity: str | None = None
    tags: list[str] | None = None
    notes: str = ""
    job_type: str = "benchmark"


class WBLogger:
    """Wrapper around W&B for logging experiments."""
    
    def __init__(self, config: WBConfig | None = None) -> None:
        self.config = config or WBConfig()
        self.run = None
        self.enabled = self.config.enabled and HAS_WANDB
        
        if self.enabled:
            self._init_wandb()
    
    def _init_wandb(self) -> None:
        """Initialize Weights & Biases run."""
        if not HAS_WANDB:
            print("⚠️ Weights & Biases not installed. Install with: pip install wandb")
            self.enabled = False
            return
        
        try:
            self.run = wandb.init(
                project=self.config.project,
                entity=self.config.entity,
                tags=self.config.tags or [],
                notes=self.config.notes,
                job_type=self.config.job_type
            )
            print(f"📊 W&B logging enabled: {self.run.get_url()}")
        except Exception as e:
            print(f"⚠️ Failed to initialize W&B: {e}")
            self.enabled = False
    
    def log_agent_benchmark(
        self,
        agent_name: str,
        games_played: int,
        wins: int,
        losses: int,
        ties: int,
        win_rate: float,
        behavioral_fidelity: float,
        action_kl: float,
        avg_decision_ms: float
    ) -> None:
        """Log agent benchmark results."""
        if not self.enabled:
            return
        
        metrics = {
            "agent": agent_name,
            "games_played": games_played,
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "win_rate": win_rate,
            "behavioral_fidelity": behavioral_fidelity,
            "action_kl": action_kl,
            "avg_decision_ms": avg_decision_ms
        }
        
        self.run.log(metrics)
    
    def log_inference_benchmark(
        self,
        engine: str,
        quantization: str,
        prompt_tokens: int,
        output_tokens: int,
        ttft_ms: float,
        tpot_ms: float,
        total_latency_ms: float,
        kv_cache_mb: float,
        scheduling_overhead_ms: float = 0.0
    ) -> None:
        """Log inference engine benchmark."""
        if not self.enabled:
            return
        
        metrics = {
            "engine": engine,
            "quantization": quantization,
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "ttft_ms": ttft_ms,
            "tpot_ms": tpot_ms,
            "total_latency_ms": total_latency_ms,
            "kv_cache_mb": kv_cache_mb,
            "scheduling_overhead_ms": scheduling_overhead_ms,
            "throughput_decisions_per_hour": 3600000 / (total_latency_ms + 0.001)
        }
        
        self.run.log(metrics)
    
    def log_comparison_chart(
        self,
        data: dict[str, Any],
        chart_type: Literal["bar", "line", "scatter"] = "bar",
        title: str = "",
        x_label: str = "",
        y_label: str = ""
    ) -> None:
        """Log a comparison chart."""
        if not self.enabled:
            return
        
        try:
            # Convert data to W&B table format
            table_data = []
            for key, values in data.items():
                if isinstance(values, dict):
                    for subkey, value in values.items():
                        table_data.append([key, subkey, value])
                else:
                    table_data.append([key, values])
            
            # Create W&B chart
            chart_key = title.lower().replace(" ", "_") or "comparison"
            self.run.log({
                chart_key: wandb.plot_table(
                    wandb.Table(data=table_data),
                    {"x": 0, "y": 1},
                    string_fields={},
                    title=title
                )
            })
        except Exception as e:
            print(f"Failed to log chart: {e}")
    
    def log_config(self, config: dict[str, Any]) -> None:
        """Log experiment configuration."""
        if not self.enabled:
            return
        
        # Save config as W&B config
        for key, value in config.items():
            self.run.config[key] = value
    
    def log_summary_metrics(self, metrics: dict[str, Any]) -> None:
        """Log summary metrics for the run."""
        if not self.enabled:
            return
        
        for key, value in metrics.items():
            self.run.summary[key] = value
    
    def finish(self) -> None:
        """Finish the W&B run."""
        if self.enabled and self.run:
            self.run.finish()
            print("✅ W&B run finished")
    
    def __enter__(self) -> WBLogger:
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.finish()


# Global logger instance
_global_logger: WBLogger | None = None


def init_wandb(config: WBConfig | None = None) -> WBLogger:
    """Initialize global W&B logger."""
    global _global_logger
    _global_logger = WBLogger(config or WBConfig())
    return _global_logger


def get_logger() -> WBLogger:
    """Get global W&B logger."""
    global _global_logger
    if _global_logger is None:
        _global_logger = WBLogger()
    return _global_logger


def log_agent_result(
    agent_name: str,
    games_played: int,
    wins: int,
    losses: int,
    ties: int,
    win_rate: float,
    behavioral_fidelity: float,
    action_kl: float,
    avg_decision_ms: float
) -> None:
    """Log agent result to global logger."""
    logger = get_logger()
    logger.log_agent_benchmark(
        agent_name=agent_name,
        games_played=games_played,
        wins=wins,
        losses=losses,
        ties=ties,
        win_rate=win_rate,
        behavioral_fidelity=behavioral_fidelity,
        action_kl=action_kl,
        avg_decision_ms=avg_decision_ms
    )


def log_inference_result(
    engine: str,
    quantization: str,
    prompt_tokens: int,
    output_tokens: int,
    ttft_ms: float,
    tpot_ms: float,
    total_latency_ms: float,
    kv_cache_mb: float,
    scheduling_overhead_ms: float = 0.0
) -> None:
    """Log inference result to global logger."""
    logger = get_logger()
    logger.log_inference_benchmark(
        engine=engine,
        quantization=quantization,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        ttft_ms=ttft_ms,
        tpot_ms=tpot_ms,
        total_latency_ms=total_latency_ms,
        kv_cache_mb=kv_cache_mb,
        scheduling_overhead_ms=scheduling_overhead_ms
    )


# Example usage
if __name__ == "__main__":
    # Enable W&B logging
    config = WBConfig(
        enabled=True,
        project="agentbench",
        tags=["demo", "benchmark"],
        notes="Demo run of AgentBench logging"
    )
    
    with init_wandb(config) as logger:
        # Log agent result
        logger.log_agent_benchmark(
            agent_name="sft",
            games_played=100,
            wins=45,
            losses=30,
            ties=25,
            win_rate=0.45,
            behavioral_fidelity=0.75,
            action_kl=0.32,
            avg_decision_ms=12.5
        )
        
        # Log inference result
        logger.log_inference_benchmark(
            engine="vllm",
            quantization="fp16",
            prompt_tokens=128,
            output_tokens=8,
            ttft_ms=45.2,
            tpot_ms=3.1,
            total_latency_ms=68.5,
            kv_cache_mb=2.3,
            scheduling_overhead_ms=1.2
        )
        
        # Log config
        logger.log_config({
            "model": "llama-3.2-1b",
            "max_turns": 500,
            "agents": ["sft", "rl", "bcrl", "agentic"]
        })
        
        # Log summary
        logger.log_summary_metrics({
            "best_agent": "rl",
            "best_engine": "vllm",
            "total_games": 1000
        })
    
    print("✅ W&B logging demo complete!")
