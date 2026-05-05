"""Evaluation metrics for agent and systems performance."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class AgentScore:
    agent_name: str
    games_played: int
    wins: int
    losses: int
    ties: int
    win_rate: float
    behavioral_fidelity: float
    action_kl: float
    avg_decision_ms: float


@dataclass
class SummaryStats:
    mean: float
    std: float
    ci95_low: float
    ci95_high: float


@dataclass
class CounterfactualOutcome:
    baseline_agent: str
    clone_agent: str
    source_result: str
    baseline_win_rate: float
    clone_win_rate: float
    lift: float


def action_distribution(actions: list[int], n_actions: int = 6) -> np.ndarray:
    v = np.zeros(n_actions, dtype=np.float64)
    for a in actions:
        v[int(a)] += 1
    if v.sum() == 0:
        return v
    return v / v.sum()


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-9) -> float:
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float(np.sum(p * np.log(p / q)))


def action_perplexity(
    predicted_probs: list[np.ndarray],
    actual_actions: list[int],
) -> float:
    """Per-action perplexity of a model's predictions against actual human actions.

    Lower = model better predicts what the human will do.
    Ref: "Synthetic User Generation" (MDPI Information, 2025) — perplexity
    over action sequences as a complementary fidelity metric.

    Args:
        predicted_probs: model's P(action) distribution at each step
        actual_actions: ground-truth human action at each step
    """
    if not predicted_probs or not actual_actions:
        return float("inf")
    n = min(len(predicted_probs), len(actual_actions))
    log_probs = []
    for i in range(n):
        p = predicted_probs[i]
        a = actual_actions[i]
        prob = float(np.clip(p[a], 1e-9, 1.0))
        log_probs.append(np.log2(prob))
    avg_neg_log = -np.mean(log_probs)
    return float(2.0 ** avg_neg_log)


def summary_stats(values: list[float]) -> SummaryStats:
    if not values:
        return SummaryStats(mean=0.0, std=0.0, ci95_low=0.0, ci95_high=0.0)
    arr = np.array(values, dtype=np.float64)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    if len(arr) == 1:
        return SummaryStats(mean=mean, std=std, ci95_low=mean, ci95_high=mean)
    margin = 1.96 * (std / float(np.sqrt(len(arr))))
    return SummaryStats(
        mean=mean,
        std=std,
        ci95_low=mean - margin,
        ci95_high=mean + margin,
    )


def counterfactual_lift(
    *,
    baseline_agent: str,
    clone_agent: str,
    source_result: str,
    baseline_results: list[int],
    clone_results: list[int],
) -> CounterfactualOutcome:
    baseline_win_rate = float(np.mean([1 if result > 0 else 0 for result in baseline_results])) if baseline_results else 0.0
    clone_win_rate = float(np.mean([1 if result > 0 else 0 for result in clone_results])) if clone_results else 0.0
    return CounterfactualOutcome(
        baseline_agent=baseline_agent,
        clone_agent=clone_agent,
        source_result=source_result,
        baseline_win_rate=baseline_win_rate,
        clone_win_rate=clone_win_rate,
        lift=clone_win_rate - baseline_win_rate,
    )
