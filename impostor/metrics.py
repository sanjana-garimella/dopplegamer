"""Impostor Effect evaluation metrics.

Covers:
  - Human detection accuracy (Turing test results)
  - Cognitive bias transfer (recency, win-streak aggression, loss aversion)
  - Behavioral fidelity at varying data sizes (Experiment 3)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from environments.rps_plus import BEST_COUNTER, N_MOVES, Move


# ─────────────────────────────────────────────────────────── data classes

@dataclass
class DetectionResult:
    impostor_type: str
    detection_rate: float        # fraction of trials human correctly flagged as AI
    false_positive_rate: float   # fraction where human misidentified as AI
    mean_confidence: float       # average self-reported confidence (0–1)


@dataclass
class BiasProfile:
    recency_bias: float            # TVD between last-3 and full move distributions
    win_streak_aggression: float   # P(POWER | 2+ consecutive wins)
    loss_aversion: float           # P(RECHARGE | 2+ consecutive losses)


@dataclass
class DriftPoint:
    timestamp: str
    game_type: str
    drift_score: float
    win_rate: float


@dataclass
class CloneReport:
    player_id: str
    best_clone: str
    fidelity_score: float
    fool_rate: float
    top_habits: list[str]
    summary: str


@dataclass
class PairedTestResult:
    metric: str
    n_players: int
    mean_delta: float
    t_stat: float
    p_value: float


@dataclass
class PowerEstimate:
    effect_size: float
    alpha: float
    power: float
    estimated_n_per_group: int


@dataclass
class SurprisalPoint:
    turn: int
    action: int
    predicted_prob: float
    surprisal: float
    confidence: float
    entropy: float
    expected_action: int


@dataclass
class RetrainingTrigger:
    needs_retraining: bool
    recent_mean_drift: float
    threshold: float
    n_recent_snapshots: int
    rationale: str


# ─────────────────────────────────────────────────────── detection metrics

def detection_accuracy(
    impostor_type: str,
    verdicts: list[bool],
    confidences: list[float] | None = None,
) -> DetectionResult:
    """Compute human detection stats from a list of trial verdicts.

    Args:
        impostor_type: label for this Impostor variant
        verdicts: True if the human correctly identified the opponent as AI
        confidences: self-reported confidence per trial (0–1), optional
    """
    n = len(verdicts)
    if n == 0:
        return DetectionResult(impostor_type, 0.0, 0.0, 0.0)
    if confidences is None:
        confidences = [0.5] * n
    detected = sum(verdicts)
    # False positive: human said "human" (wrong) with high confidence
    fp = sum(1 for v, c in zip(verdicts, confidences) if not v and c >= 0.7)
    return DetectionResult(
        impostor_type=impostor_type,
        detection_rate=detected / n,
        false_positive_rate=fp / n,
        mean_confidence=float(np.mean(confidences)),
    )


# ─────────────────────────────────────────────────────── bias metrics

def recency_bias(move_sequence: list[int], window: int = 3) -> float:
    """TVD between the last `window` moves and the full move distribution.

    Higher → player over-weights recent history.
    """
    if len(move_sequence) < window + 1:
        return 0.0

    full = np.zeros(N_MOVES)
    for m in move_sequence:
        full[m] += 1
    full /= full.sum()

    recent = np.zeros(N_MOVES)
    for m in move_sequence[-window:]:
        recent[m] += 1
    if recent.sum() > 0:
        recent /= recent.sum()

    return float(0.5 * np.abs(recent - full).sum())


def win_streak_aggression(
    move_sequence: list[int],
    outcomes: list[int],
    power_move: int | None = None,
    streak: int = 2,
) -> float:
    """P(POWER | last `streak` outcomes are wins)."""
    power = power_move if power_move is not None else int(Move.POWER)
    if len(move_sequence) < streak + 1 or len(outcomes) < streak:
        return 0.0
    streak_moves = [
        move_sequence[i]
        for i in range(streak, min(len(move_sequence), len(outcomes)))
        if all(outcomes[j] > 0 for j in range(i - streak, i))
    ]
    if not streak_moves:
        return 0.0
    return sum(1 for m in streak_moves if m == power) / len(streak_moves)


def loss_aversion(
    move_sequence: list[int],
    outcomes: list[int],
    recharge_move: int | None = None,
    streak: int = 2,
) -> float:
    """P(RECHARGE | last `streak` outcomes are losses)."""
    recharge = recharge_move if recharge_move is not None else int(Move.RECHARGE)
    if len(move_sequence) < streak + 1 or len(outcomes) < streak:
        return 0.0
    after_loss = [
        move_sequence[i]
        for i in range(streak, min(len(move_sequence), len(outcomes)))
        if all(outcomes[j] < 0 for j in range(i - streak, i))
    ]
    if not after_loss:
        return 0.0
    return sum(1 for m in after_loss if m == recharge) / len(after_loss)


def bias_profile(move_sequence: list[int], outcomes: list[int]) -> BiasProfile:
    return BiasProfile(
        recency_bias=recency_bias(move_sequence),
        win_streak_aggression=win_streak_aggression(move_sequence, outcomes),
        loss_aversion=loss_aversion(move_sequence, outcomes),
    )


def bias_correlation(human: BiasProfile, impostor: BiasProfile) -> float:
    """Pearson correlation between two bias profiles. Range [-1, 1].

    A score close to 1 means the Impostor reproduced the human's cognitive biases.
    """
    h = np.array([human.recency_bias, human.win_streak_aggression, human.loss_aversion])
    a = np.array([impostor.recency_bias, impostor.win_streak_aggression, impostor.loss_aversion])
    if h.std() < 1e-9 or a.std() < 1e-9:
        return 1.0 if np.allclose(h, a) else 0.0
    return float(np.corrcoef(h, a)[0, 1])


# ─────────────────────────────────────── CoBBLEr-style per-decision bias rates
# (Koo et al., ACL Findings 2024)
# Instead of one aggregate number, measure what *fraction* of individual decisions
# exhibit each bias — gives per-decision granularity.

@dataclass
class BiasRates:
    """Per-decision bias rates (fraction of decisions exhibiting each bias)."""
    recency_rate: float        # fraction of decisions matching last-3 opponent move counter
    aggression_rate: float     # fraction of POWER plays after win streaks vs overall POWER rate
    aversion_rate: float       # fraction of RECHARGE plays after loss streaks vs overall rate
    compounding: float         # fraction where 2+ biases fire simultaneously


def per_decision_bias_rates(
    move_sequence: list[int],
    outcomes: list[int],
    opp_sequence: list[int] | None = None,
    streak: int = 2,
    window: int = 3,
) -> BiasRates:
    """CoBBLEr-style metric: what fraction of decisions show each bias?

    Ref: Koo et al., "Benchmarking Cognitive Biases in LLMs as Evaluators" (ACL 2024)
    """
    n = min(len(move_sequence), len(outcomes))
    if n < streak + 1:
        return BiasRates(0.0, 0.0, 0.0, 0.0)

    power = int(Move.POWER)
    recharge = int(Move.RECHARGE)
    counter_map = {int(k): int(v) for k, v in BEST_COUNTER.items()}

    recency_hits = 0
    aggression_hits = 0
    aversion_hits = 0
    compound_hits = 0
    evaluated = 0

    for i in range(streak, n):
        evaluated += 1
        flags = 0
        move = move_sequence[i]

        # Recency: did agent play the counter to the most frequent of last `window` opp moves?
        if opp_sequence and i >= window:
            recent_opp = opp_sequence[max(0, i - window):i]
            if recent_opp:
                from collections import Counter
                mode_opp = Counter(recent_opp).most_common(1)[0][0]
                if move == counter_map.get(mode_opp):
                    recency_hits += 1
                    flags += 1

        # Aggression after win streak
        if all(outcomes[j] > 0 for j in range(i - streak, i)):
            if move == power:
                aggression_hits += 1
                flags += 1

        # Aversion after loss streak
        if all(outcomes[j] < 0 for j in range(i - streak, i)):
            if move == recharge:
                aversion_hits += 1
                flags += 1

        if flags >= 2:
            compound_hits += 1

    d = max(evaluated, 1)
    return BiasRates(
        recency_rate=recency_hits / d,
        aggression_rate=aggression_hits / d,
        aversion_rate=aversion_hits / d,
        compounding=compound_hits / d,
    )


# ─────────────────────────── action distribution mode analysis
# (Shafiullah et al., "Behavior Transformers", NeurIPS 2022)
# Instead of single-mode fidelity (TVD), detect whether the agent preserves
# the player's *multimodal* action patterns — e.g., "usually ROCK, sometimes POWER".

def action_mode_divergence(
    human_sequences: list[list[int]],
    agent_sequences: list[list[int]],
    n_modes: int = 3,
) -> float:
    """Measure how well the agent preserves the human's top-K action modes.

    Ref: BeT (NeurIPS 2022) k-mode behavioral cloning.
    Returns mean absolute error between human and agent top-K mode frequencies.
    Lower = better preservation of multimodal behavior.
    """
    from evaluation.metrics import action_distribution

    human_moves = [m for s in human_sequences for m in s]
    agent_moves = [m for s in agent_sequences for m in s]

    p = action_distribution(human_moves)
    q = action_distribution(agent_moves)

    # Sort by human frequency to identify top-K modes
    top_k = np.argsort(p)[::-1][:n_modes]
    return float(np.mean(np.abs(p[top_k] - q[top_k])))


def behavioral_drift(snapshot_signatures: list[dict[str, Any]]) -> list[DriftPoint]:
    points: list[DriftPoint] = []
    for snap in snapshot_signatures:
        points.append(
            DriftPoint(
                timestamp=str(snap.get("created_at", "")),
                game_type=str(snap.get("game_type", "Unknown")),
                drift_score=float(snap.get("drift_score", 0.0)),
                win_rate=float(snap.get("win_rate", 0.0)),
            )
        )
    return points


def susceptibility_score(detection_rates: list[float]) -> float:
    if not detection_rates:
        return 0.0
    rates = np.array(detection_rates, dtype=np.float64)
    return float(np.clip(1.0 - np.nanmean(rates), 0.0, 1.0))


def paired_significance(
    baseline_values: list[float],
    clone_values: list[float],
    *,
    metric: str,
) -> PairedTestResult:
    n = min(len(baseline_values), len(clone_values))
    if n == 0:
        return PairedTestResult(metric=metric, n_players=0, mean_delta=0.0, t_stat=0.0, p_value=1.0)
    base = np.array(baseline_values[:n], dtype=np.float64)
    clone = np.array(clone_values[:n], dtype=np.float64)
    diffs = clone - base
    mean_delta = float(np.mean(diffs))
    if n == 1 or np.std(diffs, ddof=1) < 1e-9:
        return PairedTestResult(metric=metric, n_players=n, mean_delta=mean_delta, t_stat=0.0, p_value=1.0)
    try:
        from scipy.stats import ttest_rel
        t_stat, p_value = ttest_rel(clone, base)
        return PairedTestResult(metric=metric, n_players=n, mean_delta=mean_delta, t_stat=float(t_stat), p_value=float(p_value))
    except Exception:
        std = float(np.std(diffs, ddof=1))
        t_stat = mean_delta / (std / np.sqrt(n)) if std > 0 else 0.0
        return PairedTestResult(metric=metric, n_players=n, mean_delta=mean_delta, t_stat=float(t_stat), p_value=1.0)


def estimate_required_sample_size(
    *,
    effect_size: float,
    alpha: float = 0.05,
    power: float = 0.8,
) -> PowerEstimate:
    effect = max(abs(float(effect_size)), 1e-6)
    z_alpha = 1.96 if alpha == 0.05 else 1.64
    z_beta = 0.84 if power == 0.8 else 1.28
    n_per_group = int(np.ceil(2 * ((z_alpha + z_beta) / effect) ** 2))
    return PowerEstimate(
        effect_size=effect,
        alpha=alpha,
        power=power,
        estimated_n_per_group=max(n_per_group, 2),
    )


def standardized_detection_prompt() -> dict[str, str]:
    return {
        "title": "Identity Check",
        "question": "Did this opponent feel human-like?",
        "scale_label": "Confidence (0 = definitely AI, 1 = definitely human)",
        "cta": "Submit Detection Judgment",
    }


def confidence_entropy(probs: np.ndarray) -> dict[str, float]:
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs[probs > 0]
    if probs.size == 0:
        return {"confidence": 0.0, "entropy": 0.0}
    entropy = float(-(probs * np.log2(probs)).sum())
    max_entropy = float(np.log2(max(len(probs), 1)))
    confidence = float(1.0 - (entropy / max_entropy)) if max_entropy > 0 else 0.0
    return {"confidence": float(np.clip(confidence, 0.0, 1.0)), "entropy": entropy}


def move_surprisal(predicted_prob: float, eps: float = 1e-9) -> float:
    prob = float(np.clip(predicted_prob, eps, 1.0))
    return float(-np.log2(prob))


def summarize_surprisal_history(history: list[dict[str, Any]]) -> dict[str, float | int]:
    if not history:
        return {
            "n_turns": 0,
            "mean_surprisal": 0.0,
            "max_surprisal": 0.0,
            "mean_confidence": 0.0,
            "high_surprisal_turns": 0,
        }
    surprisals = np.array([float(item.get("surprisal", 0.0)) for item in history], dtype=np.float64)
    confidences = np.array([float(item.get("confidence", 0.0)) for item in history], dtype=np.float64)
    return {
        "n_turns": int(len(history)),
        "mean_surprisal": float(np.mean(surprisals)),
        "max_surprisal": float(np.max(surprisals)),
        "mean_confidence": float(np.mean(confidences)),
        "high_surprisal_turns": int(np.sum(surprisals >= 2.0)),
    }


def retraining_trigger(
    snapshot_signatures: list[dict[str, Any]],
    *,
    recent_window: int = 3,
    threshold: float = 0.18,
) -> RetrainingTrigger:
    if not snapshot_signatures:
        return RetrainingTrigger(False, 0.0, threshold, 0, "No drift snapshots recorded yet.")
    recent = snapshot_signatures[-recent_window:]
    drifts = np.array([float(item.get("drift_score", 0.0)) for item in recent], dtype=np.float64)
    mean_drift = float(np.mean(drifts)) if drifts.size else 0.0
    needs = mean_drift >= threshold and len(recent) >= min(recent_window, 2)
    rationale = (
        f"Recent mean drift {mean_drift:.2f} exceeds threshold {threshold:.2f}."
        if needs
        else f"Recent mean drift {mean_drift:.2f} is below threshold {threshold:.2f}."
    )
    return RetrainingTrigger(
        needs_retraining=needs,
        recent_mean_drift=mean_drift,
        threshold=threshold,
        n_recent_snapshots=len(recent),
        rationale=rationale,
    )


def narrative_summary(
    *,
    games_played: int,
    fidelity_scores: list[float],
    drift_scores: list[float],
    fooled_count: int,
) -> str:
    avg_fidelity = float(np.mean(fidelity_scores)) if fidelity_scores else 0.0
    peak_drift = float(np.max(drift_scores)) if drift_scores else 0.0
    return (
        f"Across {games_played} games, clone fidelity averaged {avg_fidelity:.0%}. "
        f"Peak behavioral drift reached {peak_drift:.2f}, and the clone fooled the subject {fooled_count} time(s)."
    )


def build_clone_report(
    player_id: str,
    clone_rows: list[dict[str, Any]],
    top_habits: list[str],
) -> CloneReport:
    if not clone_rows:
        return CloneReport(
            player_id=player_id,
            best_clone="Not trained",
            fidelity_score=0.0,
            fool_rate=0.0,
            top_habits=top_habits,
            summary="No trained clone results yet.",
        )
    ordered = sorted(
        clone_rows,
        key=lambda row: (
            float(row.get("fidelity_score") or 0.0),
            float(row.get("fool_rate") or 0.0),
        ),
        reverse=True,
    )
    best = ordered[0]
    fidelity = float(best.get("fidelity_score") or 0.0)
    fool_rate = float(best.get("fool_rate") or 0.0)
    summary = (
        f"Top clone `{best.get('impostor_type', 'unknown')}` reached {fidelity:.0%} fidelity "
        f"and fooled players {fool_rate:.0%} of the time."
    )
    return CloneReport(
        player_id=player_id,
        best_clone=str(best.get("impostor_type", "unknown")),
        fidelity_score=fidelity,
        fool_rate=fool_rate,
        top_habits=top_habits,
        summary=summary,
    )


# ─────────────────────────────────────────────────── data regime helper

def fidelity_at_data_sizes(
    train_sequences: list[list[int]],
    eval_sequences: list[list[int]],
    data_sizes: list[int] | None = None,
) -> dict[int, float]:
    """Measure behavioral fidelity (1 - TVD) at each training data size.

    Trains an NGramImpostor on `data_sizes` moves and compares its move
    distribution to the held-out evaluation sequences.

    Returns:
        {actual_training_moves: fidelity_score}
    """
    from agents.impostor.ngram import NGramImpostor
    from evaluation.metrics import action_distribution

    if data_sizes is None:
        data_sizes = [20, 50, 100, 200, 500]

    all_eval = [m for seq in eval_sequences for m in seq]
    p_human = action_distribution(all_eval)

    results: dict[int, float] = {}
    for target in data_sizes:
        agent = NGramImpostor(n=2)
        collected = 0
        for seq in train_sequences:
            if collected >= target:
                break
            take = min(len(seq), target - collected)
            agent.train(seq[:take])
            collected += take

        if collected == 0:
            results[0] = 0.0
            continue

        # Simulate the agent on eval sequences
        pred_moves: list[int] = []
        for seq in eval_sequences:
            sim = NGramImpostor(n=2)
            sim._transitions = agent._transitions
            for m in seq:
                action = sim.act(np.zeros(3), {"legal_moves": list(range(N_MOVES))})
                pred_moves.append(action)
                sim.observe(m, 0, 0)

        q_pred = action_distribution(pred_moves) if pred_moves else p_human
        tvd = float(0.5 * np.abs(p_human - q_pred).sum())
        results[collected] = max(0.0, 1.0 - tvd)

    return results
