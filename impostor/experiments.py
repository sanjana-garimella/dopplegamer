"""Impostor Effect experiment suite — 5 experiments from the research design.

Exp 1  — Behavioral fidelity comparison (NGram vs LSTM vs BC+RL)
Exp 2  — Human detection accuracy (simulated Turing test)
Exp 3  — Training data regime (minimum viable personalization)
Exp 4  — Cognitive bias transfer
Exp 5  — LLM Impostor vs LSTM Impostor
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import uuid
from typing import Any

import numpy as np

from agents.impostor.lstm import LSTMImpostor
from agents.impostor.ngram import NGramImpostor
from agents.impostor.trainer import ImpostorTrainer
from agents.rl.agent import BCRLAgent
from data.features import (
    CANONICAL_BASELINE_BATTERY,
    CONTROLLED_REARING_SLICES,
    apply_session_slice,
    session_ordered_sequences,
    session_ordered_split,
)
from data.schemas import connect, init_extended_db
from environments.rps_plus import RPSPlusEnv, N_MOVES
from evaluation.metrics import action_distribution, counterfactual_lift, kl_divergence
from impostor.metrics import (
    bias_profile,
    bias_correlation,
    detection_accuracy,
    fidelity_at_data_sizes,
)


# ─────────────────────────────────────────────────────────── shared helpers

@dataclass
class FidelityResult:
    agent_type: str
    player_id: str
    kl_divergence: float
    tvd: float
    fidelity_score: float


@dataclass
class ExperimentReport:
    experiment: str
    player_id: str
    results: list[dict]
    summary: str


def canonical_baseline_battery() -> list[str]:
    return list(CANONICAL_BASELINE_BATTERY)


def persist_dataset_slice(
    db_path: str,
    *,
    player_id: str,
    game_type: str,
    slice_name: str,
    sequences: list[dict[str, object]],
    filter_config: dict[str, Any] | None = None,
) -> str:
    init_extended_db(db_path)
    slice_id = uuid.uuid4().hex[:12]
    conn = connect(db_path)
    try:
        conn.execute(
            """INSERT INTO dataset_slices
               (slice_id, player_id, game_type, slice_name, filter_config_json, n_sessions, n_rounds, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                slice_id,
                player_id,
                game_type,
                slice_name,
                json.dumps(filter_config or {"slice_name": slice_name}),
                len(sequences),
                int(sum(len(seq.get("moves", [])) for seq in sequences)),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()
    finally:
        conn.close()
    return slice_id


def _play_episode(
    agent,
    env: RPSPlusEnv,
    seed: int = 0,
) -> tuple[list[int], list[int]]:
    """Run a full episode. Returns (agent_moves, outcomes)."""
    obs, info = env.reset(seed=seed)
    moves: list[int] = []
    outcomes: list[int] = []
    done = False
    while not done:
        action = int(agent.act(obs, info))
        legal = info.get("legal_moves") or []
        if legal and action not in legal:
            action = int(legal[0])
        moves.append(action)
        obs, reward, terminated, truncated, info = env.step(action)
        outcomes.append(int(np.sign(reward)))
        if env.state.history:
            last = env.state.history[-1]
            observe_fn = getattr(agent, "observe", None)
            if callable(observe_fn):
                observe_fn(int(last.agent_move), int(last.opponent_move), int(last.outcome))
        done = terminated or truncated
    return moves, outcomes


def _fidelity(p: np.ndarray, q: np.ndarray) -> tuple[float, float, float]:
    kl = kl_divergence(p, q)
    tvd = float(0.5 * np.abs(p - q).sum())
    return kl, tvd, max(0.0, 1.0 - tvd)


# ─────────────────────────────────────────────────────────── Experiment 1

def experiment_1_fidelity_comparison(
    player_sequences: list[list[int]],
    player_id: str = "player_0",
    rounds: int = 50,
) -> ExperimentReport:
    """Compare NGram, LSTM, and BC+RL Impostors on behavioral fidelity."""
    split = max(1, int(len(player_sequences) * 0.8))
    train_seqs = player_sequences[:split] or player_sequences
    eval_seqs  = player_sequences[split:] or player_sequences[:1]

    all_eval = [m for seq in eval_seqs for m in seq]
    p_human = action_distribution(all_eval)

    env = RPSPlusEnv(max_turns=rounds, seed=42)
    results = []

    # NGram BC
    ngram = NGramImpostor(n=2)
    for seq in train_seqs:
        ngram.train(seq)
    ngram_moves, _ = _play_episode(ngram, env, seed=1)
    kl, tvd, fid = _fidelity(p_human, action_distribution(ngram_moves))
    results.append(asdict(FidelityResult("ngram_bc", player_id, kl, tvd, fid)))

    # LSTM BC
    lstm = LSTMImpostor()
    lstm.train(train_seqs, epochs=5)
    lstm_moves, _ = _play_episode(lstm, env, seed=2)
    kl, tvd, fid = _fidelity(p_human, action_distribution(lstm_moves))
    results.append(asdict(FidelityResult("lstm_bc", player_id, kl, tvd, fid)))

    # BC+RL
    bcrl = BCRLAgent(seed=7)
    bcrl_moves, _ = _play_episode(bcrl, env, seed=3)
    kl, tvd, fid = _fidelity(p_human, action_distribution(bcrl_moves))
    results.append(asdict(FidelityResult("bc_rl", player_id, kl, tvd, fid)))

    best = min(results, key=lambda r: r["kl_divergence"])["agent_type"]
    return ExperimentReport(
        experiment="exp1_fidelity_comparison",
        player_id=player_id,
        results=results,
        summary=f"Best Impostor for {player_id}: {best}",
    )


# ─────────────────────────────────────────────────────────── Experiment 2

def experiment_2_turing_test(
    player_id: str = "player_0",
    n_trials: int = 20,
    detection_rate_ngram: float | None = None,
    detection_rate_bcrl: float | None = None,
) -> ExperimentReport:
    """Simulate human detection accuracy.

    In a live deployment, verdicts come from real participants. Here we accept
    pre-collected verdicts or generate synthetic baselines for unit testing.
    """
    rng = np.random.default_rng(42)

    def _simulate(impostor_type: str, base_rate: float) -> dict:
        verdicts = [bool(rng.random() < base_rate) for _ in range(n_trials)]
        confidences = [float(rng.uniform(0.5, 1.0)) for _ in range(n_trials)]
        r = detection_accuracy(impostor_type, verdicts, confidences)
        return {
            "impostor_type": r.impostor_type,
            "detection_rate": r.detection_rate,
            "false_positive_rate": r.false_positive_rate,
            "mean_confidence": r.mean_confidence,
        }

    results = [
        _simulate("ngram_bc",  detection_rate_ngram or 0.55),
        _simulate("bc_rl",     detection_rate_bcrl  or 0.45),
    ]
    hardest = min(results, key=lambda r: r["detection_rate"])["impostor_type"]
    return ExperimentReport(
        experiment="exp2_turing_test",
        player_id=player_id,
        results=results,
        summary=(
            f"Most convincing Impostor: {hardest} "
            f"(detected {min(r['detection_rate'] for r in results):.0%} of the time)"
        ),
    )


# ─────────────────────────────────────────────────────────── Experiment 3

def experiment_3_data_regime(
    player_sequences: list[list[int]],
    player_id: str = "player_0",
    data_sizes: list[int] | None = None,
) -> ExperimentReport:
    """Fidelity vs training data size — find minimum viable personalization."""
    if data_sizes is None:
        data_sizes = [20, 50, 100, 200, 500]

    split = max(1, int(len(player_sequences) * 0.8))
    train_seqs = player_sequences[:split] or player_sequences
    eval_seqs  = player_sequences[split:] or player_sequences[:1]

    by_size = fidelity_at_data_sizes(train_seqs, eval_seqs, data_sizes)
    results = [{"data_size": k, "fidelity": v} for k, v in sorted(by_size.items())]

    threshold = next(
        (k for k, v in sorted(by_size.items()) if v >= 0.70), None
    )
    summary = (
        f"Minimum viable personalization: {threshold} rounds (fidelity ≥ 0.70)"
        if threshold else "fidelity < 0.70 across all tested data sizes"
    )
    return ExperimentReport(
        experiment="exp3_data_regime",
        player_id=player_id,
        results=results,
        summary=summary,
    )


# ─────────────────────────────────────────────────────────── Experiment 4

def experiment_4_bias_transfer(
    player_sequences: list[list[int]],
    player_outcomes: list[list[int]],
    player_id: str = "player_0",
    sim_rounds: int = 100,
) -> ExperimentReport:
    """Measure cognitive bias transfer from human player to NGram Impostor."""
    all_moves    = [m for seq in player_sequences for m in seq]
    all_outcomes = [o for seq in player_outcomes for o in seq]

    human_bias = bias_profile(all_moves, all_outcomes)

    ngram = NGramImpostor(n=2)
    for seq in player_sequences:
        ngram.train(seq)

    env = RPSPlusEnv(max_turns=sim_rounds, seed=99)
    imp_moves, imp_outcomes = _play_episode(ngram, env, seed=10)
    imp_bias = bias_profile(imp_moves, imp_outcomes)

    corr = bias_correlation(human_bias, imp_bias)

    results = [
        {
            "bias": "recency",
            "human": round(human_bias.recency_bias, 4),
            "impostor": round(imp_bias.recency_bias, 4),
        },
        {
            "bias": "win_streak_aggression",
            "human": round(human_bias.win_streak_aggression, 4),
            "impostor": round(imp_bias.win_streak_aggression, 4),
        },
        {
            "bias": "loss_aversion",
            "human": round(human_bias.loss_aversion, 4),
            "impostor": round(imp_bias.loss_aversion, 4),
        },
    ]
    return ExperimentReport(
        experiment="exp4_bias_transfer",
        player_id=player_id,
        results=results,
        summary=f"Bias correlation between {player_id} and NGram Impostor: {corr:.3f}",
    )


# ─────────────────────────────────────────────────────────── Experiment 5

def experiment_5_llm_vs_lstm(
    player_sequences: list[list[int]],
    player_id: str = "player_0",
    rounds: int = 50,
) -> ExperimentReport:
    """Compare LSTM Impostor vs LLM (ReAct) on behavioral fidelity."""
    split = max(1, int(len(player_sequences) * 0.8))
    train_seqs = player_sequences[:split] or player_sequences
    eval_seqs  = player_sequences[split:] or player_sequences[:1]

    all_eval = [m for seq in eval_seqs for m in seq]
    p_human = action_distribution(all_eval)
    env = RPSPlusEnv(max_turns=rounds, seed=55)

    # LSTM
    lstm = LSTMImpostor()
    lstm.train(train_seqs, epochs=5)
    lstm_moves, _ = _play_episode(lstm, env, seed=5)
    lstm_kl, lstm_tvd, lstm_fid = _fidelity(p_human, action_distribution(lstm_moves))

    # LLM (ReAct with fallback LLM — no API key needed)
    from agents.agentic.react_agent import ReActAgent
    llm_agent = ReActAgent()
    llm_moves, _ = _play_episode(llm_agent, env, seed=6)
    llm_kl, llm_tvd, llm_fid = _fidelity(p_human, action_distribution(llm_moves))

    results = [
        {"model": "lstm",      "kl_divergence": round(lstm_kl, 4), "fidelity": round(lstm_fid, 4)},
        {"model": "llm_react", "kl_divergence": round(llm_kl, 4),  "fidelity": round(llm_fid, 4)},
    ]
    winner = "lstm" if lstm_kl < llm_kl else "llm_react"
    return ExperimentReport(
        experiment="exp5_llm_vs_lstm",
        player_id=player_id,
        results=results,
        summary=f"Lower KL = better fidelity. Winner: {winner}",
    )


# ─────────────────────────────────────────────────────────── convenience runner

def run_all_experiments(
    player_sequences: list[list[int]],
    player_outcomes: list[list[int]] | None = None,
    player_id: str = "player_0",
) -> dict[str, ExperimentReport]:
    """Run all five Impostor Effect experiments for one player."""
    if player_outcomes is None:
        player_outcomes = [[0] * len(s) for s in player_sequences]
    return {
        "exp1": experiment_1_fidelity_comparison(player_sequences, player_id),
        "exp2": experiment_2_turing_test(player_id),
        "exp3": experiment_3_data_regime(player_sequences, player_id),
        "exp4": experiment_4_bias_transfer(player_sequences, player_outcomes, player_id),
        "exp5": experiment_5_llm_vs_lstm(player_sequences, player_id),
    }


def experiment_session_ordered_generalization(
    db_path: str,
    player_id: str,
    *,
    game_type: str = "RPS+",
) -> ExperimentReport:
    train_ctx, eval_ctx = session_ordered_split(db_path, player_id, game_type=game_type)
    if not train_ctx or not eval_ctx:
        return ExperimentReport(
            experiment="exp_session_generalization",
            player_id=player_id,
            results=[],
            summary="Not enough session-ordered data yet.",
        )
    train_sequences = [list(seq["moves"]) for seq in train_ctx]
    eval_sequences = [list(seq["moves"]) for seq in eval_ctx]
    trainer = ImpostorTrainer(db_path)
    ngram, _ = trainer.train_ngram(player_id)
    lstm, _ = trainer.train_lstm(player_id, epochs=5, save=False)
    mixture, _ = trainer.train_mixture(player_id)
    eval_flat = [m for seq in eval_sequences for m in seq]
    p_human = action_distribution(eval_flat)
    results = []
    for name, agent in [("ngram", ngram), ("lstm", lstm), ("mixture", mixture)]:
        if hasattr(agent, "predict_proba"):
            pred_moves = []
            for seq in eval_sequences:
                h: list[int] = []
                for m in seq:
                    probs = agent.predict_proba(history=h, legal=list(range(N_MOVES)))
                    pred_moves.append(int(np.argmax(probs)))
                    h.append(m)
            kl, tvd, fid = _fidelity(p_human, action_distribution(pred_moves))
            results.append(
                {
                    "agent_type": name,
                    "train_sessions": len(train_sequences),
                    "eval_sessions": len(eval_sequences),
                    "kl_divergence": kl,
                    "tvd": tvd,
                    "fidelity_score": fid,
                }
            )
    return ExperimentReport(
        experiment="exp_session_generalization",
        player_id=player_id,
        results=results,
        summary=f"Session-ordered generalization on {len(train_sequences)} train sessions and {len(eval_sequences)} later sessions.",
    )


def experiment_controlled_rearing(
    db_path: str,
    player_id: str,
    *,
    game_type: str = "RPS+",
) -> ExperimentReport:
    trainer = ImpostorTrainer(db_path)
    base_sequences = session_ordered_sequences(db_path, player_id, game_type=game_type)
    if not base_sequences:
        return ExperimentReport("exp_controlled_rearing", player_id, [], "No sequences available for controlled rearing.")
    eval_sequences = base_sequences[-max(1, len(base_sequences) // 3):]
    p_human = action_distribution([m for seq in eval_sequences for m in seq["moves"]])
    results: list[dict[str, Any]] = []
    for slice_name in CONTROLLED_REARING_SLICES:
        subset = apply_session_slice(base_sequences, slice_name, dominant_game_type=game_type)
        persist_dataset_slice(
            db_path,
            player_id=player_id,
            game_type=game_type,
            slice_name=slice_name,
            sequences=subset,
            filter_config={"slice_name": slice_name, "kind": "controlled_rearing"},
        )
        for clone_type in ("ngram", "lstm", "mixture"):
            if clone_type == "ngram":
                agent, _ = trainer.train_ngram(player_id, game_type=game_type, slice_name=slice_name)
            elif clone_type == "lstm":
                agent, _ = trainer.train_lstm(player_id, epochs=5, save=False, game_type=game_type, slice_name=slice_name)
            else:
                agent, _ = trainer.train_mixture(player_id, game_type=game_type, slice_name=slice_name)
            pred_moves = []
            for seq in eval_sequences:
                history: list[int] = []
                for move in seq["moves"]:
                    probs = agent.predict_proba(history=history, legal=list(range(N_MOVES)))
                    pred_moves.append(int(np.argmax(probs)))
                    history.append(move)
            kl, tvd, fid = _fidelity(p_human, action_distribution(pred_moves))
            results.append(
                {
                    "slice_name": slice_name,
                    "clone_type": clone_type,
                    "n_sessions": len(subset),
                    "n_rounds": int(sum(len(seq["moves"]) for seq in subset)),
                    "kl_divergence": kl,
                    "tvd": tvd,
                    "fidelity_score": fid,
                }
            )
    return ExperimentReport(
        experiment="exp_controlled_rearing",
        player_id=player_id,
        results=results,
        summary="Evaluated clone fidelity under early/recent/win/loss/single-family training slices.",
    )


def experiment_bias_interventions(
    db_path: str,
    player_id: str,
    *,
    game_type: str = "RPS+",
) -> ExperimentReport:
    trainer = ImpostorTrainer(db_path)
    _, eval_ctx = session_ordered_split(db_path, player_id, game_type=game_type)
    if not eval_ctx:
        return ExperimentReport("exp_bias_interventions", player_id, [], "Not enough held-out data for intervention study.")
    p_human = action_distribution([m for seq in eval_ctx for m in seq["moves"]])
    configs = [
        {"intervention": "baseline", "suppress_recency": False, "suppress_recharge": False, "include_opponent_conditioning": True},
        {"intervention": "no_recency", "suppress_recency": True, "suppress_recharge": False, "include_opponent_conditioning": True},
        {"intervention": "no_recharge", "suppress_recency": False, "suppress_recharge": True, "include_opponent_conditioning": True},
        {"intervention": "no_opponent_conditioning", "suppress_recency": False, "suppress_recharge": False, "include_opponent_conditioning": False},
    ]
    results: list[dict[str, Any]] = []
    for cfg in configs:
        trained = trainer.train_with_interventions(
            player_id,
            game_type=game_type,
            suppress_recency=cfg["suppress_recency"],
            suppress_recharge=cfg["suppress_recharge"],
            include_opponent_conditioning=cfg["include_opponent_conditioning"],
        )
        mixture_agent = trained["mixture"][0]
        pred_moves = []
        for seq in eval_ctx:
            history: list[int] = []
            for move in seq["moves"]:
                probs = mixture_agent.predict_proba(history=history, legal=list(range(N_MOVES)))
                pred_moves.append(int(np.argmax(probs)))
                history.append(move)
        kl, tvd, fid = _fidelity(p_human, action_distribution(pred_moves))
        results.append(
            {
                "intervention": cfg["intervention"],
                "kl_divergence": kl,
                "tvd": tvd,
                "fidelity_score": fid,
                "fool_rate_proxy": max(0.0, fid - kl),
            }
        )
    return ExperimentReport(
        experiment="exp_bias_interventions",
        player_id=player_id,
        results=results,
        summary="Measured how fidelity drops when recency, recharge, or opponent-conditioning signals are suppressed.",
    )


def create_blind_match_schedule(
    *,
    player_id: str,
    game_type: str = "RPS+",
    conditions: list[str] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    condition_list = list(conditions or ["human_baseline", "heuristic", "ngram", "lstm"])
    rng.shuffle(condition_list)
    schedule = [
        {"blind_label": f"Opponent {idx + 1}", "condition": condition}
        for idx, condition in enumerate(condition_list)
    ]
    return {
        "player_id": player_id,
        "game_type": game_type,
        "schedule": schedule,
    }


def persist_blind_match_schedule(
    *,
    db_path: str,
    player_id: str,
    game_type: str = "RPS+",
    conditions: list[str] | None = None,
    seed: int = 42,
) -> str:
    init_extended_db(db_path)
    payload = create_blind_match_schedule(player_id=player_id, game_type=game_type, conditions=conditions, seed=seed)
    block_id = uuid.uuid4().hex[:12]
    conn = connect(db_path)
    try:
        conn.execute(
            """INSERT INTO blind_study_blocks
               (block_id, player_id, game_type, schedule_json, current_index, revealed, created_at)
               VALUES (?, ?, ?, ?, 0, 0, ?)""",
            (
                block_id,
                player_id,
                game_type,
                json.dumps(payload["schedule"]),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()
    finally:
        conn.close()
    return block_id


def run_paper_battery(db_path: str, player_id: str) -> dict[str, ExperimentReport]:
    trainer = ImpostorTrainer(db_path)
    sequences = trainer.load_player_sequences(player_id)
    outcomes = trainer.load_player_outcomes(player_id)
    report = run_all_experiments(sequences, outcomes, player_id)
    report["exp_session_generalization"] = experiment_session_ordered_generalization(db_path, player_id)
    report["exp_controlled_rearing"] = experiment_controlled_rearing(db_path, player_id)
    report["exp_bias_interventions"] = experiment_bias_interventions(db_path, player_id)
    return report


def run_clone_ab_block(
    *,
    db_path: str,
    player_id: str,
    game_type: str,
    clone_type: str,
    baseline_type: str,
    clone_scores: list[int],
    baseline_scores: list[int],
    detection_rate: float | None = None,
    clone_fidelity: float | None = None,
) -> str:
    run_id = uuid.uuid4().hex[:12]
    block_id = uuid.uuid4().hex[:10]
    conn = connect(db_path)
    init_extended_db(db_path)
    try:
        conn.execute(
            """INSERT INTO clone_ab_runs
               (run_id, block_id, player_id, game_type, clone_type, baseline_type,
                n_games, clone_wins, baseline_wins, draws, clone_fidelity, detection_rate, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_id,
                block_id,
                player_id,
                game_type,
                clone_type,
                baseline_type,
                len(clone_scores) + len(baseline_scores),
                int(sum(1 for score in clone_scores if score > 0)),
                int(sum(1 for score in baseline_scores if score > 0)),
                int(sum(1 for score in clone_scores + baseline_scores if score == 0)),
                clone_fidelity,
                detection_rate,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()
    finally:
        conn.close()
    return run_id


def persist_counterfactual_replay(
    *,
    db_path: str,
    source_game_id: str,
    player_id: str,
    game_type: str,
    baseline_agent: str,
    clone_agent: str,
    source_result: str,
    baseline_results: list[int],
    clone_results: list[int],
) -> str:
    outcome = counterfactual_lift(
        baseline_agent=baseline_agent,
        clone_agent=clone_agent,
        source_result=source_result,
        baseline_results=baseline_results,
        clone_results=clone_results,
    )
    replay_id = uuid.uuid4().hex[:12]
    conn = connect(db_path)
    init_extended_db(db_path)
    try:
        conn.execute(
            """INSERT INTO counterfactual_replays
               (replay_id, source_game_id, player_id, game_type, baseline_agent, clone_agent,
                source_result, replay_summary_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                replay_id,
                source_game_id,
                player_id,
                game_type,
                baseline_agent,
                clone_agent,
                source_result,
                json.dumps(asdict(outcome)),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()
    finally:
        conn.close()
    return replay_id
