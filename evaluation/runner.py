"""End-to-end benchmark runner.

This runner executes:
1) Agent-vs-environment episodes for win/fidelity metrics
2) Inference engine prompt benchmarks for systems metrics
3) Persistence into SQLite tables expected by dashboard
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from agents import AGENT_REGISTRY
from agents.impostor.trainer import ImpostorTrainer
from data.backfill import canonical_game_type
from data.features import CANONICAL_BASELINE_BATTERY, session_ordered_split
from data.schemas import connect, init_db, init_extended_db
from environments import CheckersEnv, ChessEnv, ConnectFourEnv, GomokuEnv, NimEnv, OthelloEnv, RPSPlusEnv, TicTacToeEnv, WarEnv
from environments.rps_plus import N_MOVES
from evaluation.metrics import AgentScore, action_distribution, kl_divergence, summary_stats
from inference import EngineConfig, setup_inference_engines


ENV_REGISTRY = {
    "RPS+": RPSPlusEnv,
    "Tic-Tac-Toe": TicTacToeEnv,
    "Connect Four": ConnectFourEnv,
    "Chess": ChessEnv,
    "Othello": OthelloEnv,
    "Checkers": CheckersEnv,
    "Gomoku": GomokuEnv,
    "Nim": NimEnv,
    "War": WarEnv,
}


def _agent_factory(name: str):
    if name in AGENT_REGISTRY:
        return AGENT_REGISTRY[name]()
    raise ValueError(f"unknown agent `{name}`. Available: {list(AGENT_REGISTRY.keys())}")


def _benchmark_ready_agents() -> list[str]:
    return [name for name in CANONICAL_BASELINE_BATTERY if name in AGENT_REGISTRY]


def _new_env(game_type: str, max_turns: int, seed: int):
    game_type = canonical_game_type(game_type) or game_type
    env_cls = ENV_REGISTRY[game_type]
    if game_type == "RPS+":
        return env_cls(max_turns=max_turns, seed=seed)
    return env_cls(max_moves=max_turns, seed=seed)


def _insert_agent_result(conn, run_id: str, row: AgentScore) -> None:
    conn.execute(
        """INSERT OR REPLACE INTO agent_results
        (run_id, agent_name, games_played, wins, losses, ties, win_rate,
         behavioral_fidelity, action_kl, avg_decision_ms)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            run_id,
            row.agent_name,
            row.games_played,
            row.wins,
            row.losses,
            row.ties,
            row.win_rate,
            row.behavioral_fidelity,
            row.action_kl,
            row.avg_decision_ms,
        ),
    )


def _insert_inference_row(conn, run_id: str, engine_name: str, model: str, quantization: str, turn: int, result) -> None:
    conn.execute(
        """INSERT OR REPLACE INTO inference_benchmarks
        (run_id, engine, model, quantization, turn, prompt_tokens, output_tokens,
         ttft_ms, tpot_ms, total_latency_ms, kv_cache_mb, scheduling_overhead_ms)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            run_id,
            engine_name,
            model,
            quantization,
            turn,
            result.prompt_tokens,
            result.output_tokens,
            result.ttft_ms,
            result.tpot_ms,
            result.total_latency_ms,
            result.kv_cache_mb,
            result.scheduling_overhead_ms,
        ),
    )


def run_benchmark(
    *,
    rounds: int = 100,
    engines: list[str] | None = None,
    agents: list[str] | None = None,
    db_path: str | Path = "data/game_data.db",
    model_name: str = "mock",
    n_seeds: int = 1,
    games: list[str] | None = None,
) -> dict[str, Any]:
    if engines is None:
        engines = ["baseline", "vllm", "preble", "infercept"]
    if agents is None:
        agents = _benchmark_ready_agents()
    games = [canonical_game_type(g) or g for g in (games or ["RPS+"])]
    run_id = uuid.uuid4().hex

    init_db(db_path)
    conn = connect(db_path)
    try:
        # ----------------------- agent performance with multiple seeds
        all_scores: list[AgentScore] = []
        game_breakdown: list[dict[str, Any]] = []
        for seed in range(n_seeds):
            for game_type in games:
                reference_actions: list[int] = []
                for agent_name in agents:
                    agent = _agent_factory(agent_name)
                    env = _new_env(game_type, rounds, seed)
                    obs, info = env.reset(seed=seed)
                    wins = losses = ties = 0
                    action_log: list[int] = []
                    decision_latencies: list[float] = []
                    done = False
                    while not done:
                        t0 = time.perf_counter()
                        action = int(agent.act(obs, info))
                        if action not in (info.get("legal_moves") or []):
                            legal = info.get("legal_moves") or [0]
                            action = int(legal[0])
                        decision_latencies.append((time.perf_counter() - t0) * 1000.0)
                        action_log.append(action)
                        obs, reward, terminated, truncated, info = env.step(action)
                        if game_type == "RPS+":
                            if reward > 0:
                                wins += 1
                            elif reward < 0:
                                losses += 1
                            else:
                                ties += 1
                        # Let RPS+ agents update local memory with compatible move ids.
                        if game_type == "RPS+" and getattr(env.state, "history", None):
                            last = env.state.history[-1]
                            observe = getattr(agent, "observe", None)
                            if callable(observe):
                                observe(int(last.agent_move), int(last.opponent_move), int(last.outcome))
                        done = terminated or truncated

                    if game_type != "RPS+":
                        if env.state.agent_score > env.state.opponent_score:
                            wins = 1
                        elif env.state.agent_score < env.state.opponent_score:
                            losses = 1
                        else:
                            ties = 1

                    if agent_name == "sft" and game_type == "RPS+" and not reference_actions:
                        reference_actions = action_log[:]
                    if game_type == "RPS+":
                        p = action_distribution(reference_actions) if reference_actions else np.ones(6) / 6
                        q = action_distribution(action_log)
                        fidelity = max(0.0, 1.0 - float(np.abs(p - q).sum()) / 2.0)
                        action_kl = kl_divergence(p, q)
                    else:
                        fidelity = 0.0
                        action_kl = 0.0

                    score = AgentScore(
                        agent_name=f"{agent_name}::{game_type}_seed{seed}",
                        games_played=rounds if game_type == "RPS+" else 1,
                        wins=wins,
                        losses=losses,
                        ties=ties,
                        win_rate=wins / max(1, (rounds if game_type == "RPS+" else 1)),
                        behavioral_fidelity=fidelity,
                        action_kl=action_kl,
                        avg_decision_ms=float(np.mean(decision_latencies)) if decision_latencies else 0.0,
                    )
                    _insert_agent_result(conn, run_id, score)
                    all_scores.append(score)

        # Aggregate scores across seeds
        from collections import defaultdict
        aggregated_scores = defaultdict(list)
        for score in all_scores:
            base_name = score.agent_name.rsplit('_seed', 1)[0]
            aggregated_scores[base_name].append(score)

        final_scores = []
        for agent_name, scores in aggregated_scores.items():
            win_rates = [s.win_rate for s in scores]
            fidelities = [s.behavioral_fidelity for s in scores]
            kls = [s.action_kl for s in scores]
            latencies = [s.avg_decision_ms for s in scores]
            final_score = AgentScore(
                agent_name=agent_name,
                games_played=sum(s.games_played for s in scores),
                wins=sum(s.wins for s in scores),
                losses=sum(s.losses for s in scores),
                ties=sum(s.ties for s in scores),
                win_rate=float(np.mean(win_rates)),
                behavioral_fidelity=float(np.mean(fidelities)),
                action_kl=float(np.mean(kls)),
                avg_decision_ms=float(np.mean(latencies)),
            )
            final_scores.append(final_score)
            _insert_agent_result(conn, run_id + "_agg", final_score)

            game_name = agent_name.split("::", 1)[1] if "::" in agent_name else "RPS+"
            game_breakdown.append(
                {
                    "agent_name": agent_name.split("::", 1)[0],
                    "game_type": game_name,
                    "games_played": final_score.games_played,
                    "wins": final_score.wins,
                    "losses": final_score.losses,
                    "ties": final_score.ties,
                    "win_rate": final_score.win_rate,
                    "avg_decision_ms": final_score.avg_decision_ms,
                    "win_rate_stats": asdict(summary_stats(win_rates)),
                }
            )

        # Statistical significance (t-test vs random)
        random_scores = [s for s in final_scores if s.agent_name == "random"]
        if random_scores:
            random_win = random_scores[0].win_rate
            for score in final_scores:
                if score.agent_name != "random":
                    try:
                        from scipy.stats import ttest_ind
                        agent_wins = [s.win_rate for s in aggregated_scores[score.agent_name]]
                        random_wins = [s.win_rate for s in aggregated_scores["random"]]
                        if len(agent_wins) > 1 and len(random_wins) > 1 and (np.var(agent_wins) > 0 or np.var(random_wins) > 0):
                            t_stat, p_value = ttest_ind(agent_wins, random_wins)
                            significance = "significant" if p_value < 0.05 else "not significant"
                            print(f"{score.agent_name} vs random: win_rate diff {significance} (p={p_value:.3f})")
                        else:
                            print(f"{score.agent_name} vs random: insufficient variance for t-test")
                    except ImportError:
                        print("Scipy not available for stats")

        all_scores = final_scores  # For return

        # ----------------------- inference systems benchmarks
        engine_pool = setup_inference_engines(EngineConfig(model_name=model_name))
        selected = {k: v for k, v in engine_pool.items() if k in engines}
        prompts = [
            "You are playing RPS+ turn 1. Predict next move.",
            "You are playing RPS+ turn 20 with long history context. Predict next move.",
            "Tool result indicates opponent favors ROCK. Choose best response.",
        ]
        for engine_name, engine in selected.items():
            for idx in range(rounds):
                prompt = prompts[idx % len(prompts)] + f" [turn={idx+1}]"
                result = engine.generate(prompt, max_new_tokens=6)
                _insert_inference_row(
                    conn,
                    run_id=run_id,
                    engine_name=engine_name,
                    model=model_name,
                    quantization="fp16",
                    turn=idx + 1,
                    result=result,
                )

        conn.commit()
    finally:
        conn.close()

    return {
        "run_id": run_id,
        "db_path": str(db_path),
        "agents": [asdict(s) for s in all_scores],
        "game_breakdown": game_breakdown,
        "engines": engines,
        "games": games,
    }


def canonical_benchmark_agents() -> list[str]:
    return _benchmark_ready_agents()


def benchmark_clone_variants(
    *,
    player_id: str,
    db_path: str | Path = "data/game_data.db",
    game_type: str = "RPS+",
) -> list[dict[str, Any]]:
    trainer = ImpostorTrainer(db_path)
    _, eval_ctx = session_ordered_split(db_path, player_id, game_type=game_type)
    if not eval_ctx:
        return []
    eval_sequences = [list(seq["moves"]) for seq in eval_ctx]
    human_flat = [m for seq in eval_sequences for m in seq]
    p_human = action_distribution(human_flat)
    variants: list[tuple[str, Any, Any]] = []
    ngram, ngram_result = trainer.train_ngram(player_id, game_type=game_type)
    mixture, mixture_result = trainer.train_mixture(player_id, game_type=game_type)
    variants.append(("ngram", ngram, ngram_result))
    variants.append(("mixture", mixture, mixture_result))
    for name, (agent, result) in trainer.train_lstm_variants(player_id, game_type=game_type).items():
        variants.append((name, agent, result))

    conn = connect(db_path)
    run_id = uuid.uuid4().hex[:12]
    rows: list[dict[str, Any]] = []
    try:
        for name, agent, result in variants:
            predicted_moves: list[int] = []
            latencies_ms: list[float] = []
            for seq in eval_sequences:
                history: list[int] = []
                for move in seq:
                    t0 = time.perf_counter()
                    probs = agent.predict_proba(history=history, legal=list(range(N_MOVES)))
                    latencies_ms.append((time.perf_counter() - t0) * 1000.0)
                    predicted_moves.append(int(np.argmax(probs)))
                    history.append(move)
            q = action_distribution(predicted_moves)
            fid = max(0.0, 1.0 - float(np.abs(p_human - q).sum()) / 2.0)
            kl = kl_divergence(p_human, q)
            avg_decision_ms = float(np.mean(latencies_ms)) if latencies_ms else float(getattr(agent, "latency_estimate_ms", lambda: 0.5)())
            row = AgentScore(
                agent_name=name,
                games_played=len(eval_sequences),
                wins=0,
                losses=0,
                ties=len(eval_sequences),
                win_rate=0.0,
                behavioral_fidelity=fid,
                action_kl=kl,
                avg_decision_ms=avg_decision_ms,
            )
            _insert_agent_result(conn, run_id, row)
            rows.append(
                {
                    "agent_name": name,
                    "player_id": player_id,
                    "game_type": game_type,
                    "fidelity_score": fid,
                    "kl_divergence": kl,
                    "avg_decision_ms": avg_decision_ms,
                    "n_training_rounds": result.n_moves,
                    "variant": getattr(agent, "variant_name", name),
                    "quantization": getattr(agent, "quantization_mode", "fp32"),
                }
            )
        conn.commit()
    finally:
        conn.close()
    return rows


def run_clone_human_ab_evaluation(
    *,
    db_path: str | Path,
    player_id: str,
    game_type: str,
    clone_type: str,
    baseline_type: str,
    clone_scores: list[int],
    baseline_scores: list[int],
    clone_fidelity: float | None = None,
    detection_rate: float | None = None,
) -> str:
    init_db(db_path)
    init_extended_db(db_path)
    run_id = uuid.uuid4().hex[:12]
    conn = connect(db_path)
    try:
        conn.execute(
            """INSERT INTO clone_ab_runs
               (run_id, block_id, player_id, game_type, clone_type, baseline_type,
                n_games, clone_wins, baseline_wins, draws, clone_fidelity, detection_rate, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_id,
                "block_a",
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
