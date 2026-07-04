"""End-to-end benchmark runner.

This runner executes:
1) Agent-vs-environment episodes for win/fidelity metrics
2) Inference engine prompt benchmarks for systems metrics
3) Persistence into SQLite tables expected by dashboard
"""

from __future__ import annotations

from collections import defaultdict
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

# Turns per RPS+ game when `rounds` is the number of games.
RPS_TURNS_PER_GAME = 20
BOARD_MAX_MOVES = 100
SYSTEM_PROMPT = (
    "You are playing a multi-turn game. History grows each turn. "
    "Reply with a single legal move name."
)


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


def _engine_row_name(requested: str, result) -> str:
    """Label fallback rows so engine column is not misread as the real backend."""
    extra = getattr(result, "extra", None) or {}
    if not extra.get("fallback"):
        return requested
    actual = extra.get("actual_backend") or "unknown"
    return f"{requested}→{actual}"


def _checkpoint_meta(agent_name: str) -> dict[str, Any]:
    try:
        from agents.checkpoints import resolve_checkpoint

        path = resolve_checkpoint(agent_name)
        return {
            "checkpoint_path": str(path) if path is not None else None,
            "trained_vs_fallback": "trained" if path is not None else "fallback",
        }
    except Exception:
        return {"checkpoint_path": None, "trained_vs_fallback": "unknown"}


def _insert_agent_result(conn, run_id: str, row: AgentScore, meta: dict[str, Any] | None = None) -> None:
    meta = meta or {}
    cols = {r[1] for r in conn.execute("PRAGMA table_info(agent_results)").fetchall()}
    if "trained_vs_fallback" in cols and "checkpoint_path" in cols:
        conn.execute(
            """INSERT OR REPLACE INTO agent_results
            (run_id, agent_name, games_played, wins, losses, ties, win_rate,
             behavioral_fidelity, action_kl, avg_decision_ms, trained_vs_fallback, checkpoint_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
                meta.get("trained_vs_fallback"),
                meta.get("checkpoint_path"),
            ),
        )
    else:
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
    cols = {r[1] for r in conn.execute("PRAGMA table_info(inference_benchmarks)").fetchall()}
    if "prefix_cache_hit_tokens" in cols:
        conn.execute(
            """INSERT OR REPLACE INTO inference_benchmarks
            (run_id, engine, model, quantization, turn, prompt_tokens, output_tokens,
             ttft_ms, tpot_ms, total_latency_ms, kv_cache_mb, scheduling_overhead_ms,
             prefix_cache_hit_tokens, prefix_cache_miss_tokens, actual_backend)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
                result.scheduling_overhead_ms if result.scheduling_overhead_ms is not None else None,
                result.prefix_cache_hit_tokens,
                result.prefix_cache_miss_tokens,
                (result.extra or {}).get("actual_backend"),
            ),
        )
    else:
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
                result.scheduling_overhead_ms if result.scheduling_overhead_ms is not None else None,
            ),
        )


def _play_games(
    agent_name: str,
    game_type: str,
    n_games: int,
    seed: int,
) -> tuple[int, int, int, list[int], list[float]]:
    """Play `n_games` independent episodes. Returns wins, losses, ties, actions, latencies."""
    agent = _agent_factory(agent_name)
    wins = losses = ties = 0
    action_log: list[int] = []
    decision_latencies: list[float] = []
    max_turns = RPS_TURNS_PER_GAME if game_type == "RPS+" else BOARD_MAX_MOVES

    for game_i in range(n_games):
        game_seed = seed * 100_003 + game_i
        env = _new_env(game_type, max_turns, game_seed)
        reset_fn = getattr(agent, "reset", None)
        if callable(reset_fn):
            reset_fn(seed=game_seed)
        obs, info = env.reset(seed=game_seed)
        done = False
        while not done:
            t0 = time.perf_counter()
            action = int(agent.act(obs, info))
            legal = info.get("legal_moves") or []
            # Clamp illegal actions so one bad step does not abort the whole run.
            # RPS+ still raises if an illegal action reaches env.step.
            if legal and action not in legal:
                action = int(legal[0])
            decision_latencies.append((time.perf_counter() - t0) * 1000.0)
            action_log.append(action)
            obs, reward, terminated, truncated, info = env.step(action)
            if game_type == "RPS+" and getattr(env.state, "history", None):
                last = env.state.history[-1]
                observe = getattr(agent, "observe", None)
                if callable(observe):
                    observe(int(last.agent_move), int(last.opponent_move), int(last.outcome))
            done = terminated or truncated

        if env.state.agent_score > env.state.opponent_score:
            wins += 1
        elif env.state.agent_score < env.state.opponent_score:
            losses += 1
        else:
            ties += 1

    return wins, losses, ties, action_log, decision_latencies


def _game_driven_prompts(n_turns: int, seed: int) -> list[str]:
    """Build growing multi-turn prompts from a seeded RPS+ trajectory."""
    env = RPSPlusEnv(max_turns=max(n_turns, 1), seed=seed)
    obs, info = env.reset(seed=seed)
    history_lines: list[str] = []
    prompts: list[str] = []
    done = False
    turn = 0
    while not done and turn < n_turns:
        legal = info.get("legal_moves") or [0]
        action = int(legal[turn % len(legal)])
        obs, reward, terminated, truncated, info = env.step(action)
        if env.state.history:
            last = env.state.history[-1]
            history_lines.append(
                f"Turn {last.turn}: agent={int(last.agent_move)} opp={int(last.opponent_move)} "
                f"outcome={int(last.outcome)}"
            )
        history_block = "\n".join(history_lines)
        prompts.append(f"{SYSTEM_PROMPT}\n\n{history_block}\n\nPredict the next move.")
        done = terminated or truncated
        turn += 1
    # Pad if the game ended early.
    while len(prompts) < n_turns:
        prompts.append(prompts[-1] if prompts else SYSTEM_PROMPT)
    return prompts


def run_benchmark(
    *,
    rounds: int = 100,
    engines: list[str] | None = None,
    agents: list[str] | None = None,
    db_path: str | Path = "data/game_data.db",
    model_name: str = "mock",
    n_seeds: int = 1,
    games: list[str] | None = None,
    allow_fallback: bool = False,
    prompt_seed: int = 0,
) -> dict[str, Any]:
    if engines is None:
        engines = ["baseline", "vllm"]
    if agents is None:
        agents = _benchmark_ready_agents()
    games = [canonical_game_type(g) or g for g in (games or ["RPS+"])]
    run_id = uuid.uuid4().hex

    init_db(db_path)
    conn = connect(db_path)
    fidelity_reference = "none"
    try:
        all_scores: list[AgentScore] = []
        game_breakdown: list[dict[str, Any]] = []
        for seed in range(n_seeds):
            for game_type in games:
                ref_name = "none"
                if game_type == "RPS+":
                    ref_name = "sft" if "sft" in agents and "sft" in AGENT_REGISTRY else "heuristic"
                    fidelity_reference = ref_name

                # Play each agent once; reuse reference agent actions when present.
                played: dict[str, tuple[int, int, int, list[int], list[float]]] = {}
                for agent_name in agents:
                    played[agent_name] = _play_games(agent_name, game_type, rounds, seed)
                if game_type == "RPS+" and ref_name not in played and ref_name in AGENT_REGISTRY:
                    played[ref_name] = _play_games(ref_name, game_type, rounds, seed)

                reference_actions: list[int] = []
                if game_type == "RPS+" and ref_name in played:
                    reference_actions = played[ref_name][3]

                for agent_name in agents:
                    wins, losses, ties, action_log, decision_latencies = played[agent_name]
                    if game_type == "RPS+" and reference_actions:
                        p = action_distribution(reference_actions)
                        q = action_distribution(action_log)
                        fidelity = max(0.0, 1.0 - float(np.abs(p - q).sum()) / 2.0)
                        action_kl = kl_divergence(p, q)
                    else:
                        fidelity = 0.0
                        action_kl = 0.0

                    meta = _checkpoint_meta(agent_name)
                    score = AgentScore(
                        agent_name=f"{agent_name}::{game_type}_seed{seed}",
                        games_played=rounds,
                        wins=wins,
                        losses=losses,
                        ties=ties,
                        win_rate=wins / max(1, rounds),
                        behavioral_fidelity=fidelity,
                        action_kl=action_kl,
                        avg_decision_ms=float(np.mean(decision_latencies)) if decision_latencies else 0.0,
                    )
                    _insert_agent_result(conn, run_id, score, meta)
                    all_scores.append(score)

        aggregated_scores: dict[str, list[AgentScore]] = defaultdict(list)
        for score in all_scores:
            base_name = score.agent_name.rsplit("_seed", 1)[0]
            aggregated_scores[base_name].append(score)

        final_scores = []
        for agent_name, scores in aggregated_scores.items():
            win_rates = [s.win_rate for s in scores]
            fidelities = [s.behavioral_fidelity for s in scores]
            kls = [s.action_kl for s in scores]
            latencies = [s.avg_decision_ms for s in scores]
            bare = agent_name.split("::", 1)[0]
            meta = _checkpoint_meta(bare)
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
            _insert_agent_result(conn, run_id + "_agg", final_score, meta)

            game_name = agent_name.split("::", 1)[1] if "::" in agent_name else "RPS+"
            game_breakdown.append(
                {
                    "agent_name": bare,
                    "game_type": game_name,
                    "games_played": final_score.games_played,
                    "wins": final_score.wins,
                    "losses": final_score.losses,
                    "ties": final_score.ties,
                    "win_rate": final_score.win_rate,
                    "avg_decision_ms": final_score.avg_decision_ms,
                    "win_rate_stats": asdict(summary_stats(win_rates)),
                    "fidelity_reference": fidelity_reference,
                    "trained_vs_fallback": meta.get("trained_vs_fallback"),
                }
            )

        # Statistical significance (t-test vs random::* keys)
        random_keys = [k for k in aggregated_scores if k.startswith("random::") or k == "random"]
        if random_keys and n_seeds > 1:
            for score in final_scores:
                if score.agent_name.startswith("random::") or score.agent_name == "random":
                    continue
                game_suffix = score.agent_name.split("::", 1)[1] if "::" in score.agent_name else ""
                random_key = next(
                    (k for k in random_keys if k.endswith(f"::{game_suffix}")),
                    None,
                )
                if random_key is None:
                    continue
                try:
                    from scipy.stats import ttest_ind

                    agent_wins = [s.win_rate for s in aggregated_scores[score.agent_name]]
                    random_wins = [s.win_rate for s in aggregated_scores[random_key]]
                    if len(agent_wins) > 1 and len(random_wins) > 1 and (
                        np.var(agent_wins) > 0 or np.var(random_wins) > 0
                    ):
                        _, p_value = ttest_ind(agent_wins, random_wins)
                        significance = "significant" if p_value < 0.05 else "not significant"
                        print(
                            f"{score.agent_name} vs {random_key}: "
                            f"win_rate diff {significance} (p={p_value:.3f})"
                        )
                    else:
                        print(f"{score.agent_name} vs {random_key}: insufficient variance for t-test")
                except ImportError:
                    print("Scipy not available for stats")

        all_scores = final_scores

        # ----------------------- inference systems benchmarks
        engine_cfg = EngineConfig(model_name=model_name, allow_fallback=allow_fallback)
        # Lazy: construct only requested engines.
        selected = setup_inference_engines(engine_cfg, engines=engines)

        prompts = _game_driven_prompts(rounds, seed=prompt_seed) if engines else []
        for engine_name, engine in selected.items():
            warmup = getattr(engine, "warmup", None)
            if callable(warmup):
                try:
                    warmup()
                except Exception as warmup_exc:
                    # Do not silence forever: first measured turn may include cold start.
                    print(f"warmup failed for {engine_name}: {warmup_exc}")
            for idx in range(rounds):
                prompt = prompts[idx % len(prompts)]
                result = engine.generate(prompt, max_new_tokens=6)
                _insert_inference_row(
                    conn,
                    run_id=run_id,
                    engine_name=_engine_row_name(engine_name, result),
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
        "fidelity_reference": fidelity_reference,
        "model_name": model_name,
        "prompt_seed": prompt_seed,
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
            avg_decision_ms = (
                float(np.mean(latencies_ms))
                if latencies_ms
                else float(getattr(agent, "latency_estimate_ms", lambda: 0.5)())
            )
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
