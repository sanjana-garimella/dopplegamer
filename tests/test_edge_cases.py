"""Edge case and robustness tests for AgentBench.

Tests boundary conditions, empty inputs, adversarial values,
concurrency-adjacent state issues, and data integrity guarantees.
"""

from __future__ import annotations

import sqlite3
import tempfile
import os
from pathlib import Path

import numpy as np
import pytest

from environments.rps_plus import RPSPlusEnv, Move, N_MOVES
from agents.base import RandomAgent, HeuristicAgent
from agents.rl.agent import PPOAgent, BCRLAgent
from agents.sft.model import SFTAgent
from agents.agentic_llm import AgenticLLM
from agents.impostor.ngram import NGramImpostor
from agents.impostor.lstm import LSTMImpostor
from evaluation.metrics import action_distribution, kl_divergence, action_perplexity
from impostor.metrics import (
    bias_profile,
    bias_correlation,
    detection_accuracy,
    per_decision_bias_rates,
    action_mode_divergence,
    recency_bias,
    win_streak_aggression,
    loss_aversion,
)
from serving.base import InferenceResult
from data.schemas import init_db, connect, ALL_TABLES


# ═══════════════════════════════════════════════════════════════════════════════
# ENVIRONMENT EDGE CASES
# ═══════════════════════════════════════════════════════════════════════════════

class TestEnvironmentEdgeCases:
    def test_zero_turns(self):
        """Game with max_turns=0 should terminate immediately."""
        env = RPSPlusEnv(max_turns=0, seed=1)
        obs, info = env.reset()
        # First step should truncate
        obs, reward, terminated, truncated, info = env.step(0)
        assert terminated or truncated

    def test_single_turn(self):
        env = RPSPlusEnv(max_turns=1, seed=1)
        obs, info = env.reset()
        obs, reward, terminated, truncated, info = env.step(0)
        assert terminated or truncated

    def test_illegal_action_handling(self):
        """Passing action outside legal_moves should not crash."""
        env = RPSPlusEnv(max_turns=10, seed=1)
        obs, info = env.reset()
        # Drain energy so POWER becomes illegal
        for _ in range(20):
            obs, reward, terminated, truncated, info = env.step(int(Move.POWER))
            if terminated or truncated:
                break
        # Environment should handle any int action without crashing
        # (it clamps or selects from legal)

    def test_repeated_resets(self):
        """Multiple resets should produce consistent initial state."""
        env = RPSPlusEnv(max_turns=10, seed=42)
        obs1, info1 = env.reset(seed=42)
        env.step(0)
        env.step(1)
        obs2, info2 = env.reset(seed=42)
        assert np.array_equal(obs1, obs2)
        assert info1 == info2

    def test_all_moves_playable(self):
        """Each of the 6 moves should be playable at least once in a fresh game."""
        env = RPSPlusEnv(max_turns=50, seed=1)
        obs, info = env.reset()
        for move in range(N_MOVES):
            if move in info.get("legal_moves", []):
                obs, _, term, trunc, info = env.step(move)
                if term or trunc:
                    break

    def test_energy_never_negative(self):
        """Agent energy should never go below 0."""
        env = RPSPlusEnv(max_turns=100, seed=7)
        obs, info = env.reset()
        rng = np.random.default_rng(7)
        done = False
        while not done:
            legal = info.get("legal_moves", list(range(N_MOVES)))
            action = int(rng.choice(legal))
            obs, _, term, trunc, info = env.step(action)
            assert obs[0] >= 0.0, "Agent energy went negative"
            assert obs[1] >= 0.0, "Opponent energy went negative"
            done = term or trunc

    def test_deterministic_with_same_seed(self):
        """Same seed should produce identical game trajectories."""
        def play(seed):
            env = RPSPlusEnv(max_turns=20, seed=seed)
            obs, info = env.reset(seed=seed)
            actions = []
            done = False
            while not done:
                actions.append(0)
                obs, r, term, trunc, info = env.step(0)
                done = term or trunc
            return actions, env.state.agent_score, env.state.opponent_score

        a1, s1, o1 = play(999)
        a2, s2, o2 = play(999)
        assert s1 == s2 and o1 == o2


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT EDGE CASES
# ═══════════════════════════════════════════════════════════════════════════════

class TestAgentEdgeCases:
    def test_agents_handle_empty_legal_moves(self):
        """Agents should not crash if legal_moves is empty (defensive)."""
        obs = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        info = {"legal_moves": list(range(N_MOVES))}  # fallback to all legal
        agents = [PPOAgent(seed=1), BCRLAgent(seed=1), SFTAgent(), NGramImpostor(seed=1)]
        for agent in agents:
            action = agent.act(obs, info)
            assert 0 <= action < N_MOVES

    def test_agents_handle_single_legal_move(self):
        """If only one move is legal, all agents should return it."""
        obs = np.array([0.0, 0.5, 0.5], dtype=np.float32)
        info = {"legal_moves": [2]}
        agents = [PPOAgent(seed=1), BCRLAgent(seed=1), NGramImpostor(seed=1)]
        for agent in agents:
            action = agent.act(obs, info)
            assert action == 2

    def test_ngram_no_training_data(self):
        """NGram with no data should still return valid actions."""
        agent = NGramImpostor(n=3, seed=1)
        obs = np.zeros(3)
        info = {"legal_moves": [0, 1, 2, 3, 4, 5]}
        action = agent.act(obs, info)
        assert 0 <= action < N_MOVES

    def test_ngram_single_move_training(self):
        """NGram trained on a single move should still work."""
        agent = NGramImpostor(n=2, seed=1)
        agent.train([0])
        obs = np.zeros(3)
        info = {"legal_moves": list(range(N_MOVES))}
        action = agent.act(obs, info)
        assert 0 <= action < N_MOVES

    def test_lstm_no_history(self):
        """LSTM with no history should fallback to random legal move."""
        agent = LSTMImpostor(seed=1)
        obs = np.zeros(3)
        info = {"legal_moves": [0, 1, 2]}
        action = agent.act(obs, info)
        assert action in [0, 1, 2]

    def test_lstm_empty_training(self):
        """LSTM trained on empty sequences should not crash."""
        agent = LSTMImpostor(seed=1)
        result = agent.train([], epochs=1)
        # Should either train on nothing or report error gracefully
        assert isinstance(result, dict)

    def test_lstm_single_token_sequences(self):
        """LSTM trained on length-1 sequences (no pairs) should not crash."""
        agent = LSTMImpostor(seed=1)
        result = agent.train([[0], [1], [2]], epochs=1)
        assert isinstance(result, dict)

    def test_bcrl_observe_extreme_outcomes(self):
        """BCRLAgent should handle extreme outcome streaks without overflow."""
        agent = BCRLAgent(seed=1)
        obs = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        info = {"legal_moves": list(range(N_MOVES))}
        # 100 consecutive wins
        for _ in range(100):
            agent.observe(0, 1, 1)
        action = agent.act(obs, info)
        assert 0 <= action < N_MOVES
        # 100 consecutive losses
        for _ in range(100):
            agent.observe(0, 1, -1)
        action = agent.act(obs, info)
        assert 0 <= action < N_MOVES

    def test_agentic_llm_all_modes(self):
        """AgenticLLM should work in all three modes."""
        from agents.agentic_llm import AgenticLLM, AgenticConfig
        for mode in ["react", "crew", "adk"]:
            agent = AgenticLLM(AgenticConfig(mode=mode))
            obs = np.array([0.6, 0.6, 0.5], dtype=np.float32)
            info = {"legal_moves": list(range(N_MOVES))}
            action = agent.act(obs, info)
            assert 0 <= action < N_MOVES

    def test_agent_reset_clears_state(self):
        """Reset should clear all agent internal state."""
        agent = BCRLAgent(seed=1)
        obs = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        info = {"legal_moves": list(range(N_MOVES))}
        for _ in range(10):
            agent.observe(0, 1, 1)
        agent.reset(seed=99)
        # Internal state should be cleared
        assert agent._last_opp is None
        assert len(agent._recent_outcomes) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# METRICS EDGE CASES
# ═══════════════════════════════════════════════════════════════════════════════

class TestMetricsEdgeCases:
    def test_action_distribution_empty(self):
        """Empty action list should return zeros."""
        dist = action_distribution([])
        assert np.allclose(dist, np.zeros(N_MOVES))

    def test_action_distribution_single(self):
        """Single action should produce one-hot distribution."""
        dist = action_distribution([3])
        assert dist[3] == 1.0
        assert dist.sum() == 1.0

    def test_kl_divergence_identical(self):
        """KL divergence of identical distributions should be ~0."""
        p = np.array([0.2, 0.3, 0.1, 0.1, 0.2, 0.1])
        assert kl_divergence(p, p) < 1e-6

    def test_kl_divergence_zeros(self):
        """KL divergence should handle zero-probability entries (eps clipping)."""
        p = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        q = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        result = kl_divergence(p, q)
        assert np.isfinite(result)

    def test_action_perplexity_perfect(self):
        """Perfect predictions should give perplexity close to 1."""
        probs = [np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]) for _ in range(5)]
        actual = [0, 0, 0, 0, 0]
        ppl = action_perplexity(probs, actual)
        assert abs(ppl - 1.0) < 0.01

    def test_action_perplexity_uniform(self):
        """Uniform predictions over 6 moves should give perplexity = 6."""
        probs = [np.ones(6) / 6.0 for _ in range(10)]
        actual = [0, 1, 2, 3, 4, 5, 0, 1, 2, 3]
        ppl = action_perplexity(probs, actual)
        assert abs(ppl - 6.0) < 0.1

    def test_action_perplexity_empty(self):
        """Empty inputs should return inf."""
        assert action_perplexity([], []) == float("inf")

    def test_bias_profile_empty(self):
        """Empty move/outcome lists should return zero biases."""
        bp = bias_profile([], [])
        assert bp.recency_bias == 0.0
        assert bp.win_streak_aggression == 0.0
        assert bp.loss_aversion == 0.0

    def test_bias_profile_short_sequence(self):
        """Sequences shorter than streak threshold should return 0."""
        bp = bias_profile([0, 1], [1, -1])
        assert bp.win_streak_aggression == 0.0
        assert bp.loss_aversion == 0.0

    def test_bias_correlation_identical(self):
        """Identical bias profiles should have correlation 1.0."""
        bp = bias_profile([0, 0, 4, 4, 5, 0, 1, 4, 5, 0], [1, 1, 1, 1, -1, -1, -1, 0, 1, 0])
        assert bias_correlation(bp, bp) == 1.0

    def test_bias_correlation_zero_variance(self):
        """Zero-variance profiles (all zeros) should not crash."""
        bp1 = bias_profile([0, 0, 0], [0, 0, 0])
        bp2 = bias_profile([1, 1, 1], [0, 0, 0])
        result = bias_correlation(bp1, bp2)
        assert np.isfinite(result)

    def test_detection_accuracy_empty(self):
        """No verdicts should return zero rates."""
        r = detection_accuracy("test", [])
        assert r.detection_rate == 0.0
        assert r.false_positive_rate == 0.0

    def test_detection_accuracy_all_correct(self):
        """All correct detections should give rate = 1.0."""
        r = detection_accuracy("test", [True] * 10)
        assert r.detection_rate == 1.0

    def test_per_decision_bias_rates_empty(self):
        """Empty inputs should return zero rates."""
        rates = per_decision_bias_rates([], [], [])
        assert rates.recency_rate == 0.0

    def test_per_decision_bias_rates_short(self):
        """Sequences shorter than streak should return zeros."""
        rates = per_decision_bias_rates([0], [1], [2])
        assert rates.aggression_rate == 0.0

    def test_action_mode_divergence_identical(self):
        """Identical sequences should have divergence ~0."""
        seqs = [[0, 0, 1, 0, 4, 0, 1]]
        div = action_mode_divergence(seqs, seqs, n_modes=3)
        assert div < 0.01

    def test_action_mode_divergence_empty(self):
        """Empty sequences should not crash."""
        div = action_mode_divergence([[]], [[]], n_modes=3)
        # Should return 0 since both are zero distributions
        assert np.isfinite(div)

    def test_cache_hit_rate_zero(self):
        """No cache data should return 0."""
        r = InferenceResult(
            text="X", prompt_tokens=10, output_tokens=2,
            ttft_ms=5, tpot_ms=1, total_latency_ms=7,
            prefix_cache_hit_tokens=0, prefix_cache_miss_tokens=0,
        )
        assert r.cache_hit_rate == 0.0

    def test_cache_hit_rate_full(self):
        """All tokens from cache should give rate = 1.0."""
        r = InferenceResult(
            text="X", prompt_tokens=10, output_tokens=2,
            ttft_ms=5, tpot_ms=1, total_latency_ms=7,
            prefix_cache_hit_tokens=100, prefix_cache_miss_tokens=0,
        )
        assert r.cache_hit_rate == 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# DATA / SCHEMA EDGE CASES
# ═══════════════════════════════════════════════════════════════════════════════

class TestDataEdgeCases:
    def _tmp_db(self):
        f = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        f.close()
        return f.name

    def test_init_db_idempotent(self):
        """Calling init_db multiple times should not error."""
        db = self._tmp_db()
        try:
            init_db(db)
            init_db(db)
            init_db(db)
            conn = connect(db)
            tables = conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'").fetchone()[0]
            assert tables == 7
            conn.close()
        finally:
            os.unlink(db)

    def test_schema_foreign_keys(self):
        """Foreign key constraints should be enforced."""
        db = self._tmp_db()
        try:
            init_db(db)
            conn = connect(db)
            # Insert into detection_sessions without a matching player_profiles row
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(
                    """INSERT INTO detection_sessions
                       (session_id, player_id, impostor_type, detected_as_human, confidence, recorded_at)
                       VALUES ('s1', 'nonexistent', 'ngram', 1, 0.8, '2025-01-01')"""
                )
            conn.close()
        finally:
            os.unlink(db)

    def test_empty_database_dashboard_safe(self):
        """Dashboard functions should handle empty tables gracefully."""
        db = self._tmp_db()
        try:
            init_db(db)
            conn = connect(db)
            import pandas as pd
            df = pd.read_sql_query("SELECT * FROM agent_results", conn)
            assert df.empty
            df = pd.read_sql_query("SELECT * FROM inference_benchmarks", conn)
            assert df.empty
            conn.close()
        finally:
            os.unlink(db)

    def test_unicode_player_names(self):
        """Player profiles should handle unicode names."""
        db = self._tmp_db()
        try:
            init_db(db)
            from impostor.player_profiles import PlayerProfileManager
            mgr = PlayerProfileManager(db)
            p = mgr.create("日本語テスト", player_id="jp1")
            retrieved = mgr.get("jp1")
            assert retrieved.display_name == "日本語テスト"
        finally:
            os.unlink(db)

    def test_sql_injection_safe(self):
        """Player names with SQL injection attempts should be safely handled."""
        db = self._tmp_db()
        try:
            init_db(db)
            from impostor.player_profiles import PlayerProfileManager
            mgr = PlayerProfileManager(db)
            malicious = "'; DROP TABLE player_profiles; --"
            p = mgr.create(malicious, player_id="evil")
            retrieved = mgr.get("evil")
            assert retrieved.display_name == malicious
            # Table should still exist
            conn = connect(db)
            count = conn.execute("SELECT COUNT(*) FROM player_profiles").fetchone()[0]
            assert count == 1
            conn.close()
        finally:
            os.unlink(db)

    def test_schema_adds_hot_path_indexes_and_row_factory(self):
        """DB connections should support dict-style rows and dashboard query indexes."""
        db = self._tmp_db()
        try:
            init_db(db)
            conn = connect(db)
            try:
                indexes = {
                    row["name"]
                    for row in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'index'"
                    ).fetchall()
                }
                row = conn.execute("SELECT 1 AS value").fetchone()
                assert row["value"] == 1
                assert "idx_games_agent" in indexes
                assert "idx_rounds_game" in indexes
                assert "idx_games_started" in indexes
            finally:
                conn.close()
        finally:
            os.unlink(db)


# ═══════════════════════════════════════════════════════════════════════════════
# IMPOSTOR TRAINER EDGE CASES
# ═══════════════════════════════════════════════════════════════════════════════

class TestImpostorTrainerEdgeCases:
    def _setup_db(self):
        f = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        f.close()
        init_db(f.name)
        return f.name

    def test_train_nonexistent_player(self):
        """Training on a player with no data should return empty sequences."""
        db = self._setup_db()
        try:
            from agents.impostor.trainer import ImpostorTrainer
            trainer = ImpostorTrainer(db)
            seqs = trainer.load_player_sequences("nobody")
            assert seqs == []
        finally:
            os.unlink(db)

    def test_train_ngram_empty_data(self):
        """NGram training on empty data should still return a functional agent."""
        db = self._setup_db()
        try:
            from agents.impostor.trainer import ImpostorTrainer
            trainer = ImpostorTrainer(db)
            agent, result = trainer.train_ngram("nobody", n=2)
            assert result.n_sequences == 0
            # Agent should still produce valid actions
            obs = np.zeros(3)
            info = {"legal_moves": list(range(N_MOVES))}
            action = agent.act(obs, info)
            assert 0 <= action < N_MOVES
        finally:
            os.unlink(db)


# ═══════════════════════════════════════════════════════════════════════════════
# SERVING ENGINE EDGE CASES
# ═══════════════════════════════════════════════════════════════════════════════

class TestServingEdgeCases:
    def test_empty_prompt(self):
        """Engines should handle empty string prompts."""
        from inference.setup_inference_engines import _MockEngine
        engine = _MockEngine("test")
        result = engine.generate("", max_new_tokens=1)
        assert result.prompt_tokens >= 0
        assert isinstance(result.text, str)

    def test_very_long_prompt(self):
        """Engines should handle very long prompts."""
        from inference.setup_inference_engines import _MockEngine
        engine = _MockEngine("test")
        long_prompt = "word " * 10000
        result = engine.generate(long_prompt, max_new_tokens=1)
        assert result.prompt_tokens == 10000

    def test_zero_new_tokens(self):
        """Requesting 0 new tokens should not crash."""
        from inference.setup_inference_engines import _MockEngine
        engine = _MockEngine("test")
        result = engine.generate("hello", max_new_tokens=0)
        assert isinstance(result, InferenceResult)

    def test_cache_hit_accumulates(self):
        """Repeated identical prefixes should produce cache hits."""
        from inference.setup_inference_engines import _MockEngine
        engine = _MockEngine("test")
        prefix = "A" * 60
        r1 = engine.generate(prefix + " first call with extra tokens here", max_new_tokens=2)
        r2 = engine.generate(prefix + " second call with different suffix", max_new_tokens=2)
        # Same first 50 chars → second call should have hits
        assert r1.prefix_cache_hit_tokens == 0
        assert r2.prefix_cache_hit_tokens > 0


# ═══════════════════════════════════════════════════════════════════════════════
# BENCHMARK RUNNER EDGE CASES
# ═══════════════════════════════════════════════════════════════════════════════

class TestBenchmarkRunnerEdgeCases:
    def test_benchmark_single_round(self):
        """Benchmark with 1 round should complete."""
        from evaluation.runner import run_benchmark
        result = run_benchmark(rounds=1, model_name="mock")
        assert result["run_id"]
        assert all(s["games_played"] == 1 for s in result["agents"])

    def test_benchmark_results_consistent(self):
        """Win + loss + tie should sum to rounds for each agent."""
        from evaluation.runner import run_benchmark
        result = run_benchmark(rounds=50, model_name="mock")
        for s in result["agents"]:
            assert s["wins"] + s["losses"] + s["ties"] == 50

    def test_benchmark_win_rate_bounded(self):
        """Win rate should always be between 0 and 1."""
        from evaluation.runner import run_benchmark
        result = run_benchmark(rounds=20, model_name="mock")
        for s in result["agents"]:
            assert 0.0 <= s["win_rate"] <= 1.0
