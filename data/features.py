"""Behavioral feature extraction helpers for clone training and dashboard analysis."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from data.schemas import connect
from environments.rps_plus import N_MOVES


CANONICAL_BASELINE_BATTERY = [
    "random",
    "heuristic",
    "profile_counter",
    "ngram",
    "lstm",
    "adaptive_router",
]

CONTROLLED_REARING_SLICES = [
    "early_sessions_v1",
    "recent_sessions_v1",
    "wins_only_v1",
    "losses_only_v1",
    "single_game_family_v1",
]


def load_rounds(db_path: str | Path, agent_name: str | None = None) -> pd.DataFrame:
    conn = connect(db_path)
    try:
        query = "SELECT g.agent_name, g.game_type, g.started_at, r.* FROM rounds r JOIN games g USING (game_id)"
        params: tuple[object, ...] = ()
        if agent_name is not None:
            query += " WHERE g.agent_name = ?"
            params = (agent_name,)
        return pd.read_sql_query(query, conn, params=params)
    finally:
        conn.close()


def load_game_sessions(db_path: str | Path, player_id: str | None = None) -> pd.DataFrame:
    conn = connect(db_path)
    try:
        query = """
            SELECT game_id, agent_name AS player_id, opponent_name, game_type, started_at,
                   n_turns, agent_score, opponent_score
            FROM games
        """
        params: tuple[object, ...] = ()
        if player_id is not None:
            query += " WHERE agent_name = ?"
            params = (player_id,)
        query += " ORDER BY started_at"
        df = pd.read_sql_query(query, conn, params=params)
    finally:
        conn.close()
    if df.empty:
        return df
    df["result"] = np.where(
        df["agent_score"] > df["opponent_score"],
        "Win",
        np.where(df["agent_score"] < df["opponent_score"], "Loss", "Draw"),
    )
    return df


def load_session_rounds(
    db_path: str | Path,
    player_id: str,
    *,
    game_type: str | None = None,
) -> pd.DataFrame:
    rounds = load_rounds(db_path, player_id)
    if rounds.empty:
        return rounds
    if game_type is not None:
        rounds = rounds.loc[rounds["game_type"] == game_type].copy()
    return rounds.sort_values(["started_at", "game_id", "turn"])


def session_ordered_sequences(
    db_path: str | Path,
    player_id: str,
    *,
    game_type: str = "RPS+",
) -> list[dict[str, object]]:
    rounds = load_session_rounds(db_path, player_id, game_type=game_type)
    sessions = load_game_sessions(db_path, player_id)
    if rounds.empty or sessions.empty:
        return []
    sessions = sessions.loc[sessions["game_type"] == game_type].copy()
    joined = rounds.merge(
        sessions[["game_id", "started_at", "opponent_name", "result"]],
        on=["game_id", "started_at"],
        how="left",
    )
    sequences: list[dict[str, object]] = []
    for game_id, group in joined.groupby("game_id", sort=False):
        group = group.sort_values("turn")
        moves = [int(v) for v in group["agent_move"].tolist() if 0 <= int(v) < N_MOVES]
        opp = [int(v) for v in group["opponent_move"].tolist() if 0 <= int(v) < N_MOVES]
        outcomes = [int(v) for v in group["outcome"].tolist()]
        if not moves:
            continue
        sequences.append(
            {
                "game_id": game_id,
                "game_type": game_type,
                "started_at": str(group["started_at"].iloc[0]),
                "opponent_name": str(group["opponent_name"].iloc[0]),
                "result": str(group["result"].iloc[0]),
                "moves": moves,
                "opponent_moves": opp,
                "outcomes": outcomes,
            }
        )
    return sequences


def session_ordered_split(
    db_path: str | Path,
    player_id: str,
    *,
    game_type: str = "RPS+",
    train_fraction: float = 0.7,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    sequences = session_ordered_sequences(db_path, player_id, game_type=game_type)
    if not sequences:
        return [], []
    split = max(1, int(len(sequences) * train_fraction))
    train = sequences[:split]
    eval_ = sequences[split:] or sequences[-1:]
    return train, eval_


def apply_session_slice(
    sequences: list[dict[str, object]],
    slice_name: str,
    *,
    dominant_game_type: str | None = None,
) -> list[dict[str, object]]:
    if not sequences:
        return []
    if slice_name == "early_sessions_v1":
        keep = max(1, int(np.ceil(len(sequences) * 0.4)))
        return sequences[:keep]
    if slice_name == "recent_sessions_v1":
        keep = max(1, int(np.ceil(len(sequences) * 0.4)))
        return sequences[-keep:]
    if slice_name == "wins_only_v1":
        filtered = [seq for seq in sequences if str(seq.get("result")) == "Win"]
        return filtered or sequences[:1]
    if slice_name == "losses_only_v1":
        filtered = [seq for seq in sequences if str(seq.get("result")) == "Loss"]
        return filtered or sequences[:1]
    if slice_name == "single_game_family_v1":
        target = dominant_game_type
        if target is None:
            target = str(sequences[0].get("game_type", "RPS+"))
        filtered = [seq for seq in sequences if str(seq.get("game_type", target)) == target]
        return filtered or sequences[:1]
    return list(sequences)


def slice_registry_frame(
    db_path: str | Path,
    player_id: str,
    *,
    game_type: str = "RPS+",
) -> pd.DataFrame:
    sequences = session_ordered_sequences(db_path, player_id, game_type=game_type)
    if not sequences:
        return pd.DataFrame()
    rows = []
    for slice_name in CONTROLLED_REARING_SLICES:
        subset = apply_session_slice(sequences, slice_name, dominant_game_type=game_type)
        rows.append(
            {
                "slice_name": slice_name,
                "n_sessions": len(subset),
                "n_rounds": int(sum(len(seq["moves"]) for seq in subset)),
                "coverage": len(subset) / max(len(sequences), 1),
            }
        )
    return pd.DataFrame(rows)


def move_distribution(df: pd.DataFrame) -> np.ndarray:
    counts = np.zeros(N_MOVES, dtype=np.float64)
    if df.empty or "agent_move" not in df:
        return counts
    for m, c in df["agent_move"].value_counts().items():
        move = int(m)
        if 0 <= move < N_MOVES:
            counts[move] = c
    total = counts.sum()
    return counts / total if total > 0 else counts


def transition_matrix(df: pd.DataFrame) -> np.ndarray:
    M = np.zeros((N_MOVES, N_MOVES), dtype=np.float64)
    if df.empty:
        return M
    for _, game in df.sort_values(["game_id", "turn"]).groupby("game_id"):
        moves = game["agent_move"].astype(int).to_numpy()
        for a, b in zip(moves[:-1], moves[1:]):
            if 0 <= a < N_MOVES and 0 <= b < N_MOVES:
                M[a, b] += 1
    row_sums = M.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return M / row_sums


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-9) -> float:
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float(np.sum(p * np.log(p / q)))


def behavioral_fidelity(reference: pd.DataFrame, candidate: pd.DataFrame) -> dict[str, float]:
    p = move_distribution(reference)
    q = move_distribution(candidate)
    return {
        "kl_divergence": kl_divergence(p, q),
        "tvd": float(0.5 * np.abs(p - q).sum()),
    }


def behavioral_timeline(db_path: str | Path, player_id: str) -> pd.DataFrame:
    sessions = load_game_sessions(db_path, player_id)
    if sessions.empty:
        return pd.DataFrame()
    rounds = load_rounds(db_path, player_id)
    clone_metrics = load_clone_results(db_path, player_id)
    rows: list[dict[str, object]] = []
    cum_games = 0
    cum_wins = 0
    for session in sessions.itertuples(index=False):
        cum_games += 1
        if session.result == "Win":
            cum_wins += 1
        game_rounds = rounds.loc[rounds["game_id"] == session.game_id]
        dist = move_distribution(game_rounds)
        clone_slice = clone_metrics.loc[clone_metrics["game_type"] == session.game_type]
        rows.append(
            {
                "started_at": session.started_at,
                "game_id": session.game_id,
                "game_type": session.game_type,
                "cum_games": cum_games,
                "cum_win_rate": cum_wins / max(cum_games, 1),
                "move_entropy": float(-(dist[dist > 0] * np.log2(dist[dist > 0])).sum()) if dist.sum() else 0.0,
                "kl_divergence": float(clone_slice["kl_divergence"].mean()) if not clone_slice.empty else np.nan,
                "fidelity_score": float(clone_slice["fidelity_score"].mean()) if not clone_slice.empty else np.nan,
            }
        )
    return pd.DataFrame(rows)


def per_game_fingerprint(db_path: str | Path, player_id: str) -> pd.DataFrame:
    rounds = load_rounds(db_path, player_id)
    sessions = load_game_sessions(db_path, player_id)
    if rounds.empty or sessions.empty:
        return pd.DataFrame()
    session_meta = sessions[["game_id", "game_type", "result"]].rename(columns={"game_type": "session_game_type"})
    merged = rounds.merge(session_meta, on="game_id", how="left")
    if "session_game_type" in merged.columns:
        merged["analysis_game_type"] = merged["session_game_type"]
    elif "game_type" in merged.columns:
        merged["analysis_game_type"] = merged["game_type"]
    else:
        merged["analysis_game_type"] = "Unknown"
    rows = []
    for game_type, group in merged.groupby("analysis_game_type", dropna=False):
        dist = move_distribution(group)
        rows.append(
            {
                "game_type": game_type,
                "games_played": int(group["game_id"].nunique()),
                "win_rate": float((group.groupby("game_id")["result"].first() == "Win").mean()),
                "move_entropy": float(-(dist[dist > 0] * np.log2(dist[dist > 0])).sum()) if dist.sum() else 0.0,
                "top_move": int(np.argmax(dist)) if dist.sum() else -1,
                "signature_json": json.dumps(dist.tolist()),
            }
        )
    return pd.DataFrame(rows).sort_values(["games_played", "win_rate"], ascending=[False, False])


def cross_game_embedding(db_path: str | Path, player_id: str) -> np.ndarray:
    fingerprint = per_game_fingerprint(db_path, player_id)
    if fingerprint.empty:
        return np.zeros(16, dtype=np.float32)
    vector: list[float] = []
    for row in fingerprint.itertuples(index=False):
        sig = np.array(json.loads(row.signature_json), dtype=np.float32)
        vector.extend(
            [
                float(row.games_played),
                float(row.win_rate),
                float(row.move_entropy),
                float(row.top_move),
            ]
        )
        vector.extend(sig.tolist())
    arr = np.array(vector, dtype=np.float32)
    if arr.size >= 16:
        return arr[:16]
    return np.pad(arr, (0, 16 - arr.size))


def opponent_style_summary(
    db_path: str | Path,
    player_id: str,
    *,
    game_type: str = "RPS+",
) -> pd.DataFrame:
    sessions = load_game_sessions(db_path, player_id)
    rounds = load_session_rounds(db_path, player_id, game_type=game_type)
    if sessions.empty or rounds.empty:
        return pd.DataFrame()
    sessions = sessions.loc[sessions["game_type"] == game_type].copy()
    joined = rounds.merge(
        sessions[["game_id", "opponent_name", "result"]],
        on="game_id",
        how="left",
    )
    rows = []
    for opponent_name, group in joined.groupby("opponent_name", dropna=False):
        opp_dist = np.zeros(N_MOVES, dtype=np.float64)
        for m, c in group["opponent_move"].value_counts().items():
            idx = int(m)
            if 0 <= idx < N_MOVES:
                opp_dist[idx] = c
        if opp_dist.sum() > 0:
            opp_dist /= opp_dist.sum()
        rows.append(
            {
                "opponent_name": str(opponent_name),
                "games": int(group["game_id"].nunique()),
                "win_rate_vs_opponent": float((group.groupby("game_id")["result"].first() == "Win").mean()),
                "opponent_signature_json": json.dumps(opp_dist.tolist()),
            }
        )
    return pd.DataFrame(rows).sort_values("games", ascending=False)


def recency_weights(n_items: int, decay: float = 0.85) -> np.ndarray:
    if n_items <= 0:
        return np.zeros(0, dtype=np.float64)
    raw = np.array([decay ** (n_items - idx - 1) for idx in range(n_items)], dtype=np.float64)
    return raw / raw.sum()


def load_clone_results(db_path: str | Path, player_id: str) -> pd.DataFrame:
    conn = connect(db_path)
    try:
        return pd.read_sql_query(
            """
            SELECT player_id, impostor_type, game_type, fidelity_score, kl_divergence,
                   tvd, fool_rate, explanation_sample, trained_at
            FROM impostor_results
            WHERE player_id = ?
            ORDER BY trained_at
            """,
            conn,
            params=(player_id,),
        )
    finally:
        conn.close()


def session_generalization_frame(
    db_path: str | Path,
    player_id: str,
    *,
    game_type: str = "RPS+",
) -> pd.DataFrame:
    train, eval_ = session_ordered_split(db_path, player_id, game_type=game_type)
    if not train or not eval_:
        return pd.DataFrame()
    return pd.DataFrame(
        {
            "split": ["train", "eval"],
            "n_sessions": [len(train), len(eval_)],
            "n_rounds": [
                sum(len(seq["moves"]) for seq in train),
                sum(len(seq["moves"]) for seq in eval_),
            ],
            "first_session": [train[0]["started_at"], eval_[0]["started_at"]],
            "last_session": [train[-1]["started_at"], eval_[-1]["started_at"]],
        }
    )


def data_sufficiency_status(
    db_path: str | Path,
    player_id: str,
    *,
    game_type: str = "RPS+",
) -> dict[str, object]:
    sequences = session_ordered_sequences(db_path, player_id, game_type=game_type)
    rounds = sum(len(seq["moves"]) for seq in sequences)
    sessions = len(sequences)
    if rounds < 50 or sessions < 3:
        label = "Too little data"
    elif rounds < 250 or sessions < 8:
        label = "Trainable"
    else:
        label = "Paper-ready"
    return {
        "label": label,
        "rounds": rounds,
        "sessions": sessions,
    }


def player_dataset_card(
    db_path: str | Path,
    player_id: str,
    *,
    game_type: str = "RPS+",
) -> dict[str, object]:
    sequences = session_ordered_sequences(db_path, player_id, game_type=game_type)
    sessions = load_game_sessions(db_path, player_id)
    fingerprint = per_game_fingerprint(db_path, player_id)
    clone_results = load_clone_results(db_path, player_id)
    rounds = sum(len(seq["moves"]) for seq in sequences)
    session_count = len(sequences)
    drift_coverage = 0.0
    if not clone_results.empty:
        drift_coverage = float(clone_results["trained_at"].nunique()) / max(session_count, 1)
    return {
        "rounds": rounds,
        "sessions": session_count,
        "game_coverage": int(fingerprint["game_type"].nunique()) if not fingerprint.empty else 0,
        "drift_coverage": float(np.clip(drift_coverage, 0.0, 1.0)),
        "trainability": data_sufficiency_status(db_path, player_id, game_type=game_type)["label"],
    }


def clone_latency_fidelity_frame(db_path: str | Path) -> pd.DataFrame:
    conn = connect(db_path)
    try:
        clones = pd.read_sql_query(
            """
            SELECT player_id, impostor_type, game_type, fidelity_score, fool_rate, n_training_rounds
            FROM impostor_results
            ORDER BY trained_at DESC
            """,
            conn,
        )
        agents = pd.read_sql_query(
            """
            SELECT agent_name, win_rate, behavioral_fidelity, avg_decision_ms
            FROM agent_results
            """,
            conn,
        )
    finally:
        conn.close()
    agents = agents.copy() if not agents.empty else pd.DataFrame(columns=["agent_name", "win_rate", "behavioral_fidelity", "avg_decision_ms"])
    if not agents.empty:
        agents["agent_key"] = agents["agent_name"].astype(str).str.split("::").str[0].str.replace(r"_seed\d+$", "", regex=True)
        agent_agg = agents.groupby("agent_key", as_index=False).agg(
            avg_decision_ms=("avg_decision_ms", "mean"),
            win_rate=("win_rate", "mean"),
            behavioral_fidelity_runtime=("behavioral_fidelity", "mean"),
        )
    else:
        agent_agg = pd.DataFrame(columns=["agent_key", "avg_decision_ms", "win_rate", "behavioral_fidelity_runtime"])

    if clones.empty and agent_agg.empty:
        return pd.DataFrame()
    if clones.empty:
        clone_like = agent_agg.loc[agent_agg["agent_key"].astype(str).str.contains("ngram|lstm|mixture", regex=True)].copy()
        clone_like["impostor_type"] = clone_like["agent_key"]
        clone_like["fidelity_score"] = clone_like["behavioral_fidelity_runtime"]
        clone_like["fool_rate"] = np.nan
        clone_like["n_training_rounds"] = np.nan
        clone_like["player_id"] = "runtime_only"
        clone_like["game_type"] = "RPS+"
        return clone_like

    merged = clones.merge(agent_agg, left_on="impostor_type", right_on="agent_key", how="outer")
    merged["impostor_type"] = merged["impostor_type"].fillna(merged["agent_key"])
    merged["fidelity_score"] = merged["fidelity_score"].fillna(merged["behavioral_fidelity_runtime"])
    merged["game_type"] = merged["game_type"].fillna("RPS+")
    merged["player_id"] = merged["player_id"].fillna("runtime_only")
    return merged


def failure_case_gallery(
    db_path: str | Path,
    player_id: str,
) -> pd.DataFrame:
    clone_results = load_clone_results(db_path, player_id)
    if clone_results.empty:
        return pd.DataFrame()
    worst = clone_results.copy()
    worst["failure_score"] = worst["kl_divergence"].fillna(0.0) - worst["fidelity_score"].fillna(0.0)
    return worst.sort_values(["failure_score", "trained_at"], ascending=[False, False]).head(10)


def canonical_figure_frames(
    db_path: str | Path,
    player_id: str,
) -> dict[str, pd.DataFrame]:
    return {
        "timeline": behavioral_timeline(db_path, player_id),
        "fingerprint": per_game_fingerprint(db_path, player_id),
        "failures": failure_case_gallery(db_path, player_id),
        "generalization": session_generalization_frame(db_path, player_id),
    }


def contextual_hint(game_type: str, info: dict, env=None) -> str:
    legal = info.get("legal_moves") or []
    if game_type == "Tic-Tac-Toe":
        if 4 in legal:
            return "Take the center while it is still available."
        corners = [move for move in legal if move in {0, 2, 6, 8}]
        if corners:
            return "Corners are the best fallback when the center is gone."
    if game_type == "Connect Four":
        if 3 in legal:
            return "The center column gives the most follow-up threat lines."
        return "Look for a move that creates two threats on the next turn."
    if game_type == "Othello":
        corner_moves = [move for move in legal if move in {0, 7, 56, 63}]
        if corner_moves:
            return "A corner is stable territory. Take it if it is genuinely legal."
        return "Early disc count matters less than stable edges and corners."
    if game_type == "Checkers" and getattr(env, "forced_jump", False):
        return "Forced captures can still be traps. Check what square you land on next."
    if game_type == "Chess":
        return "Develop pieces and keep the king safe before chasing material."
    if game_type == "Gomoku":
        return "Extend open threes and block any open four immediately."
    if game_type == "Nim":
        total = int(np.sum(getattr(env, "piles", [0])))
        return f"There are {total} matches left. Try to leave a symmetric response if you can."
    if game_type == "RPS+":
        energy = getattr(getattr(env, "state", None), "agent_energy", None)
        if energy is not None and energy < 2:
            return "You cannot use Power yet, so plan one turn ahead."
        return "Recharge is strongest when it sets up a credible Power threat."
    return "Use the legal moves and look for the option that improves your next turn, not just this one."


def recency_bias_warning(rounds_df: pd.DataFrame) -> str | None:
    if rounds_df.empty or "agent_move_name" not in rounds_df:
        return None
    last = rounds_df["agent_move_name"].tail(3).tolist()
    if len(last) == 3 and len(set(last)) == 1:
        return f"Recency bias detected: repeated {last[0]} three times in a row."
    return None


def top_move_name(rounds_df: pd.DataFrame) -> str | None:
    if rounds_df.empty or "agent_move_name" not in rounds_df:
        return None
    counts = Counter(rounds_df["agent_move_name"].tolist())
    return counts.most_common(1)[0][0] if counts else None
