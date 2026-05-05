"""Short-term and long-term memory for the agentic LLM.

Short-term: ring buffer of the last N rounds (in-context).
Long-term: vector store over rendered game-summary documents (RAG via ChromaDB).

ChromaDB is imported lazily so this module is safe to import without it.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import uuid


@dataclass
class ShortTermMemory:
    """Recent game history in sliding window (in-context learning)."""
    window: int = 10

    def __post_init__(self) -> None:
        self.buffer: deque[dict[str, Any]] = deque(maxlen=self.window)

    def add(self, record: dict[str, Any]) -> None:
        """Add a game round to short-term memory."""
        self.buffer.append(record)

    def recent(self) -> list[dict[str, Any]]:
        """Get all recent rounds in order."""
        return list(self.buffer)

    def clear(self) -> None:
        """Clear the buffer."""
        self.buffer.clear()

    def to_text(self) -> str:
        """Render recent history as text for LLM context."""
        if not self.buffer:
            return "(no recent history)"
        lines = []
        for i, record in enumerate(self.buffer, 1):
            you_move = record.get("you_move", "?")
            opp_move = record.get("opponent_move", "?")
            outcome = record.get("outcome", 0)
            outcome_str = "won" if outcome > 0 else "lost" if outcome < 0 else "tied"
            lines.append(
                f"  Turn {i}: you played {you_move}, opponent played {opp_move} ({outcome_str})"
            )
        return "\n".join(lines)


@dataclass
class GameSummary:
    """Structured summary of a game for RAG storage."""
    game_id: str
    opponent_name: str
    agent_score: int
    opponent_score: int
    n_turns: int
    key_moments: list[str] = field(default_factory=list)
    strategy: str = ""
    outcome: str = "unknown"

    def to_text(self) -> str:
        """Render as searchable document."""
        text = (
            f"Game {self.game_id}: vs {self.opponent_name} "
            f"({self.agent_score}-{self.opponent_score}) over {self.n_turns} turns\n"
        )
        if self.strategy:
            text += f"Strategy: {self.strategy}\n"
        text += f"Outcome: {self.outcome}\n"
        if self.key_moments:
            text += "Key moments:\n"
            for moment in self.key_moments:
                text += f"  - {moment}\n"
        return text


class LongTermMemory:
    """RAG vector store for game summaries (ChromaDB-backed).
    
    Stores rendered game summaries as documents for similarity search.
    Enables retrieval of similar past games to inform current strategy.
    """

    def __init__(
        self,
        persist_dir: str | Path = "data/chroma",
        collection: str = "agentbench",
        enabled: bool = True
    ) -> None:
        self.persist_dir = Path(persist_dir)
        self.collection_name = collection
        self.enabled = enabled
        self._client = None
        self._coll = None
        self._initialized = False

    def _ensure(self) -> None:
        """Lazy initialization of ChromaDB client."""
        if self._initialized or not self.enabled:
            return
        
        try:
            import chromadb
        except ImportError:
            raise ImportError(
                "ChromaDB not installed. Install with: pip install chromadb"
            )

        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self.persist_dir))
        self._coll = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self._initialized = True

    def add_game(
        self,
        game_summary: GameSummary,
        metadata: dict[str, Any] | None = None
    ) -> None:
        """Add a completed game to long-term memory.
        
        Args:
            game_summary: GameSummary object with game details
            metadata: Optional metadata (opponent, agent, etc.)
        """
        if not self.enabled:
            return
        
        self._ensure()
        
        doc_text = game_summary.to_text()
        meta = metadata or {}
        meta.update({
            "opponent": game_summary.opponent_name,
            "outcome": game_summary.outcome,
            "score": f"{game_summary.agent_score}-{game_summary.opponent_score}"
        })
        
        self._coll.add(
            documents=[doc_text],
            ids=[game_summary.game_id],
            metadatas=[meta]
        )

    def query(
        self,
        query_text: str,
        k: int = 3,
        opponent: str | None = None
    ) -> list[dict[str, Any]]:
        """Retrieve similar games from memory.
        
        Args:
            query_text: Query string (e.g., "opponent played defensive strategy")
            k: Number of results to return
            opponent: Optional filter by opponent name
            
        Returns:
            List of similar game documents with metadata
        """
        if not self.enabled:
            return []
        
        self._ensure()
        
        where_filter = None
        if opponent:
            where_filter = {"opponent": opponent}
        
        results = self._coll.query(
            query_texts=[query_text],
            n_results=k,
            where=where_filter if opponent else None
        )
        
        retrieved = []
        if results and results.get("documents"):
            for doc, doc_id, distances, meta in zip(
                results["documents"][0],
                results["ids"][0],
                results["distances"][0],
                results["metadatas"][0]
            ):
                retrieved.append({
                    "id": doc_id,
                    "document": doc,
                    "similarity": 1 - distances,  # Convert distance to similarity
                    "metadata": meta
                })
        
        return retrieved

    def retrieve_similar_games(
        self,
        current_situation: str,
        opponent: str | None = None,
        k: int = 3
    ) -> str:
        """Retrieve and format similar past games for LLM context.
        
        Args:
            current_situation: Description of current game state
            opponent: Filter by opponent name if available
            k: Number of similar games to retrieve
            
        Returns:
            Formatted text for LLM prompt
        """
        if not self.enabled:
            return "(RAG disabled)"
        
        similar_games = self.query(current_situation, k=k, opponent=opponent)
        
        if not similar_games:
            return "(no similar games found)"
        
        text = f"Similar past games:\n"
        for i, game in enumerate(similar_games, 1):
            text += f"\n{i}. (similarity: {game['similarity']:.2%})\n"
            text += game["document"]
        
        return text

    def clear(self) -> None:
        """Clear all stored games."""
        if self._initialized and self._coll is not None:
            # Delete all documents
            all_ids = self._coll.get()["ids"]
            if all_ids:
                self._coll.delete(ids=all_ids)

    def count(self) -> int:
        """Get number of games in memory."""
        if not self._initialized or self._coll is None:
            return 0
        try:
            return self._coll.count()
        except Exception:
            return 0


# Convenience class combining both memory types
@dataclass
class AgentMemory:
    """Combined short-term + long-term memory system."""
    short_term: ShortTermMemory = field(default_factory=ShortTermMemory)
    long_term: LongTermMemory = field(default_factory=LongTermMemory)

    def reset(self) -> None:
        """Clear all memory."""
        self.short_term.clear()
        self.long_term.clear()

    def add_round(self, you_move: str, opp_move: str, outcome: int) -> None:
        """Record a game round in short-term memory."""
        self.short_term.add({
            "you_move": you_move,
            "opponent_move": opp_move,
            "outcome": outcome
        })

    def add_game_to_long_term(self, summary: GameSummary) -> None:
        """Archive a completed game to long-term memory."""
        self.long_term.add_game(summary)

    def get_context_for_llm(self, current_situation: str = "") -> str:
        """Format all memory for LLM system prompt."""
        context = "MEMORY:\n"
        context += f"Recent rounds:\n{self.short_term.to_text()}\n"
        
        if current_situation:
            context += f"\nSimilar past games:\n"
            context += self.long_term.retrieve_similar_games(current_situation, k=2)
        
        return context
