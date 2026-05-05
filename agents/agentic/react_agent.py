"""ReAct loop for the Agentic LLM — LangGraph-backed with Python fallback.

The agent runs Reason → Act → Observe → Reflect each turn using LangGraph's
StateGraph as the execution backbone.  If LangGraph is not installed, the same
loop executes as a plain Python while-loop so the module always works.

LangGraph state machine:
  START → reason → [act | tool_call] → observe → END
  tool_call loops back to reason with TOOL_RESULT appended.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, TypedDict

import numpy as np

from agents.agentic.memory import ShortTermMemory
from agents.agentic.tools import (
    GameMemory,
    analyze_opponent,
    get_game_history,
    predict_next_move,
)
from agents.base import Agent
from environments.rps_plus import Move, N_MOVES


LLM = Callable[[str], str]


SYSTEM_PROMPT = """You are an agent playing RPS+ (rock-paper-scissors plus).
Moves: ROCK PAPER SCISSORS LIZARD POWER RECHARGE.
You may call tools by writing a single line:
  TOOL: <name>
Available tools: get_game_history, analyze_opponent, predict_next_move.
After tools return, respond with a final line:
  ACTION: <MOVE_NAME>
"""


@dataclass
class ReActTrace:
    steps: list[str] = field(default_factory=list)


def fallback_llm(prompt: str) -> str:
    """Deterministic offline LLM stand-in."""
    if "TOOL_RESULT" not in prompt:
        return "TOOL: predict_next_move"
    m = re.search(r"prediction['\": ]+([A-Z]+)", prompt)
    pred = m.group(1) if m else None
    counter = {
        "ROCK": "PAPER", "PAPER": "SCISSORS", "SCISSORS": "ROCK",
        "LIZARD": "ROCK", "POWER": "RECHARGE", "RECHARGE": "POWER",
    }
    move = counter.get(pred, "ROCK")
    return f"ACTION: {move}"


# ---------------------------------------------------------------------------
# LangGraph state schema
# ---------------------------------------------------------------------------

class AgentState(TypedDict, total=False):
    prompt: str
    response: str
    trace: list[str]
    action: int
    done: bool
    legal_moves: list[int]


# ---------------------------------------------------------------------------
# LangGraph-backed agent (falls back to plain loop when unavailable)
# ---------------------------------------------------------------------------

def _build_langgraph_app(llm: LLM, memory: GameMemory, max_steps: int) -> Any:
    """Build a LangGraph StateGraph for the ReAct loop.

    Returns a compiled app with an .invoke(state) method, or None if
    langgraph is not installed.
    """
    try:
        from langgraph.graph import END, StateGraph
    except ImportError:
        return None

    def _run_tool(name: str) -> str:
        if name == "get_game_history":
            return repr(get_game_history(memory, last_n=5))
        if name == "analyze_opponent":
            return repr(analyze_opponent(memory))
        if name == "predict_next_move":
            return repr(predict_next_move(memory))
        return f"unknown tool: {name}"

    def _parse_action(text: str, legal: list[int]) -> int:
        upper = text.split(":", 1)[-1].strip().upper()
        for m in Move:
            if m.name in upper and int(m) in legal:
                return int(m)
        return int(legal[0])

    def reason_node(state: AgentState) -> AgentState:
        response = llm(state["prompt"]).strip()
        trace    = list(state.get("trace", []))
        trace.append(response)
        return {**state, "response": response, "trace": trace}

    def router(state: AgentState) -> str:
        resp = state.get("response", "")
        if resp.startswith("ACTION:"):
            return "finish"
        if resp.startswith("TOOL:"):
            return "tool_call"
        return "finish"

    def tool_node(state: AgentState) -> AgentState:
        name   = state["response"].split(":", 1)[1].strip()
        result = _run_tool(name)
        new_prompt = state["prompt"] + f"\n{state['response']}\nTOOL_RESULT: {result}\n"
        return {**state, "prompt": new_prompt}

    def finish_node(state: AgentState) -> AgentState:
        resp   = state.get("response", "")
        legal  = state.get("legal_moves", list(range(N_MOVES)))
        action = _parse_action(resp, legal)
        return {**state, "action": action, "done": True}

    graph = StateGraph(AgentState)
    graph.add_node("reason",    reason_node)
    graph.add_node("tool_call", tool_node)
    graph.add_node("finish",    finish_node)

    graph.set_entry_point("reason")
    graph.add_conditional_edges(
        "reason",
        router,
        {"finish": "finish", "tool_call": "tool_call"},
    )
    graph.add_edge("tool_call", "reason")
    graph.add_edge("finish", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# ReActAgent
# ---------------------------------------------------------------------------

class ReActAgent(Agent):
    """ReAct agent backed by LangGraph (with transparent Python fallback)."""

    name = "agentic_react"

    def __init__(
        self,
        llm: LLM | None = None,
        max_steps: int = 4,
        history_window: int = 10,
    ) -> None:
        self.llm          = llm or fallback_llm
        self.max_steps    = max_steps
        self.short_term   = ShortTermMemory(window=history_window)
        self._memory      = GameMemory(agent_moves=[], opponent_moves=[], outcomes=[])
        self.last_trace: ReActTrace | None = None
        # Build LangGraph app lazily (attempt once, cache result).
        self._lg_app = _build_langgraph_app(self.llm, self._memory, max_steps)

    def reset(self, seed: int | None = None) -> None:
        self.short_term.clear()
        self._memory = GameMemory(agent_moves=[], opponent_moves=[], outcomes=[])
        # Rebuild the graph pointing at the fresh memory object.
        self._lg_app = _build_langgraph_app(self.llm, self._memory, self.max_steps)

    def observe(self, agent_move: int, opponent_move: int, outcome: int) -> None:
        self._memory.agent_moves.append(int(agent_move))
        self._memory.opponent_moves.append(int(opponent_move))
        self._memory.outcomes.append(int(outcome))
        self.short_term.add({
            "you_move":      Move(agent_move).name,
            "opponent_move": Move(opponent_move).name,
            "outcome":       outcome,
        })

    def act(self, obs: np.ndarray, info: dict[str, Any]) -> int:
        prompt = self._initial_prompt(obs, info)
        legal  = info.get("legal_moves") or list(range(N_MOVES))

        if self._lg_app is not None:
            return self._act_langgraph(prompt, legal)
        return self._act_python(prompt, info)

    # ---------------------------------------------------------------- LangGraph path

    def _act_langgraph(self, prompt: str, legal: list[int]) -> int:
        """Execute the ReAct loop via LangGraph StateGraph."""
        try:
            state = self._lg_app.invoke({
                "prompt":      prompt,
                "trace":       [],
                "legal_moves": legal,
                "done":        False,
            })
            self.last_trace = ReActTrace(steps=state.get("trace", []))
            return int(state.get("action", legal[0]))
        except Exception:
            # LangGraph failed at runtime — fall back to Python loop.
            return self._act_python(prompt, {"legal_moves": legal})

    # ---------------------------------------------------------------- Python fallback path

    def _act_python(self, prompt: str, info: dict[str, Any]) -> int:
        trace = ReActTrace()
        for _ in range(self.max_steps):
            response = self.llm(prompt).strip()
            trace.steps.append(response)
            if response.startswith("ACTION:"):
                self.last_trace = trace
                return self._parse_action(response, info)
            if response.startswith("TOOL:"):
                tool_name = response.split(":", 1)[1].strip()
                result    = self._run_tool(tool_name)
                prompt   += f"\n{response}\nTOOL_RESULT: {result}\n"
                continue
            break
        self.last_trace = trace
        legal = info.get("legal_moves") or list(range(N_MOVES))
        return int(legal[0])

    # ---------------------------------------------------------------- internals

    def _initial_prompt(self, obs: np.ndarray, info: dict[str, Any]) -> str:
        agent_e = int(round(obs[0] * 5))
        opp_e   = int(round(obs[1] * 5))
        legal   = ", ".join(Move(m).name for m in (info.get("legal_moves") or range(N_MOVES)))
        recent  = self.short_term.recent()
        recent_str = (
            "\n".join(f"  you={r['you_move']} opp={r['opponent_move']} outcome={r['outcome']:+d}" for r in recent)
            or "  (none)"
        )
        return (
            f"{SYSTEM_PROMPT}\n\n"
            f"State: your_energy={agent_e} opp_energy={opp_e} legal_moves=[{legal}]\n"
            f"Recent:\n{recent_str}\n\n"
            f"Decide the next move. Use a tool first if useful."
        )

    def _run_tool(self, name: str) -> str:
        if name == "get_game_history":
            return repr(get_game_history(self._memory, last_n=5))
        if name == "analyze_opponent":
            return repr(analyze_opponent(self._memory))
        if name == "predict_next_move":
            return repr(predict_next_move(self._memory))
        return f"unknown tool: {name}"

    def _parse_action(self, response: str, info: dict[str, Any]) -> int:
        text  = response.split(":", 1)[1].strip().upper()
        legal = info.get("legal_moves") or list(range(N_MOVES))
        for m in Move:
            if m.name in text and int(m) in legal:
                return int(m)
        return int(legal[0])
