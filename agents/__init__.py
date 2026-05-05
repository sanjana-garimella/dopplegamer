from agents.agentic_llm import AgenticLLM
from agents.adaptive_router import AdaptiveRouterAgent
from agents.base import AGENT_REGISTRY, Agent, HeuristicAgent, OptimalAgent, RandomAgent, register_agent
from agents.impostor.lstm import LSTMImpostor
from agents.impostor.ngram import NGramImpostor
from agents.impostor.trainer import ImpostorTrainer
from agents.profile_counter import ProfileCounterAgent
from agents.rl import BCRLAgent as BCPlusRLAgent, PPOAgent as RLAgent
from agents.sft.model import SFTAgent

# Register advanced agents
register_agent("sft", SFTAgent)
register_agent("rl", RLAgent)
register_agent("ppo", RLAgent)
register_agent("bcrl", BCPlusRLAgent)
register_agent("bc_rl", BCPlusRLAgent)
register_agent("agentic", AgenticLLM)
register_agent("profile_counter", ProfileCounterAgent)
register_agent("adaptive_router", AdaptiveRouterAgent)

__all__ = [
    "Agent",
    "RandomAgent",
    "HeuristicAgent",
    "OptimalAgent",
    "SFTAgent",
    "RLAgent",
    "BCPlusRLAgent",
    "AgenticLLM",
    "ProfileCounterAgent",
    "AdaptiveRouterAgent",
    "AGENT_REGISTRY",
    # Impostor Effect
    "NGramImpostor",
    "LSTMImpostor",
    "ImpostorTrainer",
]
