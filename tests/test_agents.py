from agents import AgenticLLM, BCPlusRLAgent, RLAgent, SFTAgent
from environments.rps_plus import RPSPlusEnv
from environments.tic_tac_toe import TicTacToeEnv


def test_agents_emit_legal_actions():
    env = RPSPlusEnv(max_turns=5, seed=1)
    obs, info = env.reset(seed=1)
    agents = [SFTAgent(seed=1), RLAgent(seed=1), BCPlusRLAgent(seed=1), AgenticLLM()]
    for agent in agents:
        a = agent.act(obs, info)
        assert a in info["legal_moves"]


def test_rps_trained_agents_fallback_on_non_rps_observations():
    env = TicTacToeEnv(max_moves=5, seed=1)
    obs, info = env.reset(seed=1)

    for agent in [SFTAgent(seed=1), RLAgent(seed=1), BCPlusRLAgent(seed=1)]:
        observe = getattr(agent, "observe", None)
        if callable(observe):
            observe(7, 8, 0)
        action = agent.act(obs, info)
        assert action in info["legal_moves"]
