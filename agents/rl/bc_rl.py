"""BC+RL agent: PPO fine-tuning on top of an SFT base model with dual reward.

The SFT model provides reference logits over moves; the env provides the win
reward. We blend the two with `DualRewardConfig`. The RL policy here is a
small MLP for tractability — the SFT model is consulted purely as an oracle
for the BC bonus.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import gymnasium as gym

from agents.rl.rewards import DualRewardConfig, combine
from environments.legal_action_wrapper import LegalActionWrapper
from environments.rps_plus import N_MOVES, RPSPlusEnv


class DualRewardEnv(gym.Env):
    """Light gym wrapper that injects the BC bonus before returning reward."""

    metadata = {"render_modes": []}

    def __init__(self, env: RPSPlusEnv, ref_policy_logits, cfg: DualRewardConfig) -> None:
        super().__init__()
        # Clamp illegal actions (e.g. POWER at low energy) before RPS+ raises.
        self.env = LegalActionWrapper(env)
        self.ref = ref_policy_logits
        self.cfg = cfg
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        ref_logits = self.ref(obs, info) if self.ref else None
        # Use the action actually taken if the wrapper clamped it.
        taken = int(info.get("action", action)) if isinstance(info, dict) else int(action)
        new_r = combine(r, taken, ref_logits, self.cfg)
        return obs, new_r, term, trunc, info

    def render(self):
        return self.env.render()

    def close(self):
        return None


def uniform_logits(_obs: np.ndarray, _info: dict[str, Any]) -> np.ndarray:
    return np.zeros(N_MOVES, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft-checkpoint", default=None, help="path to SFT checkpoint (optional)")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--beta", type=float, default=0.4)
    parser.add_argument("--output", default="checkpoints/bc_rl")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = DualRewardConfig(alpha=args.alpha, beta=args.beta)

    ref = uniform_logits
    if args.sft_checkpoint:
        from agents.sft.model import SFTAgent

        sft = SFTAgent(args.sft_checkpoint)

        def ref(obs, info):
            # Approximate logits via greedy preference: 1.0 on the SFT pick, 0 elsewhere.
            choice = sft.act(obs, info)
            v = np.zeros(N_MOVES, dtype=np.float32)
            v[choice] = 1.0
            return v

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    def make():
        env = RPSPlusEnv(seed=args.seed)
        return DualRewardEnv(env, ref, cfg)

    vec = DummyVecEnv([make])
    model = PPO(
        "MlpPolicy",
        vec,
        verbose=1,
        seed=args.seed,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
    )
    model.learn(total_timesteps=args.timesteps)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output)


if __name__ == "__main__":
    main()
