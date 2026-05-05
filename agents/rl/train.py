"""PPO training for the RL agent. Uses Stable-Baselines3 against the RPS+ env.

Run:
    python -m agents.rl.train --timesteps 100000 --output checkpoints/ppo_best
"""

from __future__ import annotations

import argparse
from pathlib import Path

from environments.rps_plus import RPSPlusEnv


def make_env(seed: int = 0):
    def thunk():
        env = RPSPlusEnv(seed=seed)
        return env

    return thunk


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--output", default="checkpoints/ppo_best")
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-freq", type=int, default=10000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    args = parser.parse_args()

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import EvalCallback

    vec = DummyVecEnv([make_env(args.seed + i) for i in range(args.n_envs)])
    eval_env = DummyVecEnv([make_env(args.seed + 100)])  # Separate eval env
    eval_callback = EvalCallback(eval_env, best_model_save_path=args.output + "_best", 
                                 log_path="./logs/", eval_freq=args.eval_freq, 
                                 n_eval_episodes=args.eval_episodes)
    
    model = PPO(
        "MlpPolicy",
        vec,
        verbose=1,
        seed=args.seed,
        n_steps=2048,       # enough steps to see full episodes (max_turns=100)
        batch_size=64,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,      # entropy bonus to prevent premature convergence
    )
    model.learn(total_timesteps=args.timesteps, callback=eval_callback)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output)
    print(f"saved PPO model to {args.output}.zip")


if __name__ == "__main__":
    main()
