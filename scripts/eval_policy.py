"""Evaluate a Colab-trained SB3 PPO checkpoint locally on Ubuntu 22.04 CPU."""

from __future__ import annotations

import argparse
from pathlib import Path

from yam_peg_insert import YamPegInsertEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()

    from stable_baselines3 import PPO

    env = YamPegInsertEnv()
    model = PPO.load(str(args.checkpoint), device="cpu")

    successes = 0
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=ep)
        total_reward = 0.0
        for _ in range(env.max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        successes += int(info["is_success"])
        print(
            f"[Eval Ep {ep}] reward={total_reward:.3f} "
            f"success={info['is_success']} dist={info['tip_to_socket_dist']:.4f} m"
        )

    print(f"Success rate: {successes}/{args.episodes}")
    env.close()


if __name__ == "__main__":
    main()
