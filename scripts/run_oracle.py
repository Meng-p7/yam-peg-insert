"""Run scripted Jacobian IK oracle and optionally save trajectories to demos/."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from yam_peg_insert import ScriptedPegOracle, YamPegInsertEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=Path("demos"))
    args = parser.parse_args()

    env = YamPegInsertEnv(max_steps=300)
    oracle = ScriptedPegOracle(env)

    if args.record:
        args.out_dir.mkdir(parents=True, exist_ok=True)

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=ep)
        obs_list, act_list, rew_list = [], [], []
        for step in range(env.max_steps):
            action = oracle.predict(step)
            obs_list.append(obs)
            act_list.append(action)
            obs, reward, terminated, truncated, info = env.step(action)
            rew_list.append(reward)
            if terminated or truncated:
                break

        print(
            f"[Oracle Ep {ep}] steps={len(act_list)} "
            f"reward={sum(rew_list):.3f} dist={info['tip_to_socket_dist']:.4f} m "
            f"success={info['is_success']}"
        )
        if args.record:
            out_file = args.out_dir / f"oracle_ep_{ep:03d}.npz"
            np.savez_compressed(
                out_file,
                obs=np.asarray(obs_list, dtype=np.float32),
                actions=np.asarray(act_list, dtype=np.float32),
                rewards=np.asarray(rew_list, dtype=np.float32),
                seed=np.int64(ep),
            )

    env.close()


if __name__ == "__main__":
    main()
