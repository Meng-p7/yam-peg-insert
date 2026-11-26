"""Replay recorded .npz demonstrations in YamPegInsertEnv."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import numpy as np

from yam_peg_insert import YamPegInsertEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("npz_path", type=Path)
    parser.add_argument("--viewer", action="store_true")
    args = parser.parse_args()

    if not args.npz_path.exists():
        raise FileNotFoundError(f"Demo file not found: {args.npz_path}")

    trajectory = np.load(args.npz_path)
    actions = trajectory["actions"]
    seed = int(trajectory["seed"]) if "seed" in trajectory else 0

    env = YamPegInsertEnv(max_steps=len(actions) + 10)
    env.reset(seed=seed)
    info: dict = {"is_success": False, "tip_to_socket_dist": float("nan")}

    if args.viewer:
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            for action in actions:
                _, _, terminated, truncated, info = env.step(action)
                viewer.sync()
                time.sleep(0.04)
                if terminated or truncated:
                    break
    else:
        for action in actions:
            _, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

    print(
        f"Replay finished | success={info['is_success']} "
        f"| dist={info['tip_to_socket_dist']:.4f} m"
    )
    env.close()


if __name__ == "__main__":
    main()
