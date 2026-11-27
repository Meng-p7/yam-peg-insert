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
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--out-dir", type=Path, default=Path("demos"))
    parser.add_argument(
        "--max-j4",
        type=float,
        default=1.48,
        help="Skip recordings whose joint4 peak during the insert phase exceeds this.",
    )
    args = parser.parse_args()

    env = YamPegInsertEnv(max_steps=args.max_steps)
    oracle = ScriptedPegOracle(env)

    if args.record:
        args.out_dir.mkdir(parents=True, exist_ok=True)

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=ep)
        obs_list, act_list, rew_list = [], [], []
        max_j4_insert = 0.0
        for step in range(env.max_steps):
            # joint4 peaks during grasp/lift by construction; only the insert
            # phase is actionable for the wrist-limit filter.
            if oracle.PHASE_NAMES[oracle._phase] == "insert":
                max_j4_insert = max(max_j4_insert, float(obs[env.nu + 3]))
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
            f"success={info['is_success']} max_j4_insert={max_j4_insert:.3f} rad"
        )
        if not info["is_success"]:
            rel = (
                env.data.site_xpos[env._peg_tip_id]
                - env.data.site_xpos[env._socket_bottom_id]
            )
            print(
                f"  oracle_status: {oracle.last_status} | "
                f"dropped={info['dropped']} tipped={info['tipped']} | "
                f"tip_rel=({rel[0] * 1000:+.1f}, {rel[1] * 1000:+.1f}, "
                f"{rel[2] * 1000:+.1f}) mm"
            )
        if args.record:
            if not info["is_success"] or max_j4_insert > args.max_j4:
                print(
                    f"  -> skip recording: success={info['is_success']}, "
                    f"max_j4_insert={max_j4_insert:.3f} rad"
                )
                continue
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
