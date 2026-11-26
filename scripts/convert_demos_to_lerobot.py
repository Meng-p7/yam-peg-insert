"""Convert recorded yam-peg-insert .npz demonstrations into a LeRobotDataset.

Replays each episode's recorded actions in YamPegInsertEnv(render_mode="rgb_array")
so both proprioceptive state and overhead_cam RGB frames are captured without
re-running the oracle. Pass --state-only to skip rendering on headless CPU setups.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np

from yam_peg_insert import YamPegInsertEnv


def build_features(obs_dim: int, action_dim: int, with_video: bool) -> dict:
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (obs_dim,),
            "names": [f"obs_{i}" for i in range(obs_dim)],
        },
        "action": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": [f"act_{i}" for i in range(action_dim)],
        },
    }
    if with_video:
        features["observation.images.overhead_cam"] = {
            "dtype": "video",
            "shape": (360, 480, 3),
            "names": ["height", "width", "channels"],
        }
    return features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demos-dir", type=Path, default=Path("demos"))
    parser.add_argument("--repo-id", type=str, default="local/yam_peg_insert")
    parser.add_argument("--root", type=Path, default=Path("datasets/yam_peg_insert"))
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--task", type=str, default="Insert the peg into the socket")
    parser.add_argument("--state-only", action="store_true")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    npz_files = sorted(args.demos_dir.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz demo files found in {args.demos_dir}")

    if args.root.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output directory {args.root} already exists. Pass --overwrite to replace."
            )
        shutil.rmtree(args.root)

    with_video = not args.state_only
    env = YamPegInsertEnv(
        max_steps=1000,
        render_mode="rgb_array" if with_video else None,
    )
    obs_dim = int(env.observation_space.shape[0])
    action_dim = int(env.action_space.shape[0])

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        root=args.root,
        robot_type="i2rt_yam",
        features=build_features(obs_dim, action_dim, with_video=with_video),
        use_videos=with_video,
    )

    kept_episodes = 0
    for npz_path in npz_files:
        traj = np.load(npz_path)
        actions = traj["actions"].astype(np.float32)
        seed = int(traj["seed"]) if "seed" in traj else 0

        obs, _ = env.reset(seed=seed)
        frames_buffer = []
        last_info = {"is_success": False}

        for act in actions:
            frame = {
                "observation.state": obs.astype(np.float32),
                "action": act,
                "task": args.task,
            }
            if with_video:
                rgb = env.render()
                if rgb is None:
                    raise RuntimeError("env.render() returned None in rgb_array mode.")
                frame["observation.images.overhead_cam"] = rgb

            frames_buffer.append(frame)
            obs, _, terminated, truncated, last_info = env.step(act)
            if terminated or truncated:
                break

        if not last_info.get("is_success", False):
            print(
                f"[Skip] {npz_path.name} did not reach success under current criteria."
            )
            continue

        for frame in frames_buffer:
            dataset.add_frame(frame)
        dataset.save_episode()
        kept_episodes += 1
        print(f"[Saved] {npz_path.name} ({len(frames_buffer)} frames)")

    dataset.finalize()
    env.close()
    print(
        f"Done! Saved {kept_episodes}/{len(npz_files)} successful episodes to {args.root}"
    )

    if args.push_to_hub and kept_episodes > 0:
        dataset.push_to_hub()
        print(f"Pushed dataset to Hugging Face Hub: {args.repo_id}")


if __name__ == "__main__":
    main()
