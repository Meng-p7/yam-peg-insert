"""Closed-loop evaluation of a LeRobot ACT checkpoint on the local CPU env.

Loads the ACT checkpoint (``.../checkpoints/last/pretrained_model``) together
with its pre/post-processors, renders the ``overhead_cam`` view at every step and
runs the policy in closed loop. The Colab checkpoint stores ``cuda`` in its
processor configs, so both pipelines are rebuilt on CPU here.

Example:
    python scripts/eval_policy.py \
        --checkpoint ~/act_yam_peg_pretrained/outputs/train/act_yam_peg/checkpoints/last/pretrained_model \
        --episodes 50
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from yam_peg_insert import YamPegInsertEnv


def _failure_reason(info: dict, truncated: bool) -> str:
    if info.get("is_success", False):
        return "success"
    if info.get("dropped", False):
        return "tipped" if info.get("tipped", False) else "dropped"
    return "timeout" if truncated else "terminated"


def evaluate(
    checkpoint: Path,
    episodes: int,
    task: str,
    max_steps: int | None,
    n_action_steps: int | None,
    te_coeff: float | None,
) -> None:
    import torch
    from lerobot.policies.act.modeling_act import ACTPolicy, ACTTemporalEnsembler
    from lerobot.policies.factory import make_pre_post_processors

    policy = ACTPolicy.from_pretrained(str(checkpoint))
    policy.to("cpu").eval()
    if te_coeff is not None:
        policy.config.n_action_steps = 1
        policy.config.temporal_ensemble_coeff = te_coeff
        policy.temporal_ensembler = ACTTemporalEnsembler(
            te_coeff, policy.config.chunk_size
        )
    elif n_action_steps is not None:
        policy.config.n_action_steps = n_action_steps
    policy.reset()
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": "cpu"}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )

    env = YamPegInsertEnv(
        **({"max_steps": max_steps} if max_steps else {}), render_mode="rgb_array"
    )
    successes = 0
    success_steps: list[int] = []
    reasons: dict[str, int] = {}
    inference_s = 0.0
    try:
        for ep in range(episodes):
            obs, _ = env.reset(seed=ep)
            policy.reset()
            info: dict = {}
            truncated = False
            steps = 0
            for steps in range(1, env.max_steps + 1):
                rgb = env.render()
                if rgb is None:
                    raise RuntimeError("env.render() returned None in rgb_array mode")
                batch = {
                    "observation.state": torch.from_numpy(obs).unsqueeze(0),
                    "observation.images.overhead_cam": torch.from_numpy(rgb)
                    .permute(2, 0, 1)
                    .float()
                    .div(255.0)
                    .unsqueeze(0),
                    "task": [task],
                }
                t0 = time.perf_counter()
                with torch.inference_mode():
                    action = postprocessor(policy.select_action(preprocessor(batch)))
                inference_s += time.perf_counter() - t0
                obs, _, terminated, truncated, info = env.step(
                    action.squeeze(0).cpu().numpy()
                )
                if terminated or truncated:
                    break

            reason = _failure_reason(info, truncated)
            reasons[reason] = reasons.get(reason, 0) + 1
            success = info.get("is_success", False)
            if success:
                successes += 1
                success_steps.append(steps)
            dist = info.get("tip_to_socket_dist", float("nan"))
            print(
                f"[Eval Ep {ep}] success={success} steps={steps} "
                f"dist={dist:.4f} m result={reason}"
            )
    finally:
        env.close()

    print(f"\nSuite: {episodes} episodes, {env.max_steps} max steps/episode")
    print(
        "Replanning: "
        + (
            f"temporal ensembling coeff={te_coeff}, 1 action/inference"
            if te_coeff is not None
            else f"{policy.config.n_action_steps} actions/inference"
        )
    )
    print(f"Success rate: {successes}/{episodes} ({successes / episodes:.0%})")
    if success_steps:
        print(f"Mean success steps: {np.mean(success_steps):.1f}")
    print(f"Outcomes: {dict(sorted(reasons.items()))}")
    print(f"Policy inference time: {inference_s:.1f} s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--task", type=str, default="Insert the peg into the socket")
    parser.add_argument(
        "--n-action-steps",
        type=int,
        default=None,
        help="Replan every N steps instead of the checkpoint's 100 (no retraining).",
    )
    parser.add_argument(
        "--te-coeff",
        type=float,
        default=None,
        help="Enable temporal ensembling with this coefficient (implies 1 step).",
    )
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}. "
            "Train on Colab first and copy the exported pretrained_model here."
        )

    evaluate(
        args.checkpoint,
        args.episodes,
        args.task,
        args.max_steps,
        args.n_action_steps,
        args.te_coeff,
    )


if __name__ == "__main__":
    main()
