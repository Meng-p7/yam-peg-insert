"""Headless CPU smoke test for YamPegInsertEnv."""

from yam_peg_insert import YamPegInsertEnv


def main() -> None:
    env = YamPegInsertEnv()
    obs, _ = env.reset(seed=42)
    print(f"Reset OK | obs shape={obs.shape}, action dim={env.action_space.shape}")

    total_reward = 0.0
    for _ in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break

    print(
        f"50-step CPU rollout OK | total_reward={total_reward:.3f} "
        f"| tip_to_socket_dist={info['tip_to_socket_dist']:.4f} m"
    )
    env.close()


if __name__ == "__main__":
    main()
