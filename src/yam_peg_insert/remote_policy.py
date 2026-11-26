"""Lightweight HTTP bridge for Cloud GPU inference <-> Local Ubuntu CPU MuJoCo."""

from __future__ import annotations

import argparse
import json
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from yam_peg_insert.env import YamPegInsertEnv


class _PolicyHandler(BaseHTTPRequestHandler):
    action_dim: int = 7

    def do_POST(self) -> None:
        if self.path != "/act":
            self.send_response(404)
            self.end_headers()
            return

        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        obs = np.asarray(payload["obs"], dtype=np.float32)

        # Placeholder zero-mean policy; swap in OpenPI / SB3 PPO inference here
        action = np.tanh(obs[: self.action_dim] * 0.1).tolist()

        body = json.dumps({"action": action}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:
        return


def run_server(host: str = "0.0.0.0", port: int = 8765, action_dim: int = 7) -> None:
    _PolicyHandler.action_dim = action_dim
    server = ThreadingHTTPServer((host, port), _PolicyHandler)
    print(f"PolicyServer listening on http://{host}:{port}/act")
    server.serve_forever()


def query_policy(url: str, obs: np.ndarray) -> np.ndarray:
    req = urllib.request.Request(
        f"{url.rstrip('/')}/act",
        data=json.dumps({"obs": obs.tolist()}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=5.0) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return np.asarray(data["action"], dtype=np.float32)


def run_client(url: str, episodes: int = 1) -> None:
    env = YamPegInsertEnv(max_steps=50)
    for ep in range(episodes):
        obs, _ = env.reset(seed=ep)
        total_reward = 0.0
        for _ in range(50):
            action = query_policy(url, obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        print(
            f"Episode {ep}: total_reward={total_reward:.3f}, "
            f"dist={info['tip_to_socket_dist']:.4f} m"
        )
    env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["server", "client"], required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--episodes", type=int, default=1)
    args = parser.parse_args()

    if args.mode == "server":
        run_server(host=args.host, port=args.port)
    else:
        run_client(url=args.url, episodes=args.episodes)


if __name__ == "__main__":
    main()
