"""Gymnasium environment for YAM Peg-in-Hole manipulation."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

from yam_peg_insert.scene import load_peg_insert_model


class YamPegInsertEnv(gym.Env[np.ndarray, np.ndarray]):
    """Single-arm YAM peg-in-hole environment running natively on CPU MuJoCo."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 25}

    def __init__(
        self,
        max_steps: int = 250,
        frame_skip: int = 10,
        action_scale: float = 0.05,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.model, self.data = load_peg_insert_model()
        self.max_steps = max_steps
        self.frame_skip = frame_skip
        self.action_scale = action_scale
        self.render_mode = render_mode
        self._step_count = 0

        self.nu = self.model.nu
        self._peg_tip_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "peg_tip"
        )
        self._socket_bottom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "socket_bottom"
        )
        self._peg_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "peg_freejoint"
        )

        # Observation: [ctrl (nu), qpos_arm (nu), qvel_arm (nu), peg_to_socket (3), contact_count (1)]
        obs_dim = self.nu * 3 + 4
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.nu,), dtype=np.float32
        )
        self._renderer: mujoco.Renderer | None = None

    def _get_obs(self) -> np.ndarray:
        peg_tip = self.data.site_xpos[self._peg_tip_id]
        socket_bottom = self.data.site_xpos[self._socket_bottom_id]
        rel = socket_bottom - peg_tip

        qpos_arm = self.data.qpos[: self.nu]
        qvel_arm = self.data.qvel[: self.nu]
        ctrl = self.data.ctrl[: self.nu]
        ncon = np.array([float(self.data.ncon)], dtype=np.float64)

        return np.concatenate([ctrl, qpos_arm, qvel_arm, rel, ncon]).astype(np.float32)

    def _compute_reward(self) -> tuple[float, bool, dict[str, Any]]:
        peg_tip = self.data.site_xpos[self._peg_tip_id]
        socket_bottom = self.data.site_xpos[self._socket_bottom_id]
        dist = float(np.linalg.norm(socket_bottom - peg_tip))

        # Smooth distance reward + success bonus when tip reaches bottom of socket
        reward = -dist - 0.01 * float(np.sum(np.square(self.data.ctrl)))
        inserted = dist < 0.015
        if inserted:
            reward += 5.0

        info = {"tip_to_socket_dist": dist, "is_success": inserted}
        return reward, inserted, info

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self._step_count = 0

        # Small randomization on peg initial XY position
        if self._peg_joint_id >= 0:
            qadr = self.model.jnt_qposadr[self._peg_joint_id]
            dx, dy = self.np_random.uniform(-0.02, 0.02, size=2)
            self.data.qpos[qadr] += dx
            self.data.qpos[qadr + 1] += dy

        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.clip(action, -1.0, 1.0)
        ctrl_range = self.model.actuator_ctrlrange
        target = self.data.ctrl + action * self.action_scale
        self.data.ctrl[:] = np.clip(target, ctrl_range[:, 0], ctrl_range[:, 1])

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1
        obs = self._get_obs()
        reward, terminated, info = self._compute_reward()
        truncated = self._step_count >= self.max_steps
        return obs, reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=360, width=480)
        self._renderer.update_scene(self.data, camera="overhead_cam")
        return self._renderer.render()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
