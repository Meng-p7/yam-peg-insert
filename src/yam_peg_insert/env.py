"""Gymnasium environment for YAM Peg-in-Hole manipulation."""

from __future__ import annotations

from typing import Any, ClassVar

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from yam_peg_insert.scene import load_peg_insert_model

# Gripper opening used in the home pose (matches the finger actuator range).
GRIPPER_OPEN = 0.0375

# Workbench footprint used to decide when the peg has fallen off the table.
_TABLE_CENTER_XY = np.array([0.35, 0.0])
_TABLE_RADIUS = 0.45


class YamPegInsertEnv(gym.Env[np.ndarray, np.ndarray]):
    """Single-arm YAM peg-in-hole environment running natively on CPU MuJoCo."""

    metadata: ClassVar[dict[str, Any]] = {
        "render_modes": ["rgb_array"],
        "render_fps": 25,
    }

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
        self._grasp_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "grasp_site"
        )
        self._peg_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "peg"
        )
        self._peg_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "peg_freejoint"
        )
        self._gripper_act_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper"
        )
        self._home_key_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_KEY, "home"
        )
        self._left_finger_qadr = self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "left_finger")
        ]
        self._right_finger_qadr = self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "right_finger")
        ]

        # Every robot joint except the peg's freejoint (hinge/slide -> 1 DoF each).
        qpos_idx: list[int] = []
        qvel_idx: list[int] = []
        for joint_id in range(self.model.njnt):
            if joint_id == self._peg_joint_id:
                continue
            jtype = int(self.model.jnt_type[joint_id])
            if jtype in (
                int(mujoco.mjtJoint.mjJNT_HINGE),
                int(mujoco.mjtJoint.mjJNT_SLIDE),
            ):
                qpos_idx.append(int(self.model.jnt_qposadr[joint_id]))
                qvel_idx.append(int(self.model.jnt_dofadr[joint_id]))
        self._robot_qpos_idx = np.asarray(qpos_idx, dtype=np.int32)
        self._robot_qvel_idx = np.asarray(qvel_idx, dtype=np.int32)

        # Observation: ctrl (nu), qpos_robot, qvel_robot, grasp->peg (3),
        # peg_tip->socket (3), peg z-axis (3), contact count (1).
        obs_dim = self.nu + 2 * len(self._robot_qpos_idx) + 10
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
        peg_pos = self.data.xpos[self._peg_body_id]
        grasp_pos = self.data.site_xpos[self._grasp_site_id]
        peg_z_axis = self.data.xmat[self._peg_body_id].reshape(3, 3)[:, 2]

        qpos_robot = self.data.qpos[self._robot_qpos_idx]
        qvel_robot = self.data.qvel[self._robot_qvel_idx]
        ctrl = self.data.ctrl[: self.nu]
        ncon = np.array([float(self.data.ncon)], dtype=np.float64)

        return np.concatenate(
            [
                ctrl,
                qpos_robot,
                qvel_robot,
                peg_pos - grasp_pos,
                socket_bottom - peg_tip,
                peg_z_axis,
                ncon,
            ]
        ).astype(np.float32)

    def _compute_reward(self) -> tuple[float, bool, dict[str, Any]]:
        peg_tip = self.data.site_xpos[self._peg_tip_id]
        socket_bottom = self.data.site_xpos[self._socket_bottom_id]
        dist = float(np.linalg.norm(socket_bottom - peg_tip))

        reward = -dist - 0.01 * float(np.sum(np.square(self.data.ctrl)))

        rel = peg_tip - socket_bottom
        peg_z_axis = self.data.xmat[self._peg_body_id].reshape(3, 3)[:, 2]
        upright = bool(abs(float(peg_z_axis[2])) > 0.92)
        # The hole is 36 mm wide and the peg 24 mm, so the real lateral clearance
        # is only +-6 mm; anything looser lets a peg resting on the rim pass.
        inserted = bool(
            abs(float(rel[0])) < 0.0055
            and abs(float(rel[1])) < 0.0055
            and -0.004 < float(rel[2]) < 0.006
            and upright
        )

        peg_pos = self.data.xpos[self._peg_body_id]
        tipped = bool(abs(float(peg_z_axis[2])) < 0.50 and float(peg_pos[2]) < 0.035)
        dropped = bool(
            peg_pos[2] < -0.005
            or np.linalg.norm(peg_pos[:2] - _TABLE_CENTER_XY) > _TABLE_RADIUS
            or tipped
        )

        if inserted:
            reward += 5.0
        if dropped:
            reward -= 2.0

        info = {
            "tip_to_socket_dist": dist,
            "is_success": inserted,
            "dropped": dropped,
            "tipped": tipped,
            "peg_vertical": upright,
        }
        return reward, inserted or dropped, info

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if self._home_key_id >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, self._home_key_id)
        else:
            mujoco.mj_resetData(self.model, self.data)

        # The home keyframe holds the fingers closed, which makes the two finger
        # collision sets interpenetrate at t=0; start open instead.
        self.data.qpos[self._left_finger_qadr] = GRIPPER_OPEN
        self.data.qpos[self._right_finger_qadr] = -GRIPPER_OPEN
        self.data.ctrl[self._gripper_act_id] = GRIPPER_OPEN
        self._step_count = 0

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
