"""Damped-least-squares Jacobian IK oracle for scripted demos and data collection."""

from __future__ import annotations

import mujoco
import numpy as np

from yam_peg_insert.env import YamPegInsertEnv


class ScriptedPegOracle:
    """Simple state-machine + Jacobian IK policy to guide the arm toward the peg/socket."""

    def __init__(self, env: YamPegInsertEnv, damping: float = 1e-2) -> None:
        self.env = env
        self.damping = damping
        self._ee_body_id = self._find_ee_body_id()
        self._peg_body_id = mujoco.mj_name2id(
            env.model, mujoco.mjtObj.mjOBJ_BODY, "peg"
        )
        self._socket_site_id = env._socket_bottom_id

    def _find_ee_body_id(self) -> int:
        # Fallback to last non-world arm body before freejoint bodies
        for candidate in ("link_6", "ee_link", "gripper_base"):
            bid = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_BODY, candidate)
            if bid >= 0:
                return bid
        return max(1, self.env.model.nbody - 3)

    def predict(self, step_idx: int) -> np.ndarray:
        model = self.env.model
        data = self.env.data
        nu = self.env.nu

        ee_pos = data.xpos[self._ee_body_id]
        peg_pos = data.xpos[self._peg_body_id]
        socket_pos = data.site_xpos[self._socket_site_id]

        # 3-phase target: hover above peg -> approach peg -> move above socket
        if step_idx < 60:
            target_pos = peg_pos + np.array([0.0, 0.0, 0.08])
            gripper_cmd = 1.0
        elif step_idx < 120:
            target_pos = peg_pos + np.array([0.0, 0.0, 0.01])
            gripper_cmd = -1.0
        else:
            target_pos = socket_pos + np.array([0.0, 0.0, 0.06])
            gripper_cmd = -1.0

        err = target_pos - ee_pos
        jacp = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, jacp, None, self._ee_body_id)

        arm_dof = max(1, nu - 1)
        j_arm = jacp[:, :arm_dof]
        jjt = j_arm @ j_arm.T + (self.damping**2) * np.eye(3)
        dq = j_arm.T @ np.linalg.solve(jjt, err)

        action = np.zeros(nu, dtype=np.float32)
        action[:arm_dof] = np.clip(dq / max(self.env.action_scale, 1e-6), -1.0, 1.0)
        if nu > arm_dof:
            action[arm_dof:] = gripper_cmd
        return action
