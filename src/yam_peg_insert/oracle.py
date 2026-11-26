"""Damped-least-squares 6-DoF pose IK oracle for scripted demos and data collection."""

from __future__ import annotations

import mujoco
import numpy as np

from yam_peg_insert.env import YamPegInsertEnv

# ``grasp_site`` is the menagerie virtual TCP; the actual jaw pinch point sits
# a few centimetres away from it. The offset is measured from the model at home.
_TIP_SPHERE_RADIUS = 1e-3

_HOVER_HEIGHT = 0.10
_GRASP_HEIGHT = 0.030  # grip the upper shaft: 15 mm below the peg top
_LIFT_HEIGHT = 0.12
_TRANSFER_HEIGHT = 0.165  # pinch height above socket_bottom while crossing over it
_INSERT_HEIGHT = 0.077  # pinch height above socket_bottom at full insertion
_INSERT_TIP_DEPTH = 0.002  # desired peg-tip height above socket_bottom

_POS_TOL = 0.008
_ROT_TOL = 0.15
_GRASP_SYNC_TOL = 0.012
_GRASP_SYNC_STEPS = 3
_GRIP_CLOSE_STEPS = 25
_MAX_PHASE_STEPS = 80


def _skew(vec: np.ndarray) -> np.ndarray:
    return np.array(
        [
            [0.0, -vec[2], vec[1]],
            [vec[2], 0.0, -vec[0]],
            [-vec[1], vec[0], 0.0],
        ]
    )


def _rotation_error(rot_cur: np.ndarray, rot_des: np.ndarray) -> np.ndarray:
    """Axis-angle vector (world frame) rotating ``rot_cur`` onto ``rot_des``."""
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, (rot_des @ rot_cur.T).reshape(-1))
    if quat[0] < 0.0:
        quat = -quat
    w = float(np.clip(quat[0], -1.0, 1.0))
    sin_half = float(np.sqrt(max(0.0, 1.0 - w * w)))
    if sin_half < 1e-6:
        return np.zeros(3)
    return (2.0 * np.arctan2(sin_half, w) / sin_half) * quat[1:]


class ScriptedPegOracle:
    """State-machine + 6-DoF pose IK policy for scripted peg-in-hole demos.

    Phases: hover -> descend -> close -> lift -> transfer -> insert.
    """

    PHASE_NAMES = ("hover", "descend", "close", "lift", "transfer", "insert")

    def __init__(
        self,
        env: YamPegInsertEnv,
        damping: float = 0.05,
        gain: float = 0.8,
        slow_gain: float = 0.25,
    ) -> None:
        self.env = env
        self.damping = damping
        self.gain = gain
        self.slow_gain = slow_gain
        self._grasp_site_id = env._grasp_site_id
        self._peg_body_id = env._peg_body_id
        self._peg_tip_id = env._peg_tip_id
        self._socket_site_id = env._socket_bottom_id
        self._desired_rot, self._pinch_offset = self._measure_hand_frame()
        self._scratch = mujoco.MjData(env.model)

        home_data = mujoco.MjData(env.model)
        mujoco.mj_resetDataKeyframe(env.model, home_data, env._home_key_id)
        self._q_home = home_data.qpos[: max(1, env.nu - 1)].copy()
        self.last_status: str = "ok"

        self._phase = 0
        self._phase_steps = 0
        self._settle_steps = 0
        self._peg_ref = np.zeros(3)
        self._hold_target = np.zeros(3)
        self._insert_cmd = np.zeros(3)

    def _measure_hand_frame(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the home grasp orientation and the pinch offset in the site frame."""
        model = self.env.model
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, self.env._home_key_id)
        mujoco.mj_forward(model, data)

        rot = data.site_xmat[self._grasp_site_id].reshape(3, 3).copy()
        grasp_pos = data.site_xpos[self._grasp_site_id]
        tips = [
            data.geom_xpos[geom_id]
            for geom_id in range(model.ngeom)
            if model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_SPHERE
            and model.geom_size[geom_id][0] <= _TIP_SPHERE_RADIUS
        ]
        pinch = np.mean(tips, axis=0)
        return rot, rot.T @ (pinch - grasp_pos)

    def _pose(self, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        """Pinch position and grasp-site rotation stored in ``data``."""
        rot = data.site_xmat[self._grasp_site_id].reshape(3, 3)
        pinch = data.site_xpos[self._grasp_site_id] + rot @ self._pinch_offset
        return pinch, rot

    def _start_episode(self) -> None:
        self._phase = 0
        self._phase_steps = 0
        self._settle_steps = 0
        self._peg_ref = self.env.data.xpos[self._peg_body_id].copy()

    def _target_for_phase(self) -> tuple[np.ndarray, float]:
        socket = self.env.data.site_xpos[self._socket_site_id]
        peg = self._peg_ref

        if self._phase == 0:
            return np.array([peg[0], peg[1], peg[2] + _HOVER_HEIGHT]), 1.0
        if self._phase == 1:
            return np.array([peg[0], peg[1], peg[2] + _GRASP_HEIGHT]), 1.0
        if self._phase == 2:
            return self._hold_target, -1.0
        if self._phase == 3:
            return np.array([peg[0], peg[1], peg[2] + _LIFT_HEIGHT]), -1.0
        if self._phase == 4:
            return np.array([socket[0], socket[1], socket[2] + _TRANSFER_HEIGHT]), -1.0

        target = np.array([socket[0], socket[1], socket[2] + _INSERT_HEIGHT])
        # Close the loop on the measured peg tip: the jaws let the peg sag a few
        # millimetres under load, so an open-loop depth lands inconsistently.
        tip = self.env.data.site_xpos[self._peg_tip_id]
        target[2] += float(
            np.clip(0.5 * (socket[2] + _INSERT_TIP_DEPTH - tip[2]), -0.003, 0.003)
        )
        target[:2] += np.clip(0.5 * (socket[:2] - tip[:2]), -0.003, 0.003)
        # Rate-limit the descent: the joint-space action clip is ~7 mm/step at
        # the wrist, which slams the peg into the table and makes it slip.
        delta = target - self._insert_cmd
        max_step = 0.0025
        norm = float(np.linalg.norm(delta))
        if norm > max_step:
            delta *= max_step / norm
        self._insert_cmd += delta
        return self._insert_cmd, -1.0

    def _advance_phase(self) -> None:
        self._phase = min(self._phase + 1, len(self.PHASE_NAMES) - 1)
        self._phase_steps = 0
        self._settle_steps = 0
        if self._phase == 2:
            peg = self._peg_ref
            self._hold_target = np.array([peg[0], peg[1], peg[2] + _GRASP_HEIGHT])
        if self._phase == 5:
            socket = self.env.data.site_xpos[self._socket_site_id]
            pinch, _ = self._pose(self.env.data)
            self._insert_cmd = np.array([socket[0], socket[1], pinch[2]])

    def predict(self, step_idx: int) -> np.ndarray:
        if step_idx == 0:
            self._start_episode()

        model = self.env.model
        data = self.env.data
        nu = self.env.nu
        arm_dof = max(1, nu - 1)

        target, gripper_cmd = self._target_for_phase()

        # Plan on the commanded joint targets rather than the measured joints:
        # the position servos are under-damped, and closing the loop on the
        # measured state produces a limit cycle at these control rates.
        scratch = self._scratch
        scratch.qpos[:] = data.qpos
        scratch.qpos[:arm_dof] = data.ctrl[:arm_dof]
        mujoco.mj_forward(model, scratch)
        pinch_cmd, rot_cmd = self._pose(scratch)

        err_pos = target - pinch_cmd
        err_rot = _rotation_error(rot_cmd, self._desired_rot)
        pos_ok = float(np.linalg.norm(err_pos)) < _POS_TOL
        rot_ok = float(np.linalg.norm(err_rot)) < _ROT_TOL

        pinch_act, _ = self._pose(data)
        actual_err = float(np.linalg.norm(target - pinch_act))

        self._phase_steps += 1
        if self._phase == 2:
            if self._phase_steps >= _GRIP_CLOSE_STEPS:
                self._advance_phase()
        elif self._phase == 1:
            # Wait until the physical arm has caught up before closing the jaws.
            if pos_ok and rot_ok and actual_err < _GRASP_SYNC_TOL:
                self._settle_steps += 1
                if self._settle_steps >= _GRASP_SYNC_STEPS:
                    self._advance_phase()
            else:
                self._settle_steps = 0
        elif self._phase < 5 and pos_ok and rot_ok:
            self._advance_phase()

        if self._phase_steps > _MAX_PHASE_STEPS and self._phase < 5:
            phase_name = self.PHASE_NAMES[self._phase]
            self.last_status = (
                f"timeout in phase '{phase_name}' after {self._phase_steps} steps "
                f"(pos_err={np.linalg.norm(err_pos):.4f} m, "
                f"rot_err={np.linalg.norm(err_rot):.3f} rad)"
            )

        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, scratch, jacp, jacr, self._grasp_site_id)

        # Jacobian of the pinch point (rigid offset from the site origin).
        offset_world = rot_cmd @ self._pinch_offset
        jacp_pinch = jacp - _skew(offset_world) @ jacr

        jac = np.vstack([jacp_pinch[:, :arm_dof], jacr[:, :arm_dof]])
        err = np.concatenate([err_pos, 0.5 * err_rot])
        damping = (self.damping**2) * np.eye(6)
        jac_pinv = jac.T @ np.linalg.inv(jac @ jac.T + damping)
        dq = jac_pinv @ err
        # Pull redundant DoF back toward the home arm posture so the wrist does
        # not settle against a joint limit (joint4 at +90 deg) mid-insertion.
        null_proj = np.eye(arm_dof) - jac_pinv @ jac
        dq_null = 0.05 * (self._q_home[:arm_dof] - scratch.qpos[:arm_dof])
        dq = dq + null_proj @ dq_null

        action = np.zeros(nu, dtype=np.float32)
        gain = self.gain if self._phase < 3 else self.slow_gain
        action[:arm_dof] = np.clip(
            gain * dq / max(self.env.action_scale, 1e-6), -1.0, 1.0
        )
        if nu > arm_dof:
            action[arm_dof:] = gripper_cmd
        return action
