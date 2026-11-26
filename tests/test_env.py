import mujoco
import numpy as np

from yam_peg_insert import ScriptedPegOracle, YamPegInsertEnv

_FINGER_BODIES = (
    "link_left_finger",
    "lf_rot",
    "lf_down",
    "link_right_finger",
    "rf_rot",
    "rf_down",
)


def test_env_reset_and_step() -> None:
    env = YamPegInsertEnv(max_steps=20)
    obs, _ = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    assert np.all(np.isfinite(obs))

    obs2, reward, terminated, truncated, info = env.step(
        np.zeros(env.action_space.shape, dtype=np.float32)
    )
    assert obs2.shape == env.observation_space.shape
    assert isinstance(reward, float)
    assert "tip_to_socket_dist" in info
    assert not terminated
    assert not truncated
    env.close()


def test_reset_uses_home_pose_and_avoids_finger_penetration() -> None:
    env = YamPegInsertEnv(max_steps=10)
    env.reset(seed=0)
    model, data = env.model, env.data

    # Home keyframe is non-zero for joints 2/3; all-zero qpos is the folded pose.
    assert data.qpos[1] > 0.5
    assert data.qpos[2] > 0.5

    finger_body_ids = {
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in _FINGER_BODIES
    }
    for contact in data.contact[: data.ncon]:
        body1 = model.geom_bodyid[contact.geom1]
        body2 = model.geom_bodyid[contact.geom2]
        assert not (body1 in finger_body_ids and body2 in finger_body_ids)
    env.close()


def test_observation_contains_peg_orientation() -> None:
    env = YamPegInsertEnv(max_steps=10)
    obs, _ = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape

    # Observation tail is [peg z-axis (3), contact count (1)].
    peg_z_axis = obs[-4:-1]
    assert np.isclose(np.linalg.norm(peg_z_axis), 1.0, atol=1e-5)
    # The peg starts upright, so its world z-axis is close to (0, 0, 1).
    assert peg_z_axis[2] > 0.99
    env.close()


def test_dropped_peg_terminates_episode() -> None:
    env = YamPegInsertEnv(max_steps=10)
    env.reset(seed=0)

    qadr = env.model.jnt_qposadr[env._peg_joint_id]
    env.data.qpos[qadr + 2] = -0.1
    mujoco.mj_forward(env.model, env.data)

    _, _, terminated, _, info = env.step(
        np.zeros(env.action_space.shape, dtype=np.float32)
    )
    assert terminated
    assert info["dropped"]
    assert not info["is_success"]
    env.close()


def test_oracle_predict_shape() -> None:
    env = YamPegInsertEnv(max_steps=10)
    env.reset(seed=1)
    oracle = ScriptedPegOracle(env)
    action = oracle.predict(0)
    assert action.shape == env.action_space.shape
    assert np.all(np.isfinite(action))
    assert np.all(action >= -1.0) and np.all(action <= 1.0)
    env.close()
