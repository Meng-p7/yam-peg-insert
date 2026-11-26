import numpy as np

from yam_peg_insert import ScriptedPegOracle, YamPegInsertEnv


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


def test_oracle_predict_shape() -> None:
    env = YamPegInsertEnv(max_steps=10)
    env.reset(seed=1)
    oracle = ScriptedPegOracle(env)
    action = oracle.predict(0)
    assert action.shape == env.action_space.shape
    assert np.all(np.isfinite(action))
    assert np.all(action >= -1.0) and np.all(action <= 1.0)
    env.close()
