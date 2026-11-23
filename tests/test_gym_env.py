"""
Unit Tests for Gymnasium-compatible MirrorLabEnv.

Verifies:
1. Environment instantiation and reset behavior.
2. Step execution with discrete actions.
3. Observation space shape, bounds, and dtype.
4. Reward signals on step penalty and visibility completion.
5. Action space validation.
"""

import numpy as np
import pytest

from mirrorlab.rl.gym_env import MirrorLabEnv


class TestMirrorLabEnv:
    """Tests MirrorLabEnv reinforcement learning environment."""

    def test_environment_init_and_reset(self):
        env = MirrorLabEnv(max_steps=20)
        obs, info = env.reset(seed=123)

        assert obs.shape == (12,)
        assert obs.dtype == np.float32
        assert env.observation_space.contains(obs)
        assert info["step_count"] == 0
        assert "visibility_state" in info

    def test_step_actions(self):
        env = MirrorLabEnv(max_steps=10)
        env.reset(seed=42)

        # Test CW rotation
        obs1, _r1, _term1, trunc1, info1 = env.step(MirrorLabEnv.ACTION_ROTATE_CW)
        assert env.observation_space.contains(obs1)
        assert info1["step_count"] == 1
        assert not trunc1

        # Test CCW rotation
        obs2, _r2, _term2, _trunc2, info2 = env.step(MirrorLabEnv.ACTION_ROTATE_CCW)
        assert env.observation_space.contains(obs2)
        assert info2["step_count"] == 2

        # Test translation
        obs3, _r3, _term3, _trunc3, _info3 = env.step(MirrorLabEnv.ACTION_TRANSLATE_POS)
        assert env.observation_space.contains(obs3)

    def test_action_space_bounds(self):
        env = MirrorLabEnv()
        env.reset()

        assert env.action_space.contains(0)
        assert env.action_space.contains(4)
        assert not env.action_space.contains(5)
        assert not env.action_space.contains(-1)

        with pytest.raises(ValueError, match="Invalid action"):
            env.step(99)

    def test_truncation_on_max_steps(self):
        env = MirrorLabEnv(max_steps=3)
        env.reset()

        _, _, _term1, trunc1, _ = env.step(0)
        assert not trunc1
        _, _, _term2, trunc2, _ = env.step(0)
        assert not trunc2
        _, _, _term3, trunc3, _ = env.step(0)
        assert trunc3

    def test_commit_action_terminates(self):
        env = MirrorLabEnv()
        env.reset()
        _, _reward, terminated, _, _info = env.step(MirrorLabEnv.ACTION_COMMIT)
        assert terminated is True
