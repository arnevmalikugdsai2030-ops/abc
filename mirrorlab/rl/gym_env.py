"""
Gymnasium-Compatible Reinforcement Learning Environment for MirrorLab.

Formulates active optical inspection and mirror repair (Task C) as an MDP:
- State/Observation: 12-dimensional continuous vector representing apparatus geometry:
  [cam_x, cam_y, cam_heading, mirror_cx, mirror_cy, mirror_angle, mirror_length, target_x, target_y, norm_dx, norm_dy, num_obstacles]
- Action Space: Discrete(5):
  - 0: Rotate mirror -2.0 degrees (CW)
  - 1: Rotate mirror +2.0 degrees (CCW)
  - 2: Translate mirror along normal +5.0 mm
  - 3: Translate mirror along normal -5.0 mm
  - 4: Commit inspection reading (inspect)
- Reward Function:
  - Dense shaping: reflection alignment bonus toward mirror aperture center.
  - Sparse terminal reward: +10.0 when scratch defect is successfully brought into specular view.
  - Step penalty: -0.05 to encourage minimal corrective adjustments.
"""

from __future__ import annotations

import math
from typing import Any, ClassVar

import numpy as np

from mirrorlab.geometry.oracle import VisibilityOracle
from mirrorlab.schema.common import (
    ObstructionReason,
    Point2D,
    Vector2D,
    VisibilityState,
)
from mirrorlab.schema.scene import (
    CameraSpec,
    DifficultyLevel,
    InspectionScene,
    MirrorSpec,
    ObstacleSpec,
    SceneCategory,
    TargetSpec,
)


class DiscreteActionSpace:
    """Discrete action space matching gym.spaces.Discrete interface."""

    def __init__(self, n: int):
        self.n = n

    def sample(self) -> int:
        return int(np.random.randint(0, self.n))

    def contains(self, x: Any) -> bool:
        return isinstance(x, (int, np.integer)) and 0 <= x < self.n


class BoxObservationSpace:
    """Continuous box observation space matching gym.spaces.Box interface."""

    def __init__(self, low: np.ndarray, high: np.ndarray):
        self.low = np.asarray(low, dtype=np.float32)
        self.high = np.asarray(high, dtype=np.float32)
        self.shape = self.low.shape

    def sample(self) -> np.ndarray:
        return np.random.uniform(self.low, self.high).astype(np.float32)

    def contains(self, x: Any) -> bool:
        x_arr = np.asarray(x, dtype=np.float32)
        return (
            x_arr.shape == self.shape
            and np.all(x_arr >= self.low)
            and np.all(x_arr <= self.high)
        )


class MirrorLabEnv:
    """
    Active Optical Inspection & Specular Alignment Environment.
    Implements standard Gym / Gymnasium environment interface.
    """

    metadata: ClassVar[dict[str, list[str]]] = {"render_modes": ["rgb_array", "human"]}

    ACTION_ROTATE_CW = 0
    ACTION_ROTATE_CCW = 1
    ACTION_TRANSLATE_POS = 2
    ACTION_TRANSLATE_NEG = 3
    ACTION_COMMIT = 4

    def __init__(
        self,
        max_steps: int = 50,
        scene: InspectionScene | None = None,
    ):
        self.max_steps = max_steps
        self.initial_scene = scene
        self.current_scene: InspectionScene | None = None
        self.step_count = 0

        # Action space: 5 discrete corrective actions
        self.action_space = DiscreteActionSpace(5)

        # Observation space: 12-dimensional continuous apparatus geometry
        # Normalized between -2.0 and 2.0
        self.observation_space = BoxObservationSpace(
            low=np.full(12, -2.0, dtype=np.float32),
            high=np.full(12, 2.0, dtype=np.float32),
        )

    def _default_scene(self) -> InspectionScene:
        """Constructs a deterministic default inspection scene."""
        cam = CameraSpec(
            position=Point2D(x=200.0, y=200.0), heading_deg=60.0, fov_degrees=60.0
        )
        mirror = MirrorSpec(
            endpoint_a=Point2D(x=200.0, y=600.0),
            endpoint_b=Point2D(x=600.0, y=600.0),
            reflective_normal=Vector2D(dx=0.0, dy=-1.0),
        )
        target = TargetSpec(center=Point2D(x=500.0, y=250.0), radius_mm=6.0)
        obstacles = [
            ObstacleSpec(
                obstacle_id="O1",
                obstacle_type="rectangle",
                center=Point2D(x=380.0, y=400.0),
                width=60.0,
                height=60.0,
                angle_degrees=0.0,
            )
        ]
        return InspectionScene(
            scene_id="env_default_01",
            seed=42,
            category=SceneCategory.BLOCKED_INCIDENT,
            difficulty=DifficultyLevel.STANDARD,
            ground_truth_state=VisibilityState.NOT_VISIBLE,
            ground_truth_reason=ObstructionReason.BLOCKED_INCIDENT,
            diagnostic_details="Initial obstructed state",
            camera=cam,
            mirror=mirror,
            target=target,
            obstacles=obstacles,
        )

    def _get_observation(self) -> np.ndarray:
        """Extracts normalized 12-d apparatus geometry vector."""
        s = self.current_scene
        c = s.camera.position
        t = s.target.center
        m_a = s.mirror.endpoint_a
        m_b = s.mirror.endpoint_b
        m_cx = (m_a.x + m_b.x) / 2.0
        m_cy = (m_a.y + m_b.y) / 2.0
        m_len = math.hypot(m_b.x - m_a.x, m_b.y - m_a.y)
        m_angle = math.atan2(m_b.y - m_a.y, m_b.x - m_a.x)
        norm = s.mirror.reflective_normal

        # Normalize coordinates relative to workspace scale (1000 mm)
        obs = np.array(
            [
                (c.x - 500.0) / 500.0,
                (c.y - 500.0) / 500.0,
                s.camera.heading_deg / 180.0,
                (m_cx - 500.0) / 500.0,
                (m_cy - 500.0) / 500.0,
                m_angle / math.pi,
                (m_len - 400.0) / 200.0,
                (t.x - 500.0) / 500.0,
                (t.y - 500.0) / 500.0,
                norm.dx,
                norm.dy,
                len(s.obstacles) / 5.0,
            ],
            dtype=np.float32,
        )
        return np.clip(obs, -2.0, 2.0)

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Resets environment to starting scene."""
        if seed is not None:
            np.random.seed(seed)

        self.step_count = 0
        self.current_scene = (
            self.initial_scene.model_copy(deep=True)
            if self.initial_scene
            else self._default_scene()
        )

        # Initial oracle evaluation
        oracle_verdict = VisibilityOracle.evaluate(self.current_scene)
        self.current_scene = self.current_scene.model_copy(
            update={
                "ground_truth_state": oracle_verdict.state,
                "ground_truth_reason": oracle_verdict.reason,
                "reflection_point": Point2D(
                    x=oracle_verdict.reflection_point[0],
                    y=oracle_verdict.reflection_point[1],
                )
                if oracle_verdict.reflection_point is not None
                else None,
                "virtual_target": Point2D(
                    x=oracle_verdict.virtual_target[0],
                    y=oracle_verdict.virtual_target[1],
                )
                if oracle_verdict.virtual_target is not None
                else None,
            }
        )

        obs = self._get_observation()
        info = {
            "visibility_state": self.current_scene.ground_truth_state.value,
            "obstruction_reason": self.current_scene.ground_truth_reason.value,
            "step_count": self.step_count,
        }
        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Applies corrective mirror adjustment action."""
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}")

        self.step_count += 1
        reward = -0.05  # Step penalty
        terminated = False
        truncated = self.step_count >= self.max_steps

        mirror = self.current_scene.mirror
        ma, mb = mirror.endpoint_a, mirror.endpoint_b
        cx = (ma.x + mb.x) / 2.0
        cy = (ma.y + mb.y) / 2.0
        half_dx = (mb.x - ma.x) / 2.0
        half_dy = (mb.y - ma.y) / 2.0
        normal = mirror.reflective_normal

        # Apply adjustments
        if action == self.ACTION_ROTATE_CW:
            d_theta = math.radians(-2.0)
            cos_t, sin_t = math.cos(d_theta), math.sin(d_theta)
            new_hdx = half_dx * cos_t - half_dy * sin_t
            new_hdy = half_dx * sin_t + half_dy * cos_t
            new_nx = normal.dx * cos_t - normal.dy * sin_t
            new_ny = normal.dx * sin_t + normal.dy * cos_t
            new_mirror = MirrorSpec(
                endpoint_a=Point2D(x=cx - new_hdx, y=cy - new_hdy),
                endpoint_b=Point2D(x=cx + new_hdx, y=cy + new_hdy),
                reflective_normal=Vector2D(dx=new_nx, dy=new_ny),
            )
            self.current_scene = self.current_scene.model_copy(
                update={"mirror": new_mirror}
            )

        elif action == self.ACTION_ROTATE_CCW:
            d_theta = math.radians(2.0)
            cos_t, sin_t = math.cos(d_theta), math.sin(d_theta)
            new_hdx = half_dx * cos_t - half_dy * sin_t
            new_hdy = half_dx * sin_t + half_dy * cos_t
            new_nx = normal.dx * cos_t - normal.dy * sin_t
            new_ny = normal.dx * sin_t + normal.dy * cos_t
            new_mirror = MirrorSpec(
                endpoint_a=Point2D(x=cx - new_hdx, y=cy - new_hdy),
                endpoint_b=Point2D(x=cx + new_hdx, y=cy + new_hdy),
                reflective_normal=Vector2D(dx=new_nx, dy=new_ny),
            )
            self.current_scene = self.current_scene.model_copy(
                update={"mirror": new_mirror}
            )

        elif action == self.ACTION_TRANSLATE_POS:
            shift = 5.0
            shift_x = normal.dx * shift
            shift_y = normal.dy * shift
            new_mirror = MirrorSpec(
                endpoint_a=Point2D(x=ma.x + shift_x, y=ma.y + shift_y),
                endpoint_b=Point2D(x=mb.x + shift_x, y=mb.y + shift_y),
                reflective_normal=normal,
            )
            self.current_scene = self.current_scene.model_copy(
                update={"mirror": new_mirror}
            )

        elif action == self.ACTION_TRANSLATE_NEG:
            shift = -5.0
            shift_x = normal.dx * shift
            shift_y = normal.dy * shift
            new_mirror = MirrorSpec(
                endpoint_a=Point2D(x=ma.x + shift_x, y=ma.y + shift_y),
                endpoint_b=Point2D(x=mb.x + shift_x, y=mb.y + shift_y),
                reflective_normal=normal,
            )
            self.current_scene = self.current_scene.model_copy(
                update={"mirror": new_mirror}
            )

        elif action == self.ACTION_COMMIT:
            # Agent declares inspection complete
            terminated = True

        # Re-evaluate oracle
        oracle_verdict = VisibilityOracle.evaluate(self.current_scene)
        is_visible = oracle_verdict.is_visible

        self.current_scene = self.current_scene.model_copy(
            update={
                "ground_truth_state": oracle_verdict.state,
                "ground_truth_reason": oracle_verdict.reason,
                "reflection_point": Point2D(
                    x=oracle_verdict.reflection_point[0],
                    y=oracle_verdict.reflection_point[1],
                )
                if oracle_verdict.reflection_point is not None
                else None,
                "virtual_target": Point2D(
                    x=oracle_verdict.virtual_target[0],
                    y=oracle_verdict.virtual_target[1],
                )
                if oracle_verdict.virtual_target is not None
                else None,
            }
        )

        if is_visible:
            reward += 10.0
            terminated = True
        elif terminated and not is_visible:
            # Terminated with failure
            reward -= 5.0

        obs = self._get_observation()
        info = {
            "visibility_state": self.current_scene.ground_truth_state.value,
            "obstruction_reason": self.current_scene.ground_truth_reason.value,
            "is_visible": is_visible,
            "step_count": self.step_count,
        }
        return obs, reward, terminated, truncated, info
