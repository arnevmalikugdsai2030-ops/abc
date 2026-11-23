"""
Quantitative Difficulty Calibration Engine for MIRRORLAB.

Calculates continuous and discrete task difficulty (EASY, MEDIUM, HARD, EXPERT)
based on measurable physical and optical geometric properties:
1. Angular clearance to mirror boundary (delta_theta_margin)
2. Obstacle penetration / clearance distance margin (delta_d_clearance)
3. Incidence angle extremity (grazing incidence penalty)
4. Optical ray path segments / reflection hops
5. Distractor and obstacle clutter density
6. Counterfactual perturbation magnitude (delta_theta, delta_x)
"""

from __future__ import annotations

import math
from enum import Enum

from pydantic import BaseModel, Field

from mirrorlab.schema.scene import InspectionScene


class DifficultyTier(str, Enum):
    """Calibrated difficulty classifications."""

    EASY = "EASY"
    MEDIUM = "MEDIUM"
    HARD = "HARD"
    EXPERT = "EXPERT"


class DifficultyMetrics(BaseModel):
    """Quantitative geometric metrics driving task difficulty."""

    aperture_margin_mm: float = Field(
        ...,
        description="Distance from reflection point to nearest mirror endpoint (mm)",
    )
    obstacle_clearance_mm: float = Field(
        ...,
        description="Minimum clearance distance between optical corridor and obstacles (mm)",
    )
    incidence_angle_deg: float = Field(
        ...,
        description="Reflection angle of incidence in degrees (0 = normal, 90 = grazing)",
    )
    num_obstacles: int = Field(
        ..., description="Number of obstacles and distractors in scene"
    )
    num_optical_legs: int = Field(
        default=2,
        description="Number of optical ray legs (2 for single mirror, 3+ for multi-mirror)",
    )
    is_counterfactual_pair: bool = Field(
        default=False,
        description="Whether task is part of a sensitive counterfactual pair",
    )
    perturbation_magnitude: float | None = Field(
        default=None,
        description="Perturbation delta (degrees or mm) for counterfactual tasks",
    )
    composite_difficulty_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Continuous difficulty index from 0 (trivial) to 100 (extreme)",
    )
    calibrated_tier: DifficultyTier = Field(
        ..., description="Discrete difficulty tier (EASY, MEDIUM, HARD, EXPERT)"
    )


class DifficultyCalibrator:
    """Computes transparent, reproducible difficulty metrics from scene geometry."""

    @staticmethod
    def calculate_metrics(
        scene: InspectionScene,
        reflection_point: tuple[float, float] | None = None,
        min_obstacle_distance: float | None = None,
        incidence_angle: float | None = None,
        is_counterfactual: bool = False,
        perturbation_delta: float | None = None,
    ) -> DifficultyMetrics:
        """
        Evaluate physical constraints to assign a quantitative difficulty score.
        """
        # 1. Aperture margin: distance from R to endpoints A and B
        aperture_margin = 50.0  # default wide margin if no reflection
        if reflection_point is not None:
            rx, ry = reflection_point
            ma = scene.mirror.endpoint_a
            mb = scene.mirror.endpoint_b
            dist_a = math.hypot(rx - ma.x, ry - ma.y)
            dist_b = math.hypot(rx - mb.x, ry - mb.y)
            aperture_margin = min(dist_a, dist_b)

        # 2. Obstacle clearance margin
        clearance = min_obstacle_distance if min_obstacle_distance is not None else 35.0

        # 3. Incidence angle
        theta = incidence_angle if incidence_angle is not None else 45.0

        num_obs = len(scene.obstacles)
        num_legs = 2  # standard single mirror

        # Composite score calculation (0 to 100)
        # Tight aperture adds up to 30 pts
        aperture_score = max(0.0, min(30.0, (50.0 - aperture_margin) * 0.6))

        # Tight obstacle clearance adds up to 30 pts
        clearance_score = max(0.0, min(30.0, (40.0 - clearance) * 0.75))

        # Extreme grazing angle (> 65 deg) adds up to 20 pts
        grazing_score = (
            max(0.0, min(20.0, (theta - 45.0) * 0.8)) if theta > 45.0 else 0.0
        )

        # Clutter density adds up to 10 pts
        clutter_score = min(10.0, num_obs * 2.5)

        # Counterfactual sensitivity adds up to 15 pts if perturbation is small (<= 2 deg or <= 5mm)
        cf_score = 0.0
        if is_counterfactual:
            if perturbation_delta is not None and perturbation_delta <= 2.5:
                cf_score = 15.0
            else:
                cf_score = 8.0

        total_score = min(
            100.0,
            aperture_score + clearance_score + grazing_score + clutter_score + cf_score,
        )

        # Determine Tier
        if total_score < 25.0:
            tier = DifficultyTier.EASY
        elif total_score < 50.0:
            tier = DifficultyTier.MEDIUM
        elif total_score < 75.0:
            tier = DifficultyTier.HARD
        else:
            tier = DifficultyTier.EXPERT

        return DifficultyMetrics(
            aperture_margin_mm=round(aperture_margin, 2),
            obstacle_clearance_mm=round(clearance, 2),
            incidence_angle_deg=round(theta, 2),
            num_obstacles=num_obs,
            num_optical_legs=num_legs,
            is_counterfactual_pair=is_counterfactual,
            perturbation_magnitude=perturbation_delta,
            composite_difficulty_score=round(total_score, 1),
            calibrated_tier=tier,
        )
