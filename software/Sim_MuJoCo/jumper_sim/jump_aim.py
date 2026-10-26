"""Lean towards the track target before jumping, closed loop on the IMU attitude.

The jump spring pushes along the body axis, so the hop goes where the body
axis leans. On the ground, with a track target set, the robot leans its body
axis ``lean_angle(distance)`` from vertical towards the target: at least
``min_lean`` (so a hop always moves towards, and near it over, the target)
growing to ``max_lean`` at ``full_distance`` and beyond.

The legs are shortened to the plane ``g`` (a tilt gradient in the body x-y
plane): leg i is shortened by ``g . p_i - min_j g . p_j`` (``p`` the feet), so
the side towards ``g`` drops and the body leans that way -- the leg closest to
that direction most, the furthest one not at all. ``g`` is integrated from the
error between the desired and the IMU-measured tilt, so slopes are cancelled
automatically. It only integrates while the robot is still, is capped at
``max_shortening`` per leg (scaled as a whole, so the feet stay on a plane),
and lets go entirely on ground steeper than ``max_ground_tilt``.

Pure numpy; meant to run unchanged on the robot.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class JumpAimConfig:
    min_lean: float = math.radians(3.0)  # at the target
    # Launches tumbled at 10-14 deg (the foot slams into the ground on
    # release); keep this low until that is fixed.
    max_lean: float = math.radians(8.0)
    full_distance: float = 30.0  # m, max_lean from this distance on
    no_direction_distance: float = 0.5  # m, closer than this: stand upright
    max_ground_tilt: float = math.radians(20.0)  # no lean on steeper ground
    # Leg stroke is 0.12 m, but beyond ~0.11 m the body box reaches the
    # ground before the shortened leg's foot does.
    max_shortening: float = 0.10  # m
    response_time: float = 0.5  # s, integrator time constant
    still_rate: float = 0.3  # rad/s, only adjust below this body rate


def lean_angle(distance: float, config: JumpAimConfig = JumpAimConfig()) -> float:
    """Desired lean [rad] for a target ``distance`` [m] away horizontally."""
    if distance < config.no_direction_distance:
        return 0.0
    fraction = min(1.0, distance / config.full_distance)
    return config.min_lean + (config.max_lean - config.min_lean) * fraction


def measured_tilt(body_rotation: np.ndarray) -> np.ndarray:
    """World tilt vector of the body axis: (z_x, z_y) / z_z (tan of the lean)."""
    z = np.asarray(body_rotation, float)[:, 2]
    return z[:2] / max(z[2], 1e-6)


class JumpAim:
    def __init__(self, feet: np.ndarray, config: JumpAimConfig = JumpAimConfig()) -> None:
        self.config = config
        self._feet = np.asarray(feet, float)[:, :2]
        self.reset()

    def reset(self) -> None:
        """Straight legs (call at touchdown)."""
        self.gradient = np.zeros(2)  # body frame
        self.shortening = np.zeros(len(self._feet))
        self.desired = 0.0  # rad
        self.limited = False  # a leg is at max_shortening

    def update(
        self,
        body_rotation: np.ndarray,
        body_rate: np.ndarray,
        body_position: np.ndarray,
        target: np.ndarray,
        dt: float,
        hold: bool = False,
    ) -> np.ndarray:
        """Leg shortening [m]; ``hold`` keeps the legs as they are (take-off)."""
        if hold:
            return self.shortening
        rotation = np.asarray(body_rotation, float)
        to_target = (np.asarray(target, float) - np.asarray(body_position, float))[:2]
        distance = float(np.linalg.norm(to_target))
        self.desired = lean_angle(distance, self.config)
        wanted = (
            math.tan(self.desired) * to_target / distance if self.desired > 0.0 else np.zeros(2)
        )
        tilt = measured_tilt(rotation)
        # Ground tilt ~ measured tilt minus the lean the legs produce.
        produced = (rotation[:2, :2] @ self.gradient)
        if math.atan(float(np.linalg.norm(tilt - produced))) > self.config.max_ground_tilt:
            self.reset()
            return self.shortening
        if float(np.linalg.norm(body_rate)) < self.config.still_rate:
            error_body = rotation[:2, :2].T @ (wanted - tilt)
            self.gradient = self.gradient + error_body * dt / self.config.response_time
        reach = self._feet @ self.gradient
        shortening = reach - reach.min()
        longest = float(shortening.max())
        self.limited = longest >= self.config.max_shortening
        if longest > self.config.max_shortening:  # anti-windup: scale the plane
            scale = self.config.max_shortening / longest
            self.gradient *= scale
            shortening *= scale
        self.shortening = shortening
        return self.shortening

    def lean_towards(self, body_rotation: np.ndarray, body_position, target) -> float:
        """Measured lean [rad] of the body axis towards the target (signed)."""
        to_target = (np.asarray(target, float) - np.asarray(body_position, float))[:2]
        distance = float(np.linalg.norm(to_target))
        if distance < 1e-9:
            return 0.0
        return math.atan(float(measured_tilt(body_rotation) @ (to_target / distance)))
