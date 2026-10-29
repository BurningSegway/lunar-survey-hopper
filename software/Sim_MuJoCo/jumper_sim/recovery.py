"""Attempt to right a tipped-over robot (Space while tipped).

The robot lies on a side face. Neither the reaction wheels (1.62 N*m against
the 1.46 N*m needed just to start tipping over the edge, and saturated within
~0.3 s) nor the legs (they slide along body z, which is horizontal on the
side, so they only push along the ground) can right it on the ground (see
README, "Tipped over"). The attempt gets it into the air instead and turns it
upright there:

1. SPIN_UP: the wheels torque the body *away* from upright, into the
   ground, which costs nothing but spins them up. In the air they can then
   swing the body towards upright through their whole speed range, from
   this speed down to the limit the other way, instead of from rest;
2. the jump spring is released. The foot points sideways, but it catches
   the ground often enough to throw the robot up;
3. FLIGHT: while airborne the attitude controller is pointed at level
   (current heading) instead of the hop reference, until the robot is on the
   ground again (GROUND phase).

Measured in Moon gravity on 16 terrain spots x 4 sides (54-55 tipped poses
per run): spin-up 0.15 s -> upright after one attempt 40/54 and 42/55; 0.3 s
34-35, 0.5 s 32-36, no spin-up 25/54. Rocking the body up with the wheels
before releasing the spring (it rises ~10 deg) did worse, 6-8/54: it uses up
the wheel speed the flight needs. Extending the legs, passively or with the
full 1000 N motor push, made every variant worse (0/27), so the legs stay
pulled in.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
import math

import numpy as np


class RecoveryStage(Enum):
    SPIN_UP = auto()
    FLIGHT = auto()


@dataclass(frozen=True)
class RecoveryConfig:
    spin_up_duration: float = 0.15  # s


def righting_axis(rotation: np.ndarray) -> np.ndarray | None:
    """Body-frame unit axis that turns body z towards world z (None if upright or upside down)."""
    axis = np.cross(rotation[:, 2], (0.0, 0.0, 1.0))
    norm = float(np.linalg.norm(axis))
    if norm < 1e-6:
        return None
    return rotation.T @ (axis / norm)


def level_quaternion(rotation: np.ndarray) -> tuple[float, float, float, float]:
    """Level attitude with the body's current heading."""
    yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    return (math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw))


class RecoveryAttempt:
    """Stage timing of one attempt; the simulation applies the actuators."""

    def __init__(self, start_time: float, config: RecoveryConfig = RecoveryConfig()) -> None:
        self.config = config
        self.stage = RecoveryStage.SPIN_UP
        self._start = start_time

    def update(self, time: float) -> bool:
        """Advance the stages; returns True when the spring should be released now."""
        if self.stage is RecoveryStage.SPIN_UP and time - self._start >= self.config.spin_up_duration:
            self.stage = RecoveryStage.FLIGHT
            return True
        return False

    @property
    def wheel_torque_sign(self) -> float:
        """+1: wheel torque along the righting axis (body pushed away from upright); 0: controller."""
        return 1.0 if self.stage is RecoveryStage.SPIN_UP else 0.0
