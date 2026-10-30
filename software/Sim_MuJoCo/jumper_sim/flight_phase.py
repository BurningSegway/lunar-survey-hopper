"""Hop phases: GROUND / TAKEOFF / ASCENT / DESCENT / LANDING.

Every behaviour (target tracking, scanning, landing-plane fitting, leaning)
switches on these phases, so none of them can act in a phase it is not meant
for. Transitions:

* GROUND -> TAKEOFF: the jump command (also accepted while settling in
  LANDING, as long as the robot is on the ground).
* TAKEOFF -> ASCENT: airborne and ``ascent_height`` above the standing height
  (vertical velocity integrated from the jump command). A hop that peaks lower
  goes straight to DESCENT; a jump that never leaves the ground returns to
  GROUND after ``takeoff_timeout``.
* ASCENT -> DESCENT: the vertical velocity estimate turns negative (the apex).
* DESCENT -> LANDING: touchdown is near (the landing-plane filter's rule:
  plane < 1 m below or < 0.3 s away), or ground contact as a backup.
* LANDING -> GROUND: standing still (accelerometer ~1 g, low body rate) for
  ``settle_time``. Bounces stay in LANDING.
* GROUND -> DESCENT: airborne without a jump (pushed off, rolled over a rim).
The detector starts in DESCENT: at start-up the robot may be in the air.

The vertical velocity comes from the accelerometer. An accelerometer measures
specific force f = a - g_vec (body frame). In free flight it reads ~0 on the
way up *and* on the way down -- world-vertical acceleration is a constant -g
in both -- so the apex cannot be read from the acceleration directly. What
changes sign at the apex is the vertical velocity, so this module integrates
the world-vertical acceleration

    a_z = (R_world_from_body @ f)_z - g

from standstill (through the push-off) into a vertical-velocity estimate.
Whenever the robot is standing still on the ground (accelerometer reading ~1 g
for a short while), the estimate is reset to zero so integration drift cannot
build up between hops.

Ground contact still comes from the simulator; the rest is pure numpy and
meant to run unchanged on the robot from real IMU samples. (In the simulator
the accelerometer is synthesised from the body's velocity change per step;
see JumperSimulation._update_flight_phase.)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
import math

import numpy as np

from .landing_reference import quaternion_to_matrix


class FlightPhase(Enum):
    GROUND = auto()
    TAKEOFF = auto()
    ASCENT = auto()
    DESCENT = auto()
    LANDING = auto()


@dataclass(frozen=True)
class FlightPhaseConfig:
    # Standing still: |f| within this of g [m/s^2] for still_time [s]
    # (zero-velocity update of the vertical velocity estimate).
    still_tolerance: float = 0.5
    still_time: float = 0.2
    # Descent once the vertical velocity estimate drops below this [m/s].
    descent_velocity: float = 0.0
    ascent_height: float = 0.5  # m above the standing height
    takeoff_timeout: float = 0.5  # s on the ground after the jump command
    settle_rate: float = 0.2  # rad/s, body rate below which the robot is still
    settle_time: float = 0.5  # s still before LANDING becomes GROUND


class VerticalVelocityEstimator:
    def __init__(self, config: FlightPhaseConfig = FlightPhaseConfig()) -> None:
        self.config = config
        self.reset()

    def reset(self) -> None:
        self.vertical_velocity = 0.0
        self._still_for = 0.0

    def update(
        self,
        specific_force_body: np.ndarray,
        body_quaternion: tuple[float, float, float, float],
        gravity: float,
        dt: float,
        in_contact: bool,
    ) -> float:
        """Integrate one accelerometer sample; returns the vertical velocity."""
        f_world = quaternion_to_matrix(body_quaternion) @ np.asarray(specific_force_body, float)
        self.vertical_velocity += (float(f_world[2]) - gravity) * dt
        still = in_contact and abs(float(np.linalg.norm(f_world)) - gravity) < self.config.still_tolerance
        self._still_for = self._still_for + dt if still else 0.0
        if self._still_for >= self.config.still_time:
            self.vertical_velocity = 0.0  # zero-velocity update
        return self.vertical_velocity


class FlightPhaseDetector:
    """The hop phase state machine (see the module docstring)."""

    def __init__(self, config: FlightPhaseConfig = FlightPhaseConfig()) -> None:
        self.config = config
        self.estimator = VerticalVelocityEstimator(config)
        self.reset()

    def reset(self) -> None:
        self.estimator.reset()
        self.phase = FlightPhase.DESCENT
        self.height = 0.0  # m above the standing height, during TAKEOFF
        self._takeoff_time = 0.0
        self._settled_for = 0.0

    def update(
        self,
        specific_force_body,
        body_quaternion,
        gravity: float,
        dt: float,
        in_contact: bool,
        body_rate=None,
        jump: bool = False,
        touchdown_near: bool = False,
    ) -> FlightPhase:
        """One IMU sample. ``jump``: the jump command was given this step;
        ``touchdown_near``: the landing-plane filter froze during this descent."""
        vz = self.estimator.update(specific_force_body, body_quaternion, gravity, dt, in_contact)
        force = float(np.linalg.norm(specific_force_body))
        rate = 0.0 if body_rate is None else float(np.linalg.norm(body_rate))
        still = (
            in_contact
            and abs(force - gravity) < self.config.still_tolerance
            and rate < self.config.settle_rate
        )
        self._settled_for = self._settled_for + dt if still else 0.0

        phase = self.phase
        if jump and (phase is FlightPhase.GROUND or (phase is FlightPhase.LANDING and in_contact)):
            self.height = 0.0
            self._takeoff_time = 0.0
            self.phase = FlightPhase.TAKEOFF
            return self.phase
        if phase is FlightPhase.GROUND:
            if not in_contact:
                self.phase = FlightPhase.DESCENT
        elif phase is FlightPhase.TAKEOFF:
            self.height += vz * dt
            self._takeoff_time += dt
            if not in_contact and self.height >= self.config.ascent_height:
                self.phase = FlightPhase.ASCENT
            elif not in_contact and vz < self.config.descent_velocity:
                self.phase = FlightPhase.DESCENT  # low hop
            elif in_contact and self._takeoff_time >= self.config.takeoff_timeout:
                self.phase = FlightPhase.GROUND  # the jump did not happen
        elif phase is FlightPhase.ASCENT:
            if in_contact:
                self.phase = FlightPhase.LANDING
            elif vz < self.config.descent_velocity:
                self.phase = FlightPhase.DESCENT
        elif phase is FlightPhase.DESCENT:
            if touchdown_near or in_contact:
                self.phase = FlightPhase.LANDING
        elif phase is FlightPhase.LANDING:
            if self._settled_for >= self.config.settle_time:
                self.phase = FlightPhase.GROUND
        return self.phase

    @property
    def vertical_velocity(self) -> float:
        return self.estimator.vertical_velocity


def gravity_magnitude(gravity_vector) -> float:
    return math.sqrt(sum(float(v) ** 2 for v in gravity_vector))
