"""Attitude reference over a hop: track a point on the way up, land on the way down.

* TAKEOFF (lift-off until 0.5 m up): the attitude the robot left the
  ground with is held.
* ASCENT (only on flights started by a jump): the downward camera (body -z)
  is pointed at a settable world target point and keeps tracking it as the
  robot moves. The required tilt is capped at ``max_track_tilt``; heading
  stays at the yaw reference.
* DESCENT: the reference moves back towards level (the xy plane). As soon as
  the camera has fitted the ground below, the landing-plane reference from
  :class:`~jumper_sim.landing_reference.LandingReference` takes over -- that
  filter starts at "level" and only receives frames during descent, so
  "level until the ground is seen, then the plane" falls out of it directly.
* The commanded reference never jumps: it slews towards the desired one at
  ``slew_rate`` (also at lift-off, starting from the attitude the robot left
  the ground with), so the LQR is not hit with large reference steps.

Pure numpy; meant to run unchanged on the robot.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .flight_phase import FlightPhase
from .landing_reference import quaternion_to_matrix, tilted_reference_quaternion

Quaternion = tuple[float, float, float, float]


@dataclass(frozen=True)
class FlightReferenceConfig:
    max_track_tilt: float = math.radians(60.0)  # cap on tilt while tracking
    slew_rate: float = math.radians(180.0)  # rad/s, max reference rotation rate
    # Default target when none is set: this far ahead of the take-off point
    # along the heading, at the take-off ground height.
    default_target_distance: float = 3.0  # m
    standing_height: float = 0.27  # m, body origin above the ground when standing


def look_at_quaternion(
    camera_position: np.ndarray,
    target: np.ndarray,
    yaw_reference: float,
    max_tilt: float,
) -> Quaternion:
    """Attitude whose -z axis (the camera) points from ``camera_position`` at
    ``target``, tilt from vertical capped at ``max_tilt``, heading kept."""
    z_axis = np.asarray(camera_position, float) - np.asarray(target, float)
    norm = float(np.linalg.norm(z_axis))
    if norm < 1e-6:
        z_axis = np.array([0.0, 0.0, 1.0])
    else:
        z_axis = z_axis / norm
    tilt = math.acos(max(-1.0, min(1.0, float(z_axis[2]))))
    if tilt > max_tilt:
        horizontal = z_axis[:2]
        h_norm = float(np.linalg.norm(horizontal))
        direction = horizontal / h_norm if h_norm > 1e-9 else np.array([1.0, 0.0])
        z_axis = np.array([
            direction[0] * math.sin(max_tilt),
            direction[1] * math.sin(max_tilt),
            math.cos(max_tilt),
        ])
    return tilted_reference_quaternion(z_axis, yaw_reference)


def slew_quaternion(current: Quaternion, desired: Quaternion, max_angle: float) -> Quaternion:
    """Rotate ``current`` towards ``desired`` by at most ``max_angle`` [rad]."""
    q0 = np.asarray(current, float)
    q1 = np.asarray(desired, float)
    dot = float(q0 @ q1)
    if dot < 0.0:
        q1, dot = -q1, -dot
    dot = min(1.0, dot)
    angle = 2.0 * math.acos(dot)
    if angle <= max_angle or angle < 1e-9:
        q = q1
    else:
        # slerp by the fraction of the angle we are allowed to cover
        t = max_angle / angle
        half = angle / 2.0
        q = (math.sin((1 - t) * half) * q0 + math.sin(t * half) * q1) / math.sin(half)
    q = q / np.linalg.norm(q)
    if q[0] < 0:
        q = -q
    return tuple(float(v) for v in q)


def default_target(takeoff_position: np.ndarray, heading: float,
                   config: FlightReferenceConfig = FlightReferenceConfig()) -> np.ndarray:
    p = np.asarray(takeoff_position, float)
    return np.array([
        p[0] + config.default_target_distance * math.cos(heading),
        p[1] + config.default_target_distance * math.sin(heading),
        p[2] - config.standing_height,
    ])


class FlightReference:
    def __init__(self, config: FlightReferenceConfig = FlightReferenceConfig(),
                 target: tuple[float, float, float] | None = None) -> None:
        self.config = config
        self.target_setting = None if target is None else np.asarray(target, float)
        self.active_target: np.ndarray | None = None
        self.commanded: Quaternion = (1.0, 0.0, 0.0, 0.0)
        self.tracking = False

    def set_target(self, target: tuple[float, float, float] | None) -> None:
        self.target_setting = None if target is None else np.asarray(target, float)

    def start_flight(self, attitude: Quaternion, position: np.ndarray, heading: float,
                     track: bool) -> None:
        """Called at lift-off; ``track`` is True for flights started by a jump."""
        self.commanded = tuple(float(v) for v in attitude)
        self.tracking = track
        if track:
            self.active_target = (
                self.target_setting.copy() if self.target_setting is not None
                else default_target(position, heading, self.config)
            )

    def update(self, phase: FlightPhase, camera_position: np.ndarray, yaw_reference: float,
               landing_reference: Quaternion, dt: float) -> Quaternion:
        if phase is FlightPhase.TAKEOFF:
            return self.commanded  # hold the lift-off attitude until clear of the ground
        if phase is FlightPhase.ASCENT and self.tracking and self.active_target is not None:
            desired = look_at_quaternion(camera_position, self.active_target, yaw_reference,
                                         self.config.max_track_tilt)
        else:  # DESCENT, LANDING
            desired = landing_reference  # level until the ground is seen, then the plane
        self.commanded = slew_quaternion(self.commanded, desired, self.config.slew_rate * dt)
        return self.commanded

    def camera_pointing_error(self, attitude: Quaternion, camera_position: np.ndarray) -> float:
        """Angle [rad] between the camera axis (body -z) and the target direction."""
        if self.active_target is None:
            return math.nan
        axis = -quaternion_to_matrix(attitude)[:, 2]
        to_target = self.active_target - np.asarray(camera_position, float)
        to_target /= np.linalg.norm(to_target)
        return math.acos(max(-1.0, min(1.0, float(axis @ to_target))))
