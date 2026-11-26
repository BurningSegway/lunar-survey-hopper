"""Attitude reference over a hop: track a point on the way up, land on the way down.

* TAKEOFF (lift-off until 0.5 m up): the attitude the robot left the
  ground with is held.
* ASCENT (only on flights started by a jump): the robot's bottom (body -z,
  in the middle of the side-mounted lidar's field of view) is pointed at a
  target point and keeps tracking it as the robot moves. The target is set
  as a ground position (x, y); the tracked point is ``aim_height`` above the
  ground there, so it is never in the air or buried. The required tilt is
  capped at ``max_track_tilt``; heading stays at the yaw reference.
* DESCENT: tracking goes on until the turn to the landing reference is
  due: until the time left to impact is no more than the time the turn
  takes, ``turn_time`` = ``turn_time_base`` + ``turn_time_per_degree`` x the
  angle from the current attitude to the landing reference (measured with
  the flight LQR, see the config). Then the reference turns to the
  landing-plane reference from :class:`~jumper_sim.landing_reference.LandingPlane`
  -- it starts at "level" and only receives fits during descent, so "level
  until the landing spot is seen, then its plane" falls out of it directly.
  It keeps updating until the moment of impact (LANDING too), so the robot
  keeps aligning with the ground to the end. Without a time to impact, the
  target is tracked until LANDING (touchdown near).
* Heading in flight: the yaw reference (the lift-off heading) throughout.
* The commanded reference never jumps: it slews towards the desired one at
  ``slew_rate`` (also at lift-off, starting from the attitude the robot left
  the ground with), so the controller is not hit with large reference steps.

Pure numpy; meant to run unchanged on the robot.
"""

from __future__ import annotations

from collections.abc import Callable
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
    # along the heading.
    default_target_distance: float = 3.0  # m
    # Tracked point above the ground at the target: the middle of the 0.3 m
    # target object.
    aim_height: float = 0.15  # m
    # Time to turn to the landing reference: base + per_degree * angle [deg].
    # Measured with the flight LQR (benchmarks/turn_rate_benchmark, z axis,
    # 10-80 deg reference steps both ways, time until within 2 deg; fitted
    # line raised so no measured step is slower; 2026-11-26).
    turn_time_base: float = 0.59  # s
    turn_time_per_degree: float = 0.0039  # s/deg
    # Extra time before the turn is due, for the time to impact's error.
    turn_time_margin: float = 0.0  # s
    # Without a ground height, the ground is taken as the take-off ground.
    standing_height: float = 0.27  # m, body origin above the ground when standing

GroundHeight = Callable[[float, float], float]  # terrain height [m] at (x, y)


def look_at_quaternion(
    sensor_position: np.ndarray,
    target: np.ndarray,
    yaw_reference: float,
    max_tilt: float,
) -> Quaternion:
    """Attitude whose -z axis points from ``sensor_position`` at ``target``,
    tilt from vertical capped at ``max_tilt``, heading kept."""
    z_axis = np.asarray(sensor_position, float) - np.asarray(target, float)
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


def attitude_angle(q1: Quaternion, q2: Quaternion) -> float:
    """Angle [deg] of the rotation between two attitudes."""
    dot = abs(float(np.dot(np.asarray(q1, float), np.asarray(q2, float))))
    return math.degrees(2.0 * math.acos(min(1.0, dot)))


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
    """Ground position (x, y) ``default_target_distance`` ahead of take-off."""
    p = np.asarray(takeoff_position, float)
    return np.array([
        p[0] + config.default_target_distance * math.cos(heading),
        p[1] + config.default_target_distance * math.sin(heading),
    ])


class FlightReference:
    def __init__(self, config: FlightReferenceConfig = FlightReferenceConfig(),
                 target: tuple[float, float] | None = None,
                 ground_height: GroundHeight | None = None) -> None:
        self.config = config
        self.ground_height = ground_height
        self.target_setting: np.ndarray | None = None  # tracked point (x, y, z)
        self.set_target(target)
        self.active_target: np.ndarray | None = None
        self.commanded: Quaternion = (1.0, 0.0, 0.0, 0.0)
        self.tracking = False
        self.landing_turn = False  # the turn to the landing reference has started
        # (angle [deg], time to impact [s]) when it started
        self.landing_turn_start: tuple[float, float] | None = None

    def set_target(self, target: tuple[float, float] | None) -> None:
        """Track ``aim_height`` above the ground at ``target`` (x, y)."""
        self.target_setting = None if target is None else self.aim_point(target)

    def aim_point(self, xy, takeoff_position: np.ndarray | None = None) -> np.ndarray:
        """The tracked point ``aim_height`` above the ground at ``xy``.

        Off the terrain (or without a ground height) the ground is the take-off
        ground, or z = 0 before any take-off.
        """
        x, y = (float(v) for v in xy)
        ground = math.nan if self.ground_height is None else float(self.ground_height(x, y))
        if not math.isfinite(ground):
            ground = (0.0 if takeoff_position is None
                      else float(takeoff_position[2]) - self.config.standing_height)
        return np.array([x, y, ground + self.config.aim_height])

    def start_flight(self, attitude: Quaternion, position: np.ndarray, heading: float,
                     track: bool) -> None:
        """Called at lift-off; ``track`` is True for flights started by a jump.

        ``attitude`` is held in TAKEOFF and is where the reference starts
        from: on a jump the attitude at the jump command (before the push
        turned the body), on a bounce the attitude at lift-off.
        """
        self.commanded = tuple(float(v) for v in attitude)
        self.tracking = track
        self.landing_turn = False
        self.landing_turn_start = None
        if track:
            self.active_target = (
                self.target_setting.copy() if self.target_setting is not None
                else self.aim_point(default_target(position, heading, self.config), position)
            )

    def turn_time(self, angle_deg: float) -> float:
        """Time [s] to turn by ``angle_deg`` and settle within 2 deg."""
        return self.config.turn_time_base + self.config.turn_time_per_degree * angle_deg

    def update(self, phase: FlightPhase, sensor_position: np.ndarray, yaw_reference: float,
               landing_reference: Callable[[float], Quaternion], dt: float,
               attitude: Quaternion | None = None,
               time_to_impact: float | None = None) -> Quaternion:
        """``landing_reference(yaw)``: the landing attitude with that heading;
        ``attitude``: the robot's; ``time_to_impact`` [s]: predicted (None:
        not known yet)."""
        if phase is FlightPhase.TAKEOFF:
            return self.commanded  # hold the start_flight attitude until clear of the ground
        yaw = yaw_reference
        landing = landing_reference(yaw)  # level until the landing spot is seen, then its plane
        if (phase is FlightPhase.DESCENT and self.tracking and not self.landing_turn
                and attitude is not None and time_to_impact is not None):
            angle = attitude_angle(attitude, landing)
            if time_to_impact <= self.turn_time(angle) + self.config.turn_time_margin:
                self.landing_turn = True
                self.landing_turn_start = (angle, time_to_impact)
        if self.tracking_target(phase):
            desired = look_at_quaternion(sensor_position, self.active_target, yaw,
                                         self.config.max_track_tilt)
        else:  # DESCENT once the landing turn is due, LANDING
            desired = landing
        self.commanded = slew_quaternion(self.commanded, desired, self.config.slew_rate * dt)
        return self.commanded

    def tracking_target(self, phase: FlightPhase) -> bool:
        """Whether body -z is pointed at the target in this phase."""
        if not self.tracking or self.active_target is None:
            return False
        if phase is FlightPhase.ASCENT:
            return True
        return phase is FlightPhase.DESCENT and not self.landing_turn

    def axis_target_angle(self, attitude: Quaternion, sensor_position: np.ndarray) -> float:
        """Angle [rad] between body -z and the target direction."""
        if self.active_target is None:
            return math.nan
        axis = -quaternion_to_matrix(attitude)[:, 2]
        to_target = self.active_target - np.asarray(sensor_position, float)
        to_target /= np.linalg.norm(to_target)
        return math.acos(max(-1.0, min(1.0, float(axis @ to_target))))
