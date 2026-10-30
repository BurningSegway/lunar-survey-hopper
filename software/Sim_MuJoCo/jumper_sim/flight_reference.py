"""Attitude reference over a hop: track a point on the way up, land on the way down.

* TAKEOFF (lift-off until 0.5 m up): the attitude the robot left the
  ground with is held.
* ASCENT (only on flights started by a jump): the robot's bottom (body -z,
  in the middle of the side-mounted lidar's field of view) is pointed at a
  target point and keeps tracking it as the robot moves. The target is set
  as a ground position (x, y); the tracked point is ``aim_height`` above the
  ground there, so it is never in the air or buried. The required tilt is
  capped at ``max_track_tilt``; heading stays at the yaw reference.
* DESCENT: tracking goes on for ``descent_track_fraction`` (half) of the
  time from lift-off to apex -- the descent takes about as long as the ascent
  (to be replaced by a criterion from the predicted landing spot). Then the
  reference moves back towards level (the xy plane). As soon as
  the ground at the predicted landing spot has been fitted, the landing-plane
  reference from :class:`~jumper_sim.landing_reference.LandingReference`
  takes over -- that filter starts at "level" and only receives fits during
  descent, so "level until the landing spot is seen, then its plane" falls
  out of it directly. It keeps updating until the moment of impact
  (LANDING too), so the robot keeps aligning with the ground to the end.
* Heading in flight (ASCENT and DESCENT; the tracking axis, body -z, is
  not affected by it), for a lidar tilted off body -z towards body +x:
  with ``face_travel`` the robot turns so body +x faces its horizontal
  velocity, so the lidar looks ahead to where it will land; with ``spin``
  it turns slowly (``spin_rate``) instead, sweeping the lidar round a cone,
  and stops once the predicted touchdown is within ``spin_stop_time``.
  Turns are limited to ``turn_rate`` and ``turn_acceleration``; in LANDING
  the turn stops and the heading reached is kept.
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
    # Keep tracking into the descent for this fraction of the ascent's time.
    descent_track_fraction: float = 0.5
    # Without a ground height, the ground is taken as the take-off ground.
    standing_height: float = 0.27  # m, body origin above the ground when standing
    # Heading in flight (neither: hold the lift-off heading).
    face_travel: bool = False  # body +x towards the horizontal velocity
    spin: bool = False  # slow spin instead
    spin_rate: float = math.radians(45.0)  # rad/s, 8 s a turn
    spin_stop_time: float = 1.0  # s before the predicted touchdown: stopped
    turn_rate: float = math.radians(90.0)  # rad/s, max turn rate to face travel
    turn_acceleration: float = math.radians(90.0)  # rad/s^2
    min_travel_speed: float = 0.2  # m/s, slower: no travel direction

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
        self.heading_offset = 0.0  # rad, heading turned since lift-off
        self.heading_rate = 0.0  # rad/s, current turn rate
        self.ascent_time = 0.0  # s, lift-off to apex
        self.descent_time = 0.0  # s, since apex

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
        """Called at lift-off; ``track`` is True for flights started by a jump."""
        self.commanded = tuple(float(v) for v in attitude)
        self.tracking = track
        self.heading_offset = 0.0
        self.heading_rate = 0.0
        self.ascent_time = 0.0
        self.descent_time = 0.0
        if track:
            self.active_target = (
                self.target_setting.copy() if self.target_setting is not None
                else self.aim_point(default_target(position, heading, self.config), position)
            )

    def update(self, phase: FlightPhase, sensor_position: np.ndarray, yaw_reference: float,
               landing_reference: Callable[[float], Quaternion], dt: float,
               touchdown_in: float | None = None,
               velocity: np.ndarray | None = None) -> Quaternion:
        """``landing_reference(yaw)``: the landing attitude with that heading;
        ``touchdown_in``: predicted time to touchdown [s] (None: unknown);
        ``velocity``: the robot's (world) velocity, for ``face_travel``."""
        if phase in (FlightPhase.TAKEOFF, FlightPhase.ASCENT):
            self.ascent_time += dt
        elif phase is FlightPhase.DESCENT:
            self.descent_time += dt
        if phase is FlightPhase.TAKEOFF:
            return self.commanded  # hold the lift-off attitude until clear of the ground
        self._update_heading(phase, dt, yaw_reference, touchdown_in, velocity)
        yaw = yaw_reference + self.heading_offset
        if self.tracking_target(phase):
            desired = look_at_quaternion(sensor_position, self.active_target, yaw,
                                         self.config.max_track_tilt)
        else:  # DESCENT, LANDING
            desired = landing_reference(yaw)  # level until the landing spot is seen, then its plane
        self.commanded = slew_quaternion(self.commanded, desired, self.config.slew_rate * dt)
        return self.commanded

    def tracking_target(self, phase: FlightPhase) -> bool:
        """Whether body -z is pointed at the target in this phase."""
        if not self.tracking or self.active_target is None:
            return False
        if phase is FlightPhase.ASCENT:
            return True
        return (phase is FlightPhase.DESCENT
                and self.descent_time < self.config.descent_track_fraction * self.ascent_time)

    def _update_heading(self, phase: FlightPhase, dt: float, yaw_reference: float,
                        touchdown_in: float | None, velocity: np.ndarray | None) -> None:
        config = self.config
        wanted = 0.0
        if phase in (FlightPhase.ASCENT, FlightPhase.DESCENT):
            if config.spin:
                if touchdown_in is None or touchdown_in > config.spin_stop_time:
                    wanted = config.spin_rate
            elif (config.face_travel and velocity is not None
                  and math.hypot(velocity[0], velocity[1]) > config.min_travel_speed):
                error = math.atan2(velocity[1], velocity[0]) - (yaw_reference + self.heading_offset)
                error = math.atan2(math.sin(error), math.cos(error))
                # Fastest turn that can still stop on the heading.
                wanted = math.copysign(
                    min(config.turn_rate, math.sqrt(2.0 * config.turn_acceleration * abs(error))),
                    error)
        step = config.turn_acceleration * dt
        self.heading_rate += max(-step, min(step, wanted - self.heading_rate))
        self.heading_offset += self.heading_rate * dt

    def axis_target_angle(self, attitude: Quaternion, sensor_position: np.ndarray) -> float:
        """Angle [rad] between body -z and the target direction."""
        if self.active_target is None:
            return math.nan
        axis = -quaternion_to_matrix(attitude)[:, 2]
        to_target = self.active_target - np.asarray(sensor_position, float)
        to_target /= np.linalg.norm(to_target)
        return math.acos(max(-1.0, min(1.0, float(axis @ to_target))))
