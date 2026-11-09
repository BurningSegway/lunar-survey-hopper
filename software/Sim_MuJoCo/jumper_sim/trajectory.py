"""Hop trajectory from the IMU: the firmware's tracker (main.cpp, "trajectory
estimation"), in the world frame.

From the jump command the robot is at rest at its launch position. Every IMU
sample's specific force is rotated into the world frame and gravity is added
back, giving the linear acceleration a, which is integrated with forward
Euler (previous sample's velocity and acceleration):

    p[k] = p[k-1] + v[k-1] * dt
    v[k] = v[k-1] + a[k-1] * dt

At propulsion cutoff (the specific force drops below ``cutoff_force`` g after
the push-off exceeded ``launch_force`` g) p and v are latched (p_c, v0) and
integration stops: from then on only gravity acts, so the whole flight is the
fixed ballistic arc

    p(t) = p_c + v0 t - [0, 0, g/2] t^2,   t = time since cutoff.

It ends at touchdown. There is no reason to expect the trajectory to change
in flight, so nothing updates it. Pure numpy; in the simulator the IMU is the
body's velocity change per step (see JumperSimulation._update_flight_phase).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .landing_reference import quaternion_to_matrix


@dataclass(frozen=True)
class TrajectoryConfig:
    launch_force: float = 1.25  # g: the push-off has started
    cutoff_force: float = 0.5  # g: the push-off has ended (free flight reads ~0)


class TrajectoryTracker:
    def __init__(self, config: TrajectoryConfig = TrajectoryConfig()) -> None:
        self.config = config
        self.reset()

    def reset(self) -> None:
        """No trajectory (on the ground, or after touchdown)."""
        self.integrating = False
        self.ballistic = False
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self._acceleration = np.zeros(3)
        self._launched = False
        self.cutoff_time = 0.0
        self.cutoff_position = np.zeros(3)
        self.launch_velocity = np.zeros(3)
        self.gravity = 0.0

    def begin(self, position) -> None:
        """Jump command: integrate from rest at ``position``."""
        self.reset()
        self.integrating = True
        self.position = np.asarray(position, float).copy()

    def update(self, time: float, specific_force_body, body_quaternion,
               gravity_vector, dt: float) -> None:
        """One IMU sample taken at ``time``."""
        if not self.integrating:
            return
        gravity_vector = np.asarray(gravity_vector, float)
        self.position = self.position + self.velocity * dt
        self.velocity = self.velocity + self._acceleration * dt
        f_world = quaternion_to_matrix(body_quaternion) @ np.asarray(specific_force_body, float)
        self._acceleration = f_world + gravity_vector
        g = float(np.linalg.norm(gravity_vector))
        force = float(np.linalg.norm(specific_force_body))
        if force >= self.config.launch_force * g:
            self._launched = True
        elif self._launched and force < self.config.cutoff_force * g:
            self.latch(time, self.position, self.velocity, g)

    def latch(self, time: float, position, velocity, gravity: float) -> None:
        """Cutoff: fix the ballistic arc (also for a robot thrown or dropped
        in a test, whose state is known)."""
        self.integrating = False
        self.ballistic = True
        self.cutoff_time = time
        self.cutoff_position = np.asarray(position, float).copy()
        self.launch_velocity = np.asarray(velocity, float).copy()
        self.gravity = gravity

    def end(self) -> None:
        """Touchdown."""
        self.reset()

    def state(self, time: float) -> tuple[np.ndarray, np.ndarray]:
        """(position, velocity) on the arc at ``time`` (only while ballistic)."""
        t = time - self.cutoff_time
        down = np.array([0.0, 0.0, self.gravity])
        return (self.cutoff_position + self.launch_velocity * t - 0.5 * down * t * t,
                self.launch_velocity - down * t)
