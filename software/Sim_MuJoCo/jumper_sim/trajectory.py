"""Hop trajectory from the IMU: the firmware's tracker (main.cpp, "trajectory
estimation"), in the world frame.

From the jump command the robot is at rest at its launch position. Every IMU
sample's specific force is rotated into the world frame and gravity is added
back, giving the linear acceleration a, which is integrated with forward
Euler (previous sample's velocity and acceleration):

    p[k] = p[k-1] + v[k-1] * dt
    v[k] = v[k-1] + a[k-1] * dt

The propulsion cutoff is the first sample on which the specific force drops
below ``cutoff_force`` g after the push-off exceeded ``launch_force`` g. The
foot and springs keep pushing the body for a while after it (on the Moon
~3 g within ~10 ms of lift-off), so integration goes on for ``latch_delay``
(``FREE_FALL_DELAY``, 60 ms, the contact estimate's bounce delay) and only
then are p and v latched (p_a, v0 at t_a) and integration stops: from then
on only gravity acts, so the rest of the flight is the fixed ballistic arc

    p(t) = p_a + v0 (t - t_a) - [0, 0, g/2] (t - t_a)^2.

Latching at the cutoff put the arc ~50 cm low at a Moon jump's touchdown
(the error is nearly all vertical); 60 ms later it is ~10 cm (measured over
six hops, see CLAUDE.md).

Touchdown ends it, and the tracker runs again for the bounce: on every
sample the robot touches the ground, integration restarts from the arc's
position and velocity there (so an arc latched after a dip below the cutoff
while still sliding or rocking on the ground is thrown away again), and the
bounce's arc is latched ``latch_delay`` after its cutoff (the impact counts
as the push-off). If the robot stays down, nothing latches (it reads
~1 g) and settling on the ground resets it. There is no reason to expect a
trajectory to change in flight, so nothing updates an arc once latched. Pure numpy; in the simulator the IMU is the
body's velocity change per step (see JumperSimulation._update_flight_phase).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ground_contact import FREE_FALL_DELAY
from .landing_reference import quaternion_to_matrix


@dataclass(frozen=True)
class TrajectoryConfig:
    launch_force: float = 1.25  # g: the push-off has started
    cutoff_force: float = 0.5  # g: the push-off has ended (free flight reads ~0)
    latch_delay: float = FREE_FALL_DELAY  # s from the cutoff to the latch


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
        self.cut_off = False  # the push-off has ended; the latch follows
        self.cutoff_time = 0.0  # the cutoff sample
        # The arc: from arc_position, arc_velocity at arc_time (the latch).
        self.arc_time = 0.0
        self.arc_position = np.zeros(3)
        self.arc_velocity = np.zeros(3)
        self.gravity = 0.0

    def begin(self, position, velocity=(0.0, 0.0, 0.0)) -> None:
        """Jump command: integrate from rest at ``position`` (or from
        ``velocity``, for a bounce)."""
        self.reset()
        self.integrating = True
        self.position = np.asarray(position, float).copy()
        self.velocity = np.asarray(velocity, float).copy()

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
        if not self.cut_off:
            if force >= self.config.launch_force * g:
                self._launched = True
            elif self._launched and force < self.config.cutoff_force * g:
                self.cut_off = True
                self.cutoff_time = time
        if self.cut_off and time - self.cutoff_time >= self.config.latch_delay - 1e-9:
            self.latch(time, self.position, self.velocity, g)

    def latch(self, time: float, position, velocity, gravity: float) -> None:
        """Fix the ballistic arc (``latch_delay`` after the cutoff; also for a
        robot thrown or dropped in a test, whose state is known)."""
        self.integrating = False
        self.ballistic = True
        self.arc_time = time
        self.arc_position = np.asarray(position, float).copy()
        self.arc_velocity = np.asarray(velocity, float).copy()
        self.gravity = gravity

    def end(self) -> None:
        """No more arc (e.g. touchdown with no bounce to follow)."""
        self.reset()

    def touchdown(self, time: float) -> None:
        """A sample in ground contact (call on every one): the arc ends and
        the tracker runs again for the bounce from the arc's state at
        ``time``. While integrating (the push-off, or a bounce's contact) it
        changes nothing; without an arc there is no state to start from."""
        if not self.ballistic:
            return
        position, velocity = self.state(time)
        self.begin(position, velocity)
        self._launched = True  # the impact is the bounce's push-off

    def path(self, end_time: float, period: float = 0.02) -> np.ndarray:
        """Positions on the arc from its latch to ``end_time`` (N x 3)."""
        duration = max(0.0, end_time - self.arc_time)
        times = np.append(np.arange(0.0, duration, period), duration)
        down = np.array([0.0, 0.0, 0.5 * self.gravity])
        return (self.arc_position + np.outer(times, self.arc_velocity)
                - np.outer(times * times, down))

    def state(self, time: float) -> tuple[np.ndarray, np.ndarray]:
        """(position, velocity) on the arc at ``time`` (only while ballistic)."""
        t = time - self.arc_time
        down = np.array([0.0, 0.0, self.gravity])
        return (self.arc_position + self.arc_velocity * t - 0.5 * down * t * t,
                self.arc_velocity - down * t)
