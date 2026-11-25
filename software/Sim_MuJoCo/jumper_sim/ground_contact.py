"""Ground contact from the IMU alone: when attitude control may act.

The robot has no contact sensor. What the IMU's accelerometer gives instead
is the specific force f: every force on the robot except gravity, per unit
mass. Besides gravity the only force on the robot is the ground's, through
the foot and the legs, so |f| tells contact from flight:

* standing: ~1 g (the ground carries the weight);
* the jump's push and impacts: tens of g or more (the firm contact makes
  sharp spikes; the sim's IMU is not clipped, the real MTi-610 saturates at
  +-10 g, which changes none of the rules below);
* free flight: ~0, going up and coming down alike.

Rules (thresholds in units of the local g; ``phase`` is the hop phase
before this sample):

* Jump (TAKEOFF): the push is seen as |f| >= ``launch_force``; the first
  later sample with |f| < ``free_fall_force`` is the cutoff -- the ground no
  longer pushes -- and the robot is airborne from that sample on, with no
  delay. These are the trajectory tracker's cutoff thresholds, so control
  starts on its cutoff sample (the tracker latches its arc
  ``FREE_FALL_DELAY`` later).
* Landing (DESCENT or LANDING): an impact, |f| > ``impact_force``, ends the
  flight. Not in TAKEOFF or ASCENT: the spring's wind-back 0.4 s after the
  release reads up to ~1250 g in flight.
* Standing still: the hop phases reach GROUND on stillness (~1 g, low body
  rate for 0.5 s); a flight the estimate still holds then ends (a landing
  too soft to read as an impact).
* Bounces (and leaving the ground without a jump): airborne again only after
  |f| < ``free_fall_force`` for ``bounce_free_fall_time`` without a break.
  Rocking or sliding on the ground also dips below it, mostly for < 20 ms
  (measured over hops and drops: 60 ms catches every bounce > 0.1 s, 60 ms
  median after the true lift-off).

JumperSimulation uses it wherever the robot needs to know whether it is
off the ground: attitude control (on from lift-off to impact, wheel momentum
bleed otherwise), the hop phases, the trajectory tracker's restarts, the
lidar, the scan and the landing fits. The simulator's contact is kept only
as ground truth for tests and benchmarks.

Pure numpy, meant to run unchanged on the robot from real IMU samples.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .flight_phase import FlightPhase


# How long the IMU waits after the ground's push ends before it trusts the
# flight: a bounce counts as airborne after this long below
# ``free_fall_force`` without a break, and the trajectory tracker latches
# its arc this long after the cutoff (the foot and springs keep pushing the
# body for a while after it). One constant for both (user's choice,
# 2026-11-25).
FREE_FALL_DELAY = 0.06  # s


@dataclass(frozen=True)
class GroundContactConfig:
    launch_force: float = 1.25  # g: the jump's push has started
    free_fall_force: float = 0.5  # g: below this nothing pushes on the robot
    impact_force: float = 1.25  # g: a landing impact
    bounce_free_fall_time: float = FREE_FALL_DELAY  # s of free fall before a bounce counts


class GroundContactEstimator:
    """Airborne or not, from the accelerometer (see the module docstring)."""

    def __init__(self, config: GroundContactConfig = GroundContactConfig()) -> None:
        self.config = config
        self.reset()

    def reset(self, airborne: bool = True) -> None:
        """``airborne``: the start state (the hop phases start in DESCENT)."""
        self.airborne = airborne
        self._launched = False  # the jump's push has been seen
        self._free_fall_for = 0.0

    def update(self, specific_force_body, gravity: float, dt: float,
               phase: FlightPhase) -> bool:
        """One IMU sample; returns whether the robot is airborne."""
        force = float(np.linalg.norm(specific_force_body)) / gravity
        config = self.config
        free_fall = force < config.free_fall_force
        self._free_fall_for = self._free_fall_for + dt if free_fall else 0.0
        if phase is not FlightPhase.TAKEOFF:
            self._launched = False
        if self.airborne and phase is FlightPhase.GROUND:
            self.airborne = False  # seen standing still: a missed landing
        if self.airborne:
            if (phase in (FlightPhase.DESCENT, FlightPhase.LANDING)
                    and force > config.impact_force):
                self.airborne = False
                self._free_fall_for = 0.0
            return self.airborne
        if phase is FlightPhase.TAKEOFF:
            if force >= config.launch_force:
                self._launched = True
            elif self._launched and free_fall:
                self.airborne = True  # the cutoff
                return True
        if self._free_fall_for >= config.bounce_free_fall_time - 1e-9:
            self.airborne = True
        return self.airborne
