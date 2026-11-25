import unittest

import numpy as np

from jumper_sim.flight_phase import FlightPhase
from jumper_sim.ground_contact import GroundContactConfig, GroundContactEstimator

G = 1.62
DT = 0.001


def force(g_units: float) -> np.ndarray:
    return np.array([0.0, 0.0, g_units * G])


class GroundContactTests(unittest.TestCase):
    def feed(self, estimator, g_units, phase, steps=1):
        for _ in range(steps):
            airborne = estimator.update(force(g_units), G, DT, phase)
        return airborne

    def standing(self) -> GroundContactEstimator:
        estimator = GroundContactEstimator()
        estimator.reset(airborne=False)
        self.assertFalse(self.feed(estimator, 1.0, FlightPhase.GROUND, 500))
        return estimator

    def test_jump_is_airborne_on_the_cutoff_sample(self) -> None:
        estimator = self.standing()
        self.assertFalse(self.feed(estimator, 1.0, FlightPhase.TAKEOFF, 9))  # before the push
        self.assertFalse(self.feed(estimator, 100.0, FlightPhase.TAKEOFF, 30))  # the push
        self.assertFalse(self.feed(estimator, 0.6, FlightPhase.TAKEOFF))  # still pushing a little
        self.assertTrue(self.feed(estimator, 0.2, FlightPhase.TAKEOFF))  # the cutoff: no delay

    def test_standing_still_ends_a_missed_landing(self) -> None:
        # A touchdown too soft to read as an impact: once the phases see the
        # robot standing (GROUND), the flight ends; pushed off, it bounces.
        estimator = GroundContactEstimator()
        self.assertTrue(self.feed(estimator, 1.0, FlightPhase.LANDING, 500))
        self.assertFalse(self.feed(estimator, 1.0, FlightPhase.GROUND))
        self.assertFalse(self.feed(estimator, 1.0, FlightPhase.GROUND, 500))
        self.assertFalse(self.feed(estimator, 0.0, FlightPhase.GROUND, 59))
        self.assertTrue(self.feed(estimator, 0.0, FlightPhase.GROUND))

    def test_free_fall_before_the_push_is_not_a_lift_off(self) -> None:
        estimator = self.standing()
        self.assertFalse(self.feed(estimator, 0.1, FlightPhase.TAKEOFF, 5))

    def test_wind_back_spike_in_the_ascent_is_not_a_landing(self) -> None:
        estimator = GroundContactEstimator()
        self.assertTrue(self.feed(estimator, 1250.0, FlightPhase.ASCENT, 25))
        self.assertTrue(self.feed(estimator, 1250.0, FlightPhase.TAKEOFF, 5))

    def test_impact_ends_the_flight_in_the_descent_and_landing(self) -> None:
        for phase in (FlightPhase.DESCENT, FlightPhase.LANDING):
            estimator = GroundContactEstimator()
            self.assertTrue(self.feed(estimator, 1.2, phase, 10))  # below the impact force
            self.assertFalse(self.feed(estimator, 15.0, phase))

    def test_bounce_needs_unbroken_free_fall(self) -> None:
        estimator = GroundContactEstimator()
        self.feed(estimator, 15.0, FlightPhase.LANDING)
        needed = round(GroundContactConfig().bounce_free_fall_time / DT)
        # Rocking: dips into free fall, each shorter than needed.
        for _ in range(5):
            self.assertFalse(self.feed(estimator, 0.1, FlightPhase.LANDING, needed - 1))
            self.assertFalse(self.feed(estimator, 0.7, FlightPhase.LANDING))
        self.assertFalse(self.feed(estimator, 0.1, FlightPhase.LANDING, needed - 1))
        self.assertTrue(self.feed(estimator, 0.1, FlightPhase.LANDING))  # the bounce
        self.assertTrue(self.feed(estimator, 0.9, FlightPhase.LANDING, 5))  # legs shaking in the air
        self.assertFalse(self.feed(estimator, 8.0, FlightPhase.LANDING))  # its touchdown

    def test_reset_starts_airborne_like_the_hop_phases(self) -> None:
        estimator = GroundContactEstimator()
        estimator.reset(airborne=False)
        estimator.reset()
        self.assertTrue(estimator.airborne)


if __name__ == "__main__":
    unittest.main()
