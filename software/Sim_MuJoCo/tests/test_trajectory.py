import math
import unittest

import numpy as np

from jumper_sim.trajectory import TrajectoryTracker

G = 1.62
GRAVITY = np.array([0.0, 0.0, -G])
LEVEL = (1.0, 0.0, 0.0, 0.0)
DT = 0.001


def yaw_quaternion(yaw):
    return (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))


class TrajectoryTrackerTests(unittest.TestCase):
    def launch(self, tracker, push_world, push_time, quaternion=LEVEL, start=(1.0, 2.0, 0.5)):
        """Rest, a constant push (world-frame linear acceleration), then free
        flight; returns the true velocity and position at the end of the push."""
        r = np.array([[1 - 2 * quaternion[3] ** 2, -2 * quaternion[0] * quaternion[3], 0.0],
                      [2 * quaternion[0] * quaternion[3], 1 - 2 * quaternion[3] ** 2, 0.0],
                      [0.0, 0.0, 1.0]])  # yaw-only rotation, body -> world
        tracker.begin(start)
        time = 0.0
        push = np.asarray(push_world, float)
        for _ in range(round(push_time / DT)):
            time += DT
            tracker.update(time, r.T @ (push - GRAVITY), quaternion, GRAVITY, DT)
        for _ in range(3):  # free flight: the accelerometer reads ~0
            time += DT
            tracker.update(time, np.zeros(3), quaternion, GRAVITY, DT)
        velocity = push * push_time
        position = np.asarray(start) + 0.5 * push * push_time**2
        return time, velocity, position

    def test_latches_the_push_off_velocity_and_flies_a_fixed_arc(self) -> None:
        tracker = TrajectoryTracker()
        push = (2.0, -1.0, 40.0)
        for yaw in (0.0, 1.2):
            with self.subTest(yaw=yaw):
                time, velocity, position = self.launch(tracker, push, 0.1, yaw_quaternion(yaw))
                self.assertTrue(tracker.ballistic)
                self.assertFalse(tracker.integrating)
                # Forward Euler lags the true values by one sample's step.
                np.testing.assert_allclose(tracker.launch_velocity, velocity, rtol=0.02)
                np.testing.assert_allclose(tracker.cutoff_position, position, atol=0.01)
                self.assertAlmostEqual(tracker.gravity, G)
                # The arc: ballistic from the cutoff, and the IMU no longer counts.
                tracker.update(time + DT, np.array([50.0, 0.0, 0.0]), LEVEL, GRAVITY, DT)
                t = 2.0
                p, v = tracker.state(tracker.cutoff_time + t)
                np.testing.assert_allclose(
                    p, tracker.cutoff_position + tracker.launch_velocity * t - (0, 0, 0.5 * G * t * t))
                np.testing.assert_allclose(v, tracker.launch_velocity - (0, 0, G * t))

    def test_standing_still_does_not_latch(self) -> None:
        tracker = TrajectoryTracker()
        tracker.begin((0.0, 0.0, 0.0))
        for k in range(500):
            tracker.update(k * DT, -GRAVITY, LEVEL, GRAVITY, DT)
        self.assertTrue(tracker.integrating)
        self.assertFalse(tracker.ballistic)
        np.testing.assert_allclose(tracker.velocity, 0.0, atol=1e-12)

    def test_touchdown_ends_it_and_a_known_state_can_be_latched(self) -> None:
        tracker = TrajectoryTracker()
        self.launch(tracker, (0.0, 0.0, 30.0), 0.1)
        tracker.end()
        self.assertFalse(tracker.ballistic or tracker.integrating)
        tracker.latch(3.0, (0.0, 0.0, 5.0), (1.0, 0.0, 0.0), G)
        p, v = tracker.state(4.0)
        np.testing.assert_allclose(p, (1.0, 0.0, 5.0 - 0.5 * G))
        np.testing.assert_allclose(v, (1.0, 0.0, -G))


if __name__ == "__main__":
    unittest.main()
