import math
import unittest

import numpy as np

from jumper_sim.ground_contact import FREE_FALL_DELAY, GroundContactConfig
from jumper_sim.trajectory import TrajectoryConfig, TrajectoryTracker

G = 1.62
GRAVITY = np.array([0.0, 0.0, -G])
LEVEL = (1.0, 0.0, 0.0, 0.0)
DT = 0.001


def yaw_quaternion(yaw):
    return (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))


class TrajectoryTrackerTests(unittest.TestCase):
    def launch(self, tracker, push_world, push_time, quaternion=LEVEL, start=(1.0, 2.0, 0.5)):
        """Rest, a constant push (world-frame linear acceleration), then free
        flight up to the latch; returns its time and the true velocity and
        position then."""
        r = np.array([[1 - 2 * quaternion[3] ** 2, -2 * quaternion[0] * quaternion[3], 0.0],
                      [2 * quaternion[0] * quaternion[3], 1 - 2 * quaternion[3] ** 2, 0.0],
                      [0.0, 0.0, 1.0]])  # yaw-only rotation, body -> world
        tracker.begin(start)
        time = 0.0
        push = np.asarray(push_world, float)
        for _ in range(round(push_time / DT)):
            time += DT
            tracker.update(time, r.T @ (push - GRAVITY), quaternion, GRAVITY, DT)
        for _ in range(round(1.5 * tracker.config.latch_delay / DT) + 3):
            if tracker.ballistic:
                break
            time += DT  # free flight: the accelerometer reads ~0
            tracker.update(time, np.zeros(3), quaternion, GRAVITY, DT)
        flight = time - push_time
        velocity = push * push_time + GRAVITY * flight
        position = (np.asarray(start) + 0.5 * push * push_time**2 + push * push_time * flight
                    + 0.5 * GRAVITY * flight**2)
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
                np.testing.assert_allclose(tracker.arc_velocity, velocity, rtol=0.02)
                np.testing.assert_allclose(tracker.arc_position, position, atol=0.01)
                self.assertAlmostEqual(tracker.gravity, G)
                # Latched FREE_FALL_DELAY after the cutoff (the first free sample).
                self.assertAlmostEqual(tracker.cutoff_time, 0.1 + DT)
                self.assertAlmostEqual(tracker.arc_time, tracker.cutoff_time + FREE_FALL_DELAY)
                self.assertAlmostEqual(tracker.arc_time, time)
                # The arc: ballistic from the latch, and the IMU no longer counts.
                tracker.update(time + DT, np.array([50.0, 0.0, 0.0]), LEVEL, GRAVITY, DT)
                t = 2.0
                p, v = tracker.state(tracker.arc_time + t)
                np.testing.assert_allclose(
                    p, tracker.arc_position + tracker.arc_velocity * t - (0, 0, 0.5 * G * t * t))
                np.testing.assert_allclose(v, tracker.arc_velocity - (0, 0, G * t))

    def test_the_push_after_the_cutoff_is_in_the_arc(self) -> None:
        # The foot and springs push the body for a while after the cutoff:
        # the arc is latched FREE_FALL_DELAY later, with that push in it.
        self.assertEqual(TrajectoryConfig().latch_delay, FREE_FALL_DELAY)
        self.assertEqual(GroundContactConfig().bounce_free_fall_time, FREE_FALL_DELAY)
        results = {}
        for delay in (0.0, FREE_FALL_DELAY):
            tracker = TrajectoryTracker(TrajectoryConfig(latch_delay=delay))
            tracker.begin((0.0, 0.0, 0.0))
            time = 0.0
            # push 50 ms, free 10 ms, a 3 g push for 10 ms, then free flight
            for force in [30.0 * G] * 50 + [0.0] * 10 + [3.0 * G] * 10 + [0.0] * 100:
                time += DT
                tracker.update(time, np.array([0.0, 0.0, force]), LEVEL, GRAVITY, DT)
                if tracker.ballistic:
                    break
            self.assertTrue(tracker.ballistic)
            self.assertAlmostEqual(tracker.cutoff_time, 0.051)
            self.assertAlmostEqual(tracker.arc_time, 0.051 + delay)
            # The arc's velocity carried back to the cutoff.
            results[delay] = tracker.arc_velocity[2] + G * (tracker.arc_time - 0.051)
        true_at_cutoff = 29.0 * G * 0.05 + 2.0 * G * 0.01  # with the late push
        self.assertAlmostEqual(results[FREE_FALL_DELAY], true_at_cutoff, delta=0.05 * G)
        self.assertLess(results[0.0], true_at_cutoff - 1.5 * G * 0.01)  # misses it

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

    def test_a_bounce_is_tracked_from_the_arc_at_touchdown(self) -> None:
        tracker = TrajectoryTracker()
        tracker.latch(0.0, (0.0, 0.0, 5.0), (1.0, 0.0, 0.0), G)
        tracker.touchdown(1.0)  # contact while integrating changes nothing
        self.assertTrue(tracker.integrating)
        np.testing.assert_allclose(tracker.position, (1.0, 0.0, 5.0 - 0.5 * G))
        np.testing.assert_allclose(tracker.velocity, (1.0, 0.0, -G))
        tracker.touchdown(1.0 + DT)
        # The impact reverses the vertical velocity in 0.1 s (> 1.25 g) ...
        push = np.array([0.0, 0.0, 2.0 * G / 0.1])
        time = 1.0
        for _ in range(100):
            time += DT
            tracker.update(time, push - GRAVITY, LEVEL, GRAVITY, DT)
        self.assertFalse(tracker.ballistic)
        # ... the first sample off the ground is its cutoff, and the bounce's
        # arc is latched FREE_FALL_DELAY later.
        cutoff = time + DT
        while not tracker.ballistic and time < 2.0:
            time += DT
            tracker.update(time, np.zeros(3), LEVEL, GRAVITY, DT)
        self.assertAlmostEqual(tracker.cutoff_time, cutoff)
        self.assertAlmostEqual(tracker.arc_time, cutoff + FREE_FALL_DELAY)
        np.testing.assert_allclose(tracker.arc_velocity, (1.0, 0.0, G - G * FREE_FALL_DELAY), atol=0.05)

    def test_path_samples_the_arc_up_to_its_end(self) -> None:
        tracker = TrajectoryTracker()
        tracker.latch(1.0, (0.0, 0.0, 0.0), (1.0, 0.0, G), G)
        path = tracker.path(3.0, period=0.1)
        self.assertEqual(len(path), 21)
        np.testing.assert_allclose(path[0], (0.0, 0.0, 0.0))
        np.testing.assert_allclose(path[-1], tracker.state(3.0)[0])
        np.testing.assert_allclose(path[10], (1.0, 0.0, 0.5 * G), atol=1e-12)  # apex

    def test_contact_without_an_arc_starts_nothing(self) -> None:
        tracker = TrajectoryTracker()
        tracker.touchdown(1.0)
        self.assertFalse(tracker.ballistic or tracker.integrating)


if __name__ == "__main__":
    unittest.main()
