import math
import unittest

import numpy as np

from jumper_sim.controller import quaternion_error
from jumper_sim.landing_reference import (
    GroundMap,
    LandingReference,
    LandingReferenceConfig,
    PlaneEstimate,
    fit_ground_plane,
    matrix_to_quaternion,
    quaternion_to_matrix,
    tilted_reference_quaternion,
)


def axis_angle_quaternion(axis, angle):
    axis = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    return (math.cos(angle / 2), *(math.sin(angle / 2) * axis))


def plane_points(slope_x, slope_y, centre=(0.0, 0.0), ground=0.0, count=1000, radius=0.5,
                 sigma=0.0, rock=None, seed=0):
    """Points of the plane through (centre, ground) with the given slopes, even
    over a disc of ``radius``. ``rock`` = (x, y, radius, bump) raises the
    ground inside a disc (offsets from the centre)."""
    rng = np.random.default_rng(seed)
    r = radius * np.sqrt(rng.random(count))
    a = rng.uniform(0.0, 2.0 * math.pi, count)
    dx, dy = r * np.cos(a), r * np.sin(a)
    z = ground + slope_x * dx + slope_y * dy + rng.standard_normal(count) * sigma
    if rock is not None:
        rx, ry, rr, bump = rock
        z = z + np.where(np.hypot(dx - rx, dy - ry) < rr, bump, 0.0)
    return np.column_stack([centre[0] + dx, centre[1] + dy, z])


def normal_of(slope_x, slope_y):
    n = np.array([-slope_x, -slope_y, 1.0])
    return n / np.linalg.norm(n)


def angle_between(a, b):
    return math.degrees(math.acos(min(1.0, float(np.dot(a, b)))))


class PlaneFitTests(unittest.TestCase):
    def test_recovers_the_plane_and_the_height_above_it(self) -> None:
        for slope in ((0.0, 0.0), (0.2, -0.1), (-0.3, 0.25)):
            points = plane_points(*slope, centre=(4.0, -2.0), ground=1.5)
            estimate = fit_ground_plane(points, (4.0, -2.0), robot_z=6.5, point_sigma=0.02,
                                        timestamp=0.0)
            self.assertTrue(estimate.valid)
            self.assertLess(angle_between(estimate.normal, normal_of(*slope)), 0.05)
            self.assertAlmostEqual(estimate.height, 5.0, places=3)  # 6.5 - 1.5

    def test_rock_is_rejected_as_outlier(self) -> None:
        clean = fit_ground_plane(plane_points(0.1, 0.0), (0.0, 0.0), 3.0, 0.02, 0.0)
        rocky = fit_ground_plane(plane_points(0.1, 0.0, rock=(0.25, 0.1, 0.12, 0.15)),
                                 (0.0, 0.0), 3.0, 0.02, 0.0)
        self.assertTrue(rocky.valid)
        self.assertLess(angle_between(rocky.normal, clean.normal), 0.3)

    def test_too_few_points_is_invalid(self) -> None:
        self.assertFalse(fit_ground_plane(plane_points(0.0, 0.0, count=5), (0.0, 0.0),
                                          3.0, 0.02, 0.0).valid)
        self.assertFalse(fit_ground_plane(np.zeros((0, 3)), (0.0, 0.0), 3.0, 0.02, 0.0).valid)

    def test_uncertainty_grows_with_noise_and_with_less_coverage(self) -> None:
        def sigma(**kwargs):
            points = plane_points(0.1, 0.0, **kwargs)
            noise = kwargs.get("sigma", 0.02)
            return fit_ground_plane(points, (0.0, 0.0), 3.0, noise, 0.0).normal_sigma
        self.assertGreater(sigma(sigma=0.08), 2.0 * sigma(sigma=0.02))
        self.assertGreater(sigma(sigma=0.02, radius=0.25), sigma(sigma=0.02))


class GroundMapTests(unittest.TestCase):
    def setUp(self) -> None:
        # Flat ground at z = 1 seen over x, y in [-1, 12] x [-3, 3].
        rng = np.random.default_rng(1)
        xy = rng.uniform((-1.0, -3.0), (12.0, 3.0), size=(40000, 2))
        self.map = GroundMap(0.5)
        for chunk in np.array_split(np.column_stack([xy, np.ones(len(xy))]), 20):
            self.map.add(chunk)  # added in batches, as the lidar does

    def test_landing_is_where_the_trajectory_meets_the_ground(self) -> None:
        g, clearance = 1.62, 0.3
        landing = self.map.predict_landing((0.0, 0.0, 5.0), (2.0, 0.0, 1.0), g, clearance,
                                           step=0.001)
        self.assertIsNotNone(landing)
        spot, time = landing
        # 5 + t - g t^2 / 2 = 1 + clearance
        expected = (1.0 + math.sqrt(1.0 + 2.0 * g * (5.0 - 1.0 - clearance))) / g
        self.assertAlmostEqual(time, expected, delta=0.002)
        np.testing.assert_allclose(spot, (2.0 * expected, 0.0), atol=0.005)

    def test_no_landing_over_ground_it_has_not_seen(self) -> None:
        self.assertIsNone(self.map.predict_landing((0.0, 0.0, 5.0), (0.0, 4.0, 0.0), 1.62, 0.3))

    def test_points_near_takes_at_most_max_points_from_the_disc(self) -> None:
        near = self.map.points_near((5.0, 0.0), 0.5, 200)
        self.assertEqual(len(near), 200)  # ~400 points in the disc
        self.assertLessEqual(float(np.hypot(near[:, 0] - 5.0, near[:, 1]).max()), 0.5)
        every = self.map.points_near((5.0, 0.0), 0.5, 100000)
        self.assertGreater(len(every), 300)
        self.assertLessEqual(float(np.hypot(every[:, 0] - 5.0, every[:, 1]).max()), 0.5)
        self.assertEqual(len(self.map.points_near((50.0, 0.0), 0.5, 1000)), 0)

    def test_clear_forgets_the_jump(self) -> None:
        self.map.clear()
        self.assertEqual(self.map.count, 0)
        self.assertTrue(math.isnan(self.map.height(5.0, 0.0)))


class LandingReferenceFilterTests(unittest.TestCase):
    def estimate(self, normal, sigma_deg, timestamp=0.1):
        n = np.asarray(normal, dtype=float)
        return PlaneEstimate(True, tuple(n / np.linalg.norm(n)), math.radians(sigma_deg),
                             timestamp=timestamp)

    def test_starts_level_and_confident_fit_takes_over(self) -> None:
        reference = LandingReference()
        np.testing.assert_allclose(reference.normal, (0.0, 0.0, 1.0))
        target = normal_of(0.2, 0.0)
        self.assertTrue(reference.update(self.estimate(target, 0.2)))
        self.assertLess(angle_between(reference.normal, target), 0.1)

    def test_noisy_fit_moves_reference_little(self) -> None:
        reference = LandingReference()
        reference.update(self.estimate(normal_of(0.2, 0.0), 30.0))
        self.assertLess(angle_between(reference.normal, (0, 0, 1)), 3.0)

    def test_tilt_is_clamped(self) -> None:
        config = LandingReferenceConfig(max_tilt=math.radians(20.0))
        reference = LandingReference(config)
        reference.update(self.estimate(normal_of(0.7, 0.0), 0.2))  # ~35 deg
        self.assertAlmostEqual(angle_between(reference.normal, (0, 0, 1)), 20.0, places=6)

    def test_outlier_fit_is_rejected_after_convergence(self) -> None:
        reference = LandingReference()
        for k in range(5):
            reference.update(self.estimate(normal_of(0.1, 0.0), 0.3, timestamp=0.03 * k))
        before = reference.normal
        self.assertFalse(reference.update(self.estimate(normal_of(-0.5, 0.3), 0.3, 0.16)))
        np.testing.assert_allclose(reference.normal, before)
        self.assertEqual(reference.rejected, 1)

    def test_noisy_far_fits_are_ignored(self) -> None:
        reference = LandingReference()
        self.assertFalse(reference.update(self.estimate(normal_of(0.2, 0.0), 6.0)))
        self.assertEqual(reference.ignored, 1)
        np.testing.assert_allclose(reference.normal, (0.0, 0.0, 1.0))

    def test_reference_keeps_updating_until_touchdown_and_flags_it_near(self) -> None:
        target = normal_of(0.15, 0.0)
        other = normal_of(-0.1, 0.1)
        for freeze in (False, True):
            reference = LandingReference(LandingReferenceConfig(freeze_near_touchdown=freeze))
            # Descending at 4 m/s: fits from 3.0 m and 2.6 m ...
            for t, h in ((0.0, 3.0), (0.1, 2.6)):
                n = target / np.linalg.norm(target)
                self.assertTrue(reference.update(PlaneEstimate(True, tuple(n), math.radians(0.3), height=h, timestamp=t)))
            held = reference.normal
            # ... then one 0.25 s from impact (1.0 m at 4 m/s) and one below
            # near_height: touchdown is near, and they are still fused (the
            # old behaviour, freeze_near_touchdown, holds the reference).
            for t, h in ((0.25, 1.0), (0.35, 0.6)):
                used = reference.update(PlaneEstimate(True, tuple(other), math.radians(0.3), height=h, timestamp=t))
                self.assertEqual(used, not freeze)
            self.assertEqual(reference.near_touchdown, 2)
            self.assertEqual(bool(np.allclose(reference.normal, held)), freeze)
            self.assertAlmostEqual(reference.descent_rate, 4.0, delta=1.5)

    def test_fused_uncertainty_has_a_floor(self) -> None:
        reference = LandingReference()
        for k in range(50):
            reference.update(self.estimate(normal_of(0.1, 0.0), 0.2, timestamp=0.03 * k))
        self.assertGreaterEqual(reference.sigma, reference.config.min_reference_sigma - 1e-12)

    def test_invalid_fits_hold_and_reset_returns_level(self) -> None:
        reference = LandingReference()
        reference.update(self.estimate(normal_of(0.1, 0.1), 0.2))
        held = reference.normal
        self.assertFalse(reference.update(PlaneEstimate(False, timestamp=1.0)))
        np.testing.assert_allclose(reference.normal, held)
        reference.reset()
        np.testing.assert_allclose(reference.normal, (0.0, 0.0, 1.0))


class ReferenceAttitudeTests(unittest.TestCase):
    def test_reference_quaternion_aligns_z_and_keeps_heading(self) -> None:
        normal = normal_of(0.2, -0.1)
        for yaw in (0.0, 0.7, -2.0):
            q = tilted_reference_quaternion(normal, yaw)
            r = quaternion_to_matrix(q)
            np.testing.assert_allclose(r[:, 2], normal, atol=1e-12)
            heading = math.atan2(r[1, 0], r[0, 0])
            self.assertAlmostEqual(math.remainder(heading - yaw, 2 * math.pi), 0.0, places=1)
        level = tilted_reference_quaternion(np.array([0.0, 0.0, 1.0]), 0.7)
        np.testing.assert_allclose(level, (math.cos(0.35), 0, 0, math.sin(0.35)), atol=1e-12)

    def test_quaternion_matrix_round_trip(self) -> None:
        rng = np.random.default_rng(2)
        for _ in range(20):
            q = rng.standard_normal(4)
            q /= np.linalg.norm(q)
            q *= np.sign(q[0])
            np.testing.assert_allclose(matrix_to_quaternion(quaternion_to_matrix(q)), q, atol=1e-12)

    def test_general_error_matches_level_error_and_is_zero_at_reference(self) -> None:
        q = axis_angle_quaternion((1, -2, 0.3), 0.4)
        yaw = 0.8
        np.testing.assert_allclose(
            quaternion_error(q, (math.cos(yaw / 2), 0, 0, math.sin(yaw / 2))),
            quaternion_error(q, tilted_reference_quaternion(np.array([0.0, 0.0, 1.0]), yaw)),
        )
        reference = tilted_reference_quaternion(normal_of(0.2, 0.1), yaw)
        np.testing.assert_allclose(quaternion_error(reference, reference), (0, 0, 0), atol=1e-12)


if __name__ == "__main__":
    unittest.main()
