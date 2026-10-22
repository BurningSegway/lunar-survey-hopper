import math
import unittest

import numpy as np

from jumper_sim.controller import quaternion_attitude_error, quaternion_error
from jumper_sim.depth_camera import CameraMount, DepthCameraSpec, DepthFrame
from jumper_sim.landing_reference import (
    LandingReference,
    LandingReferenceConfig,
    PlaneEstimate,
    estimate_ground_plane,
    matrix_to_quaternion,
    quaternion_to_matrix,
    tilted_reference_quaternion,
)

MOUNT = CameraMount(
    position=(0.06, -0.06, -0.151),
    rotation=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
)


def axis_angle_quaternion(axis, angle):
    axis = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    return (math.cos(angle / 2), *(math.sin(angle / 2) * axis))


def plane_frame(slope_x, slope_y, height, body_quaternion, spec=DepthCameraSpec(noise=False),
                rock=None):
    """Exact depth image of the plane z = slope_x*x + slope_y*y - height.

    Coordinates are world-oriented and relative to the robot origin. ``rock``
    = (x, y, radius, bump) raises the ground inside a disc.
    """
    rays = spec.optical_rays()
    r_wb = quaternion_to_matrix(body_quaternion)
    origin = r_wb @ np.array(MOUNT.position)
    d = rays @ (r_wb @ MOUNT.body_from_optical).T
    t = (slope_x * origin[0] + slope_y * origin[1] - height - origin[2]) / (
        d[:, 2] - slope_x * d[:, 0] - slope_y * d[:, 1]
    )
    if rock is not None:
        rx, ry, radius, bump = rock
        hit = origin + d * t[:, None]
        inside = np.hypot(hit[:, 0] - rx, hit[:, 1] - ry) < radius
        t2 = (slope_x * origin[0] + slope_y * origin[1] - height + bump - origin[2]) / (
            d[:, 2] - slope_x * d[:, 0] - slope_y * d[:, 1]
        )
        hit2 = origin + d * t2[:, None]
        inside2 = np.hypot(hit2[:, 0] - rx, hit2[:, 1] - ry) < radius
        t = np.where(inside | inside2, np.minimum(t, t2), t)
    depth = t * rays[:, 2]
    depth[(t <= 0) | (depth < spec.min_range) | (depth > spec.max_range)] = np.nan
    return DepthFrame(depth.reshape(spec.height, spec.width), spec, 0.0, tuple(body_quaternion))


def normal_of(slope_x, slope_y):
    n = np.array([-slope_x, -slope_y, 1.0])
    return n / np.linalg.norm(n)


def angle_between(a, b):
    return math.degrees(math.acos(min(1.0, float(np.dot(a, b)))))


class PlaneFitTests(unittest.TestCase):
    def test_recovers_tilted_plane_from_tilted_robot(self) -> None:
        for slope in ((0.0, 0.0), (0.2, -0.1), (-0.3, 0.25)):
            for attitude in ((1.0, 0.0, 0.0, 0.0),
                             axis_angle_quaternion((1, 2, 0.5), math.radians(20))):
                frame = plane_frame(*slope, 2.0, attitude)
                estimate = estimate_ground_plane(frame, MOUNT)
                self.assertTrue(estimate.valid)
                self.assertLess(angle_between(estimate.normal, normal_of(*slope)), 0.05)
                self.assertAlmostEqual(estimate.height, 2.0, places=2)

    def test_robot_yaw_does_not_change_world_normal(self) -> None:
        slope = (0.15, 0.05)
        normals = [
            estimate_ground_plane(
                plane_frame(*slope, 2.0, axis_angle_quaternion((0, 0, 1), yaw)), MOUNT
            ).normal
            for yaw in (0.0, 1.0, 2.5)
        ]
        for n in normals:
            self.assertLess(angle_between(n, normal_of(*slope)), 0.05)

    def test_rock_is_rejected_as_outlier(self) -> None:
        clean = estimate_ground_plane(plane_frame(0.1, 0.0, 1.5, (1, 0, 0, 0)), MOUNT)
        rocky = estimate_ground_plane(
            plane_frame(0.1, 0.0, 1.5, (1, 0, 0, 0), rock=(0.25, 0.1, 0.12, 0.15)), MOUNT
        )
        self.assertTrue(rocky.valid)
        self.assertLess(angle_between(rocky.normal, clean.normal), 0.3)

    def test_out_of_range_ground_is_invalid(self) -> None:
        too_close = estimate_ground_plane(plane_frame(0.0, 0.0, 0.3, (1, 0, 0, 0)), MOUNT)
        too_far = estimate_ground_plane(plane_frame(0.0, 0.0, 30.0, (1, 0, 0, 0)), MOUNT)
        self.assertFalse(too_close.valid)
        self.assertFalse(too_far.valid)

    def test_uncertainty_grows_with_noise_and_altitude(self) -> None:
        noisy = DepthCameraSpec()
        sigmas = []
        for height in (2.0, 6.0):
            frame = plane_frame(0.1, 0.0, height, (1, 0, 0, 0))
            rng = np.random.default_rng(0)
            depth = frame.depth + rng.standard_normal(frame.depth.shape) * noisy.depth_sigma(frame.depth)
            frame = DepthFrame(depth, noisy, 0.0, frame.body_quaternion)
            sigmas.append(estimate_ground_plane(frame, MOUNT).normal_sigma)
        self.assertGreater(sigmas[1], 2.0 * sigmas[0])


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

    def test_reference_freezes_close_to_touchdown(self) -> None:
        config = LandingReferenceConfig()
        reference = LandingReference(config)
        target = normal_of(0.15, 0.0)
        # Descending at 4 m/s: fits from 3.0 m and 2.6 m are used ...
        for t, h in ((0.0, 3.0), (0.1, 2.6)):
            n = target / np.linalg.norm(target)
            self.assertTrue(reference.update(PlaneEstimate(True, tuple(n), math.radians(0.3), height=h, timestamp=t)))
        held = reference.normal
        # ... but one 0.25 s from impact (1.0 m at 4 m/s) and one below
        # freeze_height are not.
        other = normal_of(-0.1, 0.1)
        for t, h in ((0.25, 1.0), (0.35, 0.6)):
            self.assertFalse(reference.update(PlaneEstimate(True, tuple(other), math.radians(0.3), height=h, timestamp=t)))
        self.assertEqual(reference.frozen, 2)
        np.testing.assert_allclose(reference.normal, held)
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
            quaternion_attitude_error(q, yaw),
        )
        reference = tilted_reference_quaternion(normal_of(0.2, 0.1), yaw)
        np.testing.assert_allclose(quaternion_error(reference, reference), (0, 0, 0), atol=1e-12)


if __name__ == "__main__":
    unittest.main()
