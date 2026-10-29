import math
from dataclasses import replace
from pathlib import Path
import unittest

import mujoco
import numpy as np

from jumper_sim.lidar import LIVOX_MID360, SONY_AS_DT1, SimulatedLidar, effective_range_sigma

MODEL_PATH = Path(__file__).resolve().parents[1] / "build" / "Jumping_Robot.xml"


class FlatGround:
    """Ray caster for the plane z = 0; counts the rays cast."""

    def __init__(self) -> None:
        self.rays = 0

    def cast(self, origin, directions, max_range):
        self.rays += len(directions)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = -origin[2] / directions[:, 2]
        return np.where((t > 0) & (t <= max_range), t, np.nan)


class LidarPatternTests(unittest.TestCase):
    def test_directions_stay_in_the_field_of_view(self) -> None:
        # The recorded pattern sweeps its elevation band in ~50 ms: take 0.25 s
        # of points, and the R2 fill used without the pattern file.
        for spec in (LIVOX_MID360, replace(LIVOX_MID360, pattern=None)):
            rays = spec.directions(0, 50000)
            np.testing.assert_allclose(np.linalg.norm(rays, axis=1), 1.0)
            off_axis = np.degrees(np.arccos(np.clip(rays[:, 2], -1, 1)))
            # Blind within 38 deg of its axis, sees past 90; the recorded
            # pattern reaches ~0.2 deg past the datasheet field of view.
            self.assertAlmostEqual(off_axis.min(), 38.0, delta=0.5)
            self.assertAlmostEqual(off_axis.max(), 97.0, delta=0.5)

    def test_recorded_pattern_plays_at_the_real_rate_with_a_fraction(self) -> None:
        tenth = replace(LIVOX_MID360, point_fraction=0.1)
        # Every 10th recorded direction: the pattern still advances in real time.
        np.testing.assert_allclose(tenth.directions(5, 3), LIVOX_MID360.directions(50, 21)[::10])

    def test_as_dt1_measures_the_same_24_by_24_grid_every_frame(self) -> None:
        frame = SONY_AS_DT1.directions(0, 576)
        np.testing.assert_allclose(SONY_AS_DT1.directions(576, 576), frame)
        np.testing.assert_allclose(np.linalg.norm(frame, axis=1), 1.0)
        self.assertEqual(len(np.unique(frame.round(9), axis=0)), 576)
        # A 30 x 30 deg square around the axis: outer pixel centres half a
        # pixel inside its edges.
        for angle in (np.degrees(np.arctan2(frame[:, 0], frame[:, 2])),
                      np.degrees(np.arctan2(frame[:, 1], frame[:, 2]))):
            self.assertAlmostEqual(angle.max(), -angle.min())
            self.assertLess(angle.max(), 15.0)
            self.assertGreater(angle.max(), 14.0)
        self.assertEqual(SONY_AS_DT1.point_rate * SONY_AS_DT1.batch_period, 576)  # one frame



@unittest.skipUnless(MODEL_PATH.is_file(), "needs the robot model")
class SimulatedLidarTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        self.data = mujoco.MjData(self.model)
        self.data.qpos[:3] = (0.0, 0.0, 3.0)
        self.data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
        mujoco.mj_forward(self.model, self.data)

    def test_points_land_on_the_ground_at_the_rate_asked_for(self) -> None:
        exact = replace(LIVOX_MID360, range_sigma=0.0, angular_sigma=0.0, close_sigma=None,
                        point_fraction=0.5)
        ground = FlatGround()
        lidar = SimulatedLidar(self.model, exact, ground)
        returns = lidar.scan(self.data, 0.05)
        self.assertEqual(ground.rays, 5000)  # 200k/s x 0.5 x 50 ms
        np.testing.assert_allclose(returns.points[:, 2], 0.0, atol=1e-9)
        np.testing.assert_allclose(
            np.linalg.norm(returns.points - returns.origin, axis=1), returns.ranges
        )

    def test_side_mount_sees_straight_below_and_is_blind_along_its_axis(self) -> None:
        exact = replace(LIVOX_MID360, range_sigma=0.0, angular_sigma=0.0, point_fraction=0.5)
        returns = SimulatedLidar(self.model, exact, FlatGround()).scan(self.data, 0.1)
        height = returns.origin[2]
        offset = returns.points - returns.origin
        # Ground straight below the sensor (and the robot) is seen ...
        horizontal = np.linalg.norm(offset[:, :2], axis=1)
        self.assertLess(horizontal.min(), 0.05 * height)
        # ... the blind cone is around the axis: out of the +x face, 22.5 deg down.
        site = self.model.site("lidar").id
        axis = -self.data.site_xmat[site].reshape(3, 3)[:, 2]
        np.testing.assert_allclose(axis, (math.cos(math.radians(22.5)), 0.0,
                                          -math.sin(math.radians(22.5))), atol=1e-9)
        off_axis = np.degrees(np.arccos(offset @ axis / np.linalg.norm(offset, axis=1)))
        self.assertGreater(off_axis.min(), 37.7)
        self.assertLess(len(returns.ranges), 10000)  # rays above the horizon miss

    def test_range_noise_is_two_centimetres_and_four_within_one_metre(self) -> None:
        self.data.qpos[2] -= 2.4  # sensor ~0.3 m up: the inner ring is within 1 m
        mujoco.mj_forward(self.model, self.data)
        noisy = replace(LIVOX_MID360, angular_sigma=0.0, point_fraction=2.0)
        returns = SimulatedLidar(self.model, noisy, FlatGround()).scan(self.data, 0.05)
        # Exact range along each (exact) direction down to the flat ground.
        cos_down = (returns.origin[2] - returns.points[:, 2]) / returns.ranges
        exact_range = returns.origin[2] / cos_down
        error = returns.ranges - exact_range
        close, far = exact_range < 0.95, exact_range > 1.05
        self.assertGreater(close.sum(), 1000)
        self.assertAlmostEqual(float(np.std(error[close])), 0.04, delta=0.005)
        self.assertAlmostEqual(float(np.std(error[far])), 0.02, delta=0.005)

    def test_as_dt1_looks_down_from_the_bottom_with_rounded_ranges(self) -> None:
        returns = SimulatedLidar(self.model, SONY_AS_DT1, FlatGround()).scan(
            self.data, SONY_AS_DT1.batch_period)
        self.assertEqual(len(returns.ranges), 576)  # one frame, all on the ground
        site = self.model.site("lidar_down").id
        np.testing.assert_allclose(returns.origin, self.data.site_xpos[site])
        # Its axis is tilted 10 deg from body -z towards +x; body -z (what the
        # robot points at the target) stays 5 deg inside the grid.
        axis = -self.data.site_xmat[site].reshape(3, 3)[:, 2]
        np.testing.assert_allclose(axis, (math.sin(math.radians(10.0)), 0.0,
                                          -math.cos(math.radians(10.0))), atol=1e-9)
        height = returns.origin[2]
        offset = returns.points[:, :2] - returns.origin[:2]
        across = np.degrees(np.arctan2(offset[:, 0], height))
        self.assertAlmostEqual(float(across.min()), 10.0 - 14.4, delta=0.3)
        self.assertAlmostEqual(float(across.max()), 10.0 + 14.4, delta=0.3)
        # Reported in 25 mm steps; noise 5 mm plus that rounding.
        np.testing.assert_allclose(returns.ranges / 0.025, np.round(returns.ranges / 0.025),
                                   atol=1e-9)
        self.assertAlmostEqual(effective_range_sigma(SONY_AS_DT1), 0.0087, delta=0.0002)


if __name__ == "__main__":
    unittest.main()
