import math
from dataclasses import replace
from pathlib import Path
import unittest

import mujoco
import numpy as np

from jumper_sim.lidar import LIVOX_MID360, SimulatedLidar

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

if __name__ == "__main__":
    unittest.main()
