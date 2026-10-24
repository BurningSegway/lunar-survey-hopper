import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from jumper_sim.depth_camera import CameraMount, DepthCameraSpec, DepthFrame
from jumper_sim.pointcloud import PointCloudRecorder, ScanConfig, read_ply, write_ply

LEVEL = (1.0, 0.0, 0.0, 0.0)
# Camera at the body origin with the site unrotated, as on the robot: the
# optical axis points straight down (body -z).
DOWN = CameraMount(position=(0.0, 0.0, 0.0),
                   rotation=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))


def flat_frame(depth=2.0, t=0.5, position=(1.0, 2.0, 3.0)):
    spec = DepthCameraSpec(width=8, height=6)
    return DepthFrame(np.full((6, 8), depth), spec, t, LEVEL, position)


class PointCloudTests(unittest.TestCase):
    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        self.path = Path(self._dir.name) / "nested" / "scan.ply"

    def tearDown(self) -> None:
        self._dir.cleanup()

    def test_ply_round_trip_and_header(self) -> None:
        recorder = PointCloudRecorder(self.path)
        recorder.start_run()
        recorder.start_jump()
        recorder.add_frame(flat_frame(), DOWN, 0.0)
        recorder.save()
        header = self.path.read_bytes().split(b"end_header\n")[0].decode()
        self.assertIn("format binary_little_endian 1.0", header)
        self.assertIn("element vertex 48", header)
        cloud = read_ply(self.path)
        np.testing.assert_array_equal(cloud, recorder.points())

    def test_points_are_placed_in_the_world(self) -> None:
        recorder = PointCloudRecorder(self.path)
        recorder.start_jump()
        recorder.add_frame(flat_frame(depth=2.0, position=(1.0, 2.0, 3.0)), DOWN, 0.0)
        cloud = recorder.points()
        # Optical axis points down: every point is 2 m below the camera.
        np.testing.assert_allclose(cloud["z"], 1.0, atol=1e-6)
        self.assertAlmostEqual(float(np.mean(cloud["x"])), 1.0, places=5)
        self.assertAlmostEqual(float(np.mean(cloud["y"])), 2.0, places=5)
        # Explicit position overrides the frame's own.
        recorder.add_frame(flat_frame(), DOWN, 0.0, robot_position=np.zeros(3))
        np.testing.assert_allclose(recorder.points()["z"][-48:], -2.0, atol=1e-6)

    def test_margin_gate_and_jump_tags(self) -> None:
        recorder = PointCloudRecorder(self.path, ScanConfig(margin=math.radians(5)))
        self.assertEqual(recorder.add_frame(flat_frame(), DOWN, 0.0), 0)  # no jump yet
        recorder.start_jump()
        self.assertEqual(recorder.add_frame(flat_frame(), DOWN, math.radians(6)), 0)
        self.assertEqual(recorder.add_frame(flat_frame(), DOWN, math.radians(4)), 48)
        recorder.start_jump()
        recorder.add_frame(flat_frame(t=1.5), DOWN, 0.0)
        cloud = recorder.points()
        self.assertEqual(sorted(set(cloud["jump"].tolist())), [1, 2])
        self.assertEqual(set(cloud["t"][cloud["jump"] == 2].tolist()), {1.5})

    def test_start_run_overwrites_earlier_cloud(self) -> None:
        first = PointCloudRecorder(self.path)
        first.start_jump()
        first.add_frame(flat_frame(), DOWN, 0.0)
        first.save()
        self.assertEqual(len(read_ply(self.path)), 48)
        PointCloudRecorder(self.path).start_run()  # next program run
        self.assertEqual(len(read_ply(self.path)), 0)
        self.assertFalse(self.path.with_suffix(".ply.tmp").exists())

    def test_empty_cloud_is_valid(self) -> None:
        write_ply(self.path, np.empty(0, PointCloudRecorder(self.path).points().dtype))
        self.assertEqual(len(read_ply(self.path)), 0)


if __name__ == "__main__":
    unittest.main()
