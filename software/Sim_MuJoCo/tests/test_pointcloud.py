from pathlib import Path
import tempfile
import unittest

import numpy as np

from jumper_sim.pointcloud import (
    PREVIEW_POINTS, PointCloudRecorder, read_ply, read_touchdowns, touchdown_path, write_ply,
)


def batch(count=48, offset=(0.0, 0.0, 0.0)):
    """A lidar batch: world points and their ranges."""
    points = np.arange(3 * count, dtype=float).reshape(count, 3) * 0.01 + offset
    return points, np.linalg.norm(points, axis=1)


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
        points, ranges = batch()
        recorder.add_points(points, ranges, 0.5)
        recorder.save()
        header = self.path.read_bytes().split(b"end_header\n")[0].decode()
        self.assertIn("format binary_little_endian 1.0", header)
        self.assertIn("element vertex 48", header)
        cloud = read_ply(self.path)
        np.testing.assert_array_equal(cloud, recorder.points())
        np.testing.assert_allclose(np.column_stack([cloud["x"], cloud["y"], cloud["z"]]), points,
                                   atol=1e-6)
        np.testing.assert_allclose(cloud["range"], ranges, rtol=1e-6)
        np.testing.assert_allclose(cloud["t"], 0.5)

    def test_points_only_count_during_a_jump_and_are_tagged_with_it(self) -> None:
        recorder = PointCloudRecorder(self.path)
        self.assertEqual(recorder.add_points(*batch(), 0.0), 0)  # no jump yet
        recorder.start_jump()
        self.assertEqual(recorder.add_points(*batch(), 0.0), 48)
        recorder.start_jump()
        self.assertEqual(recorder.add_points(*batch(10), 1.0), 10)
        self.assertEqual(recorder.add_points(np.zeros((0, 3)), np.zeros(0), 1.0), 0)
        self.assertEqual(np.bincount(recorder.points()["jump"]).tolist(), [0, 48, 10])

    def test_preview_is_an_even_sample_refreshed_at_each_save(self) -> None:
        recorder = PointCloudRecorder(self.path)
        recorder.start_jump()
        recorder.add_points(*batch(10 * PREVIEW_POINTS), 0.0)
        self.assertEqual(len(recorder.preview), 0)
        recorder.save()
        self.assertEqual(len(recorder.preview), PREVIEW_POINTS)

    def test_touchdowns_are_written_next_to_the_cloud(self) -> None:
        recorder = PointCloudRecorder(self.path)
        recorder.start_run()
        self.assertEqual(read_touchdowns(self.path), [])
        recorder.add_touchdown((1.0, 2.0, 3.0))  # before any jump: ignored
        recorder.start_jump()
        recorder.add_touchdown((4.0, 5.0, 6.0))
        recorder.start_jump()
        recorder.add_touchdown((7.0, 8.0, 9.0))
        self.assertEqual(touchdown_path(self.path).name, "scan.touchdowns.json")
        touchdowns = read_touchdowns(self.path)
        self.assertEqual([jump for jump, _ in touchdowns], [1, 2])
        np.testing.assert_allclose(touchdowns[1][1], (7.0, 8.0, 9.0))
        PointCloudRecorder(self.path).start_run()  # next program run
        self.assertEqual(read_touchdowns(self.path), [])

    def test_start_run_overwrites_earlier_cloud(self) -> None:
        first = PointCloudRecorder(self.path)
        first.start_jump()
        first.add_points(*batch(), 0.0)
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
