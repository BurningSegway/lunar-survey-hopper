from pathlib import Path
import tempfile
import unittest

import numpy as np

from jumper_sim.pointcloud import (
    PREVIEW_POINTS, PointCloudRecorder, arcs_path, path_errors, read_arcs, read_ply, read_touchdowns, touchdown_path, write_ply,
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

    def test_path_errors_at_the_apex_and_at_touchdown(self) -> None:
        t = np.linspace(0.0, 2.0, 21)
        predicted = np.column_stack([t, 0 * t, 2 * t - t * t])  # apex at t = 1
        actual = predicted + (0.0, 0.0, 0.01)
        actual[-1] += (0.03, 0.04, -0.01)
        errors = path_errors(predicted, actual)
        self.assertEqual([where for where, _, _ in errors], ["apex", "touchdown"])
        np.testing.assert_allclose(errors[0][1], actual[10])
        self.assertAlmostEqual(errors[0][2], 0.01)
        self.assertAlmostEqual(errors[1][2], 0.05)
        # Latched on the way down: no apex; too short: nothing.
        self.assertEqual([w for w, _, _ in path_errors(predicted[12:], actual[12:])], ["touchdown"])
        self.assertEqual(path_errors(predicted[:1], actual[:1]), [])

    def test_predicted_paths_are_written_next_to_the_cloud(self) -> None:
        recorder = PointCloudRecorder(self.path)
        recorder.start_run()
        self.assertEqual(read_arcs(self.path), [])
        recorder.add_arc([[0.0, 0.0, 0.0], [1.0, 0.0, 1.0]])  # before any jump: ignored
        recorder.start_jump()
        recorder.add_arc([[0.0, 0.0, 0.0], [1.0, 0.0, 1.0], [2.0, 0.0, 0.0]],
                         [[0.0, 0.0, 0.0], [1.0, 0.0, 0.9]])  # with the actual path
        recorder.add_arc([[2.0, 0.0, 0.0], [2.5, 0.0, 0.2]])  # a bounce, no actual path
        self.assertEqual(arcs_path(self.path).name, "scan.arcs.json")
        arcs = read_arcs(self.path)
        self.assertEqual([jump for jump, _, _ in arcs], [1, 1])
        np.testing.assert_allclose(arcs[0][1][2], (2.0, 0.0, 0.0))
        np.testing.assert_allclose(arcs[0][2][1], (1.0, 0.0, 0.9))
        self.assertEqual(arcs[1][1].shape, (2, 3))
        self.assertEqual(arcs[1][2].shape, (0, 3))
        PointCloudRecorder(self.path).start_run()
        self.assertEqual(read_arcs(self.path), [])

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
