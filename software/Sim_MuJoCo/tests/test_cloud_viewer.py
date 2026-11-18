import os
from pathlib import Path
import tempfile
import time
import unittest

import numpy as np

from jumper_sim.cloud_viewer import ARC_SPECTRUM, POINT_SPECTRUM, CloudFile, error_text, jump_colours
from jumper_sim.pointcloud import PLY_DTYPE, arcs_path, touchdown_path, write_ply


def cloud(count, jumps=1):
    points = np.zeros(count, PLY_DTYPE)
    points["x"] = np.linspace(0.0, 2.0, count)
    points["y"] = np.linspace(-1.0, 1.0, count)
    points["jump"] = 1 + np.arange(count) % jumps
    return points


class CloudFileTests(unittest.TestCase):
    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        self.path = Path(self._dir.name) / "scan.ply"

    def tearDown(self) -> None:
        self._dir.cleanup()

    def _rewrite(self, points) -> None:
        write_ply(self.path, points)
        stamp = time.time_ns() + 10_000_000  # make sure the mtime moves on
        os.utime(self.path, ns=(stamp, stamp))

    def test_reloads_only_when_the_file_changes(self) -> None:
        source = CloudFile(self.path)
        self.assertFalse(source.poll())  # no file yet: nothing to do
        self._rewrite(cloud(300))
        self.assertTrue(source.poll())
        self.assertEqual(source.points.shape, (300, 3))
        self.assertEqual(source.jumps, 1)
        self.assertFalse(source.poll())  # unchanged
        self._rewrite(cloud(600, jumps=2))  # the next jump
        self.assertTrue(source.poll())
        self.assertEqual(len(source.points), 600)
        self.assertEqual(source.jumps, 2)
        # Jump 1 blue, the latest jump (2) green: the colours shift as jumps come.
        np.testing.assert_allclose(source.colours[0], POINT_SPECTRUM[0])
        np.testing.assert_allclose(source.colours[1], POINT_SPECTRUM[1])

    def test_emptied_file_clears_the_cloud(self) -> None:
        self._rewrite(cloud(100))
        source = CloudFile(self.path)
        source.poll()
        self._rewrite(cloud(0))  # a new simulator run starts a fresh cloud
        self.assertTrue(source.poll())
        self.assertEqual(len(source.points), 0)
        self.assertEqual(source.jumps, 0)

    def test_unreadable_file_keeps_the_last_cloud(self) -> None:
        self._rewrite(cloud(50))
        source = CloudFile(self.path)
        source.poll()
        self.path.write_bytes(b"not a ply")
        self.assertFalse(source.poll())
        self.assertEqual(len(source.points), 50)

    def test_error_labels(self) -> None:
        self.assertEqual(error_text(0.0123), "1.2 cm")
        self.assertEqual(error_text(1.234), "1.23 m")

    def test_jump_colours_run_along_a_spectrum_to_the_latest_jump(self) -> None:
        np.testing.assert_allclose(jump_colours([1], 1), [POINT_SPECTRUM[0]])
        colours = jump_colours([1, 2, 3], 3)
        np.testing.assert_allclose(colours[[0, 2]], POINT_SPECTRUM)
        np.testing.assert_allclose(colours[1], np.mean(POINT_SPECTRUM, axis=0))
        np.testing.assert_allclose(jump_colours([1, 5], 5, ARC_SPECTRUM), ARC_SPECTRUM)


class HighlightAndTouchdownTests(unittest.TestCase):
    def test_points_near_the_target_and_touchdowns(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "scan.ply"
            write_ply(path, cloud(300))  # x 0..2, y -1..1, z 0
            source = CloudFile(path, target=(1.0, 0.0, 0.0))
            self.assertTrue(source.poll())
            near_target = np.linalg.norm(source.points - (1.0, 0.0, 0.0), axis=1) <= 0.3
            np.testing.assert_array_equal(source.highlight, near_target)
            self.assertTrue(source.highlight.any() and not source.highlight.all())
            self.assertEqual(source.touchdowns, [])
            touchdown_path(path).write_text('[{"jump": 1, "position": [1, 2, 3]}]')
            self.assertTrue(source.poll())  # only the touchdowns changed
            self.assertEqual(source.touchdowns[0][0], 1)
            np.testing.assert_allclose(source.touchdowns[0][1], (1.0, 2.0, 3.0))
            self.assertFalse(source.poll())
            self.assertEqual(source.arcs, [])
            arcs_path(path).write_text(
                '[{"jump": 2, "points": [[0, 0, 0], [1, 0, 1]], "actual": [[0, 0, 0], [1, 0, 0.9]]}]')
            self.assertTrue(source.poll())  # only the predicted paths changed
            self.assertEqual(source.arcs[0][0], 2)
            # Paths: orange -> red, on the latest jump of either file (here 2).
            np.testing.assert_allclose(source.arc_colours, [ARC_SPECTRUM[1]])
            np.testing.assert_allclose(source.arcs[0][1], [[0, 0, 0], [1, 0, 1]])
            np.testing.assert_allclose(source.arcs[0][2], [[0, 0, 0], [1, 0, 0.9]])
            self.assertFalse(source.poll())
            # Without a target nothing is highlighted.
            plain = CloudFile(path)
            plain.poll()
            self.assertFalse(plain.highlight.any())


if __name__ == "__main__":
    unittest.main()
