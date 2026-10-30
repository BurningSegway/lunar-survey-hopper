from collections.abc import Callable, Generator
from contextlib import contextmanager, nullcontext
import io
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import glfw
import mujoco
import numpy as np

from jumper_sim import app
from jumper_sim.pointcloud import read_ply
from jumper_sim.simulation import JumperSimulation, ReactionWheelTuning
from jumper_sim.telemetry import TelemetryPayload

MODEL_PATH = Path(__file__).resolve().parents[1] / "build" / "Jumping_Robot.xml"
TERRAIN_ASSETS = (
    MODEL_PATH.parent / "terrain.obj",
    MODEL_PATH.parent / "terrain_height.bin",
)


@unittest.skipUnless(
    MODEL_PATH.is_file() and all(path.is_file() for path in TERRAIN_ASSETS),
    "Generate build/terrain.obj and terrain_height.bin from "
    "build/convert_usd.py before running app tests.",
)
class ViewerApplicationTests(unittest.TestCase):
    def test_gravity_cannot_be_selected_at_launch(self) -> None:
        with (
            patch("sys.stderr", new_callable=io.StringIO),
            self.assertRaises(SystemExit),
        ):
            app.build_argument_parser().parse_args(["--gravity", "moon"])

    def test_track_target_takes_x_y_and_excludes_no_target(self) -> None:
        parser = app.build_argument_parser()
        self.assertEqual(parser.parse_args(["--track-target", "4", "1"]).track_target, [4.0, 1.0])
        self.assertTrue(parser.parse_args(["--no-target"]).no_target)
        with patch("sys.stderr", new_callable=io.StringIO), self.assertRaises(SystemExit):
            parser.parse_args(["--track-target", "4", "1", "--no-target"])

    def test_random_target_is_within_reach_of_the_spawn(self) -> None:
        rng = np.random.default_rng(3)
        points = np.array([app.random_target_near((5.0, -2.0), 20.0, 2.0, rng) for _ in range(2000)])
        distances = np.hypot(points[:, 0] - 5.0, points[:, 1] + 2.0)
        self.assertGreaterEqual(distances.min(), 2.0)
        self.assertLessEqual(distances.max(), 20.0)
        # Spread over the whole disc and in every direction, not bunched up.
        self.assertGreater(distances.max(), 19.0)
        self.assertLess(distances.min(), 3.0)
        self.assertGreater(float(np.mean(distances > 10.0)), 0.6)  # area: (400-100)/(400-4)
        headings = np.arctan2(points[:, 1] + 2.0, points[:, 0] - 5.0)
        self.assertEqual(len(np.unique(np.floor((headings + np.pi) / (np.pi / 2)))), 4)

    def test_spawn_position_is_the_root_joint_start(self) -> None:
        model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        data = mujoco.MjData(model)
        self.assertEqual(app.spawn_position(model, data), (0.0, 0.0))

    def test_keyboard_callback_drives_simulation_and_telemetry(self) -> None:
        messages: list[str] = []
        telemetry_payloads: list[TelemetryPayload] = []
        simulated_data: list[mujoco.MjData] = []
        viewer_cameras: list[SimpleNamespace] = []
        base_body_ids: list[int] = []
        backspace_before: list[
            tuple[np.ndarray, np.ndarray, np.ndarray, float]
        ] = []
        backspace_after: list[
            tuple[np.ndarray, np.ndarray, np.ndarray, float]
        ] = []

        class FakeTelemetry:
            def __init__(self) -> None:
                pass

            def send(self, payload: TelemetryPayload) -> None:
                telemetry_payloads.append(payload)

            def close(self) -> None:
                pass

        class FakeViewer:
            def __init__(
                self,
                model: mujoco.MjModel,
                data: mujoco.MjData,
                key_callback: Callable[[int], None],
            ) -> None:
                self._iterations = 0
                self._model = model
                self._data = data
                self._key_callback = key_callback
                self.cam = mujoco.MjvCamera()

            def is_running(self) -> bool:
                self._iterations += 1
                if self._iterations == 2:
                    backspace_before.append(
                        (
                            self._data.qpos.copy(),
                            self._data.qvel.copy(),
                            self._data.ctrl.copy(),
                            float(self._data.time),
                        )
                    )
                    mujoco.mj_resetData(self._model, self._data)
                    for key in (
                        glfw.KEY_BACKSPACE,
                        glfw.KEY_SPACE,
                        glfw.KEY_C,
                        glfw.KEY_X,
                        glfw.KEY_C,
                        glfw.KEY_R,
                    ):
                        self._key_callback(key)
                return self._iterations <= 2

            def lock(self):
                return nullcontext()

            def sync(self) -> None:
                return None

        @contextmanager
        def launch_passive(
            _model: mujoco.MjModel,
            _data: mujoco.MjData,
            *,
            key_callback: Callable[[int], None],
        ) -> Generator[FakeViewer, None, None]:
            simulated_data.append(_data)
            self.assertGreater(_model.nbody, 0)
            base_body_ids.append(
                mujoco.mj_name2id(
                    _model, mujoco.mjtObj.mjOBJ_BODY, "base_link"
                )
            )
            viewer = FakeViewer(_model, _data, key_callback)
            viewer_cameras.append(viewer.cam)
            yield viewer

        original_handle_key = JumperSimulation.handle_key

        def record_state_on_backspace(
            simulation: JumperSimulation, key: str
        ) -> str | None:
            if key == "backspace":
                backspace_after.append(
                    (
                        simulation.data.qpos.copy(),
                        simulation.data.qvel.copy(),
                        simulation.data.ctrl.copy(),
                        float(simulation.data.time),
                    )
                )
            return original_handle_key(simulation, key)

        windows = []

        class FakeWindow:
            def __init__(self, command, **kwargs) -> None:
                self.command = command
                self.terminated = False
                windows.append(self)

            def poll(self):
                return None

            def terminate(self) -> None:
                self.terminated = True

        with (
            patch.object(app.mujoco.viewer, "launch_passive", launch_passive),
            patch.object(app, "UdpTelemetry", FakeTelemetry),
            patch.object(JumperSimulation, "handle_key", record_state_on_backspace),
            patch.object(app.time, "sleep"),
            patch("sys.stdout", new_callable=io.StringIO) as stdout,
            tempfile.TemporaryDirectory() as scan_dir,
            patch.object(app.subprocess, "Popen", FakeWindow),
        ):
            scan_file = Path(scan_dir) / "scan.ply"
            scan_file.write_bytes(b"points from an earlier run")
            result = app.main(["--controller", "lqr", "--scan-file", str(scan_file)])
            messages.append(stdout.getvalue())
            # A new run starts a fresh (empty) cloud in the scan file.
            self.assertEqual(len(read_ply(scan_file)), 0)
        # The live point cloud window is started on the scan file and closed
        # again when the viewer exits.
        # (Warp's first init also runs `uname -p` through Popen.)
        windows = [w for w in windows if "jumper_sim.cloud_viewer" in w.command]
        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0].command[1:4], ["-m", "jumper_sim.cloud_viewer", str(scan_file)])
        self.assertTrue(windows[0].terminated)
        # Without --track-target or --no-target a random target is chosen and
        # drawn in the point cloud window.
        self.assertEqual(windows[0].command[4], "--target")
        x, y, _ = (float(v) for v in windows[0].command[5:8])
        self.assertLessEqual(np.hypot(x, y), app.RANDOM_TARGET_RADIUS)

        self.assertEqual(result, 0)
        self.assertGreater(simulated_data[0].time, 0.0)
        # The robot is followed with a free camera centred on it (MuJoCo's
        # tracking camera lags and stutters with a fast robot).
        self.assertEqual(viewer_cameras[0].type, mujoco.mjtCamera.mjCAMERA_FREE)
        np.testing.assert_allclose(
            viewer_cameras[0].lookat, simulated_data[0].xpos[base_body_ids[0]]
        )
        self.assertEqual(viewer_cameras[0].distance, 3.0)
        self.assertEqual(viewer_cameras[0].azimuth, 135.0)
        self.assertEqual(viewer_cameras[0].elevation, -20.0)
        self.assertEqual(len(backspace_before), 1)
        self.assertEqual(len(backspace_after), 1)
        before = backspace_before[0]
        after = backspace_after[0]
        np.testing.assert_array_equal(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        np.testing.assert_array_equal(before[2], after[2])
        self.assertEqual(before[3], after[3])
        self.assertEqual(len(telemetry_payloads), 2)
        self.assertAlmostEqual(
            telemetry_payloads[0]["sat_max"],
            ReactionWheelTuning().max_wheel_velocity,
        )
        for expected in (
            "[CONTROLLER] Selected: lqr",
            "[GRAVITY] Selected: moon",
            "[JUMP INITIATED]",
            "[GRAVITY] Switched to Earth",
            "[CONTROLLER] Disabled",
            "[CONTROLLER] Attitude stabilization enabled",
            "[RESET] Simulation reset",
            "[LIDAR] Livox Mid-360S",
            "[SCAN] Lidar point cloud",
            "[TARGET] Random target",
            "[SCAN] Point cloud window opened",
        ):
            self.assertIn(expected, messages[0])


class LidarOptionTests(unittest.TestCase):
    def test_lidar_fraction_defaults_to_the_hardware(self) -> None:
        parser = app.build_argument_parser()
        self.assertIsNone(parser.parse_args([]).lidar_fraction)  # 1 with a GPU, 0.1 without
        self.assertEqual(parser.parse_args(["--lidar-fraction", "0.05"]).lidar_fraction, 0.05)

    def test_lidar_defaults_to_the_mid360s_with_the_as_dt1_as_an_option(self) -> None:
        parser = app.build_argument_parser()
        self.assertEqual(parser.parse_args([]).lidar, "mid360s")
        self.assertEqual(parser.parse_args(["--lidar", "as-dt1"]).lidar, "as-dt1")
        # --spin works with either lidar.
        args = parser.parse_args(["--spin"])
        self.assertEqual((args.lidar, args.spin), ("mid360s", True))
        args = parser.parse_args(["--lidar", "as-dt1", "--spin"])
        self.assertEqual((args.lidar, args.spin), ("as-dt1", True))
        with self.assertRaises(SystemExit), patch("sys.stderr", io.StringIO()):
            parser.parse_args(["--lidar", "avia"])


class RobotFollowerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.viewer = SimpleNamespace(cam=mujoco.MjvCamera())
        self.data = SimpleNamespace(xpos=np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]]))
        self.follower = app.RobotFollower(1)

    def test_camera_centre_moves_with_the_robot_without_lag(self) -> None:
        self.follower.start(self.viewer, self.data)
        self.assertEqual(self.viewer.cam.type, mujoco.mjtCamera.mjCAMERA_FREE)
        np.testing.assert_allclose(self.viewer.cam.lookat, (1.0, 2.0, 3.0))
        self.data.xpos[1] = (1.0, 2.0, 9.0)  # a fast climb
        self.viewer.cam.azimuth = 40.0  # the user rotates: still following
        self.follower.update(self.viewer, self.data)
        np.testing.assert_allclose(self.viewer.cam.lookat, (1.0, 2.0, 9.0))

    def test_panning_hands_the_camera_to_the_user_until_restarted(self) -> None:
        self.follower.start(self.viewer, self.data)
        self.viewer.cam.lookat[:] = (5.0, 5.0, 0.0)  # user pans
        self.data.xpos[1] = (1.0, 2.0, 9.0)
        self.follower.update(self.viewer, self.data)
        np.testing.assert_allclose(self.viewer.cam.lookat, (5.0, 5.0, 0.0))
        self.assertFalse(self.follower.active)
        self.follower.start(self.viewer, self.data)  # R
        np.testing.assert_allclose(self.viewer.cam.lookat, (1.0, 2.0, 9.0))

    def test_another_camera_stops_following(self) -> None:
        self.follower.start(self.viewer, self.data)
        self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self.follower.update(self.viewer, self.data)
        self.assertFalse(self.follower.active)


if __name__ == "__main__":
    unittest.main()
