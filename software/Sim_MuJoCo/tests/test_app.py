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
                self.cam = SimpleNamespace()

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
        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0].command[1:4], ["-m", "jumper_sim.cloud_viewer", str(scan_file)])
        self.assertTrue(windows[0].terminated)
        # The live point cloud window is started on the scan file and closed
        # again when the viewer exits.
        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0].command[1:4], ["-m", "jumper_sim.cloud_viewer", str(scan_file)])
        self.assertTrue(windows[0].terminated)

        self.assertEqual(result, 0)
        self.assertGreater(simulated_data[0].time, 0.0)
        self.assertEqual(
            viewer_cameras[0].type, mujoco.mjtCamera.mjCAMERA_TRACKING
        )
        self.assertEqual(
            viewer_cameras[0].trackbodyid, base_body_ids[0]
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
            "[SCAN] Target point cloud",
            "[SCAN] Point cloud window opened",
            "[SCAN] Point cloud window opened",
        ):
            self.assertIn(expected, messages[0])


if __name__ == "__main__":
    unittest.main()
