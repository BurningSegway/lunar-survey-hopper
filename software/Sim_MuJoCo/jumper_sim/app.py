from __future__ import annotations

import argparse
from collections.abc import Sequence
from contextlib import closing
import math
from pathlib import Path
import queue
import subprocess
import sys
import time

import glfw
import mujoco
import mujoco.viewer
import numpy as np

from .pointcloud import PointCloudRecorder, ScanConfig
from .simulation import MOON_GRAVITY, JumperSimulation
from .telemetry import UdpTelemetry
from .visualization import draw_flight_target, draw_landing_plane, draw_target_scan

FRAME_PERIOD = 1.0 / 60.0
MODEL_PATH = Path(__file__).resolve().parents[1] / "build" / "Jumping_Robot.xml"
SCAN_PATH = Path(__file__).resolve().parents[1] / "pointclouds" / "target_scan.ply"
# Default track target: a random ground position this far from the spawn, at
# least RANDOM_TARGET_MIN_DISTANCE away so the object is clear of the robot.
RANDOM_TARGET_RADIUS = 20.0  # m
RANDOM_TARGET_MIN_DISTANCE = 2.0  # m
KEY_NAMES = {
    glfw.KEY_SPACE: "space",
    glfw.KEY_BACKSPACE: "backspace",
    glfw.KEY_C: "c",
    glfw.KEY_X: "x",
    glfw.KEY_R: "r",
}


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m jumper_sim",
        description="Run the directional jumper in the MuJoCo Python viewer.",
    )
    parser.add_argument(
        "--controller", choices=("pid", "lqr"), default="pid",
        help="attitude controller (default: pid)",
    )
    parser.add_argument(
        "--model", type=Path, default=MODEL_PATH,
        help=f"MuJoCo MJCF path (default: {MODEL_PATH})",
    )
    target = parser.add_mutually_exclusive_group()
    target.add_argument(
        "--track-target", type=float, nargs=2, metavar=("X", "Y"), default=None,
        help="ground position the camera tracks during a jump's ascent, 0.15 m "
        "above the terrain there (default: a random point within "
        f"{RANDOM_TARGET_RADIUS:g} m of the spawn)",
    )
    target.add_argument(
        "--no-target", action="store_true",
        help="no target: no target object, and the camera tracks a point 3 m "
        "ahead of the take-off point",
    )
    parser.add_argument(
        "--scan-file", type=Path, default=SCAN_PATH,
        help="PLY file for the target point cloud collected during tracked ascents; "
        "emptied at start-up, every jump of the run adds to it ("
        f"default: {SCAN_PATH})",
    )
    parser.add_argument(
        "--scan-margin", type=float, default=5.0, metavar="DEG",
        help="only frames with the camera axis within this angle of the target "
        "count (skips the initial turn; default: 5)",
    )
    parser.add_argument(
        "--no-scan", action="store_true", help="do not collect the target point cloud",
    )
    parser.add_argument(
        "--no-scan-window", action="store_true",
        help="do not open the live point cloud window (the file is still written)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    model_path = args.model.expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"MuJoCo model not found: {model_path}")
    terrain_paths = (
        model_path.parent / "terrain.obj",
        model_path.parent / "terrain_height.bin",
    )
    missing_terrain_paths = [
        path for path in terrain_paths if not path.is_file()
    ]
    if model_path == MODEL_PATH.resolve() and missing_terrain_paths:
        raise FileNotFoundError(
            f"Terrain assets not found: {', '.join(map(str, missing_terrain_paths))}. "
            f"Extract {terrain_paths[0].with_name('moon-surface.zip')} into "
            f"build/Testing/Temporary/moon-surface and run "
            f"'{Path('.venv/bin/python')} build/convert_usd.py'."
        )

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    target_scan = None
    if not args.no_scan:
        target_scan = PointCloudRecorder(
            args.scan_file.expanduser().resolve(), ScanConfig(math.radians(args.scan_margin))
        )
    track_target = None if args.track_target is None else tuple(args.track_target)
    random_target = track_target is None and not args.no_target
    if random_target:
        track_target = random_target_near(spawn_position(model, data))
    simulation = JumperSimulation(
        model, data, args.controller,
        track_target=track_target,
        target_scan=target_scan,
    )
    if simulation.target_scan is not None:
        simulation.target_scan.start_run()  # this run starts a fresh cloud
    scan_window = None
    if simulation.target_scan is not None and not args.no_scan_window:
        scan_window = open_scan_window(simulation.target_scan.path, simulation.track_target_point)
    state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
    last_state = np.empty(mujoco.mj_stateSize(model, state_spec))
    mujoco.mj_getState(model, data, last_state, state_spec)
    key_queue: queue.SimpleQueue[str] = queue.SimpleQueue()

    def on_key(keycode: int) -> None:
        key_name = KEY_NAMES.get(keycode)
        if key_name is not None:
            key_queue.put(key_name)

    print(f"[MODEL] Loaded {model_path}")
    print(f"[CONTROLLER] Selected: {args.controller}")
    print(f"[GRAVITY] Selected: moon ({MOON_GRAVITY} m/s^2)")
    print("[SPACE] Jump | [BACKSPACE] Toggle Earth/Moon Gravity")
    print(
        "[C] Re-enable stabilization | "
        "[X] Disable/zero motor torque | [R] Reset"
    )
    print("[TELEMETRY] Streaming PlotJuggler data to 127.0.0.1:9870")
    if simulation.contact_gating:
        print(f"[CONTACT] {args.controller.upper()} is gated: attitude control in flight, wheel bleed-off in contact")
    if simulation.landing_reference is not None:
        print(
            "[CAMERA] Landing plane overlay: disc = latest fit (orange = used, "
            "grey = not used/held), green arrow = reference normal, blue dots = depth points"
        )
    if simulation.flight_reference is not None:
        target = (
            "3 m ahead of the take-off point" if simulation.track_target_point is None
            else "({:.2f}, {:.2f}, {:.2f})".format(*simulation.track_target_point)
        )
        print(
            f"[FLIGHT] Jumps: camera tracks {target} on the way up (red sphere), "
            "then levels and aligns with the ground seen on the way down"
        )
    if random_target:
        print(
            f"[TARGET] Random target within {RANDOM_TARGET_RADIUS:g} m of the spawn: "
            "--track-target {:.2f} {:.2f} repeats it, --no-target turns it off".format(*track_target)
        )
    if simulation.target_scan is not None:
        print(
            f"[SCAN] Target point cloud (frames within {args.scan_margin:g} deg of the "
            f"target, cyan dots) -> {simulation.target_scan.path}; emptied now, "
            "saved after every jump's ascent"
        )

    try:
        run_viewer(model, data, simulation, key_queue, on_key, last_state, state_spec)
    finally:
        if scan_window is not None and scan_window.poll() is None:
            scan_window.terminate()
    return 0


def spawn_position(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[float, float]:
    """Ground position (x, y) of the robot's root joint at start-up."""
    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root")
    if root < 0:
        return (0.0, 0.0)
    adr = int(model.jnt_qposadr[root])
    return (float(data.qpos[adr]), float(data.qpos[adr + 1]))


def random_target_near(
    centre: Sequence[float],
    radius: float = RANDOM_TARGET_RADIUS,
    min_distance: float = RANDOM_TARGET_MIN_DISTANCE,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Uniformly random ground position between ``min_distance`` and ``radius`` of ``centre``."""
    rng = np.random.default_rng() if rng is None else rng
    distance = math.sqrt(rng.uniform(min_distance**2, radius**2))  # uniform over the area
    heading = rng.uniform(0.0, 2.0 * math.pi)
    return (
        float(centre[0]) + distance * math.cos(heading),
        float(centre[1]) + distance * math.sin(heading),
    )


def open_scan_window(path: Path, target: Sequence[float] | None) -> subprocess.Popen | None:
    """Start the live point cloud window in its own process."""
    command = [sys.executable, "-m", "jumper_sim.cloud_viewer", str(path)]
    if target is not None:
        command += ["--target", *(str(v) for v in target)]
    try:
        window = subprocess.Popen(command, cwd=Path(__file__).resolve().parents[1])
    except OSError as error:
        print(f"[SCAN] Could not open the point cloud window: {error}")
        return None
    print("[SCAN] Point cloud window opened; it updates after every jump's ascent")
    return window


def track_robot(viewer, base_body_id: int) -> None:
    """Point the viewer's camera at the robot and follow it."""
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    viewer.cam.trackbodyid = base_body_id
    viewer.cam.distance = 3.0
    viewer.cam.azimuth = 135.0
    viewer.cam.elevation = -20.0


def run_viewer(model, data, simulation, key_queue, on_key, last_state, state_spec) -> None:
    with closing(UdpTelemetry()) as telemetry:
        with mujoco.viewer.launch_passive(
            model, data, key_callback=on_key
        ) as viewer:
            base_body_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, "base_link"
            )
            if base_body_id < 0:
                raise ValueError("MuJoCo model is missing required body 'base_link'")
            with viewer.lock():
                track_robot(viewer, base_body_id)
            overlay_scene = getattr(viewer, "user_scn", None)

            while viewer.is_running():
                frame_started = time.perf_counter()
                with viewer.lock():
                    while not key_queue.empty():
                        key = key_queue.get()
                        if key == "backspace":
                            # MuJoCo's viewer resets data on Backspace before
                            # forwarding the key to the Python callback.
                            mujoco.mj_setState(model, data, last_state, state_spec)
                            mujoco.mj_forward(model, data)
                        message = simulation.handle_key(key)
                        if message:
                            print(message)
                        if key == "r":
                            track_robot(viewer, base_body_id)  # follow the robot again

                    frame_end = float(data.time) + FRAME_PERIOD
                    while data.time < frame_end:
                        simulation.step()
                        for event in simulation.flight_events:
                            print(event)
                        simulation.flight_events.clear()
                    if overlay_scene is not None:
                        overlay_scene.ngeom = 0
                        draw_landing_plane(overlay_scene, simulation)
                        draw_flight_target(overlay_scene, simulation)
                        draw_target_scan(overlay_scene, simulation)
                    telemetry.send(simulation.telemetry_payload())
                    mujoco.mj_getState(model, data, last_state, state_spec)

                viewer.sync()
                remaining = FRAME_PERIOD - (time.perf_counter() - frame_started)
                if remaining > 0.0:
                    time.sleep(remaining)


if __name__ == "__main__":
    raise SystemExit(main())
