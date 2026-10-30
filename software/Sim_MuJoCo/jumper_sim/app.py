from __future__ import annotations

import argparse
from dataclasses import replace
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

from .gpu_ray_caster import gpu_available
from .lidar import LIDARS, SONY_AS_DT1
from .pointcloud import PointCloudRecorder
from .simulation import MOON_GRAVITY, JumperSimulation
from .telemetry import UdpTelemetry
from .visualization import (
    draw_flight_target, draw_height_label, draw_landing_plane, draw_target_scan,
)

# The viewer draws in its own thread at the display rate; hand it new state
# more often than that so every drawn frame is fresh.
SYNC_PERIOD = 1.0 / 120.0  # s of wall time between viewer syncs
# Most sim time one sync may catch up on; if a slow stretch puts the sim
# further behind the wall clock, it runs slower than real time instead.
MAX_CATCH_UP = 0.05  # s
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
        help="ground position the lidar tracks during a jump's ascent, 0.15 m "
        "above the terrain there (default: a random point within "
        f"{RANDOM_TARGET_RADIUS:g} m of the spawn)",
    )
    target.add_argument(
        "--no-target", action="store_true",
        help="no target: no target object, and the lidar tracks a point 3 m "
        "ahead of the take-off point",
    )
    parser.add_argument(
        "--scan-file", type=Path, default=SCAN_PATH,
        help="PLY file for the lidar point cloud of every tracked jump (lift-off to "
        "touchdown); emptied at start-up, every jump of the run adds to it ("
        f"default: {SCAN_PATH})",
    )
    parser.add_argument(
        "--no-scan", action="store_true", help="do not collect the lidar point cloud",
    )
    parser.add_argument(
        "--lidar", choices=sorted(LIDARS), default="mid360s",
        help="lidar: the Livox Mid-360S on the +x side face (default, used for "
        "development), or the Sony AS-DT1 (24x24 grid, 30 deg) under the bottom "
        "face, the robot facing its direction of travel in flight",
    )
    parser.add_argument(
        "--spin", action="store_true",
        help="spin slowly (45 deg/s) about the vertical in flight, sweeping the "
        "lidar round; with the AS-DT1 instead of facing the direction of travel",
    )
    parser.add_argument(
        "--lidar-fraction", type=float, default=None, metavar="F",
        help="fraction of the lidar's real point rate that is simulated "
        "(default: 1; for the Mid-360S 0.1 without an NVIDIA GPU, since the CPU "
        "caster runs its full rate 2-10x slower than real time)",
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
    fraction = args.lidar_fraction
    spec = LIDARS[args.lidar]
    if fraction is None:
        # The AS-DT1's 8640 points/s are cheap on any hardware.
        fraction = 1.0 if gpu_available() or spec is SONY_AS_DT1 else 0.1
    lidar = replace(spec, point_fraction=fraction)
    print(f"[LIDAR] {lidar.name}: {fraction:.0%} of the point rate, rays cast on the "
          f"{'GPU (Warp)' if gpu_available() else 'CPU'}")
    target_scan = None
    if not args.no_scan:
        target_scan = PointCloudRecorder(args.scan_file.expanduser().resolve())
    track_target = None if args.track_target is None else tuple(args.track_target)
    random_target = track_target is None and not args.no_target
    if random_target:
        track_target = random_target_near(spawn_position(model, data))
    simulation = JumperSimulation(
        model, data, args.controller,
        track_target=track_target,
        target_scan=target_scan,
        lidar=lidar,
        spin=args.spin,
    )
    if simulation.target_scan is not None:
        simulation.target_scan.start_run()  # this run starts a fresh cloud
    scan_window = None
    if simulation.target_scan is not None and not args.no_scan_window:
        scan_window = open_scan_window(
            simulation.target_scan.path, simulation.track_target_point,
        )
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
            "[LANDING] Landing plane overlay at the predicted landing spot: disc = latest fit "
            "(orange = used, grey = not used/held), green arrow = reference normal, "
            "blue dots = lidar points it used"
        )
    if simulation.flight_reference is not None:
        target = (
            "3 m ahead of the take-off point" if simulation.track_target_point is None
            else "({:.2f}, {:.2f}, {:.2f})".format(*simulation.track_target_point)
        )
        print(
            f"[FLIGHT] Jumps: the robot's bottom tracks {target} on the way up (red sphere), "
            "then levels and aligns with the predicted landing spot's ground"
        )
    if random_target:
        print(
            f"[TARGET] Random target within {RANDOM_TARGET_RADIUS:g} m of the spawn: "
            "--track-target {:.2f} {:.2f} repeats it, --no-target turns it off".format(*track_target)
        )
    if simulation.target_scan is not None:
        print(
            f"[SCAN] Lidar point cloud of every jump (cyan dots) -> "
            f"{simulation.target_scan.path}; emptied now, saved at every touchdown"
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
    print("[SCAN] Point cloud window opened; it updates at every touchdown")
    return window


class RobotFollower:
    """Keeps the viewer camera centred on the robot.

    MuJoCo's own tracking camera moves its centre 20 % of the way to the body
    on every drawn frame. The viewer draws in its own thread, so with a fast
    robot the camera glides while the robot jumps at each sync: visible
    stutter. Here the free camera's centre is set to the robot in the same
    locked update that moves the robot, so both always move together. Mouse
    rotate and zoom still work; panning (or picking another camera) hands
    the camera to the user until ``start`` is called again (R).
    """

    def __init__(self, base_body_id: int) -> None:
        self.base_body_id = base_body_id
        self.active = False
        self._lookat: np.ndarray | None = None

    def start(self, viewer, data) -> None:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.distance = 3.0
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -20.0
        self.active = True
        self._lookat = None
        self.update(viewer, data)

    def update(self, viewer, data) -> None:
        if not self.active:
            return
        cam = viewer.cam
        moved_by_user = self._lookat is not None and not np.allclose(cam.lookat, self._lookat)
        if cam.type != mujoco.mjtCamera.mjCAMERA_FREE or moved_by_user:
            self.active = False
            return
        cam.lookat[:] = data.xpos[self.base_body_id]
        self._lookat = cam.lookat.copy()


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
            follower = RobotFollower(base_body_id)
            with viewer.lock():
                follower.start(viewer, data)
            overlay_scene = getattr(viewer, "user_scn", None)
            # Sim time follows the wall clock from this anchor.
            wall_anchor, sim_anchor = time.perf_counter(), float(data.time)

            while viewer.is_running():
                sync_started = time.perf_counter()
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
                            follower.start(viewer, data)  # follow the robot again
                    target = sim_anchor + (sync_started - wall_anchor)
                    if target < float(data.time) - 1e-9 or target - float(data.time) > MAX_CATCH_UP:
                        # Reset (time went back) or too far behind: re-anchor.
                        target = min(max(target, float(data.time)), float(data.time) + MAX_CATCH_UP)
                        wall_anchor, sim_anchor = sync_started, target
                    while data.time < target:
                        simulation.step()
                        for event in simulation.flight_events:
                            print(event)
                        simulation.flight_events.clear()
                    if overlay_scene is not None:
                        overlay_scene.ngeom = 0
                        draw_landing_plane(overlay_scene, simulation)
                        draw_flight_target(overlay_scene, simulation)
                        draw_target_scan(overlay_scene, simulation)
                        draw_height_label(overlay_scene, simulation)
                    follower.update(viewer, data)
                    telemetry.send(simulation.telemetry_payload())
                    mujoco.mj_getState(model, data, last_state, state_spec)

                viewer.sync()
                remaining = SYNC_PERIOD - (time.perf_counter() - sync_started)
                if remaining > 0.0:
                    time.sleep(remaining)

if __name__ == "__main__":
    raise SystemExit(main())
