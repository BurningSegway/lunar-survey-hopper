from __future__ import annotations

import argparse
from collections.abc import Sequence
from contextlib import closing
from pathlib import Path
import queue
import time

import glfw
import mujoco
import mujoco.viewer
import numpy as np

from .simulation import EARTH_GRAVITY, JumperSimulation
from .telemetry import UdpTelemetry

FRAME_PERIOD = 1.0 / 60.0
MODEL_PATH = Path(__file__).resolve().parents[1] / "build" / "Jumping_Robot.xml"
KEY_NAMES = {
    glfw.KEY_SPACE: "space",
    glfw.KEY_ENTER: "enter",
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    model_path = args.model.expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"MuJoCo model not found: {model_path}")
    terrain_paths = (
        model_path.parent / "terrain.obj",
        model_path.parent / "terrain_height.png",
    )
    missing_terrain_paths = [
        path for path in terrain_paths if not path.is_file()
    ]
    if model_path == MODEL_PATH.resolve() and missing_terrain_paths:
        raise FileNotFoundError(
            f"Terrain assets not found: {', '.join(map(str, missing_terrain_paths))}. "
            f"Generate them from {terrain_paths[0].with_name('terrain_merged.usd')} "
            f"with '{Path('.venv/bin/python')} build/convert_usd.py'."
        )

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    simulation = JumperSimulation(model, data, args.controller)
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
    print(f"[GRAVITY] Selected: earth ({EARTH_GRAVITY} m/s^2)")
    print(
        "[SPACE] Jump | [ENTER] Toggle Tilt | [BACKSPACE] Toggle Earth/Moon Gravity"
    )
    print(
        "[C] Request/cancel controlled wheel rest | "
        "[X] Disable/zero motor torque | [R] Reset"
    )

    with closing(UdpTelemetry()) as telemetry:
        with mujoco.viewer.launch_passive(
            model, data, key_callback=on_key
        ) as viewer:
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

                    frame_end = float(data.time) + FRAME_PERIOD
                    while data.time < frame_end:
                        simulation.step()
                    telemetry.send(simulation.telemetry_payload())
                    mujoco.mj_getState(model, data, last_state, state_spec)

                viewer.sync()
                remaining = FRAME_PERIOD - (time.perf_counter() - frame_started)
                if remaining > 0.0:
                    time.sleep(remaining)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
