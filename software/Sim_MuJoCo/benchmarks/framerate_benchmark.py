"""Frame-rate benchmark: the viewer loop's frame rate over a Moon hop.

Runs the app's own viewer loop (``jumper_sim.app.run_viewer``: the sim paced
to the wall clock, overlays, telemetry, the point-cloud recorder) with a
stand-in for MuJoCo's passive viewer, so no window opens. Space is pressed
at 1 s; the run ends 1.5 s after the hop's touchdown is settled (LANDING ->
GROUND) or at a wall-time cap.

Per viewer sync it records the wall time and the sim time. Reported per
hop phase (ground before the jump, airborne, after touchdown):

* frame rate: syncs per wall second (the loop aims at 1 / app.SYNC_PERIOD, 60 Hz);
* worst frame: the longest wall gap between two syncs (a visible hitch);
* real-time factor: sim time advanced per wall second. Below 1 the sim
  falls behind and the loop re-anchors to the wall clock (slow motion).

Not included: drawing itself (MuJoCo's viewer renders in its own thread
from a copy made in ``sync``) and the point-cloud window (own process).

Usage:
    python -m benchmarks.framerate_benchmark --label fps_gpu
    python -m benchmarks.framerate_benchmark --label fps_cpu --cpu
    python -m benchmarks.framerate_benchmark --label fps_cpu_smooth --cpu --fraction 0.03
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import replace
import json
import queue
import tempfile
import threading
import time
from pathlib import Path
from unittest import mock

import mujoco
import numpy as np

from jumper_sim import app
from jumper_sim.flight_phase import FlightPhase
from jumper_sim.lidar import LIVOX_MID360
from jumper_sim.pointcloud import PointCloudRecorder
from jumper_sim.simulation import JumperSimulation

from .lqr_benchmark import MODEL_PATH, RESULTS_DIR

JUMP_AT = 1.0  # s of sim time
AFTER_SETTLE = 1.5  # s of sim time after LANDING -> GROUND
WALL_CAP = 120.0  # s
TRACK_OFFSET = (10.0, 0.0)  # m from the spawn: a fixed, repeatable target
SCENE_GEOMS = 10000  # the passive viewer's user scene size


class _Camera:
    def __init__(self) -> None:
        self.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.distance = self.azimuth = self.elevation = 0.0
        self.lookat = np.zeros(3)


class FakeViewer:
    """What ``run_viewer`` uses of ``mujoco.viewer.Handle``; records each sync."""

    def __init__(self, model, data, simulation, key_queue, wall_cap: float) -> None:
        self.cam = _Camera()
        self.user_scn = mujoco.MjvScene(model, maxgeom=SCENE_GEOMS)
        self._lock = threading.Lock()
        self._data = data
        self._simulation = simulation
        self._keys = key_queue
        self._wall_cap = wall_cap
        self._started = time.perf_counter()
        self._jumped = False
        self._settled_at: float | None = None
        self._was_landing = False
        self.frames: list[tuple[float, float, str, bool]] = []  # wall, sim, phase, jumped

    @contextmanager
    def lock(self):
        with self._lock:
            yield

    def sync(self) -> None:
        phase = self._simulation.flight_phase.phase
        sim_time = float(self._data.time)
        self.frames.append((time.perf_counter(), sim_time, phase.name, self._jumped))
        if not self._jumped and sim_time >= JUMP_AT:
            self._keys.put("space")
            self._jumped = True
            return
        if not self._jumped:
            return  # the spawn drop's landing is not the hop's
        if phase is FlightPhase.LANDING:
            self._was_landing = True
        elif phase is FlightPhase.GROUND and self._was_landing and self._settled_at is None:
            self._settled_at = sim_time

    def is_running(self) -> bool:
        if time.perf_counter() - self._started > self._wall_cap:
            return False
        return self._settled_at is None or float(self._data.time) < self._settled_at + AFTER_SETTLE


def _segment(frames: list[tuple[float, float, str, bool]], name: str) -> dict:
    """Frame-rate figures over consecutive syncs ``frames``."""
    if len(frames) < 2:
        return {"segment": name, "frames": len(frames)}
    wall = np.array([f[0] for f in frames])
    sim = np.array([f[1] for f in frames])
    gaps = np.diff(wall)
    span = wall[-1] - wall[0]
    return {
        "segment": name,
        "frames": len(frames),
        "wall_s": round(float(span), 3),
        "sim_s": round(float(sim[-1] - sim[0]), 3),
        "fps": round(float(len(gaps) / span), 1),
        "fps_p5": round(float(1.0 / np.percentile(gaps, 95)), 1),
        "worst_frame_ms": round(float(gaps.max() * 1e3), 1),
        "real_time_factor": round(float((sim[-1] - sim[0]) / span), 3),
    }


def split_by_phase(frames: list[tuple[float, float, str, bool]]) -> list[dict]:
    """Ground before the jump (incl. the spawn drop), the hop (Space to its
    last airborne sync, TAKEOFF..DESCENT), after touchdown."""
    airborne = {"TAKEOFF", "ASCENT", "DESCENT"}
    first = next((i for i, f in enumerate(frames) if f[3]), len(frames))
    last = max((i for i, f in enumerate(frames) if f[3] and f[2] in airborne), default=first - 1)
    return [
        _segment(frames[:first], "ground"),
        _segment(frames[first:last + 1], "airborne"),
        _segment(frames[last + 1:], "after touchdown"),
        _segment(frames, "whole run"),
    ]


def run(cpu: bool, fraction: float | None, max_range: float | None,
        wall_cap: float = WALL_CAP, scan: bool = True) -> dict:
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    patches = [mock.patch("jumper_sim.simulation.gpu_available", return_value=False)] if cpu else []
    for patch in patches:
        patch.start()
    try:
        gpu = not cpu and app.gpu_available()
        point_fraction = fraction if fraction is not None else (1.0 if gpu else 0.1)
        lidar = replace(LIVOX_MID360, point_fraction=point_fraction)
        if max_range is not None:
            lidar = replace(lidar, max_range=max_range)
        spawn = app.spawn_position(model, data)
        target = (spawn[0] + TRACK_OFFSET[0], spawn[1] + TRACK_OFFSET[1])
        with tempfile.TemporaryDirectory() as scratch:
            recorder = PointCloudRecorder(Path(scratch) / "scan.ply") if scan else None
            simulation = JumperSimulation(model, data, "lqr", track_target=target,
                                          target_scan=recorder, lidar=lidar)
            if recorder is not None:
                recorder.start_run()
            key_queue: queue.SimpleQueue[str] = queue.SimpleQueue()
            spec = mujoco.mjtState.mjSTATE_INTEGRATION
            last_state = np.empty(mujoco.mj_stateSize(model, spec))
            mujoco.mj_getState(model, data, last_state, spec)
            viewer = FakeViewer(model, data, simulation, key_queue, wall_cap)

            @contextmanager
            def launch_passive(*_args, **_kwargs):
                yield viewer

            with mock.patch("mujoco.viewer.launch_passive", launch_passive):
                app.run_viewer(model, data, simulation, key_queue, lambda _k: None,
                               last_state, spec)
    finally:
        for patch in patches:
            patch.stop()
    return {
        "caster": "GPU (Warp)" if gpu else "CPU",
        "point_fraction": point_fraction,
        "max_range": lidar.max_range,
        "segments": split_by_phase(viewer.frames),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--cpu", action="store_true", help="cast on the CPU even with a GPU")
    parser.add_argument("--fraction", type=float, default=None,
                        help="lidar point fraction (default: as the app, 1.0 GPU / 0.1 CPU)")
    parser.add_argument("--max-range", type=float, default=None, help="lidar cut-off [m]")
    args = parser.parse_args()
    result = run(args.cpu, args.fraction, args.max_range)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{args.label}.json").write_text(json.dumps(result, indent=1))
    print(f"\n=== {args.label}: {result['caster']}, {result['point_fraction']:.0%} "
          f"of the point rate, cut-off {result['max_range']:g} m ===")
    print(f"{'segment':<16} {'frames':>6} {'fps':>6} {'fps p5':>7} {'worst ms':>9} {'real time':>9}")
    for s in result["segments"]:
        if "fps" not in s:
            print(f"{s['segment']:<16} {s['frames']:>6}")
            continue
        print(f"{s['segment']:<16} {s['frames']:>6} {s['fps']:>6.1f} {s['fps_p5']:>7.1f} "
              f"{s['worst_frame_ms']:>9.1f} {s['real_time_factor']:>9.3f}")


if __name__ == "__main__":
    main()
