"""Free-flight attitude benchmark for the jumper's attitude controllers.

The robot is placed high above the terrain with an upward launch velocity, so
it is in ballistic flight for the whole run and only the reaction wheels can
change its attitude. Each scenario starts from a disturbed attitude and/or
body rate, and the run is scored against the true attitude error taken from
the free-joint quaternion (independent of what the controller itself sees).

Usage:
    python -m benchmarks.lqr_benchmark --label baseline
    python -m benchmarks.lqr_benchmark --label fix1 --compare baseline
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from jumper_sim.controller import ReactionWheelTuning, wheel_effective_inertia
from jumper_sim.simulation import MOON_GRAVITY, JumperSimulation

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "build" / "Jumping_Robot.xml"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

DURATION = 2.0
START_HEIGHT = 40.0
LAUNCH_VELOCITY = 2.0
SETTLE_THRESHOLD_DEG = 2.0
TRACE_EVERY = 10  # store every 10th step (100 Hz) for plotting


@dataclass(frozen=True)
class Scenario:
    name: str
    euler_deg: tuple[float, float, float]  # roll, pitch, yaw (ZYX)
    body_rates: tuple[float, float, float]  # rad/s, body frame
    wheel_speeds: tuple[float, float, float] = (0.0, 0.0, 0.0)  # rad/s


SCENARIOS = (
    Scenario("tilt", (20.0, -15.0, 10.0), (0.0, 0.0, 0.0)),
    Scenario("tumble", (0.0, 0.0, 0.0), (1.5, -1.0, 0.5)),
    Scenario("large_tilt", (40.0, 30.0, -20.0), (0.0, 0.0, 0.0)),
    Scenario("tilt+tumble_spun_wheels", (10.0, 25.0, 0.0), (0.5, 1.0, -0.8),
             (60.0, -60.0, 40.0)),
    Scenario("near_gimbal_lock", (30.0, 80.0, 0.0), (0.0, 0.0, 0.0)),
    Scenario("past_vertical", (0.0, 100.0, 0.0), (0.0, 0.0, 0.0)),
    Scenario("yaw_with_spun_pitch_wheel", (0.0, 0.0, 30.0), (0.0, 0.0, 0.0),
             (0.0, 70.0, 0.0)),
)


def euler_zyx_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])


def run_scenario(controller_name: str, scenario: Scenario,
                 controller_factory=None) -> dict:
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    sim = JumperSimulation(model, data, controller_name)
    if controller_factory is not None:
        sim.controller = controller_factory(model, sim.tuning)
    sim.model.opt.gravity[2] = -MOON_GRAVITY
    sim.selected_gravity = MOON_GRAVITY

    ids = sim.ids
    qa = model.jnt_qposadr[ids.root_joint]
    va = model.jnt_dofadr[ids.root_joint]
    data.qpos[qa:qa + 3] = (0.0, 0.0, START_HEIGHT)
    data.qpos[qa + 3:qa + 7] = euler_zyx_to_quat(
        *(math.radians(a) for a in scenario.euler_deg))
    data.qvel[va:va + 3] = (0.0, 0.0, LAUNCH_VELOCITY)
    data.qvel[va + 3:va + 6] = scenario.body_rates
    for joint_id, speed in zip(ids.wheel_joints, scenario.wheel_speeds):
        data.qvel[model.jnt_dofadr[joint_id]] = speed
    mujoco.mj_forward(model, data)

    tuning: ReactionWheelTuning = sim.tuning
    max_torque = wheel_effective_inertia(tuning) * tuning.max_wheel_acceleration
    steps = int(round(DURATION / model.opt.timestep))
    dt = float(model.opt.timestep)

    errors = np.empty(steps)
    rates = np.empty(steps)
    saturated = np.zeros(steps, dtype=bool)
    wheel_peak = 0.0
    abs_torque_integral = 0.0
    trace_t, trace_err, trace_axes = [], [], []

    for k in range(steps):
        sim.step()
        q = data.qpos[qa + 3:qa + 7]
        err = math.degrees(2.0 * math.acos(min(1.0, abs(float(q[0])))))
        errors[k] = err
        w = data.qvel[va + 3:va + 6]
        rates[k] = float(np.linalg.norm(w))
        ctrl = np.array([data.ctrl[a] for a in ids.wheel_actuators])
        saturated[k] = bool(np.any(np.abs(ctrl) >= 0.999 * max_torque))
        abs_torque_integral += float(np.sum(np.abs(ctrl))) * dt
        wheel_peak = max(wheel_peak, max(
            abs(float(data.qvel[model.jnt_dofadr[j]])) for j in ids.wheel_joints))
        if k % TRACE_EVERY == 0:
            trace_t.append(round(float(data.time), 4))
            trace_err.append(round(err, 4))
            # signed small-angle body-frame error vector, for per-axis plots
            sgn = 1.0 if q[0] >= 0 else -1.0
            trace_axes.append([round(math.degrees(2.0 * sgn * float(v)), 4)
                               for v in q[1:4]])

    above = np.nonzero(errors > SETTLE_THRESHOLD_DEG)[0]
    if len(above) == 0:
        settling = 0.0
    elif above[-1] == steps - 1:
        settling = None  # never settled within the window
    else:
        settling = (above[-1] + 1) * dt

    return {
        "scenario": scenario.name,
        "settling_time_s": settling,
        "peak_error_deg": float(errors.max()),
        "final_error_deg": float(errors[-1]),
        "iae_deg_s": float(errors.sum() * dt),
        "peak_body_rate_rad_s": float(rates.max()),
        "saturated_fraction": float(saturated.mean()),
        "peak_wheel_speed_rev_s": wheel_peak / (2 * math.pi),
        "torque_effort_Nms": abs_torque_integral,
        "trace": {"t": trace_t, "err": trace_err, "axes": trace_axes},
    }


def run_all(controller_name: str = "lqr", controller_factory=None) -> list[dict]:
    return [run_scenario(controller_name, s, controller_factory) for s in SCENARIOS]


def _fmt(v, nd=2):
    if v is None:
        return "  n/s "
    return f"{v:6.{nd}f}"


def print_table(label: str, results: list[dict],
                reference: list[dict] | None = None) -> None:
    print(f"\n=== {label} ===")
    header = (f"{'scenario':26s} {'settle[s]':>9s} {'peak[deg]':>9s} "
              f"{'final[deg]':>10s} {'IAE':>7s} {'sat%':>6s} {'wheel[rev/s]':>12s}")
    print(header)
    for i, r in enumerate(results):
        line = (f"{r['scenario']:26s} {_fmt(r['settling_time_s']):>9s} "
                f"{_fmt(r['peak_error_deg']):>9s} {_fmt(r['final_error_deg']):>10s} "
                f"{_fmt(r['iae_deg_s']):>7s} {100*r['saturated_fraction']:6.1f} "
                f"{r['peak_wheel_speed_rev_s']:12.2f}")
        ref = None
        if reference is not None:
            match = [x for x in reference if x["scenario"] == r["scenario"]]
            ref = match[0] if match else None
        if ref is not None:
            a, b = ref["settling_time_s"], r["settling_time_s"]
            d = "" if a is None or b is None else f"  settle {b - a:+.3f}s"
            d += f"  IAE {100*(r['iae_deg_s']/ref['iae_deg_s']-1):+.0f}%"
            line += d
        print(line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--controller", default="lqr", choices=("lqr", "pid"))
    parser.add_argument("--compare", help="label of a saved result to diff against")
    args = parser.parse_args()

    results = run_all(args.controller)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{args.label}.json").write_text(json.dumps(results))
    reference = None
    if args.compare:
        reference = json.loads((RESULTS_DIR / f"{args.compare}.json").read_text())
    print_table(args.label, results, reference)


if __name__ == "__main__":
    main()
