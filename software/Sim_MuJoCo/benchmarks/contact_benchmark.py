"""Contact-gating benchmark: landings with bounces, and jumps from the ground.

Runs the LQR (with the camera landing reference) with contact gating on and
off on:

* drops onto sloped terrain from 3 m and 6 m, continued for 5 s after the
  first touchdown so bounces and settling are included;
* a stand-then-jump on the spawn terrain under Moon gravity (jump at 2 s,
  14 s total). Note the jump's launch speed depends on the MuJoCo version:
  ~5 m/s on the pinned 3.2.3, ~1 m/s on 3.16.

Scored after the first touchdown (drops) or over the whole run (jumps):
wheel saturation (time with any wheel at >= 98 % of its speed limit), peak
wheel speed, wheel speed left at the end, number of contact phases, and the
misalignment to the true ground plane at the first touchdown.

Usage:
    python -m benchmarks.contact_benchmark --label gating_v1
"""

from __future__ import annotations

import argparse
import json
import math

import mujoco
import numpy as np

from jumper_sim.depth_camera import HeightfieldRayCaster
from jumper_sim.landing_reference import LandingReferenceConfig
from jumper_sim.simulation import MOON_GRAVITY, JumperSimulation

from .landing_benchmark import pick_spots, true_plane_normal
from .lqr_benchmark import MODEL_PATH, RESULTS_DIR, euler_zyx_to_quat

AFTER_TOUCHDOWN = 5.0
# A full hop on MuJoCo 3.2.3 (the pinned version) flies ~6.5 s on the Moon.
JUMP_DURATION = 14.0
SATURATION_FRACTION = 0.98


def _wheel_speeds(model, data, sim) -> np.ndarray:
    return np.array([data.qvel[model.jnt_dofadr[j]] for j in sim.ids.wheel_joints])


def run(kind: str, gating: bool, caster, radius, x=0.0, y=0.0, height=0.0,
        euler=(0.0, 0.0, 0.0)) -> dict:
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    sim = JumperSimulation(model, data, "lqr", contact_gating=gating)
    sim.model.opt.gravity[2] = -MOON_GRAVITY
    sim.selected_gravity = MOON_GRAVITY
    qa = model.jnt_qposadr[sim.ids.root_joint]
    limit = sim.tuning.max_wheel_velocity
    if kind == "drop":
        ground = float(caster.height(np.array([x]), np.array([y]))[0])
        data.qpos[qa:qa + 3] = (x, y, ground + height)
        data.qpos[qa + 3:qa + 7] = euler_zyx_to_quat(*(math.radians(a) for a in euler))
        mujoco.mj_forward(model, data)
        sim.landing_reference.reset(float(data.time))
        end_time, touchdown, misalignment = math.inf, None, None
    else:
        mujoco.mj_forward(model, data)
        end_time, touchdown, misalignment = JUMP_DURATION, 0.0, None

    saturated = steps = 0
    peak = 0.0
    phases_at_touchdown = 0
    jumped = False
    was_in_contact = sim.in_contact
    flight_start, longest_flight, max_tilt = float(data.time), 0.0, 0.0
    while data.time < min(end_time, JUMP_DURATION):
        if kind == "jump" and not jumped and data.time >= 2.0:
            sim.handle_key("space")
            jumped = True
        sim.step()
        if was_in_contact and not sim.in_contact:
            flight_start = float(data.time)
        if sim.in_contact and not was_in_contact:
            flight = float(data.time) - flight_start
            if touchdown is None:
                touchdown = float(data.time)
                end_time = touchdown + AFTER_TOUCHDOWN
                phases_at_touchdown = sim.contact_phases - 1
            if flight > longest_flight and (kind == "drop" or jumped):
                # Score the landing that ends the longest flight (the hop).
                longest_flight = flight
                body_z = data.xmat[model.jnt_bodyid[sim.ids.root_joint]].reshape(3, 3)[:, 2]
                truth = true_plane_normal(caster, float(data.qpos[qa]), float(data.qpos[qa + 1]), radius)
                misalignment = math.degrees(math.acos(min(1.0, float(body_z @ truth))))
        was_in_contact = sim.in_contact
        if touchdown is not None:
            body_z = data.xmat[model.jnt_bodyid[sim.ids.root_joint]].reshape(3, 3)[:, 2]
            max_tilt = max(max_tilt, math.degrees(math.acos(max(-1.0, min(1.0, float(body_z[2]))))))
        if touchdown is not None:
            w = np.abs(_wheel_speeds(model, data, sim))
            steps += 1
            saturated += bool(np.any(w >= SATURATION_FRACTION * limit))
            peak = max(peak, float(w.max()))
    final = np.abs(_wheel_speeds(model, data, sim))
    return {
        "kind": kind, "gating": gating, "x": x, "y": y, "height": height,
        "saturated_fraction": saturated / max(steps, 1),
        "peak_wheel_rev_s": peak / (2 * math.pi),
        "final_wheel_rev_s": float(final.max()) / (2 * math.pi),
        "contact_phases": sim.contact_phases - phases_at_touchdown,
        "touchdown_misalignment_deg": misalignment,
        "longest_flight_s": longest_flight,
        "max_tilt_after_touchdown_deg": max_tilt,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    radius = LandingReferenceConfig().roi_radius
    caster = HeightfieldRayCaster(mujoco.MjModel.from_xml_path(str(MODEL_PATH)))
    rng = np.random.default_rng(21)
    results = []
    for gating in (False, True):
        results.append(run("jump", gating, caster, radius))
    for x, y, _ in pick_spots(caster, radius)[::2]:
        for height in (3.0, 6.0):
            euler = rng.uniform(-15.0, 15.0, 3)
            for gating in (False, True):
                results.append(run("drop", gating, caster, radius, x, y, height, euler))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{args.label}.json").write_text(json.dumps(results, indent=1))

    print(f"\n=== {args.label} ===")
    print(f"{'run':20s} {'gating':>6s} {'sat%':>6s} {'peak':>6s} {'final':>6s} {'phases':>6s} {'td mis':>6s} {'tipped':>6s}")
    for r in results:
        name = "jump (spawn)" if r["kind"] == "jump" else f"drop {r['height']:.0f}m ({r['x']:.1f},{r['y']:.1f})"
        mis = r["touchdown_misalignment_deg"]
        tipped = "yes" if r["max_tilt_after_touchdown_deg"] > 60.0 else ""
        print(f"{name:20s} {str(r['gating']):>6s} {100*r['saturated_fraction']:6.1f} "
              f"{r['peak_wheel_rev_s']:6.1f} {r['final_wheel_rev_s']:6.2f} {r['contact_phases']:6d} "
              f"{'   n/a' if mis is None else f'{mis:6.2f}'} {tipped:>6s}")
    for gating in (False, True):
        rows = [r for r in results if r["gating"] == gating]
        sat = np.array([r["saturated_fraction"] for r in rows]) * 100
        fin = np.array([r["final_wheel_rev_s"] for r in rows])
        print(f"gating={gating!s:5}: saturated median {np.median(sat):.1f}% max {sat.max():.1f}% | "
              f"wheel speed left median {np.median(fin):.2f} rev/s max {fin.max():.2f}")


if __name__ == "__main__":
    main()
