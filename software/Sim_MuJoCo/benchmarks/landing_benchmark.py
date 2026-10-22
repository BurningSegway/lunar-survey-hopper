"""Landing-attitude benchmark for the depth-camera landing reference.

The robot is released at rest at several heights above sloped spots on the
terrain, with a disturbed initial attitude, and falls under Moon gravity until
its first contact with the terrain. Scored at touchdown:

* misalignment -- angle between the body z axis and the true area-averaged
  ground plane (disc of ``roi_radius``) below the robot at touchdown;
* estimate error -- angle between the filter's plane normal and that truth.

Each drop is run with the camera reference on and off (off = land level).

Usage:
    python -m benchmarks.landing_benchmark --label landing_v1
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from jumper_sim.depth_camera import HeightfieldRayCaster
from jumper_sim.landing_reference import LandingReferenceConfig
from jumper_sim.simulation import MOON_GRAVITY, JumperSimulation

from .lqr_benchmark import MODEL_PATH, RESULTS_DIR, euler_zyx_to_quat

HEIGHTS = (3.0, 6.0, 10.0)
SPOTS = 12
MAX_TIME = 6.0


def true_plane_normal(caster: HeightfieldRayCaster, x: float, y: float,
                      radius: float) -> np.ndarray:
    g = np.linspace(-radius, radius, 101)
    gx, gy = np.meshgrid(g, g)
    inside = gx**2 + gy**2 <= radius**2
    design = np.column_stack([gx[inside], gy[inside], np.ones(inside.sum())])
    a, b, _ = np.linalg.lstsq(design, caster.height(gx[inside] + x, gy[inside] + y),
                              rcond=None)[0]
    n = np.array([-a, -b, 1.0])
    return n / np.linalg.norm(n)


def pick_spots(caster: HeightfieldRayCaster, radius: float, seed: int = 7) -> list:
    """Spots with true average slopes spread over 0-25 deg."""
    rng = np.random.default_rng(seed)
    bins = {b: [] for b in range(5)}  # 0-5, 5-10, ... 20-25 deg
    while any(len(v) < -(-SPOTS // 5) for v in bins.values()):
        x, y = rng.uniform(-10.0, 10.0, 2)
        slope = math.degrees(math.acos(true_plane_normal(caster, x, y, radius)[2]))
        b = int(slope // 5)
        if b in bins and len(bins[b]) < -(-SPOTS // 5):
            bins[b].append((float(x), float(y), slope))
    spots = [s for b in sorted(bins) for s in bins[b]]
    return spots[:SPOTS]


def drop(x: float, y: float, height: float, euler_deg, use_camera: bool,
         caster: HeightfieldRayCaster, radius: float) -> dict:
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    sim = JumperSimulation(model, data, "lqr", landing_reference=use_camera)
    sim.model.opt.gravity[2] = -MOON_GRAVITY
    qa = model.jnt_qposadr[sim.ids.root_joint]
    ground = float(caster.height(np.array([x]), np.array([y]))[0])
    data.qpos[qa:qa + 3] = (x, y, ground + height)
    data.qpos[qa + 3:qa + 7] = euler_zyx_to_quat(*(math.radians(a) for a in euler_deg))
    mujoco.mj_forward(model, data)
    if sim.landing_reference is not None:
        sim.landing_reference.reset(float(data.time))
    while data.time < MAX_TIME and not sim.robot_touches_terrain():
        sim.step()
    body_z = data.xmat[model.jnt_bodyid[sim.ids.root_joint]].reshape(3, 3)[:, 2]
    truth = true_plane_normal(caster, float(data.qpos[qa]), float(data.qpos[qa + 1]), radius)
    result = {
        "x": x, "y": y, "height": height, "camera": use_camera,
        "touchdown_time": float(data.time),
        "true_slope_deg": math.degrees(math.acos(truth[2])),
        "misalignment_deg": math.degrees(math.acos(min(1.0, float(body_z @ truth)))),
    }
    if sim.landing_reference is not None:
        filt = sim.landing_reference.filter
        result["estimate_error_deg"] = math.degrees(
            math.acos(min(1.0, float(filt.normal @ truth))))
        result["frames_accepted"] = filt.accepted
        result["frames_rejected"] = filt.rejected
        result["reference_sigma_deg"] = math.degrees(filt.sigma)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    config = LandingReferenceConfig()
    probe = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    caster = HeightfieldRayCaster(probe)
    spots = pick_spots(caster, config.roi_radius)
    rng = np.random.default_rng(11)
    results = []
    for x, y, _ in spots:
        for height in HEIGHTS:
            euler = rng.uniform(-15.0, 15.0, 3)
            for use_camera in (False, True):
                results.append(drop(x, y, height, euler, use_camera, caster, config.roi_radius))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{args.label}.json").write_text(json.dumps(results, indent=1))

    print(f"\n=== {args.label}: misalignment to the ground plane at touchdown [deg] ===")
    print(f"{'slope':>6s} {'height':>6s} {'level ref':>9s} {'camera ref':>10s} {'est err':>7s} {'frames':>6s}")
    for level, cam in zip(results[0::2], results[1::2]):
        print(f"{cam['true_slope_deg']:6.1f} {cam['height']:6.1f} {level['misalignment_deg']:9.2f} "
              f"{cam['misalignment_deg']:10.2f} {cam['estimate_error_deg']:7.2f} {cam['frames_accepted']:6d}")
    for name, rows in (("level", results[0::2]), ("camera", results[1::2])):
        mis = np.array([r["misalignment_deg"] for r in rows])
        print(f"{name:7s} misalignment: median {np.median(mis):.2f}  p90 {np.percentile(mis, 90):.2f}  max {mis.max():.2f}")
    cams = results[1::2]
    for h in HEIGHTS:
        mis = [r["misalignment_deg"] for r in cams if r["height"] == h]
        lev = [r["misalignment_deg"] for r in results[0::2] if r["height"] == h]
        print(f"  from {h:4.1f} m: level median {np.median(lev):.2f} -> camera median {np.median(mis):.2f}")


if __name__ == "__main__":
    main()
