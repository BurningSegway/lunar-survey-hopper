"""Fit build/Jumping_Robot.xml to the robot's design values (jumper_sim/cube_model.py).

The report's model gives the whole robot's inertia J_c (wheels locked, about
the CoM) and each wheel's spin inertia J_w,i (SENSOR_CUBE); the jump mechanism
is JUMP_SPRING_COUNT JUMP_SPRINGs in parallel; the robot weighs ROBOT_MASS
(the cube and the springs). This script writes them into the MJCF, so the
simulated plant is the report's model:

- each wheel: J_w,i about its spin axis, J_w,i/2 across it (a thin disc);
- jump_joint: the springs' combined stiffness; wound to full compression
  with the foot at the bottom of the cube, released to the springs' free
  length (the joint's range is the stroke), then wound back; a third of the
  springs' mass moves with the foot (armature). Where the springs sit and how
  long they are is not modelled (user's choice for now);
- base_link: the mass left after the wheels, legs and foot, and the full
  inertia tensor that makes the whole robot's inertia exactly diag(J_c),
  products of inertia included, in the model's start pose: the foot wound
  (q = 0) and the legs at rest, as in flight;
- the spawn pose: legs and foot SPAWN_CLEARANCE above the ground.

Run from software/Sim_MuJoCo after changing the values in cube_model.py:

    .venv/Scripts/python.exe build/fit_robot_model.py
"""

from __future__ import annotations

from pathlib import Path
import re
import sys

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jumper_sim.controller import robot_inertia  # noqa: E402
from jumper_sim.cube_model import (  # noqa: E402
    JUMP_SPRING,
    JUMP_SPRING_COUNT,
    ROBOT_MASS,
    SENSOR_CUBE,
)
from jumper_sim.simulation import _leg_feet, _required_id, find_model_ids  # noqa: E402

MODEL_PATH = ROOT / "build" / "Jumping_Robot.xml"
WHEEL_AXES = {"rw_x": 0, "rw_y": 1, "rw_z": 2}
FOOT_GEOM = "foot"
BODY_GEOM = "body_box"
JUMP_JOINT = "jump_joint"
# Light damping: the springs barely lose energy (100 N*s/m would take about
# half of it at take-off speed); it settles the foot after the re-wind.
JUMP_DAMPING = 2.0  # N*s/m
# Stiff ends of the stroke (the wound stop and the free length), so the
# wind-up does not push the foot into a soft joint limit.
JUMP_LIMIT_SOLREF = "0.002 1"
# The legs and the wound foot start this far above the spawn ground: starting
# in it, the firm foot contact would throw the robot up.
SPAWN_CLEARANCE = 0.01  # m


def _replace(xml: str, pattern: str, replacement: str, what: str, flags: int = 0) -> str:
    xml, count = re.subn(pattern, lambda m: m.group(1) + replacement, xml, count=1, flags=flags)
    if count != 1:
        raise ValueError(f"no {what} in {MODEL_PATH}")
    return xml


def _set_wheels(xml: str) -> str:
    for name, axis in WHEEL_AXES.items():
        diagonal = [SENSOR_CUBE.wheel_inertia / 2.0] * 3
        diagonal[axis] = SENSOR_CUBE.wheel_inertia
        xml = _replace(
            xml,
            rf'(<body name="{name}"[^>]*>\s*<inertial [^>]*diaginertia=")[^"]*',
            " ".join(f"{v:.7g}" for v in diagonal),
            f"inertial for wheel {name!r}",
        )
    return xml


def _set_base(xml: str, mass: float, inertia: np.ndarray) -> str:
    full = (inertia[0, 0], inertia[1, 1], inertia[2, 2],
            inertia[0, 1], inertia[0, 2], inertia[1, 2])
    return _replace(
        xml,
        r'(<body name="base_link"[^>]*>.*?)<inertial [^>]*/>',
        f'<inertial pos="0 0 0" mass="{mass:.7g}" fullinertia="'
        + " ".join(f"{v:.9g}" for v in full) + '"/>',
        "inertial for base_link",
        re.DOTALL,
    )


def leg_tip_height(model: mujoco.MjModel) -> float:
    """Lowest point of the legs (at rest) in the root body frame."""
    ids = find_model_ids(model)
    tips = []
    for joint, foot in zip(ids.leg_joints, _leg_feet(model, ids.leg_joints)):
        geom = int(np.flatnonzero(model.geom_bodyid == model.jnt_bodyid[joint])[0])
        tips.append(foot[2] - model.geom_size[geom][0])
    return float(min(tips))


def wound_position(model: mujoco.MjModel) -> float:
    """jump_joint position with the foot's bottom at the bottom of the cube."""
    box = _required_id(model, mujoco.mjtObj.mjOBJ_GEOM, BODY_GEOM)
    cube_bottom = model.geom_pos[box][2] - model.geom_size[box][2]
    geom = _required_id(model, mujoco.mjtObj.mjOBJ_GEOM, FOOT_GEOM)
    body = int(model.geom_bodyid[geom])
    joint = _required_id(model, mujoco.mjtObj.mjOBJ_JOINT, JUMP_JOINT)
    if not np.allclose(model.jnt_axis[joint], (0.0, 0.0, -1.0)):
        raise ValueError("jump_joint must slide along -z")
    foot_bottom = model.body_pos[body][2] + model.geom_pos[geom][2] - model.geom_size[geom][0]
    return float(foot_bottom - cube_bottom)


def _set_jump_joint(xml: str, model: mujoco.MjModel) -> str:
    """Wound position at q = 0 (where the model starts), the stroke above it."""
    joint = _required_id(model, mujoco.mjtObj.mjOBJ_JOINT, JUMP_JOINT)
    foot_body = int(model.jnt_bodyid[joint])
    x, y, z = model.body_pos[foot_body]
    # The joint slides along -z: q = wound is the body moved down by wound.
    position = f"{x:.6g} {y:.6g} {z - wound_position(model):.6g}"
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, foot_body)
    xml = _replace(xml, rf'(<body name="{name}" )pos="[^"]*"', f'pos="{position}"', f"body {name!r}")
    stroke = JUMP_SPRING.stroke
    return _replace(
        xml,
        rf'(<joint name="{JUMP_JOINT}" )[^>]*/>',
        f'type="slide" axis="0 0 -1" '
        f'stiffness="{JUMP_SPRING_COUNT * JUMP_SPRING.stiffness:.6g}" '
        f'damping="{JUMP_DAMPING:g}" springref="{stroke:.6g}" limited="true" '
        f'range="0 {stroke:.6g}" solreflimit="{JUMP_LIMIT_SOLREF}" '
        f'armature="{JUMP_SPRING_COUNT * JUMP_SPRING.mass / 3.0:.6g}"/>',
        "jump_joint",
    )


def _set_spawn(xml: str, model: mujoco.MjModel) -> str:
    """Start the robot SPAWN_CLEARANCE above the spawn ground (z = 0)."""
    height = SPAWN_CLEARANCE - leg_tip_height(model)
    return _replace(xml, r'(<body name="base_link" )pos="[^"]*"', f'pos="0 0 {height:.4g}"', "base_link pos")


def fit() -> None:
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    ids = find_model_ids(model)
    base = int(model.jnt_bodyid[ids.root_joint])
    others = robot_inertia(model, ids.root_joint, ids.wheel_joints).mass - model.body_mass[base]
    base_mass = ROBOT_MASS - others
    if base_mass <= 0.0:
        raise ValueError(f"robot mass {ROBOT_MASS} kg is below its parts' {others:.3f} kg")
    # First pass: wheels, jump joint, base mass and a placeholder base inertia.
    placeholder = np.diag([0.01, 0.01, 0.01])
    xml = _set_wheels(MODEL_PATH.read_text(encoding="utf-8"))
    xml = _set_jump_joint(xml, model)
    xml = _set_base(xml, base_mass, placeholder)
    xml = _set_spawn(xml, model)
    MODEL_PATH.write_text(xml, encoding="utf-8")

    # The base's inertia adds linearly: swap the placeholder for what is missing.
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    total = np.array(robot_inertia(model, ids.root_joint, ids.wheel_joints).rigid_body)
    base_inertia = SENSOR_CUBE.j_c - (total - placeholder)
    if np.any(np.linalg.eigvalsh(base_inertia) <= 0.0):
        raise ValueError(f"J_c is too small for the rest of the robot: base would need {base_inertia}")
    MODEL_PATH.write_text(_set_base(xml, base_mass, base_inertia), encoding="utf-8")

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    result = robot_inertia(model, ids.root_joint, ids.wheel_joints)
    print(f"base_link: {base_mass:.4f} kg, inertia\n{np.round(base_inertia, 7)}")
    print(f"robot in flight: {result.mass:.4f} kg, J_c\n{np.round(np.array(result.rigid_body), 7)}")
    print(f"wheel spin inertia: {result.wheel_spin}")
    print(f"jump springs: {JUMP_SPRING_COUNT} x {JUMP_SPRING.stiffness:g} N/m, wound at "
          f"q = 0 with the foot at the cube's bottom ({wound_position(model) * 1000:+.1f} mm), stroke {JUMP_SPRING.stroke} m, "
          f"{JUMP_SPRING_COUNT * JUMP_SPRING.energy:.1f} J")


if __name__ == "__main__":
    fit()
