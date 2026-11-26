"""Run one Moon hop towards a target and log, per 1 ms step, what the
controller does (for render_hop_figure.py).

usage (from Sim_MuJoCo): python report_figures/simulation/scripts/record_hop.py OUT.pkl DX DY
DX, DY: the target relative to the spawn point [m]. The report's hop: 15 4.
"""
import pickle
import sys

import mujoco
import numpy as np

sys.path.insert(0, ".")
from jumper_sim.app import spawn_position  # noqa: E402
from jumper_sim.flight_phase import FlightPhase  # noqa: E402
from jumper_sim.simulation import JumperSimulation  # noqa: E402

dx, dy = float(sys.argv[2]), float(sys.argv[3])
model = mujoco.MjModel.from_xml_path("build/Jumping_Robot.xml")
data = mujoco.MjData(model)
sx, sy = spawn_position(model, data)
sim = JumperSimulation(model, data, "lqr", track_target=(sx + dx, sy + dy))
base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
while data.time < 6.0:  # settle and lean towards the target
    sim.step()
sim.handle_key("space")
t0 = float(data.time)
log = []
while data.time < t0 + 20.0:
    sim.step()
    phase = sim.phase
    fr, lr, tr = sim.flight_reference, sim.landing_reference, sim.trajectory
    log.append(dict(
        t=float(data.time) - t0, qpos=data.qpos.copy(), pos=data.xpos[base].copy(),
        phase=phase.name, airborne=bool(sim.estimated_airborne), contact=bool(sim.in_contact),
        cut_off=bool(tr.cut_off), ballistic=bool(tr.ballistic),
        tracking=bool(fr.tracking_target(phase)), accepted=int(lr.plane.accepted),
        ref=tuple(sim.reference_quaternion()),
        arc=(float(tr.arc_time) - t0, np.array(tr.arc_position, float), np.array(tr.arc_velocity, float))
        if tr.ballistic else None,
        spot=None if lr.visual is None else (lr.visual.anchor.copy(), lr.visual.measured_normal.copy(),
                                             bool(lr.visual.used)),
    ))
    if phase is FlightPhase.GROUND and t0 + 1.0 < data.time:
        break
pickle.dump(dict(log=log, target=sim.flight_reference.target_setting.copy(), start=log[0]["pos"],
                 gravity=float(sim.selected_gravity)), open(sys.argv[1], "wb"))
print("steps", len(log), "end", round(log[-1]["t"], 3))
