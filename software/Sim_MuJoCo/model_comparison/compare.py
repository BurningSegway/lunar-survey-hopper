"""MuJoCo against the report's nonlinear model (4.15) in zero gravity.

The robot floats free (no gravity, no contact) from an initial state
``x0 = [q1 q2 q3 q4 wx wy wz Ox Oy Oz]`` (quaternion scalar last, body rates
and wheel speeds relative to the body, as in the report and the MATLAB
script), with a wheel torque profile (``--profile``: ``step``, the torque held
for a while and then zero, or ``doublets``, the user's ``tau_profile.m``). The same start
and input are integrated through (4.15) (``cube_model.report_dynamics``,
RK45 at tight tolerances), and every state of both is saved and plotted
(``<label>_sdi.png``, Simulink Data Inspector style).

(4.15)'s input is ``tau = J_w Omega_dot``. With ``--input spin`` (default)
MuJoCo's wheel motors get the motor torque that realises it,
``tau + J_w w_dot`` (``cube_model.motor_torque``, at MuJoCo's state each
step); with ``--input motor`` they get ``tau`` itself, which shows what
reading (4.15)'s tau as the motor torque would cost.

The wheel bearing friction (MJCF damping) is not in (4.15), so it is turned
off unless ``--bearing-friction``. MuJoCo's wheel motors are ideal (as in
(4.15) and Simulink) unless ``--motor-limits`` gives them the bench-tested
motor's limits (``cube_model.WHEEL_MOTOR``). The jump foot is held wound and the legs
rest on their springs, as in flight.

``--view`` shows the MuJoCo run in a viewer window as it runs (real time).

Usage:
    python -m model_comparison --label default --view
    python -m model_comparison --label default
    python -m model_comparison --label spun --x0 0 0 0 1 0.5 -0.3 0.8 20 -10 15 \\
        --torque 0.02 -0.01 0.015 --torque-time 1 --duration 5
    python -m model_comparison --label doublets --profile doublets --duration 20 \
        --x0 0.29 0.11 -0.30 1 0.1 -0.2 0.15 0 2 0
    python -m model_comparison --label simulink_1khz --profile doublets --duration 20 \
        --matlab model_comparison/matlab/ode4_1khz.csv model_comparison/matlab/ode3_1khz.csv \
        --simulink-solver ode4 ode3
Writes model_comparison/results/<label>.csv and <label>_sdi.png/.pdf (one column
per --matlab run, or one against the model (4.15))
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
import math
from pathlib import Path
import time
from typing import NamedTuple

import mujoco
import numpy as np
from scipy.integrate import solve_ivp

from jumper_sim.controller import robot_inertia
from jumper_sim.cube_model import (
    SENSOR_CUBE, WHEEL_MOTOR, SensorCube, WheelMotor, motor_torque, report_dynamics,
)
from jumper_sim.simulation import JUMP_WINDUP_FORCE, find_model_ids

MODEL_PATH = Path(__file__).resolve().parents[1] / "build" / "Jumping_Robot.xml"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

STATE_NAMES = ("q1", "q2", "q3", "q4", "wx", "wy", "wz", "Omega_x", "Omega_y", "Omega_z")
DEFAULT_X0 = (0.0, 0.0, 0.0, 1.0, 0.5, -0.3, 0.8, 20.0, -10.0, 15.0)
DEFAULT_TORQUE = (0.02, -0.01, 0.015)  # N*m, (4.15)'s tau
# Base start position: above the terrain, so the viewer does not draw the robot
# sunk into it (no gravity, no contact: it does not change the motion).
START_POSITION = (0.0, 0.0, 2.0)


def free_floating_model(bearing_friction: bool = False, wheel_inertia: float | None = None
                        ) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """The robot in zero gravity with contacts off (and its wheels' J_w,i set)."""
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    model.opt.gravity[:] = 0.0
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    if not bearing_friction:
        for joint in find_model_ids(model).wheel_joints:
            model.dof_damping[model.jnt_dofadr[joint]] = 0.0
    if wheel_inertia is not None:
        set_wheel_inertia(model, wheel_inertia)
    return model, mujoco.MjData(model)


def set_wheel_inertia(model: mujoco.MjModel, wheel_inertia: float) -> None:
    """Each wheel's spin inertia J_w,i -> ``wheel_inertia``, keeping J_c.

    J_c counts the wheels locked, so the three wheels' change (each on its own
    axis, together isotropic) goes to the base instead.
    """
    ids = find_model_ids(model)
    changes = []
    for axis, joint in enumerate(ids.wheel_joints):
        inertia = model.body_inertia[model.jnt_bodyid[joint]]
        changes.append(inertia[axis] - wheel_inertia)
        inertia[axis] = wheel_inertia
    if max(changes) - min(changes) > 1e-12:
        raise ValueError("the wheels' spin inertias differ; J_c would not be kept")
    model.body_inertia[model.jnt_bodyid[ids.root_joint]] += changes[0]


def set_state(model: mujoco.MjModel, data: mujoco.MjData, state: np.ndarray,
              settle_steps: int = 500) -> None:
    """Start at ``state``, the foot held wound and settled into its stroke limit.

    Unsettled, the wind-up force pushes the foot ~1 mm into its soft limit on
    the first step and jolts the body rates by ~0.016 rad/s.
    """
    ids = find_model_ids(model)
    q_adr = model.jnt_qposadr[ids.root_joint]
    v_adr = model.jnt_dofadr[ids.root_joint]
    data.ctrl[ids.wind_up_actuator] = JUMP_WINDUP_FORCE  # foot held wound
    for _ in range(settle_steps):
        mujoco.mj_step(model, data)
    data.qpos[q_adr:q_adr + 3] = START_POSITION
    data.qpos[q_adr + 3:q_adr + 7] = np.roll(state[:4], 1)  # MuJoCo: scalar first
    data.qvel[:] = 0.0
    data.qvel[v_adr + 3:v_adr + 6] = state[4:7]  # free joint: body-frame rates
    for joint, speed in zip(ids.wheel_joints, state[7:]):
        data.qvel[model.jnt_dofadr[joint]] = speed
    data.time = 0.0
    mujoco.mj_forward(model, data)


def read_state(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    ids = find_model_ids(model)
    q_adr = model.jnt_qposadr[ids.root_joint]
    v_adr = model.jnt_dofadr[ids.root_joint]
    wheels = [data.qvel[model.jnt_dofadr[j]] for j in ids.wheel_joints]
    return np.concatenate([
        np.roll(data.qpos[q_adr + 3:q_adr + 7], -1),
        data.qvel[v_adr + 3:v_adr + 6],
        wheels,
    ])


class LiveView:
    """MuJoCo's viewer on the run (``--view``), paced to the wall clock."""

    FRAME_PERIOD = 1.0 / 60.0

    def __init__(self, speed: float = 1.0) -> None:
        self.speed = speed
        self.viewer = None

    def open(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        import mujoco.viewer

        self.viewer = mujoco.viewer.launch_passive(model, data)
        with self.viewer.lock():
            self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            self.viewer.cam.trackbodyid = model.jnt_bodyid[find_model_ids(model).root_joint]
            self.viewer.cam.distance = 1.5
            self.viewer.cam.elevation = -20.0
        self._start = time.perf_counter()
        self._next_frame = 0.0
        self.viewer.sync()

    def show(self, sim_time: float) -> None:
        """Draw a frame every 1/60 s of sim time, waiting for the wall clock."""
        if self.viewer is None or sim_time < self._next_frame or not self.viewer.is_running():
            return
        delay = self._start + sim_time / self.speed - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        self.viewer.sync()
        self._next_frame = sim_time + self.FRAME_PERIOD * self.speed

    def hold(self) -> None:
        """Keep the window open (on the last pose) until it is closed."""
        while self.viewer is not None and self.viewer.is_running():
            self.viewer.sync()
            time.sleep(self.FRAME_PERIOD)


@dataclass(frozen=True)
class TorqueProfile:
    """(4.15)'s wheel torque tau(t), piecewise constant.

    ``segments`` are ``(start time, tau)``, in time order; tau holds until the
    next start (zero before the first).
    """

    segments: tuple[tuple[float, tuple[float, float, float]], ...]

    def __call__(self, t: float) -> np.ndarray:
        tau = np.zeros(3)
        for start, value in self.segments:
            if t >= start:
                tau = np.asarray(value, dtype=float)
        return tau

    @property
    def switch_times(self) -> tuple[float, ...]:
        return tuple(start for start, _ in self.segments)


def step_profile(torque, torque_time: float) -> TorqueProfile:
    """``torque`` from 0 to ``torque_time``, then zero."""
    return TorqueProfile(((0.0, tuple(torque)), (torque_time, (0.0, 0.0, 0.0))))


def doublet_profile(amplitude: float = 0.02) -> TorqueProfile:
    """The user's MATLAB ``tau_profile.m``: a doublet on each axis (+a for 1 s,
    then -a for 1 s, at 1, 4 and 7 s), a multi-axis pulse (a, -0.75a, 0.5a) at
    10-11 s, then coasting with zero torque."""
    a = amplitude
    return TorqueProfile((
        (1.0, (a, 0.0, 0.0)), (2.0, (-a, 0.0, 0.0)), (3.0, (0.0, 0.0, 0.0)),
        (4.0, (0.0, a, 0.0)), (5.0, (0.0, -a, 0.0)), (6.0, (0.0, 0.0, 0.0)),
        (7.0, (0.0, 0.0, a)), (8.0, (0.0, 0.0, -a)), (9.0, (0.0, 0.0, 0.0)),
        (10.0, (a, -0.75 * a, 0.5 * a)), (11.0, (0.0, 0.0, 0.0)),
    ))


def run_mujoco(x0: np.ndarray, profile: TorqueProfile, duration: float,
               cube: SensorCube, spin_input: bool, bearing_friction: bool,
               view: LiveView | None = None, wheel_inertia: float | None = None,
               motor: WheelMotor | None = None) -> dict:
    """MuJoCo from ``x0``. ``motor``: the wheel motors' torque limits (the
    torque each motor delivers at its wheel's speed); None: ideal motors."""
    model, data = free_floating_model(bearing_friction, wheel_inertia)
    ids = find_model_ids(model)
    set_state(model, data, x0)
    if view is not None:
        view.open(model, data)
    steps = int(round(duration / model.opt.timestep))
    t = np.arange(steps + 1) * model.opt.timestep
    states = np.zeros((steps + 1, 10))
    applied = np.zeros((steps + 1, 3))
    for k in range(steps + 1):
        state = read_state(model, data)
        tau = profile(t[k])
        command = motor_torque(state, tau, cube) if spin_input else tau
        applied[k] = (command if motor is None
                      else motor.available_torque(command, state[7:]))
        states[k] = state
        if k == steps:
            break
        for actuator, value in zip(ids.wheel_actuators, applied[k]):
            data.ctrl[actuator] = value
        mujoco.mj_step(model, data)
        if view is not None:
            view.show(data.time)
    return {"t": t, "states": states, "motor": applied}


def run_model(x0: np.ndarray, profile: TorqueProfile, t: np.ndarray,
              cube: SensorCube) -> np.ndarray:
    """(4.15) integrated piecewise, restarted at each torque switch."""
    states = np.zeros((len(t), 10))
    state = x0.copy()
    bounds = sorted({0.0, t[-1], *(s for s in profile.switch_times if 0.0 < s < t[-1])})
    for start, end in zip(bounds[:-1], bounds[1:]):
        tau = profile(start)
        inside = (t >= start) & (t <= end)
        solution = solve_ivp(lambda _, x: report_dynamics(x, tau, cube), (start, end), state,
                             method="RK45", t_eval=t[inside], rtol=1e-10, atol=1e-12)
        states[inside] = solution.y.T
        state = solution.y[:, -1]
    return states


def attitude_difference_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation angle between two quaternion series [deg]."""
    dot = np.abs(np.sum(a / np.linalg.norm(a, axis=1, keepdims=True)
                        * b / np.linalg.norm(b, axis=1, keepdims=True), axis=1))
    return np.degrees(2.0 * np.arccos(np.clip(dot, -1.0, 1.0)))


def momentum_world(states: np.ndarray, cube: SensorCube) -> np.ndarray:
    """|H| in the inertial frame (constant in free flight)."""
    h = states[:, 4:7] @ cube.j_c.T + states[:, 7:] @ cube.j_w.T
    return np.linalg.norm(h, axis=1)


def write_csv(path, t, tau, motor, sim, model, attitude_diff) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "tau_x", "tau_y", "tau_z", "motor_x", "motor_y", "motor_z"]
                        + [f"sim_{n}" for n in STATE_NAMES]
                        + [f"model_{n}" for n in STATE_NAMES]
                        + ["attitude_diff_deg"])
        for row in zip(t, tau, motor, sim, model, attitude_diff):
            writer.writerow([f"{row[0]:.4f}"] + [f"{v:.9g}" for v in
                            np.concatenate([row[1], row[2], row[3], row[4], [row[5]]])])


def load_result(path) -> dict:
    """A run's saved results (``write_csv``), as ``compare`` returns them."""
    data = np.genfromtxt(path, delimiter=",", names=True)
    columns = lambda prefix, names: np.column_stack([data[f"{prefix}{n}"] for n in names])
    return {"t": data["t"], "tau": columns("tau_", "xyz"), "motor": columns("motor_", "xyz"),
            "sim": columns("sim_", STATE_NAMES), "model": columns("model_", STATE_NAMES),
            "attitude_diff": data["attitude_diff_deg"]}


# MATLAB's default line colours (its "lines" colormap).
MATLAB_COLORS = ("#0072BD", "#D95319", "#EDB120", "#7E2F8E")


# Simulink Data Inspector's line colours for x_val(1)..x_val(10).
SDI_COLORS = ("#E8178A", "#A2142F", "#2CA0E8", "#F07F35", "#B33CF0",
              "#64C832", "#F51DB4", "#EB3A1C", "#1FAD6B", "#0072BD")


def sample_rate_label(t: np.ndarray) -> str:
    """A run's step from its sample times: '1 kHz' (fixed) or 'variable step'."""
    steps = np.diff(t)
    if np.ptp(steps) > 1e-6 * steps.mean():
        return "variable step"
    rate = 1.0 / steps.mean()
    return f"{rate / 1000:.4g} kHz" if rate >= 1000 else f"{rate:.4g} Hz"


MUJOCO_INTEGRATORS = {
    mujoco.mjtIntegrator.mjINT_EULER: "semi-implicit Euler",
    mujoco.mjtIntegrator.mjINT_RK4: "RK4",
    mujoco.mjtIntegrator.mjINT_IMPLICIT: "implicit Euler",
    mujoco.mjtIntegrator.mjINT_IMPLICITFAST: "implicitfast Euler",
}


def mujoco_solver_label(model: mujoco.MjModel) -> str:
    """MuJoCo's integrator and rate, e.g. 'semi-implicit Euler, 1 kHz'."""
    name = MUJOCO_INTEGRATORS.get(mujoco.mjtIntegrator(model.opt.integrator),
                                  str(model.opt.integrator))
    return f"{name}, {sample_rate_label(np.array([0.0, model.opt.timestep]))}"


class Reference(NamedTuple):
    """What MuJoCo is compared with in one column of the SDI figure."""
    t: np.ndarray       # its own sample times
    states: np.ndarray  # (samples, 10)
    label: str          # "Simulink", "Model (4.15)"
    info: str = ""      # solver and rate, for the panel title


# The SDI figure is drawn at its printed size, so its text is the report's
# size (10-11 pt). One column: half the text width of an A4 landscape page
# with 2.5 cm margins (24.7 cm); more: the whole width, below it room for
# the caption.
SDI_COLUMN_SIZE = (12.0 / 2.54, 16.0 / 2.54)  # [in]
SDI_PAGE_SIZE = (24.7 / 2.54, 14.5 / 2.54)  # [in]


def plot_sdi(path, t, sim, references: list[Reference], tau: np.ndarray | None = None,
             sim_info: str = "") -> None:
    """Simulink Data Inspector style: all 10 states x_1..x_10 on one axes.

    One column per reference (a Simulink run, or the model (4.15)), top to
    bottom: the wheel torque input (with ``tau``), the reference's states (its
    own samples), MuJoCo's (same axes), and MuJoCo - reference (at the
    reference's samples). Each row shares its y-axis across the columns (the
    states' two rows one range), so the columns compare by eye. The x_i legend
    is drawn once, in a column left of the panels, the torque legend on the first
    torque panel's title line. Panel titles carry each run's solver and rate (``Reference.info``, ``sim_info``); no figure title (it goes in the
    report's caption). Written as ``path`` (PNG, 300 dpi) and the same name as
    PDF (vector, for LaTeX).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker

    columns = len(references)
    with plt.rc_context({"font.family": "sans-serif", "font.size": 10,
                         "axes.titlesize": 11, "axes.labelsize": 10,
                         "xtick.labelsize": 10, "ytick.labelsize": 10,
                         "legend.fontsize": 10, "axes.linewidth": 0.6,
                         "axes.edgecolor": "#262626", "grid.color": "#DFDFDF",
                         "grid.linewidth": 0.6, "lines.linewidth": 1.1,
                         "pdf.fonttype": 42}):
        ratios = ((1.4,) if tau is not None else ()) + (2.6, 2.6, 2.0)
        fig, grid = plt.subplots(len(ratios), columns,
                                 figsize=SDI_COLUMN_SIZE if columns == 1 else SDI_PAGE_SIZE,
                                 sharex=True, sharey="row", layout="constrained",
                                 gridspec_kw={"height_ratios": ratios}, squeeze=False)
        suffix = lambda info: f": {info}" if info else ""
        errors = [np.column_stack([np.interp(reference.t, t, sim[:, i]) for i in range(10)])
                  - reference.states for reference in references]
        # One power of ten for all error panels, in their titles (an axis
        # offset label would push one column's title out of line).
        peak = max(float(np.abs(error).max()) for error in errors)
        exponent = int(np.floor(np.log10(peak))) if peak > 0 else 0
        scale = f" [$\\times 10^{{{exponent}}}$]" if exponent else ""
        for column, reference, error in zip(grid.T, references, errors):
            axes = column
            if tau is not None:
                torque_ax, axes = column[0], column[1:]
                for i, (color, axis) in enumerate(zip(MATLAB_COLORS, "xyz")):
                    torque_ax.plot(t, tau[:, i], color=color, drawstyle="steps-post",
                                   label=f"$\\tau_{axis}$")
                torque_ax.set_title("Torque profile", fontweight="bold", loc="left")
            panels = ((axes[0], reference.t, reference.states,
                       f"{reference.label}{suffix(reference.info)}"),
                      (axes[1], t, sim, f"MuJoCo{suffix(sim_info)}"),
                      (axes[2], reference.t, error / 10.0**exponent,
                       f"Error: MuJoCo $-$ {reference.label}{scale}"))
            for ax, time, states, name in panels:
                for i, color in enumerate(SDI_COLORS):
                    ax.plot(time, states[:, i], color=color, label=f"$x_{{{i + 1}}}$")
                ax.set_title(name, fontweight="bold", loc="left")
            axes[2].set_xlabel("Time [s]")
        for ax in fig.axes:
            ax.grid(True)
            ax.set_xlim(0.0, t[-1])
            ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5, steps=(1, 2, 5, 10)))
        rows = grid[1:] if tau is not None else grid
        low = min(rows[0, 0].get_ylim()[0], rows[1, 0].get_ylim()[0])
        high = max(rows[0, 0].get_ylim()[1], rows[1, 0].get_ylim()[1])
        for ax in (rows[0, 0], rows[1, 0]):
            ax.set_ylim(low, high)
        # x_1..x_4 (the quaternion) have no unit, x_5..x_10 (rates) rad/s.
        for ax in (rows[0, 0], rows[1, 0]):
            ax.set_ylabel(r"$x_i$ [$-$, rad/s]")
        rows[2, 0].set_ylabel(r"$\Delta x_i$ [$-$, rad/s]")
        if tau is not None:
            torque_ax = grid[0, 0]
            torque_ax.set_ylabel(r"$\tau$ [N$\cdot$m]")
            peak = float(np.abs(tau).max()) or 1.0
            torque_ax.set_ylim(-1.5 * peak, 1.5 * peak)
            torque_ax.set_yticks((-peak, 0.0, peak))
            # The torque legend on the panel's title line (right of the title),
            # so it takes no room of its own.
            torque_ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=3,
                             frameon=False, borderpad=0.0, borderaxespad=0.15,
                             handlelength=1.2, columnspacing=0.8)
        # The states' legend once, in a column left of the panels: the panels
        # keep the full height.
        handles, names = rows[0, 0].get_legend_handles_labels()
        fig.legend(handles, names, loc="outside left center", ncol=1, frameon=False,
                   handlelength=1.4, labelspacing=0.5)
        fig.align_ylabels(grid[:, 0])
        fig.savefig(path, dpi=300)
        fig.savefig(Path(path).with_suffix(".pdf"))
        plt.close(fig)


def load_matlab(path) -> tuple[np.ndarray, np.ndarray]:
    """A MATLAB run's states: CSV ``t, q1..q4, wx..wz, Omega_x..Omega_z``, as
    ``matlab/export_states.m`` writes it from a Simulink timeseries."""
    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    return data[:, 0], data[:, 1:11]


def write_matlab_csv(path, t_ref, ref, t, sim, model) -> None:
    """On the MATLAB run's samples: its states, MuJoCo's and the model's
    (both interpolated from their 1 ms grid)."""
    at = lambda x: np.column_stack([np.interp(t_ref, t, x[:, i]) for i in range(10)])
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t"] + [f"matlab_{n}" for n in STATE_NAMES]
                        + [f"sim_{n}" for n in STATE_NAMES]
                        + [f"model_{n}" for n in STATE_NAMES])
        for row in np.column_stack([t_ref, ref, at(sim), at(model)]):
            writer.writerow([f"{v:.10g}" for v in row])


def compare(x0, profile: TorqueProfile, duration, cube=SENSOR_CUBE, spin_input=True,
            bearing_friction=False, view: LiveView | None = None,
            sim_wheel_inertia: float | None = None,
            motor: WheelMotor | None = None) -> dict:
    """Both runs from ``x0``; its quaternion is normalised (MuJoCo needs a unit one).

    ``sim_wheel_inertia`` sets MuJoCo's J_w,i (J_c kept); the MJCF's otherwise.
    ``motor`` limits MuJoCo's wheel motor torques (e.g. ``WHEEL_MOTOR``;
    None, the default: ideal); (4.15) has no limits.
    """
    x0 = np.asarray(x0, dtype=float).copy()
    x0[:4] /= np.linalg.norm(x0[:4])
    sim = run_mujoco(x0, profile, duration, cube, spin_input, bearing_friction, view,
                     sim_wheel_inertia, motor)
    t = sim["t"]
    model = run_model(x0, profile, t, cube)
    tau = np.array([profile(tk) for tk in t])
    return {"t": t, "tau": tau, "motor": sim["motor"], "sim": sim["states"], "model": model,
            "attitude_diff": attitude_difference_deg(sim["states"][:, :4], model[:, :4])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--label", required=True)
    parser.add_argument("--x0", type=float, nargs=10, default=None,
                        metavar=("q1", "q2", "q3", "q4", "wx", "wy", "wz", "Ox", "Oy", "Oz"),
                        help="initial state: quaternion (scalar last), body rates [rad/s], "
                             f"wheel speeds [rad/s] (default {DEFAULT_X0}, or the "
                             "--matlab run's first sample)")
    parser.add_argument("--profile", choices=("step", "doublets"), default="step",
                        help="step: --torque from 0 to --torque-time, then zero (default); "
                             "doublets: tau_profile.m, an x, y and z doublet of "
                             "+-0.02 N*m at 1, 4, 7 s, a multi-axis pulse at 10-11 s, "
                             "then zero")
    parser.add_argument("--amplitude", type=float, default=0.02,
                        help="the doublets' torque a [N*m] (default 0.02, tau_profile.m's)")
    parser.add_argument("--torque", type=float, nargs=3, default=DEFAULT_TORQUE,
                        metavar=("tx", "ty", "tz"),
                        help=f"(4.15)'s wheel torque [N*m] (default {DEFAULT_TORQUE})")
    parser.add_argument("--torque-time", type=float, default=1.0,
                        help="torque held from 0 to this time, then zero [s] (default 1)")
    parser.add_argument("--duration", type=float, default=5.0, help="[s] (default 5)")
    parser.add_argument("--input", choices=("spin", "motor"), default="spin",
                        help="spin: tau = J_w Omega_dot as in (4.15), MuJoCo's motors get "
                             "tau + J_w w_dot; motor: tau straight to MuJoCo's motors")
    parser.add_argument("--wheel-inertia", type=float, default=None,
                        help="J_w,i in both MuJoCo (J_c kept: the difference goes to the "
                             "base) and (4.15) [kg*m^2] (default the CAD value "
                             f"{SENSOR_CUBE.wheel_inertia}, which the MJCF is fitted to)")
    parser.add_argument("--model-wheel-inertia", type=float, default=None,
                        help="J_w,i in (4.15) only [kg*m^2], to see a mismatch")
    parser.add_argument("--matlab", type=Path, nargs="+", default=None, metavar="CSV",
                        help="MATLAB/Simulink runs' states (CSVs from "
                             "matlab/export_states.m) to compare MuJoCo with instead of "
                             "the model (4.15), one column each in <label>_sdi.png; "
                             "<label>_matlab_<csv name>.csv each")
    parser.add_argument("--simulink-solver", nargs="+", default=None, metavar="NAME",
                        help="the --matlab runs' solvers (e.g. ode4 ode3), in their order, "
                             "for the panel titles (the rate comes from the sample times)")
    parser.add_argument("--motor-limits", action="store_true",
                        help="give MuJoCo's wheel motors the bench-tested limits "
                             "(cube_model.WHEEL_MOTOR: torque limit, falling with wheel speed "
                             "while speeding the wheel up; default: ideal motors, as (4.15))")
    parser.add_argument("--torque-limit", type=float, default=None,
                        help=f"with --motor-limits, their torque limit [N*m] (default "
                             f"{WHEEL_MOTOR.torque_limit})")
    parser.add_argument("--bearing-friction", action="store_true",
                        help="keep MuJoCo's wheel bearing friction (not in (4.15))")
    parser.add_argument("--replot", action="store_true",
                        help="redraw <label>_sdi from the saved results/<label>.csv (and "
                             "the --matlab runs) instead of running MuJoCo and (4.15), "
                             "for layout changes")
    parser.add_argument("--view", action="store_true",
                        help="watch the MuJoCo run in a viewer window, in real time; it "
                             "stays open after the run until closed")
    parser.add_argument("--view-speed", type=float, default=1.0,
                        help="playback speed with --view (default 1 = real time)")
    args = parser.parse_args()

    sim_wheel = args.wheel_inertia
    model_wheel = args.model_wheel_inertia or args.wheel_inertia or SENSOR_CUBE.wheel_inertia
    cube = replace(SENSOR_CUBE, wheel_inertia=model_wheel)
    matlab = [load_matlab(path) for path in args.matlab or ()]
    solvers = list(args.simulink_solver or ())
    if len(solvers) not in (0, len(matlab)):
        parser.error("--simulink-solver needs one name per --matlab run")
    if args.x0 is None:
        args.x0 = list(matlab[0][1][0]) if matlab else list(DEFAULT_X0)
        if matlab:
            print("x0 from the first MATLAB run: " + " ".join(f"{v:.10g}" for v in args.x0))
    for path, (_, states) in zip(args.matlab or (), matlab):
        if not np.allclose(states[0], args.x0):
            print(f"warning: {path} starts from another x0: {states[0]}")
    view = LiveView(args.view_speed) if args.view else None
    if args.torque_limit is not None and not args.motor_limits:
        parser.error("--torque-limit needs --motor-limits")
    motor = replace(WHEEL_MOTOR, torque_limit=args.torque_limit or WHEEL_MOTOR.torque_limit) \
        if args.motor_limits else None
    profile = (doublet_profile(args.amplitude) if args.profile == "doublets"
               else step_profile(args.torque, args.torque_time))
    norm = float(np.linalg.norm(args.x0[:4]))
    if abs(norm - 1.0) > 1e-6:
        print(f"x0's quaternion has norm {norm:.4f}; normalised to "
              + " ".join(f"{v:.4f}" for v in np.asarray(args.x0[:4]) / norm))
    RESULTS_DIR.mkdir(exist_ok=True)
    stem = RESULTS_DIR / args.label
    if args.replot:
        result = load_result(stem.with_name(f"{args.label}.csv"))
    else:
        result = compare(args.x0, profile, args.duration, cube,
                         args.input == "spin", args.bearing_friction, view, sim_wheel, motor)
        keys = ("t", "tau", "motor", "sim", "model", "attitude_diff")
        write_csv(stem.with_name(f"{args.label}.csv"), *(result[k] for k in keys))
    sim_shown = sim_wheel or SENSOR_CUBE.wheel_inertia
    wheels = (f"J_w,i {model_wheel:g}" if sim_shown == model_wheel
              else f"J_w,i MuJoCo {sim_shown:g}, model {model_wheel:g}")
    motors = ("ideal wheel motors" if motor is None else
              f"wheel motors limited to {motor.torque_limit:g} N*m, falling above "
              f"{motor.knee_speed / (2 * math.pi):.2f} rev/s to 0 at "
              f"{motor.zero_torque_speed / (2 * math.pi):.2f} rev/s")
    profile_name = (f"doublets of {args.amplitude:g} N*m" if args.profile == "doublets"
                    else f"step of {' '.join(f'{v:g}' for v in args.torque)} N*m for "
                         f"{args.torque_time:g} s")
    caption = (f"zero g, {profile_name}, input {args.input}, {wheels}, {motors}"
               f"{', bearing friction' if args.bearing_friction else ''}: x0 = "
               + " ".join(f"{v:g}" for v in args.x0))
    model, data = free_floating_model(wheel_inertia=sim_wheel)
    sdi = stem.with_name(f"{args.label}_sdi.png")
    references, matlab_csvs = [], []
    for k, (path, (t_ref, ref)) in enumerate(zip(args.matlab or (), matlab)):
        solver = solvers[k] if solvers else None
        references.append(Reference(t_ref, ref, "Simulink", ", ".join(
            filter(None, (solver, sample_rate_label(t_ref))))))
        matlab_csvs.append(stem.with_name(f"{args.label}_matlab_{path.stem}.csv"))
        write_matlab_csv(matlab_csvs[-1], t_ref, ref, result["t"], result["sim"],
                         result["model"])
    if not references:
        references.append(Reference(result["t"], result["model"], "Eq. (4.15)", "RK45"))
    plot_sdi(sdi, result["t"], result["sim"], references, tau=result["tau"],
             sim_info=mujoco_solver_label(model))

    ids = find_model_ids(model)
    inertia = robot_inertia(model, ids.root_joint, ids.wheel_joints)
    print("MJCF at the flight pose: J diag", np.round(np.diag(inertia.rigid_body), 6),
          "wheel spin", np.round(inertia.wheel_spin, 6))
    print("(4.15): J_c diag", np.round(np.diag(cube.j_c), 6), "J_w,i", cube.wheel_inertia)
    error = np.abs(result["sim"] - result["model"])
    print(f"{'state':>8} {'max |sim - model|':>18} {'at end':>12}")
    for i, name in enumerate(STATE_NAMES):
        print(f"{name:>8} {error[:, i].max():18.3g} {error[-1, i]:12.3g}")
    print(f"attitude difference: max {result['attitude_diff'].max():.3g} deg, "
          f"end {result['attitude_diff'][-1]:.3g} deg")
    for name in ("sim", "model"):
        h = momentum_world(result[name], cube)
        print(f"|H| {name}: start {h[0]:.6g}, end {h[-1]:.6g} N*m*s")
    print(stem.with_name(f"{args.label}.csv"))
    for path, (t_ref, ref) in zip(args.matlab or (), matlab):
        at = lambda x: np.column_stack([np.interp(t_ref, result["t"], x[:, i]) for i in range(10)])
        for name, states in (("MuJoCo", result["sim"]), ("(4.15)", result["model"])):
            e = np.abs(at(states) - ref)
            print(f"max |{name} - {path.stem}| (on its samples): q {e[:, :4].max():.3g}, "
                  f"w {e[:, 4:7].max():.3g} rad/s, Omega {e[:, 7:].max():.3g} rad/s")
    for path in matlab_csvs:
        print(path)
    print(sdi)
    print(sdi.with_suffix(".pdf"))
    print(f"for the caption: {caption}")
    if view is not None:
        print("Close the viewer window to exit.")
        view.hold()


if __name__ == "__main__":
    main()
