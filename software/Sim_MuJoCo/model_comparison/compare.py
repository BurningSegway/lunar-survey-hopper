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
off unless ``--bearing-friction``. The jump foot is held wound and the legs
rest on their springs, as in flight.

``--view`` shows the MuJoCo run in a viewer window as it runs (real time).

Usage:
    python -m model_comparison --label default --view
    python -m model_comparison --label default
    python -m model_comparison --label spun --x0 0 0 0 1 0.5 -0.3 0.8 20 -10 15 \\
        --torque 0.02 -0.01 0.015 --torque-time 1 --duration 5
    python -m model_comparison --label doublets --profile doublets --duration 20 \
        --x0 0.29 0.11 -0.30 1 0.1 -0.2 0.15 0 2 0
Writes model_comparison/results/<label>.csv and <label>_sdi.png
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
from pathlib import Path
import time

import mujoco
import numpy as np
from scipy.integrate import solve_ivp

from jumper_sim.controller import robot_inertia
from jumper_sim.cube_model import SENSOR_CUBE, SensorCube, motor_torque, report_dynamics
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
               view: LiveView | None = None, wheel_inertia: float | None = None) -> dict:
    model, data = free_floating_model(bearing_friction, wheel_inertia)
    ids = find_model_ids(model)
    set_state(model, data, x0)
    if view is not None:
        view.open(model, data)
    steps = int(round(duration / model.opt.timestep))
    t = np.arange(steps + 1) * model.opt.timestep
    states = np.zeros((steps + 1, 10))
    motor = np.zeros((steps + 1, 3))
    for k in range(steps + 1):
        state = read_state(model, data)
        tau = profile(t[k])
        motor[k] = motor_torque(state, tau, cube) if spin_input else tau
        states[k] = state
        if k == steps:
            break
        for actuator, value in zip(ids.wheel_actuators, motor[k]):
            data.ctrl[actuator] = value
        mujoco.mj_step(model, data)
        if view is not None:
            view.show(data.time)
    return {"t": t, "states": states, "motor": motor}


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


def plot_sdi(path, t, sim, t_ref, ref, ref_label: str = "Simulink",
             title: str | None = None, tau: np.ndarray | None = None,
             ref_info: str = "", sim_info: str = "") -> None:
    """Simulink Data Inspector style: all 10 states x_1..x_10 on one axes.

    Top (with ``tau``): the wheel torque input on MuJoCo's samples; then the
    reference run (a Simulink run, or the model (4.15); its own samples);
    MuJoCo, same axes; MuJoCo - reference, at the reference's samples.
    ``ref_info``/``sim_info`` (solver, rate) go in the panel titles.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    error = np.column_stack([np.interp(t_ref, t, sim[:, i]) for i in range(10)]) - ref
    with plt.rc_context({"font.family": "sans-serif", "font.size": 10,
                         "axes.linewidth": 0.6, "axes.edgecolor": "#262626",
                         "grid.color": "#DFDFDF", "grid.linewidth": 0.8}):
        ratios = ((1.5,) if tau is not None else ()) + (3, 3, 2)
        fig, axes = plt.subplots(len(ratios), 1, figsize=(12, 12 + 2 * (tau is not None)),
                                 sharex=True, gridspec_kw={"height_ratios": ratios})
        if tau is not None:
            torque_ax, axes = axes[0], axes[1:]
            for i, (color, axis) in enumerate(zip(MATLAB_COLORS, "xyz")):
                torque_ax.plot(t, tau[:, i], color=color, linewidth=1.5,
                               drawstyle="steps-post", label=f"$\\tau_{axis}$")
            torque_ax.set_title("Torque profile (wheel torque input)", fontweight="bold",
                                loc="left")
            torque_ax.set_ylabel(r"$\tau$ [N$\cdot$m]")
            torque_ax.legend(loc="upper right", ncol=3, frameon=True, edgecolor="#262626",
                             fancybox=False, fontsize=9)
            torque_ax.grid(True)
            torque_ax.set_xlim(0.0, t[-1])
        suffix = lambda info: f": {info}" if info else ""
        panels = ((axes[0], t_ref, ref, f"{ref_label}{suffix(ref_info)}"),
                  (axes[1], t, sim, f"MuJoCo simulation{suffix(sim_info)}"),
                  (axes[2], t_ref, error, f"Error: MuJoCo $-$ {ref_label}"))
        for ax, time, states, name in panels:
            for i, color in enumerate(SDI_COLORS):
                ax.plot(time, states[:, i], color=color, linewidth=1.5,
                        label=f"$x_{{{i + 1}}}$")
            ax.set_title(name, fontweight="bold", loc="left")
            ax.grid(True)
            ax.set_xlim(0.0, t[-1])
        low = min(axes[0].get_ylim()[0], axes[1].get_ylim()[0])
        high = max(axes[0].get_ylim()[1], axes[1].get_ylim()[1])
        for ax in axes[:2]:
            ax.set_ylim(low, high)
        axes[2].set_xlabel("Time [s]")
        handles, names = axes[0].get_legend_handles_labels()
        fig.legend(handles, names, loc="upper center", ncol=10, frameon=False,
                   fontsize=9, handlelength=1.2, columnspacing=1.0,
                   bbox_to_anchor=(0.5, 0.995 if not title else 0.97))
        if title:
            fig.suptitle(title, y=0.995)
        fig.tight_layout(rect=(0, 0, 1, 0.965 if not title else 0.95))
        fig.savefig(path, dpi=120)
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
            sim_wheel_inertia: float | None = None) -> dict:
    """Both runs from ``x0``; its quaternion is normalised (MuJoCo needs a unit one).

    ``sim_wheel_inertia`` sets MuJoCo's J_w,i (J_c kept); the MJCF's otherwise.
    """
    x0 = np.asarray(x0, dtype=float).copy()
    x0[:4] /= np.linalg.norm(x0[:4])
    sim = run_mujoco(x0, profile, duration, cube, spin_input, bearing_friction, view,
                     sim_wheel_inertia)
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
    parser.add_argument("--matlab", type=Path, default=None,
                        help="a MATLAB run's states (CSV from matlab/export_states.m) to "
                             "compare MuJoCo with instead of the model (4.15): "
                             "<label>_matlab.csv, and the run in <label>_sdi.png")
    parser.add_argument("--simulink-solver", default=None, metavar="NAME",
                        help="the --matlab run's solver (e.g. ode4), for the plot titles "
                             "(its rate comes from its sample times)")
    parser.add_argument("--bearing-friction", action="store_true",
                        help="keep MuJoCo's wheel bearing friction (not in (4.15))")
    parser.add_argument("--view", action="store_true",
                        help="watch the MuJoCo run in a viewer window, in real time; it "
                             "stays open after the run until closed")
    parser.add_argument("--view-speed", type=float, default=1.0,
                        help="playback speed with --view (default 1 = real time)")
    args = parser.parse_args()

    sim_wheel = args.wheel_inertia
    model_wheel = args.model_wheel_inertia or args.wheel_inertia or SENSOR_CUBE.wheel_inertia
    cube = replace(SENSOR_CUBE, wheel_inertia=model_wheel)
    matlab = load_matlab(args.matlab) if args.matlab else None
    if args.x0 is None:
        args.x0 = list(matlab[1][0]) if matlab is not None else list(DEFAULT_X0)
        if matlab is not None:
            print("x0 from the MATLAB run: " + " ".join(f"{v:.10g}" for v in args.x0))
    view = LiveView(args.view_speed) if args.view else None
    profile = (doublet_profile() if args.profile == "doublets"
               else step_profile(args.torque, args.torque_time))
    norm = float(np.linalg.norm(args.x0[:4]))
    if abs(norm - 1.0) > 1e-6:
        print(f"x0's quaternion has norm {norm:.4f}; normalised to "
              + " ".join(f"{v:.4f}" for v in np.asarray(args.x0[:4]) / norm))
    result = compare(args.x0, profile, args.duration, cube,
                     args.input == "spin", args.bearing_friction, view, sim_wheel)
    RESULTS_DIR.mkdir(exist_ok=True)
    stem = RESULTS_DIR / args.label
    keys = ("t", "tau", "motor", "sim", "model", "attitude_diff")
    write_csv(stem.with_suffix(".csv"), *(result[k] for k in keys))
    sim_shown = sim_wheel or SENSOR_CUBE.wheel_inertia
    wheels = (f"J_w,i {model_wheel:g}" if sim_shown == model_wheel
              else f"J_w,i MuJoCo {sim_shown:g}, model {model_wheel:g}")
    title = (f"zero g, {args.profile} torque, input {args.input}, {wheels}"
             f"{', bearing friction' if args.bearing_friction else ''}: x0 = "
             + " ".join(f"{v:g}" for v in args.x0))
    model, data = free_floating_model(wheel_inertia=sim_wheel)
    sdi = stem.with_name(f"{args.label}_sdi.png")
    if matlab is not None:
        t_ref, ref = matlab
        ref_label = "Simulink"
        ref_info = ", ".join(filter(None, (args.simulink_solver, sample_rate_label(t_ref))))
        write_matlab_csv(stem.with_name(f"{args.label}_matlab.csv"), t_ref, ref,
                         result["t"], result["sim"], result["model"])
    else:
        t_ref, ref = result["t"], result["model"]
        ref_label = "Model (4.15)"
        ref_info = f"RK45 (rtol 1e-10), sampled at {sample_rate_label(t_ref)}"
    plot_sdi(sdi, result["t"], result["sim"], t_ref, ref, ref_label=ref_label, title=title,
             tau=result["tau"], ref_info=ref_info, sim_info=mujoco_solver_label(model))

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
    print(stem.with_suffix(".csv"))
    if matlab is not None:
        t_ref, ref = matlab
        at = lambda x: np.column_stack([np.interp(t_ref, result["t"], x[:, i]) for i in range(10)])
        for name, states in (("MuJoCo", result["sim"]), ("(4.15)", result["model"])):
            e = np.abs(at(states) - ref)
            print(f"max |{name} - MATLAB| (on MATLAB's samples): q {e[:, :4].max():.3g}, "
                  f"w {e[:, 4:7].max():.3g} rad/s, Omega {e[:, 7:].max():.3g} rad/s")
    if matlab is not None:
        print(stem.with_name(f"{args.label}_matlab.csv"))
    print(sdi)
    if view is not None:
        print("Close the viewer window to exit.")
        view.hold()


if __name__ == "__main__":
    main()
