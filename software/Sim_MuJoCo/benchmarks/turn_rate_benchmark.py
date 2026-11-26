"""Turn-rate test: how fast the robot turns about one axis in flight.

The robot jumps straight up (Moon). Two modes:

``--mode controller`` (default): the LQR turns the robot. From the jump
command the controller holds the attitude the robot jumped with (instead of
the hop reference); ``START_DELAY`` (1 s) after the cutoff its reference
steps by an angle about the tested axis (``--angles``, default 170, 90, 45,
30, 20 and 10 deg). The reference is the step's end attitude straight away (no slew):
a reference several full turns away is the same quaternion as one under
180 deg away (the error quaternion takes the shorter way), so 170 deg is
the largest step the controller sees. Each step is watched for
``WINDOW`` (1 s), then held until the body has settled (``Settle``), and
the next step goes back the other way, so every angle is turned in both
directions. Per step:

* the angle turned after the 1 s window;
* the time until the attitude stays within 5 and 2 deg of the reference;
* overshoot, peak body rate, peak wheel torque (share of the motor limit)
  and peak wheel speed;
* seconds per degree: the time to within 2 deg over the step angle, and
  1 s over the angle turned in the first second (only meaningful for a step
  the window does not finish).

The turn time is close to a fixed time plus a time per degree, so the
steps' times to within 5 and 2 deg are fitted (least squares, both
directions) with t = a + k * theta (``fit_turn_time``).

``--gain design`` is the flight controller as it is (Bryson's bounds of the
user's script: 0.26 N*m, 60 deg/s; its output is clipped at the motor's
1.94 N*m); ``--gain full`` designs the same LQI with the torque bound at the
motor limit (``LqrLimits.max_torque``), for this test only. Default: both.

``--mode open-loop``: no controller. 1 s after the cutoff the axis's wheel
runs at full torque for ``PULSE`` (1 s), then back at full torque until it
is at its starting speed (the body at rest again), then the same the other
way, which brings the body back (``TurnManoeuvre``): each direction spun
from rest. Per direction: peak rate and the time to it, the starting
acceleration (first 50 ms), the rest-to-rest angle and time; the slower
direction's rate and acceleration, the residual and the cross-axis rate,
and bang-bang turn times from them (2 sqrt(theta / alpha) when
theta < omega^2 / alpha, else theta / omega + omega / alpha).

In both modes another axis or step runs in the same jump while the latched
arc leaves time for it, else in a jump of its own. Wheel torques go through
the sim's limits (1.94 N*m, 20 rev/s), or with ``--motor bench`` also
through the bench-tested motor (``cube_model.WHEEL_MOTOR``: 0.43 N*m,
falling with speed while speeding up). All from the true body state.

Usage:
    python -m benchmarks.turn_rate_benchmark --label turn_z --view
    python -m benchmarks.turn_rate_benchmark --label turn_z_design --gain design --angles 170
    python -m benchmarks.turn_rate_benchmark --label pulse_xyz --mode open-loop --axes x y z
    python -m benchmarks.turn_rate_benchmark --label pulse_bench --mode open-loop --motor bench
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field, replace
import json
import math
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from jumper_sim.controller import (
    LqrController,
    LqrLimits,
    limit_wheel_command,
    quaternion_error,
    wheel_effective_inertia,
)
from jumper_sim.cube_model import WHEEL_MOTOR
from jumper_sim.lidar import LIVOX_MID360
from jumper_sim.simulation import MOON_GRAVITY, JumperSimulation

from .lqr_benchmark import MODEL_PATH, RESULTS_DIR

AXES = ("x", "y", "z")
GAINS = ("design", "full")
START_DELAY = 1.0  # s after the jump's cutoff
PULSE = 1.0  # s of full torque, each way (open loop)
WINDOW = 1.0  # s a step is watched before it may settle (controller)
ANGLES = (170.0, 90.0, 45.0, 30.0, 20.0, 10.0)  # deg, controller steps
JUMP_AT = 2.0  # s of sim time after a reset (the robot has settled)
SETTLE_AFTER = 1.0  # s after the landing before the jump's run ends
MARGIN = 0.5  # s of flight left over after a manoeuvre in the same jump
TRACE_EVERY = 10  # steps (100 Hz) per stored trace sample (open loop)
SYNC_PERIOD = 1.0 / 60.0  # s of wall time between viewer syncs


class TurnManoeuvre:
    """Open-loop wheel command for one axis (pure logic: speed in, sign out).

    ``update(time, wheel_speed)`` returns -1, 0 or +1 (times full torque).
    The wheel is first driven in ``direction``.
    """

    STEPS = ("pulse", "stop", "reverse", "return", "done")

    def __init__(self, start_time: float, start_wheel_speed: float, direction: float = 1.0,
                 pulse: float = PULSE) -> None:
        self.start_time = start_time
        self.start_wheel_speed = start_wheel_speed
        self.direction = 1.0 if direction >= 0.0 else -1.0
        self.pulse = pulse
        self.step = "pulse"
        self.step_started = start_time
        self.step_times: dict[str, float] = {"pulse": start_time}

    def _next(self, time: float) -> None:
        self.step = self.STEPS[self.STEPS.index(self.step) + 1]
        self.step_started = time
        self.step_times[self.step] = time

    def update(self, time: float, wheel_speed: float) -> float:
        s = self.direction
        relative = s * (wheel_speed - self.start_wheel_speed)  # > 0: spun in direction
        if self.step == "pulse" and time - self.step_started >= self.pulse - 1e-9:
            self._next(time)
        if self.step == "stop" and relative <= 0.0:
            self._next(time)
        if self.step == "reverse" and time - self.step_started >= self.pulse - 1e-9:
            self._next(time)
        if self.step == "return" and relative >= 0.0:
            self._next(time)
        return {"pulse": s, "stop": -s, "reverse": -s, "return": s, "done": 0.0}[self.step]

    @property
    def done(self) -> bool:
        return self.step == "done"


class Settle:
    """Whether a held step has come to rest (pure logic).

    Done once ``window`` has passed and the attitude error has stayed below
    ``max_error`` [deg] with the body rate below ``max_rate`` [deg/s] for
    ``hold`` [s], or at ``timeout`` [s] after the step.
    """

    def __init__(self, start_time: float, window: float = WINDOW, max_error: float = 1.0,
                 max_rate: float = 3.0, hold: float = 0.1, timeout: float = 3.0) -> None:
        self.start_time = start_time
        self.window, self.max_error, self.max_rate = window, max_error, max_rate
        self.hold, self.timeout = hold, timeout
        self._quiet_since: float | None = None
        self.timed_out = False

    def update(self, time: float, error_deg: float, rate_deg_s: float) -> bool:
        if error_deg < self.max_error and rate_deg_s < self.max_rate:
            if self._quiet_since is None:
                self._quiet_since = time
        else:
            self._quiet_since = None
        elapsed = time - self.start_time
        if elapsed >= self.timeout:
            self.timed_out = True
            return True
        return (elapsed >= self.window and self._quiet_since is not None
                and time - self._quiet_since >= self.hold - 1e-9)


@dataclass
class AxisRun:
    axis: int
    manoeuvre: TurnManoeuvre
    start_attitude: np.ndarray
    start_rate: np.ndarray  # body frame, rad/s (the take-off spin)
    wheel_axis: np.ndarray  # body frame
    trace: list = field(default_factory=list)  # (t, rate about axis, angle about axis, wheel, cmd)
    cross_peak: float = 0.0


@dataclass
class StepRun:
    """One controller step: the reference turned by ``angle`` [rad] about the axis."""

    axis: int
    angle: float  # signed, about the body axis
    start_time: float
    start_reference: np.ndarray
    target: np.ndarray
    wheel_axis: np.ndarray  # body frame
    settle: Settle
    trace: list = field(default_factory=list)  # (t, rate about axis, error deg, torque, wheel)


class TurnRateSimulation(JumperSimulation):
    """The hop process with the reference (controller mode) or the wheel
    commands (open loop) replaced in flight."""

    def __init__(self, *args, motor: str = "sim", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.motor = motor
        self.wheel_commands = [0.0, 0.0, 0.0]  # -1 / 0 / +1, set by the open-loop test
        self.override = False  # open loop: the wheel commands replace the controller
        self.reference_override: tuple[float, float, float, float] | None = None

    def reference_quaternion(self) -> tuple[float, float, float, float]:
        if self.reference_override is not None:
            return self.reference_override
        return super().reference_quaternion()

    def _apply_controls(self) -> None:
        super()._apply_controls()
        tuning = self.tuning
        full = wheel_effective_inertia(tuning) * tuning.max_wheel_acceleration
        for k, (actuator, joint) in enumerate(zip(self.ids.wheel_actuators, self.ids.wheel_joints)):
            speed = float(self.data.qvel[self.model.jnt_dofadr[joint]])
            if self.override:
                command = self.wheel_commands[k] * (
                    WHEEL_MOTOR.torque_limit if self.motor == "bench" else full)
                torque = limit_wheel_command(self.model, self.data, joint, command, tuning)
            else:
                torque = float(self.data.ctrl[actuator])
            if self.motor == "bench":
                torque = float(WHEEL_MOTOR.available_torque(torque, speed))
            self.data.ctrl[actuator] = torque


def body_state(sim: JumperSimulation) -> tuple[np.ndarray, np.ndarray]:
    qa = sim.model.jnt_qposadr[sim.ids.root_joint]
    da = sim.model.jnt_dofadr[sim.ids.root_joint]
    return (sim.data.qpos[qa + 3: qa + 7].copy(), sim.data.qvel[da + 3: da + 6].copy())


def wheel_axis_in_body(sim: JumperSimulation, k: int) -> np.ndarray:
    base = sim.model.jnt_bodyid[sim.ids.root_joint]
    world = sim.data.xaxis[sim.ids.wheel_joints[k]]
    axis = sim.data.xmat[base].reshape(3, 3).T @ world
    return axis / np.linalg.norm(axis)


def wheel_speed(sim: JumperSimulation, k: int) -> float:
    return float(sim.data.qvel[sim.model.jnt_dofadr[sim.ids.wheel_joints[k]]])


def rotated(q: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    """``q`` turned by ``angle`` [rad] about the body-frame ``axis``."""
    turn = np.array([math.cos(angle / 2), *(math.sin(angle / 2) * axis)])
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, np.asarray(q, float), turn)
    return out / np.linalg.norm(out)


def attitude_distance(q1, q2) -> float:
    """Angle [deg] between two attitudes."""
    return math.degrees(2.0 * math.acos(min(1.0, abs(float(np.dot(q1, q2))))))


def flight_time_left(sim: JumperSimulation, launch_z: float) -> float:
    """Time until the latched arc is back at the launch height [s]."""
    tracker = sim.trajectory
    if not tracker.ballistic:
        return 0.0
    z0, vz = float(tracker.arc_position[2]), float(tracker.arc_velocity[2])
    g = MOON_GRAVITY if sim.selected_gravity is None else sim.selected_gravity
    disc = vz * vz + 2.0 * g * (z0 - launch_z)
    t_land = tracker.arc_time + (vz + math.sqrt(max(disc, 0.0))) / g
    return t_land - float(sim.data.time)


class Pacer:
    """Optional MuJoCo viewer, synced at 60 Hz with the sim paced to the wall clock."""

    def __init__(self, sim: JumperSimulation, view: bool) -> None:
        self.data = sim.data
        self.viewer = self.follower = None
        if view:
            from jumper_sim.app import RobotFollower
            self.viewer = mujoco.viewer.launch_passive(sim.model, sim.data)
            self.follower = RobotFollower(sim.model.jnt_bodyid[sim.ids.root_joint])
            with self.viewer.lock():
                self.follower.start(self.viewer, sim.data)
                self.viewer.cam.distance = 6.0
        self.restart()

    def restart(self) -> None:
        self.wall_anchor, self.sim_anchor = time.perf_counter(), float(self.data.time)
        self.next_sync = 0.0

    def after_step(self) -> bool:
        """False once the viewer window was closed."""
        if self.viewer is None:
            return True
        if not self.viewer.is_running():
            return False
        if float(self.data.time) - self.sim_anchor >= self.next_sync:
            with self.viewer.lock():
                self.follower.update(self.viewer, self.data)
            self.viewer.sync()
            self.next_sync += SYNC_PERIOD
            lag = (float(self.data.time) - self.sim_anchor) - (time.perf_counter() - self.wall_anchor)
            if lag > 0:
                time.sleep(lag)
        return True

    def close(self) -> None:
        if self.viewer is not None:
            self.viewer.close()


def make_sim(motor: str, gain: str | None = None) -> TurnRateSimulation:
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    sim = TurnRateSimulation(model, data, "lqr", motor=motor,
                             lidar=replace(LIVOX_MID360, point_fraction=0.01))
    if gain == "full":
        full = wheel_effective_inertia(sim.tuning) * sim.tuning.max_wheel_acceleration
        sim.controller = LqrController(sim.tuning, LqrLimits(max_torque=full))
    return sim


def sim_step(sim: JumperSimulation) -> None:
    sim.step()
    for event in sim.flight_events:
        print(f"  {float(sim.data.time):6.3f} s  {event}")
    sim.flight_events.clear()


# ------------------------------------------------------------------ controller

def summarise_step(run: StepRun, max_torque: float) -> dict:
    tr = np.array(run.trace)
    t, rate, error, torque, wheel = tr[:, 0], tr[:, 1], tr[:, 2], tr[:, 3], tr[:, 4]
    sign = 1.0 if run.angle >= 0 else -1.0
    rate = sign * rate  # the step's direction positive
    turned = np.concatenate([[0.0], np.cumsum(0.5 * (rate[1:] + rate[:-1]) * np.diff(t))])
    theta = abs(math.degrees(run.angle))
    t_rel = t - run.start_time

    def stays_within(limit: float) -> float:
        outside = np.flatnonzero(error >= limit)
        if outside.size == 0:
            return 0.0
        if outside[-1] == len(error) - 1:
            return math.nan  # never settled inside
        return float(t_rel[outside[-1] + 1])

    after_window = float(np.degrees(np.interp(run.start_time + WINDOW, t, turned)))
    t5, t2 = stays_within(5.0), stays_within(2.0)
    return {
        "axis": AXES[run.axis],
        "step_deg": round(math.degrees(run.angle), 1),
        "turned_after_window_deg": round(after_window, 1),
        "error_after_window_deg": round(float(np.interp(run.start_time + WINDOW, t, error)), 1),
        "time_within_5deg_s": round(t5, 3),
        "time_within_2deg_s": round(t2, 3),
        "overshoot_deg": round(max(0.0, float(np.degrees(turned.max())) - theta), 1),
        "peak_rate_deg_s": round(float(np.degrees(rate.max())), 1),
        "peak_torque_share": round(float(np.abs(torque).max()) / max_torque, 2),
        "peak_wheel_speed_rev_s": round(float(np.abs(wheel).max()) / (2 * math.pi), 2),
        "s_per_deg_2deg": round(t2 / theta, 5) if math.isfinite(t2) else math.nan,
        "s_per_deg_window": round(WINDOW / after_window, 5) if after_window > 0 else math.nan,
        "timed_out": run.settle.timed_out,
    }


def run_controller_test(axes: list[int], gains: list[str], angles: list[float], motor: str = "sim",
                        view: bool = False) -> list[dict]:
    results: list[dict] = []
    for gain in gains:
        sim = make_sim(motor, gain)
        data = sim.data
        max_torque = wheel_effective_inertia(sim.tuning) * sim.tuning.max_wheel_acceleration
        pacer = Pacer(sim, view)
        # each angle out and back, per axis
        pending = [(k, sgn * math.radians(a)) for k in axes for a in angles for sgn in (1.0, -1.0)]
        while pending:
            print(f"{gain} gain: jump for {len(pending)} steps")
            sim.reset()
            pacer.restart()
            jumped = airborne_seen = False
            launch_z = 0.0
            done_in_jump: list[StepRun] = []
            current: StepRun | None = None
            held = None
            landed_at = None
            while True:
                t = float(data.time)
                if not jumped and t >= JUMP_AT:
                    launch_z = float(data.xpos[sim.model.jnt_bodyid[sim.ids.root_joint]][2])
                    held, _ = body_state(sim)  # the attitude it jumps with
                    sim.reference_override = tuple(held)
                    sim.handle_key("space")
                    jumped = True
                if (jumped and current is None and pending and landed_at is None
                        and sim.trajectory.cut_off and sim.estimated_airborne
                        and t >= sim.trajectory.cutoff_time + START_DELAY):
                    last = done_in_jump[-1] if done_in_jump else None
                    needs = (last.trace[-1][0] - last.start_time if last else 0.0) + MARGIN
                    if not done_in_jump or flight_time_left(sim, launch_z) >= needs:
                        k, angle = pending[0]
                        axis = wheel_axis_in_body(sim, k)
                        target = rotated(held, axis, angle)
                        current = StepRun(k, angle, t, held.copy(), target, axis, Settle(t))
                        sim.reference_override = tuple(target)
                        print(f"  {t:6.3f} s  axis {AXES[k]} step {math.degrees(angle):+.0f} deg, "
                              f"{flight_time_left(sim, launch_z):.2f} s of flight left")
                    elif done_in_jump:
                        landed_at = landed_at  # wait for the landing; the rest goes to a new jump
                airborne_seen = airborne_seen or (jumped and sim.estimated_airborne)
                if airborne_seen and not sim.estimated_airborne and landed_at is None:
                    landed_at = t
                    sim.reference_override = None  # the landing as usual
                    if current is not None:
                        print(f"  {t:6.3f} s  landed during the step: repeated in the next jump")
                        current = None
                if landed_at is not None and t >= landed_at + SETTLE_AFTER:
                    break
                sim_step(sim)
                if current is not None:
                    q, w = body_state(sim)
                    k = current.axis
                    error = attitude_distance(q, current.target)
                    torque = float(data.ctrl[sim.ids.wheel_actuators[k]])
                    current.trace.append((float(data.time), float(w @ current.wheel_axis), error,
                                          torque, wheel_speed(sim, k)))
                    if current.settle.update(float(data.time), error, math.degrees(float(np.linalg.norm(w)))):
                        done_in_jump.append(current)
                        pending.pop(0)
                        held = current.target  # the next step starts from here
                        current = None
                if not pacer.after_step():
                    pacer.close()
                    return results + [dict(gain=gain, motor=motor, **summarise_step(r, max_torque))
                                      for r in done_in_jump]
            if not done_in_jump:
                raise RuntimeError("no step fitted in a jump")
            results += [dict(gain=gain, motor=motor, **summarise_step(r, max_torque)) for r in done_in_jump]
        pacer.close()
    return results


# ------------------------------------------------------------------ open loop

def direction_summary(t, rate, angle, wheel, start, mid, end, start_wheel) -> dict:
    """One spin direction: full torque from ``start``, back to rest at ``end``
    (``rate``, ``angle`` with this direction's turn positive)."""
    half = (t >= start - 1e-9) & (t <= end + 1e-9)
    spin = half & (t <= mid + 1e-9)
    peak_i = int(np.argmax(np.where(spin, rate, -np.inf)))
    peak = float(rate[peak_i])
    reach90 = float(t[np.argmax(spin & (rate >= 0.9 * peak))] - start)
    early = half & (t <= start + 0.05)
    alpha = float(np.polyfit(t[early] - start, rate[early], 1)[0]) if early.sum() > 2 else math.nan
    turned = float(np.interp(end, t, angle) - np.interp(start, t, angle))
    return {
        "peak_rate_deg_s": round(math.degrees(peak), 1),
        "time_to_peak_s": round(float(t[peak_i] - start), 3),
        "time_to_90pct_s": round(reach90, 3),
        "start_acceleration_deg_s2": round(math.degrees(alpha), 1),
        "wheel_speed_at_pulse_end_rev_s": round(abs(float(np.interp(mid, t, wheel)) - start_wheel) / (2 * math.pi), 2),
        "rest_to_rest_angle_deg": round(math.degrees(turned), 1),
        "rest_to_rest_time_s": round(end - start, 3),
        "rest_to_rest_mean_rate_deg_s": round(math.degrees(turned) / (end - start), 1),
    }


def summarise(run: AxisRun, sim_name: str) -> dict:
    """Both spin directions of one axis: the pulse (steps 1-2) and the
    reverse (steps 3-4), each from rest at full torque."""
    tr = np.array(run.trace)
    t, rate, wheel = tr[:, 0], tr[:, 1], tr[:, 3]
    m = run.manoeuvre
    t0 = m.start_time
    times = m.step_times
    in_pulse = t <= times.get("stop", math.inf)
    sign = float(np.sign(np.mean(rate[in_pulse]))) or 1.0
    rate = sign * rate  # the pulse's turn positive
    # The angle about the wheel axis, unwrapped: the integrated rate (one
    # axis turns; the attitude quaternion wraps at 180 deg).
    angle = np.concatenate([[0.0], np.cumsum(0.5 * (rate[1:] + rate[:-1]) * np.diff(t))])
    directions = {}
    if "reverse" in times:
        directions["first"] = direction_summary(t, rate, angle, wheel, t0, times["stop"],
                                                times["reverse"], m.start_wheel_speed)
    if m.done:
        directions["second"] = direction_summary(t, -rate, -angle, wheel, times["reverse"], times["return"],
                                                 times["done"], m.start_wheel_speed)
    both = list(directions.values())
    return {
        "axis": AXES[run.axis],
        "motor": sim_name,
        "finished": m.done,
        "takeoff_rate_deg_s": [round(math.degrees(v), 2) for v in run.start_rate],
        "directions": directions,
        # the slower direction: what a turn can count on either way
        "peak_rate_deg_s": min((d["peak_rate_deg_s"] for d in both), default=math.nan),
        "start_acceleration_deg_s2": min((d["start_acceleration_deg_s2"] for d in both), default=math.nan),
        # the turn left after step 4, relative to the take-off spin
        "residual_angle_deg": round(math.degrees(float(angle[-1])), 2) if m.done else math.nan,
        "cross_axis_peak_deg_s": round(math.degrees(run.cross_peak), 1),
        "step_times_s": {k: round(v - t0, 3) for k, v in times.items()},
        "trace": [[round(float(a), 3), round(math.degrees(float(b)), 2), round(math.degrees(float(c)), 2),
                   round(float(d) / (2 * math.pi), 3), float(e)] for a, b, c, d, e in tr],
    }


def run_test(axes: list[int], motor: str = "sim", view: bool = False) -> list[dict]:
    """The open-loop pulse test."""
    sim = make_sim(motor)
    data, model = sim.data, sim.model
    pacer = Pacer(sim, view)
    pending = list(axes)
    results: list[dict] = []
    while pending:
        print(f"jump for axes {[AXES[a] for a in pending]}")
        sim.reset()
        pacer.restart()
        sim.override = False
        jumped = False
        launch_z = 0.0
        runs: list[AxisRun] = []
        current: AxisRun | None = None
        finished_at = None  # the IMU's first impact after the jump
        airborne_seen = False
        steps = 0
        while True:
            t = float(data.time)
            if not jumped and t >= JUMP_AT:
                launch_z = float(data.xpos[model.jnt_bodyid[sim.ids.root_joint]][2])
                sim.handle_key("space")
                sim.override = True  # no attitude control in this flight
                jumped = True
            if jumped and current is None and pending and sim.trajectory.cut_off and sim.estimated_airborne:
                ready = t >= sim.trajectory.cutoff_time + START_DELAY
                needs = 4.0 * PULSE + MARGIN if not runs else (
                    runs[-1].manoeuvre.step_times["done"] - runs[-1].manoeuvre.start_time + MARGIN)
                if ready and (not runs or flight_time_left(sim, launch_z) >= needs):
                    k = pending.pop(0)
                    q, w = body_state(sim)
                    current = AxisRun(k, TurnManoeuvre(t, wheel_speed(sim, k)), q, w,
                                      wheel_axis_in_body(sim, k))
                    print(f"  {t:6.3f} s  axis {AXES[k]}: {flight_time_left(sim, launch_z):.2f} s of flight left")
            if current is not None:
                k = current.axis
                cmd = current.manoeuvre.update(t, wheel_speed(sim, k))
                sim.wheel_commands = [cmd if j == k else 0.0 for j in range(3)]
                q, w = body_state(sim)
                rel = w - current.start_rate
                about = float(rel @ current.wheel_axis)
                current.cross_peak = max(current.cross_peak,
                                         float(np.linalg.norm(rel - about * current.wheel_axis)))
                if steps % TRACE_EVERY == 0 or current.manoeuvre.done:
                    turned = np.array(quaternion_error(tuple(q), tuple(current.start_attitude)))
                    current.trace.append((t, about, float(turned @ current.wheel_axis),
                                          wheel_speed(sim, k), cmd))
                if current.manoeuvre.done:
                    runs.append(current)
                    print(f"  {t:6.3f} s  axis {AXES[k]} done")
                    current = None
                    sim.wheel_commands = [0.0, 0.0, 0.0]
            airborne_seen = airborne_seen or (jumped and sim.estimated_airborne)
            if airborne_seen and not sim.estimated_airborne and finished_at is None:
                finished_at = t
                sim.override = False  # the landing as usual
                if current is not None:
                    print(f"  {t:6.3f} s  landed during axis {AXES[current.axis]}: unfinished")
                    runs.append(current)
                    current = None
            if finished_at is not None and t >= finished_at + SETTLE_AFTER:
                break
            sim_step(sim)
            steps += 1
            if not pacer.after_step():
                pacer.close()
                return results + [summarise(r, motor) for r in runs]
        results += [summarise(r, motor) for r in runs]
        if not runs and pending:
            raise RuntimeError("no manoeuvre fitted in a jump")
    pacer.close()
    return results


def bang_bang_time(theta_deg: float, rate_deg_s: float, accel_deg_s2: float) -> float:
    """Rest-to-rest time [s] of a turn at full acceleration up to a top rate."""
    if theta_deg < rate_deg_s ** 2 / accel_deg_s2:
        return 2.0 * math.sqrt(theta_deg / accel_deg_s2)
    return theta_deg / rate_deg_s + rate_deg_s / accel_deg_s2


def print_open_loop(results: list[dict]) -> None:
    for r in results:
        print(f"\naxis {r['axis']} ({r['motor']} motor){'' if r['finished'] else ' UNFINISHED'}, "
              f"take-off spin {r['takeoff_rate_deg_s']} deg/s:")
        keys = ("peak_rate_deg_s", "time_to_peak_s", "time_to_90pct_s", "start_acceleration_deg_s2",
                "wheel_speed_at_pulse_end_rev_s", "rest_to_rest_angle_deg", "rest_to_rest_time_s",
                "rest_to_rest_mean_rate_deg_s")
        names = list(r["directions"])
        print(f"  {'':32s} " + "".join(f"{n + ' direction':>18s}" for n in names))
        for key in keys:
            print(f"  {key:32s} " + "".join(f"{r['directions'][n][key]:>18}" for n in names))
        for key in ("peak_rate_deg_s", "start_acceleration_deg_s2", "residual_angle_deg",
                    "cross_axis_peak_deg_s", "step_times_s"):
            label = key + (" (slower dir.)" if key in ("peak_rate_deg_s", "start_acceleration_deg_s2") else "")
            print(f"  {label:46s} {r[key]}")
        if r["finished"]:
            print("  bang-bang turn times, slower direction (s):", {
                theta: round(bang_bang_time(theta, r["peak_rate_deg_s"], r["start_acceleration_deg_s2"]), 2)
                for theta in (10, 20, 40, 60, 80)})


def fit_turn_time(rows: list[dict], key: str) -> tuple[float, float, float]:
    """Least-squares t = a + k * theta over the steps: (a [s], k [s/deg], max residual [s])."""
    pts = [(abs(r["step_deg"]), r[key]) for r in rows if math.isfinite(r[key]) and not r["timed_out"]]
    if len(pts) < 2:
        return math.nan, math.nan, math.nan
    theta, t = np.array(pts).T
    k, a = np.polyfit(theta, t, 1)
    return float(a), float(k), float(np.abs(a + k * theta - t).max())


def print_controller(results: list[dict]) -> None:
    cols = (("step_deg", "step"), ("turned_after_window_deg", "in 1 s"), ("time_within_5deg_s", "t<5deg"),
            ("time_within_2deg_s", "t<2deg"), ("overshoot_deg", "overshoot"), ("peak_rate_deg_s", "peak rate"),
            ("peak_torque_share", "torque"), ("peak_wheel_speed_rev_s", "wheel rev/s"),
            ("s_per_deg_2deg", "s/deg (2deg)"), ("s_per_deg_window", "s/deg (1 s)"))
    for gain in dict.fromkeys(r["gain"] for r in results):
        rows = [r for r in results if r["gain"] == gain]
        print(f"\n{gain} gain ({rows[0]['motor']} motor), axis {', '.join(dict.fromkeys(r['axis'] for r in rows))}:")
        print("  " + "".join(f"{name:>13s}" for _, name in cols))
        for r in rows:
            print("  " + "".join(f"{r[key]:>13}" for key, _ in cols) + ("  (timed out)" if r["timed_out"] else ""))
        worst = max((r["s_per_deg_2deg"] for r in rows if math.isfinite(r["s_per_deg_2deg"])), default=math.nan)
        print(f"  slowest: {worst:.4f} s per degree of error (time to within 2 deg / step angle)")
        for key, name in (("time_within_5deg_s", "5 deg"), ("time_within_2deg_s", "2 deg")):
            a, k, worst_fit = fit_turn_time(rows, key)
            print(f"  fit, time to within {name}: t = {a:.3f} s + {k:.5f} s/deg * angle "
                  f"(worst step off by {worst_fit:.3f} s)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--label", required=True)
    parser.add_argument("--mode", choices=("controller", "open-loop"), default="controller")
    parser.add_argument("--axes", nargs="+", choices=AXES, default=["z"])
    parser.add_argument("--gain", nargs="+", choices=GAINS, default=list(GAINS),
                        help="controller mode: the flight gain (design) and/or one designed for full torque")
    parser.add_argument("--angles", nargs="+", type=float, default=list(ANGLES),
                        help="controller mode: step angles [deg], each turned out and back")
    parser.add_argument("--motor", choices=("sim", "bench"), default="sim")
    parser.add_argument("--view", action="store_true", help="open the MuJoCo viewer (real time)")
    args = parser.parse_args()
    axes = [AXES.index(a) for a in args.axes]
    if args.mode == "controller":
        results = run_controller_test(axes, args.gain, args.angles, args.motor, args.view)
    else:
        results = run_test(axes, args.motor, args.view)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{args.label}.json").write_text(json.dumps(results, indent=1))
    (print_controller if args.mode == "controller" else print_open_loop)(results)
    print(f"\nsaved {RESULTS_DIR / (args.label + '.json')}")


if __name__ == "__main__":
    main()
