from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol

import mujoco
import numpy as np

from .cube_model import SENSOR_CUBE, SensorCube, linearize


@dataclass(frozen=True)
class PidGains:
    kp: float = 1.1
    ki: float = 0.15
    kd: float = 0.13


@dataclass(frozen=True)
class ReactionWheelTuning:
    max_wheel_velocity: float = 20.0 * 2.0 * math.pi
    max_wheel_acceleration: float = 75.0 * 2.0 * math.pi
    wheel_rest_deceleration: float = 10.0 * 2.0 * math.pi
    wheel_rest_response_time: float = 0.25
    stationary_linear_speed: float = 0.05
    stationary_angular_speed: float = 0.05
    stationary_duration: float = 1.0
    # J_c (wheels locked) on its smallest axis, for the motor torque limits.
    robot_inertia: float = min(SENSOR_CUBE.body_inertia)
    wheel_inertia: float = SENSOR_CUBE.wheel_inertia
    # Viscous bearing friction of each wheel [N*m*s/rad]. Matches the wheel
    # joints' damping in build/Jumping_Robot.xml; on hardware, identify it from
    # a spin-down test. Set to 0.0 to disable friction compensation.
    wheel_viscous_friction: float = 0.01
    roll: PidGains = PidGains()
    pitch: PidGains = PidGains()
    yaw: PidGains = PidGains()


@dataclass(frozen=True)
class LqrLimits:
    """Q and R of the report's LQR by Bryson's rule (report ch. 5).

    Each weight is 1/(largest acceptable value)^2, so the tuning reads as:
    "use full wheel acceleration when the attitude error reaches
    ``max_angle_error``" and "a body rate of ``max_rate`` is as bad as that
    error". The attitude states are the error quaternion's vector part, so the
    angle is weighed as sin(angle/2). ``max_wheel_acceleration`` defaults to
    the motor limit in :class:`ReactionWheelTuning`.
    """

    max_angle_error: float = math.radians(9.0)  # rad
    max_rate: float = 1.6  # rad/s
    max_wheel_acceleration: float | None = None  # rad/s^2

    def wheel_acceleration(self, tuning: ReactionWheelTuning) -> float:
        """Largest wheel acceleration u [rad/s^2]."""
        if self.max_wheel_acceleration is not None:
            return self.max_wheel_acceleration
        return tuning.max_wheel_acceleration

    def weights(self, tuning: ReactionWheelTuning) -> tuple[np.ndarray, np.ndarray]:
        """(Q, R) for x_r = [q1 q2 q3 wx wy wz] and u = Omega_dot."""
        max_acceleration = self.wheel_acceleration(tuning)
        max_quaternion = math.sin(0.5 * self.max_angle_error)
        q = np.diag([1.0 / max_quaternion**2] * 3 + [1.0 / self.max_rate**2] * 3)
        r = np.eye(3) / max_acceleration**2
        return q, r


@dataclass(frozen=True)
class SensorData:
    roll: float
    pitch: float
    yaw: float
    roll_rate: float
    pitch_rate: float
    yaw_rate: float


@dataclass(frozen=True)
class TorqueCommand:
    tau_x: float
    tau_y: float
    tau_z: float


@dataclass(frozen=True)
class WheelSpeeds:
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class WheelTorques:
    x: float
    y: float
    z: float


class AttitudeController(Protocol):
    # What the simulation should feed the control law: bearing-friction
    # compensation on the motor command, and the rate error to a moving
    # reference instead of the body rate.
    compensates_wheel_friction: bool
    follows_reference_rate: bool

    def compute_torque(
        self, sensor: SensorData, wheel_speeds: WheelSpeeds, time_step: float
    ) -> TorqueCommand: ...

    def reset(self) -> None: ...


class WheelRestController:
    def __init__(self, tuning: ReactionWheelTuning) -> None:
        self._tuning = tuning
        self._target_speeds = WheelSpeeds(0.0, 0.0, 0.0)
        self._active = False

    @property
    def active(self) -> bool:
        return self._active

    @property
    def target_speeds(self) -> WheelSpeeds:
        return self._target_speeds

    def begin(self, current_speeds: WheelSpeeds) -> None:
        self._target_speeds = current_speeds
        self._active = True

    def cancel(self) -> None:
        self._active = False

    def _rest_axis(self, measured_speed: float, target_speed: float, time_step: float) -> tuple[float, float]:
        target_step = self._tuning.wheel_rest_deceleration * time_step
        target_speed = math.copysign(
            max(0.0, abs(target_speed) - target_step), target_speed
        )
        desired_acceleration = (
            target_speed - measured_speed
        ) / self._tuning.wheel_rest_response_time
        limited_acceleration = max(
            -self._tuning.wheel_rest_deceleration,
            min(self._tuning.wheel_rest_deceleration, desired_acceleration),
        )
        effective_inertia = wheel_effective_inertia(self._tuning)
        return effective_inertia * limited_acceleration, target_speed

    def compute_motor_torques(
        self, current_speeds: WheelSpeeds, time_step: float
    ) -> WheelTorques:
        if not self._active:
            return WheelTorques(0.0, 0.0, 0.0)

        x_torque, target_x = self._rest_axis(
            current_speeds.x, self._target_speeds.x, time_step
        )
        y_torque, target_y = self._rest_axis(
            current_speeds.y, self._target_speeds.y, time_step
        )
        z_torque, target_z = self._rest_axis(
            current_speeds.z, self._target_speeds.z, time_step
        )
        self._target_speeds = WheelSpeeds(target_x, target_y, target_z)
        return WheelTorques(x_torque, y_torque, z_torque)


class ReactionWheelController:
    _max_integral = 10.0
    # The gains were tuned with bearing friction acting as damping, on the raw
    # body rate: no friction compensation, no rate error to the reference.
    compensates_wheel_friction = False
    follows_reference_rate = False

    def __init__(self, tuning: ReactionWheelTuning) -> None:
        self._tuning = tuning
        self.reset()

    @property
    def tuning(self) -> ReactionWheelTuning:
        return self._tuning

    def compute_torque(
        self, sensor: SensorData, wheel_speeds: WheelSpeeds, time_step: float
    ) -> TorqueCommand:
        del wheel_speeds
        self._roll_integral = self._clamp_integral(
            self._roll_integral - sensor.roll * time_step
        )
        self._pitch_integral = self._clamp_integral(
            self._pitch_integral - sensor.pitch * time_step
        )
        self._yaw_integral = self._clamp_integral(
            self._yaw_integral - sensor.yaw * time_step
        )

        return TorqueCommand(
            self._axis_torque(-sensor.roll, -sensor.roll_rate, self._tuning.roll, self._roll_integral),
            self._axis_torque(-sensor.pitch, -sensor.pitch_rate, self._tuning.pitch, self._pitch_integral),
            self._axis_torque(-sensor.yaw, -sensor.yaw_rate, self._tuning.yaw, self._yaw_integral),
        )

    def reset(self) -> None:
        self._roll_integral = 0.0
        self._pitch_integral = 0.0
        self._yaw_integral = 0.0

    @staticmethod
    def _axis_torque(error: float, rate_error: float, gains: PidGains, integral: float) -> float:
        return gains.kp * error + gains.ki * integral + gains.kd * rate_error

    @classmethod
    def _clamp_integral(cls, value: float) -> float:
        return max(-cls._max_integral, min(cls._max_integral, value))


class LqrController:
    """The report's LQR (ch. 4-5) on the sensor cube's linearised gyrostat model.

    State x_r = [q1 q2 q3 wx wy wz]: the vector part of the error quaternion
    to the reference attitude, and the body rates (the simulator passes the
    rate error to a turning reference, see :func:`reference_rate`; the report
    regulates to a fixed one). Input u = Omega_dot, the
    wheel accelerations; u = -K x_r with K = R^-1 B_r' P from the algebraic
    Riccati equation (5.6)-(5.7) on (A_r, B_r) of
    :func:`jumper_sim.cube_model.linearize`.

    As in the report (4.21), the model is linearised with the wheels at
    rest, Omega_hat = 0: one constant K, decoupled per axis, no Riccati solve
    while flying. A_r would depend on the wheel speeds (the gyroscopic
    coupling J_w*Omega_hat/J_c between the axes, 4.18); optionally,
    ``schedule_wheel_speeds`` re-linearises at the measured wheel speeds
    whenever they have moved ``relinearize_step`` [rad/s] from the last
    point (5-7 % less integrated error with spun wheels in the free-flight
    benchmark, ~0.18 ms per re-solve).

    The wheels take speed commands, as the ODrives do in velocity mode: the
    commanded acceleration is integrated into a wheel speed command, which a
    per-wheel speed loop (``speed_loop_bandwidth`` [Hz]) follows with the
    acceleration as torque feedforward. :meth:`compute_torque` returns the
    resulting body torque (minus the motor torque), so it plugs into the
    simulator's wheel limits like the PID.
    """

    compensates_wheel_friction = True
    follows_reference_rate = True

    def __init__(
        self,
        tuning: ReactionWheelTuning,
        limits: LqrLimits | None = None,
        cube: SensorCube = SENSOR_CUBE,
        schedule_wheel_speeds: bool = False,
        relinearize_step: float = 2.0,
        speed_loop_bandwidth: float = 25.0,
    ) -> None:
        if limits is None:
            limits = LqrLimits()
        if limits.max_angle_error <= 0.0 or limits.max_rate <= 0.0:
            raise ValueError("LQR limits must be positive")
        self._tuning = tuning
        self._cube = cube
        self._q, self._r = limits.weights(tuning)
        self._max_acceleration = limits.wheel_acceleration(tuning)
        self.schedule_wheel_speeds = schedule_wheel_speeds
        self._relinearize_step = relinearize_step
        self._effective_inertia = np.array([
            effective_inertia(cube.wheel_inertia, j) for j in cube.body_inertia
        ])
        self._speed_gain = self._effective_inertia * 2.0 * math.pi * speed_loop_bandwidth
        # Anti-windup: the command stays within the speed error that already
        # asks for full motor torque.
        max_motor_torque = wheel_effective_inertia(tuning) * tuning.max_wheel_acceleration
        self._max_speed_error = max_motor_torque / self._speed_gain
        self._linearized_at = np.zeros(3)
        self._gain = self.gain_at(self._linearized_at)
        self.reset()

    @property
    def gain(self) -> np.ndarray:
        """Current K (3 x 6): u = -K x_r."""
        return self._gain

    @property
    def linearized_at(self) -> np.ndarray:
        """Omega_hat [rad/s] the current gain was computed at."""
        return self._linearized_at.copy()

    @property
    def wheel_speed_command(self) -> WheelSpeeds | None:
        """Last wheel speed command [rad/s] (None before the first step)."""
        if self._speed_command is None:
            return None
        return WheelSpeeds(*(float(v) for v in self._speed_command))

    def gain_at(self, wheel_speeds: np.ndarray) -> np.ndarray:
        a, b = linearize(wheel_speeds, self._cube)
        p = solve_continuous_are(a, b, self._q, self._r)
        return np.linalg.solve(self._r, b.T @ p)

    def compute_torque(
        self, sensor: SensorData, wheel_speeds: WheelSpeeds, time_step: float
    ) -> TorqueCommand:
        speeds = np.array([wheel_speeds.x, wheel_speeds.y, wheel_speeds.z])
        if (
            self.schedule_wheel_speeds
            and np.max(np.abs(speeds - self._linearized_at)) > self._relinearize_step
        ):
            self._linearized_at = speeds
            self._gain = self.gain_at(speeds)
        state = np.concatenate([
            error_quaternion_vector((sensor.roll, sensor.pitch, sensor.yaw)),
            (sensor.roll_rate, sensor.pitch_rate, sensor.yaw_rate),
        ])
        acceleration = np.clip(
            -self._gain @ state, -self._max_acceleration, self._max_acceleration
        )
        if self._speed_command is None:
            self._speed_command = speeds.copy()
        self._speed_command = np.clip(
            self._speed_command + acceleration * time_step,
            speeds - self._max_speed_error,
            speeds + self._max_speed_error,
        )
        self._speed_command = np.clip(
            self._speed_command,
            -self._tuning.max_wheel_velocity,
            self._tuning.max_wheel_velocity,
        )
        motor = (
            self._effective_inertia * acceleration
            + self._speed_gain * (self._speed_command - speeds)
        )
        return TorqueCommand(*(-float(v) for v in motor))

    def reset(self) -> None:
        # The next step starts the speed command from the measured speeds.
        self._speed_command: np.ndarray | None = None


def error_quaternion_vector(rotation: tuple[float, float, float]) -> np.ndarray:
    """[q1 q2 q3] of the error quaternion for a body-frame rotation vector.

    The inverse of :func:`quaternion_error` (which returns the shorter
    rotation, so q4 >= 0): axis * sin(angle / 2).
    """
    vector = np.asarray(rotation, dtype=float)
    angle = float(np.linalg.norm(vector))
    if angle < 1e-12:
        return 0.5 * vector
    return vector * (math.sin(0.5 * angle) / angle)


def solve_continuous_are(
    a: np.ndarray, b: np.ndarray, q: np.ndarray, r: np.ndarray
) -> np.ndarray:
    """Solve A'P + PA - PBR^-1B'P + Q = 0 (5.7) with the matrix sign function.

    Small, dependency-free equivalent of scipy.linalg.solve_continuous_are.
    The sign function of the Hamiltonian (Newton iteration with determinant
    scaling) gives its stable invariant subspace without eigenvectors, which
    are ill-conditioned here: the cube's axes are nearly identical, so the
    eigenvalues nearly repeat.
    """
    n = a.shape[0]
    hamiltonian = np.block([[a, -b @ np.linalg.solve(r, b.T)], [-q, -a.T]])
    sign = hamiltonian
    for _ in range(100):
        inverse = np.linalg.inv(sign)
        scale = math.exp(-np.linalg.slogdet(sign)[1] / (2 * n))
        updated = 0.5 * (scale * sign + inverse / scale)
        converged = np.linalg.norm(updated - sign, 1) <= 1e-13 * np.linalg.norm(updated, 1)
        sign = updated
        if converged:
            break
    else:
        raise ValueError("Riccati equation has no stabilising solution")
    identity = np.eye(n)
    # P spans the stable subspace: (sign + I) [I; P] = 0.
    lhs = np.vstack([sign[:n, n:], sign[n:, n:] + identity])
    rhs = -np.vstack([sign[:n, :n] + identity, sign[n:, :n]])
    p = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    return 0.5 * (p + p.T)


@dataclass(frozen=True)
class RobotInertia:
    """Whole-robot mass properties about its CoM, in the root body frame.

    ``rigid_body`` treats every body -- including the wheels -- as locked
    together (the report's J_c); ``wheel_spin`` is each wheel's inertia about
    its own spin axis (J_w,i).
    """

    rigid_body: tuple[tuple[float, float, float], ...]
    wheel_spin: tuple[float, float, float]
    mass: float


def robot_inertia(
    model: mujoco.MjModel,
    root_joint_id: int,
    wheel_joint_ids: tuple[int, int, int],
) -> RobotInertia:
    """Evaluate :class:`RobotInertia` from the model at its reference pose.

    That is the flight pose: build/fit_robot_model.py puts the wound jump
    foot at q = 0, and the legs rest at 0.
    """
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    root_body = int(model.jnt_bodyid[root_joint_id])
    bodies = [
        b for b in range(model.nbody)
        if b == root_body or _is_descendant(model, b, root_body)
    ]
    com = data.subtree_com[root_body]
    inertia_world = np.zeros((3, 3))
    for b in bodies:
        rotation = data.ximat[b].reshape(3, 3)
        inertia_world += rotation @ np.diag(model.body_inertia[b]) @ rotation.T
        r = data.xipos[b] - com
        inertia_world += model.body_mass[b] * (float(r @ r) * np.eye(3) - np.outer(r, r))
    root_rotation = data.xmat[root_body].reshape(3, 3)
    inertia_body = root_rotation.T @ inertia_world @ root_rotation

    wheel_spin = []
    for joint_id in wheel_joint_ids:
        wheel_body = int(model.jnt_bodyid[joint_id])
        rotation = data.ximat[wheel_body].reshape(3, 3)
        wheel_inertia_world = rotation @ np.diag(model.body_inertia[wheel_body]) @ rotation.T
        joint_axis = data.xaxis[joint_id]
        wheel_spin.append(float(joint_axis @ wheel_inertia_world @ joint_axis))
    return RobotInertia(
        rigid_body=tuple(tuple(float(v) for v in row) for row in inertia_body),
        wheel_spin=tuple(wheel_spin),
        mass=float(sum(model.body_mass[b] for b in bodies)),
    )


def _is_descendant(model: mujoco.MjModel, body: int, ancestor: int) -> bool:
    while body > 0:
        body = int(model.body_parentid[body])
        if body == ancestor:
            return True
    return False


def wheel_friction_compensation(
    wheel_speed: float, tuning: ReactionWheelTuning
) -> float:
    """Motor torque that cancels the wheel's viscous bearing friction.

    Bearing friction drags the wheel with -b*w and pushes the body with +b*w.
    Left uncompensated, a wheel that is spinning after soaking up momentum
    keeps feeding a constant torque into the body, which a controller without
    integral action can only balance with a steady attitude error.
    """
    return tuning.wheel_viscous_friction * wheel_speed


def effective_inertia(wheel: float, body: float) -> float:
    """Torque per wheel acceleration relative to the body [kg*m^2].

    ``body`` is the whole robot's inertia about the wheel's axis, wheels
    locked (J_c): the body turns back by wheel/body of the wheel's
    acceleration (J_c w_dot = -J_w Omega_dot), so the motor needs
    wheel * (body - wheel) / body.
    """
    return wheel * (body - wheel) / body


def wheel_effective_inertia(tuning: ReactionWheelTuning) -> float:
    return effective_inertia(tuning.wheel_inertia, tuning.robot_inertia)


def read_simulated_imu(
    model: mujoco.MjModel, data: mujoco.MjData, root_joint_id: int
) -> SensorData:
    qpos_address = model.jnt_qposadr[root_joint_id]
    dof_address = model.jnt_dofadr[root_joint_id]
    qw, qx, qy, qz = data.qpos[qpos_address + 3 : qpos_address + 7]
    roll = math.atan2(
        2.0 * (qw * qx + qy * qz),
        1.0 - 2.0 * (qx * qx + qy * qy),
    )
    sin_pitch = 2.0 * (qw * qy - qz * qx)
    pitch = (
        math.copysign(math.pi / 2.0, sin_pitch)
        if abs(sin_pitch) >= 1.0
        else math.asin(sin_pitch)
    )
    yaw = math.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )
    return SensorData(
        roll,
        pitch,
        yaw,
        float(data.qvel[dof_address + 3]),
        float(data.qvel[dof_address + 4]),
        float(data.qvel[dof_address + 5]),
    )


def read_wheel_speeds(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_ids: tuple[int, int, int],
) -> WheelSpeeds:
    return WheelSpeeds(
        *(float(data.qvel[model.jnt_dofadr[joint_id]]) for joint_id in joint_ids)
    )


def wrapped_yaw_error(yaw: float, reference: float) -> float:
    return math.remainder(yaw - reference, 2.0 * math.pi)


def quaternion_error(
    quaternion: tuple[float, float, float, float],
    reference: tuple[float, float, float, float],
) -> tuple[float, float, float]:
    """Attitude error as a body-frame rotation vector [rad].

    Returns the axis * angle of the rotation from ``reference`` to the
    current attitude (q_err = q_ref^-1 * q). Unlike Euler angles, its
    components are about the body axes -- the same axes as the gyro rates and
    the reaction wheels -- at any attitude, and it has no singularity at
    +-90 deg pitch. The shorter of the two equivalent rotations is used.
    """
    w, x, y, z = quaternion
    rw, rx, ry, rz = reference
    # conj(reference) * quaternion
    ew = rw * w + rx * x + ry * y + rz * z
    ex = rw * x - rx * w - ry * z + rz * y
    ey = rw * y + rx * z - ry * w - rz * x
    ez = rw * z - rx * y + ry * x - rz * w
    if ew < 0.0:
        ew, ex, ey, ez = -ew, -ex, -ey, -ez
    vector_norm = math.sqrt(ex * ex + ey * ey + ez * ez)
    if vector_norm < 1e-12:
        return (2.0 * ex, 2.0 * ey, 2.0 * ez)
    scale = 2.0 * math.atan2(vector_norm, ew) / vector_norm
    return (scale * ex, scale * ey, scale * ez)


def reference_rate(
    previous: tuple[float, float, float, float],
    reference: tuple[float, float, float, float],
    quaternion: tuple[float, float, float, float],
    time_step: float,
    max_rate: float,
) -> tuple[float, float, float]:
    """Angular velocity of a moving reference attitude, in the body frame [rad/s].

    From two successive reference quaternions (MuJoCo convention), rotated
    from the reference frame into the body's (``quaternion``) and capped at
    ``max_rate`` so a step in the reference does not read as a huge rate.
    """
    if reference == previous:
        return (0.0, 0.0, 0.0)
    rate = [v / time_step for v in quaternion_error(reference, previous)]
    speed = math.sqrt(rate[0] ** 2 + rate[1] ** 2 + rate[2] ** 2)
    if speed > max_rate:
        rate = [v * max_rate / speed for v in rate]
    # Into the body frame: rotate by conj(q) * reference.
    w, x, y, z = quaternion
    rw, rx, ry, rz = reference
    relative = np.array([
        w * rw + x * rx + y * ry + z * rz,
        w * rx - x * rw - y * rz + z * ry,
        w * ry + x * rz - y * rw - z * rx,
        w * rz - x * ry + y * rx - z * rw,
    ])
    body = np.zeros(3)
    mujoco.mju_rotVecQuat(body, np.array(rate), relative)
    return (float(body[0]), float(body[1]), float(body[2]))


def limit_wheel_command(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_id: int,
    command: float,
    tuning: ReactionWheelTuning,
) -> float:
    wheel_velocity = float(data.qvel[model.jnt_dofadr[joint_id]])
    time_step = float(model.opt.timestep)
    effective_inertia = wheel_effective_inertia(tuning)
    max_motor_torque = effective_inertia * tuning.max_wheel_acceleration
    limited_command = max(-max_motor_torque, min(max_motor_torque, command))

    if wheel_velocity * limited_command <= 0.0:
        return limited_command

    speed_headroom = tuning.max_wheel_velocity - abs(wheel_velocity)
    if speed_headroom <= 0.0:
        return 0.0

    allowed_acceleration = speed_headroom / time_step
    allowed_torque = effective_inertia * min(
        tuning.max_wheel_acceleration, allowed_acceleration
    )
    return max(-allowed_torque, min(allowed_torque, limited_command))


def combine_attitude_and_rest_torque(
    attitude_torque: float, rest_torque: float, tuning: ReactionWheelTuning
) -> float:
    max_motor_torque = wheel_effective_inertia(tuning) * tuning.max_wheel_acceleration
    if attitude_torque * rest_torque < 0.0:
        return max(-max_motor_torque, min(max_motor_torque, rest_torque))
    available_rest_torque = max(
        -max_motor_torque - attitude_torque,
        min(max_motor_torque - attitude_torque, rest_torque),
    )
    return attitude_torque + available_rest_torque
