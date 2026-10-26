from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol

import mujoco
import numpy as np


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
    robot_mass: float = 5.0
    robot_inertia: float = 0.075
    wheel_mass: float = 0.2
    wheel_inertia: float = 0.0036
    # Viscous bearing friction of each wheel [N*m*s/rad]. Matches the wheel
    # joints' damping in build/Jumping_Robot.xml; on hardware, identify it from
    # a spin-down test. Set to 0.0 to disable friction compensation.
    wheel_viscous_friction: float = 0.01
    roll: PidGains = PidGains()
    pitch: PidGains = PidGains()
    yaw: PidGains = PidGains()


@dataclass(frozen=True)
class LqrLimits:
    """Physical tuning knobs for the LQR (Bryson's rule).

    Each weight is 1/(largest acceptable value)^2, so the tuning reads as:
    "use full torque when the error reaches ``max_angle_error``", "a body rate
    of ``max_rate`` is as bad as that angle error", and "remove a steady error
    over roughly ``integral_time`` seconds". ``max_torque`` defaults to the
    wheel motor limit from :class:`ReactionWheelTuning`.
    """

    max_angle_error: float = math.radians(9.0)  # rad
    max_rate: float = 1.6  # rad/s
    integral_time: float = 0.3  # s; 0 disables the integrator
    max_torque: float | None = None  # N*m

    def to_weights(self, tuning: ReactionWheelTuning) -> "LqrWeights":
        max_torque = (
            self.max_torque
            if self.max_torque is not None
            else wheel_effective_inertia(tuning) * tuning.max_wheel_acceleration
        )
        integral = (
            1.0 / (self.max_angle_error * self.integral_time) ** 2
            if self.integral_time > 0.0
            else 0.0
        )
        return LqrWeights(
            angle=1.0 / self.max_angle_error**2,
            angular_rate=1.0 / self.max_rate**2,
            torque=1.0 / max_torque**2,
            integral=integral,
        )


@dataclass(frozen=True)
class LqrWeights:
    """Raw LQR weights; build them with :meth:`LqrLimits.to_weights`."""

    angle: float
    angular_rate: float
    torque: float
    # Weight on the integrated angle error (LQI); 0 disables the integrator.
    # It removes the steady error left by bearing friction the feedforward in
    # wheel_friction_compensation() does not cancel (unknown or drifting b).
    integral: float
    integral_torque_limit: float = 0.8  # N*m, anti-windup clamp
    # Only integrate while |angle error| is below this [rad], so large
    # transients do not wind the integrator up (integral separation).
    integral_zone: float = math.radians(5.0)


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


AxisInertia = tuple[float, float, float]
AxisGains = tuple[float, float]


class LqrController:
    """Per-axis LQR on a double integrator J*theta_ddot = tau.

    ``axis_inertia`` is the inertia the wheels actually push against on each
    body axis (roll, pitch, yaw). Passing ``gyroscopic_inertia`` (from
    :func:`robot_inertia`) also supplies it, and adds the w x H gyroscopic
    feedforward from :func:`gyroscopic_torque`. When neither is given,
    ``tuning.robot_inertia`` is used for all three axes. ``weights`` default
    to :class:`LqrLimits` ``().to_weights(tuning)``.

    With ``weights.integral > 0`` the state is augmented with the integrated
    angle error (LQI), which removes steady attitude errors from constant
    disturbance torques such as unmodelled bearing friction. The integrator is
    clamped so its torque never exceeds ``weights.integral_torque_limit``.
    """

    def __init__(
        self,
        tuning: ReactionWheelTuning,
        weights: LqrWeights | None = None,
        axis_inertia: AxisInertia | None = None,
        gyroscopic_inertia: "RobotInertia | None" = None,
    ) -> None:
        if weights is None:
            weights = LqrLimits().to_weights(tuning)
        self._gyroscopic_inertia = gyroscopic_inertia
        if axis_inertia is None and gyroscopic_inertia is not None:
            axis_inertia = gyroscopic_inertia.plant_axis_inertia
        if axis_inertia is None:
            axis_inertia = (tuning.robot_inertia,) * 3
        if (
            any(inertia <= 0.0 for inertia in axis_inertia)
            or weights.angle <= 0.0
            or weights.angular_rate <= 0.0
            or weights.torque <= 0.0
            or weights.integral < 0.0
            or weights.integral_torque_limit < 0.0
        ):
            raise ValueError("LQR inertia and weights must be positive")
        self._weights = weights
        self._axis_inertia = tuple(float(i) for i in axis_inertia)
        axis_solutions = [
            self._compute_axis_gains(inertia, weights)
            for inertia in self._axis_inertia
        ]
        self._gains = tuple((k[1], k[2]) for k in axis_solutions)
        self._integral_gains = tuple(k[0] for k in axis_solutions)
        self._integral_limits = tuple(
            weights.integral_torque_limit / k_i if k_i > 0.0 else 0.0
            for k_i in self._integral_gains
        )
        self.reset()

    @property
    def axis_inertia(self) -> AxisInertia:
        return self._axis_inertia

    @property
    def gains(self) -> tuple[AxisGains, AxisGains, AxisGains]:
        """(angle gain, rate gain) for roll, pitch and yaw."""
        return self._gains

    @property
    def integral_gains(self) -> tuple[float, float, float]:
        """Integrated-angle gain per axis (all zero when integral is off)."""
        return self._integral_gains

    def compute_torque(
        self, sensor: SensorData, wheel_speeds: WheelSpeeds, time_step: float
    ) -> TorqueCommand:
        feedforward = (
            gyroscopic_torque(self._gyroscopic_inertia, sensor, wheel_speeds)
            if self._gyroscopic_inertia is not None
            else (0.0, 0.0, 0.0)
        )
        angles = (sensor.roll, sensor.pitch, sensor.yaw)
        rates = (sensor.roll_rate, sensor.pitch_rate, sensor.yaw_rate)
        torques = []
        for axis in range(3):
            k_i = self._integral_gains[axis]
            if k_i > 0.0 and abs(angles[axis]) < self._weights.integral_zone:
                limit = self._integral_limits[axis]
                self._integral[axis] = max(
                    -limit, min(limit, self._integral[axis] + angles[axis] * time_step)
                )
            k_angle, k_rate = self._gains[axis]
            torques.append(
                feedforward[axis]
                - (k_i * self._integral[axis] + k_angle * angles[axis] + k_rate * rates[axis])
            )
        return TorqueCommand(*torques)

    def reset(self) -> None:
        self._integral = [0.0, 0.0, 0.0]

    @staticmethod
    def _compute_axis_gains(
        inertia: float, weights: LqrWeights
    ) -> tuple[float, float, float]:
        """Return (integral gain, angle gain, rate gain) for one axis."""
        if weights.integral == 0.0:
            # State [angle, rate]: double integrator.
            a = np.array([[0.0, 1.0], [0.0, 0.0]])
            b = np.array([[0.0], [1.0 / inertia]])
            q = np.diag([weights.angle, weights.angular_rate])
        else:
            # State [integrated angle, angle, rate].
            a = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
            b = np.array([[0.0], [0.0], [1.0 / inertia]])
            q = np.diag([weights.integral, weights.angle, weights.angular_rate])
        p = solve_continuous_are(a, b, q, np.array([[weights.torque]]))
        k = (b.T @ p / weights.torque).ravel()
        if weights.integral == 0.0:
            return 0.0, float(k[0]), float(k[1])
        return float(k[0]), float(k[1]), float(k[2])


def solve_continuous_are(
    a: np.ndarray, b: np.ndarray, q: np.ndarray, r: np.ndarray
) -> np.ndarray:
    """Solve A'P + PA - PBR^-1B'P + Q = 0 via the Hamiltonian's stable subspace.

    Small, dependency-free equivalent of scipy.linalg.solve_continuous_are for
    the low-order systems used here.
    """
    n = a.shape[0]
    r_inv = np.linalg.inv(r)
    hamiltonian = np.block([[a, -b @ r_inv @ b.T], [-q, -a.T]])
    eigenvalues, eigenvectors = np.linalg.eig(hamiltonian)
    stable = eigenvectors[:, eigenvalues.real < 0.0]
    if stable.shape[1] != n:
        raise ValueError("Riccati equation has no stabilising solution")
    p = np.real(stable[n:, :] @ np.linalg.inv(stable[:n, :]))
    return 0.5 * (p + p.T)


@dataclass(frozen=True)
class RobotInertia:
    """Whole-robot inertia about its CoM, in the root body frame.

    ``rigid_body`` treats every body -- including the wheels -- as locked
    together; ``wheel_spin`` is each wheel's inertia about its own spin axis.
    """

    rigid_body: tuple[tuple[float, float, float], ...]
    wheel_spin: tuple[float, float, float]

    @property
    def plant_axis_inertia(self) -> AxisInertia:
        """Inertia each wheel's motor reacts against (own wheel spin excluded)."""
        return tuple(
            self.rigid_body[axis][axis] - self.wheel_spin[axis] for axis in range(3)
        )


def robot_inertia(
    model: mujoco.MjModel,
    root_joint_id: int,
    wheel_joint_ids: tuple[int, int, int],
) -> RobotInertia:
    """Evaluate :class:`RobotInertia` from the model at its reference pose."""
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
    )


def gyroscopic_torque(
    inertia: RobotInertia, sensor: SensorData, wheel_speeds: WheelSpeeds
) -> tuple[float, float, float]:
    """Body torque w x H that cancels gyroscopic coupling in flight.

    H = J_rigid*w + J_spin*w_wheel is the robot's total angular momentum in
    the body frame (w_wheel relative to the body). In flight dH/dt = 0 in
    inertial space, so the body sees -w x H; adding +w x H to the commanded
    body torque leaves each axis behaving like the decoupled double integrator
    the LQR was designed for.
    """
    rate = np.array([sensor.roll_rate, sensor.pitch_rate, sensor.yaw_rate])
    momentum = np.array(inertia.rigid_body) @ rate + np.array(inertia.wheel_spin) * np.array(
        [wheel_speeds.x, wheel_speeds.y, wheel_speeds.z]
    )
    torque = np.cross(rate, momentum)
    return float(torque[0]), float(torque[1]), float(torque[2])


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


def wheel_effective_inertia(tuning: ReactionWheelTuning) -> float:
    return tuning.wheel_inertia * tuning.robot_inertia / (
        tuning.wheel_inertia + tuning.robot_inertia
    )


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
