from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol

import mujoco


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
    roll: PidGains = PidGains()
    pitch: PidGains = PidGains()
    yaw: PidGains = PidGains()


@dataclass(frozen=True)
class LqrWeights:
    angle: float = 100.0
    angular_rate: float = 10.0
    torque: float = 1.0


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


class LqrController:
    def __init__(
        self, tuning: ReactionWheelTuning, weights: LqrWeights = LqrWeights()
    ) -> None:
        if (
            tuning.robot_inertia <= 0.0
            or weights.angle <= 0.0
            or weights.angular_rate <= 0.0
            or weights.torque <= 0.0
        ):
            raise ValueError("LQR inertia and weights must be positive")
        self._gains = self._compute_axis_gains(tuning, weights)

    @property
    def gains(self) -> tuple[float, float]:
        return self._gains

    def compute_torque(
        self, sensor: SensorData, wheel_speeds: WheelSpeeds, time_step: float
    ) -> TorqueCommand:
        del wheel_speeds, time_step

        def axis_torque(angle: float, angular_rate: float) -> float:
            return -(self._gains[0] * angle + self._gains[1] * angular_rate)

        return TorqueCommand(
            axis_torque(sensor.roll, sensor.roll_rate),
            axis_torque(sensor.pitch, sensor.pitch_rate),
            axis_torque(sensor.yaw, sensor.yaw_rate),
        )

    def reset(self) -> None:
        pass

    @staticmethod
    def _compute_axis_gains(
        tuning: ReactionWheelTuning, weights: LqrWeights
    ) -> tuple[float, float]:
        inertia = tuning.robot_inertia
        p_angle_rate = inertia * math.sqrt(weights.torque * weights.angle)
        p_rate = inertia * math.sqrt(
            weights.torque * (2.0 * p_angle_rate + weights.angular_rate)
        )
        return (
            p_angle_rate / (weights.torque * inertia),
            p_rate / (weights.torque * inertia),
        )


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
