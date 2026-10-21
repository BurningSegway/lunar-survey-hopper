from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum, auto

import mujoco

from .controller import (
    AttitudeController,
    LqrController,
    ReactionWheelController,
    ReactionWheelTuning,
    WheelRestController,
    WheelSpeeds,
    combine_attitude_and_rest_torque,
    limit_wheel_command,
    read_simulated_imu,
    read_wheel_speeds,
    wrapped_yaw_error,
)
from .telemetry import TelemetryPayload, make_telemetry_payload

EARTH_GRAVITY = 9.81
MOON_GRAVITY = 1.62
JUMP_RELEASE_DURATION = 0.4
JUMP_CYCLE_DURATION = 3.0
TILT_FORCE = 200.0
JUMP_WINDUP_FORCE = -8000.0


class WheelControlMode(Enum):
    STABILIZING = auto()
    RESTING = auto()
    DISABLED = auto()


@dataclass(frozen=True)
class ModelIds:
    root_joint: int
    wheel_joints: tuple[int, int, int]
    wheel_actuators: tuple[int, int, int]
    wind_up_actuator: int
    tilt_actuators: tuple[int, int]


def _required_id(model: mujoco.MjModel, object_type: int, name: str) -> int:
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id < 0:
        raise ValueError(f"Model is missing required MuJoCo object {name!r}")
    return object_id


def find_model_ids(model: mujoco.MjModel) -> ModelIds:
    return ModelIds(
        root_joint=_required_id(model, mujoco.mjtObj.mjOBJ_JOINT, "root"),
        wheel_joints=(
            _required_id(model, mujoco.mjtObj.mjOBJ_JOINT, "rw_x_joint"),
            _required_id(model, mujoco.mjtObj.mjOBJ_JOINT, "rw_y_joint"),
            _required_id(model, mujoco.mjtObj.mjOBJ_JOINT, "rw_z_joint"),
        ),
        wheel_actuators=(
            _required_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "motor_rw_x"),
            _required_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "motor_rw_y"),
            _required_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "motor_rw_z"),
        ),
        wind_up_actuator=_required_id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, "wind_up_motor"
        ),
        tilt_actuators=(
            _required_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "tilt_fl"),
            _required_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "tilt_fr"),
        ),
    )


class JumperSimulation:
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        controller_name: str,
        tuning: ReactionWheelTuning = ReactionWheelTuning(),
    ) -> None:
        self.model = model
        self.data = data
        self.ids = find_model_ids(model)
        self.tuning = tuning
        self.selected_gravity = EARTH_GRAVITY
        self.model.opt.gravity[2] = -EARTH_GRAVITY

        if controller_name == "pid":
            self.controller: AttitudeController = ReactionWheelController(tuning)
        elif controller_name == "lqr":
            self.controller = LqrController(tuning)
        else:
            raise ValueError(f"Unknown controller {controller_name!r}; use 'pid' or 'lqr'")
        self.controller_name = controller_name
        self.wheel_rest_controller = WheelRestController(tuning)
        self.wheel_control_mode = WheelControlMode.STABILIZING
        self._stationary_duration = 0.0
        self.yaw_reference = 0.0
        self.jump_triggered = False
        self.jump_start_time = 0.0
        self.tilt_active = False

    def handle_key(self, key: str) -> str | None:
        if key == "space":
            if not self.jump_triggered:
                self.jump_triggered = True
                self.jump_start_time = float(self.data.time)
                return "[JUMP INITIATED] Releasing spring!"
        elif key == "enter":
            self.tilt_active = not self.tilt_active
            return (
                "[TILT] Front-leg shortening is "
                f"{'ON' if self.tilt_active else 'OFF'}; attitude controller remains active."
            )
        elif key == "backspace":
            self.selected_gravity = (
                MOON_GRAVITY
                if self.selected_gravity == EARTH_GRAVITY
                else EARTH_GRAVITY
            )
            self.model.opt.gravity[2] = -self.selected_gravity
            name = "Moon" if self.selected_gravity == MOON_GRAVITY else "Earth"
            return f"[GRAVITY] Switched to {name} ({self.selected_gravity} m/s^2)."
        elif key == "c":
            if self.wheel_control_mode is WheelControlMode.DISABLED:
                self.wheel_control_mode = WheelControlMode.STABILIZING
                self._stationary_duration = 0.0
                return "[CONTROLLER] Attitude stabilization enabled."
        elif key == "x":
            self.wheel_rest_controller.cancel()
            self.wheel_control_mode = WheelControlMode.DISABLED
            self._stationary_duration = 0.0
            self.controller.reset()
            return "[CONTROLLER] Disabled; reaction-wheel motor torques are zero."
        elif key == "r":
            self.reset()
            return "[RESET] Simulation reset to initial state."
        return None

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.model.opt.gravity[2] = -EARTH_GRAVITY
        mujoco.mj_forward(self.model, self.data)
        self.controller.reset()
        self.wheel_rest_controller.cancel()
        self.wheel_control_mode = WheelControlMode.STABILIZING
        self._stationary_duration = 0.0
        self.yaw_reference = 0.0
        self.jump_triggered = False
        self.jump_start_time = 0.0
        self.tilt_active = False
        self.selected_gravity = EARTH_GRAVITY

    def read_wheel_speeds(self) -> WheelSpeeds:
        return read_wheel_speeds(self.model, self.data, self.ids.wheel_joints)

    @property
    def relaxation_active(self) -> bool:
        return self.wheel_control_mode is WheelControlMode.RESTING

    def telemetry_payload(self) -> TelemetryPayload:
        sensor = read_simulated_imu(self.model, self.data, self.ids.root_joint)
        return make_telemetry_payload(
            sensor, self.yaw_reference, self.read_wheel_speeds(), self.tuning
        )

    def step(self) -> None:
        self._update_wheel_rest_state()
        self._apply_controls()
        mujoco.mj_step(self.model, self.data)

    def _update_wheel_rest_state(self) -> None:
        if self.wheel_control_mode is WheelControlMode.DISABLED:
            self._stationary_duration = 0.0
            return

        dof_address = self.model.jnt_dofadr[self.ids.root_joint]
        linear_velocity = self.data.qvel[dof_address : dof_address + 3]
        angular_velocity = self.data.qvel[dof_address + 3 : dof_address + 6]
        stationary = (
            not self.jump_triggered
            and not self.tilt_active
            and float(linear_velocity @ linear_velocity)
            <= self.tuning.stationary_linear_speed**2
            and float(angular_velocity @ angular_velocity)
            <= self.tuning.stationary_angular_speed**2
        )

        if stationary:
            self._stationary_duration += float(self.model.opt.timestep)
            if (
                self.wheel_control_mode is WheelControlMode.STABILIZING
                and self._stationary_duration >= self.tuning.stationary_duration
            ):
                self.wheel_rest_controller.begin(self.read_wheel_speeds())
                self.wheel_control_mode = WheelControlMode.RESTING
        else:
            self._stationary_duration = 0.0
            if self.wheel_control_mode is WheelControlMode.RESTING:
                self.wheel_rest_controller.cancel()
                self.wheel_control_mode = WheelControlMode.STABILIZING

    def _apply_controls(self) -> None:
        if self.wheel_control_mode is not WheelControlMode.DISABLED:
            sensor = read_simulated_imu(self.model, self.data, self.ids.root_joint)
            sensor = replace(
                sensor, yaw=wrapped_yaw_error(sensor.yaw, self.yaw_reference)
            )
            wheel_speeds = self.read_wheel_speeds()
            torques = self.controller.compute_torque(
                sensor, wheel_speeds, float(self.model.opt.timestep)
            )
            rest_torques = self.wheel_rest_controller.compute_motor_torques(
                wheel_speeds, float(self.model.opt.timestep)
            )
            body_torques = (torques.tau_x, torques.tau_y, torques.tau_z)
            rest_values = (rest_torques.x, rest_torques.y, rest_torques.z)
            for actuator_id, joint_id, body_torque, rest_torque in zip(
                self.ids.wheel_actuators,
                self.ids.wheel_joints,
                body_torques,
                rest_values,
                strict=True,
            ):
                command = combine_attitude_and_rest_torque(
                    -body_torque, rest_torque, self.tuning
                )
                self.data.ctrl[actuator_id] = limit_wheel_command(
                    self.model, self.data, joint_id, command, self.tuning
                )
        else:
            for actuator_id in self.ids.wheel_actuators:
                self.data.ctrl[actuator_id] = 0.0

        tilt_force = TILT_FORCE if self.tilt_active else 0.0
        for actuator_id in self.ids.tilt_actuators:
            self.data.ctrl[actuator_id] = tilt_force

        if not self.jump_triggered:
            self.data.ctrl[self.ids.wind_up_actuator] = JUMP_WINDUP_FORCE
        else:
            time_since_jump = float(self.data.time) - self.jump_start_time
            if time_since_jump < JUMP_RELEASE_DURATION:
                self.data.ctrl[self.ids.wind_up_actuator] = 0.0
            elif time_since_jump < JUMP_CYCLE_DURATION:
                self.data.ctrl[self.ids.wind_up_actuator] = JUMP_WINDUP_FORCE
            else:
                self.jump_triggered = False
