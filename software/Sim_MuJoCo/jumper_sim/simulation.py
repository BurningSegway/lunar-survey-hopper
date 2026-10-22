from __future__ import annotations

from dataclasses import dataclass, replace
import math
from enum import Enum, auto

import mujoco
import numpy as np

from .controller import (
    AttitudeController,
    LqrController,
    LqrLimits,
    ReactionWheelController,
    ReactionWheelTuning,
    WheelRestController,
    WheelSpeeds,
    combine_attitude_and_rest_torque,
    limit_wheel_command,
    quaternion_error,
    read_simulated_imu,
    read_wheel_speeds,
    robot_inertia,
    wheel_friction_compensation,
    wrapped_yaw_error,
)
from .depth_camera import CAMERA_SITE, TERRAIN_GEOM, DepthCameraSpec, SimulatedDepthCamera
from .landing_reference import (
    LandingReference,
    LandingReferenceConfig,
    estimate_ground_plane,
    frame_points_world,
    quaternion_to_matrix,
)
from .telemetry import TelemetryPayload, make_telemetry_payload

EARTH_GRAVITY = 9.81
MOON_GRAVITY = 1.62
JUMP_RELEASE_DURATION = 0.4
JUMP_CYCLE_DURATION = 3.0
TILT_FORCE = 200.0
JUMP_WINDUP_FORCE = -8000.0
# Contact gating: the robot counts as airborne again only after this long
# without terrain contact, so rocking on the legs or a skid reads as one
# contact phase. Real bounces in Moon gravity last >= ~0.1 s.
FLIGHT_DEBOUNCE = 0.06


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


VISUAL_POINTS = 150


@dataclass(frozen=True)
class LandingVisual:
    anchor: np.ndarray  # world point on the fitted plane below the robot
    measured_normal: np.ndarray  # latest single-frame fit
    points: np.ndarray  # sample of the camera's ground points (world)
    radius: float
    used: bool = True  # False: ignored (too noisy), frozen or rejected


class CameraLandingReference:
    """Drives the simulated D435 and the landing-plane filter.

    While the robot is airborne, a depth frame is captured at the camera's
    frame rate and becomes usable ``latency`` seconds later; each usable frame
    is fitted and fused. On the ground nothing is updated, so the last
    estimate is held. :meth:`reset` (called at take-off) returns to "land
    level" until the camera has seen the ground below.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        root_joint_id: int,
        config: LandingReferenceConfig = LandingReferenceConfig(),
        spec: DepthCameraSpec = DepthCameraSpec(),
        seed: int = 0,
    ) -> None:
        self.camera = SimulatedDepthCamera(model, root_joint_id, spec, seed=seed)
        self.filter = LandingReference(config)
        self.config = config
        self._period = 1.0 / spec.frame_rate
        self._latency = spec.latency
        self._pending: list = []
        self._next_capture = 0.0
        self._root_qpos = int(model.jnt_qposadr[root_joint_id])
        self.visual: LandingVisual | None = None

    def reset(self, time: float) -> None:
        self.filter.reset(time)
        self._pending.clear()
        self._next_capture = time
        self.visual = None

    def step(self, data: mujoco.MjData, airborne: bool) -> None:
        now = float(data.time)
        if airborne and now >= self._next_capture:
            frame = self.camera.capture(data, roi_radius=self.config.roi_radius)
            position = data.qpos[self._root_qpos : self._root_qpos + 3].copy()
            self._pending.append((now + self._latency, frame, position))
            self._next_capture = now + self._period
        while self._pending and self._pending[0][0] <= now:
            _, frame, position = self._pending.pop(0)
            estimate = estimate_ground_plane(frame, self.camera.mount, self.config)
            used = self.filter.update(estimate)
            if estimate.valid:
                self._store_visual(frame, position, estimate, used)

    def _store_visual(self, frame, position: np.ndarray, estimate, used: bool) -> None:
        """Keep the latest fit in world coordinates for the viewer (sim only)."""
        points = frame_points_world(frame, self.camera.mount)
        points = points[np.hypot(points[:, 0], points[:, 1]) <= self.config.roi_radius]
        if len(points) > VISUAL_POINTS:
            points = points[:: max(1, len(points) // VISUAL_POINTS)][:VISUAL_POINTS]
        self.visual = LandingVisual(
            anchor=position + np.array([0.0, 0.0, -estimate.height]),
            measured_normal=np.array(estimate.normal),
            points=points + position,
            radius=self.config.roi_radius,
            used=used,
        )

    def reference_quaternion(self, yaw_reference: float) -> tuple[float, float, float, float]:
        return self.filter.reference_quaternion(yaw_reference)


class JumperSimulation:
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        controller_name: str,
        tuning: ReactionWheelTuning = ReactionWheelTuning(),
        landing_reference: bool = True,
        contact_gating: bool = True,
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
            self.controller = LqrController(
                tuning,
                LqrLimits().to_weights(tuning),
                gyroscopic_inertia=robot_inertia(
                    model, self.ids.root_joint, self.ids.wheel_joints
                ),
            )
        else:
            raise ValueError(f"Unknown controller {controller_name!r}; use 'pid' or 'lqr'")
        self.controller_name = controller_name
        # Only the LQR cancels bearing friction: the PID gains were tuned with
        # that friction acting as extra rate damping, and lose damping without it.
        self.compensate_wheel_friction = controller_name == "lqr"
        # LQR only: roll/pitch reference from the downward depth camera.
        self.landing_reference: CameraLandingReference | None = None
        if (
            landing_reference
            and controller_name == "lqr"
            and mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, CAMERA_SITE) >= 0
        ):
            self.landing_reference = CameraLandingReference(model, self.ids.root_joint)
        self._robot_geoms = _robot_geom_mask(model, self.ids.root_joint)
        self._terrain_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, TERRAIN_GEOM)
        # LQR only: attitude control in flight, wheel momentum bleed-off in
        # contact (replaces the stationary-for-a-while rest trigger).
        self.contact_gating = contact_gating and controller_name == "lqr"
        self.in_contact = False
        self._time_without_contact = math.inf
        self.contact_phases = 0
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
                if self.controller_name == "lqr":
                    # Start each flight with an empty integrator rather than
                    # whatever it collected while standing on the ground.
                    self.controller.reset()
                if self.landing_reference is not None:
                    self.landing_reference.reset(float(self.data.time))
                return "[JUMP INITIATED] Releasing spring!"
        elif key == "enter":
            self.tilt_active = not self.tilt_active
            note = (
                "attitude control is gated off while in contact."
                if self.contact_gating
                else "attitude controller remains active."
            )
            return (
                "[TILT] Front-leg shortening is "
                f"{'ON' if self.tilt_active else 'OFF'}; {note}"
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
                if self.contact_gating and self.in_contact:
                    self._enter_contact_mode()
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
        if self.landing_reference is not None:
            self.landing_reference.reset(float(self.data.time))
        self.wheel_rest_controller.cancel()
        self.wheel_control_mode = WheelControlMode.STABILIZING
        self._stationary_duration = 0.0
        self.yaw_reference = 0.0
        self.jump_triggered = False
        self.jump_start_time = 0.0
        self.tilt_active = False
        self.selected_gravity = EARTH_GRAVITY
        self.in_contact = False
        self._time_without_contact = math.inf
        self.contact_phases = 0

    def read_wheel_speeds(self) -> WheelSpeeds:
        return read_wheel_speeds(self.model, self.data, self.ids.wheel_joints)

    @property
    def relaxation_active(self) -> bool:
        return self.wheel_control_mode is WheelControlMode.RESTING

    def telemetry_payload(self) -> TelemetryPayload:
        sensor = read_simulated_imu(self.model, self.data, self.ids.root_joint)
        roll_ref, pitch_ref = 0.0, 0.0
        if self.landing_reference is not None:
            r = quaternion_to_matrix(self.reference_quaternion())
            roll_ref = math.atan2(r[2, 1], r[2, 2])
            pitch_ref = math.asin(max(-1.0, min(1.0, -r[2, 0])))
        return make_telemetry_payload(
            sensor, self.yaw_reference, self.read_wheel_speeds(), self.tuning,
            roll_reference=roll_ref, pitch_reference=pitch_ref,
        )

    def robot_touches_terrain(self) -> bool:
        contacts = self.data.contact.geom[: self.data.ncon]
        if not len(contacts):
            return False
        touching = (
            (self._robot_geoms[contacts[:, 0]] & (contacts[:, 1] == self._terrain_geom))
            | (self._robot_geoms[contacts[:, 1]] & (contacts[:, 0] == self._terrain_geom))
        )
        return bool(np.any(touching))

    def reference_quaternion(self) -> tuple[float, float, float, float]:
        """Attitude the LQR regulates to: landing plane tilt + yaw reference."""
        if self.landing_reference is not None:
            return self.landing_reference.reference_quaternion(self.yaw_reference)
        half = 0.5 * self.yaw_reference
        return (math.cos(half), 0.0, 0.0, math.sin(half))

    def step(self) -> None:
        self._update_contact_phase(self.robot_touches_terrain())
        if not self.contact_gating:
            # With gating the wheel mode follows the contact phase instead.
            self._update_wheel_rest_state()
        if self.landing_reference is not None:
            self.landing_reference.step(self.data, airborne=not self.in_contact)
        self._apply_controls()
        mujoco.mj_step(self.model, self.data)

    def _update_contact_phase(self, touching: bool) -> None:
        """Contact starts on the first touching step; flight needs a quiet spell."""
        if touching:
            self._time_without_contact = 0.0
            if not self.in_contact:
                self.in_contact = True
                self.contact_phases += 1
                if self.contact_gating:
                    self._enter_contact_mode()
            return
        self._time_without_contact += float(self.model.opt.timestep)
        if self.in_contact and self._time_without_contact >= FLIGHT_DEBOUNCE:
            self.in_contact = False
            if self.contact_gating:
                self._enter_flight_mode()

    def _enter_contact_mode(self) -> None:
        # The ground now sets the attitude: stop regulating it and use the
        # contact to bleed the wheels' momentum into the terrain instead.
        if self.wheel_control_mode is WheelControlMode.DISABLED:
            return
        self.controller.reset()
        self.wheel_rest_controller.begin(self.read_wheel_speeds())
        self.wheel_control_mode = WheelControlMode.RESTING

    def _enter_flight_mode(self) -> None:
        if self.wheel_control_mode is WheelControlMode.DISABLED:
            return
        self.controller.reset()
        self.wheel_rest_controller.cancel()
        self.wheel_control_mode = WheelControlMode.STABILIZING
        # Hold the heading the robot left the ground with.
        self.yaw_reference = read_simulated_imu(
            self.model, self.data, self.ids.root_joint
        ).yaw

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
                if self.controller_name == "lqr":
                    self.controller.reset()
        else:
            self._stationary_duration = 0.0
            if self.wheel_control_mode is WheelControlMode.RESTING:
                self.wheel_rest_controller.cancel()
                self.wheel_control_mode = WheelControlMode.STABILIZING

    def _apply_controls(self) -> None:
        if self.wheel_control_mode is not WheelControlMode.DISABLED:
            sensor = read_simulated_imu(self.model, self.data, self.ids.root_joint)
            if self.controller_name == "lqr":
                # The LQR works on the body-frame quaternion error, which lines
                # up with the gyro rates and wheel axes at any attitude.
                qpos_address = self.model.jnt_qposadr[self.ids.root_joint]
                quaternion = self.data.qpos[qpos_address + 3 : qpos_address + 7]
                error = quaternion_error(
                    tuple(float(v) for v in quaternion), self.reference_quaternion()
                )
                sensor = replace(sensor, roll=error[0], pitch=error[1], yaw=error[2])
            else:
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
                wheel_speed = float(self.data.qvel[self.model.jnt_dofadr[joint_id]])
                attitude_command = -body_torque
                if self.contact_gating and self.in_contact:
                    attitude_command = 0.0  # gated: only the rest torque acts
                elif self.compensate_wheel_friction:
                    attitude_command += wheel_friction_compensation(
                        wheel_speed, self.tuning
                    )
                command = combine_attitude_and_rest_torque(
                    attitude_command, rest_torque, self.tuning
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


def _robot_geom_mask(model: mujoco.MjModel, root_joint_id: int) -> np.ndarray:
    """Boolean per geom: does it belong to the robot (the root body's subtree)?"""
    root_body = int(model.jnt_bodyid[root_joint_id])
    in_robot = np.zeros(model.nbody, dtype=bool)
    for body in range(model.nbody):
        b = body
        while b > 0 and b != root_body:
            b = int(model.body_parentid[b])
        in_robot[body] = b == root_body
    return in_robot[model.geom_bodyid]
