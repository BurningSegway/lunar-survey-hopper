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
from .flight_phase import FlightPhase, FlightPhaseDetector, gravity_magnitude
from .flight_reference import FlightReference, FlightReferenceConfig
from .jump_aim import JumpAim
from .depth_camera import (
    CAMERA_SITE,
    TERRAIN_GEOM,
    DepthCameraSpec,
    HeightfieldRayCaster,
    MeshRayCaster,
    SimulatedDepthCamera,
)
from .landing_reference import (
    LandingReference,
    LandingReferenceConfig,
    estimate_ground_plane,
    frame_points_world,
    quaternion_to_matrix,
)
from .pointcloud import PointCloudRecorder
from .telemetry import TelemetryPayload, make_telemetry_payload

EARTH_GRAVITY = 9.81
MOON_GRAVITY = 1.62
JUMP_RELEASE_DURATION = 0.4
JUMP_CYCLE_DURATION = 3.0
# Must exceed the jump spring's force when wound; keep within the
# wind_up_motor ctrlrange in the MJCF.
JUMP_WINDUP_FORCE = -13000.0
# Contact gating: the robot counts as airborne again only after this long
# without terrain contact, so rocking on the legs or a skid reads as one
# contact phase. Real bounces in Moon gravity last >= ~0.1 s.
FLIGHT_DEBOUNCE = 0.06
TARGET_GEOM = "target_object"
TARGET_PARKED_DEPTH = 100.0  # m below the terrain while no target is set
# Tipped over (on the ground): body tilt beyond TIPPED_TILT; upright again
# below UPRIGHT_TILT. While tipped the legs are pulled in as far as they go.
TIPPED_TILT = math.radians(60.0)
UPRIGHT_TILT = math.radians(45.0)
TIPPED_LEG_FORCE_MARGIN = 1.2  # x spring force at full stroke


LEGS = ("fl", "fr", "bl", "br")


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
    leg_actuators: tuple[int, int, int, int]  # fl, fr, bl, br
    leg_joints: tuple[int, int, int, int]


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
        leg_actuators=tuple(
            _required_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"tilt_{leg}")
            for leg in LEGS
        ),
        leg_joints=tuple(
            _required_id(model, mujoco.mjtObj.mjOBJ_JOINT, f"shock_{leg}")
            for leg in LEGS
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
            self._pending.append((now + self._latency, frame, np.array(frame.body_position)))
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
        flight_reference: bool = True,
        track_target: tuple[float, float, float] | None = None,
        target_scan: PointCloudRecorder | None = None,
        gravity: float = MOON_GRAVITY,
    ) -> None:
        self.model = model
        self.data = data
        self.ids = find_model_ids(model)
        self.tuning = tuning
        self.default_gravity = gravity
        self.selected_gravity = gravity
        self.model.opt.gravity[2] = -gravity

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
        # LQR + camera + gating: ascent/descent from the IMU accelerometer, and
        # the hop reference (track a point going up, level/plane coming down).
        self.flight_phase = FlightPhaseDetector()
        self._previous_velocity: np.ndarray | None = None
        self._camera_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, CAMERA_SITE)
        self.flight_reference: FlightReference | None = None
        if flight_reference and self.landing_reference is not None and self.contact_gating:
            self.flight_reference = FlightReference(FlightReferenceConfig(), track_target)
        self._jump_command = False  # given since the last phase update
        self._frozen_at_descent = 0  # landing-filter freezes before this descent
        self.flight_events: list[str] = []
        # LQR + hop reference: on the ground, lean the body axis towards the
        # track target (closed loop on the IMU attitude, by shortening legs) so
        # the next jump heads towards it. Needs a set target (--track-target).
        self.jump_aim: JumpAim | None = (
            JumpAim(_leg_feet(model, self.ids.leg_joints))
            if self.flight_reference is not None else None
        )
        self._leg_stiffness = np.array(
            [float(model.jnt_stiffness[j]) for j in self.ids.leg_joints]
        )
        self.leg_shortening = np.zeros(len(LEGS))
        self._aim_announced = False
        self._leg_stroke = np.array([float(model.jnt_range[j][1]) for j in self.ids.leg_joints])
        self.tipped = False
        # The scan target object, if the model has one: placed at the target.
        self._target_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, TARGET_GEOM)
        # MuJoCo centres a mesh on its centre of mass: the geom sits at the
        # mesh's offset (mesh_pos) from the asset origin (the object's base).
        # Taken from the mesh, as other simulations on this model move the geom.
        self._target_offset = (
            model.mesh_pos[model.geom_dataid[self._target_geom]].copy()
            if self._target_geom >= 0 else None
        )
        self._place_target_object(track_target)
        # Optional: point cloud of the target from on-target ascent frames.
        # Own camera instance (full frames, own noise stream) so the landing
        # fit sees exactly the same frames whether scanning is on or not.
        self.target_scan: PointCloudRecorder | None = None
        self._scan_camera: SimulatedDepthCamera | None = None
        self._scan_next = 0.0
        self._scan_open = False  # a tracked ascent is collecting points
        self._scan_points_before = 0
        if target_scan is not None and self.flight_reference is not None:
            self.target_scan = target_scan
            self._scan_camera = SimulatedDepthCamera(
                model, self.ids.root_joint,
                replace(self.landing_reference.camera.spec,
                        width=target_scan.config.width, height=target_scan.config.height),
                ray_caster=self.landing_reference.camera._ray_caster, seed=1,
            )
        self.wheel_rest_controller = WheelRestController(tuning)
        self.wheel_control_mode = WheelControlMode.STABILIZING
        self._stationary_duration = 0.0
        self.yaw_reference = 0.0
        self.jump_triggered = False
        self.jump_start_time = 0.0

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
                self._jump_command = True  # GROUND -> TAKEOFF
                return "[JUMP INITIATED] Releasing spring!"
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
        self.model.opt.gravity[2] = -self.default_gravity
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
        self.tipped = False
        self.selected_gravity = self.default_gravity
        self.in_contact = False
        self._time_without_contact = math.inf
        self.contact_phases = 0
        self.flight_phase.reset()
        self._previous_velocity = None
        self._jump_command = False
        self._frozen_at_descent = 0
        self._close_scan()  # keep what the run collected; the file stays as saved
        self.flight_events.clear()
        self.leg_shortening[:] = 0.0
        self._aim_announced = False
        if self.jump_aim is not None:
            self.jump_aim.reset()
        if self.flight_reference is not None:
            self.flight_reference.commanded = (1.0, 0.0, 0.0, 0.0)
            self.flight_reference.tracking = False
            self.flight_reference.active_target = None

    def set_track_target(self, target: tuple[float, float, float] | None) -> None:
        """World point the camera tracks during ascent (None: default point)."""
        if self.flight_reference is not None:
            self.flight_reference.set_target(target)
        self._place_target_object(target)

    def _place_target_object(self, target) -> None:
        """Stand the target object on the terrain at ``target`` (hide it for None)."""
        if self._target_geom < 0:
            return
        terrain = self._terrain_ray_caster()
        if terrain is not None:
            terrain.objects = []
        if target is None or terrain is None:
            self.model.geom_pos[self._target_geom] = self._target_offset - (0.0, 0.0, TARGET_PARKED_DEPTH)
            return
        x, y = float(target[0]), float(target[1])
        # Base on the lowest ground under the object, so it never floats.
        ring = np.linspace(0.0, 2.0 * math.pi, 12, endpoint=False)
        reach = 0.5 * float(np.ptp(self._target_footprint(), axis=0).max())
        xs = np.append(x + reach * np.cos(ring), x)
        ys = np.append(y + reach * np.sin(ring), y)
        base = float(np.nanmin(terrain.height(xs, ys)))
        self.model.geom_pos[self._target_geom] = self._target_offset + (x, y, base)
        terrain.objects = [MeshRayCaster.from_model(self.model, self._target_geom)]

    def _target_footprint(self) -> np.ndarray:
        mesh = int(self.model.geom_dataid[self._target_geom])
        adr, num = int(self.model.mesh_vertadr[mesh]), int(self.model.mesh_vertnum[mesh])
        return self.model.mesh_vert[adr : adr + num, :2]

    def _terrain_ray_caster(self) -> HeightfieldRayCaster | None:
        """The ray caster the depth cameras share (None without a camera)."""
        if self.landing_reference is None:
            return None
        caster = self.landing_reference.camera._ray_caster
        return caster if isinstance(caster, HeightfieldRayCaster) else None

    @property
    def phase(self) -> FlightPhase:
        return self.flight_phase.phase

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
        """Attitude the LQR regulates to.

        In flight with the hop reference: the rate-limited flight reference
        (target tracking on the way up, level then landing plane on the way
        down). Otherwise: landing plane tilt + yaw reference.
        """
        if self.flight_reference is not None and not self.in_contact:
            return self.flight_reference.commanded
        if self.landing_reference is not None:
            return self.landing_reference.reference_quaternion(self.yaw_reference)
        half = 0.5 * self.yaw_reference
        return (math.cos(half), 0.0, 0.0, math.sin(half))

    def step(self) -> None:
        self._update_contact_phase(self.robot_touches_terrain())
        if not self.contact_gating:
            # With gating the wheel mode follows the contact phase instead.
            self._update_wheel_rest_state()
        phase = self._update_flight_phase()
        if self.landing_reference is not None:
            # Without the hop reference: fit the ground whenever airborne.
            # With it: only on the way down (on the way up the camera is busy
            # tracking the target and looks at the ground at an angle).
            capture = (
                phase is FlightPhase.DESCENT
                if self.flight_reference is not None
                else not self.in_contact
            )
            accepted = self.landing_reference.filter.accepted
            self.landing_reference.step(self.data, airborne=capture)
            if (self.flight_reference is not None and accepted == 0
                    and self.landing_reference.filter.accepted > 0):
                self.flight_events.append("[FLIGHT] Ground seen: aligning with the landing plane.")
        if self.flight_reference is not None and not self.in_contact:
            self.flight_reference.update(
                phase,
                self.data.site_xpos[self._camera_site],
                self.yaw_reference,
                self.landing_reference.reference_quaternion(self.yaw_reference),
                float(self.model.opt.timestep),
            )
        self._update_tipped()
        self._update_jump_aim()
        if self._scan_open:
            if phase is FlightPhase.ASCENT and not self.in_contact:
                self._scan_step()
            elif phase is not FlightPhase.TAKEOFF:  # still on its way up to ASCENT
                self._close_scan()
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
            if self.flight_reference is not None:
                qa = self.model.jnt_qposadr[self.ids.root_joint]
                track = self.phase is FlightPhase.TAKEOFF  # a jump, not a bounce
                self.flight_reference.start_flight(
                    tuple(float(v) for v in self.data.qpos[qa + 3 : qa + 7]),
                    self.data.qpos[qa : qa + 3].copy(),
                    self.yaw_reference,
                    track,
                )
                if track:
                    if self.target_scan is not None:
                        self.target_scan.start_jump()
                        self._scan_open = True
                        self._scan_next = float(self.data.time)
                        self._scan_points_before = self.target_scan.point_count

    def _update_tipped(self) -> None:
        """Tipped over on the ground: pull the legs in as far as they go.

        Only an attempt: with the current hardware neither the legs nor the
        reaction wheels can right the robot (see README).
        """
        if self.flight_reference is None:
            return
        body = self.model.jnt_bodyid[self.ids.root_joint]
        tilt = math.acos(max(-1.0, min(1.0, float(self.data.xmat[body][8]))))
        if not self.tipped and self.phase is FlightPhase.GROUND and tilt > TIPPED_TILT:
            self.tipped = True
            self.flight_events.append(
                f"[TIPPED] Tipped over ({math.degrees(tilt):.0f} deg): legs pulled in."
            )
        elif self.tipped and tilt < UPRIGHT_TILT:
            self.tipped = False
            self.flight_events.append("[TIPPED] Upright again.")

    def _update_jump_aim(self) -> None:
        """Leg shortening that leans the body axis towards the target (ground only)."""
        target = None if self.flight_reference is None else self.flight_reference.target_setting
        phase = self.phase
        if self.jump_aim is None or target is None:
            return
        if self.tipped or phase not in (FlightPhase.GROUND, FlightPhase.TAKEOFF):
            # Straight legs from lift-off until the robot has settled again.
            self.jump_aim.reset()
            self.leg_shortening = self.jump_aim.shortening
            self._aim_announced = False
            return
        body = self.model.jnt_bodyid[self.ids.root_joint]
        rotation = self.data.xmat[body].reshape(3, 3)
        position = self.data.xpos[body]
        dof = self.model.jnt_dofadr[self.ids.root_joint]
        self.leg_shortening = self.jump_aim.update(
            rotation, self.data.qvel[dof + 3 : dof + 6], position, target,
            float(self.model.opt.timestep), hold=phase is FlightPhase.TAKEOFF,
        )
        if phase is FlightPhase.TAKEOFF:
            return
        lean = self.jump_aim.lean_towards(rotation, position, target)
        still = float(np.linalg.norm(self.data.qvel[dof + 3 : dof + 6])) < 0.05
        reached = abs(lean - self.jump_aim.desired) < math.radians(1.0)
        if not self._aim_announced and still and (reached or self.jump_aim.limited):
            self._aim_announced = True
            distance = math.hypot(target[0] - position[0], target[1] - position[1])
            note = "" if reached else (
                f" (wanted {math.degrees(self.jump_aim.desired):.1f}; legs at their limit)"
            )
            self.flight_events.append(
                f"[AIM] Target {distance:.1f} m away: leaning {math.degrees(lean):.1f} deg "
                f"towards it{note}."
            )

    def _scan_step(self) -> None:
        """Capture a full frame at the camera rate; keep it if on target."""
        now = float(self.data.time)
        if now < self._scan_next:
            return
        self._scan_next = now + 1.0 / self._scan_camera.spec.frame_rate
        # Pose as the camera will cast from it (kinematics, not the newer qpos).
        body = self.model.jnt_bodyid[self.ids.root_joint]
        error = self.flight_reference.camera_pointing_error(
            tuple(float(v) for v in self.data.xquat[body]), self.data.site_xpos[self._camera_site]
        )
        if error > self.target_scan.config.margin:
            return  # still turning towards the target: skip without ray casting
        frame = self._scan_camera.capture(self.data)
        self.target_scan.add_frame(frame, self._scan_camera.mount, error)

    def _close_scan(self) -> None:
        """End of a tracked ascent: write the cloud (all jumps so far)."""
        if not self._scan_open:
            return
        self._scan_open = False
        scan = self.target_scan
        try:
            scan.save()
        except OSError as error:
            self.flight_events.append(f"[SCAN] Could not write {scan.path}: {error}")
            return
        added = scan.point_count - self._scan_points_before
        self.flight_events.append(
            f"[SCAN] Jump {scan.jumps}: +{added} points on target; "
            f"{scan.point_count} total saved to {scan.path}."
        )

    def _update_flight_phase(self) -> FlightPhase:
        before = self.flight_phase.phase
        qa = self.model.jnt_qposadr[self.ids.root_joint]
        quaternion = tuple(float(v) for v in self.data.qpos[qa + 3 : qa + 7])
        # Simulated accelerometer: the body's actual velocity change over the
        # last step, as specific force in the body frame. MuJoCo's own
        # accelerometer sensor (imu_accel) reports the explicit acceleration,
        # which with this model's heavily damped jump joint (implicit damping
        # in the Euler integrator) does not match the velocity change -- it
        # put the integrated vertical velocity ~1 m/s off. A real IMU measures
        # the true motion, so the finite difference is the faithful stand-in.
        dof = self.model.jnt_dofadr[self.ids.root_joint]
        velocity = self.data.qvel[dof : dof + 3].copy()
        dt = float(self.model.opt.timestep)
        acceleration = (
            np.zeros(3) if self._previous_velocity is None
            else (velocity - self._previous_velocity) / dt
        )
        self._previous_velocity = velocity
        specific_force = quaternion_to_matrix(quaternion).T @ (acceleration - self.model.opt.gravity)
        touchdown_near = (
            self.landing_reference is not None
            and self.landing_reference.filter.frozen > self._frozen_at_descent
        )
        phase = self.flight_phase.update(
            specific_force, quaternion, gravity_magnitude(self.model.opt.gravity),
            float(self.model.opt.timestep), self.in_contact,
            body_rate=self.data.qvel[dof + 3 : dof + 6],
            jump=self._jump_command,
            touchdown_near=touchdown_near,
        )
        self._jump_command = False
        if phase is not before:
            self._enter_phase(phase)
        return phase

    def _enter_phase(self, phase: FlightPhase) -> None:
        if phase is FlightPhase.DESCENT and self.landing_reference is not None:
            # Only a freeze from this descent on means touchdown is near.
            self._frozen_at_descent = self.landing_reference.filter.frozen
        if self.flight_reference is None:
            return
        if phase is FlightPhase.ASCENT and self.flight_reference.tracking:
            t = self.flight_reference.active_target
            message = f"[FLIGHT] Ascent: camera tracking target ({t[0]:.2f}, {t[1]:.2f}, {t[2]:.2f})."
        else:
            message = {
                FlightPhase.GROUND: "[PHASE] On the ground.",
                FlightPhase.TAKEOFF: "[PHASE] Take-off.",
                FlightPhase.ASCENT: "[PHASE] Ascent.",
                FlightPhase.DESCENT: "[FLIGHT] Descent: levelling until the camera sees the ground.",
                FlightPhase.LANDING: "[PHASE] Landing: touchdown close."
                if not self.in_contact else "[PHASE] Landing: ground contact.",
            }[phase]
        self.flight_events.append(message)

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

        # Leg force = spring stiffness x shortening (open loop); tipped over:
        # pull every leg in to the end of its stroke.
        leg_forces = self._leg_stiffness * (
            self._leg_stroke * TIPPED_LEG_FORCE_MARGIN if self.tipped else self.leg_shortening
        )
        for actuator_id, force in zip(self.ids.leg_actuators, leg_forces, strict=True):
            self.data.ctrl[actuator_id] = float(force)

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


def _leg_feet(model: mujoco.MjModel, leg_joints) -> np.ndarray:
    """Lower end of each leg's capsule in the root body frame (legs at rest)."""
    feet = []
    for joint in leg_joints:
        body = int(model.jnt_bodyid[joint])
        geom = int(np.flatnonzero(model.geom_bodyid == body)[0])
        rotation = np.zeros(9)
        mujoco.mju_quat2Mat(rotation, model.geom_quat[geom])
        axis = rotation.reshape(3, 3)[:, 2] * model.geom_size[geom][1]
        ends = (model.geom_pos[geom] + axis, model.geom_pos[geom] - axis)
        feet.append(model.body_pos[body] + min(ends, key=lambda end: end[2]))
    return np.array(feet)


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

