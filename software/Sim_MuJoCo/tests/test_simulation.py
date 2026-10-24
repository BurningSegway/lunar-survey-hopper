
from dataclasses import astuple, replace
from pathlib import Path
import json
import math
import socket
import tempfile
import unittest

import mujoco
import numpy as np

from jumper_sim.simulation import (
    EARTH_GRAVITY,
    FLIGHT_DEBOUNCE,
    MOON_GRAVITY,
    JumperSimulation,
    WheelControlMode,
)
from jumper_sim.controller import (
    ReactionWheelTuning,
    WheelSpeeds,
    limit_wheel_command,
    robot_inertia,
    wheel_effective_inertia,
)
from jumper_sim.depth_camera import (
    TERRAIN_GEOM_GROUP,
    DepthCameraSpec,
    HeightfieldRayCaster,
    SimulatedDepthCamera,
)
from jumper_sim.flight_phase import FlightPhase
from jumper_sim.pointcloud import PointCloudRecorder, ScanConfig, read_ply
from jumper_sim.telemetry import UdpTelemetry
from jumper_sim.visualization import draw_landing_plane

def mujoco_ray_distances(model, data, origin, directions, max_range):
    """Reference distances from MuJoCo's mj_multiRay (NaN for no hit)."""
    n = len(directions)
    group = np.zeros(6, dtype=np.uint8)
    group[TERRAIN_GEOM_GROUP] = 1
    geom_ids = np.zeros(n, dtype=np.int32)
    dist = np.zeros(n)
    vec = np.ascontiguousarray(directions, dtype=float).ravel()
    pnt = np.asarray(origin, dtype=float)
    try:  # MuJoCo >= 3.3 has an extra optional "normal" output.
        mujoco.mj_multiRay(model, data, pnt, vec, group, True, -1, geom_ids, dist, None, n, max_range)
    except TypeError:
        mujoco.mj_multiRay(model, data, pnt, vec, group, True, -1, geom_ids, dist, n, max_range)
    dist[(geom_ids < 0) | (dist > max_range)] = np.nan
    return dist


MODEL_PATH = Path(__file__).resolve().parents[1] / "build" / "Jumping_Robot.xml"
TERRAIN_ASSETS = (
    MODEL_PATH.parent / "terrain.obj",
    MODEL_PATH.parent / "terrain_height.bin",
)


@unittest.skipUnless(
    MODEL_PATH.is_file() and all(path.is_file() for path in TERRAIN_ASSETS),
    "Generate build/terrain.obj and terrain_height.bin from "
    "build/convert_usd.py before running model tests.",
)
class SimulationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        self.data = mujoco.MjData(self.model)
        self.simulation = JumperSimulation(self.model, self.data, "pid")

    def test_lqr_uses_whole_robot_inertia_from_model(self) -> None:
        simulation = JumperSimulation(self.model, self.data, "lqr")
        inertia = simulation.controller.axis_inertia
        rigid = robot_inertia(
            self.model, simulation.ids.root_joint, simulation.ids.wheel_joints
        )
        np.testing.assert_allclose(rigid.wheel_spin, (0.0036,) * 3)
        # Box 0.075 + two off-axis wheels (0.0036 + 0.2*0.125^2 each) + legs
        # and the spherical foot; the axis's own wheel spin inertia is excluded.
        np.testing.assert_allclose(inertia, (0.1050, 0.1050, 0.0985), atol=5e-4)

    def test_friction_tuning_matches_model_wheel_damping(self) -> None:
        for joint_id in self.simulation.ids.wheel_joints:
            self.assertAlmostEqual(
                float(self.model.dof_damping[self.model.jnt_dofadr[joint_id]]),
                self.simulation.tuning.wheel_viscous_friction,
            )

    def test_heightfield_ray_caster_matches_mujoco(self) -> None:
        mujoco.mj_forward(self.model, self.data)
        fast = HeightfieldRayCaster(self.model)
        rng = np.random.default_rng(4)
        for origin in ((0.0, 0.0, 3.0), (5.0, -3.0, 8.0), (-12.0, 7.0, 1.0)):
            directions = rng.standard_normal((400, 3))
            directions[:, 2] = -np.abs(directions[:, 2])
            directions /= np.linalg.norm(directions, axis=1, keepdims=True)
            a = fast.cast(np.array(origin), directions, 15.0)
            b = mujoco_ray_distances(self.model, self.data, origin, directions, 15.0)
            np.testing.assert_array_equal(np.isfinite(a), np.isfinite(b))
            np.testing.assert_allclose(a[np.isfinite(a)], b[np.isfinite(b)], atol=1e-6)

    def test_depth_camera_sees_ground_only_within_d435_range(self) -> None:
        camera = SimulatedDepthCamera(
            self.model, self.simulation.ids.root_joint, DepthCameraSpec(noise=False)
        )
        caster = HeightfieldRayCaster(self.model)
        ground = float(caster.height(np.array([0.0]), np.array([0.0]))[0])
        for height, expect_valid in ((2.0, True), (0.35, False)):
            self.data.qpos[:3] = (0.0, 0.0, ground + height)
            self.data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
            mujoco.mj_forward(self.model, self.data)
            frame = camera.capture(self.data)
            self.assertEqual(frame.depth.shape, (60, 106))
            centre = frame.depth[30, 53]
            if expect_valid:
                self.assertAlmostEqual(centre, height - 0.151, delta=0.15)
            else:
                self.assertTrue(np.isnan(frame.depth).mean() > 0.9)

    def test_landing_reference_updates_only_in_flight(self) -> None:
        simulation = JumperSimulation(self.model, self.data, "lqr")
        self.assertIsNotNone(simulation.landing_reference)
        caster = HeightfieldRayCaster(self.model)
        x, y = 5.0, -3.0  # sloped spot
        ground = float(caster.height(np.array([x]), np.array([y]))[0])
        self.model.opt.gravity[2] = -MOON_GRAVITY
        self.data.qpos[:3] = (x, y, ground + 3.0)
        mujoco.mj_forward(self.model, self.data)
        simulation.landing_reference.reset(float(self.data.time))
        self.assertFalse(simulation.robot_touches_terrain())
        while not simulation.robot_touches_terrain() and self.data.time < 4.0:
            simulation.step()
        filt = simulation.landing_reference.filter
        self.assertGreater(filt.accepted, 5)
        self.assertGreater(math.degrees(math.acos(filt.normal[2])), 5.0)
        # No depth frame is ever captured while the robot is in contact
        # (bounces are flights and may capture).
        camera = simulation.landing_reference.camera
        original_capture = camera.capture
        captures_in_contact = []

        def capture(data, roi_radius=None):
            captures_in_contact.append(simulation.in_contact)
            return original_capture(data, roi_radius)

        camera.capture = capture
        for _ in range(1500):
            simulation.step()
        self.assertNotIn(True, captures_in_contact)
        telemetry = simulation.telemetry_payload()
        self.assertNotEqual((telemetry["roll_ref"], telemetry["pitch_ref"]), (0.0, 0.0))
        simulation.handle_key("space")
        np.testing.assert_allclose(filt.normal, (0.0, 0.0, 1.0))

    def _drop(self, simulation, height=1.5, x=0.0, y=0.0):
        caster = HeightfieldRayCaster(self.model)
        ground = float(caster.height(np.array([x]), np.array([y]))[0])
        self.data.qpos[:3] = (x, y, ground + height)
        self.data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
        mujoco.mj_forward(self.model, self.data)
        if simulation.landing_reference is not None:
            simulation.landing_reference.reset(float(self.data.time))

    def test_contact_gating_switches_modes_and_bleeds_wheels(self) -> None:
        simulation = JumperSimulation(self.model, self.data, "lqr")
        self.assertTrue(simulation.contact_gating)
        self.model.opt.gravity[2] = -MOON_GRAVITY
        self._drop(simulation)
        for joint_id, speed in zip(simulation.ids.wheel_joints, (30.0, -20.0, 10.0)):
            self.data.qvel[self.model.jnt_dofadr[joint_id]] = speed
        simulation.step()
        self.assertFalse(simulation.in_contact)
        self.assertIs(simulation.wheel_control_mode, WheelControlMode.STABILIZING)
        while not simulation.in_contact and self.data.time < 3.0:
            simulation.step()
        self.assertTrue(simulation.in_contact)
        self.assertIs(simulation.wheel_control_mode, WheelControlMode.RESTING)
        # In contact only the rest torque acts: it opposes each wheel's spin.
        simulation.step()
        for actuator_id, joint_id in zip(simulation.ids.wheel_actuators, simulation.ids.wheel_joints):
            speed = self.data.qvel[self.model.jnt_dofadr[joint_id]]
            if abs(speed) > 1.0:
                self.assertLessEqual(self.data.ctrl[actuator_id] * speed, 0.0)
        for _ in range(6000):
            simulation.step()
        if simulation.in_contact:
            self.assertLess(max(abs(v) for v in astuple(simulation.read_wheel_speeds())), 1.0)

    def test_short_contact_gaps_do_not_count_as_flight(self) -> None:
        simulation = JumperSimulation(self.model, self.data, "lqr")
        simulation._update_contact_phase(True)
        self.assertTrue(simulation.in_contact)
        gap_steps = int(0.5 * FLIGHT_DEBOUNCE / self.model.opt.timestep)
        for _ in range(gap_steps):
            simulation._update_contact_phase(False)
        self.assertTrue(simulation.in_contact)
        simulation._update_contact_phase(True)
        for _ in range(int(1.2 * FLIGHT_DEBOUNCE / self.model.opt.timestep)):
            simulation._update_contact_phase(False)
        self.assertFalse(simulation.in_contact)
        self.assertIs(simulation.wheel_control_mode, WheelControlMode.STABILIZING)
        self.assertEqual(simulation.contact_phases, 1)

    def test_lift_off_holds_current_heading(self) -> None:
        simulation = JumperSimulation(self.model, self.data, "lqr")
        yaw = 0.7
        self.data.qpos[3:7] = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
        mujoco.mj_forward(self.model, self.data)
        simulation._update_contact_phase(True)
        for _ in range(int(1.5 * FLIGHT_DEBOUNCE / self.model.opt.timestep)):
            simulation._update_contact_phase(False)
        self.assertAlmostEqual(simulation.yaw_reference, yaw, places=9)

    def test_disabled_wheels_stay_disabled_through_contact_changes(self) -> None:
        simulation = JumperSimulation(self.model, self.data, "lqr")
        simulation.handle_key("x")
        simulation._update_contact_phase(True)
        self.assertIs(simulation.wheel_control_mode, WheelControlMode.DISABLED)
        simulation.handle_key("c")
        self.assertIs(simulation.wheel_control_mode, WheelControlMode.RESTING)

    def test_pid_is_not_contact_gated(self) -> None:
        self.assertFalse(self.simulation.contact_gating)

    def test_landing_plane_overlay_is_drawn(self) -> None:
        simulation = JumperSimulation(self.model, self.data, "lqr")
        self.model.opt.gravity[2] = -MOON_GRAVITY
        self._drop(simulation, height=2.5, x=5.0, y=-3.0)
        scene = mujoco.MjvScene(self.model, maxgeom=500)
        self.assertEqual(draw_landing_plane(scene, simulation), 0)  # nothing fitted yet
        while self.data.time < 0.5:
            simulation.step()
        added = draw_landing_plane(scene, simulation)
        self.assertGreater(added, 2)
        disc = scene.geoms[0]
        self.assertEqual(disc.type, mujoco.mjtGeom.mjGEOM_CYLINDER)
        np.testing.assert_allclose(
            disc.mat[:, 2], simulation.landing_reference.visual.measured_normal, atol=1e-6
        )
        small = mujoco.MjvScene(self.model, maxgeom=3)
        self.assertLessEqual(draw_landing_plane(small, simulation), 3)

    def test_pid_has_no_landing_reference(self) -> None:
        self.assertIsNone(self.simulation.landing_reference)

    def test_terrain_is_firm_and_overdamped(self) -> None:
        terrain = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "terrain_collision")
        self.assertEqual(self.model.geom_priority[terrain], 1)
        self.assertAlmostEqual(float(self.model.geom_friction[terrain][0]), 0.7)
        # A soft 0.03 s contact let landing legs sink 8-17 cm and get stuck.
        self.assertLessEqual(float(self.model.geom_solref[terrain][0]), 0.01)
        self.assertGreater(float(self.model.geom_solref[terrain][1]), 1.0)  # overdamped

    def test_leans_towards_target_on_the_ground_from_the_imu(self) -> None:
        target = (-30.0, 0.0, 0.0)
        simulation = JumperSimulation(self.model, self.data, "lqr", track_target=target,
                                      gravity=EARTH_GRAVITY)
        while self.data.time < 3.0:
            simulation.step()
        self.assertTrue(simulation.in_contact)
        body = self.model.jnt_bodyid[simulation.ids.root_joint]
        lean = simulation.jump_aim.lean_towards(
            self.data.xmat[body].reshape(3, 3), self.data.xpos[body], target)
        self.assertGreater(lean, math.radians(2.0))  # towards the target
        self.assertLessEqual(lean, simulation.jump_aim.config.max_lean + math.radians(1.0))
        self.assertTrue(any("[AIM]" in e for e in simulation.flight_events))
        np.testing.assert_allclose(
            self.data.ctrl[list(simulation.ids.leg_actuators)],
            simulation._leg_stiffness * simulation.leg_shortening,
        )

    def test_target_object_stands_at_target_and_the_camera_sees_it(self) -> None:
        simulation = JumperSimulation(self.model, self.data, "lqr", track_target=(8.0, 3.0, 0.0))
        mujoco.mj_forward(self.model, self.data)
        target = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "target_object")
        caster = simulation._terrain_ray_caster()
        self.assertEqual(len(caster.objects), 1)
        # Same distances as MuJoCo's own ray cast against the object.
        groups = self.model.geom_group.copy()
        self.model.geom_group[:] = 5
        self.model.geom_group[target] = 0
        only_target = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
        rng = np.random.default_rng(2)
        origin = np.array([8.0, 3.0, 6.0])
        hits = 0
        for direction in rng.normal([0, 0, -1], 0.15, (100, 3)):
            direction /= np.linalg.norm(direction)
            geom = np.zeros(1, dtype=np.int32)
            expected = mujoco.mj_ray(self.model, self.data, origin, direction, only_target, 1, -1, geom)
            ours = caster.objects[0].cast(origin, direction[None], 20.0)[0]
            if expected < 0:
                self.assertTrue(np.isnan(ours))
            else:
                hits += 1
                self.assertAlmostEqual(ours, expected, places=6)
        self.model.geom_group[:] = groups
        self.assertGreater(hits, 10)
        # It stands on the terrain: its lowest point is at most a little below it.
        ground = caster.height(np.array([8.0]), np.array([3.0]))[0]
        bottom = caster.objects[0].centre[2] - (caster.objects[0].top - caster.objects[0].centre[2])
        self.assertLess(abs(bottom - ground), 0.5)

    def test_target_object_is_hidden_without_a_target(self) -> None:
        JumperSimulation(self.model, self.data, "lqr")
        mujoco.mj_forward(self.model, self.data)
        target = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "target_object")
        self.assertLess(self.data.geom_xpos[target][2], -50.0)

    def test_tipped_robot_pulls_its_legs_in(self) -> None:
        simulation = JumperSimulation(self.model, self.data, "lqr")
        quaternion = np.zeros(4)
        mujoco.mju_euler2Quat(quaternion, np.array([0.0, math.radians(90), 0.0]), "xyz")
        self.data.qpos[3:7] = quaternion
        self.data.qpos[2] = 0.4
        while self.data.time < 4.0 and not simulation.tipped:
            simulation.step()
        self.assertTrue(simulation.tipped)
        self.assertIs(simulation.phase, FlightPhase.GROUND)
        self.assertTrue(any("[TIPPED]" in e for e in simulation.flight_events))
        simulation.step()
        forces = self.data.ctrl[list(simulation.ids.leg_actuators)]
        np.testing.assert_allclose(forces, 1500.0 * 0.12 * 1.2)

    def test_hop_tracks_target_on_ascent_and_fits_ground_only_on_descent(self) -> None:
        target = (2.0, 0.5, 0.0)
        simulation = JumperSimulation(self.model, self.data, "lqr", track_target=target,
                                      gravity=EARTH_GRAVITY)  # hop fits the test window
        self.assertIsNotNone(simulation.flight_reference)
        camera = simulation.landing_reference.camera
        original_capture = camera.capture
        capture_phases = []

        def capture(data, roi_radius=None):
            capture_phases.append(simulation.phase)
            return original_capture(data, roi_radius)

        camera.capture = capture
        while self.data.time < 2.0:
            simulation.step()
        capture_phases.clear()
        simulation.handle_key("space")
        phases, errors = [], []
        while self.data.time < 6.0:
            simulation.step()
            if not phases or phases[-1] is not simulation.phase:
                phases.append(simulation.phase)
            if simulation.phase is FlightPhase.ASCENT:
                errors.append(simulation.flight_reference.camera_pointing_error(
                    tuple(self.data.qpos[3:7]), self.data.site_xpos[simulation._camera_site]))
        self.assertEqual(phases[:4], [FlightPhase.TAKEOFF, FlightPhase.ASCENT,
                                      FlightPhase.DESCENT, FlightPhase.LANDING])
        np.testing.assert_allclose(simulation.flight_reference.active_target, target)
        self.assertLess(math.degrees(errors[-1]), 5.0)  # on target at the apex
        self.assertTrue(capture_phases)
        self.assertEqual(set(capture_phases), {FlightPhase.DESCENT})
        self.assertGreater(simulation.landing_reference.filter.accepted, 0)
        self.assertTrue(any("Ascent" in e for e in simulation.flight_events))
        self.assertTrue(any("Descent" in e for e in simulation.flight_events))

    def test_target_scan_collects_on_target_ascent_points_across_jumps(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "scan.ply"
            recorder = PointCloudRecorder(path, ScanConfig(margin=math.radians(5)))
            simulation = JumperSimulation(self.model, self.data, "lqr", gravity=EARTH_GRAVITY,
                                          track_target=(2.0, 0.5, 0.0), target_scan=recorder)
            self.assertIs(simulation.target_scan, recorder)
            # Noise-free so the geometry can be checked exactly.
            spec = replace(simulation._scan_camera.spec, noise=False)
            simulation._scan_camera = SimulatedDepthCamera(self.model, simulation.ids.root_joint, spec)
            camera = simulation._scan_camera
            original_capture = camera.capture
            captures = []

            def capture(data, roi_radius=None):
                body = self.model.jnt_bodyid[simulation.ids.root_joint]
                captures.append((simulation.phase, simulation.flight_reference.camera_pointing_error(
                    tuple(data.xquat[body]), data.site_xpos[simulation._camera_site])))
                return original_capture(data, roi_radius)

            camera.capture = capture
            recorder.start_run()
            for jump in (1, 2):
                if jump == 2:
                    simulation.reset()  # the cloud survives a reset (R key)
                while self.data.time < 2.0:
                    simulation.step()
                simulation.handle_key("space")
                while self.data.time < 4.0:
                    simulation.step()
                self.assertEqual(len(read_ply(path)), recorder.point_count)  # saved after ascent
            self.assertTrue(captures)
            self.assertEqual({phase for phase, _ in captures}, {FlightPhase.ASCENT})
            self.assertLessEqual(max(error for _, error in captures), math.radians(5))
            cloud = read_ply(path)
            self.assertEqual(sorted(set(cloud["jump"].tolist())), [1, 2])
            self.assertGreater(len(cloud), 10000)
            xyz = np.column_stack([cloud["x"], cloud["y"], cloud["z"]]).astype(float)
            ground = HeightfieldRayCaster(self.model).height(xyz[:, 0], xyz[:, 1])
            self.assertLess(float(np.max(np.abs(xyz[:, 2] - ground))), 1e-3)
            self.assertTrue(any("[SCAN] Jump 2" in e for e in simulation.flight_events))

    def test_target_scan_needs_the_hop_reference(self) -> None:
        recorder = PointCloudRecorder(Path(tempfile.gettempdir()) / "unused.ply")
        simulation = JumperSimulation(self.model, self.data, "pid", target_scan=recorder)
        self.assertIsNone(simulation.target_scan)

    def test_estimated_vertical_velocity_matches_truth_in_flight(self) -> None:
        simulation = JumperSimulation(self.model, self.data, "lqr")
        while self.data.time < 2.0:
            simulation.step()
        simulation.handle_key("space")
        worst = 0.0
        while self.data.time < 3.2:
            # The estimate is updated at the start of a step, from the motion
            # up to then, so compare it with the velocity before the step.
            velocity_before = float(self.data.qvel[2])
            simulation.step()
            if not simulation.in_contact:
                worst = max(worst, abs(simulation.flight_phase.vertical_velocity - velocity_before))
        self.assertLess(worst, 0.05)

    def test_pid_has_no_flight_reference(self) -> None:
        self.assertIsNone(self.simulation.flight_reference)

    def test_model_loads_and_steps_headlessly(self) -> None:
        self.simulation.step()
        self.assertAlmostEqual(self.data.time, self.model.opt.timestep)

    def test_gravity_toggle_and_reset_restore_launch_gravity(self) -> None:
        self.assertEqual(self.model.opt.gravity[2], -MOON_GRAVITY)  # Moon by default
        self.simulation.step()
        qpos_before = self.data.qpos.copy()
        qvel_before = self.data.qvel.copy()
        time_before = self.data.time

        self.simulation.handle_key("backspace")
        self.assertEqual(self.model.opt.gravity[2], -EARTH_GRAVITY)
        self.assertEqual(self.data.time, time_before)
        self.assertEqual(self.data.qpos.tolist(), qpos_before.tolist())
        self.assertEqual(self.data.qvel.tolist(), qvel_before.tolist())

        self.simulation.reset()
        self.assertEqual(self.simulation.selected_gravity, MOON_GRAVITY)
        self.assertEqual(self.model.opt.gravity[2], -MOON_GRAVITY)
        self.simulation.handle_key("backspace")
        self.assertEqual(self.model.opt.gravity[2], -EARTH_GRAVITY)

    def test_model_has_no_separate_plane_collision_geom(self) -> None:
        plane_type = mujoco.mjtGeom.mjGEOM_PLANE
        self.assertNotIn(plane_type, self.model.geom_type)
        visual_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "terrain_visual"
        )
        self.assertGreaterEqual(visual_id, 0)
        self.assertEqual(
            self.model.geom_type[visual_id], mujoco.mjtGeom.mjGEOM_MESH
        )
        self.assertEqual(self.model.geom_contype[visual_id], 0)
        self.assertEqual(self.model.geom_conaffinity[visual_id], 0)

        collision_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "terrain_collision"
        )
        self.assertGreaterEqual(collision_id, 0)
        self.assertEqual(
            self.model.geom_type[collision_id], mujoco.mjtGeom.mjGEOM_HFIELD
        )
        self.assertGreater(self.model.geom_contype[collision_id], 0)
        self.assertGreater(self.model.geom_conaffinity[collision_id], 0)

    def test_reaction_wheels_are_inside_box_and_cannot_contact_ground(self) -> None:
        for name, expected_pos in (
            ("rw_x", (0.125, 0.0, 0.0)),
            ("rw_y", (0.0, 0.125, 0.0)),
            ("rw_z", (0.0, 0.0, 0.125)),
        ):
            body_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, name
            )
            self.assertGreaterEqual(body_id, 0)
            np.testing.assert_allclose(
                self.model.body_pos[body_id], expected_pos
            )

            geom_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, f"{name}_geom"
            )
            self.assertGreaterEqual(geom_id, 0)
            self.assertEqual(self.model.geom_contype[geom_id], 0)
            self.assertEqual(self.model.geom_conaffinity[geom_id], 0)

    def test_heightfield_collision_matches_visual_terrain(self) -> None:
        mujoco.mj_forward(self.model, self.data)
        visual_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "terrain_visual"
        )
        collision_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "terrain_collision"
        )
        self.model.geom_group[:] = 2
        self.model.geom_group[visual_id] = 0
        self.model.geom_group[collision_id] = 1
        visual_group = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
        collision_group = np.array([0, 1, 0, 0, 0, 0], dtype=np.uint8)

        for x, y in ((-20.0, -20.0), (0.0, 0.0), (15.0, 15.0), (-45.0, 40.0), (42.0, -44.0)):
            ray_origin = np.array([x, y, 20.0])
            ray_direction = np.array([0.0, 0.0, -1.0])
            heights = []
            for geom_group in (visual_group, collision_group):
                hit_geom = np.zeros(1, dtype=np.int32)
                distance = mujoco.mj_ray(
                    self.model,
                    self.data,
                    ray_origin,
                    ray_direction,
                    geom_group,
                    1,
                    -1,
                    hit_geom,
                )
                self.assertGreaterEqual(distance, 0.0)
                heights.append(ray_origin[2] - distance)

            self.assertAlmostEqual(heights[0], heights[1], delta=0.25)

    def test_jump_control_updates_actuators(self) -> None:
        self.simulation.handle_key("space")
        self.simulation.step()
        self.assertTrue(self.simulation.jump_triggered)
        # No target to lean towards: the leg motors stay off.
        self.assertEqual(
            self.data.ctrl[list(self.simulation.ids.leg_actuators)].tolist(),
            [0.0, 0.0, 0.0, 0.0],
        )
        self.assertEqual(
            self.data.ctrl[self.simulation.ids.wind_up_actuator], 0.0
        )

    def test_stationary_robot_automatically_restores_wheels(self) -> None:
        time_step = float(self.model.opt.timestep)
        tuning = ReactionWheelTuning(stationary_duration=2.0 * time_step)
        self.simulation = JumperSimulation(
            self.model, self.data, "pid", tuning
        )
        self.simulation.yaw_reference = 0.75
        root_dof = self.model.jnt_dofadr[self.simulation.ids.root_joint]
        self.data.qvel[root_dof : root_dof + 6] = 0.0
        wheel_speeds = (10.0, -5.0, 2.0)
        for joint_id, speed in zip(
            self.simulation.ids.wheel_joints, wheel_speeds, strict=True
        ):
            self.data.qvel[self.model.jnt_dofadr[joint_id]] = speed

        self.simulation._update_wheel_rest_state()
        self.assertEqual(
            self.simulation.wheel_control_mode, WheelControlMode.STABILIZING
        )
        self.simulation._update_wheel_rest_state()
        self.assertEqual(
            self.simulation.wheel_control_mode, WheelControlMode.RESTING
        )
        self.assertTrue(self.simulation.relaxation_active)
        self.assertEqual(self.simulation.yaw_reference, 0.75)
        self.assertEqual(
            self.simulation.wheel_rest_controller.target_speeds,
            WheelSpeeds(*wheel_speeds),
        )

        self.data.qvel[root_dof] = tuning.stationary_linear_speed + 0.1
        self.simulation._update_wheel_rest_state()
        self.assertEqual(
            self.simulation.wheel_control_mode, WheelControlMode.STABILIZING
        )
        self.assertFalse(self.simulation.relaxation_active)
        self.assertFalse(self.simulation.wheel_rest_controller.active)

    def test_relaxation_slows_wheels_and_telemetry_reports_joint_velocity(self) -> None:
        # Ground-rest behaviour only; the camera's landing reference (which the
        # spawn drop would pick up) is covered by the landing tests.
        self.simulation = JumperSimulation(
            self.model, self.data, "lqr", landing_reference=False
        )
        for joint_id, speed in zip(
            self.simulation.ids.wheel_joints, (20.0, -15.0, 10.0), strict=True
        ):
            self.data.qvel[self.model.jnt_dofadr[joint_id]] = speed

        for _ in range(5000):
            self.simulation.step()

        self.assertTrue(self.simulation.relaxation_active)
        wheel_speeds = self.simulation.read_wheel_speeds()
        self.assertLess(max(abs(wheel_speeds.x), abs(wheel_speeds.y), abs(wheel_speeds.z)), 0.1)
        telemetry = self.simulation.telemetry_payload()
        for axis, speed in zip("xyz", (wheel_speeds.x, wheel_speeds.y, wheel_speeds.z), strict=True):
            self.assertEqual(telemetry[f"vel_{axis}"], speed)

    def test_disable_and_reenable_stabilization(self) -> None:
        self.simulation.handle_key("x")
        self.assertEqual(self.simulation.wheel_control_mode, WheelControlMode.DISABLED)
        self.simulation.step()
        self.assertEqual(
            self.data.ctrl[list(self.simulation.ids.wheel_actuators)].tolist(),
            [0.0, 0.0, 0.0],
        )
        self.simulation.handle_key("c")
        self.assertEqual(
            self.simulation.wheel_control_mode, WheelControlMode.STABILIZING
        )

    def test_wheel_command_respects_speed_headroom(self) -> None:
        tuning = ReactionWheelTuning()
        joint_id = self.simulation.ids.wheel_joints[0]
        dof_id = self.model.jnt_dofadr[joint_id]
        speed_headroom = 0.01
        self.data.qvel[dof_id] = tuning.max_wheel_velocity - speed_headroom

        command = limit_wheel_command(
            self.model, self.data, joint_id, 100.0, tuning
        )
        expected = wheel_effective_inertia(tuning) * (
            speed_headroom / self.model.opt.timestep
        )
        self.assertAlmostEqual(command, expected)

        self.data.qvel[dof_id] = tuning.max_wheel_velocity
        self.assertEqual(
            limit_wheel_command(self.model, self.data, joint_id, 100.0, tuning),
            0.0,
        )

    def test_reset_clears_jump_and_disabled_state(self) -> None:
        self.simulation.handle_key("space")
        self.simulation.handle_key("x")
        self.simulation.reset()
        self.assertFalse(self.simulation.jump_triggered)
        self.assertEqual(
            self.simulation.wheel_control_mode, WheelControlMode.STABILIZING
        )
        self.assertEqual(self.data.time, 0.0)


class TelemetryTests(unittest.TestCase):
    def test_udp_datagram_matches_plotjuggler_signal_names(self) -> None:
        receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        receiver.bind(("127.0.0.1", 0))
        receiver.settimeout(1.0)
        telemetry = UdpTelemetry(port=receiver.getsockname()[1])
        payload = {
            "roll": 0.1,
            "roll_ref": 0.0,
            "pitch": 0.2,
            "pitch_ref": 0.0,
            "yaw": 0.3,
            "yaw_ref": 0.4,
            "vel_x": 1.0,
            "vel_y": 2.0,
            "vel_z": 3.0,
            "sat_max": 4.0,
            "sat_min": -4.0,
        }
        try:
            telemetry.send(payload)
            datagram, _ = receiver.recvfrom(2048)
        finally:
            telemetry.close()
            receiver.close()

        self.assertEqual(json.loads(datagram), payload)


if __name__ == "__main__":
    unittest.main()
