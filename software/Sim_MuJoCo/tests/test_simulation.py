
from pathlib import Path
import json
import socket
import unittest

import mujoco
import numpy as np

from jumper_sim.simulation import (
    EARTH_GRAVITY,
    MOON_GRAVITY,
    JumperSimulation,
    WheelControlMode,
)
from jumper_sim.controller import (
    ReactionWheelTuning,
    WheelSpeeds,
    limit_wheel_command,
    wheel_effective_inertia,
)
from jumper_sim.telemetry import UdpTelemetry

MODEL_PATH = Path(__file__).resolve().parents[1] / "build" / "Jumping_Robot.xml"
TERRAIN_ASSETS = (
    MODEL_PATH.parent / "terrain.obj",
    MODEL_PATH.parent / "terrain_height.png",
)


@unittest.skipUnless(
    MODEL_PATH.is_file() and all(path.is_file() for path in TERRAIN_ASSETS),
    "Generate build/terrain.obj and terrain_height.png from "
    "build/terrain_merged.usd before running model tests.",
)
class SimulationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        self.data = mujoco.MjData(self.model)
        self.simulation = JumperSimulation(self.model, self.data, "pid")

    def test_model_loads_and_steps_headlessly(self) -> None:
        self.simulation.step()
        self.assertAlmostEqual(self.data.time, self.model.opt.timestep)

    def test_gravity_toggle_and_reset_restore_launch_gravity(self) -> None:
        self.simulation.step()
        qpos_before = self.data.qpos.copy()
        qvel_before = self.data.qvel.copy()
        time_before = self.data.time

        self.simulation.handle_key("backspace")
        self.assertEqual(self.model.opt.gravity[2], -MOON_GRAVITY)
        self.assertEqual(self.data.time, time_before)
        self.assertEqual(self.data.qpos.tolist(), qpos_before.tolist())
        self.assertEqual(self.data.qvel.tolist(), qvel_before.tolist())

        self.simulation.reset()
        self.assertEqual(self.simulation.selected_gravity, EARTH_GRAVITY)
        self.assertEqual(self.model.opt.gravity[2], -EARTH_GRAVITY)
        self.simulation.handle_key("backspace")
        self.assertEqual(self.model.opt.gravity[2], -MOON_GRAVITY)

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
            ("rw_x", (0.10, 0.0, 0.0)),
            ("rw_y", (0.0, 0.10, 0.0)),
            ("rw_z", (0.0, 0.0, 0.10)),
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

        for x, y in ((-20.0, -20.0), (0.0, 0.0), (15.0, 15.0)):
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

    def test_jump_and_tilt_controls_update_actuators(self) -> None:
        self.simulation.handle_key("space")
        self.simulation.handle_key("enter")
        self.simulation.step()
        self.assertTrue(self.simulation.jump_triggered)
        self.assertEqual(
            self.data.ctrl[list(self.simulation.ids.tilt_actuators)].tolist(),
            [200.0, 200.0],
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
        self.simulation = JumperSimulation(self.model, self.data, "lqr")
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

    def test_reset_clears_jump_tilt_and_disabled_state(self) -> None:
        self.simulation.handle_key("space")
        self.simulation.handle_key("enter")
        self.simulation.handle_key("x")
        self.simulation.reset()
        self.assertFalse(self.simulation.jump_triggered)
        self.assertFalse(self.simulation.tilt_active)
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
