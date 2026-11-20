import math
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import mujoco
import numpy as np

from jumper_sim.cube_model import (
    EQUILIBRIUM,
    SENSOR_CUBE,
    controllability_matrix,
    cross_matrix,
    dynamics,
    jacobians,
    linearize,
    motor_torque,
    quaternion_rate_matrix,
    report_dynamics,
    wheel_accelerations,
)

from model_comparison.compare import (
    DEFAULT_TORQUE, DEFAULT_X0, LiveView, compare, doublet_profile, free_floating_model,
    load_matlab, plot_sdi, plot_simple, step_profile,
)
from jumper_sim.controller import robot_inertia
from jumper_sim.simulation import find_model_ids

MODEL_PATH = Path(__file__).resolve().parents[1] / "build" / "Jumping_Robot.xml"
MATLAB_DIR = Path(__file__).resolve().parents[1] / "model_comparison" / "matlab"
MATLAB_RUN = MATLAB_DIR / "doublets_jw00225.csv"
MATLAB_CAD_RUN = MATLAB_DIR / "doublets_cad_ode4_1ms.csv"
MATLAB_FIXED_RUN = MATLAB_DIR / "fixed_jw0044.csv"
RNG = np.random.default_rng(7)


def random_state() -> np.ndarray:
    q = RNG.standard_normal(4)
    return np.concatenate([q / np.linalg.norm(q), RNG.standard_normal(3)])


class CubeModelTests(unittest.TestCase):
    def test_quaternion_kinematics_is_hamilton_body_rate_scalar_last(self) -> None:
        # (4.3)-(4.4) == 1/2 q * (0, w) in MuJoCo's (w, x, y, z) convention.
        for _ in range(5):
            x = random_state()
            q_dot = 0.5 * quaternion_rate_matrix(x[4:]) @ x[:4]
            expected = np.zeros(4)
            mujoco.mju_mulQuat(expected, np.roll(x[:4], 1), np.concatenate([[0.0], x[4:]]))
            np.testing.assert_allclose(q_dot, 0.5 * np.roll(expected, -1), atol=1e-12)
            self.assertAlmostEqual(float(x[:4] @ q_dot), 0.0, places=12)  # |q| stays 1

    def test_cross_matrix_is_the_cross_product(self) -> None:
        a, b = RNG.standard_normal(3), RNG.standard_normal(3)
        np.testing.assert_allclose(cross_matrix(a) @ b, np.cross(a, b), atol=1e-12)

    def test_equilibrium_is_at_rest_with_any_constant_wheel_speeds(self) -> None:
        # (4.12)-(4.15): w0 = 0 and Omega_dot = 0, whatever Omega_hat is.
        np.testing.assert_allclose(dynamics(EQUILIBRIUM, np.array([80.0, -30.0, 10.0]), np.zeros(3)), 0.0)

    def test_free_flight_conserves_angular_momentum(self) -> None:
        # dH/dt + w x H = 0 (4.7): |H| is constant, wheel torques included.
        x, speeds, dt = random_state(), np.array([40.0, -20.0, 10.0]), 1e-4
        x[4:] *= 0.5

        def momentum_norm(x, speeds):
            return np.linalg.norm(SENSOR_CUBE.j_c @ x[4:] + SENSOR_CUBE.j_w @ speeds)

        start = momentum_norm(x, speeds)
        for k in range(5000):
            torques = 0.5 * np.sin(np.array([3.0, 5.0, 7.0]) * k * dt)
            speeds_dot = wheel_accelerations(x, speeds, torques)
            x = x + dt * dynamics(x, speeds, torques)
            speeds = speeds + dt * speeds_dot
        self.assertAlmostEqual(momentum_norm(x, speeds) / start, 1.0, places=3)

    def test_jacobians_match_finite_differences(self) -> None:
        # A (4.18 with J_c - J_w in front) and B = df/dtau, away from the
        # equilibrium too.
        for _ in range(3):
            x, speeds = random_state(), RNG.uniform(-100.0, 100.0, 3)
            torques = RNG.uniform(-2.0, 2.0, 3)
            a, b = jacobians(x, speeds)
            step = 1e-6
            numeric_a = np.column_stack([
                (dynamics(x + step * e, speeds, torques)
                 - dynamics(x - step * e, speeds, torques)) / (2 * step)
                for e in np.eye(7)
            ])
            numeric_b = np.column_stack([
                (dynamics(x, speeds, torques + step * e)
                 - dynamics(x, speeds, torques - step * e)) / (2 * step)
                for e in np.eye(3)
            ])
            np.testing.assert_allclose(a, numeric_a, atol=1e-6)
            np.testing.assert_allclose(b, numeric_b, atol=1e-6)

    def test_linearisation_has_the_reports_structure(self) -> None:
        # (4.21) with row/column q4 stripped (5.1), torque input.
        jw = SENSOR_CUBE.wheel_inertia
        bx, by, bz = (j - jw for j in SENSOR_CUBE.body_inertia)
        a, b = linearize(np.zeros(3))  # the report's Omega_hat = 0
        expected_a = np.zeros((6, 6))
        expected_a[:3, 3:] = 0.5 * np.eye(3)
        np.testing.assert_allclose(a, expected_a)
        np.testing.assert_allclose(b[:3], 0.0)
        np.testing.assert_allclose(b[3:], -np.diag([1 / bx, 1 / by, 1 / bz]))
        # With spun wheels, (J_c - J_w)^-1 S(J_w Omega_hat) couples the axes.
        speeds = np.array([60.0, -60.0, 40.0])
        a, _ = linearize(speeds)
        np.testing.assert_allclose(
            a[3:, 3:], np.diag([1 / bx, 1 / by, 1 / bz]) @ cross_matrix(jw * speeds)
        )
        self.assertAlmostEqual(a[3, 4], -jw * speeds[2] / bx)
        self.assertAlmostEqual(a[3, 5], jw * speeds[1] / bx)

    def test_wheel_torque_spins_the_wheel_and_turns_the_body_back(self) -> None:
        # J_w (Omega_dot + w_dot) = tau and (J_c - J_w) w_dot = -tau at rest:
        # the momentum J_c w + J_w Omega stays zero.
        torques = np.array([0.3, -0.5, 0.2])
        w_dot = dynamics(EQUILIBRIUM, np.zeros(3), torques)[4:]
        omega_dot = wheel_accelerations(EQUILIBRIUM, np.zeros(3), torques)
        np.testing.assert_allclose(SENSOR_CUBE.j_w @ (omega_dot + w_dot), torques)
        np.testing.assert_allclose(
            SENSOR_CUBE.j_c @ w_dot + SENSOR_CUBE.j_w @ omega_dot, 0.0, atol=1e-12
        )

    def test_report_model_is_the_motor_torque_model_with_its_own_input(self) -> None:
        # (4.15)'s tau = J_w Omega_dot; the motor torque tau + J_w w_dot gives
        # the same motion through dynamics() and wheel_accelerations().
        for _ in range(5):
            x, speeds, tau = random_state(), 30.0 * RNG.standard_normal(3), RNG.standard_normal(3)
            rates = report_dynamics(np.concatenate([x, speeds]), tau)
            motor = motor_torque(np.concatenate([x, speeds]), tau)
            np.testing.assert_allclose(rates[:7], dynamics(x, speeds, motor), atol=1e-10)
            np.testing.assert_allclose(rates[7:], wheel_accelerations(x, speeds, motor), atol=1e-9)
            np.testing.assert_allclose(SENSOR_CUBE.j_w @ rates[7:], tau, atol=1e-12)

    def test_reduced_system_is_controllable(self) -> None:
        # Kalman rank condition (5.1)-(5.2), with and without wheel momentum.
        for speeds in (np.zeros(3), np.array([120.0, -90.0, 60.0])):
            a, b = linearize(speeds)
            self.assertEqual(np.linalg.matrix_rank(controllability_matrix(a, b)), 6)

    def test_cad_values(self) -> None:
        np.testing.assert_allclose(SENSOR_CUBE.body_inertia, (0.07066, 0.07067, 0.07333))
        # The CAD 0.004422 rounded, as in the user's fixed Simulink model.
        self.assertAlmostEqual(SENSOR_CUBE.wheel_inertia, 0.0044)
        self.assertTrue(math.isclose(SENSOR_CUBE.mass, 3.3))


@unittest.skipUnless(MODEL_PATH.is_file(), "needs the robot model")
class ZeroGravityModelComparisonTests(unittest.TestCase):
    def test_mujoco_follows_the_reports_nonlinear_model(self) -> None:
        # Zero g, no bearing friction: tumbling with spun wheels, torque on for
        # 0.5 s. Measured: 0.03 deg, 0.0008 rad/s, 0.00013 rad/s (1 ms steps).
        result = compare(DEFAULT_X0, step_profile(DEFAULT_TORQUE, 0.5), 2.0)
        self.assertLess(result["attitude_diff"].max(), 0.1)
        error = np.abs(result["sim"] - result["model"])
        self.assertLess(error[:, 4:7].max(), 0.005)
        self.assertLess(error[:, 7:].max(), 0.001)

    def test_motor_torque_is_not_the_reports_input(self) -> None:
        # Fed straight to the motors, (4.15)'s tau turns the robot differently
        # (measured: 3.3 deg, 0.65 rad/s on the wheels).
        result = compare(DEFAULT_X0, step_profile(DEFAULT_TORQUE, 0.5), 2.0,
                         spin_input=False)
        self.assertGreater(result["attitude_diff"].max(), 1.0)

    def test_live_view_draws_frames_without_changing_the_run(self) -> None:
        # A stand-in for MuJoCo's viewer window; 20x speed keeps the test short.
        args = (DEFAULT_X0, step_profile(DEFAULT_TORQUE, 0.05), 0.5)
        with mock.patch("mujoco.viewer.launch_passive") as launch:
            launch.return_value.is_running.return_value = True
            shown = compare(*args, view=LiveView(speed=20.0))
        frames = launch.return_value.sync.call_count
        self.assertGreaterEqual(frames, 0.5 * 60 / 20)
        self.assertLessEqual(frames, 0.5 * 60 / 20 + 2)
        np.testing.assert_array_equal(shown["sim"], compare(*args)["sim"])

    def test_doublets_are_the_matlab_tau_profile(self) -> None:
        def tau_profile(t):  # tau_profile.m, line for line
            a, tau = 0.02, np.zeros(3)
            if 1 <= t < 2: tau[0] = a
            if 2 <= t < 3: tau[0] = -a
            if 4 <= t < 5: tau[1] = a
            if 5 <= t < 6: tau[1] = -a
            if 7 <= t < 8: tau[2] = a
            if 8 <= t < 9: tau[2] = -a
            if 10 <= t < 11: tau = np.array([a, -0.75 * a, 0.5 * a])
            return tau

        profile = doublet_profile()
        for t in np.concatenate([np.arange(0.0, 20.0, 0.01), np.arange(13) - 1e-9]):
            np.testing.assert_array_equal(profile(t), tau_profile(t), err_msg=f"t = {t}")

    def test_wheel_inertia_is_set_and_j_c_kept(self) -> None:
        model, _ = free_floating_model(wheel_inertia=0.00225)
        ids = find_model_ids(model)
        inertia = robot_inertia(model, ids.root_joint, ids.wheel_joints)
        np.testing.assert_allclose(inertia.wheel_spin, 0.00225, atol=1e-12)
        np.testing.assert_allclose(np.diag(inertia.rigid_body), SENSOR_CUBE.body_inertia,
                                   atol=1e-9)

    @unittest.skipUnless(MATLAB_RUN.is_file() and MATLAB_FIXED_RUN.is_file(),
                         "needs the MATLAB runs")
    def test_mujoco_follows_the_matlab_runs_until_their_first_late_switch(self) -> None:
        # The user's Simulink runs (J_w,i 0.00225, then fixed to 0.0044): their
        # torque switches land a few ms off from 2 s on (variable step), so
        # compare up to there. Measured: q 7.9e-5 / 6.5e-5, w 7.0e-5 / 3.4e-5
        # rad/s, Omega 4.5e-5 / 3.5e-5 rad/s.
        for run, wheel_inertia in ((MATLAB_RUN, 0.00225), (MATLAB_FIXED_RUN, 0.0044)):
            with self.subTest(run=run.name):
                t_ref, ref = load_matlab(run)
                cube = replace(SENSOR_CUBE, wheel_inertia=wheel_inertia)
                result = compare(ref[0], doublet_profile(), 1.99, cube,
                                 sim_wheel_inertia=wheel_inertia)
                inside = t_ref <= 1.99
                sim = np.column_stack([np.interp(t_ref[inside], result["t"], result["sim"][:, i])
                                       for i in range(10)])
                error = np.abs(sim - ref[inside])
                self.assertLess(error[:, :4].max(), 3e-4)
                self.assertLess(error[:, 4:7].max(), 2e-4)
                self.assertLess(error[:, 7:].max(), 1e-4)

    @unittest.skipUnless(MATLAB_CAD_RUN.is_file(), "needs the MATLAB run")
    def test_mujoco_follows_the_fixed_step_simulink_run_with_cad_wheels(self) -> None:
        # The user's Simulink model with the CAD J_w,i (0.004422, not the
        # design's rounded 0.0044), ode4 at 1 ms, 20 s of doublets.
        # Measured: q 5.1e-4, w 1.4e-4 rad/s, Omega 8.1e-4 rad/s.
        t_ref, ref = load_matlab(MATLAB_CAD_RUN)
        cube = replace(SENSOR_CUBE, wheel_inertia=0.004422)
        result = compare(ref[0], doublet_profile(), 20.0, cube, sim_wheel_inertia=0.004422)
        error = np.abs(result["sim"] - ref)  # same 1 ms grid
        self.assertLess(error[:, :4].max(), 1e-3)
        self.assertLess(error[:, 4:7].max(), 3e-4)
        self.assertLess(error[:, 7:].max(), 1.5e-3)

    def test_simple_plot_is_written(self) -> None:
        result = compare(DEFAULT_X0, step_profile(DEFAULT_TORQUE, 0.05), 0.1)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "run_simple.png"
            plot_simple(path, result["t"], result["sim"], result["t"], result["model"])
            self.assertGreater(path.stat().st_size, 10_000)
            path = Path(folder) / "run_sdi.png"
            plot_sdi(path, result["t"], result["sim"], result["t"][::7], result["model"][::7])
            self.assertGreater(path.stat().st_size, 10_000)


if __name__ == "__main__":
    unittest.main()
