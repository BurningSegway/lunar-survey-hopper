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
    DEFAULT_TORQUE, DEFAULT_X0, STATE_NAMES, LiveView, Reference, compare, doublet_profile,
    free_floating_model, load_matlab, main, mujoco_solver_label, plot_sdi, sample_rate_label,
    step_profile,
)
from jumper_sim.controller import robot_inertia
from jumper_sim.simulation import find_model_ids

MODEL_PATH = Path(__file__).resolve().parents[1] / "build" / "Jumping_Robot.xml"
MATLAB_DIR = Path(__file__).resolve().parents[1] / "model_comparison" / "matlab"
MATLAB_ODE4_RUN = MATLAB_DIR / "ode4_1khz.csv"
MATLAB_ODE3_RUN = MATLAB_DIR / "ode3_1khz.csv"
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

    @unittest.skipUnless(MATLAB_ODE4_RUN.is_file() and MATLAB_ODE3_RUN.is_file(),
                         "needs the MATLAB runs")
    def test_mujoco_follows_the_fixed_step_simulink_runs(self) -> None:
        # The user's Simulink model at 1 kHz, 20 s of doublets. Measured
        # (q, w, Omega): ode4 5.0e-4, 1.4e-4, 8.1e-4 (its last stage sees a
        # switch ~1/6 ms early); ode3 4.9e-4, 1.4e-4, 9.8e-5.
        for run, omega_bound in ((MATLAB_ODE4_RUN, 1.5e-3), (MATLAB_ODE3_RUN, 3e-4)):
            with self.subTest(run=run.name):
                t_ref, ref = load_matlab(run)
                result = compare(ref[0], doublet_profile(), 20.0)
                error = np.abs(result["sim"] - ref)  # same 1 ms grid
                self.assertLess(error[:, :4].max(), 1e-3)
                self.assertLess(error[:, 4:7].max(), 3e-4)
                self.assertLess(error[:, 7:].max(), omega_bound)
                # ode3 never evaluates past a switch: it is (4.15) itself, to
                # its 3rd-order step in q (measured 2.5e-5, w 7e-11).
                if run == MATLAB_ODE3_RUN:
                    error = np.abs(result["model"] - ref)
                    self.assertLess(error[:, :4].max(), 1e-4)
                    self.assertLess(error[:, 4:].max(), 1e-8)

    def test_rate_and_solver_labels(self) -> None:
        self.assertEqual(sample_rate_label(np.linspace(0.0, 20.0, 20001)), "1 kHz")
        self.assertEqual(sample_rate_label(np.array([0.0, 0.01, 0.02])), "100 Hz")
        self.assertEqual(sample_rate_label(np.array([0.0, 0.001, 0.4])), "variable step")
        model, _ = free_floating_model()
        self.assertEqual(mujoco_solver_label(model), "semi-implicit Euler, 1 kHz")

    def test_sdi_plot_is_drawn_at_its_printed_size(self) -> None:
        import matplotlib.image
        result = compare(DEFAULT_X0, step_profile(DEFAULT_TORQUE, 0.05), 0.1)
        reference = Reference(result["t"][::7], result["model"][::7], "Simulink", "ode4, 1 kHz")
        with tempfile.TemporaryDirectory() as folder:
            # One column: half an A4 landscape text width (12 cm at 300 dpi);
            # two: the whole width (24.7 cm).
            for columns, width_cm in ((1, 12.0), (2, 24.7)):
                path = Path(folder) / f"run{columns}_sdi.png"
                plot_sdi(path, result["t"], result["sim"], [reference] * columns,
                         tau=result["tau"], sim_info="semi-implicit Euler, 1 kHz")
                width = matplotlib.image.imread(path).shape[1]
                self.assertAlmostEqual(width, width_cm / 2.54 * 300, delta=2)
                self.assertTrue(path.with_suffix(".pdf").is_file())

    def _write_run(self, path: Path, t: np.ndarray, states: np.ndarray) -> None:
        # As matlab/export_states.m writes a Simulink run.
        header = "t," + ",".join(STATE_NAMES)
        np.savetxt(path, np.column_stack([t, states]), delimiter=",", header=header,
                   comments="")

    def test_a_run_writes_its_csvs_and_one_sdi_plot(self) -> None:
        result = compare(DEFAULT_X0, step_profile(DEFAULT_TORQUE, 1.0), 0.05)
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            for name in ("ode4", "ode3"):
                self._write_run(folder / f"{name}.csv", result["t"], result["model"])
            results = folder / "results"
            argv = ["compare", "--label", "run", "--duration", "0.05",
                    "--matlab", str(folder / "ode4.csv"), str(folder / "ode3.csv")]
            with mock.patch("model_comparison.compare.RESULTS_DIR", results), \
                    mock.patch("builtins.print"):
                with mock.patch("sys.argv", argv + ["--simulink-solver", "ode4", "ode3"]):
                    main()
                self.assertEqual(sorted(p.name for p in results.iterdir()),
                                 ["run.csv", "run_matlab_ode3.csv", "run_matlab_ode4.csv",
                                  "run_sdi.pdf", "run_sdi.png"])
                with mock.patch("sys.argv", argv + ["--simulink-solver", "ode4"]), \
                        mock.patch("sys.stderr"), self.assertRaises(SystemExit):
                    main()  # one solver name per run
            with mock.patch("model_comparison.compare.RESULTS_DIR", folder / "model"), \
                    mock.patch("sys.argv", ["compare", "--label", "run", "--duration", "0.05"]), \
                    mock.patch("builtins.print"):
                main()  # no --matlab: against the model (4.15)
            self.assertEqual(sorted(p.name for p in (folder / "model").iterdir()),
                             ["run.csv", "run_sdi.pdf", "run_sdi.png"])
            # --replot redraws from the saved results, without running anything.
            (folder / "model" / "run_sdi.png").unlink()
            with mock.patch("model_comparison.compare.RESULTS_DIR", folder / "model"), \
                    mock.patch("sys.argv", ["compare", "--label", "run", "--replot"]), \
                    mock.patch("model_comparison.compare.compare",
                               side_effect=AssertionError("ran")), \
                    mock.patch("builtins.print"):
                main()
            self.assertTrue((folder / "model" / "run_sdi.png").is_file())


if __name__ == "__main__":
    unittest.main()
