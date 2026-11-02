import math
import unittest

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
    quaternion_rate_matrix,
    wheel_accelerations,
)

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

    def test_reduced_system_is_controllable(self) -> None:
        # Kalman rank condition (5.1)-(5.2), with and without wheel momentum.
        for speeds in (np.zeros(3), np.array([120.0, -90.0, 60.0])):
            a, b = linearize(speeds)
            self.assertEqual(np.linalg.matrix_rank(controllability_matrix(a, b)), 6)

    def test_cad_values(self) -> None:
        np.testing.assert_allclose(SENSOR_CUBE.body_inertia, (0.07066, 0.07067, 0.07333))
        self.assertAlmostEqual(SENSOR_CUBE.wheel_inertia, 0.004422)
        self.assertTrue(math.isclose(SENSOR_CUBE.mass, 3.3))


if __name__ == "__main__":
    unittest.main()
