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
        # dH/dt + w x H = 0 (4.7): |H| is constant, wheel accelerations included.
        x, speeds, dt = random_state(), np.array([40.0, -20.0, 10.0]), 1e-4
        x[4:] *= 0.5

        def momentum_norm(x, speeds):
            return np.linalg.norm(SENSOR_CUBE.j_c @ x[4:] + SENSOR_CUBE.j_w @ speeds)

        start = momentum_norm(x, speeds)
        for k in range(5000):
            accelerations = 30.0 * np.sin(np.array([3.0, 5.0, 7.0]) * k * dt)
            x = x + dt * dynamics(x, speeds, accelerations)
            speeds = speeds + dt * accelerations
        self.assertAlmostEqual(momentum_norm(x, speeds) / start, 1.0, places=3)

    def test_jacobians_match_finite_differences(self) -> None:
        # (4.18) and (4.19), away from the equilibrium too.
        for _ in range(3):
            x, speeds = random_state(), RNG.uniform(-100.0, 100.0, 3)
            accelerations = RNG.uniform(-50.0, 50.0, 3)
            a, b = jacobians(x, speeds)
            step = 1e-6
            numeric_a = np.column_stack([
                (dynamics(x + step * e, speeds, accelerations)
                 - dynamics(x - step * e, speeds, accelerations)) / (2 * step)
                for e in np.eye(7)
            ])
            numeric_b = np.column_stack([
                (dynamics(x, speeds, accelerations + step * e)
                 - dynamics(x, speeds, accelerations - step * e)) / (2 * step)
                for e in np.eye(3)
            ])
            np.testing.assert_allclose(a, numeric_a, atol=1e-6)
            np.testing.assert_allclose(b, numeric_b, atol=1e-6)

    def test_linearisation_has_the_reports_structure(self) -> None:
        # (4.21) with row/column q4 stripped (5.1).
        speeds = np.array([60.0, -60.0, 40.0])
        a, b = linearize(speeds)
        jw = SENSOR_CUBE.wheel_inertia
        jx, jy, jz = SENSOR_CUBE.body_inertia
        np.testing.assert_allclose(a[:3, :3], 0.0)
        np.testing.assert_allclose(a[:3, 3:], 0.5 * np.eye(3))
        np.testing.assert_allclose(a[3:, :3], 0.0)
        # J_c^-1 S(J_w Omega_hat): the wheels' momentum couples the axes.
        np.testing.assert_allclose(
            a[3:, 3:], np.diag([1 / jx, 1 / jy, 1 / jz]) @ cross_matrix(jw * speeds)
        )
        self.assertAlmostEqual(a[3, 4], -jw * speeds[2] / jx)
        self.assertAlmostEqual(a[3, 5], jw * speeds[1] / jx)
        np.testing.assert_allclose(b[:3], 0.0)
        np.testing.assert_allclose(b[3:], -np.diag([jw / jx, jw / jy, jw / jz]))

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
