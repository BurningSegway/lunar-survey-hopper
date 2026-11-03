from dataclasses import astuple
import math
import unittest

import numpy as np

from jumper_sim.controller import (
    LqrController,
    LqrLimits,
    ReactionWheelController,
    ReactionWheelTuning,
    SensorData,
    WheelRestController,
    WheelSpeeds,
    combine_attitude_and_rest_torque,
    error_quaternion_vector,
    quaternion_error,
    reference_rate,
    augment,
    solve_continuous_are,
    tracking_feedforward,
    wheel_effective_inertia,
    wheel_friction_compensation,
    wrapped_yaw_error,
)
from jumper_sim.cube_model import SENSOR_CUBE, dynamics, linearize, wheel_accelerations


def level_error(q, yaw):
    """Error from a level reference with heading ``yaw``."""
    return quaternion_error(q, (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)))


def care_residual(a, b, q, r, p):
    return a.T @ p + p @ a - p @ b @ np.linalg.solve(r, b.T) @ p + q


ZERO_WHEELS = WheelSpeeds(0.0, 0.0, 0.0)
SPUN_WHEELS = np.array([60.0, -60.0, 40.0])


class ControllerTests(unittest.TestCase):
    def test_pid_opposes_angle_and_rate_and_reset_clears_integral(self) -> None:
        controller = ReactionWheelController(ReactionWheelTuning())
        sensor = SensorData(0.1, 0.0, 0.0, 0.1, 0.0, 0.0)
        command = controller.compute_torque(sensor, WheelSpeeds(0.0, 0.0, 0.0), 0.01)
        self.assertLess(command.tau_x, 0.0)
        controller.reset()
        reset_command = controller.compute_torque(
            SensorData(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            WheelSpeeds(0.0, 0.0, 0.0),
            0.01,
        )
        self.assertEqual(reset_command.tau_x, 0.0)

    def test_lqr_opposes_angle_rate_and_has_zero_equilibrium_output(self) -> None:
        controller = LqrController(ReactionWheelTuning())
        angle_command = controller.compute_torque(
            SensorData(0.1, 0.0, 0.0, 0.0, 0.0, 0.0),
            WheelSpeeds(0.0, 0.0, 0.0),
            0.001,
        )
        # The integral is state: start each case afresh.
        controller.reset()
        rate_command = controller.compute_torque(
            SensorData(0.0, 0.0, 0.0, 0.1, 0.0, 0.0),
            WheelSpeeds(0.0, 0.0, 0.0),
            0.001,
        )
        controller.reset()
        neutral_command = controller.compute_torque(
            SensorData(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            WheelSpeeds(0.0, 0.0, 0.0),
            0.001,
        )
        self.assertLess(angle_command.tau_x, 0.0)
        self.assertLess(rate_command.tau_x, 0.0)
        self.assertEqual(neutral_command.tau_x, 0.0)

    def test_lqr_gain_solves_the_riccati_equation_and_stabilises(self) -> None:
        q, r = LqrLimits().weights()
        controller = LqrController(ReactionWheelTuning())
        for speeds in (np.zeros(3), SPUN_WHEELS):
            with self.subTest(wheel_speeds=speeds):
                a, b = augment(*linearize(speeds))
                p = solve_continuous_are(a, b, q, r)
                np.testing.assert_allclose(
                    care_residual(a, b, q, r, p), 0.0, atol=1e-9 * np.abs(p).max()
                )
                self.assertTrue(np.all(np.linalg.eigvalsh(p) > 0.0))
                # (5.6): K = R^-1 B' P
                gain = controller.gain_at(speeds)
                self.assertEqual(gain.shape, (3, 9))
                np.testing.assert_allclose(gain, np.linalg.solve(r, b.T @ p), rtol=1e-9)
                self.assertTrue(np.all(np.linalg.eigvals(a - b @ gain).real < 0.0))

    def test_care_matches_closed_form_double_integrator(self) -> None:
        inertia, q1, q2, r = 0.1, 40.0, 0.4, 0.25
        # Analytic CARE solution for J*theta_ddot = u.
        p12 = inertia * math.sqrt(r * q1)
        p22 = inertia * math.sqrt(r * (2.0 * p12 + q2))
        a = np.array([[0.0, 1.0], [0.0, 0.0]])
        b = np.array([[0.0], [1.0 / inertia]])
        p = solve_continuous_are(a, b, np.diag([q1, q2]), np.array([[r]]))
        np.testing.assert_allclose(p[0, 1], p12, rtol=1e-9)
        np.testing.assert_allclose(p[1, 1], p22, rtol=1e-9)

    def test_lqr_limits_follow_brysons_rule(self) -> None:
        limits = LqrLimits(max_angle_error=0.2, max_rate=1.0, max_integral_error=0.1,
                           max_torque=1.5)
        q, r = limits.weights()
        np.testing.assert_allclose(np.diag(q), [1 / 0.1**2] * 3 + [1.0] * 3 + [1 / 0.1**2] * 3)
        np.testing.assert_allclose(np.diag(r), [1 / 1.5**2] * 3)
        # Without the integrator, decoupled at Omega_hat = 0: Bryson's torque
        # at the largest acceptable attitude error, K_q = sqrt(Q_q / R).
        plain = LqrLimits(max_angle_error=0.2, max_rate=1.0, max_integral_error=None,
                          max_torque=1.5)
        self.assertEqual(plain.weights()[0].shape, (6, 6))
        gain = LqrController(ReactionWheelTuning(), plain).gain
        self.assertEqual(gain.shape, (3, 6))
        np.testing.assert_allclose(np.abs(np.diag(gain[:, :3])), 1.5 / 0.1)
        # The default torque bound is the script's 10 rev/s^2 wheel acceleration.
        j_c = np.array(SENSOR_CUBE.body_inertia)
        j_w = SENSOR_CUBE.wheel_inertia
        np.testing.assert_allclose(
            LqrLimits().torques(), j_w * (j_c - j_w) / j_c * 10.0 * 2.0 * math.pi
        )

    def test_lqr_matches_the_matlab_script(self) -> None:
        # angular_reference_integral.mlx: the input there is the wheel
        # acceleration Omega_dot, B = [0; -J_c^-1 J_w], and its weights.
        j_c = np.diag(SENSOR_CUBE.body_inertia)
        j_w = SENSOR_CUBE.wheel_inertia * np.eye(3)
        a6 = np.zeros((6, 6))
        a6[:3, 3:] = 0.5 * np.eye(3)
        b6 = np.vstack([np.zeros((3, 3)), -np.linalg.solve(j_c, j_w)])
        a_aug, b_aug = augment(a6, b6)
        deg = math.pi / 180
        q_aug = np.diag([1 / (15 * deg / 2) ** 2] * 3 + [1 / (60 * deg) ** 2] * 3
                        + [1 / (5 * deg) ** 2] * 3)
        r = np.eye(3) / (10 * 2 * math.pi) ** 2
        k_script = np.linalg.solve(r, b_aug.T @ solve_continuous_are(a_aug, b_aug, q_aug, r))
        # tau_i = J_eff,i Omega_dot_i, so the torque gain is J_eff,i times it.
        j_eff = np.diag(j_w) * (np.diag(j_c) - np.diag(j_w)) / np.diag(j_c)
        gain = LqrController(ReactionWheelTuning()).gain
        np.testing.assert_allclose(gain, j_eff[:, None] * k_script, rtol=1e-7, atol=1e-12)
        # The script's printed gains (its K_bar = -lqr(...)).
        np.testing.assert_allclose(np.diag(-k_script[:, :3]), [770.52, 770.53, 774.40], atol=0.01)
        np.testing.assert_allclose(np.diag(-k_script[:, 3:6]), [126.14, 126.15, 128.23], atol=0.01)
        np.testing.assert_allclose(np.diag(-k_script[:, 6:]), [720.0] * 3, atol=0.01)
        a, b = augment(*linearize(np.zeros(3)))
        self.assertTrue(np.allclose(
            np.sort_complex(np.linalg.eigvals(a - b @ gain)),
            np.sort_complex(np.linalg.eigvals(a_aug - b_aug @ k_script)),
        ))

    def test_reference_feedforward_is_the_error_to_the_reference(self) -> None:
        # The script's u = -K x_r + N r, N = N_u + K N_x, is -K [q_v - r; w]:
        # what the controller gets from the error quaternion.
        controller = LqrController(ReactionWheelTuning())
        a, b = linearize(np.zeros(3))
        n_x, n_u, n = tracking_feedforward(a, b, controller.state_gain)
        np.testing.assert_allclose(n_x, np.vstack([np.eye(3), np.zeros((3, 3))]), atol=1e-12)
        np.testing.assert_allclose(n_u, 0.0, atol=1e-12)
        np.testing.assert_allclose(n, controller.state_gain[:, :3], atol=1e-12)
        x_r = np.array([0.05, -0.02, 0.01, 0.1, 0.2, -0.1])
        ref = np.array([0.03, 0.01, -0.02])
        error = x_r - np.concatenate([ref, np.zeros(3)])
        np.testing.assert_allclose(
            -controller.state_gain @ x_r + n @ ref, -controller.state_gain @ error, atol=1e-12
        )

    def test_lqr_gain_is_the_reports_at_rest_unless_scheduled(self) -> None:
        tuning = ReactionWheelTuning()
        level = SensorData(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        scheduled = LqrController(tuning, schedule_wheel_speeds=True)
        fixed = LqrController(tuning)  # the report: Omega_hat = 0
        zero_gain = scheduled.gain.copy()
        for controller in (scheduled, fixed):
            controller.compute_torque(level, WheelSpeeds(*SPUN_WHEELS), 0.001)
        np.testing.assert_allclose(scheduled.linearized_at, SPUN_WHEELS)
        np.testing.assert_allclose(scheduled.gain, scheduled.gain_at(SPUN_WHEELS))
        np.testing.assert_allclose(fixed.gain, zero_gain)
        np.testing.assert_allclose(fixed.linearized_at, 0.0)
        # The wheels' momentum couples the axes (4.18), and so the gain.
        self.assertGreater(abs(scheduled.gain[0, 1]), 0.05 * abs(scheduled.gain[0, 0]))
        self.assertAlmostEqual(zero_gain[0, 1], 0.0, delta=1e-12)
        # Small changes keep the gain.
        scheduled.compute_torque(level, WheelSpeeds(*(SPUN_WHEELS + 1.0)), 0.001)
        np.testing.assert_allclose(scheduled.linearized_at, SPUN_WHEELS)

    def test_lqr_output_is_the_clipped_motor_torque(self) -> None:
        tuning = ReactionWheelTuning()
        controller = LqrController(tuning)
        state = np.array([0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0])  # small roll rate
        torque = controller.compute_torque(
            SensorData(0.0, 0.0, 0.0, 0.05, 0.0, 0.0), ZERO_WHEELS, 0.001
        )
        # u = tau = -K_bar z on the wheel; the body gets -tau.
        np.testing.assert_allclose(astuple(torque), controller.gain @ state, atol=1e-12)
        self.assertLess(torque.tau_x, 0.0)
        # Clipped at the motor limit, not at Bryson's (softer) torque bound.
        large = controller.compute_torque(SensorData(1.0, -1.0, 0.5, 8.0, -8.0, 8.0), ZERO_WHEELS, 0.001)
        np.testing.assert_allclose(
            np.abs(astuple(large)), wheel_effective_inertia(tuning) * tuning.max_wheel_acceleration
        )

    def test_lqr_integrates_the_error_and_reset_empties_it(self) -> None:
        controller = LqrController(ReactionWheelTuning())
        error = SensorData(0.02, 0.0, -0.01, 0.0, 0.0, 0.0)
        first = controller.compute_torque(error, ZERO_WHEELS, 0.01)
        for _ in range(99):
            later = controller.compute_torque(error, ZERO_WHEELS, 0.01)
        np.testing.assert_allclose(
            controller.integral, error_quaternion_vector((0.02, 0.0, -0.01)) * 1.0, rtol=1e-12
        )
        # The integral pushes harder against a lasting error.
        self.assertLess(later.tau_x, first.tau_x)
        self.assertGreater(later.tau_z, first.tau_z)
        controller.reset()
        np.testing.assert_allclose(controller.integral, 0.0)
        np.testing.assert_allclose(
            astuple(controller.compute_torque(error, ZERO_WHEELS, 0.01)), astuple(first)
        )

    def test_lqr_integral_has_anti_windup(self) -> None:
        tuning = ReactionWheelTuning()
        controller = LqrController(tuning)
        # Integral separation: only axes with an error below 5 deg integrate.
        controller.compute_torque(SensorData(0.2, 0.05, 0.0, 0.0, 0.0, 0.0), ZERO_WHEELS, 0.01)
        self.assertEqual(controller.integral[0], 0.0)
        self.assertGreater(controller.integral[1], 0.0)
        # Clamp: the integral's torque K_I x_I stays within 0.8 N*m.
        controller.reset()
        for _ in range(5000):
            controller.compute_torque(SensorData(0.08, -0.08, 0.0, 0.0, 0.0, 0.0), ZERO_WHEELS, 0.01)
        np.testing.assert_allclose(
            np.abs(np.diag(controller.integral_gain) * controller.integral), [0.8, 0.8, 0.0],
            atol=1e-12,
        )
        # Off: the script's pure integrator.
        faithful = LqrController(tuning, LqrLimits(integral_zone=None, integral_torque_limit=None))
        for _ in range(5000):
            faithful.compute_torque(SensorData(0.2, 0.0, 0.0, 0.0, 0.0, 0.0), ZERO_WHEELS, 0.01)
        np.testing.assert_allclose(faithful.integral[0], 50.0 * error_quaternion_vector((0.2, 0, 0))[0])

    def test_lqr_regulates_the_nonlinear_cube_model(self) -> None:
        # Report model (4.11) in closed loop with u = -K_bar z, from a 40 deg
        # tilt with a tumbling body and spun wheels, and a constant
        # disturbance torque (unmodelled bearing drag) that only the
        # integrator removes.
        tuning = ReactionWheelTuning()
        controller = LqrController(tuning)
        axis = np.array([1.0, -2.0, 0.5]) / np.linalg.norm([1.0, -2.0, 0.5])
        angle = math.radians(40.0)
        x = np.concatenate(
            [axis * math.sin(angle / 2), [math.cos(angle / 2)], [0.5, -0.3, 0.4]]
        )
        speeds = np.array([30.0, -20.0, 10.0])
        disturbance = np.array([0.01, -0.005, 0.008])  # N*m on the wheels
        dt = 0.001
        for _ in range(8000):
            sign = 1.0 if x[3] >= 0.0 else -1.0  # the shorter rotation
            rotation = 2.0 * math.atan2(np.linalg.norm(x[:3]), abs(x[3])) * sign * x[:3] \
                / max(np.linalg.norm(x[:3]), 1e-12)
            command = controller.compute_torque(
                SensorData(*rotation, *x[4:]), WheelSpeeds(*speeds), dt
            )
            u = -np.array(astuple(command)) + disturbance
            speeds_dot = wheel_accelerations(x, speeds, u)
            x = x + dt * dynamics(x, speeds, u)
            x[:4] /= np.linalg.norm(x[:4])
            speeds = speeds + dt * speeds_dot
        self.assertLess(math.degrees(2 * math.acos(min(1.0, abs(x[3])))), 0.1)
        self.assertLess(np.linalg.norm(x[4:]), 1e-3)

    def test_friction_compensation_pushes_wheel_along_its_spin(self) -> None:
        tuning = ReactionWheelTuning()
        self.assertAlmostEqual(
            wheel_friction_compensation(50.0, tuning), tuning.wheel_viscous_friction * 50.0
        )
        self.assertLess(wheel_friction_compensation(-50.0, tuning), 0.0)

    def test_quaternion_error_is_body_rotation_vector(self) -> None:
        self.assertEqual(level_error((1.0, 0.0, 0.0, 0.0), 0.0), (0.0, 0.0, 0.0))
        for axis in range(3):
            for angle in (0.3, -1.2, math.radians(100.0)):
                q = [math.cos(angle / 2), 0.0, 0.0, 0.0]
                q[axis + 1] = math.sin(angle / 2)
                error = level_error(tuple(q), 0.0)
                expected = [0.0, 0.0, 0.0]
                expected[axis] = angle
                np.testing.assert_allclose(error, expected, atol=1e-12)

    def test_quaternion_error_uses_yaw_reference_and_shortest_rotation(self) -> None:
        yaw = math.radians(170.0)
        q = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
        np.testing.assert_allclose(
            level_error(q, yaw), (0.0, 0.0, 0.0), atol=1e-12
        )
        # 170 deg heading vs a -170 deg reference is a 20 deg (not 340) error.
        error = level_error(q, math.radians(-170.0))
        np.testing.assert_allclose(error, (0.0, 0.0, math.radians(-20.0)), atol=1e-12)
        # Negated quaternion is the same attitude.
        negated = tuple(-v for v in q)
        np.testing.assert_allclose(
            level_error(negated, 0.0), level_error(q, 0.0)
        )

    def test_error_quaternion_vector_inverts_quaternion_error(self) -> None:
        for q in ((0.9, 0.1, -0.3, 0.2), (-0.2, 0.7, 0.1, -0.6), (1.0, 0.0, 0.0, 0.0)):
            q = tuple(float(v) for v in np.array(q) / np.linalg.norm(q))
            vector = error_quaternion_vector(level_error(q, 0.0))
            sign = 1.0 if q[0] >= 0.0 else -1.0
            np.testing.assert_allclose(vector, sign * np.array(q[1:]), atol=1e-12)

    def test_reference_rate_is_in_the_body_frame_and_capped(self) -> None:
        def about(axis, angle):
            q = [math.cos(angle / 2), 0.0, 0.0, 0.0]
            q[axis + 1] = math.sin(angle / 2)
            return tuple(q)

        dt = 0.001
        previous, reference = about(2, 0.3), about(2, 0.3 + 0.5 * dt)  # 0.5 rad/s yaw
        np.testing.assert_allclose(
            reference_rate(previous, reference, reference, dt, 10.0), (0.0, 0.0, 0.5), atol=1e-9
        )
        # Body rolled 90 deg from the reference: the yaw rate is along body -y.
        body = (math.cos(0.15) * math.cos(math.pi / 4), math.cos(0.15) * math.sin(math.pi / 4),
                math.sin(0.15) * math.sin(math.pi / 4), math.sin(0.15) * math.cos(math.pi / 4))
        np.testing.assert_allclose(
            reference_rate(previous, reference, body, dt, 10.0), (0.0, 0.5, 0.0), atol=1e-6
        )
        capped = reference_rate(about(0, 0.0), about(0, 0.1), about(0, 0.1), dt, 2.0)
        np.testing.assert_allclose(capped, (2.0, 0.0, 0.0), atol=1e-9)

    def test_lqr_rejects_non_positive_limits(self) -> None:
        with self.assertRaises(ValueError):
            LqrController(ReactionWheelTuning(), LqrLimits(max_rate=0.0))

    def test_wheel_rest_ramps_targets_and_cancels(self) -> None:
        tuning = ReactionWheelTuning()
        rest_controller = WheelRestController(tuning)
        rest_controller.begin(WheelSpeeds(10.0, -10.0, 0.0))
        torques = rest_controller.compute_motor_torques(
            WheelSpeeds(10.0, -10.0, 0.0), 0.1
        )
        self.assertGreater(rest_controller.target_speeds.x, 0.0)
        self.assertLess(rest_controller.target_speeds.x, 10.0)
        self.assertLess(rest_controller.target_speeds.y, 0.0)
        self.assertGreater(rest_controller.target_speeds.y, -10.0)
        max_rest_torque = (
            wheel_effective_inertia(tuning) * tuning.wheel_rest_deceleration
        )
        self.assertLessEqual(abs(torques.x), max_rest_torque)
        self.assertLess(torques.x, 0.0)
        self.assertGreater(torques.y, 0.0)

        for _ in range(20):
            rest_controller.compute_motor_torques(
                WheelSpeeds(0.0, 0.0, 0.0), 0.1
            )
        self.assertEqual(rest_controller.target_speeds, WheelSpeeds(0.0, 0.0, 0.0))

        rest_controller.cancel()
        cancelled = rest_controller.compute_motor_torques(
            WheelSpeeds(10.0, -5.0, 1.0), 0.1
        )
        self.assertEqual(cancelled.x, 0.0)
        self.assertEqual(cancelled.y, 0.0)
        self.assertEqual(cancelled.z, 0.0)

    def test_combined_torque_stays_within_motor_limit(self) -> None:
        tuning = ReactionWheelTuning()
        maximum = wheel_effective_inertia(tuning) * tuning.max_wheel_acceleration
        output = combine_attitude_and_rest_torque(maximum * 0.75, maximum, tuning)
        self.assertLessEqual(abs(output), maximum)

    def test_wheel_rest_torque_takes_priority_when_attitude_conflicts(self) -> None:
        tuning = ReactionWheelTuning()
        output = combine_attitude_and_rest_torque(0.5, -0.1, tuning)
        self.assertEqual(output, -0.1)

    def test_wrapped_yaw_error_uses_shortest_angle(self) -> None:
        self.assertAlmostEqual(wrapped_yaw_error(-math.pi + 0.1, math.pi - 0.1), 0.2)


if __name__ == "__main__":
    unittest.main()
