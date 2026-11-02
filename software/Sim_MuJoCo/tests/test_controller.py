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
    solve_continuous_are,
    wheel_effective_inertia,
    wheel_friction_compensation,
    wrapped_yaw_error,
)
from jumper_sim.cube_model import dynamics, linearize, wheel_accelerations


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
        # The wheel speed command is state: start each case afresh.
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
        tuning = ReactionWheelTuning()
        q, r = LqrLimits().weights(tuning)
        controller = LqrController(tuning)
        for speeds in (np.zeros(3), SPUN_WHEELS):
            with self.subTest(wheel_speeds=speeds):
                a, b = linearize(speeds)
                p = solve_continuous_are(a, b, q, r)
                np.testing.assert_allclose(
                    care_residual(a, b, q, r, p), 0.0, atol=1e-9 * np.abs(p).max()
                )
                self.assertTrue(np.all(np.linalg.eigvalsh(p) > 0.0))
                # (5.6): K = R^-1 B' P
                gain = controller.gain_at(speeds)
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
        tuning = ReactionWheelTuning()
        limits = LqrLimits(max_angle_error=0.2, max_rate=1.0, max_torque=1.5)
        q, r = limits.weights(tuning)
        np.testing.assert_allclose(np.diag(q), [1 / math.sin(0.1) ** 2] * 3 + [1.0] * 3)
        np.testing.assert_allclose(np.diag(r), [1 / 1.5**2] * 3)
        # Decoupled at Omega_hat = 0: full motor torque at the largest
        # acceptable attitude error, K_q = sqrt(Q_q / R) = tau_max / q_max.
        gain = LqrController(tuning, limits).gain
        np.testing.assert_allclose(np.abs(np.diag(gain[:, :3])), 1.5 / math.sin(0.1))
        # The default is the motor limit the simulator enforces.
        self.assertAlmostEqual(
            LqrLimits().torque(tuning),
            wheel_effective_inertia(tuning) * tuning.max_wheel_acceleration,
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
        self.assertEqual(zero_gain[0, 1], 0.0)
        # Small changes keep the gain.
        scheduled.compute_torque(level, WheelSpeeds(*(SPUN_WHEELS + 1.0)), 0.001)
        np.testing.assert_allclose(scheduled.linearized_at, SPUN_WHEELS)

    def test_lqr_output_is_the_clipped_motor_torque(self) -> None:
        tuning = ReactionWheelTuning()
        controller = LqrController(tuning)
        state = np.array([0.0, 0.0, 0.0, 0.05, 0.0, 0.0])  # small roll rate
        torque = controller.compute_torque(
            SensorData(0.0, 0.0, 0.0, 0.05, 0.0, 0.0), ZERO_WHEELS, 0.001
        )
        # u = tau = -K x_r on the wheel; the body gets -tau.
        np.testing.assert_allclose(astuple(torque), controller.gain @ state, atol=1e-12)
        self.assertLess(torque.tau_x, 0.0)
        large = controller.compute_torque(SensorData(1.0, -1.0, 0.5, 0.0, 0.0, 0.0), ZERO_WHEELS, 0.001)
        np.testing.assert_allclose(np.abs(astuple(large)), LqrLimits().torque(tuning))

    def test_lqr_regulates_the_nonlinear_cube_model(self) -> None:
        # Report model (4.11) in closed loop with the report's u = -K x_r, from
        # a 40 deg tilt with a tumbling body and spun wheels.
        controller = LqrController(ReactionWheelTuning())
        axis = np.array([1.0, -2.0, 0.5]) / np.linalg.norm([1.0, -2.0, 0.5])
        angle = math.radians(40.0)
        x = np.concatenate(
            [axis * math.sin(angle / 2), [math.cos(angle / 2)], [0.5, -0.3, 0.4]]
        )
        speeds = np.array([30.0, -20.0, 10.0])
        limit = LqrLimits().torque(ReactionWheelTuning())
        dt = 0.001
        for _ in range(2000):
            u = np.clip(-controller.gain @ np.delete(x, 3), -limit, limit)
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
