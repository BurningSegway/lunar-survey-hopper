from dataclasses import astuple, replace
import math
import unittest

import numpy as np

from jumper_sim.controller import (
    LqrController,
    LqrLimits,
    RobotInertia,
    ReactionWheelController,
    ReactionWheelTuning,
    SensorData,
    WheelRestController,
    WheelSpeeds,
    combine_attitude_and_rest_torque,
    gyroscopic_torque,
    quaternion_error,
    solve_continuous_are,
    wheel_effective_inertia,
    wheel_friction_compensation,
    wrapped_yaw_error,
)


def level_error(q, yaw):
    """Error from a level reference with heading ``yaw``."""
    return quaternion_error(q, (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)))


def limits_weights(**overrides):
    """The simulator's default weights, optionally with fields replaced."""
    return replace(LqrLimits().to_weights(ReactionWheelTuning()), **overrides)


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
        rate_command = controller.compute_torque(
            SensorData(0.0, 0.0, 0.0, 0.1, 0.0, 0.0),
            WheelSpeeds(0.0, 0.0, 0.0),
            0.001,
        )
        neutral_command = controller.compute_torque(
            SensorData(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            WheelSpeeds(0.0, 0.0, 0.0),
            0.001,
        )
        self.assertLess(angle_command.tau_x, 0.0)
        self.assertLess(rate_command.tau_x, 0.0)
        self.assertEqual(neutral_command.tau_x, 0.0)

    def test_lqr_gains_solve_the_riccati_equation_per_axis(self) -> None:
        weights = limits_weights(integral=0.0)
        controller = LqrController(
            ReactionWheelTuning(), weights, axis_inertia=(0.1, 0.12, 0.08)
        )
        for inertia, (k_angle, k_rate) in zip(
            controller.axis_inertia, controller.gains, strict=True
        ):
            a = np.array([[0.0, 1.0], [0.0, 0.0]])
            b = np.array([[0.0], [1.0 / inertia]])
            q = np.diag([weights.angle, weights.angular_rate])
            # Reconstruct P from K = R^-1 B^T P and check the CARE residual.
            p12 = k_angle * weights.torque * inertia
            p22 = k_rate * weights.torque * inertia
            p11 = p12 * p22 / (weights.torque * inertia**2)
            p = np.array([[p11, p12], [p12, p22]])
            residual = (
                a.T @ p + p @ a - p @ b @ b.T @ p / weights.torque + q
            )
            np.testing.assert_allclose(residual, 0.0, atol=1e-9)
            closed_loop = a - b @ np.array([[k_angle, k_rate]])
            self.assertTrue(np.all(np.linalg.eigvals(closed_loop).real < 0.0))

    def test_gains_match_closed_form_double_integrator(self) -> None:
        inertia, weights = 0.1, limits_weights(integral=0.0)
        q1, q2, r = weights.angle, weights.angular_rate, weights.torque
        # Analytic CARE solution for J*theta_ddot = u.
        p12 = inertia * math.sqrt(r * q1)
        p22 = inertia * math.sqrt(r * (2.0 * p12 + q2))
        expected = (p12 / (r * inertia), p22 / (r * inertia))
        controller = LqrController(ReactionWheelTuning(), weights, (inertia,) * 3)
        np.testing.assert_allclose(controller.gains[0], expected, rtol=1e-9)
        a = np.array([[0.0, 1.0], [0.0, 0.0]])
        b = np.array([[0.0], [1.0 / inertia]])
        p = solve_continuous_are(a, b, np.diag([q1, q2]), np.array([[r]]))
        np.testing.assert_allclose(p[0, 1], p12, rtol=1e-9)

    def test_lqi_is_stable_and_removes_constant_disturbance(self) -> None:
        inertia = 0.1
        controller = LqrController(
            ReactionWheelTuning(), limits_weights(), axis_inertia=(inertia,) * 3
        )
        k_i, (k_angle, k_rate) = controller.integral_gains[0], controller.gains[0]
        closed_loop = np.array([
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [-k_i / inertia, -k_angle / inertia, -k_rate / inertia],
        ])
        self.assertTrue(np.all(np.linalg.eigvals(closed_loop).real < 0.0))

        # Double integrator with a constant 0.2 N*m disturbance on roll.
        angle, rate, dt = 0.0, 0.0, 0.001
        for _ in range(4000):
            sensor = SensorData(angle, 0.0, 0.0, rate, 0.0, 0.0)
            torque = controller.compute_torque(sensor, WheelSpeeds(0.0, 0.0, 0.0), dt)
            rate += (torque.tau_x + 0.2) / inertia * dt
            angle += rate * dt
        self.assertLess(abs(angle), 1e-3)

        without_integral = LqrController(
            ReactionWheelTuning(), limits_weights(integral=0.0), (inertia,) * 3
        )
        steady_error = 0.2 / without_integral.gains[0][0]
        self.assertGreater(steady_error, 1e-2)

    def test_lqi_integrator_is_clamped_outside_zone_and_reset(self) -> None:
        weights = limits_weights(integral_torque_limit=0.5)
        controller = LqrController(ReactionWheelTuning(), weights, (0.1,) * 3)
        k_i = controller.integral_gains[0]
        k_angle = controller.gains[0][0]
        small = weights.integral_zone / 2.0
        for _ in range(100000):
            command = controller.compute_torque(
                SensorData(small, 0.0, 0.0, 0.0, 0.0, 0.0), WheelSpeeds(0.0, 0.0, 0.0), 0.01
            )
        self.assertAlmostEqual(command.tau_x + k_angle * small, -0.5, places=9)
        controller.reset()
        large = weights.integral_zone * 2.0
        command = controller.compute_torque(
            SensorData(large, 0.0, 0.0, 0.0, 0.0, 0.0), WheelSpeeds(0.0, 0.0, 0.0), 0.01
        )
        self.assertAlmostEqual(command.tau_x, -k_angle * large, places=12)
        self.assertGreater(k_i, 0.0)

    def test_friction_compensation_pushes_wheel_along_its_spin(self) -> None:
        tuning = ReactionWheelTuning()
        self.assertAlmostEqual(
            wheel_friction_compensation(50.0, tuning), tuning.wheel_viscous_friction * 50.0
        )
        self.assertLess(wheel_friction_compensation(-50.0, tuning), 0.0)

    def test_lqr_limits_follow_brysons_rule(self) -> None:
        tuning = ReactionWheelTuning()
        limits = LqrLimits(max_angle_error=0.2, integral_time=0.0, max_torque=1.5)
        weights = limits.to_weights(tuning)
        self.assertEqual(weights.integral, 0.0)
        controller = LqrController(tuning, weights, (0.1,) * 3)
        # Double-integrator LQR: angle gain = sqrt(q/r) = max_torque / max_angle.
        self.assertAlmostEqual(controller.gains[0][0], 1.5 / 0.2)
        default = LqrLimits().to_weights(tuning)
        self.assertAlmostEqual(
            1.0 / math.sqrt(default.torque),
            wheel_effective_inertia(tuning) * tuning.max_wheel_acceleration,
        )

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

    def test_gyroscopic_torque_is_rate_cross_total_momentum(self) -> None:
        inertia = RobotInertia(
            rigid_body=((0.11, 0.0, 0.0), (0.0, 0.12, 0.0), (0.0, 0.0, 0.10)),
            wheel_spin=(0.0036, 0.0036, 0.0036),
        )
        np.testing.assert_allclose(inertia.plant_axis_inertia, (0.1064, 0.1164, 0.0964))
        still = SensorData(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self.assertEqual(
            gyroscopic_torque(inertia, still, WheelSpeeds(50.0, 50.0, 50.0)), (0.0, 0.0, 0.0)
        )
        sensor = SensorData(0.0, 0.0, 0.0, 0.0, 0.0, 1.0)  # yawing
        torque = gyroscopic_torque(inertia, sensor, WheelSpeeds(0.0, 70.0, 0.0))
        # w = z, H_y = 0.0036*70 -> z x y = -x
        np.testing.assert_allclose(torque, (-0.0036 * 70.0, 0.0, 0.0), atol=1e-12)
        with_ff = LqrController(ReactionWheelTuning(), gyroscopic_inertia=inertia)
        without = LqrController(ReactionWheelTuning(), axis_inertia=inertia.plant_axis_inertia)
        delta = np.subtract(
            astuple(with_ff.compute_torque(sensor, WheelSpeeds(0.0, 70.0, 0.0), 0.001)),
            astuple(without.compute_torque(sensor, WheelSpeeds(0.0, 70.0, 0.0), 0.001)),
        )
        np.testing.assert_allclose(delta, torque, atol=1e-12)

    def test_lqr_rejects_non_positive_axis_inertia(self) -> None:
        with self.assertRaises(ValueError):
            LqrController(ReactionWheelTuning(), axis_inertia=(0.1, 0.0, 0.1))

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
