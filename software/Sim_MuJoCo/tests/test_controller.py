import math
import unittest

from jumper_sim.controller import (
    LqrController,
    ReactionWheelController,
    ReactionWheelTuning,
    SensorData,
    WheelRestController,
    WheelSpeeds,
    combine_attitude_and_rest_torque,
    wheel_effective_inertia,
    wrapped_yaw_error,
)


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

    def test_wrapped_yaw_error_uses_shortest_angle(self) -> None:
        self.assertAlmostEqual(wrapped_yaw_error(-math.pi + 0.1, math.pi - 0.1), 0.2)


if __name__ == "__main__":
    unittest.main()
