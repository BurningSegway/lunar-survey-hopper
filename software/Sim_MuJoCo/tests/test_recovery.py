import math
import unittest

import numpy as np

from jumper_sim.recovery import (
    RecoveryAttempt,
    RecoveryConfig,
    RecoveryStage,
    level_quaternion,
    righting_axis,
)


def rotation_about(axis, angle):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * k @ k


class RecoveryTests(unittest.TestCase):
    def test_righting_axis_turns_body_z_towards_world_z(self) -> None:
        for axis, angle in (((0, 1, 0), 1.6), ((1, 0, 0), -1.8), ((1, 1, 0), 2.0)):
            rotation = rotation_about(axis, angle)
            body_axis = righting_axis(rotation)
            # A small turn about it (body frame) reduces the tilt.
            turned = rotation @ rotation_about(body_axis, 0.01)
            self.assertLess(math.acos(turned[2, 2]), math.acos(rotation[2, 2]))
            self.assertAlmostEqual(float(np.linalg.norm(body_axis)), 1.0)
        self.assertIsNone(righting_axis(np.eye(3)))

    def test_level_quaternion_keeps_the_heading(self) -> None:
        rotation = rotation_about((0, 0, 1), 0.7) @ rotation_about((1, 0, 0), 1.5)
        w, x, y, z = level_quaternion(rotation)
        self.assertAlmostEqual(x, 0.0)
        self.assertAlmostEqual(y, 0.0)
        self.assertAlmostEqual(2.0 * math.atan2(z, w), 0.7, places=6)

    def test_spins_up_then_releases_the_spring_once(self) -> None:
        attempt = RecoveryAttempt(10.0, RecoveryConfig(spin_up_duration=0.15))
        self.assertIs(attempt.stage, RecoveryStage.SPIN_UP)
        self.assertEqual(attempt.wheel_torque_sign, 1.0)
        self.assertFalse(attempt.update(10.1))
        self.assertTrue(attempt.update(10.15))
        self.assertIs(attempt.stage, RecoveryStage.FLIGHT)
        self.assertEqual(attempt.wheel_torque_sign, 0.0)
        self.assertFalse(attempt.update(10.2))  # released only once


if __name__ == "__main__":
    unittest.main()
