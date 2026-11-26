import math
import unittest

import numpy as np

from benchmarks.turn_rate_benchmark import (
    Settle,
    TurnManoeuvre,
    attitude_distance,
    bang_bang_time,
    fit_turn_time,
    rotated,
)


def run(manoeuvre: TurnManoeuvre, accel: float = 10.0, limit: float = 4.0, dt: float = 0.001):
    """A wheel with a speed limit, driven by the manoeuvre; returns (time, speed) per step."""
    t, speed, out = manoeuvre.start_time, manoeuvre.start_wheel_speed, []
    while not manoeuvre.done and t < manoeuvre.start_time + 10.0:
        command = manoeuvre.update(t, speed)
        speed = max(-limit, min(limit, speed + command * accel * dt))
        out.append((t, speed, command))
        t += dt
    return out


class TurnManoeuvreTest(unittest.TestCase):
    def test_pulse_stop_reverse_return(self):
        m = TurnManoeuvre(start_time=2.0, start_wheel_speed=0.5, pulse=1.0)
        trace = run(m)
        self.assertTrue(m.done)
        times = {k: v - 2.0 for k, v in m.step_times.items()}
        self.assertAlmostEqual(times["stop"], 1.0, places=6)
        # stop: from the speed limit back to the start speed at full torque
        self.assertAlmostEqual(times["reverse"] - times["stop"], (4.0 - 0.5) / 10.0, delta=0.002)
        self.assertAlmostEqual(times["return"] - times["reverse"], 1.0, places=6)
        self.assertAlmostEqual(times["done"] - times["return"], (4.0 + 0.5) / 10.0, delta=0.002)
        self.assertAlmostEqual(trace[-1][1], 0.5, delta=0.011)  # wheel back at its start speed
        self.assertEqual(m.update(9.0, 0.5), 0.0)  # and no torque afterwards

    def test_negative_direction_mirrors(self):
        m = TurnManoeuvre(start_time=0.0, start_wheel_speed=0.0, direction=-1.0)
        trace = run(m)
        self.assertTrue(m.done)
        self.assertEqual(trace[0][2], -1.0)
        self.assertAlmostEqual(min(s for _, s, _ in trace), -4.0)

    def test_bang_bang_time(self):
        # short turn: accelerate half, brake half, never at the top rate
        self.assertAlmostEqual(bang_bang_time(10.0, 100.0, 1000.0), 2.0 * (10.0 / 1000.0) ** 0.5)
        # long turn: cruise at the top rate in between
        self.assertAlmostEqual(bang_bang_time(100.0, 100.0, 1000.0), 1.0 + 0.1)


class ControllerStepTest(unittest.TestCase):
    def test_settle_waits_for_the_window_and_a_quiet_spell(self):
        s = Settle(start_time=1.0, window=1.0, hold=0.1, timeout=3.0)
        self.assertFalse(s.update(1.5, 0.5, 1.0))  # quiet, but inside the window
        self.assertFalse(s.update(1.95, 0.5, 1.0))  # quiet for 0.45 s: still inside the window
        self.assertTrue(s.update(2.0, 0.5, 1.0))  # window over, quiet for > 0.1 s
        s = Settle(start_time=0.0)
        self.assertFalse(s.update(1.5, 0.5, 1.0))
        self.assertFalse(s.update(1.55, 3.0, 1.0))  # error too large: the quiet spell restarts
        self.assertFalse(s.update(1.6, 0.5, 1.0))
        self.assertTrue(s.update(1.7, 0.5, 1.0))
        self.assertFalse(s.timed_out)

    def test_settle_times_out(self):
        s = Settle(start_time=0.0, timeout=3.0)
        self.assertFalse(s.update(2.9, 10.0, 50.0))
        self.assertTrue(s.update(3.0, 10.0, 50.0))
        self.assertTrue(s.timed_out)

    def test_rotated_turns_about_the_body_axis(self):
        level = np.array([1.0, 0.0, 0.0, 0.0])
        q = rotated(level, np.array([0.0, 0.0, 1.0]), math.radians(170.0))
        self.assertAlmostEqual(attitude_distance(level, q), 170.0, places=6)
        # about the body axis: from a tilted start, the tilt stays
        tilted = rotated(level, np.array([1.0, 0.0, 0.0]), math.radians(30.0))
        q = rotated(tilted, np.array([0.0, 0.0, 1.0]), math.radians(90.0))
        self.assertAlmostEqual(attitude_distance(tilted, q), 90.0, places=6)
        z_axis = lambda quat: np.array([2 * (quat[1] * quat[3] + quat[0] * quat[2]),
                                        2 * (quat[2] * quat[3] - quat[0] * quat[1]),
                                        1 - 2 * (quat[1] ** 2 + quat[2] ** 2)])
        np.testing.assert_allclose(z_axis(q), z_axis(tilted), atol=1e-12)

    def test_fit_turn_time(self):
        rows = [{"step_deg": s * a, "t": 0.4 + 0.003 * a, "timed_out": False}
                for a in (10.0, 45.0, 90.0) for s in (1, -1)]
        a, k, worst = fit_turn_time(rows, "t")
        self.assertAlmostEqual(a, 0.4)
        self.assertAlmostEqual(k, 0.003)
        self.assertAlmostEqual(worst, 0.0)
        rows.append({"step_deg": 170.0, "t": 9.0, "timed_out": True})  # left out
        self.assertAlmostEqual(fit_turn_time(rows, "t")[1], 0.003)


if __name__ == "__main__":
    unittest.main()
