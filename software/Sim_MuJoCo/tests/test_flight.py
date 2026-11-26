import math
import unittest

import numpy as np

from jumper_sim.flight_phase import FlightPhase, FlightPhaseDetector, VerticalVelocityEstimator
from jumper_sim.flight_reference import (
    FlightReference,
    FlightReferenceConfig,
    attitude_angle,
    default_target,
    look_at_quaternion,
    slew_quaternion,
)
from jumper_sim.jump_aim import JumpAim, JumpAimConfig, lean_angle
from jumper_sim.landing_reference import quaternion_to_matrix

G = 9.81
DT = 0.001
LEVEL = (1.0, 0.0, 0.0, 0.0)


def run_hop(detector, push_accel=20.0, push_time=0.3, bias=0.0, settle=1000):
    """Stand, jump, push off, fly ballistically, land. Returns the phase per step."""
    phases = []

    def step(force_z, contact, jump=False):
        phases.append(detector.update(np.array([0.0, 0.0, force_z + bias]), LEVEL, G, DT,
                                      contact, body_rate=np.zeros(3), jump=jump))

    for _ in range(700):  # standing still: specific force = +g
        step(G, True)
    step(G + push_accel, True, jump=True)
    for _ in range(int(push_time / DT) - 1):  # push-off on the ground
        step(G + push_accel, True)
    v0 = push_accel * push_time
    for _ in range(int(2 * v0 / G / DT)):  # free flight: accelerometer ~0
        step(0.0, False)
    for _ in range(settle):
        step(G, True)
    return phases, v0


class FlightPhaseTests(unittest.TestCase):
    def test_hop_goes_through_all_phases(self) -> None:
        detector = FlightPhaseDetector()
        phases, v0 = run_hop(detector, push_accel=60.0, push_time=0.1)  # 0.3 m stroke
        sequence = [p for i, p in enumerate(phases) if i == 0 or p is not phases[i - 1]]
        self.assertEqual(sequence, [
            FlightPhase.LANDING,  # starts as DESCENT; on the ground at once
            FlightPhase.GROUND, FlightPhase.TAKEOFF, FlightPhase.ASCENT,
            FlightPhase.DESCENT, FlightPhase.LANDING, FlightPhase.GROUND,
        ])
        # The switch to descent happens at the apex, v0/g after lift-off.
        lift_off = 700 + int(0.1 / DT)
        apex = phases.index(FlightPhase.DESCENT)
        self.assertAlmostEqual((apex - lift_off) * DT, v0 / G, delta=0.01)
        # Ascent starts 0.5 m above the standing height (push-off included).
        ascent = phases.index(FlightPhase.ASCENT)
        t = (ascent - lift_off) * DT
        height = 0.5 * 60.0 * 0.1**2 + v0 * t - 0.5 * G * t * t
        self.assertAlmostEqual(height, 0.5, delta=0.02)

    def test_landing_settles_to_ground_after_standing_still(self) -> None:
        detector = FlightPhaseDetector()
        phases, _ = run_hop(detector, settle=400)  # < settle_time after touchdown
        self.assertIs(phases[-1], FlightPhase.LANDING)
        for _ in range(200):
            detector.update(np.array([0.0, 0.0, G]), LEVEL, G, DT, True, body_rate=np.zeros(3))
        self.assertIs(detector.phase, FlightPhase.GROUND)

    def test_standing_still_settles_whatever_the_contact_input(self) -> None:
        # A landing the contact estimate missed: still ~1 g, no rotation.
        detector = FlightPhaseDetector()
        detector.phase = FlightPhase.LANDING
        steps = 0
        while detector.phase is FlightPhase.LANDING and steps < 1000:
            detector.update(np.array([0.0, 0.0, G]), LEVEL, G, DT, False, body_rate=np.zeros(3))
            steps += 1
        self.assertIs(detector.phase, FlightPhase.GROUND)
        self.assertAlmostEqual(steps * DT, detector.config.settle_time, delta=2 * DT)
        # Free fall is never still.
        detector.phase = FlightPhase.LANDING
        for _ in range(2000):
            detector.update(np.zeros(3), LEVEL, G, DT, False, body_rate=np.zeros(3))
        self.assertIs(detector.phase, FlightPhase.LANDING)

    def test_rocking_robot_is_not_settled(self) -> None:
        detector = FlightPhaseDetector()
        for _ in range(2000):
            detector.update(np.array([0.0, 0.0, G]), LEVEL, G, DT, True,
                            body_rate=np.array([0.5, 0.0, 0.0]))
        self.assertIs(detector.phase, FlightPhase.LANDING)

    def test_failed_jump_returns_to_ground(self) -> None:
        detector = FlightPhaseDetector()
        for _ in range(700):
            detector.update(np.array([0.0, 0.0, G]), LEVEL, G, DT, True, body_rate=np.zeros(3))
        detector.update(np.array([0.0, 0.0, G]), LEVEL, G, DT, True, jump=True)
        self.assertIs(detector.phase, FlightPhase.TAKEOFF)
        for _ in range(600):  # never leaves the ground
            detector.update(np.array([0.0, 0.0, G]), LEVEL, G, DT, True, body_rate=np.zeros(3))
        self.assertIs(detector.phase, FlightPhase.GROUND)

    def test_low_hop_skips_ascent(self) -> None:
        detector = FlightPhaseDetector()
        phases, _ = run_hop(detector, push_accel=5.0, push_time=0.1)  # ~5 cm hop
        self.assertNotIn(FlightPhase.ASCENT, phases)
        self.assertIn(FlightPhase.DESCENT, phases[phases.index(FlightPhase.TAKEOFF):])

    def test_descent_lands_when_touchdown_is_near(self) -> None:
        detector = FlightPhaseDetector()
        detector.update(np.zeros(3), LEVEL, G, DT, False)
        self.assertIs(detector.phase, FlightPhase.DESCENT)
        detector.update(np.zeros(3), LEVEL, G, DT, False, touchdown_near=True)
        self.assertIs(detector.phase, FlightPhase.LANDING)
        for _ in range(100):  # still in the air: stays LANDING
            detector.update(np.zeros(3), LEVEL, G, DT, False)
        self.assertIs(detector.phase, FlightPhase.LANDING)

    def test_zero_velocity_update_removes_drift_while_standing(self) -> None:
        estimator = VerticalVelocityEstimator()
        for _ in range(2000):
            estimator.update(np.array([0.0, 0.0, G + 0.3]), LEVEL, G, DT, True)
        self.assertEqual(estimator.vertical_velocity, 0.0)
        # ... but not in flight, where it must integrate.
        for _ in range(100):
            estimator.update(np.zeros(3), LEVEL, G, DT, False)
        self.assertAlmostEqual(estimator.vertical_velocity, -G * 0.1, places=6)

    def test_tilted_body_uses_world_vertical(self) -> None:
        estimator = VerticalVelocityEstimator()
        tilt = math.radians(30)
        q = (math.cos(tilt / 2), math.sin(tilt / 2), 0.0, 0.0)
        # Specific force of a robot at rest, expressed in the tilted body frame.
        f_body = quaternion_to_matrix(q).T @ np.array([0.0, 0.0, G])
        for _ in range(100):
            estimator.update(f_body, q, G, DT, False)
        self.assertAlmostEqual(estimator.vertical_velocity, 0.0, places=9)

    def test_descent_is_latched_until_landing(self) -> None:
        detector = FlightPhaseDetector()
        detector.update(np.zeros(3), LEVEL, G, DT, False)
        self.assertIs(detector.phase, FlightPhase.DESCENT)  # dropped from rest
        for _ in range(100):  # a brief upward kick does not flip it back
            detector.update(np.array([0.0, 0.0, 3 * G]), LEVEL, G, DT, False)
        self.assertIs(detector.phase, FlightPhase.DESCENT)


class FlightReferenceTests(unittest.TestCase):
    def test_look_at_points_camera_at_target_and_keeps_heading(self) -> None:
        camera = np.array([0.0, 0.0, 3.0])
        target = np.array([1.5, -1.0, 0.0])
        q = look_at_quaternion(camera, target, 0.4, math.radians(60))
        r = quaternion_to_matrix(q)
        to_target = (target - camera) / np.linalg.norm(target - camera)
        np.testing.assert_allclose(-r[:, 2], to_target, atol=1e-12)
        heading = math.atan2(r[1, 0], r[0, 0])
        self.assertLess(abs(math.remainder(heading - 0.4, 2 * math.pi)), math.radians(25))

    def test_look_at_tilt_is_capped(self) -> None:
        q = look_at_quaternion(np.array([0.0, 0.0, 1.0]), np.array([10.0, 0.0, 0.0]), 0.0,
                               math.radians(60))
        tilt = math.degrees(math.acos(quaternion_to_matrix(q)[2, 2]))
        self.assertAlmostEqual(tilt, 60.0, places=9)
        # Tilted towards the target (camera axis leans to +x).
        self.assertGreater(-quaternion_to_matrix(q)[0, 2], 0.0)

    def test_slew_is_rate_limited_and_converges(self) -> None:
        target = (math.cos(math.radians(45)), math.sin(math.radians(45)), 0.0, 0.0)  # 90 deg
        q = LEVEL
        for _ in range(10):
            new = slew_quaternion(q, target, math.radians(10))
            step = 2 * math.acos(min(1.0, abs(float(np.dot(new, q)))))
            self.assertLessEqual(step, math.radians(10) + 1e-9)
            q = new
        np.testing.assert_allclose(q, target, atol=1e-12)
        negated = tuple(-v for v in target)
        np.testing.assert_allclose(slew_quaternion(target, negated, 0.01), target, atol=1e-12)

    def test_default_target_is_ahead_of_take_off(self) -> None:
        config = FlightReferenceConfig(default_target_distance=3.0)
        np.testing.assert_allclose(
            default_target(np.array([1.0, 2.0, 0.5]), math.pi / 2, config), (1.0, 5.0),
            atol=1e-12,
        )

    def test_target_is_tracked_just_above_the_ground(self) -> None:
        def slope(x: float, y: float) -> float:
            return 0.5 * x if abs(x) < 50.0 else math.nan  # off the terrain beyond 50 m

        config = FlightReferenceConfig(default_target_distance=3.0, aim_height=0.15,
                                       standing_height=0.27)
        reference = FlightReference(config, target=(4.0, 1.0), ground_height=slope)
        np.testing.assert_allclose(reference.target_setting, (4.0, 1.0, 2.15))
        reference.start_flight(LEVEL, np.array([0.0, 0.0, 0.27]), 0.0, track=True)
        np.testing.assert_allclose(reference.active_target, (4.0, 1.0, 2.15))
        # The default point 3 m ahead is on the ground there, not at take-off height.
        reference.set_target(None)
        reference.start_flight(LEVEL, np.array([0.0, 0.0, 0.27]), 0.0, track=True)
        np.testing.assert_allclose(reference.active_target, (3.0, 0.0, 1.65))
        # Off the terrain: the take-off ground.
        reference.set_target((60.0, 0.0))
        np.testing.assert_allclose(reference.target_setting, (60.0, 0.0, 0.15))
        reference.start_flight(LEVEL, np.array([48.0, 0.0, 24.27]), 0.0, track=True)
        np.testing.assert_allclose(reference.active_target, (60.0, 0.0, 0.15))
        reference.set_target(None)
        reference.start_flight(LEVEL, np.array([48.0, 0.0, 24.27]), 0.0, track=True)
        np.testing.assert_allclose(reference.active_target, (51.0, 0.0, 24.15))

    def test_ascent_tracks_descent_follows_landing_reference(self) -> None:
        reference = FlightReference(FlightReferenceConfig(slew_rate=math.radians(1e6)),
                                    target=(2.0, 0.0))
        reference.start_flight(LEVEL, np.zeros(3), 0.0, track=True)
        camera = np.array([0.0, 0.0, 2.0])
        q = reference.update(FlightPhase.ASCENT, camera, 0.0, lambda yaw: LEVEL, DT)
        self.assertLess(reference.axis_target_angle(q, camera), 1e-6)
        plane = (math.cos(0.05), math.sin(0.05), 0.0, 0.0)
        q = reference.update(FlightPhase.DESCENT, camera, 0.0, lambda yaw: plane, DT,
                             attitude=q, time_to_impact=0.0)  # the landing turn is due
        np.testing.assert_allclose(q, plane, atol=1e-12)
        # Flights not started by a jump (bounces) never track.
        reference.start_flight(LEVEL, np.zeros(3), 0.0, track=False)
        q = reference.update(FlightPhase.ASCENT, camera, 0.0, lambda yaw: LEVEL, DT)
        np.testing.assert_allclose(q, LEVEL, atol=1e-12)


    def test_target_tracked_until_the_landing_turn_is_due(self) -> None:
        config = FlightReferenceConfig(slew_rate=math.radians(1e6))
        reference = FlightReference(config, target=(2.0, 0.0))
        reference.start_flight(LEVEL, np.zeros(3), 0.0, track=True)
        camera = np.array([0.0, 0.0, 2.0])
        plane = (math.cos(0.05), math.sin(0.05), 0.0, 0.0)
        tracked = reference.update(FlightPhase.ASCENT, camera, 0.0, lambda yaw: plane, DT)
        angle = attitude_angle(tracked, plane)
        self.assertAlmostEqual(angle, 47.6, delta=0.1)  # 45 deg to the target, plane tilted 5.7 deg
        turn = config.turn_time_base + config.turn_time_per_degree * angle
        self.assertAlmostEqual(reference.turn_time(angle), turn)
        # No time to impact yet, or more than the turn takes: still on the target.
        for left in (None, 5.0, turn + 0.01):
            q = reference.update(FlightPhase.DESCENT, camera, 0.0, lambda yaw: plane, DT,
                                 attitude=tracked, time_to_impact=left)
            self.assertLess(reference.axis_target_angle(q, camera), 1e-6)
            self.assertTrue(reference.tracking_target(FlightPhase.DESCENT))
        # As soon as the turn takes as long as the time left: the landing reference.
        q = reference.update(FlightPhase.DESCENT, camera, 0.0, lambda yaw: plane, DT,
                             attitude=tracked, time_to_impact=turn - 0.001)
        np.testing.assert_allclose(q, plane, atol=1e-12)
        self.assertFalse(reference.tracking_target(FlightPhase.DESCENT))
        self.assertAlmostEqual(reference.landing_turn_start[0], angle)
        self.assertAlmostEqual(reference.landing_turn_start[1], turn - 0.001)
        # It stays turned, even if a later prediction leaves more time.
        q = reference.update(FlightPhase.DESCENT, camera, 0.0, lambda yaw: plane, DT,
                             attitude=tracked, time_to_impact=5.0)
        np.testing.assert_allclose(q, plane, atol=1e-12)
        # A smaller angle turns later: 0 deg needs only the base time.
        reference.start_flight(LEVEL, np.zeros(3), 0.0, track=True)
        self.assertIsNone(reference.landing_turn_start)
        reference.update(FlightPhase.DESCENT, camera, 0.0, lambda yaw: plane, DT,
                         attitude=plane, time_to_impact=config.turn_time_base + 0.001)
        self.assertTrue(reference.tracking_target(FlightPhase.DESCENT))
        reference.update(FlightPhase.DESCENT, camera, 0.0, lambda yaw: plane, DT,
                         attitude=plane, time_to_impact=config.turn_time_base - 0.001)
        self.assertFalse(reference.tracking_target(FlightPhase.DESCENT))
        # LANDING (touchdown near) never tracks.
        self.assertFalse(reference.tracking_target(FlightPhase.LANDING))

    def test_turn_time_margin_starts_the_landing_turn_earlier(self) -> None:
        config = FlightReferenceConfig(slew_rate=math.radians(1e6), turn_time_margin=0.15)
        reference = FlightReference(config, target=(2.0, 0.0))
        reference.start_flight(LEVEL, np.zeros(3), 0.0, track=True)
        camera = np.array([0.0, 0.0, 2.0])
        due = config.turn_time_base + 0.15  # attitude already on the plane: 0 deg
        reference.update(FlightPhase.DESCENT, camera, 0.0, lambda yaw: LEVEL, DT,
                         attitude=LEVEL, time_to_impact=due + 0.001)
        self.assertTrue(reference.tracking_target(FlightPhase.DESCENT))
        reference.update(FlightPhase.DESCENT, camera, 0.0, lambda yaw: LEVEL, DT,
                         attitude=LEVEL, time_to_impact=due - 0.001)
        self.assertFalse(reference.tracking_target(FlightPhase.DESCENT))

    def test_holds_the_lift_off_heading_in_flight(self) -> None:
        reference = FlightReference(FlightReferenceConfig(slew_rate=math.radians(1e6)),
                                    target=(0.0, 0.0))
        reference.start_flight(LEVEL, np.zeros(3), 0.3, track=True)
        camera = np.array([0.0, 0.0, 2.0])  # straight above the target: no tilt
        for _ in range(int(1.0 / DT)):
            q = reference.update(FlightPhase.ASCENT, camera, 0.3, lambda yaw: LEVEL, DT)
        self.assertLess(reference.axis_target_angle(q, camera), 1e-6)
        heading = quaternion_to_matrix(q)[:, 0]
        self.assertAlmostEqual(math.atan2(heading[1], heading[0]), 0.3, places=6)
        q = reference.update(FlightPhase.LANDING, camera, 0.3,
                             lambda yaw: (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)), DT)
        np.testing.assert_allclose(q, (math.cos(0.15), 0.0, 0.0, math.sin(0.15)), atol=1e-9)

class JumpAimTests(unittest.TestCase):
    FEET = np.array([[0.2, 0.2, 0], [0.2, -0.2, 0], [-0.2, 0.2, 0], [-0.2, -0.2, 0]])

    def test_lean_grows_with_distance_and_is_capped(self) -> None:
        config = JumpAimConfig()
        self.assertEqual(lean_angle(0.1, config), 0.0)  # no direction
        self.assertAlmostEqual(lean_angle(1.0, config), config.min_lean, delta=math.radians(0.2))
        self.assertAlmostEqual(lean_angle(15.0, config), (config.min_lean + config.max_lean) / 2)
        self.assertEqual(lean_angle(100.0, config), config.max_lean)

    def test_shortens_legs_towards_the_tilt_error_closest_most(self) -> None:
        aim = JumpAim(self.FEET)
        target = (30.0, 30.0, 0.0)  # along the front-left leg, robot level
        for _ in range(100):
            s = aim.update(np.eye(3), np.zeros(3), np.zeros(3), target, 0.01)
        self.assertGreater(s[0], s[1])
        self.assertAlmostEqual(s[1], s[2])
        self.assertAlmostEqual(s[3], 0.0)

    def test_stops_where_the_measured_lean_is_right(self) -> None:
        aim = JumpAim(self.FEET)
        lean = lean_angle(40.0)
        leaning = np.array([[math.cos(lean), 0, math.sin(lean)], [0, 1, 0],
                            [-math.sin(lean), 0, math.cos(lean)]])
        aim.update(leaning, np.zeros(3), np.zeros(3), (40.0, 0.0, 0.0), 0.01)
        before = aim.gradient.copy()
        aim.update(leaning, np.zeros(3), np.zeros(3), (40.0, 0.0, 0.0), 0.01)
        np.testing.assert_allclose(aim.gradient, before, atol=1e-12)

    def test_holds_while_moving_or_told_to(self) -> None:
        aim = JumpAim(self.FEET)
        aim.update(np.eye(3), np.array([0.0, 1.0, 0.0]), np.zeros(3), (40.0, 0.0, 0.0), 0.01)
        self.assertFalse(np.any(aim.gradient))
        aim.update(np.eye(3), np.zeros(3), np.zeros(3), (40.0, 0.0, 0.0), 0.01, hold=True)
        self.assertFalse(np.any(aim.gradient))

    def test_cap_scales_all_legs_and_reports_the_limit(self) -> None:
        aim = JumpAim(self.FEET)
        for _ in range(2000):  # robot never moves: the integrator runs into the cap
            s = aim.update(np.eye(3), np.zeros(3), np.zeros(3), (300.0, 300.0, 0.0), 0.01)
        self.assertAlmostEqual(s[0], JumpAimConfig().max_shortening)
        self.assertAlmostEqual(s[1], s[0] / 2)
        self.assertTrue(aim.limited)

    def test_no_lean_on_steep_ground(self) -> None:
        tilt = math.radians(25)
        steep = np.array([[1, 0, 0], [0, math.cos(tilt), -math.sin(tilt)],
                          [0, math.sin(tilt), math.cos(tilt)]])  # 25 deg roll
        s = JumpAim(self.FEET).update(steep, np.zeros(3), np.zeros(3), (40.0, 0.0, 0.0), 0.01)
        self.assertFalse(np.any(s))


if __name__ == "__main__":
    unittest.main()
