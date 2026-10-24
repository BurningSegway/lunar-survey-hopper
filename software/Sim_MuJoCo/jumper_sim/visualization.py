"""Viewer overlays: the camera's estimated landing plane.

Drawn into a ``mujoco.MjvScene`` (the passive viewer's ``user_scn``):

* a translucent disc on the latest single-frame plane fit, the size of the
  fitting region: orange when that fit updated the reference, grey when it
  did not (too far away to trust, frozen just before touchdown, rejected as
  an outlier) or while the robot is in contact and the estimate is held;
* a green arrow along the filtered reference normal the LQR aligns body z to;
* small blue dots: a sample of the depth points the fit used;
* the camera's tracking target (red sphere) while a hop is under way, with a
  red line from the camera to it during the ascent;
* cyan dots: a sample of the target point cloud collected so far (with
  ``--scan-file`` scanning on), refreshed after each jump's ascent.
"""

from __future__ import annotations

import mujoco
import numpy as np

from .simulation import JumperSimulation

DISC_FLIGHT_RGBA = np.array([1.0, 0.55, 0.1, 0.35], dtype=np.float32)
DISC_HELD_RGBA = np.array([0.6, 0.6, 0.6, 0.25], dtype=np.float32)
ARROW_RGBA = np.array([0.1, 0.85, 0.2, 0.9], dtype=np.float32)
POINT_RGBA = np.array([0.2, 0.5, 1.0, 0.9], dtype=np.float32)
TARGET_RGBA = np.array([0.95, 0.15, 0.15, 0.9], dtype=np.float32)
SIGHT_RGBA = np.array([0.95, 0.15, 0.15, 0.6], dtype=np.float32)
TARGET_RADIUS = 0.06
ARROW_LENGTH = 0.45
POINT_RADIUS = 0.008
SCAN_RGBA = np.array([0.1, 0.9, 0.9, 0.8], dtype=np.float32)
SCAN_POINT_RADIUS = 0.01


def frame_from_normal(normal: np.ndarray) -> np.ndarray:
    """Rotation matrix whose z axis is ``normal``."""
    z = np.asarray(normal, dtype=float)
    z = z / np.linalg.norm(z)
    helper = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = np.cross(helper, z)
    x /= np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z])


def _add_geom(scene: mujoco.MjvScene) -> mujoco.MjvGeom | None:
    if scene.ngeom >= scene.maxgeom:
        return None
    geom = scene.geoms[scene.ngeom]
    scene.ngeom += 1
    return geom


def draw_landing_plane(scene: mujoco.MjvScene, simulation: JumperSimulation,
                       show_points: bool = True) -> int:
    """Append the landing-plane overlay to ``scene``; returns geoms added."""
    reference = simulation.landing_reference
    if reference is None or reference.visual is None:
        return 0
    visual = reference.visual
    start = scene.ngeom
    disc = _add_geom(scene)
    if disc is not None:
        mujoco.mjv_initGeom(
            disc,
            mujoco.mjtGeom.mjGEOM_CYLINDER,
            np.array([visual.radius, 0.002, 0.0]),
            visual.anchor + 0.003 * visual.measured_normal,
            frame_from_normal(visual.measured_normal).ravel(),
            DISC_FLIGHT_RGBA if visual.used and not simulation.in_contact else DISC_HELD_RGBA,
        )
    arrow = _add_geom(scene)
    if arrow is not None:
        tip = visual.anchor + ARROW_LENGTH * reference.filter.normal
        mujoco.mjv_initGeom(arrow, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3),
                            np.zeros(3), np.eye(3).ravel(), ARROW_RGBA)
        mujoco.mjv_connector(arrow, mujoco.mjtGeom.mjGEOM_ARROW, 0.012,
                             visual.anchor, tip)
    if show_points:
        for point in visual.points:
            geom = _add_geom(scene)
            if geom is None:
                break
            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE,
                                np.array([POINT_RADIUS, 0.0, 0.0]), point,
                                np.eye(3).ravel(), POINT_RGBA)
    return scene.ngeom - start


def draw_flight_target(scene: mujoco.MjvScene, simulation: JumperSimulation) -> int:
    """Append the tracking target (and line of sight during ascent)."""
    reference = simulation.flight_reference
    if reference is None or reference.active_target is None or not reference.tracking:
        return 0
    from .flight_phase import FlightPhase

    start = scene.ngeom
    marker = _add_geom(scene)
    if marker is not None:
        mujoco.mjv_initGeom(marker, mujoco.mjtGeom.mjGEOM_SPHERE,
                            np.array([TARGET_RADIUS, 0.0, 0.0]), reference.active_target,
                            np.eye(3).ravel(), TARGET_RGBA)
    if simulation.phase is FlightPhase.ASCENT:
        line = _add_geom(scene)
        if line is not None:
            mujoco.mjv_initGeom(line, mujoco.mjtGeom.mjGEOM_LINE, np.zeros(3), np.zeros(3),
                                np.eye(3).ravel(), SIGHT_RGBA)
            mujoco.mjv_connector(line, mujoco.mjtGeom.mjGEOM_LINE, 2.0,
                                 simulation.data.site_xpos[simulation._camera_site],
                                 reference.active_target)
    return scene.ngeom - start


def draw_target_scan(scene: mujoco.MjvScene, simulation: JumperSimulation) -> int:
    """Append an even sample of the target point cloud collected so far."""
    scan = simulation.target_scan
    if scan is None:
        return 0
    start = scene.ngeom
    for point in scan.preview:
        geom = _add_geom(scene)
        if geom is None:
            break
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE,
                            np.array([SCAN_POINT_RADIUS, 0.0, 0.0]), point,
                            np.eye(3).ravel(), SCAN_RGBA)
    return scene.ngeom - start
