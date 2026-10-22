"""Landing attitude reference from the downward depth camera.

Pipeline (identical in simulation and on hardware -- only the DepthFrame
source differs):

1. :func:`estimate_ground_plane` turns one depth frame into points, rotates
   them into a level, yaw-carrying frame with the IMU attitude stored in the
   frame, keeps the points in a disc of ``roi_radius`` straight below the
   robot, and fits the area-weighted "average" plane with outlier rejection.
   It also estimates how uncertain the fitted normal is.
2. :class:`LandingReference` fuses successive plane normals with a small
   Kalman filter on the tilt vector (n_x, n_y): fits that are too noisy
   (far away) are ignored, clean fits dominate, and the reference freezes
   shortly before touchdown. The tilt is clamped to ``max_tilt``.
3. :meth:`LandingReference.reference_quaternion` builds the attitude the
   controller should hold: body z along the plane normal, heading unchanged.

The normal is stored in the IMU's world frame, so yaw rotation of the robot
between frames is accounted for automatically. A slowly drifting IMU yaw only
rotates the frame the normal and the reference both live in, so it does not
bias the result; IMU roll/pitch at the frame timestamp do matter.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .depth_camera import CameraMount, DepthFrame


@dataclass(frozen=True)
class LandingReferenceConfig:
    roi_radius: float = 0.5  # m, disc below the robot that defines the plane
    min_points: int = 12  # ground cells (not pixels) needed for a fit
    max_tilt: float = math.radians(20.0)  # never ask for a steeper landing
    prior_sigma: float = math.radians(15.0)  # trust in "level" at take-off
    process_sigma: float = math.radians(10.0)  # tilt change per sqrt(second)
    outlier_gate: float = 4.0  # reject fits this many sigmas from the estimate
    cell_size: float = 0.0625  # m, ground grid for area-weighted fitting
    outlier_mad_factor: float = 3.0  # cells further than this many sigmas are outliers
    min_residual_floor: float = 0.02  # m, outlier threshold floor
    roughness_correlation: float = 0.25  # m, size of independent terrain bumps
    # Fits whose depth-noise uncertainty exceeds this are ignored (in
    # practice: from above ~5-6 m). Terrain roughness does not count here --
    # it does not shrink as the robot descends.
    max_fit_sigma: float = math.radians(3.0)
    # Successive fits see the same ground, so their roughness errors are
    # correlated; never let the fused estimate claim more certainty than this.
    min_reference_sigma: float = math.radians(1.0)
    # Freeze the reference shortly before touchdown: close-range fits only see
    # part of the disc, and a late change is one the LQR cannot follow anyway.
    freeze_height: float = 1.0  # m of plane below the robot
    freeze_time: float = 0.3  # s to touchdown at the measured descent rate


@dataclass(frozen=True)
class PlaneEstimate:
    valid: bool
    normal: tuple[float, float, float] = (0.0, 0.0, 1.0)  # world frame, up
    normal_sigma: float = math.inf  # rad, 1-sigma uncertainty of the normal
    noise_sigma: float = math.inf  # rad, the part of it due to depth noise
    height: float = math.nan  # m, plane below the robot origin
    rms: float = math.nan  # m, fit residual
    points: int = 0
    timestamp: float = math.nan


def quaternion_to_matrix(q: tuple[float, float, float, float]) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def matrix_to_quaternion(r: np.ndarray) -> tuple[float, float, float, float]:
    trace = r[0, 0] + r[1, 1] + r[2, 2]
    if trace > 0:
        s = 2.0 * math.sqrt(1.0 + trace)
        q = (0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s)
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = 2.0 * math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2])
        q = ((r[2, 1] - r[1, 2]) / s, 0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s)
    elif r[1, 1] > r[2, 2]:
        s = 2.0 * math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2])
        q = ((r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s)
    else:
        s = 2.0 * math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1])
        q = ((r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s)
    if q[0] < 0:
        q = tuple(-v for v in q)
    return tuple(float(v) for v in q)


def frame_points_world(frame: DepthFrame, mount: CameraMount) -> np.ndarray:
    """Valid pixels as points relative to the robot origin, world-oriented."""
    fx, fy, cx, cy = frame.spec.intrinsics
    v, u = np.nonzero(np.isfinite(frame.depth))
    z = frame.depth[v, u]
    optical = np.stack([(u - cx) / fx * z, (v - cy) / fy * z, z], axis=1)
    body = optical @ mount.body_from_optical.T + np.array(mount.position)
    return body @ quaternion_to_matrix(frame.body_quaternion).T


def estimate_ground_plane(
    frame: DepthFrame,
    mount: CameraMount,
    config: LandingReferenceConfig = LandingReferenceConfig(),
) -> PlaneEstimate:
    """Area-weighted "average" plane of the ground in the disc below the robot.

    Pixels are dense where the camera looks straight down and on faces that
    point at it, and sparse (or occluded) elsewhere, so fitting raw points
    lets those regions dominate -- on rough ground that can even flip the
    tilt. Points are therefore first reduced to one mean point per
    horizontal grid cell, and z = a*x + b*y + c is fitted to the cells, with
    iterative rejection of outlier cells (rocks, holes).
    """
    points = frame_points_world(frame, mount)
    if len(points):
        points = points[np.hypot(points[:, 0], points[:, 1]) <= config.roi_radius]
    depths = frame.depth[np.isfinite(frame.depth)]
    cells, counts = _cell_medians(points, config.cell_size)
    if len(cells) < config.min_points:
        return PlaneEstimate(False, points=len(cells), timestamp=frame.timestamp)

    inliers = np.ones(len(cells), dtype=bool)
    for _ in range(4):
        coefficients, residual = _fit_height_plane(cells, inliers)
        mad = np.median(np.abs(residual[inliers] - np.median(residual[inliers])))
        threshold = max(config.outlier_mad_factor * 1.4826 * mad, config.min_residual_floor)
        new_inliers = np.abs(residual) <= threshold
        if new_inliers.sum() < config.min_points or np.array_equal(new_inliers, inliers):
            break
        inliers = new_inliers
    coefficients, residual = _fit_height_plane(cells, inliers)
    a, b, c = coefficients
    normal = np.array([-a, -b, 1.0]) / math.sqrt(a * a + b * b + 1.0)
    if normal[2] < math.cos(math.radians(60.0)):
        return PlaneEstimate(False, points=int(inliers.sum()), timestamp=frame.timestamp)

    selected = cells[inliers]
    rms = float(np.sqrt(np.mean(residual[inliers] ** 2)))
    # Slope uncertainty of a least-squares plane is sigma / sqrt(spread in
    # x-y). Split the residual into sensor noise (independent per cell, from
    # the camera's noise model) and terrain roughness, which is correlated
    # over ``roughness_correlation`` and so only averages down with the
    # number of independent terrain patches seen, not the number of cells.
    xy = selected[:, :2] - selected[:, :2].mean(axis=0)
    spread = max(float(np.linalg.eigvalsh(xy.T @ xy)[0]), 1e-12)
    cell_count = len(selected)
    pixel_sigma = float(frame.spec.depth_sigma(np.median(depths))) if frame.spec.noise else 0.0
    cell_noise = pixel_sigma / math.sqrt(max(1.0, float(np.median(counts[inliers]))))
    roughness = math.sqrt(max(rms**2 - cell_noise**2, 0.0))
    patches = max(
        3.0, cell_count * config.cell_size**2 / config.roughness_correlation**2
    )
    radius_of_gyration = math.sqrt(spread / cell_count)
    noise_variance = cell_noise**2 / spread
    roughness_variance = roughness**2 / (radius_of_gyration**2 * min(patches, cell_count))
    # Unseen parts of the disc (outside the field of view, occluded) can tilt
    # the true average plane in ways the fit cannot see.
    disc_cells = math.pi * config.roi_radius**2 / config.cell_size**2
    coverage = min(1.0, cell_count / disc_cells)
    roughness_variance /= coverage**2
    normal_sigma = math.sqrt(noise_variance + roughness_variance + 1e-8)
    noise_sigma = math.sqrt(noise_variance)
    return PlaneEstimate(
        valid=True,
        normal=tuple(float(v) for v in normal),
        normal_sigma=float(normal_sigma),
        noise_sigma=float(noise_sigma),
        height=float(-c),  # plane height at x = y = 0, relative to the robot
        rms=rms,
        points=int(inliers.sum()),
        timestamp=frame.timestamp,
    )


def _cell_medians(points: np.ndarray, cell_size: float) -> tuple[np.ndarray, np.ndarray]:
    if not len(points):
        return np.zeros((0, 3)), np.zeros(0, dtype=int)
    keys = np.floor(points[:, :2] / cell_size).astype(np.int64)
    order = np.lexsort((keys[:, 1], keys[:, 0]))
    keys, points = keys[order], points[order]
    boundaries = np.flatnonzero(np.any(np.diff(keys, axis=0) != 0, axis=1)) + 1
    groups = np.split(points, boundaries)
    # Mean point per cell: exact on a plane, and noise averages down.
    cells = np.array([g.mean(axis=0) for g in groups])
    counts = np.array([len(g) for g in groups])
    return cells, counts


def _fit_height_plane(cells: np.ndarray, inliers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    design = np.column_stack([cells[:, 0], cells[:, 1], np.ones(len(cells))])
    coefficients = np.linalg.lstsq(design[inliers], cells[inliers, 2], rcond=None)[0]
    return coefficients, cells[:, 2] - design @ coefficients


class LandingReference:
    """Kalman filter on the plane tilt vector plus the reference attitude."""

    def __init__(self, config: LandingReferenceConfig = LandingReferenceConfig()) -> None:
        self.config = config
        self.reset()

    def reset(self, time: float | None = None) -> None:
        """Back to "land level" with prior uncertainty (e.g. at take-off)."""
        self._tilt = np.zeros(2)
        self._variance = self.config.prior_sigma**2
        self._time = time
        self.last_estimate: PlaneEstimate | None = None
        self.accepted = 0
        self.rejected = 0  # outliers
        self.ignored = 0  # too noisy (too far away)
        self.frozen = 0  # skipped because touchdown is close
        self._last_height: tuple[float, float] | None = None  # (time, height)
        self.descent_rate = 0.0  # m/s, from successive plane heights

    @property
    def tilt_vector(self) -> np.ndarray:
        """(n_x, n_y) of the reference normal, clamped to ``max_tilt``."""
        limit = math.sin(self.config.max_tilt)
        magnitude = float(np.linalg.norm(self._tilt))
        if magnitude > limit:
            return self._tilt * (limit / magnitude)
        return self._tilt.copy()

    @property
    def normal(self) -> np.ndarray:
        tx, ty = self.tilt_vector
        return np.array([tx, ty, math.sqrt(max(0.0, 1.0 - tx * tx - ty * ty))])

    @property
    def sigma(self) -> float:
        return math.sqrt(self._variance)

    def update(self, estimate: PlaneEstimate) -> bool:
        """Fuse one plane fit; returns whether it was accepted."""
        self.last_estimate = estimate
        if self._time is not None and math.isfinite(estimate.timestamp):
            dt = max(0.0, estimate.timestamp - self._time)
            self._variance += self.config.process_sigma**2 * dt
        if math.isfinite(estimate.timestamp):
            self._time = estimate.timestamp
        if not estimate.valid:
            return False
        self._update_descent_rate(estimate)
        if self.touchdown_close(estimate.height):
            self.frozen += 1
            return False
        noise = estimate.noise_sigma if math.isfinite(estimate.noise_sigma) else estimate.normal_sigma
        if noise > self.config.max_fit_sigma:
            self.ignored += 1
            return False
        measured = np.array(estimate.normal[:2])
        measurement_variance = estimate.normal_sigma**2
        innovation = measured - self._tilt
        if np.linalg.norm(innovation) > self.config.outlier_gate * math.sqrt(
            self._variance + measurement_variance
        ):
            self.rejected += 1
            return False
        gain = self._variance / (self._variance + measurement_variance)
        self._tilt = self._tilt + gain * innovation
        self._variance = max(self._variance * (1.0 - gain), self.config.min_reference_sigma**2)
        self.accepted += 1
        return True

    def touchdown_close(self, height: float) -> bool:
        if height < self.config.freeze_height:
            return True
        return self.descent_rate > 0.0 and height / self.descent_rate < self.config.freeze_time

    def _update_descent_rate(self, estimate: PlaneEstimate) -> None:
        if not (math.isfinite(estimate.height) and math.isfinite(estimate.timestamp)):
            return
        if self._last_height is not None:
            dt = estimate.timestamp - self._last_height[0]
            if dt > 1e-6:
                rate = (self._last_height[1] - estimate.height) / dt
                self.descent_rate += 0.3 * (rate - self.descent_rate)
        self._last_height = (estimate.timestamp, estimate.height)

    def reference_quaternion(self, yaw_reference: float) -> tuple[float, float, float, float]:
        """Attitude with body z along the reference normal and the given heading."""
        return tilted_reference_quaternion(self.normal, yaw_reference)


def tilted_reference_quaternion(
    normal: np.ndarray, yaw_reference: float
) -> tuple[float, float, float, float]:
    z_axis = np.asarray(normal, dtype=float)
    z_axis = z_axis / np.linalg.norm(z_axis)
    heading = np.array([math.cos(yaw_reference), math.sin(yaw_reference), 0.0])
    x_axis = heading - (heading @ z_axis) * z_axis
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    return matrix_to_quaternion(np.column_stack([x_axis, y_axis, z_axis]))
