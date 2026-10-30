"""Landing attitude reference from the lidar points of the current jump.

Pipeline (identical in simulation and on hardware -- only the point source
differs):

1. :class:`GroundMap` collects the jump's lidar points (world frame) in a
   coarse horizontal grid. Everything since the jump started is kept, so
   ground the lidar saw earlier in the hop counts too (its blind cone and
   the band's edges leave parts unseen at any one moment).
2. :meth:`GroundMap.predict_landing` follows the robot's ballistic trajectory
   (position, velocity, gravity) to where it first meets the mapped ground,
   ``touchdown_clearance`` below the base.
3. :func:`fit_ground_plane` fits the area-weighted "average" plane to up to
   ``max_points`` of the jump's points within ``roi_radius`` of that spot,
   with outlier rejection, and estimates how uncertain its normal is.
4. :class:`LandingReference` fuses successive plane normals with a small
   Kalman filter on the tilt vector (n_x, n_y): fits that are too noisy are
   ignored, clean fits dominate, and it keeps updating until touchdown (the
   lidar keeps seeing the spot, so the robot can still correct just before
   impact; ``freeze_near_touchdown`` restores the old freeze). It also tells
   when touchdown is near (it starts the LANDING phase). The tilt is clamped
   to ``max_tilt``.
5. :meth:`LandingReference.reference_quaternion` builds the attitude the
   controller should hold: body z along the plane normal, heading unchanged.

The points and the trajectory are in the world frame. In the simulator the
pose and velocity are the true ones; on the robot they come from odometry /
the state estimate.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class LandingReferenceConfig:
    roi_radius: float = 0.5  # m, disc around the predicted landing spot
    max_points: int = 1000  # most lidar points one fit uses (an even subsample)
    min_points: int = 12  # ground cells (not points) needed for a fit
    fit_period: float = 0.05  # s between fits
    map_cell: float = 0.5  # m, ground map grid for the landing prediction
    # Base height above the ground when the legs touch down: the trajectory
    # lands where the base comes this close to the mapped ground.
    touchdown_clearance: float = 0.3  # m
    max_tilt: float = math.radians(20.0)  # never ask for a steeper landing
    prior_sigma: float = math.radians(15.0)  # trust in "level" at take-off
    process_sigma: float = math.radians(10.0)  # tilt change per sqrt(second)
    outlier_gate: float = 4.0  # reject fits this many sigmas from the estimate
    cell_size: float = 0.0625  # m, ground grid for area-weighted fitting
    outlier_mad_factor: float = 3.0  # cells further than this many sigmas are outliers
    min_residual_floor: float = 0.02  # m, outlier threshold floor
    roughness_correlation: float = 0.25  # m, size of independent terrain bumps
    # Fits whose sensor-noise uncertainty exceeds this are ignored. Terrain
    # roughness does not count here -- it does not shrink as the robot descends.
    max_fit_sigma: float = math.radians(3.0)
    # Successive fits see the same ground, so their roughness errors are
    # correlated; never let the fused estimate claim more certainty than this.
    min_reference_sigma: float = math.radians(1.0)
    # Touchdown is near (the LANDING phase starts) below this height or this
    # time to touchdown. The reference keeps updating then, unless frozen.
    near_height: float = 1.0  # m of the landing plane below the robot
    near_time: float = 0.3  # s to touchdown at the measured descent rate
    freeze_near_touchdown: bool = False


@dataclass(frozen=True)
class PlaneEstimate:
    valid: bool
    normal: tuple[float, float, float] = (0.0, 0.0, 1.0)  # world frame, up
    normal_sigma: float = math.inf  # rad, 1-sigma uncertainty of the normal
    noise_sigma: float = math.inf  # rad, the part of it due to sensor noise
    height: float = math.nan  # m, robot origin above the plane at the landing spot
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


class GroundMap:
    """The current jump's lidar points in a horizontal grid of ``cell`` metres."""

    def __init__(self, cell: float = 0.5) -> None:
        self.cell = cell
        self.clear()

    def clear(self) -> None:
        self._cells: dict[tuple[int, int], list[np.ndarray]] = {}
        self._heights: dict[tuple[int, int], float] = {}  # cached median z per cell
        self.count = 0

    def add(self, points: np.ndarray) -> None:
        if not len(points):
            return
        keys = np.floor(points[:, :2] / self.cell).astype(np.int64)
        unique, inverse = np.unique(keys, axis=0, return_inverse=True)
        inverse = inverse.ravel()
        order = np.argsort(inverse, kind="stable")
        bounds = np.searchsorted(inverse[order], np.arange(len(unique) + 1))
        for (i, j), start, end in zip(unique, bounds[:-1], bounds[1:]):
            key = (int(i), int(j))
            self._cells.setdefault(key, []).append(points[order[start:end]])
            self._heights.pop(key, None)
        self.count += len(points)

    def _cell_points(self, key: tuple[int, int]) -> np.ndarray | None:
        chunks = self._cells.get(key)
        if chunks is None:
            return None
        if len(chunks) > 1:
            chunks[:] = [np.concatenate(chunks)]
        return chunks[0]

    def _cell_height(self, key: tuple[int, int]) -> float:
        if key not in self._heights:
            points = self._cell_points(key)
            self._heights[key] = math.nan if points is None else float(np.median(points[:, 2]))
        return self._heights[key]

    def height(self, x: float, y: float) -> float:
        """Median ground height in the cell at (x, y), else the mean over its
        seen neighbours; NaN if none of them has points."""
        i, j = math.floor(x / self.cell), math.floor(y / self.cell)
        own = self._cell_height((i, j))
        if math.isfinite(own):
            return own
        around = [self._cell_height((i + di, j + dj)) for di in (-1, 0, 1) for dj in (-1, 0, 1)]
        around = [h for h in around if math.isfinite(h)]
        return float(np.mean(around)) if around else math.nan

    def predict_landing(self, position, velocity, gravity: float, clearance: float,
                        step: float = 0.02, horizon: float = 15.0) -> tuple[np.ndarray, float] | None:
        """(x, y) and time [s] where the base's ballistic trajectory first
        comes within ``clearance`` of the mapped ground; None if it does not
        within ``horizon`` (that ground not seen)."""
        position = np.asarray(position, float)
        velocity = np.asarray(velocity, float)
        for t in np.arange(step, horizon, step):
            z = position[2] + velocity[2] * t - 0.5 * gravity * t * t
            x, y = position[:2] + velocity[:2] * t
            ground = self.height(x, y)
            if math.isfinite(ground) and z - clearance <= ground:
                return np.array([x, y]), float(t)
        return None

    def points_near(self, centre, radius: float, max_points: int) -> np.ndarray:
        """Up to ``max_points`` (an even subsample) of the points within
        ``radius`` (horizontally) of ``centre``."""
        reach = math.ceil(radius / self.cell)
        i0, j0 = math.floor(centre[0] / self.cell), math.floor(centre[1] / self.cell)
        chunks = [self._cell_points((i, j)) for i in range(i0 - reach, i0 + reach + 1)
                  for j in range(j0 - reach, j0 + reach + 1)]
        chunks = [c for c in chunks if c is not None]
        if not chunks:
            return np.zeros((0, 3))
        points = np.concatenate(chunks)
        points = points[np.hypot(points[:, 0] - centre[0], points[:, 1] - centre[1]) <= radius]
        if len(points) > max_points:
            points = points[np.linspace(0, len(points) - 1, max_points).astype(int)]
        return points


def fit_ground_plane(
    points: np.ndarray,
    centre,
    robot_z: float,
    point_sigma: float,
    timestamp: float,
    config: LandingReferenceConfig = LandingReferenceConfig(),
) -> PlaneEstimate:
    """Area-weighted "average" plane of the ground ``points`` (world) in the
    disc of ``roi_radius`` around ``centre`` (x, y).

    Points are dense where the lidar passed often and on faces that point at
    it, and sparse (or occluded) elsewhere, so fitting raw points lets those
    regions dominate -- on rough ground that can even flip the tilt. Points
    are therefore first reduced to one mean point per horizontal grid cell,
    and z = a*x + b*y + c is fitted to the cells (x, y relative to the
    centre), with iterative rejection of outlier cells (rocks, holes).
    """
    relative = np.asarray(points, float).reshape(-1, 3) - (centre[0], centre[1], 0.0)
    cells, counts = _cell_medians(relative, config.cell_size)
    if len(cells) < config.min_points:
        return PlaneEstimate(False, points=len(cells), timestamp=timestamp)

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
        return PlaneEstimate(False, points=int(inliers.sum()), timestamp=timestamp)

    selected = cells[inliers]
    rms = float(np.sqrt(np.mean(residual[inliers] ** 2)))
    # Slope uncertainty of a least-squares plane is sigma / sqrt(spread in
    # x-y). Split the residual into sensor noise (independent per cell) and
    # terrain roughness, which is correlated over ``roughness_correlation``
    # and so only averages down with the number of independent terrain
    # patches seen, not the number of cells.
    xy = selected[:, :2] - selected[:, :2].mean(axis=0)
    spread = max(float(np.linalg.eigvalsh(xy.T @ xy)[0]), 1e-12)
    cell_count = len(selected)
    cell_noise = point_sigma / math.sqrt(max(1.0, float(np.median(counts[inliers]))))
    roughness = math.sqrt(max(rms**2 - cell_noise**2, 0.0))
    patches = max(
        3.0, cell_count * config.cell_size**2 / config.roughness_correlation**2
    )
    radius_of_gyration = math.sqrt(spread / cell_count)
    noise_variance = cell_noise**2 / spread
    roughness_variance = roughness**2 / (radius_of_gyration**2 * min(patches, cell_count))
    # Unseen parts of the disc (never in view, occluded) can tilt the true
    # average plane in ways the fit cannot see.
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
        height=float(robot_z - c),  # robot above the plane at the landing spot
        rms=rms,
        points=int(inliers.sum()),
        timestamp=timestamp,
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
        self.near_touchdown = 0  # fits with touchdown near
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
            self.near_touchdown += 1
            if self.config.freeze_near_touchdown:
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
        if height < self.config.near_height:
            return True
        return self.descent_rate > 0.0 and height / self.descent_rate < self.config.near_time

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
