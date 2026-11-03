"""Simulated lidar: the Livox Mid-360S.

The Mid-360S scans non-repetitively: the pattern never repeats, so the field of view
fills in over time instead of returning the same lines every frame. The
directions are Livox's own recorded pattern (4 s, in time order, from the
``mujoco-lidar`` package), played back at the real point rate and looped. It
sweeps the whole -7..52 deg elevation band about every 50 ms, so one 10 ms
batch covers only part of it. Simulating only a ``point_fraction`` of the
points takes every 1/fraction-th one, so the pattern still advances in real
time. Without the package, a low-discrepancy (R2) sequence fills the field of
view evenly instead.

The patterns are from MuJoCo-LiDAR / DISCOVERSE (Jia et al. 2025,
arXiv:2507.21981); cite it with lidar results (README, "Citations").

The sensor sits on the ``lidar`` site; its rotation axis (the "up" of the
sensor) is the site's -z axis. Its field of view is a band all the way round
that axis, from 38 to 97 deg off it (-7..52 deg elevation); within 38 deg of
the axis it is blind. On the robot (``build/Jumping_Robot.xml``) the site is
on the +x side face, flush with the bottom, with the axis pointing out of the
face and tilted 22.5 deg down: straight down (body -z) is then in the middle
of the band, so the robot can point its bottom straight at the target and
also sees the ground below it. The blind cone points out and slightly down
from that face. Rays are not blocked by the robot itself.

Points are cast in batches of ``batch_period`` from the pose at that moment,
i.e. perfectly de-skewed (on the robot, the per-point timestamps and the IMU
do that). Each point gets Gaussian range and direction noise from the
datasheet precision, which does not grow with range (except 4 cm within 1 m).

Datasheet values (Livox Mid-360S): 200k points/s (first return); range
0.1-40 m at 10 % reflectivity (it cuts off at 100 m on brighter surfaces;
regolith is ~10 %); range precision 2 cm (1 sigma at 10 m; 4 cm at 0.2 m,
quoted for 0.1-1 m); angular precision 0.15 deg.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
import importlib.util
import math
from pathlib import Path

import mujoco
import numpy as np

from .ray_casting import LIDAR_SITE, RayCaster, _required

# R2 low-discrepancy sequence (Roberts 2018): the generalised golden ratio.
_R2 = (0.7548776662466927, 0.5698402909980532)

# Sensor frame (z along the axis) in the lidar site frame: the axis is the
# site's -z.
SITE_FROM_SENSOR = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])


@cache
def scan_pattern(name: str) -> np.ndarray | None:
    """Recorded Livox pattern ``name`` (rows of two angles [rad], in emission
    order) from the mujoco-lidar package; None if it is not installed."""
    spec = importlib.util.find_spec("mujoco_lidar")
    if spec is None or not spec.submodule_search_locations:
        return None
    path = Path(spec.submodule_search_locations[0]) / "scan_mode" / f"{name}.npy"
    if not path.is_file():
        return None
    return np.load(path, allow_pickle=False).astype(float)


@dataclass(frozen=True)
class LidarSpec:
    name: str
    point_rate: float  # points/s
    min_range: float  # m
    max_range: float  # m
    range_sigma: float  # m, 1 sigma
    angular_sigma: float  # rad, 1 sigma
    # Range sigma within a close range instead: (range, sigma).
    close_sigma: tuple[float, float] | None = None
    # Field of view about the axis: off-axis angles [min, max], all the way round.
    ring: tuple[float, float] = (0.0, math.pi)
    # Recorded Livox pattern (mujoco-lidar ``scan_mode`` file name): rows of
    # (azimuth, elevation above the plane normal to the axis).
    pattern: str | None = None
    # Simulation only: points are cast in batches this long, and only this
    # fraction of the real point rate is cast (ray casting dominates the cost).
    batch_period: float = 0.01  # s
    point_fraction: float = 1.0

    def directions(self, start: int, count: int) -> np.ndarray:
        """Unit directions of simulated points ``start`` .. ``start + count``
        in the sensor frame (z along the axis), shape (count, 3)."""
        emitted = np.floor(np.arange(start, start + count) / self.point_fraction).astype(np.int64)
        table = None if self.pattern is None else scan_pattern(self.pattern)
        if table is not None:
            azimuth, elevation = table[emitted % len(table)].T
            return np.column_stack([np.cos(elevation) * np.cos(azimuth),
                                    np.cos(elevation) * np.sin(azimuth), np.sin(elevation)])
        index = np.arange(start, start + count, dtype=float)
        u = np.mod(0.5 + _R2[0] * index, 1.0)
        v = np.mod(0.5 + _R2[1] * index, 1.0)
        # Even in solid angle: cos(off-axis angle) uniform between the bounds.
        cos_low, cos_high = math.cos(self.ring[0]), math.cos(self.ring[1])
        cos_theta = cos_low - v * (cos_low - cos_high)
        sin_theta = np.sqrt(np.maximum(0.0, 1.0 - cos_theta**2))
        azimuth = 2.0 * math.pi * u
        return np.column_stack([sin_theta * np.cos(azimuth), sin_theta * np.sin(azimuth), cos_theta])


LIVOX_MID360 = LidarSpec(
    name="Livox Mid-360S",
    point_rate=200_000.0,
    min_range=0.1,
    max_range=40.0,
    range_sigma=0.02,
    angular_sigma=math.radians(0.15),
    close_sigma=(1.0, 0.04),
    ring=(math.radians(90.0 - 52.0), math.radians(90.0 + 7.0)),
    pattern="mid360",
)

@dataclass(frozen=True)
class LidarPoints:
    points: np.ndarray  # (n, 3) world positions of the returns
    ranges: np.ndarray  # (n,) measured range [m]
    origin: np.ndarray  # sensor position (world) at capture
    timestamp: float


class SimulatedLidar:
    """Casts the points the lidar emits over a time span from the site's pose."""

    def __init__(self, model: mujoco.MjModel, spec: LidarSpec, ray_caster: RayCaster,
                 seed: int = 0) -> None:
        self.spec = spec
        self._site_id = _required(model, mujoco.mjtObj.mjOBJ_SITE, LIDAR_SITE)
        self._ray_caster = ray_caster
        self._rng = np.random.default_rng(seed)
        self._next_index = 0
        self._carry = 0.0  # fractional points owed from earlier batches

    def scan(self, data: mujoco.MjData, duration: float) -> LidarPoints:
        """Returns of the points emitted over the last ``duration`` seconds."""
        spec = self.spec
        wanted = spec.point_rate * spec.point_fraction * duration + self._carry
        count = int(wanted)
        self._carry = wanted - count
        directions = spec.directions(self._next_index, count)
        self._next_index += count
        rotation = data.site_xmat[self._site_id].reshape(3, 3)
        origin = data.site_xpos[self._site_id].copy()
        world = directions @ (rotation @ SITE_FROM_SENSOR).T
        distance = self._ray_caster.cast(origin, world, spec.max_range)
        hit = np.isfinite(distance) & (distance >= spec.min_range)
        distance, world = distance[hit], world[hit]
        # Measurement noise: range along the ray, and a small direction error.
        sigma = np.full(len(distance), spec.range_sigma)
        if spec.close_sigma is not None:
            sigma[distance < spec.close_sigma[0]] = spec.close_sigma[1]
        distance = distance + self._rng.standard_normal(len(distance)) * sigma
        world = world + self._rng.standard_normal(world.shape) * spec.angular_sigma
        world /= np.linalg.norm(world, axis=1, keepdims=True)
        return LidarPoints(origin + world * distance[:, None], distance, origin, float(data.time))
