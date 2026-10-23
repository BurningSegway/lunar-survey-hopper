"""Simulated downward-looking depth camera (Intel RealSense D435 stand-in).

The camera is a MuJoCo ``<site>`` on the robot. Its optical axis is the site's
-z axis, image x is the site's +x axis and image y is the site's -y axis, so a
site with the body's orientation on the bottom face looks straight down.

Depth is produced by ray casting one ray per (decimated) pixel against the
terrain. Two ray casters are provided:

* :class:`HeightfieldRayCaster` -- vectorised ray marching against the terrain
  heightfield. Fast enough for 30 Hz in the interactive viewer.
* :class:`MujocoRayCaster` -- ``mj_multiRay`` against any geoms in a geom
  group. Exact for any geometry, but ~30 us per ray on the 257x257 terrain
  heightfield, so it is mainly a reference for testing.

The output :class:`DepthFrame` is what a real D435 driver would hand over: a
z-depth image (NaN where invalid), the intrinsics, a timestamp and the body
attitude from the IMU at capture time. Everything downstream of it
(:mod:`jumper_sim.landing_reference`) is meant to run unchanged on hardware.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol

import mujoco
import numpy as np

TERRAIN_GEOM = "terrain_collision"
TERRAIN_GEOM_GROUP = 2
CAMERA_SITE = "realsense"

# Optical frame (x right, y down, z forward) expressed in the camera site frame.
SITE_FROM_OPTICAL = np.array(
    [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]
)


@dataclass(frozen=True)
class DepthCameraSpec:
    """RealSense D435 depth stream, decimated for simulation speed.

    Native depth is 848x480 at 87 x 58 deg; the default 8x decimation keeps the
    field of view and gives 106x60 rays. Noise follows the stereo model
    sigma_z = z^2 * subpixel / (f * baseline) at the native focal length, which
    is about 0.7 % at 2 m (Intel specifies < 2 %).
    """

    width: int = 106
    height: int = 60
    horizontal_fov: float = math.radians(87.0)
    vertical_fov: float = math.radians(58.0)
    min_range: float = 0.3  # m, closer returns are invalid
    max_range: float = 10.0  # m
    frame_rate: float = 30.0  # Hz
    latency: float = 0.05  # s from exposure until the frame is usable
    native_width: int = 848
    baseline: float = 0.050  # m
    subpixel_rms: float = 0.08  # disparity noise [px]
    dropout_probability: float = 0.02  # random invalid pixels
    noise: bool = True

    @property
    def intrinsics(self) -> tuple[float, float, float, float]:
        """(fx, fy, cx, cy) in decimated pixels."""
        fx = 0.5 * self.width / math.tan(0.5 * self.horizontal_fov)
        fy = 0.5 * self.height / math.tan(0.5 * self.vertical_fov)
        return fx, fy, 0.5 * (self.width - 1), 0.5 * (self.height - 1)

    def depth_sigma(self, depth: np.ndarray) -> np.ndarray:
        native_fx = 0.5 * self.native_width / math.tan(0.5 * self.horizontal_fov)
        return depth**2 * self.subpixel_rms / (native_fx * self.baseline)

    def optical_rays(self) -> np.ndarray:
        """Unit ray directions in the optical frame, shape (height*width, 3)."""
        fx, fy, cx, cy = self.intrinsics
        u, v = np.meshgrid(np.arange(self.width), np.arange(self.height))
        rays = np.stack(
            [(u - cx) / fx, (v - cy) / fy, np.ones_like(u, dtype=float)], axis=-1
        ).reshape(-1, 3)
        return rays / np.linalg.norm(rays, axis=1, keepdims=True)


@dataclass(frozen=True)
class CameraMount:
    """Fixed pose of the camera site in the robot's base body frame."""

    position: tuple[float, float, float]
    rotation: tuple[tuple[float, float, float], ...]  # body_from_site

    @classmethod
    def from_model(cls, model: mujoco.MjModel, site_name: str = CAMERA_SITE) -> "CameraMount":
        site_id = _required(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        rotation = np.zeros(9)
        mujoco.mju_quat2Mat(rotation, model.site_quat[site_id])
        return cls(
            position=tuple(float(v) for v in model.site_pos[site_id]),
            rotation=tuple(tuple(float(v) for v in row) for row in rotation.reshape(3, 3)),
        )

    @property
    def body_from_optical(self) -> np.ndarray:
        return np.array(self.rotation) @ SITE_FROM_OPTICAL


@dataclass(frozen=True)
class DepthFrame:
    depth: np.ndarray  # (height, width) z-depth in metres, NaN = invalid
    spec: DepthCameraSpec
    timestamp: float
    body_quaternion: tuple[float, float, float, float]  # IMU attitude (w, x, y, z)


class RayCaster(Protocol):
    def cast(self, origin: np.ndarray, directions: np.ndarray, max_range: float) -> np.ndarray:
        """Distance along each unit direction to the terrain; NaN for no hit."""
        ...


class MujocoRayCaster:
    """``mj_multiRay`` against the geoms in ``TERRAIN_GEOM_GROUP``."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self._model = model
        self._data = data
        self._group = np.zeros(6, dtype=np.uint8)
        self._group[TERRAIN_GEOM_GROUP] = 1

    def cast(self, origin: np.ndarray, directions: np.ndarray, max_range: float) -> np.ndarray:
        n = len(directions)
        geom_ids = np.zeros(n, dtype=np.int32)
        dist = np.zeros(n)
        vec = np.ascontiguousarray(directions, dtype=float).ravel()
        pnt = np.asarray(origin, dtype=float)
        try:  # MuJoCo >= 3.3 has an extra optional "normal" output.
            mujoco.mj_multiRay(self._model, self._data, pnt, vec, self._group, True,
                               -1, geom_ids, dist, None, n, max_range)
        except TypeError:
            mujoco.mj_multiRay(self._model, self._data, pnt, vec, self._group, True,
                               -1, geom_ids, dist, n, max_range)
        dist[geom_ids < 0] = np.nan
        return dist


class HeightfieldRayCaster:
    """Vectorised ray marching against an axis-aligned MuJoCo heightfield geom.

    The surface is the heightfield's grid with linear interpolation inside
    each triangle, using the same cell split as MuJoCo's collision mesh.
    Agreement with ``mj_multiRay`` is checked in the tests.
    """

    def __init__(self, model: mujoco.MjModel, geom_name: str = TERRAIN_GEOM,
                 ) -> None:
        geom_id = _required(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        if model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_HFIELD:
            raise ValueError(f"{geom_name!r} is not a heightfield geom")
        if not np.allclose(model.geom_quat[geom_id], (1.0, 0.0, 0.0, 0.0)):
            raise ValueError("HeightfieldRayCaster needs an axis-aligned heightfield")
        hfield = int(model.geom_dataid[geom_id])
        rows, cols = int(model.hfield_nrow[hfield]), int(model.hfield_ncol[hfield])
        size_x, size_y, size_z, _ = model.hfield_size[hfield]
        adr = int(model.hfield_adr[hfield])
        position = model.geom_pos[geom_id]
        self._heights = (
            model.hfield_data[adr : adr + rows * cols].reshape(rows, cols) * size_z
            + position[2]
        )
        self._x0 = position[0] - size_x
        self._y0 = position[1] - size_y
        self._dx = 2.0 * size_x / (cols - 1)
        self._dy = 2.0 * size_y / (rows - 1)
        self._rows, self._cols = rows, cols
        self._max_height = float(self._heights.max())

    def height(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Terrain height at (x, y); NaN outside the heightfield."""
        fx = (np.asarray(x) - self._x0) / self._dx
        fy = (np.asarray(y) - self._y0) / self._dy
        inside = (fx >= 0) & (fx <= self._cols - 1) & (fy >= 0) & (fy <= self._rows - 1)
        i = np.clip(np.floor(fy).astype(int), 0, self._rows - 2)
        j = np.clip(np.floor(fx).astype(int), 0, self._cols - 2)
        u = np.clip(fx - j, 0.0, 1.0)
        v = np.clip(fy - i, 0.0, 1.0)
        h00 = self._heights[i, j]
        h10 = self._heights[i, j + 1]
        h01 = self._heights[i + 1, j]
        h11 = self._heights[i + 1, j + 1]
        # Each cell is split along its (0,0)-(1,1) diagonal into two triangles.
        lower = u >= v
        h = np.where(
            lower,
            h00 + u * (h10 - h00) + v * (h11 - h10),
            h00 + v * (h01 - h00) + u * (h11 - h01),
        )
        return np.where(inside, h, np.nan)

    def cast(self, origin: np.ndarray, directions: np.ndarray, max_range: float) -> np.ndarray:
        """Exact ray/heightfield intersection.

        Along a ray the gap between ray and terrain is piecewise linear: it
        only changes slope where the ray crosses a grid line (x or y) or a
        cell diagonal. Stepping from breakpoint to breakpoint therefore finds
        the first sign change exactly, and the hit is a linear interpolation.
        """
        origin = np.asarray(origin, dtype=float)
        n = len(directions)
        result = np.full(n, np.nan)
        rate_x = directions[:, 0] / self._dx
        rate_y = directions[:, 1] / self._dy
        rate_w = rate_x - rate_y  # diagonal family: fx - fy = integer
        dz = directions[:, 2]
        start = np.zeros(n)
        if origin[2] > self._max_height:
            with np.errstate(divide="ignore"):
                start = np.where(dz < 0, (origin[2] - self._max_height) / -dz, np.inf)

        def gap(idx: np.ndarray, t: np.ndarray) -> np.ndarray:
            p = origin + directions[idx] * t[:, None]
            return p[:, 2] - self.height(p[:, 0], p[:, 1])

        def next_crossing(f: np.ndarray, rate: np.ndarray) -> np.ndarray:
            with np.errstate(divide="ignore", invalid="ignore"):
                target = np.where(rate > 0, np.floor(f) + 1.0, np.ceil(f) - 1.0)
                dt = (target - f) / rate
            return np.where(np.abs(rate) > 1e-12, dt, np.inf)

        fx0 = (origin[0] - self._x0) / self._dx
        fy0 = (origin[1] - self._y0) / self._dy
        active = np.flatnonzero(start < max_range)
        t = start[active]
        g = gap(active, t)
        alive = g > 0  # below ground or off the map: no hit
        active, t, g = active[alive], t[alive], g[alive]
        while active.size:
            fx = fx0 + t * rate_x[active]
            fy = fy0 + t * rate_y[active]
            dt = np.minimum.reduce([
                next_crossing(fx, rate_x[active]),
                next_crossing(fy, rate_y[active]),
                next_crossing(fx - fy, rate_w[active]),
            ])
            t_next = np.minimum(t + np.maximum(dt, 1e-9), max_range)
            g_next = gap(active, t_next)
            crossed = g_next <= 0  # NaN (left the map) compares False
            if np.any(crossed):
                g0, g1 = g[crossed], g_next[crossed]
                t0, t1 = t[crossed], t_next[crossed]
                result[active[crossed]] = t0 + (t1 - t0) * g0 / (g0 - g1)
            keep = ~crossed & np.isfinite(g_next) & (t_next < max_range)
            active, t, g = active[keep], t_next[keep], g_next[keep]
        return result

    def max_height_near(self, x: float, y: float, radius: float) -> float:
        """Highest terrain point within ``radius`` (square window) of (x, y)."""
        j0 = int(np.clip(np.floor((x - radius - self._x0) / self._dx), 0, self._cols - 1))
        j1 = int(np.clip(np.ceil((x + radius - self._x0) / self._dx), 0, self._cols - 1))
        i0 = int(np.clip(np.floor((y - radius - self._y0) / self._dy), 0, self._rows - 1))
        i1 = int(np.clip(np.ceil((y + radius - self._y0) / self._dy), 0, self._rows - 1))
        return float(self._heights[i0 : i1 + 1, j0 : j1 + 1].max())


class SimulatedDepthCamera:
    """Produces :class:`DepthFrame` s from the robot's camera site."""

    def __init__(
        self,
        model: mujoco.MjModel,
        root_joint_id: int,
        spec: DepthCameraSpec = DepthCameraSpec(),
        ray_caster: RayCaster | None = None,
        site_name: str = CAMERA_SITE,
        seed: int = 0,
    ) -> None:
        self.spec = spec
        self.mount = CameraMount.from_model(model, site_name)
        self._model = model
        self._site_id = _required(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        self._qpos_address = int(model.jnt_qposadr[root_joint_id])
        self._ray_caster = ray_caster or HeightfieldRayCaster(model)
        self._optical_rays = spec.optical_rays()
        self._site_rays = self._optical_rays @ SITE_FROM_OPTICAL.T
        self._rng = np.random.default_rng(seed)

    def capture(self, data: mujoco.MjData, roi_radius: float | None = None) -> DepthFrame:
        """Render one depth frame.

        ``roi_radius`` is a simulation-only speed-up: only pixels whose rays
        can reach the ground within ``roi_radius`` (horizontally) of the point
        straight below the camera are cast; all others are NaN. The plane fit
        only reads that region, so its result is unchanged. A real camera
        delivers the full frame at no cost, so leave it None there.
        """
        site_rotation = data.site_xmat[self._site_id].reshape(3, 3)
        origin = data.site_xpos[self._site_id]
        directions = self._site_rays @ site_rotation.T
        selected = np.arange(len(directions))
        if roi_radius is not None and isinstance(self._ray_caster, HeightfieldRayCaster):
            reach = 1.5 * roi_radius
            top = self._ray_caster.max_height_near(origin[0], origin[1], reach)
            drop = origin[2] - top
            if drop > 0.0:
                horizontal = np.hypot(directions[:, 0], directions[:, 1])
                down = -directions[:, 2]
                with np.errstate(divide="ignore", invalid="ignore"):
                    selected = np.flatnonzero((down > 0) & (horizontal / down <= reach / drop))
        distance = np.full(len(directions), np.nan)
        distance[selected] = self._ray_caster.cast(
            origin, directions[selected], self.spec.max_range * 1.6
        )
        depth = distance * self._optical_rays[:, 2]
        if self.spec.noise:
            valid = np.isfinite(depth)
            depth[valid] += self._rng.standard_normal(valid.sum()) * self.spec.depth_sigma(depth[valid])
            depth[self._rng.random(depth.shape) < self.spec.dropout_probability] = np.nan
        with np.errstate(invalid="ignore"):
            depth[(depth < self.spec.min_range) | (depth > self.spec.max_range)] = np.nan
        q = data.qpos[self._qpos_address + 3 : self._qpos_address + 7]
        return DepthFrame(
            depth=depth.reshape(self.spec.height, self.spec.width),
            spec=self.spec,
            timestamp=float(data.time),
            body_quaternion=tuple(float(v) for v in q),
        )


def _required(model: mujoco.MjModel, object_type: int, name: str) -> int:
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id < 0:
        raise ValueError(f"Model is missing required MuJoCo object {name!r}")
    return object_id
