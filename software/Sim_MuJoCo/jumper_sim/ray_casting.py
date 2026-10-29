"""Ray casting against the terrain heightfield and objects on it (the lidar).

:class:`HeightfieldRayCaster` is an exact ray/heightfield intersection,
vectorised in numpy. MuJoCo's own ``mj_multiRay`` gives the same distances
(checked in the tests) but costs ~30 us per ray on the 257x257 terrain
heightfield. Objects standing on the terrain (the scan target) are added to it
as :class:`MeshRayCaster` s in ``HeightfieldRayCaster.objects``; a ray returns
the nearest of all. :class:`jumper_sim.gpu_ray_caster.WarpRayCaster` casts the
same triangles on the GPU. Any object with a matching ``cast`` method (see
:class:`RayCaster`) can be used instead, e.g. for non-heightfield terrain.
"""

from __future__ import annotations

from typing import Protocol

import mujoco
import numpy as np

TERRAIN_GEOM = "terrain_collision"
TERRAIN_GEOM_GROUP = 2
LIDAR_SITE = "lidar"
LIDAR_DOWN_SITE = "lidar_down"  # Sony AS-DT1, bottom face


class RayCaster(Protocol):
    def cast(self, origin: np.ndarray, directions: np.ndarray, max_range: float) -> np.ndarray:
        """Distance along each unit direction to the terrain; NaN for no hit."""
        ...


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
        # Static meshes on the terrain the camera also sees (e.g. the target).
        self.objects: list[MeshRayCaster] = []

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
        """Distance to the terrain or the nearest object on it; NaN for no hit."""
        result = self.cast_terrain(origin, directions, max_range)
        for obj in self.objects:
            hit = obj.cast(origin, directions, max_range)
            result = np.fmin(result, hit)  # fmin: a NaN (miss) never wins
        return result

    def cast_terrain(self, origin: np.ndarray, directions: np.ndarray, max_range: float) -> np.ndarray:
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


class MeshRayCaster:
    """Exact ray/triangle intersection (Moller-Trumbore) with a static mesh.

    Only rays passing the mesh's bounding sphere are tested against its
    triangles, so a small object far from the view costs next to nothing.
    """

    CHUNK = 64  # rays tested against all triangles at once

    def __init__(self, vertices: np.ndarray, faces: np.ndarray) -> None:
        vertices = np.asarray(vertices, float)
        faces = np.asarray(faces, int)
        self.vertices, self.faces = vertices, faces
        self._v0 = vertices[faces[:, 0]]
        self._e1 = vertices[faces[:, 1]] - self._v0
        self._e2 = vertices[faces[:, 2]] - self._v0
        self._det_axis = np.cross(self._e2, self._e1)
        lower, upper = vertices.min(axis=0), vertices.max(axis=0)
        self.centre = 0.5 * (lower + upper)
        self.radius = float(np.linalg.norm(vertices - self.centre, axis=1).max())
        self.top = float(upper[2])

    @classmethod
    def from_model(cls, model: mujoco.MjModel, geom_id: int) -> "MeshRayCaster":
        """World-frame triangles of a static (world body) mesh geom."""
        mesh = int(model.geom_dataid[geom_id])
        vadr, vnum = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
        fadr, fnum = int(model.mesh_faceadr[mesh]), int(model.mesh_facenum[mesh])
        rotation = np.zeros(9)
        mujoco.mju_quat2Mat(rotation, model.geom_quat[geom_id])
        vertices = model.mesh_vert[vadr : vadr + vnum] @ rotation.reshape(3, 3).T + model.geom_pos[geom_id]
        return cls(vertices, model.mesh_face[fadr : fadr + fnum])

    def cast(self, origin: np.ndarray, directions: np.ndarray, max_range: float) -> np.ndarray:
        origin = np.asarray(origin, float)
        result = np.full(len(directions), np.nan)
        to_centre = self.centre - origin
        along = directions @ to_centre
        miss_sq = float(to_centre @ to_centre) - along**2
        candidates = np.flatnonzero((miss_sq <= self.radius**2) & (along + self.radius > 0.0))
        if not len(candidates):
            return result
        # Moller-Trumbore with the triple products rewritten so that only the
        # ray direction d varies: det = d.(e2 x e1), u*det = d.(e2 x s),
        # v*det = d.(s x e1), t*det = e2.(s x e1). Each is then one matrix
        # product per chunk of rays instead of (rays, faces, 3) cross products.
        # (einsum, not @: OpenBLAS threads these small products and that is slower.)
        s = origin - self._v0  # (f, 3)
        q = np.cross(s, self._e1)
        u_axis = np.cross(self._e2, s)
        t_num = np.einsum("fk,fk->f", self._e2, q)
        for start in range(0, len(candidates), self.CHUNK):
            rays = candidates[start : start + self.CHUNK]
            d = directions[rays]
            det = np.einsum("rk,fk->rf", d, self._det_axis)
            with np.errstate(divide="ignore", invalid="ignore"):
                inv = 1.0 / det
                u = np.einsum("rk,fk->rf", d, u_axis) * inv
                v = np.einsum("rk,fk->rf", d, q) * inv
                t = t_num[None] * inv
            ok = (np.abs(det) > 1e-12) & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 0) & (t <= max_range)
            t = np.where(ok, t, np.inf).min(axis=1)
            result[rays] = np.where(np.isfinite(t), t, np.nan)
        return result


def _required(model: mujoco.MjModel, object_type: int, name: str) -> int:
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id < 0:
        raise ValueError(f"Model is missing required MuJoCo object {name!r}")
    return object_id
