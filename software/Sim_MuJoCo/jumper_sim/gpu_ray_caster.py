"""GPU ray casting (NVIDIA Warp) for the lidars.

A lidar casts 200-240k rays/s, mostly near-horizontal ones that cross a lot of
terrain: the numpy :class:`HeightfieldRayCaster` runs the Mid-360 at ~10x
slower than real time. This casts the same triangles (the heightfield grid,
split along the same cell diagonal, plus the caster's objects such as the
target) with Warp's BVH ray queries on the GPU: 200k rays in ~10 ms, hits
identical to the numpy caster.
"""

from __future__ import annotations

from functools import cache

import numpy as np

from .ray_casting import HeightfieldRayCaster

try:
    import warp as wp
except ImportError:  # pragma: no cover - depends on the install
    wp = None


@cache
def gpu_available() -> bool:
    """True if Warp is installed and sees a CUDA device."""
    if wp is None:
        return False
    try:
        wp.config.log_level = wp.LOG_WARNING  # no banner or kernel-load lines
        wp.init()
        return wp.get_cuda_device_count() > 0
    except Exception:  # noqa: BLE001 - no driver, broken install, ...
        return False


if wp is not None:
    @wp.kernel
    def _cast_kernel(mesh: wp.uint64, origin: wp.vec3, directions: wp.array(dtype=wp.vec3),
                     max_range: float, distances: wp.array(dtype=float)):
        i = wp.tid()
        query = wp.mesh_query_ray(mesh, origin, directions[i], max_range)
        if query.result:
            distances[i] = query.t
        else:
            distances[i] = -1.0


class WarpRayCaster:
    """Drop-in :class:`RayCaster` for a :class:`HeightfieldRayCaster`, on the GPU.

    Follows the terrain caster's ``objects`` (the target is moved by
    re-placing it): the mesh is rebuilt when they change.
    """

    def __init__(self, terrain: HeightfieldRayCaster, device: str = "cuda") -> None:
        if wp is None:
            raise RuntimeError("warp is not installed")
        self.terrain = terrain
        self._device = device
        rows, cols = terrain._rows, terrain._cols
        x = terrain._x0 + terrain._dx * np.arange(cols)
        y = terrain._y0 + terrain._dy * np.arange(rows)
        xx, yy = np.meshgrid(x, y)
        self._terrain_vertices = np.column_stack([xx.ravel(), yy.ravel(), terrain._heights.ravel()])
        v00 = (np.arange(rows - 1)[:, None] * cols + np.arange(cols - 1)[None, :]).ravel()
        v10, v01 = v00 + 1, v00 + cols
        v11 = v01 + 1
        # The same split as the numpy caster: along the (0,0)-(1,1) diagonal.
        self._terrain_faces = np.concatenate([
            np.column_stack([v00, v10, v11]), np.column_stack([v00, v11, v01]),
        ])
        self._objects: list = []
        self._mesh = None
        self._build()

    def _build(self) -> None:
        vertices, faces = [self._terrain_vertices], [self._terrain_faces]
        count = len(self._terrain_vertices)
        for obj in self.terrain.objects:
            vertices.append(obj.vertices)
            faces.append(obj.faces + count)
            count += len(obj.vertices)
        self._objects = list(self.terrain.objects)
        self._mesh = wp.Mesh(
            points=wp.array(np.concatenate(vertices).astype(np.float32), dtype=wp.vec3, device=self._device),
            indices=wp.array(np.concatenate(faces).astype(np.int32).ravel(), dtype=wp.int32, device=self._device),
        )

    def cast(self, origin: np.ndarray, directions: np.ndarray, max_range: float) -> np.ndarray:
        """Distance along each unit direction to the terrain or an object; NaN for no hit."""
        if len(self._objects) != len(self.terrain.objects) or any(
            a is not b for a, b in zip(self._objects, self.terrain.objects)
        ):
            self._build()
        n = len(directions)
        if n == 0:
            return np.empty(0)
        rays = wp.array(np.asarray(directions, np.float32), dtype=wp.vec3, device=self._device)
        distances = wp.empty(n, dtype=float, device=self._device)
        wp.launch(_cast_kernel, dim=n, device=self._device,
                  inputs=[self._mesh.id, wp.vec3(*np.asarray(origin, float)), rays, float(max_range), distances])
        result = distances.numpy().astype(float)
        result[result < 0.0] = np.nan
        return result
