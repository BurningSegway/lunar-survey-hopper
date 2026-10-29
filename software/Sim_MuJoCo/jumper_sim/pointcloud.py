"""Point cloud of everything the lidar sees during tracked jumps.

From lift-off to touchdown of each jump (the ascent tracks the target), every
lidar return is added as a world-frame point. The cloud is written as a binary
little-endian PLY (x, y, z as float32, ``jump`` as uint16, ``t`` and
``range`` -- lidar-to-point distance -- as float32 per point), which
CloudCompare, MeshLab, Open3D and most point cloud tools read directly.

Where the robot touched down after each jump goes in a small JSON file next
to the cloud (:func:`touchdown_path`: ``[{"jump": n, "position": [x, y, z]},
...]``, the ground under the robot at first contact), written at touchdown,
so the landing spots are easy to find in the cloud.

:meth:`PointCloudRecorder.start_run` empties the file, so each program run
starts a fresh cloud; every jump of the run then adds to it, and the file is
rewritten at each touchdown so it is always up to date.

The world placement uses the robot pose at capture time. In the simulator
that is the true pose; on the robot it would come from an odometry /
localisation estimate.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

PREVIEW_POINTS = 600
PLY_DTYPE = np.dtype([
    ("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("jump", "<u2"), ("t", "<f4"), ("range", "<f4"),
])


class PointCloudRecorder:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._chunks: list[np.ndarray] = []
        self.jumps = 0
        self.preview = np.empty((0, 3))
        self.touchdowns: list[tuple[int, tuple[float, float, float]]] = []  # (jump, x y z)

    @property
    def point_count(self) -> int:
        return int(sum(len(c) for c in self._chunks))

    def start_run(self) -> None:
        """Forget earlier points and overwrite the file with an empty cloud."""
        self._chunks.clear()
        self.jumps = 0
        self.touchdowns.clear()
        self.save()
        self._save_touchdowns()

    def add_touchdown(self, position) -> None:
        """Record where the current jump touched down (written at once)."""
        if self.jumps == 0:
            return
        self.touchdowns.append((self.jumps, tuple(float(v) for v in position)))
        self._save_touchdowns()

    def _save_touchdowns(self) -> None:
        path = touchdown_path(self.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(
            [{"jump": jump, "position": list(position)} for jump, position in self.touchdowns]
        ))
        temporary.replace(path)

    def start_jump(self) -> int:
        self.jumps += 1
        return self.jumps

    def add_points(self, points: np.ndarray, ranges: np.ndarray, timestamp: float) -> int:
        """Add world points (with their capture range) to the current jump;
        returns points added."""
        if self.jumps == 0 or not len(points):
            return 0
        chunk = np.empty(len(points), PLY_DTYPE)
        chunk["x"], chunk["y"], chunk["z"] = points[:, 0], points[:, 1], points[:, 2]
        chunk["jump"] = self.jumps
        chunk["t"] = timestamp
        chunk["range"] = ranges
        self._chunks.append(chunk)
        return len(chunk)

    def points(self) -> np.ndarray:
        return np.concatenate(self._chunks) if self._chunks else np.empty(0, PLY_DTYPE)

    def save(self) -> Path:
        points = self.points()
        write_ply(self.path, points)
        # Even subsample for the viewer overlay, refreshed at each save.
        step = max(1, len(points) // PREVIEW_POINTS)
        self.preview = np.column_stack([points["x"], points["y"], points["z"]])[::step].astype(float)
        return self.path


def write_ply(path: str | Path, points: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment jumper_sim lidar scan (world frame, metres)\n"
        f"element vertex {len(points)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property ushort jump\n"
        "property float t\n"
        "property float range\n"
        "end_header\n"
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "wb") as handle:
        handle.write(header.encode("ascii"))
        handle.write(np.ascontiguousarray(points, PLY_DTYPE).tobytes())
    temporary.replace(path)  # never leave a half-written file behind


def touchdown_path(cloud_path: str | Path) -> Path:
    """The touchdown file that goes with a cloud: ``scan.ply`` -> ``scan.touchdowns.json``."""
    return Path(cloud_path).with_suffix(".touchdowns.json")


def read_touchdowns(cloud_path: str | Path) -> list[tuple[int, np.ndarray]]:
    """(jump, position) per touchdown; empty if there is no file."""
    try:
        entries = json.loads(touchdown_path(cloud_path).read_text())
    except (OSError, ValueError):
        return []
    return [(int(e["jump"]), np.array(e["position"], float)) for e in entries]


def read_ply(path: str | Path) -> np.ndarray:
    """Read a cloud written by :func:`write_ply`."""
    data = Path(path).read_bytes()
    end = data.index(b"end_header\n") + len(b"end_header\n")
    header = data[:end].decode("ascii").splitlines()
    count = next(int(line.split()[2]) for line in header if line.startswith("element vertex"))
    return np.frombuffer(data[end:], PLY_DTYPE, count=count)
