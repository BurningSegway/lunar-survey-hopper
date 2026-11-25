"""Point cloud of everything the lidar sees during tracked jumps.

From lift-off to touchdown of each jump (the ascent tracks the target), every
lidar return is added as a world-frame point. The cloud is written as a binary
little-endian PLY (x, y, z as float32, ``jump`` as uint16, ``t`` and
``range`` -- lidar-to-point distance -- as float32 per point), which
CloudCompare, MeshLab, Open3D and most point cloud tools read directly.

Where the robot touched down after each jump goes in a small JSON file next
to the cloud (:func:`touchdown_path`: ``[{"jump": n, "position": [x, y, z]},
...]``, the ground under the robot at first contact), written at touchdown,
so the landing spots are easy to find in the cloud. The predicted paths go
in another (:func:`arcs_path`: ``[{"jump": n, "points": [[x, y, z], ...],
"actual": [[x, y, z], ...]}, ...]``): every ballistic arc the trajectory
tracker latched during the jump and its bounces, sampled from its latch to
where it ended (touchdown), with the robot's actual (true) base path over
the same time, written as each one ends.

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
        self.arcs: list[tuple[int, np.ndarray, np.ndarray]] = []  # (jump, predicted, actual)

    @property
    def point_count(self) -> int:
        return int(sum(len(c) for c in self._chunks))

    def start_run(self) -> None:
        """Forget earlier points and overwrite the file with an empty cloud."""
        self._chunks.clear()
        self.jumps = 0
        self.touchdowns.clear()
        self.arcs.clear()
        self.save()
        self._save_touchdowns()
        self._save_arcs()

    def add_touchdown(self, position) -> None:
        """Record where the current jump touched down (written at once)."""
        if self.jumps == 0:
            return
        self.touchdowns.append((self.jumps, tuple(float(v) for v in position)))
        self._save_touchdowns()

    def add_arc(self, points, actual=()) -> None:
        """Record one predicted path (N x 3) of the current jump or its
        bounces, and the actual path over the same time (M x 3; written at
        once)."""
        if self.jumps == 0:
            return
        self.arcs.append((self.jumps, np.asarray(points, float).reshape(-1, 3),
                          np.asarray(actual, float).reshape(-1, 3)))
        self._save_arcs()

    def _save_touchdowns(self) -> None:
        _write_json(touchdown_path(self.path),
                    [{"jump": jump, "position": list(position)} for jump, position in self.touchdowns])

    def _save_arcs(self) -> None:
        _write_json(arcs_path(self.path),
                    [{"jump": jump, "points": np.round(points, 4).tolist(),
                      "actual": np.round(actual, 4).tolist()}
                     for jump, points, actual in self.arcs])

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


def _write_json(path: Path, entries) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(entries))
    temporary.replace(path)


def arcs_path(cloud_path: str | Path) -> Path:
    """The predicted-path file that goes with a cloud: ``scan.ply`` -> ``scan.arcs.json``."""
    return Path(cloud_path).with_suffix(".arcs.json")


def read_arcs(cloud_path: str | Path) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """(jump, N x 3 predicted path, M x 3 actual path) per arc; empty if
    there is no file."""
    try:
        entries = json.loads(arcs_path(cloud_path).read_text())
    except (OSError, ValueError):
        return []
    return [(int(e["jump"]), np.array(e["points"], float).reshape(-1, 3),
             np.array(e.get("actual", []), float).reshape(-1, 3)) for e in entries]


def path_errors(predicted: np.ndarray, actual: np.ndarray) -> list[tuple[str, np.ndarray, float]]:
    """Prediction error of one arc: ``(where, actual position, distance)``
    at its apex (the predicted path's highest sample, if the arc rises
    first) and at its end (touchdown). Both paths are sampled on the same
    grid from the latch and end at the touchdown."""
    predicted, actual = np.asarray(predicted, float), np.asarray(actual, float)
    if len(predicted) < 2 or len(actual) < 2:
        return []
    errors = []
    apex = int(np.argmax(predicted[:, 2]))
    if 0 < apex < min(len(predicted), len(actual)) - 1:
        errors.append(("apex", actual[apex], float(np.linalg.norm(predicted[apex] - actual[apex]))))
    errors.append(("touchdown", actual[-1], float(np.linalg.norm(predicted[-1] - actual[-1]))))
    return errors


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
