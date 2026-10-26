"""Point cloud of the tracked target, composed from depth frames over jumps.

While a jump's ascent keeps the downward camera pointed at the track target,
every depth frame taken with the camera axis within ``margin`` of the target
direction is turned into world-frame points; frames from the initial turn
towards the target are skipped. Only points inside the keep sphere
(:meth:`PointCloudRecorder.keep_within`: around the target object, radius its
length) are added, so the cloud holds the target and the ground right at it
rather than everything the camera saw. The cloud is written as
a binary little-endian PLY (x, y, z as float32, ``jump`` as uint16 and ``t``
as float32 per point), which CloudCompare, MeshLab, Open3D and most point
cloud tools read directly.

:meth:`PointCloudRecorder.start_run` empties the file, so each program run
starts a fresh cloud; every jump of the run then adds to it, and the file is
rewritten after each jump's ascent so it is always up to date.

The world placement uses the robot position at capture time. In the
simulator that is the true position; on the robot it would come from an
odometry / localisation estimate.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np

from .depth_camera import CameraMount, DepthFrame
from .landing_reference import frame_points_world

PREVIEW_POINTS = 600
PLY_DTYPE =np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("jump", "<u2"), ("t", "<f4")])


@dataclass(frozen=True)
class ScanConfig:
    margin: float = math.radians(5.0)  # max camera-to-target angle for a frame to count
    # Scan frame size in rays, same field of view as the camera. Ray casting
    # dominates the ascent's cost; the full 106x60 frame slowed the sim down.
    width: int = 53
    height: int = 30


class PointCloudRecorder:
    def __init__(self, path: str | Path, config: ScanConfig = ScanConfig()) -> None:
        self.path = Path(path)
        self.config = config
        self._chunks: list[np.ndarray] = []
        self.jumps = 0
        self.frames = 0
        self.preview = np.empty((0, 3))
        # Keep sphere (centre, radius); None keeps every point.
        self.region: tuple[np.ndarray, float] | None = None

    @property
    def point_count(self) -> int:
        return int(sum(len(c) for c in self._chunks))

    def start_run(self) -> None:
        """Forget earlier points and overwrite the file with an empty cloud."""
        self._chunks.clear()
        self.jumps = 0
        self.frames = 0
        self.save()

    def start_jump(self) -> int:
        self.jumps += 1
        return self.jumps

    def keep_within(self, centre, radius: float | None) -> None:
        """Only add points within ``radius`` of ``centre`` (None: keep all)."""
        self.region = None if radius is None else (np.asarray(centre, float), float(radius))

    def add_frame(self, frame: DepthFrame, mount: CameraMount, pointing_error: float,
                  robot_position: np.ndarray | None = None) -> int:
        """Add one frame's points if it was taken on target; returns points added.

        ``robot_position`` (world, at capture) defaults to the frame's own
        ``body_position``.
        """
        if not pointing_error <= self.config.margin or self.jumps == 0:
            return 0
        if robot_position is None:
            robot_position = frame.body_position
        points = frame_points_world(frame, mount) + np.asarray(robot_position, float)
        if self.region is not None:
            centre, radius = self.region
            points = points[np.sum((points - centre) ** 2, axis=1) <= radius**2]
        if not len(points):
            return 0
        chunk = np.empty(len(points), PLY_DTYPE)
        chunk["x"], chunk["y"], chunk["z"] = points[:, 0], points[:, 1], points[:, 2]
        chunk["jump"] = self.jumps
        chunk["t"] = frame.timestamp
        self._chunks.append(chunk)
        self.frames += 1
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
        "comment jumper_sim target scan (world frame, metres)\n"
        f"element vertex {len(points)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property ushort jump\n"
        "property float t\n"
        "end_header\n"
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "wb") as handle:
        handle.write(header.encode("ascii"))
        handle.write(np.ascontiguousarray(points, PLY_DTYPE).tobytes())
    temporary.replace(path)  # never leave a half-written file behind


def read_ply(path: str | Path) -> np.ndarray:
    """Read a cloud written by :func:`write_ply`."""
    data = Path(path).read_bytes()
    end = data.index(b"end_header\n") + len(b"end_header\n")
    count = next(
        int(line.split()[2]) for line in data[:end].decode("ascii").splitlines()
        if line.startswith("element vertex")
    )
    return np.frombuffer(data[end:], PLY_DTYPE, count=count)
