"""Live Open3D window for the target point cloud.

Watches the PLY file written by :class:`jumper_sim.pointcloud.PointCloudRecorder`
and redraws whenever it changes, i.e. after every jump's ascent. Points are
coloured by jump; the track target (if given) is a red sphere and a world
axis frame marks the origin (x red, y green, z blue, 0.5 m). The camera view
you set with the mouse is kept across updates; it is only fitted to the cloud
the first time points arrive (press R in the window to refit).

The simulator starts this automatically in its own process, so the Open3D
window cannot interfere with the MuJoCo viewer. It can also be run on its own:

    .venv/bin/python -m jumper_sim.cloud_viewer pointclouds/target_scan.ply

Open3D's own controls apply: left-drag rotate, right/ctrl-drag pan, wheel
zoom, +/- point size, R refit view, Q/Esc close.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import time

import numpy as np

from .pointcloud import read_ply

POLL_PERIOD = 0.5  # s between file checks
FRAME_PERIOD = 1.0 / 30.0  # s between window redraws
POINT_SIZE = 2.0
TARGET_RADIUS = 0.12
VIEW_FRONT = (-0.55, -0.45, 0.70)  # initial view: oblique, from above
VIEW_ZOOM = 0.7
BACKGROUND = (0.08, 0.09, 0.11)

# Colour per jump (cycled): a tab10-like palette that stays distinct on a
# dark background.
JUMP_COLOURS = np.array([
    (0.12, 0.47, 0.71), (1.00, 0.50, 0.05), (0.17, 0.63, 0.17), (0.84, 0.15, 0.16),
    (0.58, 0.40, 0.74), (0.55, 0.34, 0.29), (0.89, 0.47, 0.76), (0.50, 0.50, 0.50),
    (0.74, 0.74, 0.13), (0.09, 0.75, 0.81),
])


def load_cloud(path: Path) -> np.ndarray | None:
    """The cloud, or None while the file is missing or unreadable."""
    try:
        return read_ply(path)
    except (OSError, ValueError, StopIteration):
        return None


def jump_colours(jumps: np.ndarray) -> np.ndarray:
    return JUMP_COLOURS[(np.asarray(jumps, int) - 1) % len(JUMP_COLOURS)]


class CloudFile:
    """Watches the PLY file; hands out points and colours when it changes.

    Kept free of Open3D so it can be tested without a window.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._signature = None
        self.points = np.empty((0, 3))
        self.colours = np.empty((0, 3))
        self.jumps = 0

    def poll(self) -> bool:
        """Reload if the file changed; returns True if the cloud changed."""
        try:
            stat = self.path.stat()
            signature = (stat.st_mtime_ns, stat.st_size)
        except OSError:
            signature = None
        if signature == self._signature:
            return False
        cloud = load_cloud(self.path) if signature is not None else None
        if signature is not None and cloud is None:
            return False  # unreadable right now; try again on the next poll
        self._signature = signature
        if cloud is None or not len(cloud):
            self.points, self.colours, self.jumps = np.empty((0, 3)), np.empty((0, 3)), 0
        else:
            self.points = np.column_stack([cloud["x"], cloud["y"], cloud["z"]]).astype(float)
            self.colours = jump_colours(cloud["jump"])
            self.jumps = len(np.unique(cloud["jump"]))
        return True


def _title(source: CloudFile) -> str:
    return f"Target point cloud: {len(source.points)} points, {source.jumps} jump(s)"


def run_window(path: Path, target: Sequence[float] | None = None) -> int:
    import open3d as o3d

    source = CloudFile(path)
    window = o3d.visualization.Visualizer()
    if not window.create_window(window_name="Target point cloud", width=960, height=720):
        print("[SCAN] Open3D could not open a window (no display?). "
              f"The file is still written: {path}")
        return 1
    options = window.get_render_option()
    options.point_size = POINT_SIZE
    options.background_color = np.array(BACKGROUND)

    cloud = o3d.geometry.PointCloud()
    window.add_geometry(cloud)
    window.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5),
                        reset_bounding_box=False)
    if target is not None:
        marker = o3d.geometry.TriangleMesh.create_sphere(radius=TARGET_RADIUS)
        marker.translate(np.asarray(target, float))
        marker.paint_uniform_color((0.95, 0.15, 0.15))
        marker.compute_vertex_normals()
        window.add_geometry(marker, reset_bounding_box=False)

    fitted = False
    next_poll = 0.0
    print(f"[SCAN] Watching {path}")
    while True:
        now = time.monotonic()
        if now >= next_poll:
            next_poll = now + POLL_PERIOD
            if source.poll():
                cloud.points = o3d.utility.Vector3dVector(source.points)
                cloud.colors = o3d.utility.Vector3dVector(source.colours)
                if len(source.points) and not fitted:
                    # First points: re-add so the view fits them; afterwards
                    # keep whatever view the user has set.
                    window.remove_geometry(cloud, reset_bounding_box=False)
                    window.add_geometry(cloud, reset_bounding_box=True)
                    view = window.get_view_control()
                    view.set_front(VIEW_FRONT)
                    view.set_up((0.0, 0.0, 1.0))
                    view.set_zoom(VIEW_ZOOM)
                    fitted = True
                else:
                    window.update_geometry(cloud)
                if not len(source.points):
                    fitted = False  # a new run emptied the file
                print(f"[SCAN] {_title(source)}")
        if not window.poll_events():
            break  # window closed
        window.update_renderer()
        time.sleep(FRAME_PERIOD)
    window.destroy_window()
    return 0


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m jumper_sim.cloud_viewer",
        description="Show the target point cloud in Open3D and redraw it whenever the file changes.",
    )
    parser.add_argument("path", type=Path, help="PLY file written by the simulator")
    parser.add_argument("--target", type=float, nargs=3, metavar=("X", "Y", "Z"),
                        help="draw the track target as a red sphere")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        import open3d  # noqa: F401
    except ImportError:
        print("[SCAN] Open3D is not installed, so the point cloud window cannot open "
              "(pip --python .venv/bin/python install open3d). "
              f"The file is still written: {args.path}")
        return 1
    return run_window(args.path, args.target)


if __name__ == "__main__":
    raise SystemExit(main())
