"""Live Open3D window for the lidar point cloud.

Watches the PLY file written by :class:`jumper_sim.pointcloud.PointCloudRecorder`
and redraws whenever it changes, i.e. at every touchdown. Points are
coloured by jump; points within 0.3 m of the target are white and larger,
so the target is easy to find; touchdown spots (from the ``.touchdowns.json``
file next to the cloud) are labelled green spheres. The track target (if
given) is a small red sphere and a
world axis frame marks the origin (x red, y green, z blue, 0.5 m). The camera
view you set with the mouse is kept across updates; it is only fitted to the
cloud the first time points arrive (press R in the window to refit).

The simulator starts this automatically in its own process, so the Open3D
window cannot interfere with the MuJoCo viewer. It can also be run on its own:

    .venv/bin/python -m jumper_sim.cloud_viewer pointclouds/target_scan.ply

Open3D's own controls apply: left-drag rotate, ctrl-drag pan, wheel zoom.
R refits the view, Q/Esc closes. (Uses Open3D's GUI scene widget, which,
unlike the classic Visualizer, has a point size per cloud.)
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import threading
import time

import numpy as np

from .pointcloud import read_ply, read_touchdowns, touchdown_path

POLL_PERIOD = 0.5  # s between file checks
POINT_SIZE = 3.0  # px
TARGET_RADIUS = 0.03  # m, small: it sits inside the bunny
# Points this close to the target (the bunny and the ground at its feet) are
# drawn larger and white, so the target stands out in a big lidar cloud.
HIGHLIGHT_RADIUS = 0.3  # m
HIGHLIGHT_POINT_SIZE = 6.0  # px
HIGHLIGHT_COLOUR = (1.0, 1.0, 1.0)
TOUCHDOWN_RADIUS = 0.08  # m
TOUCHDOWN_COLOUR = (0.2, 1.0, 0.3)
VIEW_FRONT = (-0.55, -0.45, 0.70)  # initial view: oblique, from above
VIEW_FOV = 60.0  # deg
VIEW_DISTANCE = 1.5  # initial camera distance, in cloud bounding-box diagonals
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

    def __init__(self, path: Path, target: Sequence[float] | None = None) -> None:
        self.path = Path(path)
        self.target = None if target is None else np.asarray(target, float)
        self._signature = None
        self._touchdown_signature = None
        self.points = np.empty((0, 3))
        self.colours = np.empty((0, 3))
        self.highlight = np.empty(0, bool)  # within HIGHLIGHT_RADIUS of the target
        self.touchdowns: list[tuple[int, np.ndarray]] = []
        self.jumps = 0

    def poll(self) -> bool:
        """Reload if the files changed; returns True if the cloud or the touchdowns changed."""
        touchdowns_changed = self._poll_touchdowns()
        try:
            stat = self.path.stat()
            signature = (stat.st_mtime_ns, stat.st_size)
        except OSError:
            signature = None
        if signature == self._signature:
            return touchdowns_changed
        cloud = load_cloud(self.path) if signature is not None else None
        if signature is not None and cloud is None:
            return False  # unreadable right now; try again on the next poll
        self._signature = signature
        if cloud is None or not len(cloud):
            self.points, self.colours, self.jumps = np.empty((0, 3)), np.empty((0, 3)), 0
            self.highlight = np.empty(0, bool)
        else:
            self.points = np.column_stack([cloud["x"], cloud["y"], cloud["z"]]).astype(float)
            self.colours = jump_colours(cloud["jump"])
            self.highlight = (
                np.zeros(len(self.points), bool) if self.target is None
                else np.sum((self.points - self.target) ** 2, axis=1) <= HIGHLIGHT_RADIUS**2
            )
            self.jumps = len(np.unique(cloud["jump"]))
        return True

    def _poll_touchdowns(self) -> bool:
        try:
            stat = touchdown_path(self.path).stat()
            signature = (stat.st_mtime_ns, stat.st_size)
        except OSError:
            signature = None
        if signature == self._touchdown_signature:
            return False
        self._touchdown_signature = signature
        self.touchdowns = read_touchdowns(self.path) if signature is not None else []
        return True


def _title(source: CloudFile) -> str:
    return f"Lidar point cloud: {len(source.points)} points, {source.jumps} jump(s)"


def run_window(path: Path, target: Sequence[float] | None = None) -> int:
    import open3d as o3d
    from open3d.visualization import gui, rendering

    source = CloudFile(path, target)
    app = gui.Application.instance
    try:
        app.initialize()
        window = app.create_window("Lidar point cloud", 960, 720)
    except RuntimeError as error:
        print(f"[SCAN] Open3D could not open a window ({error}). The file is still written: {path}")
        return 1
    widget = gui.SceneWidget()
    widget.scene = rendering.Open3DScene(window.renderer)
    widget.scene.set_background((*BACKGROUND, 1.0))
    window.add_child(widget)
    scene = widget.scene

    def material(size: float, shader: str = "defaultUnlit") -> rendering.MaterialRecord:
        record = rendering.MaterialRecord()
        record.shader = shader
        record.point_size = size
        return record

    scene.add_geometry("axes", o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5),
                       material(1.0, "defaultLit"))
    if target is not None:
        marker = o3d.geometry.TriangleMesh.create_sphere(radius=TARGET_RADIUS)
        marker.translate(np.asarray(target, float))
        marker.paint_uniform_color((0.95, 0.15, 0.15))
        marker.compute_vertex_normals()
        scene.add_geometry("target", marker, material(1.0, "defaultLit"))
    sizes = {"points": POINT_SIZE, "highlight": HIGHLIGHT_POINT_SIZE}
    shown: set[str] = set()
    labels: list = []
    displayed = [np.empty((0, 3))]  # points on screen, for refitting the view
    fit_pending = [False]  # a fit asked for before the widget had a size

    def fit_view() -> None:
        points = displayed[0]
        if not len(points):
            return
        if widget.frame.width <= 0 or widget.frame.height <= 0:
            fit_pending[0] = True  # the camera needs the viewport size: fit on layout
            return
        fit_pending[0] = False
        lower, upper = points.min(axis=0), points.max(axis=0)
        centre = 0.5 * (lower + upper)
        diagonal = max(float(np.linalg.norm(upper - lower)), 0.5)
        front = np.array(VIEW_FRONT) / np.linalg.norm(VIEW_FRONT)
        bounds = o3d.geometry.AxisAlignedBoundingBox(lower, upper)
        widget.setup_camera(VIEW_FOV, bounds, centre)
        widget.look_at(centre, centre + front * diagonal * VIEW_DISTANCE, (0.0, 0.0, 1.0))

    def show_cloud(points, colours, highlight, touchdowns, title: str, fit: bool) -> None:
        for name in shown:
            scene.remove_geometry(name)
        shown.clear()
        for label in labels:
            widget.remove_3d_label(label)
        labels.clear()
        groups = (("points", ~highlight), ("highlight", highlight))
        for name, mask in groups:
            if not mask.any():
                continue
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(points[mask])
            cloud.colors = o3d.utility.Vector3dVector(
                np.tile(HIGHLIGHT_COLOUR, (int(mask.sum()), 1)) if name == "highlight"
                else colours[mask]
            )
            scene.add_geometry(name, cloud, material(sizes[name]))
            shown.add(name)
        for jump, position in touchdowns:
            marker = o3d.geometry.TriangleMesh.create_sphere(radius=TOUCHDOWN_RADIUS)
            marker.translate(position)
            marker.paint_uniform_color(TOUCHDOWN_COLOUR)
            marker.compute_vertex_normals()
            name = f"touchdown {jump}"
            scene.add_geometry(name, marker, material(1.0, "defaultLit"))
            shown.add(name)
            labels.append(widget.add_3d_label(position + (0.0, 0.0, 3 * TOUCHDOWN_RADIUS), name))
        displayed[0] = points
        if fit:
            fit_view()
        window.title = title
        window.post_redraw()
        print(f"[SCAN] {title}")

    def on_key(event) -> bool:
        if event.type != gui.KeyEvent.Type.DOWN:
            return False
        if event.key == gui.KeyName.R:
            fit_view()
            return True
        if event.key in (gui.KeyName.Q, gui.KeyName.ESCAPE):
            window.close()
            return True
        return False

    def on_layout(context) -> None:
        widget.frame = window.content_rect
        if fit_pending[0]:
            fit_view()

    window.set_on_key(on_key)
    window.set_on_layout(on_layout)

    def watch() -> None:
        # File polling off the GUI thread; drawing is handed back to it.
        fitted = False
        while True:
            if source.poll():
                fit = len(source.points) > 0 and not fitted
                fitted = fitted or fit
                if not len(source.points):
                    fitted = False  # a new run emptied the file
                # Snapshot: the next poll replaces these while drawing waits.
                snapshot = (source.points, source.colours, source.highlight,
                            list(source.touchdowns), _title(source), fit)
                app.post_to_main_thread(window, lambda snapshot=snapshot: show_cloud(*snapshot))
            time.sleep(POLL_PERIOD)

    print(f"[SCAN] Watching {path}")
    threading.Thread(target=watch, daemon=True).start()
    app.run()
    return 0


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m jumper_sim.cloud_viewer",
        description="Show the lidar point cloud in Open3D and redraw it whenever the file changes.",
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
