"""Convert a terrain mesh to the simulator's collision heightfield.

Input is an OBJ, or a ZIP holding one (default: the moon surface's
``MoonSurface.zip``). The terrain is converted to Z-up and scaled to metres,
then centred on the world origin: x/y on the middle of the terrain, z so the
ground right at the origin (where the robot spawns) is at height 0. The
heightfield (MuJoCo's float32 binary format, or an 8-bit PNG) is what the
simulator uses for contact, the lidar and drawing; the matching MJCF
``size`` and ``pos`` values are printed.
"""

import argparse
from pathlib import Path
import struct
import tempfile
import zipfile
import zlib

import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
import trimesh

BUILD_DIR = Path(__file__).resolve().parent
MOON_SURFACE = BUILD_DIR / "Testing" / "Temporary" / "moon-surface"


def _write_grayscale_png(path: Path, image: np.ndarray) -> None:
    height, width = image.shape
    raw = b"".join(b"\x00" + row.tobytes() for row in image)

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(
            b"IHDR",
            struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0),
        )
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )
    path.write_bytes(png)


def _load_vertices(path: Path) -> np.ndarray:
    """The mesh's vertex positions (an OBJ, or the first OBJ in a ZIP)."""
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive, tempfile.TemporaryDirectory() as tmp:
            names = [n for n in archive.namelist() if n.lower().endswith(".obj")]
            if not names:
                raise RuntimeError(f"No .obj file in {path}")
            return _load_vertices(Path(archive.extract(names[0], tmp)))
    mesh = trimesh.load(path, force="mesh", process=False)
    return np.asarray(mesh.vertices, float)


def _to_z_up(vertices: np.ndarray, up: str) -> np.ndarray:
    if up == "z":
        return vertices
    # Y-up to Z-up: rotate +90 deg about x, (x, y, z) -> (x, -z, y).
    return np.column_stack([vertices[:, 0], -vertices[:, 2], vertices[:, 1]])


def _tile_heightfield(heights: np.ndarray, tiles: int) -> np.ndarray:
    """Repeat a heightfield so the copies sit one tile period apart.

    The first and last rows/columns of a tile lie on its edges, so adjacent
    copies share one sample and the grid spacing stays the same everywhere.
    """
    rows, cols = heights.shape
    row_index = np.arange(tiles * (rows - 1) + 1) % (rows - 1)
    col_index = np.arange(tiles * (cols - 1) + 1) % (cols - 1)
    row_index[-1], col_index[-1] = rows - 1, cols - 1
    return heights[np.ix_(row_index, col_index)]


def _sample_heights(vertices: np.ndarray, resolution: int) -> np.ndarray:
    """Heights on a regular grid over the mesh bounds; rows advance in +Y."""
    if resolution < 2:
        raise ValueError("Heightfield resolution must be at least 2")

    xy = vertices[:, :2]
    unique_xy, unique_indices = np.unique(xy, axis=0, return_index=True)
    heights = vertices[unique_indices, 2]
    x_coordinates = np.linspace(xy[:, 0].min(), xy[:, 0].max(), resolution)
    y_coordinates = np.linspace(xy[:, 1].min(), xy[:, 1].max(), resolution)
    grid_x, grid_y = np.meshgrid(x_coordinates, y_coordinates)

    linear = LinearNDInterpolator(unique_xy, heights, fill_value=np.nan)
    sampled_heights = linear(grid_x, grid_y)
    missing = np.isnan(sampled_heights)
    if np.any(missing):
        nearest = NearestNDInterpolator(unique_xy, heights)
        sampled_heights[missing] = nearest(grid_x[missing], grid_y[missing])
    return sampled_heights


def _heightfield_image(heights: np.ndarray) -> np.ndarray:
    min_height = float(heights.min())
    height_range = float(heights.max() - min_height)
    normalized = np.clip((heights - min_height) / height_range, 0.0, 1.0)
    # MuJoCo reads PNG rows top-to-bottom while heightfield rows advance in +Y.
    return np.rint(normalized[::-1] * 255.0).astype(np.uint8)


def _write_heightfield(path: Path, heights: np.ndarray) -> None:
    """PNG: 8-bit, heights in 1/255 steps of the range. Any other suffix:
    MuJoCo's binary format (int32 rows, int32 cols, float32 heights, rows in
    +Y), which keeps full precision; MuJoCo normalises it to [0, 1] itself.

    8 bits over this terrain's 5 m relief are ~2 cm steps on 25 cm cells,
    which tilt the ground facets by up to ~4 deg and gave the jumping foot
    an off-axis push (take-off spin up to 7 rad/s).
    """
    if path.suffix.lower() == ".png":
        _write_grayscale_png(path, _heightfield_image(heights))
        return
    rows, cols = heights.shape
    path.write_bytes(
        struct.pack("<ii", rows, cols) + heights.astype("<f4").tobytes()
    )


def convert(
    input_path: Path,
    heightfield_path: Path,
    heightfield_resolution: int,
    tiles: int = 1,
    scale: float = 1.0,
    up: str = "z",
) -> None:
    if tiles < 1 or tiles % 2 == 0:
        raise ValueError("--tiles must be an odd number >= 1")
    if not input_path.is_file():
        raise FileNotFoundError(f"Terrain input file not found: {input_path}")

    vertices = _to_z_up(_load_vertices(input_path), up) * scale
    lower, upper = vertices.min(axis=0), vertices.max(axis=0)
    vertices[:, :2] -= 0.5 * (lower[:2] + upper[:2])
    heights = _sample_heights(vertices, heightfield_resolution)
    middle = (heightfield_resolution - 1) // 2
    heights -= float(heights[middle, middle])  # the ground at the origin: z = 0
    heights = _tile_heightfield(heights, tiles)  # tiles touch edge to edge

    heightfield_path.parent.mkdir(parents=True, exist_ok=True)
    _write_heightfield(heightfield_path, heights)
    rows, cols = heights.shape
    print(f"Exported {cols}x{rows} collision heightfield ({tiles}x{tiles} tiles) "
          f"to {heightfield_path}")

    half = 0.5 * tiles * (upper - lower)  # the tiles are centred on the original
    print(
        "MJCF values:\n"
        f'  <hfield ... size="{half[0]:.4f} {half[1]:.4f} '
        f'{float(heights.max() - heights.min()):.4f} 0.1"/>\n'
        f'  <geom name="terrain_collision" ... pos="0.0000 0.0000 '
        f'{float(heights.min()):.4f}"/>'
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a terrain mesh (OBJ, or a ZIP with an OBJ) to a MuJoCo heightfield."
    )
    parser.add_argument(
        "--input", type=Path, default=MOON_SURFACE / "source" / "MoonSurface.zip"
    )
    parser.add_argument(
        "--heightfield",
        type=Path,
        default=BUILD_DIR / "terrain_height.bin",
        help="collision heightfield: .bin (float32, default) or .png (8-bit)",
    )
    parser.add_argument(
        "--resolution", type=int, default=401,
        help="heightfield samples per side of one tile",
    )
    parser.add_argument(
        "--tiles", type=int, default=1,
        help="odd number of terrain copies per side, centred on the original",
    )
    parser.add_argument(
        "--scale", type=float, default=0.01,
        help="metres per input unit (moon surface: centimetres)",
    )
    parser.add_argument(
        "--up", choices=("y", "z"), default="y", help="up axis of the input"
    )
    args = parser.parse_args()
    convert(
        args.input.expanduser().resolve(),
        args.heightfield.expanduser().resolve(),
        args.resolution,
        args.tiles,
        args.scale,
        args.up,
    )


if __name__ == "__main__":
    main()
