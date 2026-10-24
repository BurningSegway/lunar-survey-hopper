"""Convert a terrain mesh to MuJoCo assets: a visual OBJ, a collision heightfield
(float32 binary) and, optionally (--texture), a PNG texture for the visual mesh.

Input is a USD stage (first mesh prim), an OBJ, or a ZIP holding an OBJ. The
terrain is converted to Z-up and scaled to metres, then centred on the world
origin: x/y on the middle of the terrain, z so the ground right at the origin
(where the robot spawns) is at height 0. The visual mesh therefore goes into
the MJCF at pos="0 0 0"; the matching heightfield values are printed.
"""

import argparse
from collections import Counter
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


def _load_usd(path: Path) -> tuple[np.ndarray, np.ndarray, None]:
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise RuntimeError(f"Could not open USD stage: {path}")
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        usd_mesh = UsdGeom.Mesh(prim)
        vertices = usd_mesh.GetPointsAttr().Get()
        faces = np.asarray(usd_mesh.GetFaceVertexIndicesAttr().Get())
        counts = usd_mesh.GetFaceVertexCountsAttr().Get()
        if vertices is None or not len(vertices) or not faces.size:
            raise RuntimeError(f"USD mesh contains no geometry: {prim.GetPath()}")
        if any(count != 3 for count in counts):
            raise RuntimeError(
                f"USD mesh is not triangulated; cannot export {prim.GetPath()} to MuJoCo OBJ"
            )
        if faces.size != len(counts) * 3:
            raise RuntimeError(f"USD mesh has inconsistent face indices: {prim.GetPath()}")
        return np.asarray(vertices, float), faces.reshape((-1, 3)), None
    raise RuntimeError(f"No USD mesh found in stage: {path}")


def _load_obj(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Vertices, triangles and per-vertex UVs (None if the OBJ has none)."""
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive, tempfile.TemporaryDirectory() as tmp:
            names = [n for n in archive.namelist() if n.lower().endswith(".obj")]
            if not names:
                raise RuntimeError(f"No .obj file in {path}")
            return _load_obj(Path(archive.extract(names[0], tmp)))
    # process=False keeps a separate vertex per (position, uv) pair.
    mesh = trimesh.load(path, force="mesh", process=False)
    uv = getattr(mesh.visual, "uv", None)
    return np.asarray(mesh.vertices, float), np.asarray(mesh.faces), (
        None if uv is None else np.asarray(uv, float)
    )


def _load(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    if path.suffix.lower() in (".usd", ".usda", ".usdc"):
        return _load_usd(path)
    return _load_obj(path)


def _to_z_up(vertices: np.ndarray, up: str) -> np.ndarray:
    if up == "z":
        return vertices
    # Y-up to Z-up: rotate +90 deg about x, (x, y, z) -> (x, -z, y).
    return np.column_stack([vertices[:, 0], -vertices[:, 2], vertices[:, 1]])


def _ground_face_mask(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Faces of the largest connected surface (the ground).

    Terrains can carry loose rock meshes resting on and partly buried in the
    ground. A heightfield cannot represent them (their buried undersides
    corrupted the sampled heights), so they had no real collision; drop them.
    """
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.merge_vertices()  # USD triangle soups share no vertices; keeps face order
    labels = trimesh.graph.connected_component_labels(
        mesh.face_adjacency, node_count=len(mesh.faces)
    )
    counts = Counter(labels.tolist())
    ground_label, ground_faces = counts.most_common(1)[0]
    if len(counts) > 1:
        print(
            f"Kept the ground surface ({ground_faces} faces); removed "
            f"{len(counts) - 1} other objects ({len(faces) - ground_faces} faces)"
        )
    return labels == ground_label


def _keep_faces(vertices, faces, uv, mask):
    faces = faces[mask]
    used, faces = np.unique(faces, return_inverse=True)
    return vertices[used], faces.reshape((-1, 3)), None if uv is None else uv[used]


def _tile_mesh(vertices, faces, uv, tiles: int, period: np.ndarray):
    """Copies in a ``tiles`` x ``tiles`` grid, centred on the original."""
    half = tiles // 2
    offsets = [
        (i * period[0], j * period[1], 0.0)
        for j in range(-half, half + 1)
        for i in range(-half, half + 1)
    ]
    return (
        np.concatenate([vertices + offset for offset in offsets]),
        np.concatenate([faces + k * len(vertices) for k in range(len(offsets))]),
        None if uv is None else np.tile(uv, (len(offsets), 1)),
    )


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


def _write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray, uv: np.ndarray | None) -> None:
    with path.open("w") as obj:
        np.savetxt(obj, vertices, fmt="v %.5f %.5f %.5f")
        if uv is None:
            np.savetxt(obj, faces + 1, fmt="f %d %d %d")
        else:
            np.savetxt(obj, uv, fmt="vt %.6f %.6f")
            np.savetxt(obj, np.repeat(faces + 1, 2, axis=1), fmt="f %d/%d %d/%d %d/%d")


def _write_texture(source: Path, output: Path) -> None:
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None  # the moon surface texture is 8192x8192
    with Image.open(source) as image:
        image.convert("RGB").save(output, compress_level=3)


def convert(
    input_path: Path,
    output_path: Path,
    heightfield_path: Path,
    heightfield_resolution: int,
    tiles: int = 1,
    scale: float = 1.0,
    up: str = "z",
    texture_path: Path | None = None,
    texture_output_path: Path | None = None,
) -> None:
    if tiles < 1 or tiles % 2 == 0:
        raise ValueError("--tiles must be an odd number >= 1")
    if not input_path.is_file():
        raise FileNotFoundError(f"Terrain input file not found: {input_path}")

    vertices, faces, uv = _load(input_path)
    vertices = _to_z_up(vertices, up) * scale
    vertices, faces, uv = _keep_faces(
        vertices, faces, uv, _ground_face_mask(vertices, faces)
    )

    lower, upper = vertices.min(axis=0), vertices.max(axis=0)
    vertices[:, :2] -= 0.5 * (lower[:2] + upper[:2])
    heights = _sample_heights(vertices, heightfield_resolution)
    middle = (heightfield_resolution - 1) // 2
    ground_at_origin = float(heights[middle, middle])
    vertices[:, 2] -= ground_at_origin
    heights -= ground_at_origin

    period = upper - lower  # tiles touch edge to edge
    heights = _tile_heightfield(heights, tiles)
    vertices, faces, uv = _tile_mesh(vertices, faces, uv, tiles, period)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    heightfield_path.parent.mkdir(parents=True, exist_ok=True)
    _write_obj(output_path, vertices, faces, uv)
    _write_heightfield(heightfield_path, heights)
    print(
        f"Exported {len(vertices)} vertices and {len(faces)} faces "
        f"({tiles}x{tiles} tiles{', with UVs' if uv is not None else ''}) to {output_path}"
    )
    rows, cols = heights.shape
    print(f"Exported {cols}x{rows} collision heightfield to {heightfield_path}")
    if texture_path is not None and texture_output_path is not None:
        _write_texture(texture_path, texture_output_path)
        print(f"Exported texture to {texture_output_path}")

    lower, upper = vertices.min(axis=0), vertices.max(axis=0)
    centre = 0.5 * (lower + upper)
    half = 0.5 * (upper - lower)
    print(
        'MJCF values (terrain_visual at pos="0 0 0"):\n'
        f'  <hfield ... size="{half[0]:.4f} {half[1]:.4f} '
        f'{float(heights.max() - heights.min()):.4f} 0.1"/>\n'
        f'  <geom name="terrain_collision" ... pos="{centre[0]:.4f} {centre[1]:.4f} '
        f'{float(heights.min()):.4f}"/>'
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a terrain mesh (USD, OBJ or ZIP with an OBJ) to MuJoCo assets."
    )
    parser.add_argument(
        "--input", type=Path, default=MOON_SURFACE / "source" / "MoonSurface.zip"
    )
    parser.add_argument("--output", type=Path, default=BUILD_DIR / "terrain.obj")
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
    parser.add_argument(
        "--texture", type=Path, nargs="?", const=MOON_SURFACE / "textures" / "Diffuse.jpeg",
        help="also write a PNG texture for the visual mesh from this colour image "
        "(default with no path: the moon surface's Diffuse.jpeg; needs UVs in the "
        "input); off by default",
    )
    parser.add_argument(
        "--texture-output", type=Path, default=BUILD_DIR / "terrain_diffuse.png"
    )
    args = parser.parse_args()
    convert(
        args.input.expanduser().resolve(),
        args.output.expanduser().resolve(),
        args.heightfield.expanduser().resolve(),
        args.resolution,
        args.tiles,
        args.scale,
        args.up,
        None if args.texture is None else args.texture.expanduser().resolve(),
        args.texture_output.expanduser().resolve(),
    )


if __name__ == "__main__":
    main()
