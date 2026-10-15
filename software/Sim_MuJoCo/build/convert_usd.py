import argparse
from pathlib import Path
import struct
import zlib

import numpy as np
from pxr import Usd, UsdGeom
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
import trimesh


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


def _make_heightfield_image(vertices: np.ndarray, resolution: int) -> np.ndarray:
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

    min_height = float(vertices[:, 2].min())
    height_range = float(vertices[:, 2].max() - min_height)
    normalized = np.clip((sampled_heights - min_height) / height_range, 0.0, 1.0)
    # MuJoCo reads PNG rows top-to-bottom while heightfield rows advance in +Y.
    return np.rint(normalized[::-1] * 255.0).astype(np.uint8)


def convert(
    input_path: Path,
    output_path: Path,
    heightfield_path: Path,
    heightfield_resolution: int,
) -> None:
    if not input_path.is_file():
        raise FileNotFoundError(f"USD input file not found: {input_path}")

    stage = Usd.Stage.Open(str(input_path))
    if stage is None:
        raise RuntimeError(f"Could not open USD stage: {input_path}")

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

        mesh = trimesh.Trimesh(
            vertices=np.asarray(vertices),
            faces=faces.reshape((-1, 3)),
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        heightfield_path.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(output_path)
        heightfield_image = _make_heightfield_image(
            np.asarray(mesh.vertices), heightfield_resolution
        )
        _write_grayscale_png(heightfield_path, heightfield_image)
        print(
            f"Exported {len(mesh.vertices)} vertices and {len(mesh.faces)} "
            f"faces to {output_path}"
        )
        print(
            f"Exported {heightfield_resolution}x{heightfield_resolution} "
            f"collision heightfield to {heightfield_path}"
        )
        return

    raise RuntimeError(f"No USD mesh found in stage: {input_path}")


def main() -> None:
    build_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Convert a USD terrain mesh to OBJ.")
    parser.add_argument("--input", type=Path, default=build_dir / "terrain_merged.usd")
    parser.add_argument("--output", type=Path, default=build_dir / "terrain.obj")
    parser.add_argument(
        "--heightfield",
        type=Path,
        default=build_dir / "terrain_height.png",
    )
    parser.add_argument("--resolution", type=int, default=257)
    args = parser.parse_args()
    convert(
        args.input.expanduser().resolve(),
        args.output.expanduser().resolve(),
        args.heightfield.expanduser().resolve(),
        args.resolution,
    )


if __name__ == "__main__":
    main()
