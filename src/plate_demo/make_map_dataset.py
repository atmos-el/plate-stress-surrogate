from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import re

import numpy as np
import pandas as pd


NODE_RE = re.compile(r"^\s*(\d+)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)")
STRESS_RE = re.compile(
    r"^\s*(\d+)\s+(\d+)\s+([+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)\s+([+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)\s+([+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)\s+([+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)"
)


@dataclass(frozen=True)
class RasterConfig:
    width: int = 64
    height: int = 64


def load_scalar_dataset(dataset_path: Path) -> pd.DataFrame:
    return pd.read_csv(dataset_path)


def build_case_maps(case_row: pd.Series, run_dir: Path, raster: RasterConfig) -> tuple[np.ndarray, np.ndarray]:
    inp_path = run_dir / f"{case_row['case_id']}.inp"
    dat_path = run_dir / f"{case_row['case_id']}.dat"

    nodes, elements = parse_inp(inp_path)
    stress_by_element = parse_element_stress(dat_path)
    stress_by_node = build_nodal_stress(nodes, elements, stress_by_element)

    image = np.zeros((9, raster.height, raster.width), dtype=np.float32)
    target = np.zeros((1, raster.height, raster.width), dtype=np.float32)

    """ NOTE：②  """
    width = float(case_row["width"])
    height = float(case_row["height"])
    edge_load = float(case_row["edge_load"])
    thickness = float(case_row["thickness"])
    hole_cx = float(case_row["hole_center_x"])
    hole_cy = float(case_row["hole_center_y"])
    hole_radius = float(case_row["hole_radius"])
    # Use the same pixel-centre coordinates for masks and field rasterization.
    x_coords = width * ((np.arange(raster.width, dtype=np.float32) + 0.5) / raster.width)
    y_coords = height * (1.0 - ((np.arange(raster.height, dtype=np.float32) + 0.5) / raster.height))
    grid_x, grid_y = np.meshgrid(x_coords, y_coords)

    hole_distance = np.sqrt((grid_x - hole_cx) ** 2 + (grid_y - hole_cy) ** 2)
    material = (hole_distance > hole_radius).astype(np.float32)
    image[0] = material
    left_band = (grid_x <= (2.0 * width / raster.width)).astype(np.float32)
    right_band = (grid_x >= (width - 2.0 * width / raster.width)).astype(np.float32)
    image[1] = left_band * material
    image[2] = right_band * material
    image[3, :, :] = edge_load / 1400.0
    image[4, :, :] = thickness / 12.0
    image[5, :, :] = grid_x / width
    image[6, :, :] = grid_y / height
    image[7, :, :] = width / 180.0
    image[8, :, :] = height / 100.0
    for _, conn in elements.items():
        coords = np.array([nodes[nid] for nid in conn], dtype=np.float32)
        node_values = np.array([stress_by_node[nid] for nid in conn], dtype=np.float32)
        rasterize_triangle_field(target[0], coords, node_values, width, height, raster)
    target[0] *= material
    """ END """

    return image, target


def build_nodal_stress(
    nodes: dict[int, tuple[float, float]],
    elements: dict[int, tuple[int, ...]],
    stress_by_element: dict[int, float],
) -> dict[int, float]:
    accum: dict[int, list[float]] = {nid: [] for nid in nodes}
    for eid, conn in elements.items():
        value = stress_by_element.get(eid, 0.0)
        for nid in conn:
            accum[nid].append(value)
    return {nid: float(np.mean(values)) if values else 0.0 for nid, values in accum.items()}


def parse_inp(inp_path: Path) -> tuple[dict[int, tuple[float, float]], dict[int, tuple[int, ...]]]:
    nodes: dict[int, tuple[float, float]] = {}
    elements: dict[int, tuple[int, ...]] = {}
    mode = None

    """ NOTE：③  """
    for raw in inp_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("*"):
            upper = line.upper()
            if upper.startswith("*NODE"):
                mode = "node"
            elif upper.startswith("*ELEMENT"):
                mode = "element"
            else:
                mode = None
            continue
        if mode == "node":
            match = NODE_RE.match(line)
            if match:
                nid = int(match.group(1))
                nodes[nid] = (float(match.group(2)), float(match.group(3)))
        elif mode == "element":
            parts = [part.strip() for part in line.split(",") if part.strip()]
            if len(parts) >= 4:
                eid = int(parts[0])
                elements[eid] = tuple(int(part) for part in parts[1:])
    """ END """
    return nodes, elements


def parse_element_stress(dat_path: Path) -> dict[int, float]:
    stress_accum: dict[int, list[float]] = {}
    in_stress = False

    """ NOTE：④  """
    for raw in dat_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        upper = raw.upper()
        if "STRESSES" in upper:
            in_stress = True
            continue
        if in_stress and not raw.strip():
            continue
        if in_stress:
            match = STRESS_RE.match(raw)
            if not match:
                if raw.strip():
                    in_stress = False
                continue
            eid = int(match.group(1))
            sxx = float(match.group(3))
            syy = float(match.group(4))
            sxy = float(match.group(6))
            mises = math.sqrt(max(sxx * sxx - sxx * syy + syy * syy + 3.0 * sxy * sxy, 0.0))
            stress_accum.setdefault(eid, []).append(mises)
    """ END """
    return {eid: float(np.mean(values)) for eid, values in stress_accum.items()}


def to_pixel_bounds(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    width: float,
    height: float,
    raster: RasterConfig,
) -> tuple[int, int, int, int]:
    px0 = max(0, min(raster.width - 1, int(math.floor((x_min / width) * raster.width))))
    px1 = max(px0 + 1, min(raster.width, int(math.ceil((x_max / width) * raster.width))))
    py0f = (1.0 - (y_max / height)) * raster.height
    py1f = (1.0 - (y_min / height)) * raster.height
    py0 = max(0, min(raster.height - 1, int(math.floor(py0f))))
    py1 = max(py0 + 1, min(raster.height, int(math.ceil(py1f))))
    return px0, px1, py0, py1


def rasterize_triangle_field(
    canvas: np.ndarray,
    coords: np.ndarray,
    node_values: np.ndarray,
    width: float,
    height: float,
    raster: RasterConfig,
) -> None:
    x_min = float(coords[:, 0].min())
    x_max = float(coords[:, 0].max())
    y_min = float(coords[:, 1].min())
    y_max = float(coords[:, 1].max())
    px0, px1, py0, py1 = to_pixel_bounds(x_min, x_max, y_min, y_max, width, height, raster)

    for py in range(py0, py1):
        y = height * (1.0 - ((py + 0.5) / raster.height))
        for px in range(px0, px1):
            x = width * ((px + 0.5) / raster.width)
            bary = barycentric_coordinates(x, y, coords)
            if bary is None:
                continue
            value = float(np.dot(bary, node_values))
            canvas[py, px] = max(canvas[py, px], value)


def barycentric_coordinates(x: float, y: float, triangle: np.ndarray) -> np.ndarray | None:
    ax, ay = float(triangle[0, 0]), float(triangle[0, 1])
    bx, by = float(triangle[1, 0]), float(triangle[1, 1])
    cx, cy = float(triangle[2, 0]), float(triangle[2, 1])
    det = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
    if abs(det) < 1e-12:
        return None
    l1 = ((by - cy) * (x - cx) + (cx - bx) * (y - cy)) / det
    l2 = ((cy - ay) * (x - cx) + (ax - cx) * (y - cy)) / det
    l3 = 1.0 - l1 - l2
    eps = -1e-6
    if l1 < eps or l2 < eps or l3 < eps:
        return None
    return np.array([l1, l2, l3], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build rasterized stress-map dataset for U-Net")
    parser.add_argument("--cases", default="data/cases.csv", help="case metadata path")
    parser.add_argument("--run-root", default="data/runs", help="CalculiX run directory")
    parser.add_argument("--output", default="data/map_dataset.npz", help="output npz path")
    parser.add_argument("--image-size", type=int, default=64, help="square raster image size")
    parser.add_argument("--seed", type=int, default=7, help="dataset split seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = load_scalar_dataset(Path(args.cases))
    run_root = Path(args.run_root)
    raster = RasterConfig(width=args.image_size, height=args.image_size)

    """ NOTE：⑤  """
    images = []
    targets = []
    case_ids = []
    for _, row in frame.iterrows():
        case_id = str(row["case_id"])
        image, target = build_case_maps(row, run_root / case_id, raster)
        images.append(image)
        targets.append(target)
        case_ids.append(case_id)

    inputs_np = np.stack(images)
    targets_np = np.stack(targets)
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(case_ids))
    train_end = int(len(order) * 0.70)
    validation_end = int(len(order) * 0.85)
    splits = np.full(len(order), "test", dtype="<U10")
    splits[order[:train_end]] = "train"
    splits[order[train_end:validation_end]] = "validation"
    train_max = float(targets_np[splits == "train"].max())
    stress_scale = train_max if train_max > 0.0 else 1.0
    targets_np = targets_np / stress_scale

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        inputs=inputs_np,
        targets=targets_np,
        case_ids=np.array(case_ids),
        splits=splits,
        stress_scale=np.array([stress_scale], dtype=np.float32),
    )
    """ END """
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
