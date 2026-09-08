from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path
import subprocess

import numpy as np

from .inp_builder import PlateCase, build_plate_input


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate plate dataset with CalculiX in Docker")
    parser.add_argument("--cases", type=int, default=500, help="number of analysis cases")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--image", default="plate-stress-calculix", help="docker image name")
    parser.add_argument("--workdir", default="data/runs", help="analysis run directory")
    parser.add_argument("--cases-out", default="data/cases.csv", help="case metadata csv path")
    parser.add_argument("--force", action="store_true", help="overwrite existing run outputs")
    return parser.parse_args()


def latin_hypercube(count: int, dimensions: int, seed: int) -> np.ndarray:
    """ NOTE：⑤  """
    if count <= 0:
        raise ValueError("count must be positive")
    if dimensions <= 0:
        raise ValueError("dimensions must be positive")

    rng = np.random.default_rng(seed)
    unit = (np.arange(count, dtype=float)[:, None] + rng.random((count, dimensions))) / count
    for column in range(dimensions):
        rng.shuffle(unit[:, column])
    """ END """
    return unit


def sample_cases(count: int, seed: int) -> list[PlateCase]:
    # The hole-related columns are transformed to valid, case-dependent
    # ranges after width, height, and radius are known.
    unit = latin_hypercube(count=count, dimensions=7, seed=seed)

    samples = []
    for index in range(count):

        """ NOTE：⑥  """
        width = 120.0 + unit[index, 0] * (180.0 - 120.0)
        height = 60.0 + unit[index, 1] * (100.0 - 60.0)
        max_radius = min(width, height) * 0.18
        hole_radius = 8.0 + unit[index, 2] * (max_radius - 8.0)

        margin_x = max(12.0, hole_radius + width * 0.12)
        margin_y = max(10.0, hole_radius + height * 0.15)
        hole_center_x = margin_x + unit[index, 3] * (width - 2.0 * margin_x)
        hole_center_y = margin_y + unit[index, 4] * (height - 2.0 * margin_y)
        thickness = 4.0 + unit[index, 5] * (12.0 - 4.0)
        edge_load = 400.0 + unit[index, 6] * (1400.0 - 400.0)
        samples.append(
            PlateCase(
                case_id=f"plate_{index:03d}",
                width=float(width),
                height=float(height),
                hole_radius=float(hole_radius),
                hole_center_x=float(hole_center_x),
                hole_center_y=float(hole_center_y),
                thickness=float(thickness),
                edge_load=float(edge_load),
            )
        )
        """ END """
    return samples


def run_case(case: PlateCase, image: str, run_root: Path, force: bool) -> None:
    case_dir = run_root / case.case_id
    if case_dir.exists() and force:
        for child in case_dir.iterdir():
            if child.is_file():
                child.unlink()
    case_dir.mkdir(parents=True, exist_ok=True)

    inp_path = build_plate_input(case, case_dir)
    dat_path = case_dir / f"{case.case_id}.dat"

    if not dat_path.exists() or force:

        """ NOTE：⑦  """
        cmd = [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{case_dir.resolve()}:/work",
            image,
            "-i",
            case.case_id,
        ]
        completed = subprocess.run(cmd, cwd=case_dir, capture_output=True, text=True)
        """ END """
        if completed.returncode != 0:
            raise RuntimeError(
                f"CalculiX failed for {case.case_id}\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )

def write_cases(case_path: Path, cases: list[PlateCase]) -> None:
    case_path.parent.mkdir(parents=True, exist_ok=True)
    records = [asdict(case) for case in cases]
    with case_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def main() -> None:

    """ NOTE：⑧  """
    args = parse_args()
    run_root = Path(args.workdir)
    case_path = Path(args.cases_out)
    cases = sample_cases(args.cases, args.seed)
    for case in cases:
        run_case(case, args.image, run_root, args.force)
        print(f"finished {case.case_id}")

    write_cases(case_path, cases)
    print(f"wrote {case_path}")
    """ END """


if __name__ == "__main__":
    main()
