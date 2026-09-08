from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re


NODE_RE = re.compile(r"^\s*(\d+)\s+([+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)\s+([+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)")
ELEM_RE = re.compile(
    r"^\s*(\d+)\s+(\d+)\s+([+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)\s+([+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)\s+([+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)\s+([+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse CalculiX result files into a scalar dataset")
    parser.add_argument("--cases", default="data/cases.csv", help="case metadata csv path")
    parser.add_argument("--run-root", default="data/runs", help="CalculiX run directory")
    parser.add_argument("--output", default="data/dataset.csv", help="output dataset csv path")
    return parser.parse_args()


def parse_dat_file(dat_path: Path) -> dict[str, float]:
    text = dat_path.read_text(encoding="utf-8", errors="ignore").splitlines()

    max_disp = 0.0
    max_mises = 0.0
    node_section = False
    elem_section = False

    for line in text:
        upper = line.upper()
        if "DISPLACEMENTS" in upper:
            node_section = True
            elem_section = False
            continue
        if "STRESSES" in upper:
            elem_section = True
            node_section = False
            continue
        if not line.strip():
            continue


        """ NOTE：①  """
        if node_section:
            match = NODE_RE.match(line)
            if match:
                ux = float(match.group(2))
                uy = float(match.group(3))
                disp = (ux * ux + uy * uy) ** 0.5
                max_disp = max(max_disp, disp)
        elif elem_section:
            match = ELEM_RE.match(line)
            if match:
                sxx = float(match.group(3))
                syy = float(match.group(4))
                sxy = float(match.group(6))
                mises = (sxx * sxx - sxx * syy + syy * syy + 3.0 * sxy * sxy) ** 0.5
                max_mises = max(max_mises, mises)
        """ END """

    if max_disp == 0.0 and max_mises == 0.0:
        raise ValueError(f"failed to parse result file: {dat_path}")

    return {
        "max_displacement": max_disp,
        "max_von_mises": max_mises,
    }


def load_cases(case_path: Path) -> list[dict[str, str]]:
    with case_path.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def build_scalar_dataset(case_path: Path, run_root: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for row in load_cases(case_path):
        case_id = row["case_id"]
        dat_path = run_root / case_id / f"{case_id}.dat"
        record: dict[str, object] = dict(row)
        record.update(parse_dat_file(dat_path))
        records.append(record)
    return records


def write_dataset(output_path: Path, records: list[dict[str, object]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    args = parse_args()
    records = build_scalar_dataset(Path(args.cases), Path(args.run_root))
    write_dataset(Path(args.output), records)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
