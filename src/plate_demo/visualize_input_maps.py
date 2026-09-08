from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CHANNEL_NAMES = [
    "material",
    "left_bc",
    "right_bc",
    "edge_load",
    "thickness",
    "x_coord",
    "y_coord",
    "width",
    "height",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize U-Net input channels and target stress map")
    parser.add_argument("--dataset", default="data/map_dataset.npz", help="map dataset path")
    parser.add_argument("--case-index", type=int, default=0, help="case index to visualize")
    parser.add_argument(
        "--output",
        default="outputs/map_prediction/plots/input_output_case00.png",
        help="output png path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = np.load(Path(args.dataset))
    inputs = data["inputs"]
    targets = data["targets"]
    case_ids = data["case_ids"]

    case_index = max(0, min(args.case_index, len(inputs) - 1))
    sample = inputs[case_index]
    target = targets[case_index, 0]
    case_id = str(case_ids[case_index])

    fig = plt.figure(figsize=(12, 10.5), constrained_layout=True)
    grid = fig.add_gridspec(4, 4, width_ratios=[0.55, 1.0, 1.0, 1.0])
    cmap_scalar = plt.colormaps["viridis"]

    input_label_ax = fig.add_subplot(grid[0:3, 0])
    input_label_ax.text(0.0, 0.5, "Input\n(9 channels)", va="center", fontsize=13, weight="bold")
    input_label_ax.axis("off")

    output_label_ax = fig.add_subplot(grid[3, 0])
    output_label_ax.text(0.0, 0.5, "Output\n(1 channel)", va="center", fontsize=13, weight="bold")
    output_label_ax.axis("off")

    for idx, name in enumerate(CHANNEL_NAMES):
        ax = fig.add_subplot(grid[idx // 3, (idx % 3) + 1])
        channel = sample[idx]
        im = ax.imshow(channel, cmap=cmap_scalar, vmin=0.0, vmax=1.0)
        ax.set_title(name)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    target_ax = fig.add_subplot(grid[3, 2])
    stress_cmap = plt.colormaps["viridis"].copy()
    stress_cmap.set_bad(color="white")
    target_display = np.where(sample[0] > 0.5, target, np.nan)
    target_max = float(np.nanmax(target_display))
    target_image = target_ax.imshow(
        target_display,
        cmap=stress_cmap,
        vmin=0.0,
        vmax=target_max if target_max > 0.0 else 1.0,
    )
    target_ax.set_title("true_stress")
    target_ax.axis("off")
    fig.colorbar(target_image, ax=target_ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"U-Net input and output: {case_id}", fontsize=15)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
