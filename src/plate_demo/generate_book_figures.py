from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import gmsh

from plate_demo.inp_builder import PlateCase


OUTPUT_DIR = Path(__file__).resolve().parents[4] / "book" / "figures"


def save_plate_problem() -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 120.0
    height = 80.0
    hole_cx = 52.0
    hole_cy = 42.0
    hole_r = 14.0

    ax.add_patch(patches.Rectangle((0, 0), width, height, facecolor="#dbeafe", edgecolor="#1f2937", lw=2))
    ax.add_patch(patches.Circle((hole_cx, hole_cy), hole_r, facecolor="white", edgecolor="#1f2937", lw=2))

    for yy in np.linspace(8, height - 8, 8):
        ax.plot([0, -6], [yy, yy], color="#dc2626", lw=1.8)
    for yy in np.linspace(12, height - 12, 6):
        ax.arrow(width, yy, 10, 0, width=0.5, head_width=3.2, head_length=4.5, color="#2563eb", length_includes_head=True)

    ax.annotate("", xy=(0, -10), xytext=(width, -10), arrowprops=dict(arrowstyle="<->", lw=1.5))
    ax.text(width / 2, -18, "width", ha="center", va="top", fontsize=11)
    ax.annotate("", xy=(-10, 0), xytext=(-10, height), arrowprops=dict(arrowstyle="<->", lw=1.5))
    ax.text(-18, height / 2, "height", ha="right", va="center", rotation=90, fontsize=11)
    ax.annotate("", xy=(hole_cx, hole_cy), xytext=(hole_cx + hole_r, hole_cy), arrowprops=dict(arrowstyle="<->", lw=1.5))
    ax.text(hole_cx + hole_r + 4, hole_cy - 1.5, "hole_radius", ha="left", va="center", fontsize=10)
    ax.plot(hole_cx, hole_cy, "ko", ms=3)
    ax.annotate(
        "(hole_center_x, hole_center_y)",
        xy=(hole_cx, hole_cy),
        xytext=(hole_cx - 6, hole_cy + 24),
        textcoords="data",
        fontsize=10,
        arrowprops=dict(arrowstyle="->", lw=1.2),
    )
    ax.text(4, height * 0.75, "fixed wall", color="#dc2626", ha="left", va="center", fontsize=11)
    ax.text(width + 16, height / 2, "edge_load", color="#2563eb", ha="left", va="center", fontsize=11)

    ax.set_xlim(-24, width + 34)
    ax.set_ylim(-22, height + 8)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "chapter02_plate_problem.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def draw_case(ax: plt.Axes, case: PlateCase, load_min: float, load_max: float) -> None:
    ax.add_patch(patches.Rectangle((0, 0), case.width, case.height, facecolor="#dbeafe", edgecolor="#1f2937", lw=1.3))
    ax.add_patch(patches.Circle((case.hole_center_x, case.hole_center_y), case.hole_radius, facecolor="white", edgecolor="#1f2937", lw=1.3))

    load_ratio = (case.edge_load - load_min) / (load_max - load_min)
    arrow_length = case.width * (0.06 + 0.10 * load_ratio)
    for yy in np.linspace(case.height * 0.18, case.height * 0.82, 5):
        ax.annotate(
            "",
            xy=(case.width + arrow_length, yy),
            xytext=(case.width, yy),
            arrowprops=dict(arrowstyle="-|>", color="#2563eb", lw=1.3, mutation_scale=9),
        )

    ax.set_xlim(-2, case.width * 1.19)
    ax.set_ylim(-2, case.height + 2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        f"W={case.width:.0f}, H={case.height:.0f}\n"
        f"R={case.hole_radius:.1f}, hole_x={case.hole_center_x:.1f}, hole_y={case.hole_center_y:.1f}\n"
        f"t={case.thickness:.1f}, edge_load={case.edge_load:.0f}",
        fontsize=8.5,
    )


def save_shape_variations() -> None:
    cases = [
        PlateCase("plate_000", 153.2451, 94.4711, 8.2913, 112.8590, 66.7614, 7.5408, 582.4691),
        PlateCase("plate_001", 164.6438, 86.9239, 14.3771, 87.8972, 53.7391, 6.3343, 1331.3922),
        PlateCase("plate_002", 137.2150, 65.5229, 9.3097, 108.5124, 28.0803, 5.3037, 1176.7106),
        PlateCase("plate_003", 143.8872, 76.8150, 13.0977, 46.7532, 25.4067, 10.0367, 990.8965),
        PlateCase("plate_004", 179.7077, 88.1019, 9.7580, 101.2010, 23.5118, 11.9505, 506.8758),
        PlateCase("plate_005", 148.6954, 76.0050, 10.1227, 64.0281, 50.8372, 8.7338, 598.5988),
        PlateCase("plate_006", 120.4025, 62.9717, 9.3454, 95.8501, 31.1805, 4.4918, 1295.5220),
        PlateCase("plate_007", 167.3609, 95.0546, 12.5899, 70.5661, 34.4416, 11.9680, 670.4968),
        PlateCase("plate_008", 152.3699, 97.2081, 8.3977, 51.7062, 71.0173, 7.7472, 1308.1154),
    ]
    load_min = min(case.edge_load for case in cases)
    load_max = max(case.edge_load for case in cases)
    fig, axes = plt.subplots(3, 3, figsize=(12, 8.8))
    for ax, case in zip(axes.ravel(), cases, strict=False):
        draw_case(ax, case, load_min, load_max)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "chapter03_shape_variations.png", dpi=180)
    plt.close(fig)


def build_mesh(case: PlateCase) -> tuple[np.ndarray, np.ndarray]:
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        gmsh.model.add(case.case_id)
        occ = gmsh.model.occ
        rect = occ.addRectangle(0.0, 0.0, 0.0, case.width, case.height)
        hole = occ.addDisk(case.hole_center_x, case.hole_center_y, 0.0, case.hole_radius, case.hole_radius)
        cut_result, _ = occ.cut([(2, rect)], [(2, hole)])
        occ.synchronize()
        surface_tag = cut_result[0][1]
        boundary = gmsh.model.getBoundary([(2, surface_tag)], oriented=False)
        hole_lines = []
        for dim, line_tag in boundary:
            if dim != 1:
                continue
            xmin, ymin, _, xmax, ymax, _ = gmsh.model.getBoundingBox(1, line_tag)
            if (
                abs(xmin - (case.hole_center_x - case.hole_radius)) < 1e-3
                and abs(xmax - (case.hole_center_x + case.hole_radius)) < 1e-3
                and abs(ymin - (case.hole_center_y - case.hole_radius)) < 1e-3
                and abs(ymax - (case.hole_center_y + case.hole_radius)) < 1e-3
            ):
                hole_lines.append(line_tag)
        distance = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(distance, "CurvesList", hole_lines)
        gmsh.model.mesh.field.setNumber(distance, "Sampling", 100)
        threshold = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(threshold, "InField", distance)
        gmsh.model.mesh.field.setNumber(threshold, "SizeMin", case.mesh_size_hole)
        gmsh.model.mesh.field.setNumber(threshold, "SizeMax", case.mesh_size_far)
        gmsh.model.mesh.field.setNumber(threshold, "DistMin", case.hole_radius * 0.25)
        gmsh.model.mesh.field.setNumber(threshold, "DistMax", case.refine_distance)
        gmsh.model.mesh.field.setAsBackgroundMesh(threshold)
        gmsh.model.mesh.generate(2)

        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        coords = np.array(node_coords, dtype=float).reshape(-1, 3)[:, :2]
        id_to_index = {int(tag): i for i, tag in enumerate(node_tags)}
        element_types, _, node_tags_by_type = gmsh.model.mesh.getElements(2, surface_tag)
        tris = []
        for etype, conn in zip(element_types, node_tags_by_type, strict=False):
            name, _, _, num_nodes, _, _ = gmsh.model.mesh.getElementProperties(etype)
            if name != "Triangle 3" or num_nodes != 3:
                continue
            for i in range(0, len(conn), 3):
                tris.append([id_to_index[int(conn[i])], id_to_index[int(conn[i + 1])], id_to_index[int(conn[i + 2])]])
        return coords, np.array(tris, dtype=int)
    finally:
        gmsh.finalize()


def save_mesh_figures() -> None:
    case = PlateCase("mesh_case", 120.0, 80.0, 14.0, 54.0, 42.0, 10.0, 1100.0)
    coords, tris = build_mesh(case)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.triplot(coords[:, 0], coords[:, 1], tris, color="#334155", lw=0.45)
    ax.set_aspect("equal")
    ax.set_title("Representative unstructured mesh")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "chapter03_mesh_overview.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.triplot(coords[:, 0], coords[:, 1], tris, color="#334155", lw=0.45)
    ax.set_xlim(case.hole_center_x - 26, case.hole_center_x + 26)
    ax.set_ylim(case.hole_center_y - 26, case.hole_center_y + 26)
    ax.set_aspect("equal")
    ax.set_title("Refined mesh around hole")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "chapter03_mesh_zoom.png", dpi=180)
    plt.close(fig)


def save_sampling_resolution_figure() -> None:
    x = np.linspace(0.0, 1.0, 400)
    nonlinear_response = 0.48 + 0.22 * np.sin(3.0 * np.pi * (x + 0.08)) + 0.18 * np.exp(-((x - 0.72) / 0.09) ** 2)
    curves = [
        (0.20 + 0.52 * x + 0.008 * np.sin(2.0 * np.pi * x) + 0.15 * x ** 2, np.array([0.1, 0.9]), 1, "(a) Linear response: 2 samples", None),
        (0.18 + 1.65 * (x - 0.42) ** 2 + 0.14 * x ** 3 + 0.025 * np.sin(2.0 * np.pi * x), np.array([0.1, 0.5, 0.9]), 2, "(b) Quadratic response: 3 samples", None),
        (nonlinear_response, np.array([0.1, 0.5, 0.9]), 2, "(c) Wide design range: 3 samples", None),
        (nonlinear_response, np.array([0.55, 0.70, 0.85]), 2, "(d) Practical range: 3 samples", (0.5, 0.9)),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(9, 7), sharex=True, sharey=True)
    for ax, (response, sample_x, degree, title, practical_range) in zip(axes.ravel(), curves, strict=True):
        sample_y = np.interp(sample_x, x, response)
        coefficients = np.polyfit(sample_x, sample_y, deg=degree)
        prediction = np.polyval(coefficients, x)

        if practical_range is not None:
            ax.axvspan(*practical_range, color="#d1d5db", alpha=0.55, label="Practical range")
        ax.plot(x, response, color="#9ca3af", lw=3.0, label="Actual response")
        prediction_mask = np.ones_like(x, dtype=bool)
        if practical_range is not None:
            prediction_mask = (x >= practical_range[0]) & (x <= practical_range[1])
        ax.plot(x[prediction_mask], prediction[prediction_mask], color="#2563eb", lw=2.4, label="Regression")
        ax.scatter(
            sample_x,
            sample_y,
            s=60,
            color="#dc2626",
            edgecolor="white",
            linewidth=0.9,
            zorder=3,
            label="Samples",
        )
        ax.set_title(title, fontsize=10.5)
        ax.set_xlabel("Design variable")
        ax.grid(alpha=0.22)
        ax.set_xlim(-0.04, 1.04)
        ax.set_ylim(0.0, 0.95)
        ax.set_xticks([0.0, 0.5, 0.9, 1.0])
        ax.set_yticks([])
        ax.legend(loc="upper left", fontsize=7.5, frameon=False)

    axes[0, 0].set_ylabel("Objective function")
    axes[1, 0].set_ylabel("Objective function")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "chapter07_sampling_resolution.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_plate_problem()
    save_shape_variations()
    save_mesh_figures()
    save_sampling_resolution_figure()


if __name__ == "__main__":
    main()
