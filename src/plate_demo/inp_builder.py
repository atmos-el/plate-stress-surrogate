from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import gmsh


""" NOTE：①  """
@dataclass(frozen=True)
class PlateCase:
    case_id: str
    width: float
    height: float
    hole_radius: float
    hole_center_x: float
    hole_center_y: float
    thickness: float
    edge_load: float
    mesh_size_far: float = 8.0
    mesh_size_hole: float = 2.5
    refine_distance: float = 12.0
    young: float = 210000.0
    poisson: float = 0.3
""" END """


def build_plate_input(case: PlateCase, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:

        """ NOTE：②  """
        gmsh.model.add(case.case_id)
        occ = gmsh.model.occ
        rect = occ.addRectangle(0.0, 0.0, 0.0, case.width, case.height)
        hole = occ.addDisk(case.hole_center_x, case.hole_center_y, 0.0, case.hole_radius, case.hole_radius)
        cut_result, _ = occ.cut([(2, rect)], [(2, hole)])
        occ.synchronize()
        """ END """
        if not cut_result:
            raise ValueError(f"failed to create perforated plate geometry for {case.case_id}")

        surface_tag = cut_result[0][1]
        boundary = gmsh.model.getBoundary([(2, surface_tag)], oriented=False)
        line_tags = [tag for dim, tag in boundary if dim == 1]
        left_lines: list[int] = []
        right_lines: list[int] = []
        hole_lines: list[int] = []
        tol = 1e-6
        for line_tag in line_tags:
            xmin, ymin, _, xmax, ymax, _ = gmsh.model.getBoundingBox(1, line_tag)
            if abs(xmin) < tol and abs(xmax) < tol:
                left_lines.append(line_tag)
            elif abs(xmin - case.width) < tol and abs(xmax - case.width) < tol:
                right_lines.append(line_tag)
            elif (
                abs(xmin - (case.hole_center_x - case.hole_radius)) < 1e-3
                and abs(xmax - (case.hole_center_x + case.hole_radius)) < 1e-3
                and abs(ymin - (case.hole_center_y - case.hole_radius)) < 1e-3
                and abs(ymax - (case.hole_center_y + case.hole_radius)) < 1e-3
            ):
                hole_lines.append(line_tag)

        if not left_lines or not right_lines or not hole_lines:
            raise ValueError(f"failed to classify boundaries for {case.case_id}")

        surf_group = gmsh.model.addPhysicalGroup(2, [surface_tag])
        gmsh.model.setPhysicalName(2, surf_group, "EALL")
        left_group = gmsh.model.addPhysicalGroup(1, left_lines)
        gmsh.model.setPhysicalName(1, left_group, "LEFT")
        right_group = gmsh.model.addPhysicalGroup(1, right_lines)
        gmsh.model.setPhysicalName(1, right_group, "RIGHT")
        hole_group = gmsh.model.addPhysicalGroup(1, hole_lines)
        gmsh.model.setPhysicalName(1, hole_group, "HOLE")

        gmsh.option.setNumber("Mesh.Algorithm", 6)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)

        """ NOTE：③  """
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
        gmsh.model.mesh.removeDuplicateNodes()
        """ END """

        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        coords_by_tag = {
            int(tag): (float(node_coords[i * 3]), float(node_coords[i * 3 + 1]))
            for i, tag in enumerate(node_tags)
        }
        used_node_tags, elem_conn = _extract_surface_triangles(surface_tag)
        left_nodes = _get_group_nodes(1, left_group, used_node_tags, coords_by_tag)
        right_nodes = _get_group_nodes(1, right_group, used_node_tags, coords_by_tag)
        if not elem_conn or len(left_nodes) < 2 or len(right_nodes) < 2:
            raise ValueError(f"invalid mesh connectivity for {case.case_id}")

        node_id_map = {tag: index + 1 for index, tag in enumerate(sorted(used_node_tags))}
        # Convert a uniform edge traction into consistent nodal forces for
        # linear boundary segments.  End nodes receive half of an adjacent
        # segment load; the sum remains case.edge_load (the total force).
        right_node_loads = _consistent_edge_loads(right_nodes, coords_by_tag, case.edge_load)
        inp_path = output_dir / f"{case.case_id}.inp"
        lines = [
            "*HEADING",
            f"2D plate with hole - {case.case_id}",
            "*NODE",
        ]
        for old_tag in sorted(used_node_tags):
            x, y = coords_by_tag[old_tag]
            lines.append(f"{node_id_map[old_tag]}, {x:.6f}, {y:.6f}, 0.0")

        lines.append("*ELEMENT, TYPE=CPS3, ELSET=EALL")
        for eid, conn in enumerate(elem_conn, start=1):
            remapped = ", ".join(str(node_id_map[tag]) for tag in conn)
            lines.append(f"{eid}, {remapped}")

        """ NOTE：④  """
        lines.extend(
            [
                "*NSET, NSET=LEFT",
                _join_ids([node_id_map[tag] for tag in left_nodes]),
                "*NSET, NSET=RIGHT",
                _join_ids([node_id_map[tag] for tag in right_nodes]),
                "*NSET, NSET=NALL",
                _join_ids(list(range(1, len(used_node_tags) + 1))),
                "*MATERIAL, NAME=STEEL",
                "*ELASTIC",
                f"{case.young:.3f}, {case.poisson:.6f}",
                "*SOLID SECTION, ELSET=EALL, MATERIAL=STEEL",
                f"{case.thickness:.6f}",
                "*STEP",
                "*STATIC",
                "*BOUNDARY",
                "LEFT, 1, 1, 0.0",
                "LEFT, 2, 2, 0.0",
                "*CLOAD",
            ]
        )
        """ END """
        for tag in right_nodes:
            lines.append(f"{node_id_map[tag]}, 1, {right_node_loads[tag]:.6f}")

        lines.extend(
            [
                "*NODE FILE",
                "U",
                "*EL FILE",
                "S",
                "*NODE PRINT, NSET=NALL",
                "U",
                "*EL PRINT, ELSET=EALL",
                "S",
                "*END STEP",
            ]
        )
        inp_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return inp_path
    finally:
        gmsh.finalize()


def _extract_surface_triangles(surface_tag: int) -> tuple[set[int], list[tuple[int, int, int]]]:
    element_types, _, node_tags_by_type = gmsh.model.mesh.getElements(2, surface_tag)
    triangles: list[tuple[int, int, int]] = []
    used_nodes: set[int] = set()
    for etype, node_tags in zip(element_types, node_tags_by_type, strict=False):
        name, _, _, num_nodes, _, _ = gmsh.model.mesh.getElementProperties(etype)
        if name != "Triangle 3" or num_nodes != 3:
            continue
        for i in range(0, len(node_tags), 3):
            conn = (int(node_tags[i]), int(node_tags[i + 1]), int(node_tags[i + 2]))
            triangles.append(conn)
            used_nodes.update(conn)
    return used_nodes, triangles


def _get_group_nodes(
    dim: int,
    physical_group: int,
    used_node_tags: set[int],
    coords_by_tag: dict[int, tuple[float, float]],
) -> list[int]:
    node_tags, _ = gmsh.model.mesh.getNodesForPhysicalGroup(dim, physical_group)
    filtered = sorted({int(tag) for tag in node_tags if int(tag) in used_node_tags})
    return sorted(filtered, key=lambda tag: (coords_by_tag[tag][1], coords_by_tag[tag][0]))


def _join_ids(ids: list[int]) -> str:
    chunks: list[str] = []
    for i in range(0, len(ids), 12):
        chunks.append(", ".join(str(v) for v in ids[i : i + 12]))
    return "\n".join(chunks)


def _consistent_edge_loads(
    node_tags: list[int],
    coords_by_tag: dict[int, tuple[float, float]],
    total_force: float,
) -> dict[int, float]:
    segment_lengths = [
        abs(coords_by_tag[b][1] - coords_by_tag[a][1])
        for a, b in zip(node_tags[:-1], node_tags[1:], strict=True)
    ]
    edge_length = sum(segment_lengths)
    if edge_length <= 0.0:
        raise ValueError("right edge has zero length")
    traction = total_force / edge_length
    loads = {tag: 0.0 for tag in node_tags}
    for (a, b), length in zip(zip(node_tags[:-1], node_tags[1:], strict=True), segment_lengths, strict=True):
        nodal_force = traction * length / 2.0
        loads[a] += nodal_force
        loads[b] += nodal_force
    return loads
