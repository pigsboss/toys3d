"""
geometrics 包回归测试。

目标：
1. 验证 4 模块拆分后各子模块可以独立导入；
2. 验证 utils 与 topology 的关键算法在合成网格上行为正确；
3. 验证包的 `__init__` 是否仍能提供旧接口，确保迁移不破坏调用方。

运行：
    pytest tests/test_geometrics.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh

# 保证可以导入 src/toys3d
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from toys3d.geometrics.euclidean import polygon_area_from_3d_ccw
from toys3d.geometrics.discrete import compute_mesh_stats
from toys3d.geometrics.topology import (
    analyze_mesh_defects,
    extract_boundary_loops,
    compute_open_edge_data,
    compute_face_topology_codes,
    group_faces_by_topology_codes,
    code_to_hex,
    hex_to_code,
)
from toys3d.geometrics.utilities import (
    compute_hole_area_stats,
    repair_mesh_by_removing_duplicates,
    weld_small_holes,
    build_hole_diagnosis_data,
    analyze_uncovered_open_edge_components,
)


# ---------------------------------------------------------------------------
# 合成网格构造
# ---------------------------------------------------------------------------

def make_single_triangle_mesh():
    """单个开放三角形：3 条开放边。"""
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float64)

    faces = np.array([[0, 1, 2]], dtype=np.int64)

    return trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        process=False,
        validate=False,
    )


def make_open_square_mesh():
    """
    由两个三角形组成的开放平面方形。
    外边界 4 条开放边，内部对角线 1 条流形边。
    """
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float64)

    faces = np.array([
        [0, 1, 2],
        [0, 2, 3],
    ], dtype=np.int64)

    return trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        process=False,
        validate=False,
    )


def make_branching_mesh():
    """
    两个三角形只共享顶点 0，不共享边。
    形成含分支顶点 0 的未覆盖开放边分量。
    """
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
    ], dtype=np.float64)

    faces = np.array([
        [0, 1, 2],
        [0, 3, 4],
    ], dtype=np.int64)

    return trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        process=False,
        validate=False,
    )


# ---------------------------------------------------------------------------
# euclidean
# ---------------------------------------------------------------------------

def test_polygon_area_from_3d_ccw_planar_triangle():
    pts = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
    ], dtype=np.float64)

    area = polygon_area_from_3d_ccw(pts)
    assert area == pytest.approx(2.0)


def test_polygon_area_from_3d_ccw_degenerate():
    pts = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
    ], dtype=np.float64)

    assert polygon_area_from_3d_ccw(pts) == 0.0


# ---------------------------------------------------------------------------
# discrete
# ---------------------------------------------------------------------------

def test_compute_mesh_stats_box():
    box = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
    stats = compute_mesh_stats(box)

    assert stats["vertices"] == 8
    assert stats["faces"] == 12
    assert stats["edges"] == 18
    assert stats["is_watertight"] is True
    assert stats["mean_edge_length"] > 0.0


# ---------------------------------------------------------------------------
# topology
# ---------------------------------------------------------------------------

def test_analyze_mesh_defects_single_triangle():
    mesh = make_single_triangle_mesh()
    defect_stats, open_mask, nonmanifold_mask = analyze_mesh_defects(mesh)

    assert defect_stats["open_edges"] == 3
    assert defect_stats["nonmanifold_edges"] == 0
    assert defect_stats["open_faces"] == 1
    assert defect_stats["nonmanifold_faces"] == 0
    assert np.all(open_mask)
    assert not np.any(nonmanifold_mask)


def test_analyze_mesh_defects_open_square():
    mesh = make_open_square_mesh()
    defect_stats, open_mask, nonmanifold_mask = analyze_mesh_defects(mesh)

    assert defect_stats["open_edges"] == 4
    assert defect_stats["nonmanifold_edges"] == 0
    assert defect_stats["open_faces"] == 2
    assert np.all(open_mask)
    assert not np.any(nonmanifold_mask)


def test_extract_boundary_loops_open_square():
    mesh = make_open_square_mesh()
    loops = extract_boundary_loops(mesh)

    assert len(loops) == 1
    loop = loops[0]
    assert len(loop) == 4
    assert set(int(v) for v in loop) == {0, 1, 2, 3}


def test_extract_boundary_loops_empty_on_watertight():
    box = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
    assert extract_boundary_loops(box) == []


def test_compute_open_edge_data_single_triangle():
    mesh = make_single_triangle_mesh()
    data = compute_open_edge_data(mesh)

    assert len(data["open_edge_vertex_pairs"]) == 3
    assert len(data["open_edge_face_ids"]) == 3
    assert len(data["open_edge_keys"]) == 3
    assert np.all(data["vertex_degree"] == 2)


def test_compute_face_topology_codes_shape():
    mesh = make_open_square_mesh()
    vertex_face_counts = np.bincount(
        mesh.faces.ravel(), minlength=len(mesh.vertices)
    ).astype(np.int64)

    from toys3d.geometrics.topology import compute_face_edge_types

    face_edge_types = compute_face_edge_types(mesh)
    codes, code_to_id, id_to_code = compute_face_topology_codes(
        mesh,
        np.arange(len(mesh.faces), dtype=np.int64),
        vertex_face_counts,
        face_edge_types,
    )

    assert codes.shape == (2, 6)
    assert codes.dtype == np.uint8


def test_group_faces_by_topology_codes():
    mesh = make_open_square_mesh()
    vertex_face_counts = np.bincount(
        mesh.faces.ravel(), minlength=len(mesh.vertices)
    ).astype(np.int64)

    from toys3d.geometrics.topology import compute_face_edge_types

    face_edge_types = compute_face_edge_types(mesh)
    grouped = group_faces_by_topology_codes(
        mesh,
        np.arange(len(mesh.faces), dtype=np.int64),
        vertex_face_counts,
        face_edge_types,
        valence_threshold=5,
    )

    # 两个面片的拓扑编码可能相同，也可能不同；仅要求分组非空且覆盖所有面
    assert len(grouped) >= 1
    all_faces = np.concatenate(list(grouped.values()))
    assert set(all_faces.tolist()) == {0, 1}


def test_code_hex_roundtrip():
    code = np.array([1, 2, 3, 4, 5, 6], dtype=np.uint8)
    hex_str = code_to_hex(code)
    recovered = hex_to_code(hex_str)
    assert np.array_equal(recovered, code)


def test_build_hole_diagnosis_data_open_square():
    mesh = make_open_square_mesh()
    data = build_hole_diagnosis_data(mesh)

    assert len(data["hole_vertex_lists"]) == 1
    assert len(data["hole_vertex_lists"][0]) == 4
    assert len(data["hole_edge_lists"]) == 1
    assert len(data["hole_edge_lists"][0]) == 4
    assert len(data["uncovered_edge_ids"]) == 0


def test_build_hole_diagnosis_data_branching_mesh():
    mesh = make_branching_mesh()
    data = build_hole_diagnosis_data(mesh)

    # 所有开放边均未形成健康孔洞
    assert len(data["hole_vertex_lists"]) == 0
    assert len(data["uncovered_edge_ids"]) == 6


def test_analyze_uncovered_open_edge_components_branching():
    mesh = make_branching_mesh()
    hole_data = build_hole_diagnosis_data(mesh)
    components = analyze_uncovered_open_edge_components(mesh, hole_data)

    assert len(components) == 1
    comp = components[0]
    assert comp["num_edges"] == 6
    assert comp["branch_vertices"] == [0]
    assert comp["endpoints"] == []
    assert comp["is_cycle"] is False


# ---------------------------------------------------------------------------
# utilities
# ---------------------------------------------------------------------------

def test_compute_hole_area_stats_open_square():
    mesh = make_open_square_mesh()
    stats = compute_hole_area_stats(mesh)

    assert stats["count"] == 1
    assert stats["total_area"] == pytest.approx(1.0)
    assert stats["max_area"] == pytest.approx(1.0)


def test_repair_mesh_by_removing_duplicates():
    # 构造带重复顶点的网格，修复后应无重复
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],  # 与 vertices[1] 重复
        [0.0, 1.0, 0.0],  # 与 vertices[2] 重复
    ], dtype=np.float64)
    faces = np.array([
        [0, 1, 2],
        [0, 3, 4],
    ], dtype=np.int64)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    repaired = repair_mesh_by_removing_duplicates(mesh)
    assert len(repaired.vertices) < len(mesh.vertices)


def test_weld_small_holes_open_square():
    mesh = make_open_square_mesh()

    # 默认焊接面积小于面片面积 p5 的小孔洞
    welded = weld_small_holes(
        mesh,
        threshold=0.5,
        quantile=5.0,
        min_edges=3,
        verbose=False,
    )

    # 焊接后开放边应减少（边界顶点合并为中心点）
    before_defects, _, _ = analyze_mesh_defects(mesh)
    after_defects, _, _ = analyze_mesh_defects(welded)

    assert after_defects["open_edges"] <= before_defects["open_edges"]


# ---------------------------------------------------------------------------
# 包迁移回归：旧导入路径仍应可用
# ---------------------------------------------------------------------------

def test_geometrics_package_legacy_exports():
    import toys3d.geometrics as geom

    for name in [
        "compute_mesh_stats",
        "analyze_mesh_defects",
        "extract_boundary_loops",
        "compute_hole_area_stats",
        "polygon_area_from_3d_ccw",
        "repair_mesh_by_removing_duplicates",
        "project_vertices_to_shell",
        "weld_small_holes",
        "compute_vertex_face_counts",
        "compute_face_edge_types",
        "compute_face_topology_codes",
        "compute_edge_to_faces",
        "compute_face_edge_keys",
        "compute_face_edge_valences",
        "compute_class_neighbor_stats",
        "compute_single_face_neighbor_stats",
        "get_face_topology_code_and_order",
        "code_to_hex",
        "hex_to_code",
        "save_codes",
        "group_faces_by_topology_codes",
        "build_hole_diagnosis_data",
        "analyze_uncovered_open_edge_components",
        "build_manifold_face_adjacency",
        "is_manifold_closed_boundary",
        "find_minimal_enclosing_manifold_boundary_greedy",
        "fit_watertight_patch_from_component",
    ]:
        assert hasattr(geom, name), f"geometrics 包缺少 {name}"
