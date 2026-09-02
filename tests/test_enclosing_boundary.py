import argparse
import json
from pathlib import Path

import numpy as np
import trimesh

from toys3d.meshinspect import perform_hole_diagnosis
from toys3d.geometrics import find_minimal_enclosing_manifold_boundary_greedy


def build_test_mesh_with_uncovered_component():
    """
    构造一个带未覆盖开放边组件且外部存在流形包络边界的平面网格。
    返回 (mesh, expected_seed_faces)。
    """
    vertices = [
        [0.0, 0.0, 0.0],          # 0 O
        [1.0, 0.0, 0.0],          # 1 P0
        [0.0, 1.0, 0.0],          # 2 P1
        [-1.0, 0.0, 0.0],         # 3 P2
        [0.0, -1.0, 0.0],         # 4 P3
        [2.0, 0.0, 0.0],          # 5 Q0
        [0.0, 2.0, 0.0],          # 6 Q1
        [-2.0, 0.0, 0.0],         # 7 Q2
        [0.0, -2.0, 0.0],         # 8 Q3
        [1.0, -1.0, 0.0],         # 9 额外顶点，用于构造流形边 4-1
    ]

    # 内部三个三角形（种子面片）
    inner_faces = [
        [0, 1, 2],   # O-P0-P1
        [0, 2, 3],   # O-P1-P2
        [0, 3, 4],   # O-P2-P3
    ]

    # 外部环带三角形，用于构建流形包络边界
    outer_faces = [
        [1, 5, 2], [2, 5, 6], [2, 6, 3], [3, 6, 7],
        [3, 7, 4], [4, 7, 8], [4, 8, 1], [1, 8, 5],
        [4, 1, 9],   # 新增，使 4-1 成为流形边
    ]

    faces = inner_faces + outer_faces
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    expected_seed = {0, 1, 2}
    return mesh, expected_seed


def test_find_minimal_enclosing_boundary_greedy():
    mesh, seed_faces = build_test_mesh_with_uncovered_component()
    component = {"face_ids": list(seed_faces)}

    result = find_minimal_enclosing_manifold_boundary_greedy(mesh, component, max_depth=5)

    assert result['success'] is True
    assert seed_faces.issubset(set(result['enclosed_faces']))
    assert len(result['boundary_edges']) > 0
    assert len(result['boundary_vertices']) > 0


def test_hole_diagnosis_with_enclosing_boundaries(tmp_path):
    mesh, _ = build_test_mesh_with_uncovered_component()

    args = argparse.Namespace(
        hole_diagnosis_output=str(tmp_path),
        compute_enclosing_boundaries=True,
    )
    perform_hole_diagnosis(mesh, args)

    json_path = tmp_path / "uncovered_component_analysis.json"
    with open(json_path) as f:
        data = json.load(f)

    components = data["components"]
    assert len(components) > 0

    # 找到包含种子面片 0 的组件
    target_comp = None
    for comp in components:
        if 0 in comp["face_ids"]:
            target_comp = comp
            break
    assert target_comp is not None
    assert target_comp["minimal_enclosing_boundary"]["success"] is True
