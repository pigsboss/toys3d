import os
import sys

# 确保能导入 src/toys3d 下的模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import numpy as np
import pytest
import trimesh

from toys3d.geometrics import (
    compute_vertex_face_counts,
    compute_face_edge_types,
    compute_face_topology_codes,
)


def validate_topology_code(code: str) -> bool:
    """
    验证一个 6 位拓扑编码是否合法。

    编码格式：v0 e0 v1 e1 v2 e2
    - 奇数位（索引 0,2,4）必须是顶点编码：'1'（独占）或 '2'（共享）
    - 偶数位（索引 1,3,5）必须是边编码：'1'（开放），'2'（流形），'3'（非流形）
    - 拓扑约束：
        * 若顶点为 '1'（独占），则与之相邻的两条边必须为 '1'（开放）
        * 若边为 '2' 或 '3'，则其两个端点必须为 '2'（共享）
    """
    if len(code) != 6:
        return False

    # 格式检查
    vertices = [code[0], code[2], code[4]]
    edges = [code[1], code[3], code[5]]

    if any(v not in ('1', '2') for v in vertices):
        return False
    if any(e not in ('1', '2', '3') for e in edges):
        return False

    v0, e0, v1, e1, v2, e2 = code

    # 拓扑约束
    # 顶点 v0 独占 -> e0 和 e2 必须开放
    if v0 == '1' and (e0 != '1' or e2 != '1'):
        return False
    # 顶点 v1 独占 -> e0 和 e1 必须开放
    if v1 == '1' and (e0 != '1' or e1 != '1'):
        return False
    # 顶点 v2 独占 -> e1 和 e2 必须开放
    if v2 == '1' and (e1 != '1' or e2 != '1'):
        return False

    # 边 e0 流形/非流形 -> v0 和 v1 必须共享
    if e0 in ('2', '3') and (v0 != '2' or v1 != '2'):
        return False
    # 边 e1 流形/非流形 -> v1 和 v2 必须共享
    if e1 in ('2', '3') and (v1 != '2' or v2 != '2'):
        return False
    # 边 e2 流形/非流形 -> v2 和 v0 必须共享
    if e2 in ('2', '3') and (v2 != '2' or v0 != '2'):
        return False

    return True


@pytest.fixture
def single_triangle_mesh() -> trimesh.Trimesh:
    """一个孤立三角形，所有顶点独占，所有边开放。"""
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    faces = np.array([[0, 1, 2]], dtype=int)
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def test_validate_format_checks():
    """格式合法性检查"""
    valid_format = ['111111', '222222', '211122', '112111']
    for code in valid_format:
        assert validate_topology_code(code) is True

    invalid_format = [
        '11111',      # 长度错误
        '1111111',    # 长度错误
        '1a1111',     # 非法字符
        '111116',     # 顶点位出现 6
        '113111',     # 边位出现 3 但格式仍正确？此例顶点位1,1,1 边位1,3,1，格式合法但拓扑可能非法，看拓扑约束
    ]
    # '113111'：v0='1', e0='1', v1='1', e1='3', v2='1', e2='1'。
    # v1='1' 要求 e0 和 e1 都 '1'，但 e1='3'，所以非法。
    # 这里作为格式+拓扑联合测试，我们期望抛出 False。
    invalid_format.append('113111')
    for code in invalid_format:
        assert validate_topology_code(code) is False


def test_validate_rejects_specific_bad_codes():
    """从真实诊断结果中提取的已知非法编码"""
    bad_codes = [
        '112222', '121212', '121222', '121232',
        '122122', '122222', '122232', '123222', '123232',
    ]
    for code in bad_codes:
        assert validate_topology_code(code) is False


def test_validate_accepts_good_codes():
    """已知合法编码"""
    good_codes = ['111111', '222222', '211122', '112111']
    for code in good_codes:
        assert validate_topology_code(code) is True


def test_single_triangle_isolated(single_triangle_mesh):
    """孤立三角形应产生标准编码 '111111'"""
    mesh = single_triangle_mesh
    v_counts = compute_vertex_face_counts(mesh)
    e_types = compute_face_edge_types(mesh)
    face_indices = np.arange(len(mesh.faces))

    codes, _, _ = compute_face_topology_codes(
        mesh, face_indices, v_counts, e_types
    )
    assert len(codes) == 1
    assert codes[0] == '111111'
    assert validate_topology_code(codes[0])


def test_two_triangles_share_edge_valid():
    """两个三角形共享一条边，编码必须合法"""
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float)
    faces = np.array([[0, 1, 2], [0, 3, 1]], dtype=int)  # 共享边 (0,1)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    v_counts = compute_vertex_face_counts(mesh)
    e_types = compute_face_edge_types(mesh)
    face_indices = np.arange(len(mesh.faces))

    codes, _, _ = compute_face_topology_codes(
        mesh, face_indices, v_counts, e_types
    )
    for code in codes:
        assert validate_topology_code(code), f"invalid code: {code}"


def test_three_triangles_share_vertex_valid():
    """三个三角形共享一个顶点（非流形顶点），编码必须合法"""
    vertices = np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0],
        [-1, 0, 0], [0, -1, 0]
    ], dtype=float)
    # 三个面都包含顶点 0，形成扇状结构
    faces = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 4]], dtype=int)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    v_counts = compute_vertex_face_counts(mesh)
    e_types = compute_face_edge_types(mesh)
    face_indices = np.arange(len(mesh.faces))

    codes, _, _ = compute_face_topology_codes(
        mesh, face_indices, v_counts, e_types
    )
    for code in codes:
        assert validate_topology_code(code), f"invalid code: {code}"


def test_nonmanifold_edge_valid():
    """三个三角形共享同一条边（非流形边），编码必须合法"""
    vertices = np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0],
        [1, 1, 0], [0.5, 2, 0]
    ], dtype=float)
    # 三个面都包含边 (0,1)
    faces = np.array([[0, 1, 2], [1, 0, 3], [0, 1, 4]], dtype=int)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    v_counts = compute_vertex_face_counts(mesh)
    e_types = compute_face_edge_types(mesh)
    face_indices = np.arange(len(mesh.faces))

    codes, _, _ = compute_face_topology_codes(
        mesh, face_indices, v_counts, e_types
    )
    for code in codes:
        assert validate_topology_code(code), f"invalid code: {code}"
