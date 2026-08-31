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
    compute_edge_to_faces,
    compute_face_edge_keys,
    compute_class_neighbor_stats,
    compute_single_face_neighbor_stats,
    analyze_mesh_defects,
)
from scipy.sparse import csr_matrix


def _code_to_raw(code):
    """
    将字符串或一维数组/列表形式的拓扑编码规范化为长度为 6 的整数列表。
    若无法解析，返回 None。
    """
    if isinstance(code, str):
        if len(code) != 6 or not code.isdigit():
            return None
        return [int(ch) for ch in code]
    try:
        arr = list(code)
    except TypeError:
        return None
    if len(arr) != 6:
        return None
    for x in arr:
        if not isinstance(x, (int, np.integer)):
            return None
    return [int(x) for x in arr]


def validate_topology_code(code) -> bool:
    """
    验证一个 6 位拓扑编码是否合法。

    编码格式：v0 e0 v1 e1 v2 e2
    - 奇数位（索引 0,2,4）是顶点元，表示顶点关联的面片数（>=0）
    - 偶数位（索引 1,3,5）是边元，表示边关联的面片数（1=开放边，2=流形边，3+=非流形边）
    - 拓扑约束：
        * 每条边 e 的两个端点顶点的关联面数都必须 >= e
        * 若顶点关联面数为 1，则与它相邻的两条边必须都是开放边（e==1）
    """
    raw = _code_to_raw(code)
    if raw is None:
        return False

    v0, e0, v1, e1, v2, e2 = raw

    # 边元不能为 0，且当前实现中只出现 1/2/3
    if any(e < 1 or e > 3 for e in [e0, e1, e2]):
        return False

    # 顶点元可以为 0（孤立点），但出现在面中的顶点至少为 1；
    # 不强制限制上限。

    # 每条边的两个端点必须具有足够的关联面数
    if v0 < e0 or v1 < e0:
        return False
    if v1 < e1 or v2 < e1:
        return False
    if v2 < e2 or v0 < e2:
        return False

    # 顶点关联面数为 1 时，其两条邻边必须都是开放边
    if v0 == 1 and (e0 != 1 or e2 != 1):
        return False
    if v1 == 1 and (e0 != 1 or e1 != 1):
        return False
    if v2 == 1 and (e1 != 1 or e2 != 1):
        return False

    return True


def _build_vertex_faces_csr(mesh):
    """根据 mesh.faces 构建 (n_vertices, n_faces) 的 CSR 矩阵。"""
    n_vertices = len(mesh.vertices)
    n_faces = len(mesh.faces)
    row = mesh.faces.ravel()
    col = np.repeat(np.arange(n_faces), 3)
    data = np.ones(3 * n_faces, dtype=np.int8)
    return csr_matrix((data, (row, col)), shape=(n_vertices, n_faces))


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
        '111116',     # 边位出现 6（>3）
    ]
    # '111311'：v1='1', e1='3'，违反 v1>=e1，拓扑非法
    invalid_format.append('111311')
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
    # 将 uint8 数组转为可比较的字符串
    assert ''.join(str(int(x)) for x in codes[0]) == '111111'
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


def test_edge_to_faces_two_triangles_share_edge():
    vertices = np.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0]], float)
    faces = np.array([[0,1,2],[0,3,1]], int)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    edge_keys, edge_faces = compute_edge_to_faces(mesh)
    assert len(edge_keys) == 1
    assert len(edge_faces[0]) == 2
    face_edge_keys = compute_face_edge_keys(mesh)
    assert face_edge_keys.shape == (2,3)


def test_class_neighbor_stats_two_triangles_share_edge():
    vertices = np.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0]], float)
    faces = np.array([[0,1,2],[0,3,1]], int)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    _, open_face_mask, nonmanifold_face_mask = analyze_mesh_defects(mesh)
    edge_keys, edge_faces = compute_edge_to_faces(mesh)
    edge_to_faces = {int(k): v for k, v in zip(edge_keys, edge_faces)}
    face_edge_keys = compute_face_edge_keys(mesh)
    vertex_faces_csr = _build_vertex_faces_csr(mesh)

    point_counts, edge_counts = compute_class_neighbor_stats(
        mesh, [0,1], open_face_mask, nonmanifold_face_mask,
        vertex_faces_csr, edge_to_faces, face_edge_keys
    )
    # 两个面共享一条边，每个面的邻居都是开放面（因为每个面都有开放边），
    # 所以边邻均为 open。
    assert edge_counts == {'normal': 0, 'open': 2, 'nonmanifold': 0}
    # 无点邻
    assert point_counts == {'normal': 0, 'open': 0, 'nonmanifold': 0}


def test_single_face_neighbor_stats_two_triangles_share_edge():
    # 类似上面，检查面0的逐边/逐顶点统计
    vertices = np.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0]], float)
    faces = np.array([[0,1,2],[0,3,1]], int)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    _, open_face_mask, nonmanifold_face_mask = analyze_mesh_defects(mesh)
    edge_keys, edge_faces = compute_edge_to_faces(mesh)
    edge_to_faces = {int(k): v for k, v in zip(edge_keys, edge_faces)}
    face_edge_keys = compute_face_edge_keys(mesh)
    vertex_faces_csr = _build_vertex_faces_csr(mesh)

    v_stats, e_stats = compute_single_face_neighbor_stats(
        mesh, 0, open_face_mask, nonmanifold_face_mask,
        vertex_faces_csr, edge_to_faces, face_edge_keys
    )
    # 面0的边(0,1)是共享边，邻居面1是开放面（它也有开放边）
    # 其他两条边开放，无边邻
    assert e_stats[0] == {'normal': 0, 'open': 1, 'nonmanifold': 0}
    for i in [1,2]:
        assert e_stats[i] == {'normal': 0, 'open': 0, 'nonmanifold': 0}
    # 三个顶点均无边外点邻
    for vs in v_stats:
        assert vs == {'normal': 0, 'open': 0, 'nonmanifold': 0}


def test_class_neighbor_stats_three_triangles_share_vertex():
    # 三个三角形仅共享顶点0，无边共享 => 所有边开放，点邻=两个开放面
    vertices = np.array([
        [0,0,0],[1,0,0],[0,1,0],
        [2,0,0],[3,0,0],
        [4,0,0],[5,0,0]
    ], float)
    faces = np.array([
        [0,1,2],
        [0,3,4],
        [0,5,6]
    ], int)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    _, open_face_mask, nonmanifold_face_mask = analyze_mesh_defects(mesh)
    edge_keys, edge_faces = compute_edge_to_faces(mesh)
    edge_to_faces = {int(k): v for k, v in zip(edge_keys, edge_faces)}
    face_edge_keys = compute_face_edge_keys(mesh)
    vertex_faces_csr = _build_vertex_faces_csr(mesh)

    point_counts, edge_counts = compute_class_neighbor_stats(
        mesh, [0,1,2], open_face_mask, nonmanifold_face_mask,
        vertex_faces_csr, edge_to_faces, face_edge_keys
    )
    # 每个面通过顶点0与另外两个面点邻，所以总点邻 open=6
    assert point_counts == {'normal': 0, 'open': 6, 'nonmanifold': 0}
    # 无边邻
    assert edge_counts == {'normal': 0, 'open': 0, 'nonmanifold': 0}


def test_single_face_neighbor_stats_nonmanifold_edge():
    # 三个三角形共享同一条非流形边(0,1)
    vertices = np.array([
        [0,0,0],[1,0,0],[0,1,0],
        [1,1,0],[0.5,2,0]
    ], float)
    faces = np.array([
        [0,1,2],
        [1,0,3],
        [0,1,4]
    ], int)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    _, open_face_mask, nonmanifold_face_mask = analyze_mesh_defects(mesh)
    edge_keys, edge_faces = compute_edge_to_faces(mesh)
    edge_to_faces = {int(k): v for k, v in zip(edge_keys, edge_faces)}
    face_edge_keys = compute_face_edge_keys(mesh)
    vertex_faces_csr = _build_vertex_faces_csr(mesh)

    # 临时诊断输出，便于定位 edge_to_faces 映射问题
    print("edge_to_faces keys:", list(edge_to_faces.keys()))
    print("nonmanifold_face_mask:", nonmanifold_face_mask)
    print("face_edge_keys[0]:", face_edge_keys[0])
    print("nonmanifold edge key:", int(face_edge_keys[0, 0]))
    print("edge_to_faces.get(key):",
          edge_to_faces.get(int(face_edge_keys[0, 0])))

    v_stats, e_stats = compute_single_face_neighbor_stats(
        mesh, 0, open_face_mask, nonmanifold_face_mask,
        vertex_faces_csr, edge_to_faces, face_edge_keys
    )
    # 面0的三条边：边0(0,1)非流形，邻居面1、面2（均为非流形面）=> nonmanifold=2
    assert e_stats[0] == {'normal': 0, 'open': 0, 'nonmanifold': 2}
    # 其他边开放，无边邻
    assert e_stats[1] == {'normal': 0, 'open': 0, 'nonmanifold': 0}
    assert e_stats[2] == {'normal': 0, 'open': 0, 'nonmanifold': 0}
    # 顶点点邻应无，因为所有相邻面均通过边相连
    for vs in v_stats:
        assert vs == {'normal': 0, 'open': 0, 'nonmanifold': 0}
