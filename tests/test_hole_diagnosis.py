import argparse
import json
from pathlib import Path

import numpy as np
import pytest
import trimesh

from toys3d.meshinspect import perform_hole_diagnosis


def _run_hole_diagnosis(tmp_path, mesh):
    """在临时目录中运行 hole diagnosis，并返回输出目录路径。"""
    args = argparse.Namespace(
        hole_diagnosis_output=str(tmp_path),
    )
    perform_hole_diagnosis(mesh, args)
    return tmp_path


def _load_hole_outputs(out_dir):
    """加载 hole diagnosis 的三个核心输出文件。"""
    npz = np.load(out_dir / "hole_diagnosis_data.npz")
    with open(out_dir / "hole_diagnosis.json", "r") as f:
        json_data = json.load(f)
    html_text = (out_dir / "hole_report.html").read_text(encoding="utf-8")
    return npz, json_data, html_text


def _assert_npz_structure(npz):
    """检查 npz 文件的必要字段是否存在且类型基本正确。"""
    required_keys = [
        "open_edge_vertex_pairs",
        "open_edge_face_ids",
        "open_edge_keys",
        "hole_ids_per_edge",
        "uncovered_edge_ids",
        "uncovered_category",
    ]
    for key in required_keys:
        assert key in npz, f"npz missing key: {key}"


def _assert_json_structure(json_data):
    """检查 JSON 报告的必要字段是否存在。"""
    required_keys = [
        "total_open_edges",
        "total_healthy_holes",
        "covered_open_edges",
        "uncovered_open_edges",
        "healthy_holes",
        "uncovered_categories_summary",
    ]
    for key in required_keys:
        assert key in json_data, f"json missing key: {key}"


def test_hole_diagnosis_single_isolated_triangle(tmp_path):
    """单个孤立三角形：3 条开放边形成一个健康孔洞。"""
    vertices = np.array([[0,0,0],[1,0,0],[0,1,0]], dtype=float)
    faces = np.array([[0,1,2]], dtype=int)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    out_dir = _run_hole_diagnosis(tmp_path, mesh)
    npz, json_data, html_text = _load_hole_outputs(out_dir)

    _assert_npz_structure(npz)
    _assert_json_structure(json_data)

    # 开放边总数 = 3
    assert json_data["total_open_edges"] == 3
    # 健康孔洞数 = 1（三角形闭合环）
    assert json_data["total_healthy_holes"] == 1
    # 覆盖开放边 = 3，未覆盖 = 0
    assert json_data["covered_open_edges"] == 3
    assert json_data["uncovered_open_edges"] == 0

    # hole_ids_per_edge 全部为 0
    hole_ids = npz["hole_ids_per_edge"]
    assert hole_ids.shape == (3,)
    assert np.all(hole_ids == 0)

    # uncovered_category 为空
    assert len(npz["uncovered_edge_ids"]) == 0
    assert len(npz["uncovered_category"]) == 0

    # JSON 中未覆盖分类为空
    assert json_data["uncovered_categories_summary"] == {}

    # HTML 包含关键统计信息
    assert "总开放边: 3" in html_text
    assert "健康孔洞数: 1" in html_text
    assert "未覆盖开放边: 0" in html_text


def test_hole_diagnosis_two_triangles_share_edge(tmp_path):
    """两个三角形共享一条边，形成一个四边形健康孔洞。"""
    vertices = np.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0]], dtype=float)
    faces = np.array([[0,1,2],[0,1,3]], dtype=int)  # 共享边 (0,1)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    out_dir = _run_hole_diagnosis(tmp_path, mesh)
    npz, json_data, html_text = _load_hole_outputs(out_dir)

    _assert_npz_structure(npz)
    _assert_json_structure(json_data)

    # 开放边：4 条（四边形边界）
    assert json_data["total_open_edges"] == 4
    # 健康孔洞数 = 1
    assert json_data["total_healthy_holes"] == 1
    assert json_data["covered_open_edges"] == 4
    assert json_data["uncovered_open_edges"] == 0

    # 健康孔洞信息：边数 = 4
    healthy_holes = json_data["healthy_holes"]
    assert len(healthy_holes) == 1
    assert healthy_holes[0]["num_edges"] == 4

    # hole_ids_per_edge 全部为 0
    hole_ids = npz["hole_ids_per_edge"]
    assert hole_ids.shape == (4,)
    assert np.all(hole_ids == 0)

    # HTML 包含关键统计信息
    assert "总开放边: 4" in html_text
    assert "健康孔洞数: 1" in html_text


def test_hole_diagnosis_three_triangles_share_vertex(tmp_path):
    """三个三角形仅共享一个顶点（扇形），无健康孔洞，开放边全部悬空。"""
    vertices = np.array([
        [0,0,0],[1,0,0],[0,1,0],
        [2,0,0],[3,0,0],
        [4,0,0],[5,0,0]
    ], dtype=float)
    faces = np.array([
        [0,1,2],
        [0,3,4],
        [0,5,6]
    ], dtype=int)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    out_dir = _run_hole_diagnosis(tmp_path, mesh)
    npz, json_data, html_text = _load_hole_outputs(out_dir)

    _assert_npz_structure(npz)
    _assert_json_structure(json_data)

    # 每个三角形三条边，没有共享边，所以 9 条开放边
    assert json_data["total_open_edges"] == 9
    # 中心顶点度数为 6，其他顶点度数为 2，无法形成闭合环
    assert json_data["total_healthy_holes"] == 0
    assert json_data["covered_open_edges"] == 0
    assert json_data["uncovered_open_edges"] == 9

    # 未覆盖开放边分类：全部为“分支内部开放边”（分类 2）
    assert json_data["uncovered_categories_summary"] == {"分支内部开放边": 9}

    # uncovered_category 数组长度 9，值全部 2
    uncovered_category = npz["uncovered_category"]
    assert uncovered_category.shape == (9,)
    assert np.all(uncovered_category == 2)


def test_hole_diagnosis_three_triangles_share_edge_nonmanifold(tmp_path):
    """三个三角形共享同一条边（非流形），开放边全部标记为非流形关联。"""
    vertices = np.array([
        [0,0,0],[1,0,0],[0,1,0],
        [1,1,0],[0.5,2,0]
    ], dtype=float)
    faces = np.array([
        [0,1,2],
        [0,1,3],
        [0,1,4]
    ], dtype=int)  # 共享边 (0,1)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    out_dir = _run_hole_diagnosis(tmp_path, mesh)
    npz, json_data, html_text = _load_hole_outputs(out_dir)

    _assert_npz_structure(npz)
    _assert_json_structure(json_data)

    # 共享非流形边不计入开放边，每个其他边只被一个面使用，共 6 条
    assert json_data["total_open_edges"] == 6
    assert json_data["total_healthy_holes"] == 0
    assert json_data["covered_open_edges"] == 0
    assert json_data["uncovered_open_edges"] == 6

    # 分类：所有开放边所属面片均为非流形面片 -> 分类 4
    assert json_data["uncovered_categories_summary"] == {"非流形关联开放边": 6}

    uncovered_category = npz["uncovered_category"]
    assert uncovered_category.shape == (6,)
    assert np.all(uncovered_category == 4)


def test_hole_diagnosis_two_isolated_triangles(tmp_path):
    """两个独立三角形：形成两个独立健康孔洞，无未覆盖开放边。"""
    vertices = np.array([
        [0,0,0],[1,0,0],[0,1,0],
        [2,0,0],[3,0,0],[2,1,0]
    ], dtype=float)
    faces = np.array([
        [0,1,2],
        [3,4,5]
    ], dtype=int)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    out_dir = _run_hole_diagnosis(tmp_path, mesh)
    npz, json_data, html_text = _load_hole_outputs(out_dir)

    _assert_npz_structure(npz)
    _assert_json_structure(json_data)

    # 每个三角形 3 条开放边，共 6 条
    assert json_data["total_open_edges"] == 6
    # 两个独立孔洞
    assert json_data["total_healthy_holes"] == 2
    assert json_data["covered_open_edges"] == 6
    assert json_data["uncovered_open_edges"] == 0

    healthy_holes = json_data["healthy_holes"]
    assert len(healthy_holes) == 2
    # 每个孔洞 3 条边
    assert [h["num_edges"] for h in healthy_holes] == [3, 3]

    # hole_ids_per_edge 长度为 6，且包含 0 和 1 两个孔洞 ID
    hole_ids = npz["hole_ids_per_edge"]
    assert hole_ids.shape == (6,)
    assert set(hole_ids.tolist()) == {0, 1}


def test_compute_open_edge_data_empty_mesh():
    """空网格应返回空开放边数据，不报错。"""
    from toys3d.geometrics import compute_open_edge_data

    mesh = trimesh.Trimesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=int), process=False)
    data = compute_open_edge_data(mesh)

    assert data['open_edge_vertex_pairs'].shape == (0, 2)
    assert data['open_edge_face_ids'].shape == (0,)
    assert data['open_edge_keys'].shape == (0,)
    assert data['open_edge_key_to_id'] == {}
    assert data['vertex_open_edges_csr'].shape == (0, 0)
    assert data['vertex_degree'].shape == (0,)
