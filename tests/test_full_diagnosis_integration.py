import argparse
import json
from pathlib import Path

import numpy as np
import pytest
import trimesh

from toys3d.geometrics import code_to_hex
from toys3d.meshinspect import perform_full_diagnosis


def _run_diagnosis(tmp_path, mesh):
    """在临时目录中运行 full diagnosis，并返回输出目录路径。"""
    args = argparse.Namespace(
        diagnosis_output=str(tmp_path),
        resume=False,
        diagnosis_format='html',
    )
    perform_full_diagnosis(mesh, args)
    return tmp_path


def _load_outputs(out_dir):
    """加载 full diagnosis 的三个核心输出文件。"""
    codes = np.load(out_dir / "face_codes.npy")
    with open(out_dir / "abnormal_truncated_classes.json", "r") as f:
        abnormal_data = json.load(f)
    html_text = (out_dir / "report.html").read_text(encoding="utf-8")
    return codes, abnormal_data, html_text


def _codes_to_hex_set(codes):
    """将 uint8 形状 (n,6) 的面片编码数组转换为 hex 字符串集合。"""
    return {code_to_hex(codes[i]) for i in range(len(codes))}


def _assert_common_structure(codes, abnormal_data, html_text, expected_n_faces):
    """通用结构断言。"""
    # 所有面片的真实编码
    assert codes.ndim == 2
    assert codes.shape[1] == 6
    assert codes.dtype == np.uint8
    assert codes.shape[0] == expected_n_faces

    # JSON 基本字段
    assert abnormal_data["valence_threshold"] == 5
    assert "classes" in abnormal_data

    # 每个类条目结构
    for hex_code, cls in abnormal_data["classes"].items():
        assert isinstance(hex_code, str)
        assert len(hex_code) == 12  # 6 字节 hex
        assert "face_indices" in cls
        assert isinstance(cls["face_indices"], list)
        assert cls["count"] == len(cls["face_indices"])
        assert cls["status"] == "done"

    # HTML 存在且包含基本结构
    assert "<svg" in html_text


def test_full_diagnosis_single_triangle(tmp_path):
    """单个孤立三角形：3 条开放边，3 个独占顶点。"""
    vertices = np.array([[0,0,0],[1,0,0],[0,1,0]], dtype=float)
    faces = np.array([[0,1,2]], dtype=int)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    out_dir = _run_diagnosis(tmp_path, mesh)
    codes, abnormal_data, html_text = _load_outputs(out_dir)

    _assert_common_structure(codes, abnormal_data, html_text, expected_n_faces=1)

    # 真实编码（未截断）：[1,1,1,1,1,1] -> hex 010101010101
    assert _codes_to_hex_set(codes) == {"010101010101"}

    # 异常分类：只有一个类，键为 010101010101，面片索引 [0]
    classes = abnormal_data["classes"]
    assert set(classes.keys()) == {"010101010101"}
    cls = classes["010101010101"]
    assert set(cls["face_indices"]) == {0}
    assert cls["count"] == 1

    # HTML 中包含该类标题
    assert "Class 010101010101" in html_text


def test_full_diagnosis_two_triangles_share_edge(tmp_path):
    """两个三角形共享一条边：两个面各有一条开放边和一条流形边。"""
    vertices = np.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0]], dtype=float)
    faces = np.array([[0,1,2],[0,1,3]], dtype=int)  # 共享边 (0,1)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    out_dir = _run_diagnosis(tmp_path, mesh)
    codes, abnormal_data, html_text = _load_outputs(out_dir)

    _assert_common_structure(codes, abnormal_data, html_text, expected_n_faces=2)

    # 两个面的真实规范化编码相同：010101020202
    assert _codes_to_hex_set(codes) == {"010101020202"}

    classes = abnormal_data["classes"]
    assert set(classes.keys()) == {"010101020202"}
    cls = classes["010101020202"]
    # 两个面都是异常面，索引 {0,1}
    assert set(cls["face_indices"]) == {0,1}
    assert cls["count"] == 2

    assert "Class 010101020202" in html_text


def test_full_diagnosis_three_triangles_share_vertex(tmp_path):
    """三个三角形仅共享顶点0（无共享边），中心顶点 valence=3。"""
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

    out_dir = _run_diagnosis(tmp_path, mesh)
    codes, abnormal_data, html_text = _load_outputs(out_dir)

    _assert_common_structure(codes, abnormal_data, html_text, expected_n_faces=3)

    # 三个面的真实规范化编码相同：010101010103
    assert _codes_to_hex_set(codes) == {"010101010103"}

    classes = abnormal_data["classes"]
    assert set(classes.keys()) == {"010101010103"}
    cls = classes["010101010103"]
    assert set(cls["face_indices"]) == {0,1,2}
    assert cls["count"] == 3

    assert "Class 010101010103" in html_text


def test_full_diagnosis_three_triangles_share_edge_nonmanifold(tmp_path):
    """三个三角形共享同一条边 (0,1)，产生非流形边，边元=3。"""
    vertices = np.array([
        [0,0,0],[1,0,0],[0,1,0],
        [1,1,0],[0.5,2,0]
    ], dtype=float)
    faces = np.array([
        [0,1,2],
        [0,1,3],
        [0,1,4]
    ], dtype=int)   # 共享边 (0,1)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    out_dir = _run_diagnosis(tmp_path, mesh)
    codes, abnormal_data, html_text = _load_outputs(out_dir)

    _assert_common_structure(codes, abnormal_data, html_text, expected_n_faces=3)

    # 三个面的真实规范化编码相同：010101030303
    assert _codes_to_hex_set(codes) == {"010101030303"}

    classes = abnormal_data["classes"]
    assert set(classes.keys()) == {"010101030303"}
    cls = classes["010101030303"]
    assert set(cls["face_indices"]) == {0,1,2}
    assert cls["count"] == 3

    assert "Class 010101030303" in html_text


def test_full_diagnosis_high_valence_vertex(tmp_path):
    """六个三角形共享顶点0，中心顶点 valence=6，截断后顶点元=5。"""
    vertices = np.array([
        [0,0,0],
        [1,0,0],[0,1,0],
        [2,0,0],[3,0,0],
        [4,0,0],[5,0,0],
        [6,0,0],[7,0,0],
        [8,0,0],[9,0,0],
        [10,0,0],[11,0,0]
    ], dtype=float)
    # 构造 6 个扇形三角形，每个包含中心顶点0
    faces = np.array([
        [0,1,2],
        [0,3,4],
        [0,5,6],
        [0,7,8],
        [0,9,10],
        [0,11,12]
    ], dtype=int)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    out_dir = _run_diagnosis(tmp_path, mesh)
    codes, abnormal_data, html_text = _load_outputs(out_dir)

    _assert_common_structure(codes, abnormal_data, html_text, expected_n_faces=6)

    # 真实编码中，中心顶点 valence=6，所以真实编码某位置应为 6，
    # 但截断后的分类中，顶点元为 5。
    # 真实编码未截断，所以每个面的顶点元包含 6，规范化后最小字典序为 010101010106
    assert _codes_to_hex_set(codes) == {"010101010106"}

    # 截断分类：顶点元 6 -> 5，所以键为 010101010105
    classes = abnormal_data["classes"]
    assert set(classes.keys()) == {"010101010105"}
    cls = classes["010101010105"]
    assert set(cls["face_indices"]) == set(range(6))
    assert cls["count"] == 6

    assert "Class 010101010105" in html_text


def test_full_diagnosis_single_triangle_plus_degenerate(tmp_path):
    """一个正常孤立三角形 + 一个退化孤立三角形：两者编码相同，归为一类。"""
    vertices = np.array([
        [0,0,0],[1,0,0],[0,1,0],   # 正常三角形
        [2,0,0],[3,0,0],[4,0,0]   # 共线退化三角形
    ], dtype=float)
    faces = np.array([
        [0,1,2],
        [3,4,5]
    ], dtype=int)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    out_dir = _run_diagnosis(tmp_path, mesh)
    codes, abnormal_data, html_text = _load_outputs(out_dir)

    _assert_common_structure(codes, abnormal_data, html_text, expected_n_faces=2)

    # 两个孤立三角形编码相同：010101010101
    assert _codes_to_hex_set(codes) == {"010101010101"}

    classes = abnormal_data["classes"]
    assert set(classes.keys()) == {"010101010101"}
    cls = classes["010101010101"]
    assert set(cls["face_indices"]) == {0,1}
    assert cls["count"] == 2

    # 面积统计中最小值应为 0（退化面）
    area_stats = cls.get("area_stats")
    assert area_stats is not None
    assert area_stats["min"] == pytest.approx(0.0, abs=1e-12)

    assert "Class 010101010101" in html_text
