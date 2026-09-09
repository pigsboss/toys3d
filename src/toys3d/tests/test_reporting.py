"""
reporting 模块测试。

覆盖：
1. load_uncovered_edge_data 从 npz 加载未覆盖开放边数据；
2. load_boundary_component_data 支持 uncovered 和 healthy 两种模式；
3. 边界组件加载后的数据结构完整。
"""

import sys
import json
from pathlib import Path

import numpy as np
import pytest

# 确保可以导入 src/toys3d
PROJECT_ROOT = Path(__file__).resolve().parents[3]  # 仓库根目录
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from toys3d.reporting import (
    load_uncovered_edge_data,
    load_boundary_component_data,
)


def _make_hole_diagnosis_npz(tmp_path: Path):
    """创建最小的 hole_diagnosis_data.npz 文件。"""
    npz_path = tmp_path / "hole_diagnosis_data.npz"
    uncovered_ids = np.array([2, 5], dtype=np.int64)
    all_vertex_pairs = np.array([[1, 2], [2, 3], [4, 5], [5, 6], [7, 8]],
                                dtype=np.int64)
    categories = np.array([1, 4], dtype=np.int8)
    np.savez_compressed(
        npz_path,
        uncovered_edge_ids=uncovered_ids,
        open_edge_vertex_pairs=all_vertex_pairs,
        uncovered_category=categories,
    )
    return npz_path


def test_load_uncovered_edge_data(tmp_path):
    """加载未覆盖开放边数据的 happy path。"""
    npz_path = _make_hole_diagnosis_npz(tmp_path)

    uncovered_ids, all_pairs, categories = load_uncovered_edge_data(tmp_path)

    assert isinstance(uncovered_ids, np.ndarray)
    assert uncovered_ids.tolist() == [2, 5]
    assert all_pairs.shape == (5, 2)
    assert categories.tolist() == [1, 4]


def test_load_uncovered_edge_data_missing_file(tmp_path):
    """文件不存在时抛出 FileNotFoundError。"""
    with pytest.raises(FileNotFoundError):
        load_uncovered_edge_data(tmp_path)


def test_load_boundary_component_data_uncovered(tmp_path):
    """加载 uncovered 分量 JSON。"""
    comp_json = {
        "total_components": 2,
        "components": [
            {
                "component_id": 0,
                "num_edges": 6,
                "num_vertices": 5,
                "vertices": [1, 2, 3, 4, 5],
                "edge_vertex_pairs": [[1, 2], [2, 3], [3, 4], [4, 5], [5, 1], [2, 5]],
                "endpoints": [],
                "branch_vertices": [2, 5],
                "is_cycle": False,
                "face_ids": [10, 11, 12],
                "open_face_count": 3,
                "nonmanifold_face_count": 0,
                "candidate_breaks": [{"v0": 1, "v1": 4, "distance": 2.0}],
            },
            {
                "component_id": 1,
                "num_edges": 3,
                "num_vertices": 3,
                "vertices": [7, 8, 9],
                "edge_vertex_pairs": [[7, 8], [8, 9], [9, 7]],
                "endpoints": [],
                "branch_vertices": [],
                "is_cycle": True,
                "face_ids": [20, 21],
                "open_face_count": 2,
                "nonmanifold_face_count": 0,
                "candidate_breaks": [],
            },
        ],
    }
    comp_path = tmp_path / "uncovered_component_analysis.json"
    comp_path.write_text(json.dumps(comp_json), encoding="utf-8")

    comp = load_boundary_component_data(tmp_path, boundary_id=1, boundary_type="uncovered")

    assert comp["component_id"] == 1
    assert comp["num_edges"] == 3
    assert comp["is_cycle"] is True
    assert comp["face_ids"] == [20, 21]
    assert "endpoints" in comp
    assert "branch_vertices" in comp
    assert "candidate_breaks" in comp


def test_load_boundary_component_data_uncovered_invalid_id(tmp_path):
    """无效 ID 抛出 ValueError。"""
    comp_json = {
        "total_components": 1,
        "components": [{"component_id": 0, "num_edges": 1}],
    }
    (tmp_path / "uncovered_component_analysis.json").write_text(
        json.dumps(comp_json), encoding="utf-8"
    )

    with pytest.raises(ValueError):
        load_boundary_component_data(tmp_path, boundary_id=2, boundary_type="uncovered")


def test_load_boundary_component_data_healthy(tmp_path):
    """加载 healthy 孔洞组件。"""
    npz_path = tmp_path / "hole_diagnosis_data.npz"
    hole_ids_per_edge = np.array([0, 0, 0, 0, -1], dtype=np.int32)
    open_edge_vertex_pairs = np.array([[1, 2], [2, 3], [3, 4], [4, 1], [5, 6]],
                                      dtype=np.int64)
    open_edge_face_ids = np.array([100, 101, 102, 103, 200], dtype=np.int64)
    np.savez_compressed(
        npz_path,
        hole_ids_per_edge=hole_ids_per_edge,
        open_edge_vertex_pairs=open_edge_vertex_pairs,
        open_edge_face_ids=open_edge_face_ids,
    )

    diag_json = {
        "healthy_holes": [
            {
                "hole_id": 0,
                "vertex_indices": [1, 2, 3, 4],
                "num_edges": 4,
            }
        ]
    }
    (tmp_path / "hole_diagnosis.json").write_text(json.dumps(diag_json), encoding="utf-8")

    comp = load_boundary_component_data(tmp_path, boundary_id=0, boundary_type="healthy")

    assert comp["component_id"] == 0
    assert comp["is_cycle"] is True
    assert comp["num_edges"] == 4
    assert comp["healthy_hole_vertex_indices"] == [1, 2, 3, 4]
    assert comp["face_ids"] == [100, 101, 102, 103]
