# src/toys3d/extract_components.py
"""
批量提取健康孔洞或未覆盖开放边连通分量。

用法示例：
    python src/toys3d/extract_components.py input.stl \
        --boundary-type healthy \
        --output-dir extracted_components \
        --boundary-neighborhood-depth 2 \
        --ids 3,5,7 \
        --overwrite
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import trimesh

# 确保可以导入 toys3d 包
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_src_parent = os.path.dirname(_project_root)
if _src_parent not in sys.path:
    sys.path.insert(0, _src_parent)

from toys3d.geometrics import expand_face_neighborhood
from toys3d.geometrics import export_component_package


def load_mesh(input_file):
    """加载网格并返回 trimesh.Trimesh，与 meshinspect.py 保持一致。"""
    mesh = trimesh.load(input_file, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    return mesh


def load_healthy_holes(diag_dir):
    """
    从 hole diagnosis 目录加载健康孔洞列表。
    每个孔洞包含：
        hole_id, vertex_indices, num_edges, area, perimeter, face_ids
    """
    diag_dir = Path(diag_dir)
    diag_json = diag_dir / "hole_diagnosis.json"
    npz_path = diag_dir / "hole_diagnosis_data.npz"
    if not diag_json.exists() or not npz_path.exists():
        raise FileNotFoundError(
            f"缺少诊断文件: {diag_json} 或 {npz_path}"
        )

    with open(diag_json, "r", encoding="utf-8") as f:
        diagnosis = json.load(f)

    npz = np.load(npz_path)
    hole_ids_per_edge = npz["hole_ids_per_edge"]
    open_edge_face_ids = npz["open_edge_face_ids"]

    healthy_holes = []
    for hole in diagnosis.get("healthy_holes", []):
        hole_id = hole["hole_id"]
        # 反推种子面片
        edge_mask = hole_ids_per_edge == hole_id
        face_ids = list(set(open_edge_face_ids[edge_mask].tolist()))
        hole = dict(hole)
        hole["face_ids"] = face_ids
        healthy_holes.append(hole)

    return healthy_holes


def load_uncovered_components(diag_dir):
    """从 hole diagnosis 目录加载未覆盖开放边连通分量。"""
    diag_dir = Path(diag_dir)
    comp_json = diag_dir / "uncovered_component_analysis.json"
    if not comp_json.exists():
        raise FileNotFoundError(f"缺少未覆盖分量分析文件: {comp_json}")

    with open(comp_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    components = data.get("components", [])
    for comp in components:
        # 确保关键字段存在
        comp.setdefault("face_ids", [])
        comp.setdefault("endpoints", [])
        comp.setdefault("branch_vertices", [])
        comp.setdefault("candidate_breaks", [])
        comp.setdefault("edge_vertex_pairs", [])
    return components


def _enumerate_components(diag_dir, boundary_type):
    """
    枚举诊断目录中的组件，仅用于提供 boundary_id 列表和提取所需的原始组件数据。
    返回 (component_id, component_dict) 列表。
    """
    if boundary_type == "healthy":
        raw_components = load_healthy_holes(diag_dir)
        enumerated = []
        for hole in raw_components:
            comp = dict(hole)
            comp["component_id"] = int(hole["hole_id"])
            comp.setdefault("face_ids", [])
            comp.setdefault("vertices", [])
            comp.setdefault("edge_vertex_pairs", [])
            comp.setdefault("endpoints", [])
            comp.setdefault("branch_vertices", [])
            comp.setdefault("candidate_breaks", [])
            comp.setdefault("healthy_hole_vertex_indices", [])
            enumerated.append(comp)
        return enumerated

    elif boundary_type == "uncovered":
        raw_components = load_uncovered_components(diag_dir)
        enumerated = []
        for comp in raw_components:
            c = dict(comp)
            c.setdefault("healthy_hole_vertex_indices", [])
            enumerated.append(c)
        return enumerated

    else:
        raise ValueError(f"未知的 boundary_type: {boundary_type}")


def extract_component_by_id(
    mesh,
    comp_original,
    boundary_type,
    boundary_id,
    neighborhood_depth,
    output_dir,
    overwrite,
):
    """
    提取单个组件，保存 PLY 和 JSON。
    返回 (success: bool, face_count: int, vertex_count: int, message: str)
    """
    stem = f"{boundary_type}_{boundary_id}_depth{neighborhood_depth}"
    ply_path = Path(output_dir) / f"{stem}.ply"
    json_path = Path(output_dir) / f"{stem}.json"

    source_file = ""
    try:
        source_file = mesh.metadata.get("file_name", "") or ""
    except Exception:
        source_file = ""

    result = export_component_package(
        mesh,
        comp_original,
        boundary_type=boundary_type,
        boundary_id=boundary_id,
        neighborhood_depth=neighborhood_depth,
        ply_path=ply_path,
        json_path=json_path,
        source_file=source_file,
        overwrite=overwrite,
    )

    if not result['success']:
        return False, 0, 0, result['message']

    return (
        True,
        result['local_face_count'],
        result['local_vertex_count'],
        "",
    )


def main():
    parser = argparse.ArgumentParser(
        description="批量提取健康孔洞或未覆盖开放边连通分量。"
    )
    parser.add_argument("input_file", help="输入网格文件路径")
    parser.add_argument(
        "--boundary-type",
        choices=["healthy", "uncovered"],
        default="uncovered",
        help="组件类型：healthy（健康孔洞）或 uncovered（未覆盖开放边分量）",
    )
    parser.add_argument(
        "--output-dir",
        default="extracted_components",
        help="输出目录（默认 extracted_components）",
    )
    parser.add_argument(
        "--hole-diagnosis-dir",
        default="hole_diagnosis_report",
        help="hole diagnosis 输出目录（默认 hole_diagnosis_report）",
    )
    parser.add_argument(
        "--boundary-neighborhood-depth",
        type=int,
        default=1,
        help="邻域深度（默认 1，即仅组件面片本身）",
    )
    parser.add_argument(
        "--ids",
        type=str,
        default=None,
        help="逗号分隔的组件 ID 列表，例如 3,5,7；不指定则全部",
    )
    parser.add_argument(
        "--max-components",
        type=int,
        default=None,
        help="最多处理多少个组件（默认全部）",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已存在的输出文件",
    )

    args = parser.parse_args()

    # 输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Hey! Loading {args.input_file}")
    mesh = load_mesh(args.input_file)

    # 枚举组件（作为 boundary_id 的来源；实际提取仍走共享的
    # export_component_package，以保证与 meshinspect.py 行为一致）
    components = _enumerate_components(args.hole_diagnosis_dir, args.boundary_type)

    if not components:
        print("没有找到任何组件。")
        return

    # 按 ID 过滤
    if args.ids is not None:
        id_set = set(int(x) for x in args.ids.split(",") if x.strip())
        components = [
            c for c in components if int(c.get("component_id", -1)) in id_set
        ]

    # 限制数量
    if args.max_components is not None:
        components = components[: args.max_components]

    success_count = 0
    fail_count = 0

    for comp in components:
        boundary_id = int(comp.get("component_id", -1))
        success, face_cnt, vert_cnt, msg = extract_component_by_id(
            mesh,
            comp,
            args.boundary_type,
            boundary_id,
            args.boundary_neighborhood_depth,
            output_dir,
            args.overwrite,
        )
        if success:
            success_count += 1
            print(
                f"[OK] {args.boundary_type}_{boundary_id}: "
                f"局部面片数 {face_cnt}, 局部顶点数 {vert_cnt}"
            )
        else:
            fail_count += 1
            print(f"[FAIL] {args.boundary_type}_{boundary_id}: {msg}")

    print("\n汇总:")
    print(f"成功提取: {success_count} 个组件")
    print(f"失败:     {fail_count} 个组件")
    print(f"输出目录: {output_dir}")


if __name__ == "__main__":
    main()
