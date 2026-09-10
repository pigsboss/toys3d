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

from toys3d.geometrics import export_component_package
from toys3d.reporting import load_boundary_component_data


def load_mesh(input_file):
    """加载网格并返回 trimesh.Trimesh，与 meshinspect.py 保持一致。"""
    mesh = trimesh.load(input_file, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    return mesh


def _enumerate_component_ids(diag_dir, boundary_type):
    """
    仅枚举诊断目录中可用的组件 ID 列表。
    真正的组件数据由 load_boundary_component_data 提供。
    """
    diag_dir = Path(diag_dir)

    if boundary_type == "healthy":
        diag_json = diag_dir / "hole_diagnosis.json"
        if not diag_json.exists():
            raise FileNotFoundError(f"缺少诊断文件: {diag_json}")
        with open(diag_json, "r", encoding="utf-8") as f:
            diagnosis = json.load(f)
        return [int(h["hole_id"]) for h in diagnosis.get("healthy_holes", [])]

    elif boundary_type == "uncovered":
        comp_json = diag_dir / "uncovered_component_analysis.json"
        if not comp_json.exists():
            raise FileNotFoundError(f"缺少未覆盖分量分析文件: {comp_json}")
        with open(comp_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [int(c["component_id"]) for c in data.get("components", [])]

    else:
        raise ValueError(f"未知的 boundary_type: {boundary_type}")


def extract_component_by_id(
    mesh,
    input_file,
    diag_dir,
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
    try:
        comp = load_boundary_component_data(
            str(diag_dir),
            boundary_id,
            boundary_type,
        )
    except Exception as e:
        return False, 0, 0, f"加载组件失败: {e}"

    stem = f"{boundary_type}_{boundary_id}_depth{neighborhood_depth}"
    ply_path = Path(output_dir) / f"{stem}.ply"
    json_path = Path(output_dir) / f"{stem}.json"

    source_file = str(Path(input_file).resolve())

    result = export_component_package(
        mesh,
        comp,
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

    # 仅枚举组件 ID；实际提取走 load_boundary_component_data，
    # 保证与 meshinspect.py --extract-component-package 完全一致。
    try:
        ids = _enumerate_component_ids(args.hole_diagnosis_dir, args.boundary_type)
    except Exception as e:
        print(f"[ERROR] 枚举组件 ID 失败: {e}")
        return

    if not ids:
        print("没有找到任何组件。")
        return

    # 按 ID 过滤
    if args.ids is not None:
        id_set = set(int(x) for x in args.ids.split(",") if x.strip())
        ids = [i for i in ids if i in id_set]

    # 限制数量
    if args.max_components is not None:
        ids = ids[: args.max_components]

    success_count = 0
    fail_count = 0

    for boundary_id in ids:
        success, face_cnt, vert_cnt, msg = extract_component_by_id(
            mesh,
            args.input_file,
            args.hole_diagnosis_dir,
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
