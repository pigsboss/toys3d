# src/toys3d/meshrepair.py
"""
网格修复工具集。

当前主要功能：
- 健康孔洞的 Seifert 极小曲面（固定边界）生成
- Seifert 填充前后局部拓扑统计
- Seifert 曲面曲率统计
"""

import numpy as np
import trimesh
import json
import argparse
from pathlib import Path

from toys3d.geometrics import analyze_mesh_defects
from toys3d.geometrics.generation import (
    generate_seifert_surface,
    compute_seifert_fill_stats,
    print_seifert_fill_stats,
    compute_seifert_curvature_stats,
    apply_seifert_patch_to_mesh,
    repair_healthy_hole,
    repair_all_healthy_holes,
)


# ---------------------------------------------------------------------------
# 全局网格统计（与 Seifert 功能无关）
# ---------------------------------------------------------------------------

def compute_global_mesh_stats(mesh):
    """
    统计网格全局边/面属性。
    """
    defect_stats, open_face_mask, nonmanifold_face_mask = analyze_mesh_defects(mesh)

    total_edges = len(mesh.edges_unique)
    open_edges = defect_stats["open_edges"]
    nonmanifold_edges = defect_stats["nonmanifold_edges"]
    manifold_edges = total_edges - open_edges - nonmanifold_edges

    total_faces = len(mesh.faces)
    open_faces = int(open_face_mask.sum())
    nonmanifold_faces = int(nonmanifold_face_mask.sum())
    union_mask = open_face_mask | nonmanifold_face_mask
    manifold_faces = total_faces - int(union_mask.sum())

    return {
        "total_edges": total_edges,
        "open_edges": open_edges,
        "manifold_edges": manifold_edges,
        "nonmanifold_edges": nonmanifold_edges,
        "total_faces": total_faces,
        "open_faces": open_faces,
        "manifold_faces": manifold_faces,
        "nonmanifold_faces": nonmanifold_faces,
    }


def print_global_mesh_stats(label, stats):
    print(f"  {label}:")
    print(f"    开放边:     {stats['open_edges']}")
    print(f"    流形边:     {stats['manifold_edges']}")
    print(f"    非流形边:   {stats['nonmanifold_edges']}")
    print(f"    总边数:     {stats['total_edges']}")
    print(f"    开放面片:   {stats['open_faces']}")
    print(f"    流形面片:   {stats['manifold_faces']}")
    print(f"    非流形面片: {stats['nonmanifold_faces']}")
    print(f"    总面片数:   {stats['total_faces']}")


# ---------------------------------------------------------------------------
# 组件包加载辅助
# ---------------------------------------------------------------------------

def _load_component_package(input_file, component_package_arg):
    """
    加载组件包 JSON：
    - 显式给出 --component-package 时，读取该路径；
    - 否则，自动查找与输入 PLY 同名的 .json 文件；
    - 若文件不存在或不含 component.healthy_hole_vertex_indices，返回 (None, None)。

    返回 (data, path) 或 (None, None)。
    """
    if component_package_arg is not None:
        path = Path(component_package_arg)
        if not path.exists():
            raise FileNotFoundError(f"未找到组件包: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f), path

    input_path = Path(input_file)
    auto_path = input_path.with_suffix(".json")
    if not auto_path.exists():
        return None, None

    try:
        with open(auto_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None, None

    comp = data.get("component", {})
    if "healthy_hole_vertex_indices" not in comp:
        return None, None
    return data, auto_path


def _build_hole_from_component(component_data, override_hole_id=None):
    """
    将组件包转换为 repair_healthy_hole 期望的 hole 字典。
    """
    comp = component_data.get("component", {})
    boundary_type = component_data.get("boundary_type", "healthy")
    if boundary_type != "healthy":
        raise ValueError(
            f"组件包边界类型不是 healthy，而是 {boundary_type}"
        )

    verts = comp.get("healthy_hole_vertex_indices", [])
    if not verts:
        raise ValueError("组件包中 healthy_hole_vertex_indices 为空")

    boundary_id = int(component_data.get("boundary_id", 0))
    if override_hole_id is not None and override_hole_id != boundary_id:
        print(
            f"  [WARN] --hole-id={override_hole_id} 与组件包 "
            f"boundary_id={boundary_id} 不一致，将使用组件包中的 ID"
        )

    vert_list = [int(v) for v in verts]
    return {
        "hole_id": boundary_id,
        "vertex_indices": vert_list,
        "num_vertices": len(vert_list),
        "num_edges": len(vert_list),
    }


# ---------------------------------------------------------------------------
# 命令行入口
# ---------------------------------------------------------------------------

def _main():
    parser = argparse.ArgumentParser(
        description="网格修复工具：对健康孔洞生成并合并 Seifert 极小曲面。"
    )
    parser.add_argument("input_file", help="输入网格文件 (ply/stl/obj)")
    parser.add_argument("output_file", help="输出修补后的网格文件")
    parser.add_argument(
        "--hole-diagnosis-dir",
        default="hole_diagnosis_report",
        help="hole diagnosis 输出目录（默认 hole_diagnosis_report）",
    )
    parser.add_argument(
        "--component-package",
        type=str,
        default=None,
        help="组件包 JSON 路径。默认会自动查找与输入 PLY 同名的 .json 文件。",
    )
    parser.add_argument(
        "--hole-id",
        type=int,
        default=None,
        help="指定修补的健康孔洞 ID；不指定则修补所有健康孔洞",
    )
    parser.add_argument(
        "--seifert-optimize-iterations",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--seifert-step-size",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--seifert-tolerance",
        type=float,
        default=1e-7,
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="打印详细过程",
    )

    args = parser.parse_args()

    print(f"Hey! Loading {args.input_file}")
    mesh = trimesh.load(args.input_file, force="mesh")

    comp_data, comp_path = _load_component_package(
        args.input_file, args.component_package
    )

    if comp_data is not None:
        print(f"自动找到组件包: {comp_path}")
        try:
            hole = _build_hole_from_component(comp_data, args.hole_id)
        except Exception as e:
            raise RuntimeError(f"解析组件包失败: {e}")

        healthy_holes = [hole]
        component_mode = True
    else:
        diag_path = Path(args.hole_diagnosis_dir) / "hole_diagnosis.json"
        if not diag_path.exists():
            raise FileNotFoundError(f"未找到 {diag_path}")

        with open(diag_path, "r", encoding="utf-8") as f:
            diag = json.load(f)

        healthy_holes = diag.get("healthy_holes", [])
        component_mode = False

    if not healthy_holes:
        print("未找到健康孔洞，无需修补。")
        mesh.export(args.output_file)
        return

    seifert_options = {
        "optimize_iterations": args.seifert_optimize_iterations,
        "step_size": args.seifert_step_size,
        "tol": args.seifert_tolerance,
        "verbose": args.verbose,
    }

    print("\n修补前网格统计:")
    before_stats = compute_global_mesh_stats(mesh)
    print_global_mesh_stats("修补前", before_stats)
    print(f"  健康孔洞总数: {len(healthy_holes)}")

    if component_mode:
        hole = healthy_holes[0]
        repaired_mesh, msg = repair_healthy_hole(
            mesh, hole, seifert_options, verbose=args.verbose
        )
        if repaired_mesh is None:
            raise RuntimeError(f"修补失败: {msg}")
        repaired_ids = [hole["hole_id"]]
        failed_records = []

    elif args.hole_id is not None:
        hole = next(
            (h for h in healthy_holes if h["hole_id"] == args.hole_id),
            None,
        )
        if hole is None:
            raise ValueError(f"未找到 hole_id={args.hole_id} 的健康孔洞")
        repaired_mesh, msg = repair_healthy_hole(
            mesh, hole, seifert_options, verbose=args.verbose
        )
        if repaired_mesh is None:
            raise RuntimeError(f"修补失败: {msg}")
        repaired_ids = [args.hole_id]
        failed_records = []

    else:
        repaired_mesh, repaired_ids, failed_records = repair_all_healthy_holes(
            mesh, healthy_holes, seifert_options, verbose=args.verbose
        )

    print("\n修补后网格统计:")
    after_stats = compute_global_mesh_stats(repaired_mesh)
    print_global_mesh_stats("修补后", after_stats)

    print(f"\n成功修补孔洞: {repaired_ids}")
    print(f"剩余健康孔洞: {len(healthy_holes) - len(repaired_ids)}")
    if failed_records:
        print("失败记录:")
        for rec in failed_records:
            print(f"  hole_id={rec['hole_id']}: {rec['message']}")

    repaired_mesh.export(args.output_file)
    print(f"\n修补后网格已保存: {args.output_file}")


if __name__ == "__main__":
    _main()
