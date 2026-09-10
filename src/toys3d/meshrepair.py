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
import sys
from pathlib import Path

from toys3d.geometrics import analyze_mesh_defects, polygon_area_from_3d_ccw
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


def _compute_input_features(mesh, vertex_indices):
    """
    计算健康孔洞边界的几何与形状特征，用于失败原因分析。
    """
    verts = np.asarray(vertex_indices, dtype=np.int64)
    pts = mesh.vertices[verts]
    n = len(pts)
    if n < 3:
        return {"boundary_vertex_count": int(n)}

    edges = np.roll(pts, -1, axis=0) - pts
    edge_lengths = np.linalg.norm(edges, axis=1)
    perimeter = float(edge_lengths.sum())

    centroid = pts.mean(axis=0)
    centered = pts - centroid
    try:
        _, sv, vh = np.linalg.svd(centered, full_matrices=False)
    except Exception:
        _, sv, vh = np.linalg.svd(centered)
    planarity = float(sv[2] / (sv[0] + 1e-12))

    try:
        area_3d = float(polygon_area_from_3d_ccw(pts))
    except Exception:
        area_3d = 0.0

    projected_area = 0.0
    self_intersect = None
    try:
        u, v = vh[0], vh[1]
        flat = np.column_stack([centered @ u, centered @ v])
        from shapely.geometry import Polygon as _ShapelyPolygon
        poly = _ShapelyPolygon(flat)
        self_intersect = not poly.is_valid
        if poly.is_valid:
            projected_area = float(poly.area)
    except Exception:
        pass

    incoming = pts - np.roll(pts, 1, axis=0)
    outgoing = np.roll(pts, -1, axis=0) - pts
    in_norm = np.linalg.norm(incoming, axis=1, keepdims=True) + 1e-12
    out_norm = np.linalg.norm(outgoing, axis=1, keepdims=True) + 1e-12
    cos_turn = np.clip(
        np.sum((incoming / in_norm) * (outgoing / out_norm), axis=1),
        -1.0, 1.0,
    )
    turn_angles = np.degrees(np.arccos(cos_turn))

    return {
        "boundary_vertex_count": int(n),
        "perimeter": perimeter,
        "area_3d": area_3d,
        "projected_area": projected_area,
        "area_ratio_3d_over_2d": (
            area_3d / projected_area if projected_area > 1e-12 else None
        ),
        "planarity_ratio": planarity,
        "bbox_extents": (pts.max(axis=0) - pts.min(axis=0)).tolist(),
        "edge_length_min": float(edge_lengths.min()),
        "edge_length_max": float(edge_lengths.max()),
        "edge_length_ratio": float(
            edge_lengths.max() / (edge_lengths.min() + 1e-12)
        ),
        "turn_angle_max_deg": float(turn_angles.max()),
        "turn_angle_mean_deg": float(turn_angles.mean()),
        "self_intersecting_projection": self_intersect,
    }


def _compute_patch_stats(before_mesh, after_mesh):
    """
    比较修补前后网格，统计新增补丁面片、面积与曲率。
    """
    before_keys = set()
    for face in before_mesh.faces:
        key = tuple(sorted(int(x) for x in face))
        before_keys.add(key)

    patch_face_indices = []
    for i, face in enumerate(after_mesh.faces):
        key = tuple(sorted(int(x) for x in face))
        if key not in before_keys:
            patch_face_indices.append(i)

    result = {
        "new_face_count": int(len(patch_face_indices)),
        "new_vertex_count": int(len(after_mesh.vertices) - len(before_mesh.vertices)),
        "patch_area": 0.0,
    }

    if not patch_face_indices:
        return result

    idx = np.asarray(patch_face_indices, dtype=np.int64)
    result["patch_area"] = float(after_mesh.area_faces[idx].sum())

    try:
        patch_mesh = after_mesh.submesh([idx], append=True)
        try:
            curv = compute_seifert_curvature_stats(patch_mesh)
            if isinstance(curv, dict):
                result["curvature"] = curv
        except Exception as e:
            result["curvature_error"] = str(e)
    except Exception as e:
        result["patch_submesh_error"] = str(e)

    return result


def _classify_failure_stage(message):
    if not message:
        return "unknown"
    msg = str(message)
    if "过于复杂" in msg or "初始" in msg:
        return "initial_disk"
    if "优化" in msg:
        return "optimize_minimal_surface"
    if "验证" in msg:
        return "validate_patch"
    if "解析" in msg:
        return "parse_component"
    if "加载" in msg or "Loading" in msg:
        return "input_load"
    return "unknown"


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
        "--status-json",
        type=str,
        default=None,
        help="状态 JSON 输出路径；默认 <output_file>.status.json",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="打印详细过程",
    )

    args = parser.parse_args()

    status_path = (
        Path(args.status_json)
        if args.status_json
        else Path(args.output_file).with_suffix(".status.json")
    )
    status_path.parent.mkdir(parents=True, exist_ok=True)

    status = {
        "input_file": str(Path(args.input_file).resolve()),
        "input_stem": Path(args.input_file).stem,
        "output_file": str(Path(args.output_file).resolve()),
        "status_json": str(status_path.resolve()),
        "component_package": None,
        "boundary_type": None,
        "hole_id": None,
        "method": "seifert_minimal_surface",
        "success": False,
        "message": "",
        "failure_stage": None,
        "seifert_options": None,
        "input_features": None,
        "before_stats": None,
        "after_stats": None,
        "patch": None,
    }

    def _write_and_exit(code):
        try:
            status_path.write_text(
                json.dumps(status, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as e:
            print(f"[WARN] 状态 JSON 写入失败: {e}")
        sys.exit(code)

    try:
        print(f"Hey! Loading {args.input_file}")
        mesh = trimesh.load(args.input_file, force="mesh")

        comp_data, comp_path = _load_component_package(
            args.input_file, args.component_package
        )

        if comp_data is not None:
            print(f"自动找到组件包: {comp_path}")
            status["component_package"] = str(Path(comp_path).resolve())
            status["boundary_type"] = comp_data.get("boundary_type", "healthy")
            try:
                status["hole_id"] = int(comp_data.get("boundary_id", 0))
            except Exception:
                status["hole_id"] = None

            try:
                hole = _build_hole_from_component(comp_data, args.hole_id)
            except Exception as e:
                status["message"] = f"解析组件包失败: {e}"
                status["failure_stage"] = "parse_component"
                print(f"[ERROR] {status['message']}")
                _write_and_exit(1)

            healthy_holes = [hole]
            component_mode = True
        else:
            diag_path = Path(args.hole_diagnosis_dir) / "hole_diagnosis.json"
            if not diag_path.exists():
                status["message"] = f"未找到 {diag_path}"
                status["failure_stage"] = "input_load"
                print(f"[ERROR] {status['message']}")
                _write_and_exit(1)

            with open(diag_path, "r", encoding="utf-8") as f:
                diag = json.load(f)

            healthy_holes = diag.get("healthy_holes", [])
            component_mode = False

        if not healthy_holes:
            print("未找到健康孔洞，无需修补。")
            mesh.export(args.output_file)
            status["success"] = True
            status["message"] = "无健康孔洞，直接复制输入网格"
            _write_and_exit(0)

        if component_mode:
            try:
                status["input_features"] = _compute_input_features(
                    mesh, healthy_holes[0]["vertex_indices"]
                )
            except Exception as e:
                status["input_features"] = {"error": str(e)}

        seifert_options = {
            "optimize_iterations": args.seifert_optimize_iterations,
            "step_size": args.seifert_step_size,
            "tol": args.seifert_tolerance,
            "verbose": args.verbose,
        }
        status["seifert_options"] = {
            "optimize_iterations": args.seifert_optimize_iterations,
            "step_size": args.seifert_step_size,
            "tol": args.seifert_tolerance,
        }

        print("\n修补前网格统计:")
        before_stats = compute_global_mesh_stats(mesh)
        print_global_mesh_stats("修补前", before_stats)
        status["before_stats"] = before_stats
        print(f"  健康孔洞总数: {len(healthy_holes)}")

        if component_mode:
            hole = healthy_holes[0]
            repaired_mesh, msg = repair_healthy_hole(
                mesh, hole, seifert_options, verbose=args.verbose
            )
            if repaired_mesh is None:
                status["message"] = str(msg) if msg else "修补失败"
                status["failure_stage"] = _classify_failure_stage(msg)
                print(f"[ERROR] {status['message']}")
                _write_and_exit(1)
            repaired_ids = [hole["hole_id"]]
            failed_records = []

        elif args.hole_id is not None:
            hole = next(
                (h for h in healthy_holes if h["hole_id"] == args.hole_id),
                None,
            )
            if hole is None:
                status["message"] = f"未找到 hole_id={args.hole_id}"
                status["failure_stage"] = "parse_component"
                print(f"[ERROR] {status['message']}")
                _write_and_exit(1)

            repaired_mesh, msg = repair_healthy_hole(
                mesh, hole, seifert_options, verbose=args.verbose
            )
            if repaired_mesh is None:
                status["message"] = str(msg) if msg else "修补失败"
                status["failure_stage"] = _classify_failure_stage(msg)
                print(f"[ERROR] {status['message']}")
                _write_and_exit(1)
            repaired_ids = [args.hole_id]
            failed_records = []

        else:
            repaired_mesh, repaired_ids, failed_records = repair_all_healthy_holes(
                mesh, healthy_holes, seifert_options, verbose=args.verbose
            )

        print("\n修补后网格统计:")
        after_stats = compute_global_mesh_stats(repaired_mesh)
        print_global_mesh_stats("修补后", after_stats)
        status["after_stats"] = after_stats

        try:
            status["patch"] = _compute_patch_stats(mesh, repaired_mesh)
        except Exception as e:
            status["patch"] = {"error": str(e)}

        print(f"\n成功修补孔洞: {repaired_ids}")
        if failed_records:
            print("失败记录:")
            for rec in failed_records:
                print(f"  hole_id={rec['hole_id']}: {rec['message']}")

        repaired_mesh.export(args.output_file)
        print(f"\n修补后网格已保存: {args.output_file}")

        status["success"] = True
        status["message"] = "修补成功"
        _write_and_exit(0)

    except SystemExit:
        raise
    except Exception as e:
        if not status["message"]:
            status["message"] = f"{type(e).__name__}: {e}"
        if status["failure_stage"] is None:
            status["failure_stage"] = "exception"
        print(f"[ERROR] {status['message']}")
        _write_and_exit(1)


if __name__ == "__main__":
    _main()
