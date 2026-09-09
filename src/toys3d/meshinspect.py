# src/toys3d/meshinspect.py
import sys
import os

# Ensure src directory is on the path so that 'toys3d' can be imported
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_src_parent = os.path.dirname(_project_root)
if _src_parent not in sys.path:
    sys.path.insert(0, _src_parent)

# If the current working directory contains an `inspect.py`, it would shadow
# the standard library module and cause circular imports (e.g., in NumPy).
if os.path.exists(os.path.join(os.getcwd(), 'inspect.py')):
    if '' in sys.path:
        sys.path.remove('')

import argparse
import json
from pathlib import Path
import numpy as np
import trimesh
from collections import deque, Counter
from scipy.sparse import csr_matrix

import matplotlib
matplotlib.use('Agg')  # 无头模式，避免弹出窗口
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D

from toys3d.geometrics import (
    compute_mesh_stats,
    analyze_mesh_defects,
    compute_hole_area_stats,
    extract_boundary_loops,
    polygon_area_from_3d_ccw,
    repair_mesh_by_removing_duplicates,
    project_vertices_to_shell,
    weld_small_holes,
    compute_vertex_face_counts,
    build_vertex_face_csr,
    compute_face_edge_types,
    compute_face_topology_codes,
    compute_edge_to_faces,
    compute_face_edge_keys,
    compute_face_edge_valences,
    compute_class_neighbor_stats,
    compute_single_face_neighbor_stats,
    get_face_topology_code_and_order,
    code_to_hex,
    hex_to_code,
    save_codes,
    group_faces_by_topology_codes,
    build_hole_diagnosis_data,
    analyze_uncovered_open_edge_components,
    build_manifold_face_adjacency,
    is_manifold_closed_boundary,
    find_minimal_enclosing_manifold_boundary_greedy,
    fit_watertight_patch_from_component,
    compute_face_area_stats,
    compute_bounding_box_stats,
    compute_volume_if_closed,
    expand_face_neighborhood,
    compute_face_distances,
    point_in_polygon_2d,
    normalize,
    extract_component_submesh,
)

from toys3d.reporting import (
    _generate_component_3d_diagram,
    generate_html_report_from_json,
    load_boundary_component_data,
)

from toys3d.visualization import (
    _parse_color_string,
    _parse_color_string_flexible,
    _print_boundary_component_diagnostics,
    _show_scene_with_camera_info,
    add_boundary_projection_to_scene,
    add_hole_boundaries_to_scene,
    add_proxy_overlay_to_scene as _add_proxy_overlay_to_scene,
    add_uncovered_edges_to_scene,
    add_wireframe_to_scene,
    build_defect_visualization,
    build_reliable_visualization,
    load_proxy_mesh as _load_proxy_mesh,
    make_double_sided,
    set_face_alpha,
    visualize_boundary_component,
)


def load_proxy_mesh(args):
    return _load_proxy_mesh(args.overlay_proxy)


def add_proxy_overlay_to_scene(scene, args, proxy):
    return _add_proxy_overlay_to_scene(
        scene,
        proxy,
        color=args.proxy_color,
        alpha=args.proxy_alpha,
        double_sided=args.proxy_double_sided,
    )


def build_reliable_visualization(mesh, distances, min_distance):
    """Compatibility wrapper; the actual implementation lives in visualization.py."""
    import toys3d.visualization as _vis
    return _vis.build_reliable_visualization(mesh, distances, min_distance)


def build_defect_visualization(mesh, open_face_mask, nonmanifold_face_mask):
    """Compatibility wrapper; implementation is in visualization.py."""
    import toys3d.visualization as _vis
    return _vis.build_defect_visualization(mesh, open_face_mask, nonmanifold_face_mask)


def make_double_sided(mesh, backface_color=None):
    """Compatibility wrapper for visualization.make_double_sided."""
    import toys3d.visualization as _vis
    return _vis.make_double_sided(mesh, backface_color)


def set_face_alpha(mesh, alpha):
    """Compatibility wrapper for visualization.set_face_alpha."""
    import toys3d.visualization as _vis
    return _vis.set_face_alpha(mesh, alpha)


def add_wireframe_to_scene(scene, mesh, color=None, radius=None):
    """Compatibility wrapper for visualization.add_wireframe_to_scene."""
    import toys3d.visualization as _vis
    return _vis.add_wireframe_to_scene(scene, mesh, color, radius)


def add_hole_boundaries_to_scene(scene, mesh, radius=None,
                                 min_edges=3, min_area=0.0, verbose=False):
    """Compatibility wrapper for visualization.add_hole_boundaries_to_scene."""
    import toys3d.visualization as _vis
    return _vis.add_hole_boundaries_to_scene(scene, mesh, radius, min_edges, min_area, verbose)


def add_boundary_projection_to_scene(scene, boundary_mesh, proxy_mesh,
                                     radius=None, verbose=False,
                                     print_enclosed_vertices=False,
                                     max_report_loops=None,
                                     max_report_vertices=20):
    """Compatibility wrapper for visualization.add_boundary_projection_to_scene."""
    import toys3d.visualization as _vis
    return _vis.add_boundary_projection_to_scene(
        scene, boundary_mesh, proxy_mesh, radius, verbose,
        print_enclosed_vertices, max_report_loops, max_report_vertices
    )


def add_uncovered_edges_to_scene(scene, mesh, data_dir, radius=None, verbose=False):
    """Compatibility wrapper for visualization.add_uncovered_edges_to_scene."""
    import toys3d.visualization as _vis
    return _vis.add_uncovered_edges_to_scene(scene, mesh, data_dir, radius, verbose)


def _show_scene_with_camera_info(scene, args, scene_translation=None):
    """Compatibility wrapper for visualization._show_scene_with_camera_info."""
    import toys3d.visualization as _vis
    return _vis._show_scene_with_camera_info(scene, args, scene_translation)


def print_scene_debug_info(scene, title="Scene Debug Info"):
    """Compatibility wrapper for visualization.print_scene_debug_info."""
    import toys3d.visualization as _vis
    return _vis.print_scene_debug_info(scene, title)


def _print_camera_info(info):
    """Compatibility wrapper for visualization._print_camera_info."""
    import toys3d.visualization as _vis
    return _vis._print_camera_info(info)


def _capture_vedo_camera_info(plotter_or_viewer):
    """Compatibility wrapper."""
    import toys3d.visualization as _vis
    return _vis._capture_vedo_camera_info(plotter_or_viewer)


def _try_get_trimesh_vedo_viewer():
    """Compatibility wrapper."""
    import toys3d.visualization as _vis
    return _vis._try_get_trimesh_vedo_viewer()


def _show_scene_with_vedo(scene):
    """Compatibility wrapper."""
    import toys3d.visualization as _vis
    return _vis._show_scene_with_vedo(scene)


def _filter_camera_core_points(points):
    """Compatibility wrapper."""
    import toys3d.visualization as _vis
    return _vis._filter_camera_core_points(points)


def _generate_topology_diagram(code, output_path):
    """Compatibility wrapper delegating to reporting._generate_topology_diagram."""
    from toys3d.reporting import _generate_topology_diagram as _rep
    return _rep(code, output_path)


def _generate_component_3d_diagram(component, mesh, output_path):
    """Compatibility wrapper delegating to reporting function."""
    from toys3d.reporting import _generate_component_3d_diagram as _rep
    return _rep(component, mesh, output_path)


def load_uncovered_edge_data(data_dir):
    """Compatibility wrapper delegating to reporting."""
    from toys3d.reporting import load_uncovered_edge_data as _rep
    return _rep(data_dir)


def generate_html_report(output_dir, id_to_code, results):
    """Compatibility wrapper delegating to reporting.generate_html_report."""
    from toys3d.reporting import generate_html_report as _rep
    return _rep(output_dir, id_to_code, results)


def generate_latex_report(output_dir, id_to_code, results):
    """Compatibility wrapper delegating to reporting.generate_latex_report."""
    from toys3d.reporting import generate_latex_report as _rep
    return _rep(output_dir, id_to_code, results)


def generate_html_report_from_json(output_dir):
    """Compatibility wrapper delegating to reporting.generate_html_report_from_json."""
    from toys3d.reporting import generate_html_report_from_json as _rep
    return _rep(output_dir)


def _parse_color_string(s):
    """Compatibility wrapper for visualization._parse_color_string."""
    import toys3d.visualization as _vis
    return _vis._parse_color_string(s)


def _parse_color_string_flexible(s):
    """Compatibility wrapper for visualization._parse_color_string_flexible."""
    import toys3d.visualization as _vis
    return _vis._parse_color_string_flexible(s)


def _print_boundary_component_diagnostics(mesh, comp, boundary_type, boundary_id,
                                         neighborhood_depth, print_distribution=False):
    """Compatibility wrapper for visualization._print_boundary_component_diagnostics."""
    import toys3d.visualization as _vis
    return _vis._print_boundary_component_diagnostics(
        mesh, comp, boundary_type, boundary_id, neighborhood_depth, print_distribution
    )


def visualize_boundary_component(mesh, args):
    """Compatibility wrapper for visualization.visualize_boundary_component."""
    import toys3d.visualization as _vis
    return _vis.visualize_boundary_component(mesh, args)


def extract_component_package(mesh, args):
    comp = load_boundary_component_data(
        args.boundary_data_dir,
        args.boundary_id,
        args.boundary_type,
    )

    _print_boundary_component_diagnostics(
        mesh,
        comp,
        args.boundary_type,
        args.boundary_id,
        args.boundary_neighborhood_depth,
        print_distribution=args.print_neighborhood_distribution,
    )

    local_mesh, _faces_idx, _old_to_new, comp_new = extract_component_submesh(
        mesh,
        comp,
        neighborhood_depth=args.boundary_neighborhood_depth,
    )

    if local_mesh is None:
        print("[ERROR] 没有可提取的面片")
        return

    local_vertices = local_mesh.vertices
    local_faces = local_mesh.faces

    input_stem = Path(args.input_file).stem
    ply_path = Path(
        args.component_output
        or f"{input_stem}_component_{args.boundary_id}.ply"
    )
    json_path = Path(
        args.component_json
        or f"{input_stem}_component_{args.boundary_id}.json"
    )

    local_mesh.export(ply_path)
    print(f"局部网格已保存: {ply_path}")

    package_data = {
        "source_file": str(Path(args.input_file).resolve()),
        "boundary_type": args.boundary_type,
        "boundary_id": args.boundary_id,
        "neighborhood_depth": args.boundary_neighborhood_depth,
        "local_vertex_count": int(len(local_vertices)),
        "local_face_count": int(len(local_faces)),
        "component": comp_new,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(package_data, f, indent=2, ensure_ascii=False)
    print(f"组件数据已保存: {json_path}")

    print(
        f"提取完成：局部面片数 {len(local_faces)}，"
        f"局部顶点数 {len(local_vertices)}"
    )


def run_full_diagnosis_pass1(mesh, output_dir, valence_threshold=5):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Full Diagnosis Pass 1 ===")
    print("分析网格缺陷...")
    _, open_face_mask, nonmanifold_face_mask = analyze_mesh_defects(mesh)
    abnormal_mask = open_face_mask | nonmanifold_face_mask
    abnormal_indices = np.where(abnormal_mask)[0]

    print(f"异常面片总数: {len(abnormal_indices)}")

    vertex_face_counts = compute_vertex_face_counts(mesh)
    face_edge_types = compute_face_edge_types(mesh)

    # 计算所有面片的拓扑编码
    all_face_indices = np.arange(len(mesh.faces))
    codes_all, _, _ = compute_face_topology_codes(
        mesh, all_face_indices, vertex_face_counts, face_edge_types
    )
    save_codes(codes_all, output_dir / "face_codes.npy")

    if len(abnormal_indices) == 0:
        # 创建空的分类 JSON 和 checkpoint
        empty_classes = {
            "valence_threshold": valence_threshold,
            "total_abnormal_faces": 0,
            "classes": {}
        }
        with open(output_dir / "abnormal_truncated_classes.json", "w") as f:
            json.dump(empty_classes, f, indent=2)
        checkpoint = {
            "valence_threshold": valence_threshold,
            "classes": {},
            "total_classes": 0,
            "abnormal_count": 0
        }
        with open(output_dir / "checkpoint.json", "w") as f:
            json.dump(checkpoint, f, indent=2)
        return {}, abnormal_indices

    # 对异常面片进行截断聚类
    print("截断聚类异常面片...")
    grouped = group_faces_by_topology_codes(
        mesh, abnormal_indices, vertex_face_counts, face_edge_types,
        valence_threshold=valence_threshold
    )

    class_faces = {}
    classes_json = {}
    for key, face_list in grouped.items():
        hex_code = code_to_hex(key)
        class_faces[hex_code] = face_list
        classes_json[hex_code] = {
            "face_indices": face_list.tolist(),
            "count": int(len(face_list)),
            "status": "pending"
        }

    abnormal_data = {
        "valence_threshold": valence_threshold,
        "total_abnormal_faces": int(len(abnormal_indices)),
        "classes": classes_json
    }
    with open(output_dir / "abnormal_truncated_classes.json", "w") as f:
        json.dump(abnormal_data, f, indent=2)

    checkpoint = {
        "valence_threshold": valence_threshold,
        "classes": {hex_code: "pending" for hex_code in class_faces},
        "total_classes": len(class_faces),
        "abnormal_count": int(len(abnormal_indices))
    }
    with open(output_dir / "checkpoint.json", "w") as f:
        json.dump(checkpoint, f, indent=2)

    print(f"发现 {len(class_faces)} 个不同拓扑类别（截断）。")
    return class_faces, abnormal_indices


def run_full_diagnosis_pass2(mesh, output_dir, class_faces,
                             open_face_mask, nonmanifold_face_mask,
                             valence_threshold=5,
                             resume=False):
    output_dir = Path(output_dir)
    checkpoint_path = output_dir / "checkpoint.json"
    classes_json_path = output_dir / "abnormal_truncated_classes.json"

    if resume and checkpoint_path.exists() and classes_json_path.exists():
        with open(checkpoint_path, "r") as f:
            checkpoint = json.load(f)
        pending_classes = [hex_code for hex_code, status in checkpoint["classes"].items()
                           if status == "pending"]
        print(f"恢复模式：已完成 {len(checkpoint['classes']) - len(pending_classes)} 类，"
              f"剩余 {len(pending_classes)} 类")
    else:
        pending_classes = list(class_faces.keys())

    if not pending_classes:
        print("没有待分析类别。")
        return {}

    # 预计算共享数据
    vertex_faces_csr = build_vertex_face_csr(mesh)
    vertex_face_counts = compute_vertex_face_counts(mesh)
    edge_keys, edge_faces = compute_edge_to_faces(mesh)
    edge_to_faces = {int(k): v for k, v in zip(edge_keys, edge_faces)}
    face_edge_keys = compute_face_edge_keys(mesh)
    edge_valences_all = compute_face_edge_valences(mesh, edge_to_faces, face_edge_keys)

    results = {}
    for idx, hex_code in enumerate(pending_classes):
        face_indices = np.asarray(class_faces[hex_code], dtype=np.int64)
        print(f"\n=== 分析类别 {idx+1}/{len(pending_classes)} ===", flush=True)
        print(f"  编码: {hex_code}, 面片数: {len(face_indices)}", flush=True)

        areas = mesh.area_faces[face_indices]
        area_stats = {
            'count': int(len(face_indices)),
            'mean': float(np.mean(areas)),
            'min': float(np.min(areas)),
            'p1': float(np.percentile(areas, 1)),
            'p5': float(np.percentile(areas, 5)),
            'p10': float(np.percentile(areas, 10)),
            'p25': float(np.percentile(areas, 25)),
            'p50': float(np.percentile(areas, 50)),
            'p75': float(np.percentile(areas, 75)),
            'p90': float(np.percentile(areas, 90)),
            'p95': float(np.percentile(areas, 95)),
            'p99': float(np.percentile(areas, 99)),
            'max': float(np.max(areas)),
        }

        point_counts, edge_counts = compute_class_neighbor_stats(
            mesh, face_indices, open_face_mask, nonmanifold_face_mask,
            vertex_faces_csr, edge_to_faces, face_edge_keys
        )

        rep_face = int(face_indices[0])
        vertex_stats, edge_stats = compute_single_face_neighbor_stats(
            mesh, rep_face, open_face_mask, nonmanifold_face_mask,
            vertex_faces_csr, edge_to_faces, face_edge_keys
        )

        _, v_order, e_order = get_face_topology_code_and_order(
            mesh, rep_face, vertex_face_counts, edge_to_faces, face_edge_keys
        )

        aligned_vertex_stats = [vertex_stats[i] for i in v_order]
        aligned_edge_stats = [edge_stats[i] for i in e_order]

        # 截断字段的真实值分布
        verts = mesh.faces[face_indices]                # (k,3)
        v_counts = vertex_face_counts[verts]            # (k,3)
        e_counts = edge_valences_all[face_indices]      # (k,3)

        # 顶点元截断分布：只统计 >= valence_threshold 的字段
        truncated_v_mask = v_counts >= valence_threshold
        truncated_v_vals = v_counts[truncated_v_mask]
        truncated_v_dist = Counter(truncated_v_vals.tolist())

        # 边元截断分布：只统计 >= 3 的字段
        truncated_e_mask = e_counts >= 3
        truncated_e_vals = e_counts[truncated_e_mask]
        truncated_e_dist = Counter(truncated_e_vals.tolist())

        class_result = {
            'class_id': hex_code,
            'code': hex_code,
            'area_stats': area_stats,
            'point_counts': point_counts,
            'edge_counts': edge_counts,
            'representative_vertex_stats': aligned_vertex_stats,
            'representative_edge_stats': aligned_edge_stats,
            'truncated_vertex_valence_dist': dict(sorted(truncated_v_dist.items())),
            'truncated_edge_valence_dist': dict(sorted(truncated_e_dist.items())),
        }
        results[hex_code] = class_result

        # 更新 abnormal_truncated_classes.json
        with open(classes_json_path, "r") as f:
            data = json.load(f)
        if hex_code not in data["classes"]:
            # 兼容恢复时 dict 中可能没有该键
            data["classes"][hex_code] = {
                "face_indices": face_indices.tolist(),
                "count": len(face_indices)
            }
        data["classes"][hex_code]["area_stats"] = area_stats
        data["classes"][hex_code]["point_counts"] = point_counts
        data["classes"][hex_code]["edge_counts"] = edge_counts
        data["classes"][hex_code]["representative_vertex_stats"] = aligned_vertex_stats
        data["classes"][hex_code]["representative_edge_stats"] = aligned_edge_stats
        data["classes"][hex_code]["status"] = "done"

        data["classes"][hex_code]["truncated_vertex_valence_dist"] = \
            dict(sorted(truncated_v_dist.items()))
        data["classes"][hex_code]["truncated_edge_valence_dist"] = \
            dict(sorted(truncated_e_dist.items()))

        with open(classes_json_path, "w") as f:
            json.dump(data, f, indent=2)

        # 更新 checkpoint
        with open(checkpoint_path, "r") as f:
            ckpt = json.load(f)
        if hex_code not in ckpt["classes"]:
            ckpt["classes"][hex_code] = "pending"  # 兼容可能缺失
        ckpt["classes"][hex_code] = "done"
        with open(checkpoint_path, "w") as f:
            json.dump(ckpt, f, indent=2)

        print(f"  面积: 平均={area_stats['mean']:.6f}, p50={area_stats['p50']:.6f}", flush=True)
        print(f"  点邻: 流形={point_counts['normal']}, 开放={point_counts['open']}, "
              f"非流形={point_counts['nonmanifold']}", flush=True)
        print(f"  边邻: 流形={edge_counts['normal']}, 开放={edge_counts['open']}, "
              f"非流形={edge_counts['nonmanifold']}", flush=True)

    return results


def perform_full_diagnosis(mesh, args):
    """
    Full Diagnosis 入口，协调 Pass 1 和 Pass 2。
    """
    output_dir = Path(args.diagnosis_output)
    output_dir.mkdir(parents=True, exist_ok=True)

    _, open_face_mask, nonmanifold_face_mask = analyze_mesh_defects(mesh)

    resume_checkpoint = output_dir / "checkpoint.json"
    classes_json_path = output_dir / "abnormal_truncated_classes.json"

    if args.resume and resume_checkpoint.exists() and classes_json_path.exists():
        with open(classes_json_path, "r") as f:
            classes_data = json.load(f)
        class_faces = {hex_code: np.asarray(entry["face_indices"], dtype=np.int64)
                       for hex_code, entry in classes_data.get("classes", {}).items()}
        print("Resume: loading existing classifications from Pass 1.")
    else:
        class_faces, _ = run_full_diagnosis_pass1(
            mesh, output_dir,
            valence_threshold=args.valence_threshold
        )

    results = run_full_diagnosis_pass2(
        mesh, output_dir, class_faces,
        open_face_mask, nonmanifold_face_mask,
        valence_threshold=args.valence_threshold,
        resume=args.resume
    )

    if args.diagnosis_format == "html":
        generate_html_report_from_json(output_dir)
    else:
        # 新的架构中以 JSON 为单一数据源，暂未适配 LaTeX 生成，这里回退到 HTML
        print("[WARN] LaTeX report generation is not yet adapted to the new architecture.")
        print("Falling back to HTML report generation.")
        generate_html_report_from_json(output_dir)

    # 计算异常面总数（用于 meta；不再单独生成 diagnosis_results.json）
    abnormal_mask = open_face_mask | nonmanifold_face_mask
    abnormal_count = int(abnormal_mask.sum())
    print(f"\nFull Diagnosis 完成，异常面总数：{abnormal_count}，"
          f"报告已生成到 {output_dir}")


def perform_hole_diagnosis(mesh, args):
    """执行孔洞诊断，输出健康孔洞与未覆盖开放边分析。"""
    output_dir = Path(args.hole_diagnosis_output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Hole Diagnosis ===")
    print("提取开放边与健康孔洞...")
    hole_data = build_hole_diagnosis_data(mesh)

    # 1. 保存二进制数据
    print("保存开放边与孔洞数据...")
    open_edge_ids = np.arange(len(hole_data['open_edge_face_ids']), dtype=np.int64)
    np.savez_compressed(
        output_dir / "hole_diagnosis_data.npz",
        open_edge_ids=open_edge_ids,
        open_edge_vertex_pairs=hole_data['open_edge_vertex_pairs'],
        open_edge_face_ids=hole_data['open_edge_face_ids'],
        open_edge_keys=hole_data['open_edge_keys'],
        hole_ids_per_edge=hole_data['hole_ids_per_edge'],
        uncovered_edge_ids=hole_data['uncovered_edge_ids'],
        uncovered_category=hole_data['uncovered_category'],
    )

    # 2. 分析未覆盖开放边连通分量（异常孔洞）
    print("分析未覆盖开放边连通分量...")
    components = analyze_uncovered_open_edge_components(mesh, hole_data)
    component_json_path = output_dir / "uncovered_component_analysis.json"
    with open(component_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_components": len(components),
            "components": components,
        }, f, indent=2, ensure_ascii=False)
    print(f"未覆盖开放边分量分析已保存: {component_json_path}")

    # 计算最小包络流形边界（可选）
    if getattr(args, 'compute_enclosing_boundaries', False):
        print("计算最小包络流形边界...")
        for comp in components:
            comp_id = comp['component_id']
            bound = find_minimal_enclosing_manifold_boundary_greedy(mesh, comp)
            comp['minimal_enclosing_boundary'] = bound
            if bound['success']:
                print(f"  Component {comp_id}: 包络边界深度 {bound['depth']}, "
                      f"内部面片 {len(bound['enclosed_faces'])}, "
                      f"边界边 {len(bound['boundary_edges'])}")
            else:
                print(f"  Component {comp_id}: 未找到包络边界 "
                      f"(最大深度 {bound['depth']})")

        # 更新 JSON 文件
        with open(component_json_path, "w", encoding="utf-8") as f:
            json.dump({
                "total_components": len(components),
                "components": components,
            }, f, indent=2, ensure_ascii=False)
        print(f"已更新包含包络边界信息的: {component_json_path}")

    # 3. 构造 JSON 报告
    print("生成孔洞诊断 JSON...")
    total_open_edges = len(open_edge_ids)
    total_holes = len(hole_data['hole_vertex_lists'])
    uncovered_count = len(hole_data['uncovered_edge_ids'])
    covered_count = total_open_edges - uncovered_count

    hole_info_list = []
    for hole_id, (vert_list, edge_list) in enumerate(zip(
        hole_data['hole_vertex_lists'],
        hole_data['hole_edge_lists']
    )):
        # 计算周长和面积
        pts = mesh.vertices[np.asarray(vert_list, dtype=np.int64)]
        # 面积调用 polygon_area_from_3d_ccw
        area = polygon_area_from_3d_ccw(pts)
        perimeter = float(np.sum(np.linalg.norm(pts - np.roll(pts, -1, axis=0), axis=1)))
        hole_info_list.append({
            "hole_id": hole_id,
            "num_edges": len(edge_list),
            "num_vertices": len(vert_list),
            "area": area,
            "perimeter": perimeter,
            "vertex_indices": vert_list,
        })

    # 未覆盖开放边分类统计
    category_counts = Counter(hole_data['uncovered_category'].tolist())
    category_names = {
        0: "孤立开放链",
        1: "悬空开放边",
        2: "分支内部开放边",
        4: "非流形关联开放边",
        5: "其他复杂开放边",
    }
    uncovered_categories_summary = {
        category_names.get(int(cat), f"cat_{cat}"): int(cnt)
        for cat, cnt in category_counts.items()
    }

    diagnosis_json = {
        "total_open_edges": total_open_edges,
        "total_healthy_holes": total_holes,
        "covered_open_edges": covered_count,
        "uncovered_open_edges": uncovered_count,
        "healthy_holes": hole_info_list,
        "uncovered_categories_summary": uncovered_categories_summary,
        "open_edge_face_ids": hole_data['open_edge_face_ids'].tolist(),  # 可用于关联面片
    }
    with open(output_dir / "hole_diagnosis.json", "w", encoding="utf-8") as f:
        json.dump(diagnosis_json, f, indent=2, ensure_ascii=False)

    # 4. 生成 HTML 报告
    print("生成孔洞诊断 HTML...")
    html_path = output_dir / "hole_report.html"
    html = ["<html><head><meta charset='utf-8'><title>Hole Diagnosis</title>",
            "<style>body{font-family:sans-serif;margin:20px}",
            "table{border-collapse:collapse;width:100%;margin-bottom:20px}",
            "th,td{border:1px solid #ccc;padding:4px 8px;text-align:center}",
            "th{background:#f0f0f0}",
            ".component-block{margin-bottom:20px;border:1px solid #ddd;padding:10px}",
            "</style></head><body>",
            "<h1>Hole Diagnosis Report</h1>",
            f"<p>总开放边: {total_open_edges}</p>",
            f"<p>健康孔洞数: {total_holes}</p>",
            f"<p>覆盖开放边: {covered_count}</p>",
            f"<p>未覆盖开放边: {uncovered_count}</p>",
            "<h2>未覆盖开放边分类</h2>",
            "<table><tr><th>分类</th><th>数量</th></tr>"]
    for cat, cnt in uncovered_categories_summary.items():
        html.append(f"<tr><td>{cat}</td><td>{cnt}</td></tr>")
    html.append("</table>")
    html.append("<h2>健康孔洞概览</h2>")
    html.append("<table><tr><th>孔洞ID</th><th>边数</th><th>面积</th><th>周长</th></tr>")
    for hole in hole_info_list:
        html.append(f"<tr><td>{hole['hole_id']}</td><td>{hole['num_edges']}</td>"
                    f"<td>{hole['area']:.6f}</td><td>{hole['perimeter']:.6f}</td></tr>")
    html.append("</table>")
    html.append("<h2>未覆盖开放边组件分析</h2>")
    html.append("<table><tr><th>组件ID</th><th>边数</th><th>端点</th><th>分支点</th>"
                "<th>断裂候选</th></tr>")
    for comp in components:
        html.append(f"<tr><td>{comp['component_id']}</td><td>{comp['num_edges']}</td>"
                    f"<td>{len(comp['endpoints'])}</td><td>{len(comp['branch_vertices'])}</td>"
                    f"<td>{len(comp['candidate_breaks'])}</td></tr>")
    html.append("</table>")

    # 生成并嵌入每个组件的 3D 图
    component_diagrams_dir = output_dir / "component_diagrams"
    component_diagrams_dir.mkdir(exist_ok=True)

    html.append("<h2>未覆盖开放边组件三维视图</h2>")
    for comp in components:
        comp_id = comp['component_id']
        diagram_path = component_diagrams_dir / f"component_{comp_id}.svg"
        try:
            _generate_component_3d_diagram(comp, mesh, str(diagram_path))
        except Exception as e:
            print(f"  [WARN] Component {comp_id} 3D diagram failed: {e}")
            diagram_path = None

        if diagram_path and diagram_path.exists():
            svg_content = diagram_path.read_text(encoding="utf-8")
        else:
            svg_content = "<svg xmlns='http://www.w3.org/2000/svg' width='200' height='170'><text x='10' y='80'>Diagram not generated</text></svg>"

        html.append("<div class='component-block'>")
        html.append(f"<h3>组件 {comp_id}（边数 {comp['num_edges']}）</h3>")
        html.append("<div class='diagram-container'>")
        html.append(svg_content)
        html.append("</div>")
        html.append("<p>")
        html.append(f"顶点数: {comp['num_vertices']} | 端点: {len(comp['endpoints'])} | "
                    f"分支点: {len(comp['branch_vertices'])} | 断裂候选: {len(comp['candidate_breaks'])}")
        html.append("</p>")
        html.append("</div>")

    html.append("</body></html>")
    html_path.write_text("\n".join(html), encoding="utf-8")
    print(f"孔洞诊断 HTML 报告已保存: {html_path}")

    print(f"\nHole Diagnosis 完成，健康孔洞：{total_holes}，未覆盖开放边：{uncovered_count}，"
          f"报告已生成到 {output_dir}")


def print_separator(title=None):
    if title:
        print(f"\n{'=' * 60}")
        print(f" {title}")
        print(f"{'=' * 60}")
    else:
        print(f"\n{'=' * 60}")


def inspect_mesh(mesh, args):
    """
    主检查函数：输出统计信息并可选返回可视化场景。
    """
    if args.weld_small_holes:
        print_separator("Weld Small Holes")
        mesh = weld_small_holes(
            mesh,
            threshold=args.weld_hole_threshold,
            quantile=args.weld_hole_quantile,
            min_edges=args.weld_min_hole_edges,
            verbose=True,
        )

    stats = compute_mesh_stats(mesh)
    (
        defect_stats,
        open_face_mask,
        nonmanifold_face_mask,
        open_edge_per_face,
        manifold_edge_per_face,
        nonmanifold_edge_per_face,
    ) = analyze_mesh_defects(mesh, return_face_edge_counts=True)
    area_stats = compute_face_area_stats(mesh)
    bbox_stats = compute_bounding_box_stats(mesh)
    volume = compute_volume_if_closed(mesh)

    defect_mask = open_face_mask | nonmanifold_face_mask
    if np.any(defect_mask):
        distances = compute_face_distances(mesh, defect_mask)
        reliable_mask = (~defect_mask) & (distances >= args.reliable_distance)
    else:
        distances = np.full(len(mesh.faces), args.reliable_distance + 1, dtype=np.int32)
        reliable_mask = np.ones(len(mesh.faces), dtype=bool)

    print_separator("Mesh Topology")
    print(f"  vertices:          {stats['vertices']}")
    print(f"  faces:             {stats['faces']}")
    print(f"  edges (unique):    {stats['edges']}")
    print(f"  watertight:        {stats['is_watertight']}")
    print(f"  open edges:        {defect_stats['open_edges']}")
    print(f"  nonmanifold edges: {defect_stats['nonmanifold_edges']}")
    print(f"  open faces:        {defect_stats['open_faces']}")
    print(f"  nonmanifold faces: {defect_stats['nonmanifold_faces']}")

    hole_stats = compute_hole_area_stats(mesh)
    print(f"  open boundary loops: {hole_stats['count']}")
    print(f"  total hole area:     {hole_stats['total_area']:.6f}")
    if hole_stats['count'] > 0:
        print(f"  hole area percentiles: "
              f"p1={hole_stats['p1_area']:.6f}, "
              f"p5={hole_stats['p5_area']:.6f}, "
              f"p25={hole_stats['p25_area']:.6f}, "
              f"p50={hole_stats['p50_area']:.6f}, "
              f"p75={hole_stats['p75_area']:.6f}, "
              f"p90={hole_stats['p90_area']:.6f}, "
              f"p95={hole_stats['p95_area']:.6f}, "
              f"p99={hole_stats['p99_area']:.6f}, "
              f"max={hole_stats['max_area']:.6f}")

    print_separator("Defect Face Details")
    both_mask = open_face_mask & nonmanifold_face_mask
    open_only_mask = open_face_mask & ~nonmanifold_face_mask
    nonmanifold_only_mask = nonmanifold_face_mask & ~open_face_mask
    print(f"  open-only faces:                 {open_only_mask.sum()}")
    print(f"  nonmanifold-only faces:          {nonmanifold_only_mask.sum()}")
    print(f"  both open & nonmanifold faces:   {both_mask.sum()}")

    print_separator("Open Face Diagnostics")

    open_face_areas = mesh.area_faces[open_face_mask]
    if len(open_face_areas) == 0:
        print("  no open faces")
    else:
        print(f"  open face count: {len(open_face_areas)}")

        print("  face area percentiles:")
        for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
            print(f"    p{p}: {np.percentile(open_face_areas, p):.6f}")
        print(f"    mean: {open_face_areas.mean():.6f}")
        print(f"    min:  {open_face_areas.min():.6f}")
        print(f"    max:  {open_face_areas.max():.6f}")

        # 开放边数量分布（每个开放面片有几条边是开放边）
        open_edge_hist = np.bincount(open_edge_per_face[open_face_mask])
        print("  open-edge count per open face:")
        for k, cnt in enumerate(open_edge_hist):
            if cnt > 0:
                print(f"    {k} open edge(s): {cnt} faces")

        # 流形边数量分布
        manifold_edge_hist = np.bincount(manifold_edge_per_face[open_face_mask])
        print("  manifold-edge count per open face:")
        for k, cnt in enumerate(manifold_edge_hist):
            if cnt > 0:
                print(f"    {k} manifold edge(s): {cnt} faces")

        # 非流形边数量分布
        nonmanifold_edge_hist = np.bincount(nonmanifold_edge_per_face[open_face_mask])
        if nonmanifold_edge_hist.size > 1:
            print("  nonmanifold-edge count per open face:")
            for k, cnt in enumerate(nonmanifold_edge_hist):
                if cnt > 0:
                    print(f"    {k} nonmanifold edge(s): {cnt} faces")

    print_separator("Topological Reliability Distribution")
    max_display = 5
    for d in range(max_display + 1):
        cnt = int(np.sum(distances == d))
        print(f"  distance {d}: {cnt} faces")
    rest = int(np.sum(distances > max_display))
    print(f"  distance > {max_display}: {rest} faces")

    reliable_count = int(reliable_mask.sum())
    total_faces = len(mesh.faces)
    pct = 100.0 * reliable_count / max(total_faces, 1)
    print(f"  reliable faces (distance >= {args.reliable_distance}): "
          f"{reliable_count} ({pct:.2f}%)")

    print_separator("Bounding Box")
    print(f"  min:      [{bbox_stats['min'][0]:.4f}, "
          f"{bbox_stats['min'][1]:.4f}, {bbox_stats['min'][2]:.4f}]")
    print(f"  max:      [{bbox_stats['max'][0]:.4f}, "
          f"{bbox_stats['max'][1]:.4f}, {bbox_stats['max'][2]:.4f}]")
    print(f"  extents:  [{bbox_stats['extents'][0]:.4f}, "
          f"{bbox_stats['extents'][1]:.4f}, {bbox_stats['extents'][2]:.4f}]")
    print(f"  diagonal: {bbox_stats['diagonal']:.4f}")
    print(f"  centroid: [{bbox_stats['centroid'][0]:.4f}, "
          f"{bbox_stats['centroid'][1]:.4f}, {bbox_stats['centroid'][2]:.4f}]")

    print_separator("Edge Length Statistics")
    print(f"  mean: {stats['mean_edge_length']:.6f}")
    print(f"  p1:   {stats['edge_length_p1']:.6f}")
    print(f"  p5:   {stats['edge_length_p5']:.6f}")
    print(f"  p50:  {stats['edge_length_p50']:.6f}")
    print(f"  p95:  {stats['edge_length_p95']:.6f}")
    print(f"  p99:  {stats['edge_length_p99']:.6f}")

    print_separator("Face Area Statistics")
    print(f"  count: {area_stats['count']}")
    print(f"  mean:  {area_stats['mean']:.6f}")
    print(f"  min:   {area_stats['min']:.6f}")
    print(f"  p1:    {area_stats['p1']:.6f}")
    print(f"  p5:    {area_stats['p5']:.6f}")
    print(f"  p10:   {area_stats['p10']:.6f}")
    print(f"  p25:   {area_stats['p25']:.6f}")
    print(f"  p50:   {area_stats['p50']:.6f}")
    print(f"  p75:   {area_stats['p75']:.6f}")
    print(f"  p90:   {area_stats['p90']:.6f}")
    print(f"  p95:   {area_stats['p95']:.6f}")
    print(f"  p99:   {area_stats['p99']:.6f}")
    print(f"  max:   {area_stats['max']:.6f}")

    print_separator("Surface & Volume")
    print(f"  total surface area: {mesh.area:.6f}")
    if np.isfinite(volume):
        print(f"  volume (watertight): {volume:.6f}")
    else:
        print(f"  volume: N/A (mesh is not watertight)")

    # 可视化
    scene = None

    if len(mesh.faces) == 0 or len(mesh.vertices) == 0:
        if args.show or args.output:
            print("  Mesh is empty; visualization skipped.")
        return scene

    if args.show or args.output:
        # 提前加载代理网格，供叠加显示、投影和透明度控制使用
        proxy_mesh = load_proxy_mesh(args) if args.overlay_proxy else None

        if args.keep_reliable_only:
            print_separator("Reliable-Only Extracted Mesh")
            print(f"  min_distance: {args.reliable_distance}")

            # 使用之前计算好的可靠面片掩码
            reliable_count = int(reliable_mask.sum())
            print(f"  reliable faces: {reliable_count}/{len(mesh.faces)}")

            if reliable_count == 0:
                print("  No reliable faces selected; skipping extraction.")
                return None

            # 提取可靠面片子网格
            reliable_faces = np.asarray(mesh.faces, dtype=np.int64)[reliable_mask]
            flat_faces = reliable_faces.ravel()
            unique_verts, inverse = np.unique(flat_faces, return_inverse=True)
            reliable_mesh = trimesh.Trimesh(
                vertices=mesh.vertices[unique_verts],
                faces=inverse.reshape(-1, 3),
                process=False,
            )
            # 清理提取后的网格
            reliable_mesh.remove_unreferenced_vertices()
            reliable_mesh.merge_vertices()
            reliable_mesh = repair_mesh_by_removing_duplicates(reliable_mesh)

            if args.weld_small_holes:
                print_separator("Weld Small Holes (Reliable Mesh)")
                reliable_mesh = weld_small_holes(
                    reliable_mesh,
                    threshold=args.weld_hole_threshold,
                    quantile=args.weld_hole_quantile,
                    min_edges=args.weld_min_hole_edges,
                    verbose=True,
                )

            # 打印提取后的简要统计
            extracted_stats = compute_mesh_stats(reliable_mesh)
            extracted_defects, _, _ = analyze_mesh_defects(reliable_mesh)
            print(f"  extracted vertices: {extracted_stats['vertices']}")
            print(f"  extracted faces:    {extracted_stats['faces']}")
            print(f"  extracted open edges:        {extracted_defects['open_edges']}")
            print(f"  extracted nonmanifold edges: {extracted_defects['nonmanifold_edges']}")

            # 导出或显示
            if args.output:
                reliable_mesh.export(args.output)
                print(f"\nReliable-only mesh saved to: {args.output}")

            # 构造用于显示的双面网格，并设置输入网格透明度
            if args.double_sided:
                display_mesh = make_double_sided(
                    reliable_mesh,
                    backface_color=args.backface_color,
                )
            else:
                display_mesh = reliable_mesh

            if args.input_alpha < 1.0:
                set_face_alpha(display_mesh, args.input_alpha)

            scene = trimesh.Scene(display_mesh)

            # 若用户要求线框，可叠加在提取网格上
            if args.wireframe:
                add_wireframe_to_scene(
                    scene, reliable_mesh,
                    color=args.wireframe_color,
                    radius=args.wireframe_radius
                )

            # 若用户要求高亮孔洞，也可以基于提取网格显示
            if args.highlight_holes:
                add_hole_boundaries_to_scene(
                    scene, reliable_mesh,
                    radius=args.hole_radius,
                    min_edges=args.min_hole_edges,
                    min_area=args.min_hole_area,
                    verbose=True,
                )

            # 孔洞边界投影到代理网格
            if args.boundary_projection and proxy_mesh is not None:
                add_boundary_projection_to_scene(
                    scene, reliable_mesh, proxy_mesh,
                    radius=args.boundary_projection_radius,
                    verbose=True,
                    print_enclosed_vertices=args.print_shell_enclosed_vertices,
                    max_report_loops=args.max_report_boundary_loops,
                    max_report_vertices=args.max_report_shell_vertices,
                )

            # 最后叠加代理网格
            if proxy_mesh is not None:
                add_proxy_overlay_to_scene(scene, args, proxy_mesh)

            return scene

        if args.highlight_reliable:
            print_separator("Topological Reliable Visualization")
            print(f"  min_distance: {args.reliable_distance}")
            print("  green:  reliable faces (distance >= min_distance)")
            print("  yellow: intermediate faces (0 < distance < min_distance)")
            print("  red:    defect faces (distance = 0)")

            vis = build_reliable_visualization(mesh, distances, args.reliable_distance)
        else:
            print_separator("Defect Visualization")
            print("  gray:    normal faces")
            print("  yellow:  faces adjacent to open edges")
            print("  red:     faces adjacent to non-manifold edges")
            print("  orange:  faces with both defects")

            vis = build_defect_visualization(mesh, open_face_mask, nonmanifold_face_mask)

        # 默认双面渲染：薄壳从任何一侧观察都可见
        if args.double_sided:
            vis = make_double_sided(vis, args.backface_color)

        if args.input_alpha < 1.0:
            set_face_alpha(vis, args.input_alpha)

        scene = trimesh.Scene(vis)

        # 导出着色网格
        if args.output:
            vis.export(args.output)
            if args.highlight_reliable:
                print(f"\nReliable-neighborhood mesh saved to: {args.output}")
            else:
                print(f"\nColored defect mesh saved to: {args.output}")

        # 再叠加线框到场景用于可视化
        if args.wireframe:
            add_wireframe_to_scene(
                scene, vis,
                color=args.wireframe_color,
                radius=args.wireframe_radius
            )

        # 高亮显示闭合孔洞边界
        if args.highlight_holes:
            add_hole_boundaries_to_scene(
                scene, mesh,
                radius=args.hole_radius,
                min_edges=args.min_hole_edges,
                min_area=args.min_hole_area,
                verbose=True,
            )

        # 孔洞边界投影到代理网格
        if args.boundary_projection and proxy_mesh is not None:
            add_boundary_projection_to_scene(
                scene, mesh, proxy_mesh,
                radius=args.boundary_projection_radius,
                verbose=True,
                print_enclosed_vertices=args.print_shell_enclosed_vertices,
                max_report_loops=args.max_report_boundary_loops,
                max_report_vertices=args.max_report_shell_vertices,
            )

    if scene is not None and proxy_mesh is not None:
        add_proxy_overlay_to_scene(scene, args, proxy_mesh)

    if args.highlight_uncovered_edges and scene is not None:
        add_uncovered_edges_to_scene(
            scene, mesh,
            data_dir=args.uncovered_data_dir,
            radius=args.uncovered_radius,
            verbose=True,
        )

    return scene


def main():
    parser = argparse.ArgumentParser(
        description="检查并可视化网格模型，输出拓扑、边长、面积等统计信息。"
    )
    parser.add_argument("input_file", help="输入网格文件路径 (stl/ply/obj)")
    parser.add_argument("-o", "--output",
                        help="输出带缺陷着色的网格文件路径 (可选)")
    parser.add_argument("--show", action="store_true",
                        help="显示可视化窗口")
    parser.add_argument("--wireframe", action="store_true",
                        help="在可视化中叠加黑色线框，观察三角剖分")
    parser.add_argument("--wireframe-radius", type=float, default=None,
                        help="线框圆柱半径（默认按包围盒自动计算）")
    parser.add_argument("--wireframe-color", type=str, default="0,0,0,255",
                        help="线框 RGBA 颜色，默认 '0,0,0,255'")
    parser.add_argument("--highlight-holes", action="store_true",
                        help="高亮显示闭合孔洞边界，相邻孔洞使用不同高饱和度颜色")
    parser.add_argument("--highlight-uncovered-edges", action="store_true",
                        help="在 --show 或 --output 时高亮 hole diagnosis 中未覆盖的开放边")
    parser.add_argument("--uncovered-data-dir", type=str,
                        default="hole_diagnosis_report",
                        help="hole diagnosis 输出目录，用于加载未覆盖开放边数据 "
                             "（默认 hole_diagnosis_report）")
    parser.add_argument("--uncovered-radius", type=float, default=None,
                        help="未覆盖开放边圆柱半径（默认自动计算，略粗于普通线框）")
    parser.add_argument("--highlight-reliable", action="store_true",
                        help="高亮显示可靠邻域（绿色=可靠，黄色=缺陷邻域，红色=不可靠）")
    parser.add_argument("--keep-reliable-only", action="store_true",
                        help="只保留可靠面片，删除其余面片。"
                             "需配合 --output 或 --show 使用。")
    parser.add_argument("--reliable-threshold", type=float, default=None,
                        help="[已废弃] 请使用 --reliable-distance。"
                             "该参数不再生效，仅保留兼容。")
    parser.add_argument("--reliable-distance", type=int, default=2,
                        help="可靠面片距离开放/非流形面的最小拓扑距离"
                             "（面邻接跳数），默认 2。")
    parser.add_argument("--hole-radius", type=float, default=None,
                        help="孔洞边界圆柱半径（默认自动计算，通常比普通线框略粗）")
    parser.add_argument("--double-sided", dest='double_sided',
                        action='store_true', default=True,
                        help="双面渲染薄壳网格（默认开启）")
    parser.add_argument("--no-double-sided", dest='double_sided',
                        action='store_false',
                        help="关闭双面渲染，恢复默认背面剔除")
    parser.add_argument("--backface-color", type=str, default=None,
                        help="双面渲染时背面子颜色，格式 'R,G,B,A'。"
                             "默认自动根据正面颜色生成同色系暗色")
    parser.add_argument("--min-hole-edges", type=int, default=3,
                        help="高亮孔洞的最小边界边数（默认 3）")
    parser.add_argument("--min-hole-area", type=float, default=0.0,
                        help="高亮孔洞的最小面积（默认 0，不过滤）")
    parser.add_argument("--weld-small-holes", action="store_true",
                        help="自动焊接面积小于阈值的小孔洞，适用于扫描去重后的伪孔洞")
    parser.add_argument("--weld-hole-threshold", type=float, default=None,
                        help="焊接孔洞的绝对面积阈值；默认使用面片面积百分位")
    parser.add_argument("--weld-hole-quantile", type=float, default=5.0,
                        help="用于计算焊接阈值的面片面积百分位，默认 5")
    parser.add_argument("--weld-min-hole-edges", type=int, default=3,
                        help="焊接孔洞的最小边数，默认 3")
    parser.add_argument("--overlay-proxy", type=str, default=None,
                        help="同时显示输入的代理网格文件（例如体素壳），路径为 STL/PLY/OBJ")
    parser.add_argument("--input-alpha", type=float, default=1.0,
                        help="输入网格（或可靠子网格）不透明度，范围 0~1，"
                             "默认 1.0。与 --overlay-proxy 配合使用时，"
                             "设置为 0.3~0.6 效果较好")
    parser.add_argument("--boundary-projection", action="store_true",
                        help="将输入网格（或可靠子网格）的孔洞边界投影到代理网格表面显示。"
                             "仅在 --overlay-proxy 且 --highlight-holes 或 --keep-reliable-only 时有效")
    parser.add_argument("--boundary-projection-radius", type=float, default=None,
                        help="边界投影线圆柱半径（默认使用与孔洞边界相同的半径）")
    parser.add_argument(
        "--print-shell-enclosed-vertices",
        action="store_true",
        help="在 --boundary-projection 模式下，打印投影孔洞多边形及被包围的代理网格顶点信息"
    )
    parser.add_argument(
        "--max-report-boundary-loops",
        type=int,
        default=None,
        help="最多打印多少个孔洞边界的投影诊断信息；默认全部"
    )
    parser.add_argument(
        "--max-report-shell-vertices",
        type=int,
        default=20,
        help="每个孔洞最多列出多少个被包围的代理网格顶点索引；默认 20"
    )
    parser.add_argument("--proxy-alpha", type=float, default=0.45,
                        help="代理网格透明度，0=全透明，1=不透明，默认 0.45")
    parser.add_argument("--proxy-color", type=str, default=None,
                        help="代理网格统一颜色，格式 'R,G,B' 或 'R,G,B,A'。"
                             "默认使用浅蓝灰色半透明 [128, 180, 255, alpha]")
    parser.add_argument("--proxy-double-sided", dest='proxy_double_sided',
                        action='store_true', default=True,
                        help="双面渲染代理网格（默认开启）")
    parser.add_argument("--no-proxy-double-sided", dest='proxy_double_sided',
                        action='store_false',
                        help="关闭代理网格双面渲染")
    parser.add_argument("--full-diagnosis", action="store_true",
                        help="执行 Full Diagnosis（2-Pass），生成异常面拓扑分类报告")
    parser.add_argument("--hole-diagnosis", action="store_true",
                        help="执行孔洞诊断（健康孔洞提取及未覆盖开放边分类）")
    parser.add_argument("--compute-enclosing-boundaries", action="store_true",
                        help="在 hole diagnosis 中计算每个未覆盖开放边组件的最小包络流形边界")
    parser.add_argument(
        "--extract-component-package",
        action="store_true",
        help="提取指定边界组件或健康孔洞的局部网格和组件数据并保存"
    )
    parser.add_argument(
        "--component-output",
        type=str,
        default=None,
        help="提取的局部网格 PLY 输出路径（默认：<输入文件名>_component_<boundary_id>.ply）"
    )
    parser.add_argument(
        "--component-json",
        type=str,
        default=None,
        help="提取的组件 JSON 输出路径（默认：<输入文件名>_component_<boundary_id>.json）"
    )
    parser.add_argument(
        "--component-package",
        type=str,
        default=None,
        help="加载已提取的组件包 JSON（与 --visualize-boundary-component 配合使用）"
    )
    parser.add_argument("--visualize-boundary-component", action="store_true",
                        help="可视化特定孔洞/开放边分量及其三角面片")
    parser.add_argument("--boundary-type", choices=["uncovered", "healthy"],
                        default="uncovered",
                        help="要可视化的边界类型：uncovered（未覆盖开放边分量）或 healthy（健康孔洞）")
    parser.add_argument("--boundary-id", type=int, default=0,
                        help="边界组件或孔洞的 ID（默认 0）")
    parser.add_argument("--boundary-data-dir", type=str, default="hole_diagnosis_report",
                        help="hole diagnosis 输出目录（默认 hole_diagnosis_report）")
    parser.add_argument("--boundary-neighborhood-depth", type=int, default=1,
                        help="边界邻域深度：0 仅边界，1 边界所在面片，"
                             "2 边界面片+直接邻居，依此类推（默认 1）")
    parser.add_argument("--boundary-radius", type=float, default=None,
                        help="边界圆柱半径（默认自动计算）")
    parser.add_argument("--boundary-show-original", action="store_true",
                        help="同时显示原始网格（半透明背景）")
    parser.add_argument(
        "--print-neighborhood-distribution",
        action="store_true",
        help="打印边界组件邻域面片距离分布（默认关闭，开启会增加耗时）",
    )
    parser.add_argument("--debug-scene", action="store_true",
                        help="在显示或导出边界组件场景前，打印场景内所有几何对象的位置与大小")
    parser.add_argument("--fit-watertight-patch", action="store_true",
                        help="在可视化边界组件时，拟合亏格0水密曲面并显示包络交线")
    parser.add_argument("--patch-method",
                        choices=["poisson", "convex_hull", "concave_hull"],
                        default="poisson",
                        help="水密包络曲面生成算法（默认 poisson）")
    parser.add_argument("--patch-neighborhood-depth", type=int, default=2,
                        help="点云提取的邻域深度（默认 2）")
    parser.add_argument("--patch-poisson-depth", type=int, default=8,
                        help="泊松重建深度（默认 8）")
    parser.add_argument("--patch-density-quantile", type=float, default=0.2,
                        help="泊松密度过滤分位（默认 0.2）")
    parser.add_argument("--patch-alpha", type=float, default=1.5,
                        help="凹包算法的 alpha 参数（默认 1.5）")
    parser.add_argument("--patch-opacity", type=float, default=0.3,
                        help="拟合水密曲面的不透明度，范围 0~1，默认 0.3")
    parser.add_argument("--allow-non-genus0", action="store_true",
                        help="允许水密但亏格非0的拟合曲面通过（用于可视化调试）")
    parser.add_argument(
        "--generate-seifert-surface",
        action="store_true",
        help="在可视化健康孔洞时，生成并显示 Seifert 曲面（固定边界极小曲面优化）"
    )
    parser.add_argument(
        "--seifert-optimize-iterations",
        type=int,
        default=200,
        help="Seifert 曲面内部顶点优化迭代次数（默认 200）"
    )
    parser.add_argument(
        "--seifert-step-size",
        type=float,
        default=1.0,
        help="Seifert 曲面 Dirichlet 求解松弛系数（默认 1.0）"
    )
    parser.add_argument(
        "--seifert-tolerance",
        type=float,
        default=1e-7,
        help="Seifert 曲面优化收敛容差（默认 1e-7）"
    )
    parser.add_argument(
        "--seifert-color",
        type=str,
        default="255,215,0,255",
        help="Seifert 曲面显示颜色，格式 'R,G,B,A'，默认金色"
    )
    parser.add_argument(
        "--seifert-curvature-report",
        action="store_true",
        help="打印 Seifert 曲面曲率统计信息"
    )
    parser.add_argument(
        "--print-camera-info",
        action="store_true",
        help="在可视化窗口关闭后输出窗口大小、摄像机位置/朝向等参数"
    )
    parser.add_argument(
        "--camera-info-output",
        type=str,
        default=None,
        help="将摄像机信息保存为 JSON 文件（可选）"
    )
    parser.add_argument("--hole-diagnosis-output", type=str, default="hole_diagnosis_report",
                        help="孔洞诊断输出目录（默认 hole_diagnosis_report）")
    parser.add_argument("--diagnosis-output", type=str, default="diagnosis_report",
                        help="Full Diagnosis 输出目录（默认 diagnosis_report）")
    parser.add_argument("--diagnosis-format", type=str, default="html",
                        choices=["html", "latex"],
                        help="报告格式：html 或 latex（默认 html）")
    parser.add_argument("--resume", action="store_true",
                        help="从现有检查点继续 Full Diagnosis Pass 2（跳过已完成类别）")
    parser.add_argument("--valence-threshold",
                        type=int,
                        default=5,
                        help="顶点元 valence 截断阈值，默认 5。"
                             "当顶点被引用的面片数 >= 阈值时，截断编码归并为该阈值。")

    if len(sys.argv) == 1:
        parser.print_help()
        return

    args = parser.parse_args()

    # 可视化边界组件时，自动开启显示窗口
    if args.visualize_boundary_component and not args.show:
        args.show = True
        print("  [info] --visualize-boundary-component 已自动开启 --show")

    if args.reliable_threshold is not None:
        print("[WARNING] --reliable-threshold is deprecated and ignored. "
              "Use --reliable-distance instead.")

    args.wireframe_color = _parse_color_string(args.wireframe_color)

    if args.backface_color is not None:
        args.backface_color = _parse_color_string(args.backface_color)

    if args.proxy_color is not None:
        args.proxy_color = _parse_color_string_flexible(args.proxy_color)

    if args.seifert_color is not None:
        args.seifert_color = _parse_color_string(args.seifert_color)

    print(f"Hey! Loading {args.input_file}")
    mesh = trimesh.load(args.input_file, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump(concatenate=True)
        print("Multiple meshes detected, merged.")

    # 提取组件包入口
    if args.extract_component_package:
        extract_component_package(mesh, args)
        return

    # 新增：局部边界组件可视化入口
    if args.visualize_boundary_component:
        visualize_boundary_component(mesh, args)
        return

    if args.hole_diagnosis:
        perform_hole_diagnosis(mesh, args)
        return

    if args.full_diagnosis:
        perform_full_diagnosis(mesh, args)
        return

    scene = inspect_mesh(mesh, args)

    if args.show and scene is not None:
        try:
            os.environ['TRIMESH_DEFAULT_VIEWER'] = 'vedo'
            _show_scene_with_camera_info(scene, args)
        except Exception as e:
            print(f"\n[ERROR] Visualization failed: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
