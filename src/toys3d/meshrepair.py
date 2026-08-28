import sys
import os

# Ensure src directory is on the path so that 'toys3d' can be imported
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_src_parent = os.path.dirname(_project_root)
if _src_parent not in sys.path:
    sys.path.insert(0, _src_parent)

import argparse
import numpy as np
import trimesh

from toys3d.geometrics import (
    compute_mesh_stats,
    analyze_mesh_defects,
    repair_mesh_by_removing_duplicates,
    repair_nonmanifold_edges,
    fill_small_boundary_loops,
    fill_holes_with_proxy,
    extract_boundary_loops,
    compute_loop_flatness,
    polygon_area_from_3d_ccw,
    remove_small_open_edge_chains,
    remove_pseudo_holes,
    trim_hanging_open_faces,
    repair_to_watertight,
    weld_small_holes,
)


def print_defect_summary(tag, defect_stats):
    """打印缺陷摘要。"""
    print(
        f"  {tag}: open_edges={defect_stats['open_edges']}, "
        f"nonmanifold_edges={defect_stats['nonmanifold_edges']}, "
        f"open_faces={defect_stats['open_faces']}, "
        f"nonmanifold_faces={defect_stats['nonmanifold_faces']}"
    )


def print_boundary_loop_stats(loops, mesh):
    """
    打印开放边界环的边数分布与平坦度统计。
    """
    if not loops:
        print("  Boundary loops: 0")
        return

    lengths = np.array([len(loop) for loop in loops])
    flatnesses = np.array([compute_loop_flatness(mesh, loop)[0]
                           for loop in loops])

    def pct(arr, p):
        return int(np.percentile(arr, p, method='lower'))

    print(f"  Boundary loops: {len(lengths)}")
    print(f"    edges: min={int(np.min(lengths))}, "
          f"p1={pct(lengths, 1)}, p5={pct(lengths, 5)}, "
          f"p25={pct(lengths, 25)}, p50={pct(lengths, 50)}, "
          f"p75={pct(lengths, 75)}, p90={pct(lengths, 90)}, "
          f"p95={pct(lengths, 95)}, p99={pct(lengths, 99)}, "
          f"max={int(np.max(lengths))}")
    print(f"    flatness: min={flatnesses.min():.3f}, "
          f"p50={np.percentile(flatnesses, 50):.3f}, "
          f"p95={np.percentile(flatnesses, 95):.3f}, "
          f"max={flatnesses.max():.3f}")

    areas = []
    for loop in loops:
        pts = mesh.vertices[np.array(loop)]
        area = polygon_area_from_3d_ccw(pts)
        areas.append(area)
    areas = np.array(areas)

    print(f"    hole area percentiles: "
          f"total={areas.sum():.4f}, "
          f"p1={np.percentile(areas, 1):.4f}, "
          f"p5={np.percentile(areas, 5):.4f}, "
          f"p25={np.percentile(areas, 25):.4f}, "
          f"p50={np.percentile(areas, 50):.4f}, "
          f"p75={np.percentile(areas, 75):.4f}, "
          f"p90={np.percentile(areas, 90):.4f}, "
          f"p95={np.percentile(areas, 95):.4f}, "
          f"p99={np.percentile(areas, 99):.4f}, "
          f"max={areas.max():.4f}")


def repair_mesh_iterative(mesh,
                          max_iterations=5,
                          normal_repair=True,
                          remove_duplicate=True,
                          repair_nonmanifold=True,
                          fill_holes=True,
                          verbose=True):
    """
    迭代修复开放边和非流形边。

    Parameters
    ----------
    mesh : trimesh.Trimesh
    max_iterations : int
        最大修复轮数
    normal_repair : bool
        是否在最终修复后执行法向一致性修复（默认 True）。
    remove_duplicate : bool
        是否去重/去退化面
    repair_nonmanifold : bool
        是否修复非流形边
    fill_holes : bool
        是否封闭小开放边界环
    verbose : bool
        是否输出每轮缺陷统计

    Returns
    -------
    repaired : trimesh.Trimesh
    """
    repaired = mesh.copy()

    # 修复前先合并重合顶点，去除重复顶点导致的伪开放边/非流形边
    repaired.merge_vertices()
    repaired.remove_unreferenced_vertices()

    if verbose:
        print("\n[Initial defects]")
        defect_stats, _, _ = analyze_mesh_defects(repaired)
        print_defect_summary("input", defect_stats)

    for it in range(1, max_iterations + 1):
        if verbose:
            print(f"\n[Repair iteration {it}/{max_iterations}]")

        # 1. 去重 / 去退化面
        if remove_duplicate:
            repaired = repair_mesh_by_removing_duplicates(repaired)

        # 2. 修复非流形边
        if repair_nonmanifold:
            defects, _, _ = analyze_mesh_defects(repaired)
            if defects['nonmanifold_edges'] > 0:
                repaired = repair_nonmanifold_edges(
                    repaired, max_iterations=10
                )

        # 3. 只填充小型和平坦的边界环，避免暴力填充导致拓扑恶化
        if fill_holes:
            defects, _, _ = analyze_mesh_defects(repaired)
            if defects['open_edges'] > 0:
                repaired = fill_small_boundary_loops(
                    repaired,
                    max_edges=5,
                    max_flatness=0.5,
                    verbose=verbose,
                )

        # 重新统计
        defect_stats, _, _ = analyze_mesh_defects(repaired)
        if verbose:
            print_defect_summary("after", defect_stats)

        # 如果已经无开放边且无非流形边，提前结束
        if (defect_stats['open_edges'] == 0 and
                defect_stats['nonmanifold_edges'] == 0):
            if verbose:
                print("\nAll open and non-manifold edges resolved.")
            break

    # 最终清理
    repaired = repaired.copy()
    repaired.merge_vertices()
    repaired.remove_unreferenced_vertices()

    if normal_repair:
        repaired.fix_normals()

    return repaired


def main():
    parser = argparse.ArgumentParser(
        description="按需修复网格拓扑缺陷。默认不执行任何修复，仅导出输入网格副本；"
                    "通过选项显式激活预清理或主修复模式。"
    )
    parser.add_argument("input_file", help="输入网格文件路径")
    parser.add_argument("-o", "--output", required=True,
                        help="输出修复后网格文件路径")

    parser.add_argument("--max-iterations", type=int, default=5,
                        help="最大修复轮数（默认 5）")

    parser.add_argument("--no-normal-repair", action="store_true",
                        help="关闭最终法向一致性修复")

    parser.add_argument("--no-duplicate", action="store_true",
                        help="关闭去重 / 去退化面")
    parser.add_argument("--no-nonmanifold", action="store_true",
                        help="关闭非流形边修复")
    parser.add_argument("--no-fill-holes", action="store_true",
                        help="关闭小孔封闭")

    # 伪孔洞与前处理清理
    parser.add_argument("--trim-hanging-faces", action="store_true",
                        help="修剪开放边数 >= 2 的悬挂面片，减少开放边界悬臂")

    parser.add_argument("--remove-pseudo-holes", action="store_true",
                        help="在拓扑修复前执行伪孔洞清理，删除短小开放边链")
    parser.add_argument("--pseudo-hole-max-edges", type=int, default=2,
                        help="伪孔洞最大开放边链边数（默认 2）")
    parser.add_argument("--pseudo-hole-iterations", type=int, default=5,
                        help="伪孔洞清理迭代次数（默认 5）")

    # 焊接小孔洞参数
    parser.add_argument("--weld-small-holes", action="store_true",
                        help="焊接面积小于阈值的小孔洞，适用于扫描去重后的伪孔洞")
    parser.add_argument("--weld-hole-threshold", type=float, default=None,
                        help="焊接孔洞的绝对面积阈值；默认使用面片面积百分位")
    parser.add_argument("--weld-hole-quantile", type=float, default=5.0,
                        help="用于计算焊接阈值的面片面积百分位，默认 5")
    parser.add_argument("--weld-min-hole-edges", type=int, default=3,
                        help="焊接孔洞的最小边数，默认 3")

    # 代理壳修补模式参数
    parser.add_argument("--proxy-shell", type=str, default=None,
                        help="指定水密代理壳网格文件路径（如体素壳），"
                             "启用可靠面片提取 + 代理支撑孔洞修补模式")
    parser.add_argument("--proxy-face-center-threshold", type=int, default=20,
                        help="投影多边形内代理面片中心数量阈值，"
                             "达到该数量才使用代理补丁，否则平面修补（默认 20）")
    parser.add_argument("--proxy-max-projection-distance", type=float, default=None,
                        help="孔洞边界投影到代理壳的最大允许距离，"
                             "超过则放弃代理补丁，默认不限制")
    parser.add_argument("--proxy-min-loop-edges", type=int, default=12,
                        help="使用代理补丁的最小边界边数，"
                             "小于此值一律平面修补（默认 12）")
    parser.add_argument("--prefill-small-holes", action="store_true",
                        help="在代理壳修补前，先填充边数较少的小孔洞")
    parser.add_argument("--prefill-max-edges", type=int, default=6,
                        help="预填充小孔洞的最大边界边数（默认 6）")
    parser.add_argument("--prefill-max-flatness", type=float, default=0.50,
                        help="预填充小孔洞的最大边界平坦度（默认 0.50）")

    parser.add_argument("--watertight", action="store_true",
                        help="使用体素化 + Marching Cubes 重建水密外壳")
    parser.add_argument("--iterative-repair", action="store_true",
                        help="显式激活迭代修复流程（去重、非流形修复、小孔填充）")
    parser.add_argument("--watertight-resolution", type=int, default=256,
                        help="水密模式默认体素分辨率（默认 256）")
    parser.add_argument("--watertight-voxel-size", type=float, default=None,
                        help="水密模式显式体素边长；优先级高于分辨率")
    parser.add_argument("--watertight-project-to-input", action="store_true",
                        help="将水密代理壳顶点投影回原始输入网格表面，减少锯齿感")
    parser.add_argument("--watertight-smooth", action="store_true",
                        help="对水密重建结果执行 Taubin 平滑")
    parser.add_argument("--watertight-smooth-iterations", type=int, default=10,
                        help="Taubin 平滑迭代次数（默认 10）")

    parser.add_argument("-v", "--verbose", action="store_true",
                        help="输出详细统计信息")

    args = parser.parse_args()

    print(f"Hey! Loading {args.input_file}")
    mesh = trimesh.load(args.input_file, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump(concatenate=True)
        print("Multiple meshes detected, merged.")

    if args.verbose:
        print("\n[Input mesh stats]")
        for k, v in compute_mesh_stats(mesh).items():
            print(f"  {k}: {v}")

    # 默认不执行任何修复，仅复制输入网格
    repaired = mesh.copy()
    repair_activated = False

    # --- 预清理步骤（按顺序执行，所有步骤可选） ---
    if args.trim_hanging_faces:
        repair_activated = True
        if args.verbose:
            print("\n[Pre-clean] trimming hanging open faces...")
            before, _, _ = analyze_mesh_defects(repaired)
            print(f"  open_edges before trim: {before['open_edges']}")
        repaired = trim_hanging_open_faces(repaired, verbose=args.verbose)
        if args.verbose:
            after, _, _ = analyze_mesh_defects(repaired)
            print(f"  open_edges after trim:  {after['open_edges']}")
            print("[Pre-clean] trim done")

    if args.remove_pseudo_holes:
        repair_activated = True
        if args.verbose:
            print("\n[Pre-clean] removing pseudo holes...")
        repaired = remove_pseudo_holes(
            repaired,
            max_chain_edges=args.pseudo_hole_max_edges,
            max_iterations=args.pseudo_hole_iterations,
            verbose=args.verbose,
        )
        if args.verbose:
            print("[Pre-clean] done")

    if args.weld_small_holes:
        repair_activated = True
        if args.verbose:
            print("\n[Pre-clean] welding small holes...")
        repaired = weld_small_holes(
            repaired,
            threshold=args.weld_hole_threshold,
            quantile=args.weld_hole_quantile,
            min_edges=args.weld_min_hole_edges,
            verbose=args.verbose,
        )
        if args.verbose:
            print("[Pre-clean] weld done")

    if args.prefill_small_holes:
        repair_activated = True
        if args.verbose:
            print("\n[Pre-proxy] filling small boundary loops...")
        repaired = fill_small_boundary_loops(
            repaired,
            max_edges=args.prefill_max_edges,
            max_flatness=args.prefill_max_flatness,
            verbose=args.verbose,
        )
        if args.verbose:
            print("[Pre-proxy] small hole fill done")

    # --- 主修复模式（互斥） ---
    modes = sum([args.proxy_shell is not None, args.watertight, args.iterative_repair])
    if modes > 1:
        print("[ERROR] --proxy-shell, --watertight, --iterative-repair are mutually exclusive.")
        return 1

    if args.proxy_shell is not None:
        repair_activated = True
        print(f"Hey! Loading shell: {args.proxy_shell}")
        shell_mesh = trimesh.load(args.proxy_shell, force="mesh")
        if not isinstance(shell_mesh, trimesh.Trimesh):
            shell_mesh = shell_mesh.dump(concatenate=True)
            print("Multiple shell meshes detected, merged.")
        repaired = fill_holes_with_proxy(
            repaired,
            shell_mesh,
            proxy_face_center_threshold=args.proxy_face_center_threshold,
            max_projection_distance=args.proxy_max_projection_distance,
            min_proxy_loop_edges=args.proxy_min_loop_edges,
            verbose=args.verbose,
        )
    elif args.watertight:
        repair_activated = True
        repaired = repair_to_watertight(
            repaired,
            resolution=args.watertight_resolution,
            voxel_size=args.watertight_voxel_size,
            project_to_input=args.watertight_project_to_input,
            smooth_watertight=args.watertight_smooth,
            smooth_iterations=args.watertight_smooth_iterations,
        )
    elif args.iterative_repair:
        repair_activated = True
        repaired = repair_mesh_iterative(
            repaired,
            max_iterations=args.max_iterations,
            remove_duplicate=not args.no_duplicate,
            repair_nonmanifold=not args.no_nonmanifold,
            fill_holes=not args.no_fill_holes,
            normal_repair=not args.no_normal_repair,
            verbose=args.verbose,
        )

    if not repair_activated:
        if args.verbose:
            print("\nNo repair options specified; exporting original mesh.")
    else:
        # 最终清理（安全操作）
        repaired = repair_mesh_by_removing_duplicates(repaired)
        repaired.fix_normals()

        # 缺陷检查
        defects, _, _ = analyze_mesh_defects(repaired)
        if defects['open_edges'] > 0 or defects['nonmanifold_edges'] > 0:
            print(f"\n[WARNING] Output is NOT watertight: "
                  f"open_edges={defects['open_edges']}, "
                  f"nonmanifold_edges={defects['nonmanifold_edges']}")

    if args.verbose:
        print("\n[Output mesh stats]")
        for k, v in compute_mesh_stats(repaired).items():
            print(f"  {k}: {v}")

    repaired.export(args.output)
    print(f"\nRepaired mesh saved to: {args.output}")
    print(f"  vertices: {repaired.vertices.shape[0]}, "
          f"faces: {repaired.faces.shape[0]}")


if __name__ == "__main__":
    main()
