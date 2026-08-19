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
    fill_holes_adaptive,
    extract_boundary_loops,
    compute_loop_flatness,
    polygon_area_from_3d_ccw,
    repair_normals,
    remove_small_open_edge_chains,
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
                          max_hole_edges=50,
                          strategy='flatness',
                          max_fan_edges=15,
                          max_fan_flatness=0.05,
                          max_earclip_edges=100,
                          max_earclip_flatness=0.15,
                          max_surface_fit_edges=500,
                          max_surface_fit_flatness=0.40,
                          edge_count_small_p=50.0,
                          edge_count_large_p=95.0,
                          g2=False,
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
    max_hole_edges : int
        允许封闭的开放边界环最大边数 (已弃用，保留兼容)
    strategy : str
        孔洞填补策略：'flatness' 或 'edge-count'
    max_fan_edges : int
        fan fill 最大边数
    max_fan_flatness : float
        fan fill 最大平坦度
    max_earclip_edges : int
        ear clip 最大边数
    max_earclip_flatness : float
        ear clip 最大平坦度
    max_surface_fit_edges : int
        surface fit 最大边数
    max_surface_fit_flatness : float
        surface fit 最大平坦度
    edge_count_small_p : float
        edge-count 策略下小孔边数百分位
    edge_count_large_p : float
        edge-count 策略下中孔边数百分位
    g2 : bool
        是否使用 G2 光滑曲面拟合（预留接口）
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
                    repaired, max_iterations=10, verbose=False
                )

        # 3. 封闭小开放边界环
        if fill_holes:
            defects, _, _ = analyze_mesh_defects(repaired)
            if defects['open_edges'] > 0:
                repaired = remove_small_open_edge_chains(
                    repaired,
                    max_chain_edges=2,
                    verbose=verbose,
                )

                if verbose:
                    print("  Boundary loops before filling:")
                    loops = extract_boundary_loops(repaired)
                    print_boundary_loop_stats(loops, repaired)

                repaired = fill_holes_adaptive(
                    repaired,
                    strategy=strategy,
                    max_fan_edges=max_fan_edges,
                    max_fan_flatness=max_fan_flatness,
                    max_earclip_edges=max_earclip_edges,
                    max_earclip_flatness=max_earclip_flatness,
                    max_surface_fit_edges=max_surface_fit_edges,
                    max_surface_fit_flatness=max_surface_fit_flatness,
                    edge_count_small_p=edge_count_small_p,
                    edge_count_large_p=edge_count_large_p,
                    g2=g2,
                    verbose=verbose
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

    # 最终清理并修复法向
    repaired = repaired.copy()
    repaired.merge_vertices()
    repaired.remove_unreferenced_vertices()

    if normal_repair:
        repaired = repair_normals(repaired, verbose=verbose)
    else:
        repaired.fix_normals()

    return repaired


def main():
    parser = argparse.ArgumentParser(
        description="迭代修复网格拓扑缺陷：开放边、非流形边、小孔洞。"
    )
    parser.add_argument("input_file", help="输入网格文件路径")
    parser.add_argument("-o", "--output", required=True,
                        help="输出修复后网格文件路径")

    parser.add_argument("--max-iterations", type=int, default=5,
                        help="最大修复轮数（默认 5）")
    parser.add_argument("--max-hole-edges", type=int, default=50,
                        help="(已弃用，自适应填补不再使用)")

    parser.add_argument("--hole-strategy", type=str, default="flatness",
                        choices=["flatness", "edge-count"],
                        help="孔洞填补策略：flatness=基于平坦度，"
                             "edge-count=基于边数统计（默认 flatness）")

    # flatness 策略阈值
    parser.add_argument("--max-fan-edges", type=int, default=15,
                        help="fan fill 最大边数（默认 15）")
    parser.add_argument("--max-fan-flatness", type=float, default=0.05,
                        help="fan fill 最大平坦度（默认 0.05）")
    parser.add_argument("--max-earclip-edges", type=int, default=100,
                        help="ear clip 最大边数（默认 100）")
    parser.add_argument("--max-earclip-flatness", type=float, default=0.15,
                        help="ear clip 最大平坦度（默认 0.15）")
    parser.add_argument("--max-surface-fit-edges", type=int, default=500,
                        help="surface fit 最大边数（默认 500）")
    parser.add_argument("--max-surface-fit-flatness", type=float, default=0.40,
                        help="surface fit 最大平坦度（默认 0.40）")

    # edge-count 策略阈值
    parser.add_argument("--edge-count-small-p", type=float, default=50.0,
                        help="edge-count 策略下小孔边数百分位（默认 50）")
    parser.add_argument("--edge-count-large-p", type=float, default=95.0,
                        help="edge-count 策略下中孔边数百分位（默认 95）")

    parser.add_argument("--g2", action="store_true",
                        help="使用 G2 光滑曲面拟合（当前预留接口，未实现）")
    parser.add_argument("--no-normal-repair", action="store_true",
                        help="关闭最终法向一致性修复")

    parser.add_argument("--no-duplicate", action="store_true",
                        help="关闭去重 / 去退化面")
    parser.add_argument("--no-nonmanifold", action="store_true",
                        help="关闭非流形边修复")
    parser.add_argument("--no-fill-holes", action="store_true",
                        help="关闭小孔封闭")

    parser.add_argument("-v", "--verbose", action="store_true",
                        help="输出详细统计信息")

    args = parser.parse_args()

    if args.max_hole_edges != 50:
        print("[WARNING] --max-hole-edges is deprecated; "
              "adaptive hole filling uses strategy-based thresholds.")

    if args.g2:
        print("[WARNING] --g2 is reserved but not yet implemented; "
              "will raise NotImplementedError if a surface-fit hole is encountered.")

    print(f"Loading: {args.input_file}")
    mesh = trimesh.load(args.input_file, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump(concatenate=True)
        print("Multiple meshes detected, merged.")

    if args.verbose:
        print("\n[Input mesh stats]")
        for k, v in compute_mesh_stats(mesh).items():
            print(f"  {k}: {v}")

    repaired = repair_mesh_iterative(
        mesh,
        max_iterations=args.max_iterations,
        max_hole_edges=args.max_hole_edges,
        strategy=args.hole_strategy,
        max_fan_edges=args.max_fan_edges,
        max_fan_flatness=args.max_fan_flatness,
        max_earclip_edges=args.max_earclip_edges,
        max_earclip_flatness=args.max_earclip_flatness,
        max_surface_fit_edges=args.max_surface_fit_edges,
        max_surface_fit_flatness=args.max_surface_fit_flatness,
        edge_count_small_p=args.edge_count_small_p,
        edge_count_large_p=args.edge_count_large_p,
        g2=args.g2,
        remove_duplicate=not args.no_duplicate,
        repair_nonmanifold=not args.no_nonmanifold,
        fill_holes=not args.no_fill_holes,
        normal_repair=not args.no_normal_repair,
        verbose=args.verbose,
    )

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
