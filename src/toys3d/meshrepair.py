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
    fill_small_holes,
    extract_boundary_loops,
)


def print_defect_summary(tag, defect_stats):
    """打印缺陷摘要。"""
    print(
        f"  {tag}: open_edges={defect_stats['open_edges']}, "
        f"nonmanifold_edges={defect_stats['nonmanifold_edges']}, "
        f"open_faces={defect_stats['open_faces']}, "
        f"nonmanifold_faces={defect_stats['nonmanifold_faces']}"
    )


def print_boundary_loop_stats(loops, max_hole_edges):
    """
    打印开放边界环的边数分布统计。

    Parameters
    ----------
    loops : list of list of int
        每个边界环的顶点索引列表。
    max_hole_edges : int
        允许封闭的最大边界环边数。
    """
    if not loops:
        print("  Boundary loops: 0")
        return

    lengths = np.array([len(loop) for loop in loops])
    stats = {
        'count': len(lengths),
        'mean': float(np.mean(lengths)),
        'min': int(np.min(lengths)),
        'p1': float(np.percentile(lengths, 1)),
        'p5': float(np.percentile(lengths, 5)),
        'p25': float(np.percentile(lengths, 25)),
        'p50': float(np.percentile(lengths, 50)),
        'p75': float(np.percentile(lengths, 75)),
        'p90': float(np.percentile(lengths, 90)),
        'p95': float(np.percentile(lengths, 95)),
        'p99': float(np.percentile(lengths, 99)),
        'max': int(np.max(lengths)),
        'too_large': int(np.sum(lengths > max_hole_edges)),
    }

    print(f"  Boundary loops: {stats['count']}")
    print(f"    edges: min={stats['min']}, "
          f"p1={stats['p1']:.1f}, p5={stats['p5']:.1f}, "
          f"p25={stats['p25']:.1f}, p50={stats['p50']:.1f}, "
          f"p75={stats['p75']:.1f}, p90={stats['p90']:.1f}, "
          f"p95={stats['p95']:.1f}, p99={stats['p99']:.1f}, "
          f"max={stats['max']}")
    print(f"    too large to fill (> {max_hole_edges} edges): "
          f"{stats['too_large']}")


def repair_mesh_iterative(mesh, max_iterations=5, max_hole_edges=50,
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
        允许封闭的开放边界环最大边数
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
                if verbose:
                    print("  Boundary loops before filling:")
                    loops = extract_boundary_loops(repaired)
                    print_boundary_loop_stats(loops, max_hole_edges)

                repaired = fill_small_holes(
                    repaired, max_loop_edges=max_hole_edges, verbose=False
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
                        help="允许封闭的开放边界环最大边数（默认 50）")

    parser.add_argument("--no-duplicate", action="store_true",
                        help="关闭去重 / 去退化面")
    parser.add_argument("--no-nonmanifold", action="store_true",
                        help="关闭非流形边修复")
    parser.add_argument("--no-fill-holes", action="store_true",
                        help="关闭小孔封闭")

    parser.add_argument("-v", "--verbose", action="store_true",
                        help="输出详细统计信息")

    args = parser.parse_args()

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
        remove_duplicate=not args.no_duplicate,
        repair_nonmanifold=not args.no_nonmanifold,
        fill_holes=not args.no_fill_holes,
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
