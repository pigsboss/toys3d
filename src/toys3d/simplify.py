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
    build_proxy_mesh,
)


def main():
    parser = argparse.ArgumentParser(
        description="简化扫描网格，生成代理模型：修复拓扑缺陷 + 简化 + 清理。"
    )
    parser.add_argument("input_file", help="输入网格文件路径 (stl/ply/obj)")
    parser.add_argument("-o", "--output", required=True,
                        help="输出代理网格文件路径")
    parser.add_argument("--target-faces", type=int, default=50000,
                        help="目标面片数（默认 50000）")
    parser.add_argument("--max-edge-length", type=float, default=None,
                        help="最大允许边长，默认自动=包围盒对角线*0.02")
    parser.add_argument("--repair-iterations", type=int, default=3,
                        help="修复迭代次数（默认 3，仅用于统计说明）")
    parser.add_argument("--fill-loop-edges", type=int, default=50,
                        help="补洞时允许的最大边界环边数（默认 50）")
    parser.add_argument("--smooth", action="store_true",
                        help="是否做轻微 Laplacian 平滑")
    parser.add_argument("--no-repair", action="store_true",
                        help="跳过拓扑修复（仅简化）")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="输出详细统计信息")
    args = parser.parse_args()

    print(f"Loading: {args.input_file}")
    mesh = trimesh.load(args.input_file, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump(concatenate=True)
        print("Multiple meshes detected, merged.")

    # 初始统计
    if args.verbose:
        print("\n[Input mesh stats]")
        for k, v in compute_mesh_stats(mesh).items():
            print(f"  {k}: {v}")
        defect_stats, _, _ = analyze_mesh_defects(mesh)
        print("\n[Input defect analysis]")
        print(f"  open edges: {defect_stats['open_edges']}")
        print(f"  nonmanifold edges: {defect_stats['nonmanifold_edges']}")
        print(f"  watertight (by count): {defect_stats['watertight_by_count']}")

    # 简化/生成代理模型
    print(f"\nBuilding proxy mesh (target_faces={args.target_faces})...")
    if args.no_repair:
        # 不做修复，直接简化
        proxy = mesh.copy()
        if len(proxy.faces) > args.target_faces:
            proxy = proxy.simplify_quadric_decimation(face_count=args.target_faces)
        proxy.merge_vertices()
        proxy.remove_unreferenced_vertices()
    else:
        proxy = build_proxy_mesh(
            mesh,
            target_faces=args.target_faces,
            max_edge_length=args.max_edge_length,
            iterations=args.repair_iterations,
            smooth=args.smooth,
        )

    # 最终统计
    if args.verbose:
        print("\n[Proxy mesh stats]")
        for k, v in compute_mesh_stats(proxy).items():
            print(f"  {k}: {v}")
        defect_stats, _, _ = analyze_mesh_defects(proxy)
        print("\n[Proxy defect analysis]")
        print(f"  open edges: {defect_stats['open_edges']}")
        print(f"  nonmanifold edges: {defect_stats['nonmanifold_edges']}")
        print(f"  watertight (by count): {defect_stats['watertight_by_count']}")

    # 保存
    proxy.export(args.output)
    print(f"\nProxy mesh saved to: {args.output}")
    print(f"  vertices: {proxy.vertices.shape[0]}, faces: {proxy.faces.shape[0]}")


if __name__ == "__main__":
    main()
