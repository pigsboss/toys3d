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
    trim_isolated_faces,
    repair_to_watertight,
)


def main():
    parser = argparse.ArgumentParser(
        description="检查并修复网格拓扑缺陷。"
    )
    parser.add_argument("input_file", help="输入网格文件路径")
    parser.add_argument("-o", "--output", required=True,
                        help="输出修复后网格文件路径")

    parser.add_argument("--trim-isolated-faces", action="store_true",
                        help="删除完全孤立的面片（开放边数 == 3）")

    parser.add_argument("--watertight", action="store_true",
                        help="使用体素化 + Marching Cubes 重建水密外壳")
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

    print(f"Yo! Loading {args.input_file}")
    mesh = trimesh.load(args.input_file, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump(concatenate=True)
        print("Multiple meshes detected, merged.")

    if args.verbose:
        print("\n[Input mesh stats]")
        for k, v in compute_mesh_stats(mesh).items():
            print(f"  {k}: {v}")

    repaired = mesh.copy()

    if args.trim_isolated_faces:
        if args.verbose:
            print("\n[trim-isolated-faces]")
            before, _, _ = analyze_mesh_defects(repaired)
            print(f"  open_edges before trim: {before['open_edges']}")
        repaired = trim_isolated_faces(repaired, verbose=args.verbose)
        if args.verbose:
            after, _, _ = analyze_mesh_defects(repaired)
            print(f"  open_edges after trim:  {after['open_edges']}")

    if args.watertight:
        if args.verbose:
            print("\n[watertight]")
        repaired = repair_to_watertight(
            repaired,
            resolution=args.watertight_resolution,
            voxel_size=args.watertight_voxel_size,
            project_to_input=args.watertight_project_to_input,
            smooth_watertight=args.watertight_smooth,
            smooth_iterations=args.watertight_smooth_iterations,
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
