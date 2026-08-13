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
    remove_isolated_components,
    remove_quasi_isolated_components,
)


def main():
    parser = argparse.ArgumentParser(
        description="清理网格：删除孤立结构和准孤立结构，"
                    "去除由点云修补产生的多余网格。"
    )
    parser.add_argument("input_file", help="输入网格文件路径")
    parser.add_argument("-o", "--output", required=True,
                        help="输出清理后网格文件路径")

    # 孤立结构移除
    parser.add_argument("--remove-isolated", action="store_true",
                        help="移除孤立连通分量")
    parser.add_argument("--min-component-faces", type=int, default=20,
                        help="孤立分量最少保留面片数（默认 20）")
    parser.add_argument("--min-component-area", type=float, default=None,
                        help="孤立分量最少保留面积（默认不启用）")
    parser.add_argument("--min-component-ratio", type=float, default=0.001,
                        help="孤立分量面片数占比阈值（默认 0.001）")

    # 准孤立结构移除
    parser.add_argument("--remove-quasi-isolated", action="store_true",
                        help="移除准孤立结构")
    parser.add_argument("--quasi-radius", type=float, default=None,
                        help="准孤立检测球邻域半径（默认自动估计）")
    parser.add_argument("--quasi-sample", type=int, default=2000,
                        help="准孤立检测随机采样球心数量（默认 2000）")
    parser.add_argument("--quasi-min-faces", type=int, default=30,
                        help="准孤立子结构最少面片数（默认 30）")
    parser.add_argument("--quasi-max-ratio", type=float, default=0.05,
                        help="准孤立子结构面片占比上限（默认 0.05）")
    parser.add_argument("--quasi-remove-bridge", action="store_true",
                        help="同时删除连接颈部的球内面片")
    parser.add_argument("--quasi-rng-seed", type=int, default=None,
                        help="随机种子，便于复现")

    parser.add_argument("-v", "--verbose", action="store_true",
                        help="输出详细统计和进度")
    args = parser.parse_args()

    if not args.remove_isolated and not args.remove_quasi_isolated:
        print("No cleaning operation selected. "
              "Use --remove-isolated and/or --remove-quasi-isolated.")
        return

    print(f"Loading: {args.input_file}")
    mesh = trimesh.load(args.input_file, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump(concatenate=True)
        print("Multiple meshes detected, merged.")

    if args.verbose:
        print("\n[Input stats]")
        for k, v in compute_mesh_stats(mesh).items():
            print(f"  {k}: {v}")

    cleaned = mesh.copy()

    if args.remove_isolated:
        print("\nRemoving isolated components...")
        cleaned = remove_isolated_components(
            cleaned,
            min_faces=args.min_component_faces,
            min_area=args.min_component_area,
            min_ratio=args.min_component_ratio,
            verbose=args.verbose,
        )

    if args.remove_quasi_isolated:
        print("\nRemoving quasi-isolated components...")
        rng = np.random.default_rng(args.quasi_rng_seed) \
            if args.quasi_rng_seed is not None else None
        cleaned = remove_quasi_isolated_components(
            cleaned,
            radius=args.quasi_radius,
            n_sample=args.quasi_sample,
            min_faces=args.quasi_min_faces,
            max_ratio=args.quasi_max_ratio,
            remove_bridge=args.quasi_remove_bridge,
            rng=rng,
            verbose=args.verbose,
        )

    # 最终清理
    cleaned = cleaned.copy()
    cleaned.merge_vertices()
    cleaned.remove_unreferenced_vertices()
    cleaned.fix_normals()

    if args.verbose:
        print("\n[Output stats]")
        for k, v in compute_mesh_stats(cleaned).items():
            print(f"  {k}: {v}")

    cleaned.export(args.output)
    print(f"\nCleaned mesh saved to: {args.output}")
    print(f"  vertices: {cleaned.vertices.shape[0]}, "
          f"faces: {cleaned.faces.shape[0]}")


if __name__ == "__main__":
    main()
