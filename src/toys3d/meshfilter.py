import sys
import os

# Ensure src directory is on the path so that 'toys3d' can be imported
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_src_parent = os.path.dirname(_project_root)
if _src_parent not in sys.path:
    sys.path.insert(0, _src_parent)

import argparse
import time
import numpy as np
import trimesh

from scipy.spatial import cKDTree
from toys3d.geometrics import compute_mesh_stats


# ------------------------------------------------------------------
#  工具函数
# ------------------------------------------------------------------

def build_vertex_neighbors(faces, n_vertices):
    """
    从面片索引构建每个顶点的 1-ring 邻域顶点集合。
    """
    neighbors = [set() for _ in range(n_vertices)]
    for f in faces:
        a, b, c = int(f[0]), int(f[1]), int(f[2])
        neighbors[a].update([b, c])
        neighbors[b].update([a, c])
        neighbors[c].update([a, b])
    return neighbors


def vertex_median_filter_topology(vertices, faces, iterations=1,
                                  strength=1.0, verbose=True):
    """
    基于拓扑 1-ring 邻域的顶点中值滤波。

    Parameters
    ----------
    vertices : (V, 3) ndarray
    faces : (F, 3) ndarray
    iterations : int
    strength : float
        新位置 = (1 - strength) * old + strength * median
    verbose : bool
        是否输出进度

    Returns
    -------
    vertices : (V, 3) ndarray
    """
    vertices = np.asarray(vertices, dtype=np.float64).copy()
    n_vertices = len(vertices)

    if verbose:
        print("  Building topology neighbor list...")
    neighbors = build_vertex_neighbors(faces, n_vertices)
    if verbose:
        print(f"    done, n_vertices={n_vertices}")

    report_interval = max(1, n_vertices // 10)

    for it in range(iterations):
        if verbose:
            print(f"  Median iteration {it + 1}/{iterations}...")
        t0 = time.time()
        new_vertices = np.empty_like(vertices)

        for i in range(n_vertices):
            if verbose and i % report_interval == 0:
                print(f"    processing {i}/{n_vertices} "
                      f"({100 * i / n_vertices:.0f}%) "
                      f"+{time.time() - t0:.2f}s")
            nb = list(neighbors[i]) + [i]
            median_pos = np.median(vertices[nb], axis=0)
            new_vertices[i] = (1.0 - strength) * vertices[i] + strength * median_pos

        if verbose:
            print(f"    iteration done in {time.time() - t0:.2f}s")
        vertices = new_vertices

    return vertices


def vertex_median_filter_euclidean(vertices, faces, k_neighbors=10,
                                   iterations=1, strength=1.0, verbose=True):
    """
    基于 k-d 树最近邻的顶点中值滤波。

    Parameters
    ----------
    vertices : (V, 3) ndarray
    faces : (F, 3) ndarray
    k_neighbors : int
    iterations : int
    strength : float
    verbose : bool

    Returns
    -------
    vertices : (V, 3) ndarray
    """
    vertices = np.asarray(vertices, dtype=np.float64).copy()
    n_vertices = len(vertices)
    k = min(k_neighbors, n_vertices)

    report_interval = max(1, n_vertices // 10)

    for it in range(iterations):
        if verbose:
            print(f"  Building k-d tree for iteration {it + 1}/{iterations}...")
        t0 = time.time()
        tree = cKDTree(vertices)
        _, idxs = tree.query(vertices, k=k)
        if verbose:
            print(f"    k-d tree built in {time.time() - t0:.2f}s")

        if verbose:
            print(f"  Median iteration {it + 1}/{iterations}...")
        t0 = time.time()
        new_vertices = np.empty_like(vertices)

        for i in range(n_vertices):
            if verbose and i % report_interval == 0:
                print(f"    processing {i}/{n_vertices} "
                      f"({100 * i / n_vertices:.0f}%) "
                      f"+{time.time() - t0:.2f}s")
            median_pos = np.median(vertices[idxs[i]], axis=0)
            new_vertices[i] = (1.0 - strength) * vertices[i] + strength * median_pos

        if verbose:
            print(f"    iteration done in {time.time() - t0:.2f}s")
        vertices = new_vertices

    return vertices


def laplacian_smooth(vertices, faces, iterations=1, alpha=0.5, verbose=True):
    """
    经典 Laplacian 平滑。

    Parameters
    ----------
    vertices : (V, 3) ndarray
    faces : (F, 3) ndarray
    iterations : int
    alpha : float
    verbose : bool

    Returns
    -------
    vertices : (V, 3) ndarray
    """
    vertices = np.asarray(vertices, dtype=np.float64).copy()
    n_vertices = len(vertices)

    if verbose:
        print("  Building topology neighbor list...")
    neighbors = build_vertex_neighbors(faces, n_vertices)
    if verbose:
        print(f"    done, n_vertices={n_vertices}")

    report_interval = max(1, n_vertices // 10)

    for it in range(iterations):
        if verbose:
            print(f"  Laplacian iteration {it + 1}/{iterations}...")
        t0 = time.time()
        new_vertices = vertices.copy()

        for i in range(n_vertices):
            if verbose and i % report_interval == 0:
                print(f"    processing {i}/{n_vertices} "
                      f"({100 * i / n_vertices:.0f}%) "
                      f"+{time.time() - t0:.2f}s")
            nb = list(neighbors[i])
            if len(nb) == 0:
                continue
            centroid = np.mean(vertices[nb], axis=0)
            new_vertices[i] = (1.0 - alpha) * vertices[i] + alpha * centroid

        if verbose:
            print(f"    iteration done in {time.time() - t0:.2f}s")
        vertices = new_vertices

    return vertices


# ------------------------------------------------------------------
#  主程序
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="网格滤波/去噪/平滑处理工具。"
    )
    parser.add_argument("input_file", help="输入网格文件路径")
    parser.add_argument("-o", "--output", required=True,
                        help="输出网格文件路径")

    parser.add_argument("--mode", type=str, default="median",
                        choices=["median", "laplacian"],
                        help="滤波模式（默认 median）")

    parser.add_argument("--neighborhood", type=str, default="topology",
                        choices=["topology", "euclidean"],
                        help="median 模式下邻域类型（默认 topology）")

    parser.add_argument("--iterations", type=int, default=1,
                        help="迭代次数（默认 1）")

    parser.add_argument("--k-neighbors", type=int, default=10,
                        help="euclidean 模式下最近邻数量（默认 10）")

    parser.add_argument("--strength", type=float, default=1.0,
                        help="中值滤波强度，0~1（默认 1.0）")

    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Laplacian 平滑强度，0~1（默认 0.5）")

    parser.add_argument("-v", "--verbose", action="store_true",
                        help="输出处理前后的统计信息和进度")

    args = parser.parse_args()

    print(f"Loading: {args.input_file}")
    mesh = trimesh.load(args.input_file, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump(concatenate=True)
        print("Multiple meshes detected, merged.")

    if args.verbose:
        print("\n[Input stats]")
        for k, v in compute_mesh_stats(mesh).items():
            print(f"  {k}: {v}")

    print(f"\nApplying {args.mode} filter "
          f"(neighborhood={args.neighborhood}, iterations={args.iterations})...")

    if args.mode == "median":
        if args.neighborhood == "topology":
            new_vertices = vertex_median_filter_topology(
                mesh.vertices, mesh.faces,
                iterations=args.iterations,
                strength=args.strength,
                verbose=args.verbose
            )
        else:  # euclidean
            new_vertices = vertex_median_filter_euclidean(
                mesh.vertices, mesh.faces,
                k_neighbors=args.k_neighbors,
                iterations=args.iterations,
                strength=args.strength,
                verbose=args.verbose
            )
    elif args.mode == "laplacian":
        new_vertices = laplacian_smooth(
            mesh.vertices, mesh.faces,
            iterations=args.iterations,
            alpha=args.alpha,
            verbose=args.verbose
        )
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    result = trimesh.Trimesh(vertices=new_vertices,
                             faces=mesh.faces.copy(),
                             process=False)

    if args.verbose:
        print("\n[Output stats]")
        for k, v in compute_mesh_stats(result).items():
            print(f"  {k}: {v}")

    result.export(args.output)
    print(f"\nFiltered mesh saved to: {args.output}")
    print(f"  vertices: {result.vertices.shape[0]}, faces: {result.faces.shape[0]}")


if __name__ == "__main__":
    main()
