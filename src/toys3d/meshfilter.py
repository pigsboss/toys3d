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
from toys3d.geometrics import (
    compute_mesh_stats,
    build_face_adjacency,   # 新增
)


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


def build_k_ring_vertex_neighbors(faces, n_vertices, k=1):
    """
    构建每个顶点的 k-ring 拓扑邻域顶点集合（包含自身）。

    Parameters
    ----------
    faces : (F, 3) ndarray
    n_vertices : int
    k : int
        环数，k=1 即 1-ring

    Returns
    -------
    neighbors : list of set
    """
    one_ring = build_vertex_neighbors(faces, n_vertices)
    neighbors = [set(s) for s in one_ring]

    current = [set(s) for s in one_ring]
    for _ in range(1, k):
        next_ring = [set() for _ in range(n_vertices)]
        for i in range(n_vertices):
            for nb in current[i]:
                next_ring[i].update(one_ring[nb])
            next_ring[i].discard(i)
            next_ring[i] -= neighbors[i]
            neighbors[i].update(next_ring[i])
        current = next_ring

    for i in range(n_vertices):
        neighbors[i].add(i)

    return neighbors


def build_k_ring_face_neighbors(mesh, k=1):
    """
    构建每个面片的 k-ring 拓扑邻域面片集合（包含自身）。

    Parameters
    ----------
    mesh : trimesh.Trimesh
    k : int

    Returns
    -------
    neighbor_lists : list of ndarray
    """
    adjacency = build_face_adjacency(mesh)
    N = len(mesh.faces)
    neighbors = [set([i]) for i in range(N)]
    current = [set([i]) for i in range(N)]

    for _ in range(k):
        next_ring = [set() for _ in range(N)]
        for i in range(N):
            for nb in current[i]:
                next_ring[i].update(adjacency[nb])
            next_ring[i] -= neighbors[i]
            neighbors[i].update(next_ring[i])
        current = next_ring

    return [np.array(list(s), dtype=int) for s in neighbors]


def build_face_neighbor_lists(mesh, neighborhood='topology',
                              kernel_size=1):
    """
    构建每个面片的邻域面片索引列表（包含自身）。

    Parameters
    ----------
    mesh : trimesh.Trimesh
    neighborhood : 'topology' | 'euclidean'
    kernel_size : int
        topology 模式下为 k-ring 环数，euclidean 模式下为 k 近邻数量

    Returns
    -------
    neighbor_lists : list of ndarray
    """
    N = len(mesh.faces)
    if N == 0:
        return []

    if neighborhood == 'topology':
        return build_k_ring_face_neighbors(mesh, k=kernel_size)
    else:
        centers = np.asarray(mesh.triangles_center, dtype=np.float64)
        k = min(kernel_size, N)
        tree = cKDTree(centers)
        _, idxs = tree.query(centers, k=k)
        return [idxs[i] for i in range(N)]


def vertex_median_filter_topology(vertices, faces, iterations=1,
                                  strength=1.0, kernel_ring=1, verbose=True):
    """
    基于拓扑 k-ring 邻域的顶点中值滤波。

    Parameters
    ----------
    vertices : (V, 3) ndarray
    faces : (F, 3) ndarray
    iterations : int
    strength : float
        新位置 = (1 - strength) * old + strength * median
    kernel_ring : int
        k-ring 环数，默认 1（即 1-ring）
    verbose : bool
        是否输出进度

    Returns
    -------
    vertices : (V, 3) ndarray
    """
    vertices = np.asarray(vertices, dtype=np.float64).copy()
    n_vertices = len(vertices)

    if verbose:
        print(f"  Building topology {kernel_ring}-ring neighbor list...")
    neighbors = build_k_ring_vertex_neighbors(faces, n_vertices, k=kernel_ring)
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


def filter_normals_median(normals, neighbor_lists, areas, verbose=True):
    """
    对每个面片，在邻域法向中选取加权球面中值。
    """
    N = len(normals)
    new_normals = np.empty_like(normals)
    report_interval = max(1, N // 10)
    t0 = time.time()

    for i in range(N):
        if verbose and i % report_interval == 0:
            print(f"    processing {i}/{N} ({100 * i / N:.0f}%) "
                  f"+{time.time() - t0:.2f}s")

        nb = neighbor_lists[i]
        nb_normals = normals[nb]
        nb_areas = areas[nb]

        dots = nb_normals @ nb_normals.T
        abs_dots = np.clip(np.abs(dots), 0.0, 1.0)
        costs = (1.0 - abs_dots) @ nb_areas
        best = int(np.argmin(costs))
        n = nb_normals[best]
        n_norm = np.linalg.norm(n)
        new_normals[i] = n / n_norm if n_norm > 1e-12 else normals[i]

    return new_normals


def update_vertices_from_normals(mesh, target_normals, strength=1.0):
    """
    根据目标面法向，把每个顶点向相邻面片的切平面投影并加权平均。
    """
    vertices = mesh.vertices.copy().astype(np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    centers = mesh.triangles_center
    areas = mesh.area_faces
    n = np.asarray(target_normals, dtype=np.float64)

    c0 = centers - vertices[faces[:, 0]]
    c1 = centers - vertices[faces[:, 1]]
    c2 = centers - vertices[faces[:, 2]]

    d0 = np.einsum('ij,ij->i', c0, n)
    d1 = np.einsum('ij,ij->i', c1, n)
    d2 = np.einsum('ij,ij->i', c2, n)

    disp0 = (areas * d0)[:, None] * n
    disp1 = (areas * d1)[:, None] * n
    disp2 = (areas * d2)[:, None] * n

    disp = np.zeros_like(vertices)
    wsum = np.zeros(len(vertices), dtype=np.float64)

    np.add.at(disp, faces[:, 0], disp0)
    np.add.at(disp, faces[:, 1], disp1)
    np.add.at(disp, faces[:, 2], disp2)

    np.add.at(wsum, faces[:, 0], areas)
    np.add.at(wsum, faces[:, 1], areas)
    np.add.at(wsum, faces[:, 2], areas)

    avg_disp = disp / (wsum[:, None] + 1e-12)
    return vertices + strength * avg_disp


def normal_median_filter(mesh, neighborhood='topology', kernel_size=1,
                         iterations=1, strength=1.0, verbose=True):
    """
    法向中值滤波：对面片法向做中值滤波，再反投影更新顶点。
    """
    mesh = mesh.copy()
    mesh.fix_normals()

    vertices = mesh.vertices.copy()
    faces = mesh.faces.copy()

    for it in range(iterations):
        if verbose:
            print(f"  Normal-median iteration {it + 1}/{iterations} "
                  f"(neighborhood={neighborhood})...")

        m = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        m.fix_normals()

        normals = m.face_normals.copy()
        areas = m.area_faces

        if verbose:
            print("    Building neighbor lists...")
        t0 = time.time()
        neighbor_lists = build_face_neighbor_lists(
            m, neighborhood=neighborhood, kernel_size=kernel_size
        )
        if verbose:
            print(f"    neighbor lists built in {time.time() - t0:.2f}s")

        if verbose:
            print("    Filtering face normals...")
        t0 = time.time()
        new_normals = filter_normals_median(normals, neighbor_lists, areas,
                                            verbose=verbose)
        if verbose:
            print(f"    normals filtered in {time.time() - t0:.2f}s")

        if verbose:
            print("    Updating vertex positions...")
        t0 = time.time()
        vertices = update_vertices_from_normals(
            trimesh.Trimesh(vertices=vertices, faces=faces, process=False),
            new_normals, strength=strength
        )
        if verbose:
            print(f"    vertices updated in {time.time() - t0:.2f}s")

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
                        choices=["median", "laplacian", "normal-median"],
                        help="滤波模式（默认 median；normal-median=法向中值滤波）")

    parser.add_argument("--neighborhood", type=str, default="topology",
                        choices=["topology", "euclidean"],
                        help="median 模式下邻域类型（默认 topology）")

    parser.add_argument("--kernel-size", type=int, default=1,
                        help="邻域大小：topology 模式为 k-ring 环数，"
                             "euclidean 模式为 k 近邻数量（默认 1）")

    parser.add_argument("--iterations", type=int, default=1,
                        help="迭代次数（默认 1）")

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
                kernel_ring=args.kernel_size,
                verbose=args.verbose
            )
        else:  # euclidean
            new_vertices = vertex_median_filter_euclidean(
                mesh.vertices, mesh.faces,
                k_neighbors=args.kernel_size,
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
    elif args.mode == "normal-median":
        new_vertices = normal_median_filter(
            mesh,
            neighborhood=args.neighborhood,
            kernel_size=args.kernel_size,
            iterations=args.iterations,
            strength=args.strength,
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
