# src/toys3d/geometrics/utilities.py
"""
综合几何工具：孔洞统计、修复、点云拟合等。
"""
import numpy as np
import trimesh
from scipy.sparse import csr_matrix
from collections import deque

from .euclidean import polygon_area_from_3d_ccw
from .topology import (
    extract_boundary_loops,
    analyze_mesh_defects,
    compute_open_edge_data,
    compute_face_edge_keys,
    compute_edge_to_faces,
    build_manifold_face_adjacency,
    is_manifold_closed_boundary,
    _extract_boundary_loops_from_edge_keys,
    _expand_face_neighborhood_geometrics,
)
from .discrete import _repair_to_watertight_mesh


def compute_hole_area_stats(mesh):
    loops = extract_boundary_loops(mesh)
    if not loops:
        return {
            'count': 0, 'total_area': 0.0,
            'p1_area': 0.0, 'p5_area': 0.0, 'p25_area': 0.0,
            'p50_area': 0.0, 'p75_area': 0.0, 'p90_area': 0.0,
            'p95_area': 0.0, 'p99_area': 0.0, 'max_area': 0.0,
        }

    areas = []
    for loop in loops:
        pts = mesh.vertices[np.asarray(loop, dtype=np.int64)]
        areas.append(polygon_area_from_3d_ccw(pts))
    areas = np.array(areas)

    return {
        'count': len(loops),
        'total_area': float(areas.sum()),
        'p1_area': float(np.percentile(areas, 1)),
        'p5_area': float(np.percentile(areas, 5)),
        'p25_area': float(np.percentile(areas, 25)),
        'p50_area': float(np.percentile(areas, 50)),
        'p75_area': float(np.percentile(areas, 75)),
        'p90_area': float(np.percentile(areas, 90)),
        'p95_area': float(np.percentile(areas, 95)),
        'p99_area': float(np.percentile(areas, 99)),
        'max_area': float(areas.max()),
    }


def repair_mesh_by_removing_duplicates(mesh):
    m = mesh.copy()
    m.merge_vertices()
    m.remove_unreferenced_vertices()

    if len(m.faces) > 0:
        area = m.area_faces
        degenerate = area < 1e-12
        if np.any(degenerate):
            mask = ~degenerate
            m.update_faces(mask)
            m.remove_unreferenced_vertices()
    return m


def project_vertices_to_shell(points, mesh):
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(pts) == 0:
        return np.zeros((0, 3)), np.zeros(0), np.zeros(0, dtype=np.int64)
    closest, distance, triangle_id = mesh.nearest.on_surface(pts)
    return closest, distance, triangle_id


def weld_small_holes(mesh, threshold=None, quantile=5.0, min_edges=3, verbose=False):
    m = mesh.copy()
    loops = extract_boundary_loops(m)
    if not loops:
        return m

    areas = np.array([polygon_area_from_3d_ccw(m.vertices[np.asarray(l)])
                      for l in loops])
    if threshold is None:
        threshold = float(np.percentile(areas, quantile)) if len(areas) > 0 else 0.0

    weld_loops = [l for l, a in zip(loops, areas)
                  if a < threshold and len(l) >= min_edges]
    if not weld_loops:
        return m

    if verbose:
        print(f"  weld threshold: {threshold:.6f}")
        print(f"  welding {len(weld_loops)} small holes")

    old_to_new = {}
    extra_vertices = list(m.vertices)

    for loop in weld_loops:
        ids = np.asarray(loop, dtype=np.int64)
        centroid = np.mean(m.vertices[ids], axis=0)

        new_idx = len(extra_vertices)
        extra_vertices.append(centroid)

        for v in ids:
            old_to_new[int(v)] = new_idx

    n_old = len(m.vertices)
    lookup = np.arange(n_old, dtype=np.int64)
    for old_v, new_v in old_to_new.items():
        lookup[old_v] = new_v

    new_vertices = np.asarray(extra_vertices, dtype=np.float64)

    faces = np.asarray(m.faces, dtype=np.int64)
    flat_faces = faces.ravel()
    new_flat = lookup[flat_faces]
    new_faces = new_flat.reshape(-1, 3)

    valid = ~((new_faces[:, 0] == new_faces[:, 1]) |
              (new_faces[:, 1] == new_faces[:, 2]) |
              (new_faces[:, 2] == new_faces[:, 0]))
    new_faces = new_faces[valid]

    m2 = trimesh.Trimesh(vertices=new_vertices, faces=new_faces, process=False)
    m2.remove_unreferenced_vertices()
    m2.merge_vertices()
    return repair_mesh_by_removing_duplicates(m2)


def trim_isolated_faces(mesh, verbose=False):
    m = mesh.copy()

    (
        defect_stats,
        open_face_mask,
        nonmanifold_face_mask,
        open_edge_per_face,
        manifold_edge_per_face,
        nonmanifold_edge_per_face,
    ) = analyze_mesh_defects(m, return_face_edge_counts=True)

    bad_mask = open_edge_per_face == 3
    count_removed = int(np.sum(bad_mask))

    if verbose:
        print(
            f"  [trim_isolated_faces] found {count_removed} isolated faces "
            f"(3 open edges)"
        )

        if count_removed > 0:
            bad_face_indices = np.where(bad_mask)[0]
            bad_faces = m.faces[bad_face_indices]
            bad_areas = m.area_faces[bad_face_indices]

            print("    area stats:")
            print(f"      min:  {bad_areas.min():.6f}")
            print(f"      max:  {bad_areas.max():.6f}")
            print(f"      mean: {bad_areas.mean():.6f}")
            print(f"      p50:  {np.percentile(bad_areas, 50):.6f}")
            print(f"      p90:  {np.percentile(bad_areas, 90):.6f}")
            print(f"      p99:  {np.percentile(bad_areas, 99):.6f}")

            all_face_vertices = m.faces.ravel()
            vertex_face_counts = np.bincount(
                all_face_vertices, minlength=len(m.vertices)
            )

            counts_per_face = vertex_face_counts[bad_faces]
            is_solo = counts_per_face == 1
            solo_counts = is_solo.sum(axis=1)

            n_all_shared = int(np.sum(solo_counts == 0))
            n_one_solo   = int(np.sum(solo_counts == 1))
            n_two_solo   = int(np.sum(solo_counts == 2))
            n_all_solo   = int(np.sum(solo_counts == 3))

            print("    vertex sharing (among all faces):")
            print(f"      0 solo vertices (all shared):            {n_all_shared}")
            print(f"      1 solo vertex:                          {n_one_solo}")
            print(f"      2 solo vertices:                        {n_two_solo}")
            print(f"      3 solo vertices (fully isolated):       {n_all_solo}")

            mean_ref_count = counts_per_face.mean()
            print(f"    mean vertex reference count (in these faces): {mean_ref_count:.2f}")

    if count_removed > 0:
        m.update_faces(~bad_mask)
        m.remove_unreferenced_vertices()

    return m


def fix_winding_consistency(mesh):
    m = mesh.copy()
    faces = np.asarray(m.faces, dtype=np.int64)
    n_faces = len(faces)
    if n_faces == 0:
        return 0, m

    edge_map = {}
    for fid, face in enumerate(faces):
        for i in range(3):
            v0, v1 = face[i], face[(i + 1) % 3]
            key = (v0, v1) if v0 < v1 else (v1, v0)
            edge_map.setdefault(key, []).append(fid)

    adjacency = [[] for _ in range(n_faces)]
    for key, face_list in edge_map.items():
        if len(face_list) != 2:
            continue
        f0, f1 = face_list
        i0 = next(i for i in range(3) if
                  (faces[f0][i], faces[f0][(i+1)%3]) == key or
                  (faces[f0][(i+1)%3], faces[f0][i]) == key)
        i1 = next(i for i in range(3) if
                  (faces[f1][i], faces[f1][(i+1)%3]) == key or
                  (faces[f1][(i+1)%3], faces[f1][i]) == key)
        adjacency[f0].append((f1, key, i0, i1))
        adjacency[f1].append((f0, key, i1, i0))

    visited = np.zeros(n_faces, dtype=bool)
    flip = np.zeros(n_faces, dtype=bool)

    for start in range(n_faces):
        if visited[start]:
            continue
        queue = deque([start])
        visited[start] = True
        while queue:
            fid = queue.popleft()
            for neighbor, key, local_idx, neighbor_local_idx in adjacency[fid]:
                if visited[neighbor]:
                    continue
                face_fid = faces[fid]
                dir1 = (face_fid[local_idx], face_fid[(local_idx + 1) % 3])
                face_neighbor = faces[neighbor]
                dir2 = (face_neighbor[neighbor_local_idx],
                        face_neighbor[(neighbor_local_idx + 1) % 3])
                same_orientation = (dir1[0] == dir2[0] and dir1[1] == dir2[1])
                flip[neighbor] = flip[fid] if same_orientation else not flip[fid]
                visited[neighbor] = True
                queue.append(neighbor)

    flipped_count = 0
    for fid in range(n_faces):
        if flip[fid]:
            faces[fid] = faces[fid][::-1]
            flipped_count += 1

    m = trimesh.Trimesh(vertices=m.vertices, faces=faces, process=False)
    return flipped_count, m


def _check_watertight_genus0(mesh):
    if not mesh.is_watertight:
        return False
    try:
        euler = mesh.euler_number
    except AttributeError:
        V = len(mesh.vertices)
        E = len(mesh.edges_unique)
        F = len(mesh.faces)
        euler = V - E + F
    return abs(euler - 2) < 1e-6


def _extract_component_point_cloud(mesh, component, neighborhood_depth):
    from collections import deque  # not used directly but kept for consistency

    seed_faces = set(map(int, component.get('face_ids', [])))
    if not seed_faces:
        return np.zeros((0, 3)), np.zeros((0, 3))

    expanded = _expand_face_neighborhood_geometrics(
        mesh, seed_faces, neighborhood_depth
    )
    if not expanded:
        expanded = seed_faces

    faces_idx = np.asarray(sorted(expanded), dtype=np.int64)
    submesh = mesh.submesh([faces_idx])[0]

    verts = submesh.vertices
    if hasattr(submesh, 'vertex_normals') and submesh.vertex_normals is not None:
        vert_normals = submesh.vertex_normals
    else:
        face_normals = submesh.face_normals
        vertex_face_count = np.bincount(submesh.faces.ravel(),
                                        minlength=len(verts))
        vert_normals = np.zeros_like(verts)
        for i, face in enumerate(submesh.faces):
            for v in face:
                vert_normals[v] += face_normals[i]
        norms = np.linalg.norm(vert_normals, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        vert_normals = vert_normals / norms

    face_centers = submesh.triangles_center
    face_normals = submesh.face_normals

    points = np.vstack([verts, face_centers])
    normals = np.vstack([vert_normals, face_normals])

    centroid = points.mean(axis=0)
    to_centroid = centroid - points
    dot = np.sum(normals * to_centroid, axis=1)
    flip = dot < 0
    normals[flip] *= -1

    return points, normals


def fit_watertight_patch_from_component(
    mesh,
    component,
    method='poisson',
    neighborhood_depth=2,
    poisson_depth=8,
    density_quantile=0.2,
    alpha=1.5,
    allow_non_genus0=False,
):
    from collections import deque  # not directly used but kept

    points, normals = _extract_component_point_cloud(
        mesh, component, neighborhood_depth
    )
    print(f"  [DEBUG] 点云点数: {len(points)}")
    if len(points) > 0:
        print(f"  [DEBUG] 点云坐标范围: min={points.min(axis=0)}, max={points.max(axis=0)}")
    if len(points) < 4:
        return {
            'success': False,
            'message': '点云点数不足，无法拟合曲面',
            'watertight_mesh': None,
            'intersection_vertices': [],
            'intersection_edges': [],
        }

    watertight_mesh = None
    message = ''
    try:
        if method == 'poisson':
            try:
                import open3d as o3d
            except ImportError:
                return {
                    'success': False,
                    'message': '未安装 Open3D，无法使用泊松重建',
                    'watertight_mesh': None,
                    'intersection_vertices': [],
                    'intersection_edges': [],
                }

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.normals = o3d.utility.Vector3dVector(normals)

            mesh_poisson, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=poisson_depth, scale=1.1, linear_fit=False
            )
            print(f"  [DEBUG] Open3D 泊松重建完成，输出顶点数: {np.asarray(mesh_poisson.vertices).shape[0]}, "
                  f"面数: {np.asarray(mesh_poisson.triangles).shape[0]}")

            densities = np.asarray(densities)
            if density_quantile > 0 and len(densities) > 0:
                threshold = np.quantile(densities, density_quantile)
                vertices_to_remove = densities < threshold
                mesh_poisson.remove_vertices_by_mask(vertices_to_remove)

            watertight_mesh = trimesh.Trimesh(
                vertices=np.asarray(mesh_poisson.vertices),
                faces=np.asarray(mesh_poisson.triangles),
                process=False,
            )
            print(f"  [DEBUG] 密度过滤后顶点数: {len(watertight_mesh.vertices)}, 面数: {len(watertight_mesh.faces)}")
            print(f"  [DEBUG] 水密性: {watertight_mesh.is_watertight}, 欧拉数: {watertight_mesh.euler_number}")
            if not _check_watertight_genus0(watertight_mesh):
                print(f"  [DEBUG] 原始结果: is_watertight={watertight_mesh.is_watertight}, "
                      f"euler={watertight_mesh.euler_number}")
                watertight_mesh = _repair_to_watertight_mesh(watertight_mesh)
                print(f"  [DEBUG] 修复后: is_watertight={watertight_mesh.is_watertight}, "
                      f"euler={watertight_mesh.euler_number}")
                if not _check_watertight_genus0(watertight_mesh):
                    if not allow_non_genus0:
                        return {
                            'success': False,
                            'message': '泊松重建结果未通过水密/亏格0检查',
                            'watertight_mesh': None,
                            'intersection_vertices': [],
                            'intersection_edges': [],
                        }
                    else:
                        print("  [WARN] 允许非亏格0水密曲面（allow_non_genus0=True）")
                        message = '泊松重建成功（水密但亏格非0）'
                else:
                    message = '泊松重建成功'
            else:
                message = '泊松重建成功'

            flipped, fixed_mesh = fix_winding_consistency(watertight_mesh)
            watertight_mesh = fixed_mesh
            print(f"  [DEBUG] 泊松重建绕序修正：翻转 {flipped} 个面片")
            print(f"  [DEBUG] 修正后水密性: {watertight_mesh.is_watertight}, "
                  f"欧拉数: {watertight_mesh.euler_number}")

        elif method == 'convex_hull':
            try:
                from scipy.spatial import ConvexHull
            except ImportError:
                return {
                    'success': False,
                    'message': '未安装 scipy，无法使用凸包算法',
                    'watertight_mesh': None,
                    'intersection_vertices': [],
                    'intersection_edges': [],
                }
            hull = ConvexHull(points)
            watertight_mesh = trimesh.Trimesh(
                vertices=hull.points,
                faces=hull.simplices,
                process=False,
            )
            message = '凸包生成成功'

            flipped, fixed_mesh = fix_winding_consistency(watertight_mesh)
            watertight_mesh = fixed_mesh
            print(f"  [DEBUG] 凸包绕序修正：翻转 {flipped} 个面片")
            print(f"  [DEBUG] 修正后水密性: {watertight_mesh.is_watertight}, "
                  f"欧拉数: {watertight_mesh.euler_number}")

        elif method == 'concave_hull':
            try:
                import open3d as o3d
            except ImportError:
                return {
                    'success': False,
                    'message': '未安装 Open3D，无法使用凹包算法',
                    'watertight_mesh': None,
                    'intersection_vertices': [],
                    'intersection_edges': [],
                }

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.normals = o3d.utility.Vector3dVector(normals)

            mesh_alpha = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
                pcd, alpha
            )
            print(f"  [DEBUG] Open3D Alpha Shape 完成，输出顶点数: {np.asarray(mesh_alpha.vertices).shape[0]}, "
                  f"面数: {np.asarray(mesh_alpha.triangles).shape[0]}")

            watertight_mesh = trimesh.Trimesh(
                vertices=np.asarray(mesh_alpha.vertices),
                faces=np.asarray(mesh_alpha.triangles),
                process=False,
            )
            print(f"  [DEBUG] 凹包结果顶点数: {len(watertight_mesh.vertices)}, 面数: {len(watertight_mesh.faces)}")
            print(f"  [DEBUG] 水密性: {watertight_mesh.is_watertight}, 欧拉数: {watertight_mesh.euler_number}")
            if not _check_watertight_genus0(watertight_mesh):
                print(f"  [DEBUG] 原始结果: is_watertight={watertight_mesh.is_watertight}, "
                      f"euler={watertight_mesh.euler_number}")
                watertight_mesh = _repair_to_watertight_mesh(watertight_mesh)
                print(f"  [DEBUG] 修复后: is_watertight={watertight_mesh.is_watertight}, "
                      f"euler={watertight_mesh.euler_number}")
                if not _check_watertight_genus0(watertight_mesh):
                    if not allow_non_genus0:
                        return {
                            'success': False,
                            'message': '凹包结果未通过水密/亏格0检查',
                            'watertight_mesh': None,
                            'intersection_vertices': [],
                            'intersection_edges': [],
                        }
                    else:
                        print("  [WARN] 允许非亏格0水密曲面（allow_non_genus0=True）")
                        message = '凹包生成成功（水密但亏格非0）'
                else:
                    message = '凹包生成成功'
            else:
                message = '凹包生成成功'

            flipped, fixed_mesh = fix_winding_consistency(watertight_mesh)
            watertight_mesh = fixed_mesh
            print(f"  [DEBUG] 凹包绕序修正：翻转 {flipped} 个面片")
            print(f"  [DEBUG] 修正后水密性: {watertight_mesh.is_watertight}, "
                  f"欧拉数: {watertight_mesh.euler_number}")

        else:
            return {
                'success': False,
                'message': f'未知算法: {method}',
                'watertight_mesh': None,
                'intersection_vertices': [],
                'intersection_edges': [],
            }
    except Exception as e:
        return {
            'success': False,
            'message': str(e),
            'watertight_mesh': None,
            'intersection_vertices': [],
            'intersection_edges': [],
        }

    seed_faces = set(map(int, component.get('face_ids', [])))
    expanded = _expand_face_neighborhood_geometrics(
        mesh, seed_faces, neighborhood_depth
    )
    if not expanded:
        expanded = seed_faces
    submesh = mesh.submesh([np.asarray(sorted(expanded), dtype=np.int64)])[0]
    boundary_loops = extract_boundary_loops(submesh)
    print(f"  [DEBUG] 邻域子网格边界环数量: {len(boundary_loops)}")

    intersection_vertices = []
    intersection_edges = []

    for loop in boundary_loops:
        loop_verts = np.asarray(loop, dtype=np.int64)
        pts = submesh.vertices[loop_verts]
        proj_pts, _, _ = watertight_mesh.nearest.on_surface(pts)
        start_idx = len(intersection_vertices)
        for p in proj_pts:
            intersection_vertices.append(p.tolist())
        for i in range(len(loop)):
            v0 = start_idx + i
            v1 = start_idx + (i + 1) % len(loop)
            intersection_edges.append([v0, v1])

    print(f"  [DEBUG] 提取交线顶点数: {len(intersection_vertices)}, 边数: {len(intersection_edges)}")
    if not intersection_vertices:
        return {
            'success': False,
            'message': '未能提取到交线',
            'watertight_mesh': None,
            'intersection_vertices': [],
            'intersection_edges': [],
        }

    return {
        'success': True,
        'message': message,
        'watertight_mesh': watertight_mesh,
        'intersection_vertices': intersection_vertices,
        'intersection_edges': intersection_edges,
    }


def build_hole_diagnosis_data(mesh):
    open_data = compute_open_edge_data(mesh)
    vertex_pairs = open_data['open_edge_vertex_pairs']
    edge_keys = open_data['open_edge_keys']
    key_to_id = open_data['open_edge_key_to_id']

    E = len(vertex_pairs)
    hole_ids_per_edge = np.full(E, -1, dtype=np.int32)
    hole_vertex_lists = []
    hole_edge_lists = []

    loops = extract_boundary_loops(mesh)

    for loop in loops:
        loop_vertices = [int(v) for v in loop]
        edge_ids = []
        valid = True
        for i in range(len(loop_vertices)):
            v0 = loop_vertices[i]
            v1 = loop_vertices[(i + 1) % len(loop_vertices)]
            min_v = min(v0, v1)
            max_v = max(v0, v1)
            key = min_v * (len(mesh.vertices) + 1) + max_v
            if key not in key_to_id:
                valid = False
                break
            edge_ids.append(key_to_id[key])
        if not valid:
            continue

        hole_id = len(hole_vertex_lists)
        hole_vertex_lists.append(loop_vertices)
        hole_edge_lists.append(edge_ids)
        for eid in edge_ids:
            hole_ids_per_edge[eid] = hole_id

    uncovered = np.where(hole_ids_per_edge == -1)[0].astype(np.int64)

    degree = open_data['vertex_degree']
    _, _, nonmanifold_face_mask = analyze_mesh_defects(mesh)

    categories = np.zeros(len(uncovered), dtype=np.int8)
    for idx, edge_id in enumerate(uncovered):
        v0, v1 = vertex_pairs[edge_id]
        d0 = degree[v0]
        d1 = degree[v1]

        face_id = open_data['open_edge_face_ids'][edge_id]
        if nonmanifold_face_mask[face_id]:
            categories[idx] = 4
            continue

        if d0 == 1 and d1 == 1:
            categories[idx] = 0
        elif (d0 == 1 and d1 >= 2) or (d1 == 1 and d0 >= 2):
            categories[idx] = 1
        elif d0 >= 2 and d1 >= 2:
            categories[idx] = 2
        else:
            categories[idx] = 5

    return {
        **open_data,
        'hole_vertex_lists': hole_vertex_lists,
        'hole_edge_lists': hole_edge_lists,
        'hole_ids_per_edge': hole_ids_per_edge,
        'uncovered_edge_ids': uncovered,
        'uncovered_category': categories,
        'nonmanifold_face_mask': nonmanifold_face_mask,
    }


def analyze_uncovered_open_edge_components(mesh, hole_data, spatial_threshold=None):
    from collections import deque
    from scipy.spatial import cKDTree

    uncovered_ids = hole_data['uncovered_edge_ids']
    all_vertex_pairs = hole_data['open_edge_vertex_pairs']
    all_face_ids = hole_data['open_edge_face_ids']
    vertex_csr = hole_data['vertex_open_edges_csr']
    hole_ids_per_edge = hole_data['hole_ids_per_edge']
    nonmanifold_face_mask = hole_data['nonmanifold_face_mask']

    U = len(uncovered_ids)
    if U == 0:
        return []

    uncovered_index = np.full(len(hole_ids_per_edge), -1, dtype=np.int64)
    uncovered_index[uncovered_ids] = np.arange(U, dtype=np.int64)

    visited = np.zeros(U, dtype=bool)
    components = []

    for start in range(U):
        if visited[start]:
            continue
        comp_indices = []
        queue = deque([start])
        visited[start] = True

        while queue:
            ue_idx = queue.popleft()
            comp_indices.append(ue_idx)
            global_eid = uncovered_ids[ue_idx]
            v0, v1 = all_vertex_pairs[global_eid]

            for v in (v0, v1):
                row_start = vertex_csr.indptr[v]
                row_end = vertex_csr.indptr[v + 1]
                for j in range(row_start, row_end):
                    gid = vertex_csr.indices[j]
                    if hole_ids_per_edge[gid] != -1:
                        continue
                    nu = uncovered_index[gid]
                    if nu == -1 or visited[nu]:
                        continue
                    visited[nu] = True
                    queue.append(int(nu))

        comp_local = np.array(comp_indices, dtype=np.int64)
        comp_global = uncovered_ids[comp_local]
        comp_edges = all_vertex_pairs[comp_global]
        comp_face_ids = np.unique(all_face_ids[comp_global])

        verts = comp_edges.ravel()
        unique_verts, counts = np.unique(verts, return_counts=True)
        endpoints = unique_verts[counts == 1]
        branch_vertices = unique_verts[counts >= 3]
        is_cycle = bool(np.all(counts == 2) and len(unique_verts) == len(comp_global))

        candidate_breaks = []
        if len(endpoints) >= 2:
            endpoint_pts = mesh.vertices[endpoints]
            if spatial_threshold is None:
                edge_lengths = np.linalg.norm(
                    mesh.vertices[comp_edges[:, 1]] - mesh.vertices[comp_edges[:, 0]],
                    axis=1
                )
                if len(edge_lengths) == 0:
                    spatial_threshold = 0.0
                else:
                    spatial_threshold = float(np.median(edge_lengths)) * 3.0

            if spatial_threshold > 0:
                tree = cKDTree(endpoint_pts)
                pairs = tree.query_pairs(spatial_threshold, output_type='ndarray')

                direct_edges = set()
                for a, b in comp_edges:
                    direct_edges.add((int(a), int(b)))
                    direct_edges.add((int(b), int(a)))

                for i, j in pairs:
                    u = int(endpoints[i])
                    v = int(endpoints[j])
                    if (u, v) in direct_edges or (v, u) in direct_edges:
                        continue
                    dist = float(np.linalg.norm(mesh.vertices[u] - mesh.vertices[v]))
                    candidate_breaks.append({'v0': u, 'v1': v, 'distance': dist})

        nonmanifold_face_count = int(np.sum(nonmanifold_face_mask[comp_face_ids]))
        open_face_count = len(comp_face_ids)

        component_info = {
            'component_id': len(components),
            'num_edges': int(len(comp_global)),
            'num_vertices': int(len(unique_verts)),
            'vertices': unique_verts.tolist(),
            'edge_vertex_pairs': comp_edges.tolist(),
            'endpoints': endpoints.tolist(),
            'branch_vertices': branch_vertices.tolist(),
            'is_cycle': bool(is_cycle),
            'face_ids': comp_face_ids.tolist(),
            'open_face_count': open_face_count,
            'nonmanifold_face_count': nonmanifold_face_count,
            'candidate_breaks': candidate_breaks,
        }
        components.append(component_info)

    return components


def find_minimal_enclosing_manifold_boundary_greedy(
    mesh, component, max_depth=12
):
    seed_faces = set(map(int, component.get('face_ids', [])))
    if not seed_faces:
        return {
            'success': False,
            'depth': 0,
            'enclosed_faces': [],
            'boundary_vertices': [],
            'boundary_edges': [],
        }

    adj = build_manifold_face_adjacency(mesh)
    current = set(seed_faces)

    for depth in range(1, max_depth + 1):
        if is_manifold_closed_boundary(mesh, current):
            face_edge_keys = compute_face_edge_keys(mesh)
            edge_keys, edge_faces = compute_edge_to_faces(mesh)
            edge_to_faces = {}
            for key, faces_list in zip(edge_keys, edge_faces):
                edge_to_faces[int(key)] = faces_list

            boundary_edge_keys = []
            for fid in current:
                for j in range(3):
                    key = int(face_edge_keys[fid, j])
                    shared = edge_to_faces.get(key, [])
                    inner_count = sum(
                        1 for f in shared if int(f) in current
                    )
                    if inner_count == 1:
                        boundary_edge_keys.append(key)

            boundary_vertices = _extract_boundary_loops_from_edge_keys(
                mesh, boundary_edge_keys
            )

            return {
                'success': True,
                'depth': depth,
                'enclosed_faces': sorted(current),
                'boundary_vertices': boundary_vertices,
                'boundary_edges': boundary_edge_keys,
            }

        next_faces = set(current)
        for f in current:
            for nb in adj[f]:
                if nb not in next_faces:
                    next_faces.add(nb)

        if len(next_faces) == len(current):
            break
        current = next_faces

    return {
        'success': False,
        'depth': max_depth,
        'enclosed_faces': [],
        'boundary_vertices': [],
        'boundary_edges': [],
    }


def generate_initial_seifert_disk(mesh, loop_vertices):
    """
    以健康孔洞边界环为边界生成初始拓扑圆盘。
    """
    loop_vertices = [int(v) for v in loop_vertices]
    if len(loop_vertices) < 3:
        return None, []

    pts = np.asarray(mesh.vertices[loop_vertices], dtype=np.float64)

    try:
        from shapely.geometry import Polygon

        centroid = pts.mean(axis=0)
        _, _, vh = np.linalg.svd(pts - centroid)
        u = vh[0]
        v = vh[1]

        poly2d = np.column_stack([
            (pts - centroid) @ u,
            (pts - centroid) @ v,
        ])

        polygon = Polygon(poly2d)
        if not polygon.is_valid:
            polygon = polygon.buffer(0)

        triangulated = trimesh.creation.triangulate_polygon(polygon)
        if triangulated is None:
            raise ValueError("triangulate_polygon returned None")

        tri_vertices_2d, tri_faces = triangulated
        tri_vertices_2d = np.asarray(tri_vertices_2d, dtype=np.float64)
        tri_faces = np.asarray(tri_faces, dtype=np.int64)

        if tri_vertices_2d.ndim != 2 or tri_vertices_2d.shape[1] != 2:
            raise ValueError("triangulate_polygon returned invalid 2D vertices")
        if tri_faces.ndim != 2 or tri_faces.shape[1] != 3 or len(tri_faces) == 0:
            raise ValueError("triangulate_polygon returned empty or invalid faces")

        v3d = centroid + tri_vertices_2d[:, 0:1] * u + tri_vertices_2d[:, 1:2] * v

        boundary_indices = []
        for p2d in poly2d:
            dists = np.linalg.norm(tri_vertices_2d - p2d, axis=1)
            idx = int(np.argmin(dists))
            if dists[idx] > 1e-8:
                raise ValueError("boundary point not found in triangulation")
            boundary_indices.append(idx)

        disk = trimesh.Trimesh(
            vertices=v3d,
            faces=tri_faces,
            process=False,
        )

        if len(boundary_indices) != len(poly2d):
            raise ValueError("boundary indices mismatch")

        return disk, boundary_indices

    except Exception as e:
        print(f"  [WARN] 初始 Seifert 圆盘生成失败: {e}")
        return None, []


def extract_intersection_faces_by_vertex_state(W, N, eps=None):
    """
    基于顶点内外状态，提取邻域网格 N 中与水密包络 W 相交的面片。
    """
    n_verts = len(N.vertices)
    n_faces = len(N.faces)

    if n_verts == 0 or n_faces == 0:
        return np.zeros(n_faces, dtype=bool), np.array([], dtype=np.int64), np.zeros(n_verts, dtype=np.int8)

    closest, dist, _ = W.nearest.on_surface(N.vertices)
    inside = W.contains(N.vertices)

    if eps is None:
        median_w = np.median(W.edges_unique_length) if len(W.edges_unique) else 0.0
        median_n = np.median(N.edges_unique_length) if len(N.edges_unique) else 0.0
        eps = max(median_w, median_n) * 1e-6

    on_surface = dist <= eps
    outside = (~inside) & (~on_surface)
    inside_only = inside & (~on_surface)

    vertex_state = np.zeros(n_verts, dtype=np.int8)
    vertex_state[inside_only] = 0
    vertex_state[outside] = 1
    vertex_state[on_surface] = 2

    face_mask = np.zeros(n_faces, dtype=bool)

    faces = np.asarray(N.faces, dtype=np.int64)
    for fid, tri in enumerate(faces):
        vs = vertex_state[tri]
        has_inside = np.any(vs == 0)
        has_outside = np.any(vs == 1)
        has_surface = np.any(vs == 2)

        if has_surface or (has_inside and has_outside):
            face_mask[fid] = True

    return face_mask, np.where(face_mask)[0], vertex_state


def segment_tubular_regions(normals, areas=None, threshold=0.1, min_faces=100,
                            max_regions=5, max_iterations=1000, rng=None):
    """
    使用 RANSAC 根据面片法线将三角网格分割为不同的管状区域。

    管状区域的特征是所有面片法线大致垂直于该区域的轴线。随机抽取两个面片，
    其法线叉乘作为候选轴线，将法线与轴线点积绝对值小于阈值的面片作为内点。
    重复此过程以取出多个区域。

    Parameters
    ----------
    normals : (N, 3) np.ndarray
        面片单位法向量（必须已归一化）。
    areas : (N,) np.ndarray or None
        面片面积，用于加权评分。若为 None，则使用等权（每个面片权重为 1）。
    threshold : float
        内点判定阈值：若 |dot(normal, axis)| <= threshold，该面片属于当前区域。
        数值越小越严格。
    min_faces : int
        一个区域必须包含的最少面片数。
    max_regions : int
        最多提取的区域数量。
    max_iterations : int
        每个区域 RANSAC 的最大迭代次数。
    rng : numpy.random.Generator or None
        随机数生成器，用于可重复性。若为 None 则使用默认随机状态。

    Returns
    -------
    labels : (N,) np.ndarray (int)
        面片区域标签。0 表示未归类，1..k 表示第 k 个区域。
    axes : list of ndarray (3,)
        每个区域对应的单位轴线方向（顺序与标签编号一致）。
    """
    N = normals.shape[0]
    if N == 0:
        return np.zeros(0, dtype=int), []

    # 面积权重
    if areas is None:
        weights = np.ones(N, dtype=np.float64)
    else:
        weights = np.asarray(areas, dtype=np.float64)

    # 检查并过滤零/无效法向量，避免除零
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    valid_mask = norms[:, 0] > 1e-12
    n = np.zeros_like(normals, dtype=np.float64)
    n[valid_mask] = normals[valid_mask] / norms[valid_mask]

    # 随机数生成器
    if rng is None:
        rng = np.random.default_rng()

    labels = np.zeros(N, dtype=int)
    axes = []

    # 当前可用的面片索引（未归类的且法向有效）
    active = np.where((labels == 0) & valid_mask)[0]

    for region_id in range(1, max_regions + 1):
        if len(active) < 2 or len(active) < min_faces:
            break

        # 当前活动子集
        active_n = n[active]
        active_w = weights[active]

        best_score = -1.0
        best_axis = None
        best_inliers = None  # 相对于 active 的布尔掩码

        # 自适应迭代次数：考虑期望内点比例至少 min_faces/len(active)
        desired_inlier_ratio = max(min_faces / len(active), 0.01)
        # 概率采样模型所需最少迭代次数（保证至少一次全内点抽样概率 0.99）
        iters = min(
            max_iterations,
            int(np.log(0.01) / np.log(1 - desired_inlier_ratio**2))
        )
        iters = max(1, iters)

        for _ in range(iters):
            # 随机抽取两个不同的活动面片
            i, j = rng.choice(len(active), size=2, replace=False)
            n1, n2 = active_n[i], active_n[j]

            cross = np.cross(n1, n2)
            norm_cross = np.linalg.norm(cross)
            if norm_cross < 1e-9:   # 法线几乎平行，无法定义轴线
                continue
            axis = cross / norm_cross

            # 计算所有活动面片与该轴线的点积绝对值
            dots = np.abs(active_n @ axis)
            inliers = dots <= threshold

            # 评分 = 内点权重之和
            score = np.sum(active_w[inliers])

            # 内点数必须满足最小要求
            if np.sum(inliers) < min_faces:
                continue

            if score > best_score:
                best_score = score
                best_axis = axis.copy()
                best_inliers = inliers

        # 如果没有找到合格模型，停止
        if best_axis is None:
            break

        # 将内点标记到全局 labels
        active_indices = active[best_inliers]
        labels[active_indices] = region_id
        axes.append(best_axis)

        # 更新 active 列表
        active = np.where(labels == 0)[0]

    return labels, axes


def repair_nonmanifold_edges(mesh, max_iterations=10, verbose=True):
    """
    策略2：对每个非流形边，保留法向最一致的两个面，删除其余面片。
    迭代直到没有非流形边（或达到迭代上限）。
    """
    for it in range(max_iterations):
        faces = np.asarray(mesh.faces, dtype=np.int64).reshape(-1, 3)

        edge_face_map = {}
        for fi, face in enumerate(faces):
            v1, v2, v3 = int(face[0]), int(face[1]), int(face[2])
            for a, b in [(v1, v2), (v2, v3), (v3, v1)]:
                key = (a, b) if a < b else (b, a)
                edge_face_map.setdefault(key, []).append(fi)

        nonmanifold = {e: fl for e, fl in edge_face_map.items()
                       if len(fl) > 2}
        if not nonmanifold:
            if verbose:
                print(f"  [Iter {it}] No nonmanifold edges remain.")
            break

        if verbose:
            print(f"  [Iter {it}] {len(nonmanifold)} nonmanifold edges, "
                  f"removing extra faces...")

        normals = mesh.face_normals
        areas = mesh.area_faces
        faces_to_remove = set()

        for edge, fl in nonmanifold.items():
            if verbose:
                va, vb = mesh.vertices[edge[0]], mesh.vertices[edge[1]]
                print(f"    edge {edge} at {va} <-> {vb}, "
                      f"shared by {len(fl)} faces")
                for fi in fl:
                    print(f"      face {fi}: area={areas[fi]:.4f}, "
                          f"normal={normals[fi].round(3)}")

            best_pair, best_key = None, -np.inf
            for i in range(len(fl)):
                for j in range(i + 1, len(fl)):
                    dot = np.dot(normals[fl[i]], normals[fl[j]])
                    score = dot + 1e-6 * min(areas[fl[i]], areas[fl[j]])
                    if score > best_key:
                        best_key = score
                        best_pair = (fl[i], fl[j])

            for fi in fl:
                if fi not in best_pair:
                    faces_to_remove.add(fi)

        keep = np.ones(len(faces), dtype=bool)
        keep[list(faces_to_remove)] = False
        mesh = trimesh.Trimesh(vertices=mesh.vertices,
                               faces=faces[keep], process=False)

    mesh = mesh.copy()
    mesh.remove_unreferenced_vertices()
    return mesh


def fill_small_holes(mesh, max_loop_edges=50, verbose=True):
    """
    用质心扇形三角化封闭小边界环。
    使用基于边界边集合的 DFS，按方向连续性在分叉处选择下一条边。
    """
    faces = np.asarray(mesh.faces, dtype=np.int64).reshape(-1, 3)

    # 构建 edge -> faces 映射，识别边界边
    edge_face_map = {}
    for fi, face in enumerate(faces):
        v1, v2, v3 = int(face[0]), int(face[1]), int(face[2])
        for a, b in [(v1, v2), (v2, v3), (v3, v1)]:
            key = (a, b) if a < b else (b, a)
            edge_face_map.setdefault(key, []).append(fi)

    boundary_edges = set(e for e, fl in edge_face_map.items() if len(fl) == 1)

    if not boundary_edges:
        if verbose:
            print("  No boundary edges, nothing to fill.")
        return mesh

    # 稳健地提取所有边界环
    loops = []
    edge_set = set(boundary_edges)

    while edge_set:
        e0 = edge_set.pop()
        v_start, v_curr = e0
        loop = [v_start, v_curr]
        v_prev = v_start

        while True:
            # 找与 v_curr 相连且未访问的边界边
            candidates = [e for e in edge_set if v_curr in e]

            if not candidates:
                # 无法闭合，放弃这条路径
                if verbose and len(loop) > 2:
                    print(f"  Dropped unclosed boundary path ({len(loop)} edges)")
                break

            # 如果有多个候选，按方向连续性选择最自然的延续
            if len(candidates) > 1:
                dir_curr = mesh.vertices[v_curr] - mesh.vertices[v_prev]
                dir_curr = dir_curr / (np.linalg.norm(dir_curr) + 1e-12)

                best_edge = None
                best_score = -np.inf
                for e in candidates:
                    v_next = e[0] if e[1] == v_curr else e[1]
                    dir_next = mesh.vertices[v_next] - mesh.vertices[v_curr]
                    dn = np.linalg.norm(dir_next)
                    if dn < 1e-12:
                        continue
                    dir_next = dir_next / dn

                    # 偏好与当前方向夹角最小的延续
                    dot = np.dot(dir_curr, dir_next)
                    # 惩罚反向转弯
                    score = dot if dot >= 0 else -0.5 * dot
                    if score > best_score:
                        best_score = score
                        best_edge = e
                next_edge = best_edge
            else:
                next_edge = candidates[0]

            edge_set.remove(next_edge)
            v_next = next_edge[0] if next_edge[1] == v_curr else next_edge[1]
            loop.append(v_next)

            if v_next == v_start:
                # 成功闭合
                loops.append(loop[:-1])  # 去掉重复的起点
                break

            v_prev, v_curr = v_curr, v_next

            # 安全上限，防止异常拓扑导致无限循环
            if len(loop) > max(max_loop_edges * 3, 500):
                if verbose:
                    print(f"  Dropped overly long boundary path ({len(loop)} edges)")
                break

    # 扇形封闭找到的边界环
    new_vertices = [mesh.vertices]
    new_faces = [faces]
    for loop in loops:
        if len(loop) > max_loop_edges:
            if verbose:
                print(f"  Skipping large boundary loop ({len(loop)} edges).")
            continue
        loop_pts = mesh.vertices[np.array(loop)]
        centroid = loop_pts.mean(axis=0)
        c_idx = sum(len(v) for v in new_vertices)
        new_vertices.append(centroid[None, :])
        tris = []
        for i in range(len(loop)):
            tris.append([c_idx, loop[i], loop[(i + 1) % len(loop)]])
        new_faces.append(np.array(tris))
        if verbose:
            print(f"  Filled boundary loop with {len(loop)} edges.")

    vertices = np.vstack(new_vertices)
    faces = np.vstack(new_faces)
    out = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    out.fix_normals()
    return out
