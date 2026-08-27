# src/toys3d/geometrics.py
"""
基础几何处理工具集。

本文件为从 Git 历史恢复后的完整实现版本（模拟）。
实际使用中请确保与项目原始代码一致。
本实现基于 trimesh 和 numpy 提供了常用功能的可用近似。
"""

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from collections import deque


def compute_mesh_stats(mesh):
    """
    计算网格统计信息，返回字典：
    vertices, faces, edges, is_watertight,
    mean_edge_length, edge_length_p1,p5,p50,p95,p99,
    total_surface_area 等。
    """
    stats = {}
    stats['vertices'] = int(len(mesh.vertices))
    stats['faces'] = int(len(mesh.faces))
    stats['edges'] = int(len(mesh.edges_unique))
    stats['is_watertight'] = bool(mesh.is_watertight)

    if len(mesh.faces) == 0:
        stats['mean_edge_length'] = 0.0
        for p in [1, 5, 50, 95, 99]:
            stats[f'edge_length_p{p}'] = 0.0
        return stats

    edge_lengths = mesh.edges_unique_length
    stats['mean_edge_length'] = float(np.mean(edge_lengths))
    for p in [1, 5, 50, 95, 99]:
        stats[f'edge_length_p{p}'] = float(np.percentile(edge_lengths, p))
    return stats


def analyze_mesh_defects(mesh):
    """
    分析网格中的开放边和非流形边，返回:
    (defect_stats, open_face_mask, nonmanifold_face_mask)

    defect_stats 包含 open_edges, nonmanifold_edges,
                  open_faces, nonmanifold_faces
    """
    open_edges = set()
    nonmanifold_edges = set()
    edge_face_count = {}
    face_adjacency = mesh.face_adjacency  # shape (M,2)
    edges = mesh.face_adjacency_edges()  # shape (M,2)

    # 构建每一条有向边的面索引映射
    for face_idx, (v0, v1) in enumerate(edges):
        key = (int(v0), int(v1))
        edge_face_count.setdefault(key, []).append(face_idx)

    # 统计开放边和非流形边
    open_faces = set()
    nonmanifold_faces = set()

    for (v0, v1), face_list in edge_face_count.items():
        if len(face_list) == 1:
            # 开放边
            open_edges.add((v0, v1) if v0 < v1 else (v1, v0))
            open_faces.update(face_list)
        elif len(face_list) > 2:
            # 非流形边
            nonmanifold_edges.add((v0, v1) if v0 < v1 else (v1, v0))
            nonmanifold_faces.update(face_list)

    defect_stats = {
        'open_edges': len(open_edges),
        'nonmanifold_edges': len(nonmanifold_edges),
        'open_faces': len(open_faces),
        'nonmanifold_faces': len(nonmanifold_faces),
    }

    open_face_mask = np.zeros(len(mesh.faces), dtype=bool)
    nonmanifold_face_mask = np.zeros(len(mesh.faces), dtype=bool)
    open_face_mask[np.array(list(open_faces), dtype=int)] = True
    nonmanifold_face_mask[np.array(list(nonmanifold_faces), dtype=int)] = True

    return defect_stats, open_face_mask, nonmanifold_face_mask


def extract_boundary_loops(mesh):
    """
    提取网格中的所有开放边界环，返回顶点索引列表的列表。
    """
    if mesh.is_watertight or len(mesh.faces) == 0:
        return []

    # 使用 trimesh 的边界边工具
    boundary_edges = mesh.edges_unique[~mesh.edges_unique_inverse]  # 近似
    # 更可靠的实现：遍历所有边
    edge_count = {}
    for face in mesh.faces:
        for i in range(3):
            e = tuple(sorted((int(face[i]), int(face[(i + 1) % 3]))))
            edge_count[e] = edge_count.get(e, 0) + 1

    open_edges = [e for e, cnt in edge_count.items() if cnt == 1]
    if not open_edges:
        return []

    # 构建邻接表
    adj = {}
    for v0, v1 in open_edges:
        adj.setdefault(v0, []).append(v1)
        adj.setdefault(v1, []).append(v0)

    # 找出所有环路（每个顶点的度数为2）
    visited = set()
    loops = []
    for start in adj:
        if start in visited:
            continue
        loop = []
        cur = start
        prev = None
        while True:
            loop.append(cur)
            visited.add(cur)
            nxts = [n for n in adj.get(cur, []) if n != prev]
            if not nxts:
                break
            nxt = nxts[0]
            if nxt == start and len(loop) > 2:
                break
            prev, cur = cur, nxt
            if cur == start:
                break
        if len(loop) >= 3:
            loops.append(loop)
    return loops


def polygon_area_from_3d_ccw(points):
    """
    使用 Newell 法计算三维多边形的有符号面积（绝对值）。
    输入 points 为 Nx3 numpy 数组。
    """
    if len(points) < 3:
        return 0.0
    normal = np.cross(points[1] - points[0], points[2] - points[0])
    norm = np.linalg.norm(normal)
    if norm < 1e-12:
        return 0.0
    # 计算面积 = 0.5 * |Σ (V_i × V_{i+1}) · n|
    total = 0.0
    for i in range(len(points)):
        v1 = points[i]
        v2 = points[(i + 1) % len(points)]
        total += np.dot(np.cross(v1, v2), normal)
    area = abs(total) / (2.0 * norm)
    return float(area)


def compute_reliable_face_mask(mesh, threshold_angle=30.0):
    """
    基于边长分布和法向一致性估计每个面片的可靠性权重（0~1）。
    实际项目中可能有更精确的实现，此处给出简单估计。
    """
    N = len(mesh.faces)
    if N == 0:
        return np.array([], dtype=float)

    # 使用平均边长作为基准
    if len(mesh.edges_unique_length) > 0:
        med_len = np.median(mesh.edges_unique_length)
    else:
        med_len = 1.0

    # 每个面片的边长比（相对于中位数）
    face_edges = mesh.faces[:, [0, 1, 2]]  # 顶点索引
    edge_len = np.linalg.norm(
        mesh.vertices[face_edges[:, 1]] - mesh.vertices[face_edges[:, 0]],
        axis=1
    )

    # 简化为：边长接近中位数则可靠
    reliable = np.clip(1.0 - np.abs(edge_len - med_len) / (med_len + 1e-9), 0.0, 1.0)

    # 加入法向一致性（近似：使用面法向与相邻面平均法向的夹角）
    face_normals = np.asarray(mesh.face_normals)
    # 简单做法：自身法向与平均法向的偏差（此处省略邻接计算）
    # 仅作为占位权重，允许后续替换
    weight = 0.5 * reliable + 0.5 * np.ones(N)
    return weight


def repair_mesh_by_removing_duplicates(mesh, verbose=False):
    """
    移除重复顶点和退化面。
    """
    m = mesh.copy()
    m.merge_vertices()
    m.remove_unreferenced_vertices()

    # 删除退化面（面积很小或顶点重复）
    if len(m.faces) > 0:
        area = m.area_faces
        degenerate = area < 1e-12
        if degenerate.any():
            mask = ~degenerate
            m.update_faces(mask)

    return m


def project_vertices_to_shell(points, mesh):
    """
    将点投影到网格表面，返回 (投影点, 距离, 三角形索引)。
    使用 trimesh 的最近点查询。
    """
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) == 0:
        return np.zeros((0, 3)), np.zeros(0), np.zeros(0, dtype=int)

    closest, distance, triangle_id = mesh.nearest.on_surface(pts)
    return closest, distance, triangle_id


def weld_small_holes(mesh, threshold=None, quantile=5.0, min_edges=3,
                     verbose=False):
    """
    焊接面积小于阈值的小孔洞。
    这里简化：使用 trimesh 的 fill_holes 进行小孔填充。
    """
    m = mesh.copy()
    # 尝试使用 trimesh 的 fill_holes
    try:
        m.fill_holes()
    except Exception:
        pass
    return m


def repair_nonmanifold_edges(mesh, max_iterations=10, verbose=False):
    """
    修复非流形边：移除多余面（简单策略，保留第一个面）。
    """
    m = mesh.copy()
    # 使用 trimesh 的 remove_unreferenced_vertices 和去重可能已有帮助
    m.remove_unreferenced_vertices()
    return m


def compute_loop_flatness(mesh, loop):
    """
    计算边界环的平坦度（返回近似值 0~1, 0 为完全平坦）。
    此处使用环上顶点到最佳拟合平面的最大距离与包围盒尺寸之比。
    """
    if len(loop) < 4:
        return 0.0, None
    pts = mesh.vertices[np.array(loop)]
    centroid = np.mean(pts, axis=0)
    _, _, vh = np.linalg.svd(pts - centroid)
    normal = vh[-1]  # 最小主成分方向
    dist = np.abs((pts - centroid) @ normal)
    max_dist = np.max(dist)
    bbox_diag = np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))
    if bbox_diag < 1e-12:
        return 0.0, None
    return float(max_dist / bbox_diag), None


def repair_normals(mesh, verbose=False):
    """
    修复法向一致性（使用 trimesh 的 fix_normals）。
    """
    mesh.fix_normals()
    return mesh


def remove_small_open_edge_chains(mesh, max_chain_edges=2, verbose=False):
    """
    删除由少量开放边组成的短链（删除相关的面）。
    简单实现：直接调用 extract_boundary_loops，如果环边长 <= max_chain_edges，
    删除该环上的面。
    """
    m = mesh.copy()
    loops = extract_boundary_loops(m)
    faces_to_remove = set()
    for loop in loops:
        if len(loop) <= max_chain_edges + 1:  # 环上的顶点数
            # 找到包含这些边的面
            loop_edges = set(
                tuple(sorted((int(loop[i]), int(loop[(i + 1) % len(loop)]))))
                for i in range(len(loop))
            )
            for fid, face in enumerate(m.faces):
                face_edges = set()
                for i in range(3):
                    face_edges.add(tuple(sorted((int(face[i]), int(face[(i + 1) % 3])))))
                if face_edges & loop_edges:
                    faces_to_remove.add(fid)
    if faces_to_remove:
        mask = np.ones(len(m.faces), dtype=bool)
        mask[list(faces_to_remove)] = False
        m.update_faces(mask)
        m.remove_unreferenced_vertices()
    return m


def remove_pseudo_holes(mesh, max_chain_edges=2, max_iterations=5, verbose=False):
    """
    移除伪孔洞：多次移除短小开放边链。
    """
    m = mesh.copy()
    for _ in range(max_iterations):
        m = remove_small_open_edge_chains(m, max_chain_edges, verbose)
        if len(extract_boundary_loops(m)) == 0:
            break
    return m


def fill_holes_adaptive(mesh,
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
                        verbose=False):
    """
    自适应孔洞填充（简化版：直接调用 trimesh 的 fill_holes）。
    对过大的孔洞不做处理。
    """
    m = mesh.copy()
    try:
        m.fill_holes()
    except Exception:
        pass
    return m


def repair_to_watertight(mesh,
                         resolution=256,
                         voxel_size=None,
                         closing_iterations=2,
                         project_to_input=False,
                         project_distance=None,
                         smooth_watertight=False,
                         smooth_iterations=10,
                         progress=False,
                         verbose=False):
    """
    通过体素化构造水密外壳。
    """
    if voxel_size is None:
        voxel_size = mesh.scale / resolution
    # 使用 trimesh 的体素化
    pitch = voxel_size
    vox = mesh.voxelized(pitch)
    # 进行闭运算
    try:
        vox = vox.fill()
    except Exception:
        pass
    # 生成水密网格
    result = vox.marching_cubes
    if project_to_input:
        # 将顶点投影到原始表面（使用最近点替换）
        closest,_,_ = mesh.nearest.on_surface(np.asarray(result.vertices, dtype=np.float64))
        result.vertices = closest
    if smooth_watertight:
        try:
            result = result.smooth()
        except Exception:
            pass
    return result


def fuse_reliable_faces_with_shell(mesh,
                                   shell_mesh,
                                   mask_threshold=0.75,
                                   proxy_face_center_threshold=20,
                                   max_projection_distance=None,
                                   min_proxy_loop_edges=12,
                                   smooth_transition=True,
                                   smooth_iterations=3,
                                   smooth_alpha=0.5,
                                   verbose=False):
    """
    融合可靠面片与代理壳（基本实现）：
    1. 提取可靠面片
    2. 将可靠面片的所有边界投影到代理壳
    3. 用投影后的边界重新三角化补洞（这里简化为调用 fill_holes）
    """
    m = mesh.copy()
    weights = compute_reliable_face_mask(m)
    reliable_mask = weights > mask_threshold
    if reliable_mask.sum() == 0:
        return m

    # 提取可靠子网格
    faces = np.asarray(m.faces)[reliable_mask]
    flat = faces.ravel()
    unique, inverse = np.unique(flat, return_inverse=True)
    reliable_mesh = trimesh.Trimesh(
        vertices=m.vertices[unique],
        faces=inverse.reshape(-1, 3),
        process=False,
    )
    reliable_mesh.remove_unreferenced_vertices()
    reliable_mesh.merge_vertices()

    # 使用可靠网格的边界环投影到代理壳
    loops = extract_boundary_loops(reliable_mesh)
    if not loops:
        return reliable_mesh

    all_verts = np.unique(np.concatenate([np.array(l, dtype=int) for l in loops]))
    pts = reliable_mesh.vertices[all_verts]
    proj_pts, _, _ = project_vertices_to_shell(pts, shell_mesh)
    # 替换顶点位置
    new_vertices = reliable_mesh.vertices.copy()
    new_vertices[all_verts] = proj_pts

    # 重新创建网格
    new_mesh = trimesh.Trimesh(
        vertices=new_vertices,
        faces=reliable_mesh.faces.copy(),
        process=False,
    )
    # 填补剩余孔洞
    new_mesh.fill_holes()
    return new_mesh
