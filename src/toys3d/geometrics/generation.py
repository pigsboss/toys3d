# src/toys3d/geometrics/generation.py
"""生成/修复类工具。"""
import numpy as np
import trimesh
from .discrete import (
    build_cotangent_laplacian,
    laplacian_smooth_fixed_boundary,
    compute_curvature_statistics,
)
from .euclidean import polygon_area_from_3d_ccw


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


# ---------------------------------------------------------------------------
# Seifert 曲面生成与修补
# ---------------------------------------------------------------------------

def _project_points_to_plane(points, plane_normal, centroid=None):
    n = np.asarray(plane_normal, dtype=np.float64)
    n = n / (np.linalg.norm(n) + 1e-12)

    if centroid is None:
        centroid = points.mean(axis=0)

    helper = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(n, helper)) > 0.9:
        helper = np.array([1.0, 0.0, 0.0])

    u = np.cross(n, helper)
    u = u / (np.linalg.norm(u) + 1e-12)
    v = np.cross(n, u)
    v = v / (np.linalg.norm(v) + 1e-12)

    flat = np.column_stack([
        (points - centroid) @ u,
        (points - centroid) @ v,
    ])
    return flat, u, v, centroid


def _is_valid_simple_projection(flat, original_area, min_area_ratio=0.2):
    from shapely.geometry import Polygon

    try:
        polygon = Polygon(flat)
    except Exception:
        return False, 0.0, None

    if not polygon.is_valid or polygon.is_empty:
        return False, 0.0, None

    area2d = float(polygon.area)
    if area2d < 1e-12:
        return False, 0.0, None

    if original_area < 1e-12:
        return False, 0.0, None

    if area2d / original_area < min_area_ratio:
        return False, area2d, None

    return True, area2d, polygon


def _candidate_projection_normals(points, mesh=None, loop_vertices=None):
    centroid = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - centroid)

    normals = []
    # 所有主成分方向
    for i in range(3):
        normals.append(vh[i].copy())
    # 三个坐标轴
    normals.extend([
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    ])
    # 边界顶点所在面片的加权平均法向
    if mesh is not None and loop_vertices is not None:
        face_normals = []
        if hasattr(mesh, 'vertex_faces'):
            for v in loop_vertices:
                faces = mesh.vertex_faces[v]
                for f in faces:
                    if f >= 0:
                        face_normals.append(mesh.face_normals[f])
        if face_normals:
            avg = np.mean(face_normals, axis=0)
            if np.linalg.norm(avg) > 1e-12:
                normals.append(avg / np.linalg.norm(avg))
    return normals


def _find_valid_boundary_projection(pts, mesh=None, loop_vertices=None):
    original_area = polygon_area_from_3d_ccw(pts)
    if original_area < 1e-12:
        return None

    for n in _candidate_projection_normals(pts, mesh, loop_vertices):
        flat, u, v, centroid = _project_points_to_plane(pts, n)
        valid, _area2d, polygon = _is_valid_simple_projection(
            flat, original_area
        )
        if valid and polygon is not None:
            return flat, u, v, centroid, polygon

    return None


def _is_simple_planar_loop(mesh, loop_vertices, max_vertices=50, planar_ratio=0.1):
    if len(loop_vertices) > max_vertices:
        return False

    pts = mesh.vertices[np.asarray(loop_vertices, dtype=np.int64)]
    centroid = pts.mean(axis=0)
    _, _, vh = np.linalg.svd(pts - centroid)
    normal = vh[2]
    dists = np.abs((pts - centroid) @ normal)
    extent = np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))
    if extent < 1e-12:
        return False
    max_dist = np.max(dists)
    return max_dist / extent < planar_ratio


def _generate_fallback_fan_disk(mesh, loop_vertices):
    pts = mesh.vertices[np.asarray(loop_vertices, dtype=np.int64)]
    centroid = pts.mean(axis=0)
    n = len(loop_vertices)

    vertices = np.vstack([pts, centroid])
    faces = []
    c_idx = n
    for i in range(n):
        faces.append([c_idx, i, (i + 1) % n])
    faces = np.array(faces, dtype=np.int64)

    disk = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    # 边界索引就是前 n 个顶点，顺序与输入 loop_vertices 一致
    boundary_indices = list(range(n))
    return disk, boundary_indices


def generate_initial_seifert_disk(mesh, loop_vertices, verbose=False):
    loop_vertices = [int(v) for v in loop_vertices]

    # 快速去除连续重复顶点
    loop = []
    for v in loop_vertices:
        if not loop or loop[-1] != v:
            loop.append(v)
    if len(loop) >= 2 and loop[0] == loop[-1]:
        loop = loop[:-1]

    if verbose:
        print(f"  [Seifert 初始圆盘] 输入边界顶点数: {len(loop_vertices)}")
        print(f"  [Seifert 初始圆盘] 去重后边界顶点数: {len(loop)}")

    if len(loop) < 3:
        if verbose:
            print("  [Seifert 初始圆盘] 失败: 去重后有效顶点数 < 3")
        return None, []

    loop_vertices = loop
    pts = np.asarray(mesh.vertices[loop_vertices], dtype=np.float64)

    if verbose:
        print(f"  [Seifert 初始圆盘] 原始边界面积: "
              f"{polygon_area_from_3d_ccw(pts):.6f}")

    projection = _find_valid_boundary_projection(pts, mesh, loop_vertices)
    if projection is None:
        if _is_simple_planar_loop(mesh, loop_vertices):
            if verbose:
                print("  [Seifert 初始圆盘] 投影失败，尝试简单平面 fallback")
                print("  [Seifert 初始圆盘] 使用质心扇形三角化作为初始圆盘")
            return _generate_fallback_fan_disk(mesh, loop_vertices)
        else:
            if verbose:
                print("  [Seifert 初始圆盘] 失败: 未找到有效投影且不满足简单平面条件")
                print(f"  [DEBUG] 原始面积={polygon_area_from_3d_ccw(pts):.6f}")
                for n in _candidate_projection_normals(pts, mesh, loop_vertices):
                    flat, _, _, _ = _project_points_to_plane(pts, n)
                    valid, area2d, _ = _is_valid_simple_projection(
                        flat, polygon_area_from_3d_ccw(pts)
                    )
                    print(f"  [DEBUG] 法向 {n}, valid={valid}, area2d={area2d:.6f}")
            return None, []
    else:
        if verbose:
            flat, u, v, centroid, polygon = projection
            print("  [Seifert 初始圆盘] 找到有效投影：")
            print(f"    centroid = {centroid}")
            print(f"    u = {u}")
            print(f"    v = {v}")
            print(f"    polygon.area = {polygon.area:.6f}")

    flat, u, v, centroid, polygon = projection

    try:
        triangulated = trimesh.creation.triangulate_polygon(polygon)
        if triangulated is None:
            raise ValueError("triangulate_polygon returned None")

        tri_vertices_2d, tri_faces = triangulated
        tri_vertices_2d = np.asarray(tri_vertices_2d, dtype=np.float64)
        tri_faces = np.asarray(tri_faces, dtype=np.int64)

        if tri_vertices_2d.ndim != 2 or tri_vertices_2d.shape[1] != 2:
            raise ValueError("invalid 2D vertex array")
        if tri_faces.ndim != 2 or tri_faces.shape[1] != 3 or len(tri_faces) == 0:
            raise ValueError("invalid face array")

        if verbose:
            print(f"  [Seifert 初始圆盘] 三角化成功: "
                  f"顶点数={len(tri_vertices_2d)}, 面片数={len(tri_faces)}")
    except Exception as e:
        if verbose:
            print(f"  [Seifert 初始圆盘] 失败: 平面三角化异常: {e}")
        if _is_simple_planar_loop(mesh, loop_vertices):
            if verbose:
                print("  [Seifert 初始圆盘] 尝试简单平面 fallback")
            return _generate_fallback_fan_disk(mesh, loop_vertices)
        return None, []

    # 严格对照原始投影点，不允许近似匹配失败或重复
    boundary_indices = []
    for i, p2d in enumerate(flat):
        dists = np.linalg.norm(tri_vertices_2d - p2d, axis=1)
        idx = int(np.argmin(dists))
        if dists[idx] > 1e-8:
            if verbose:
                print(f"  [Seifert 初始圆盘] 失败: 投影点 {i} 无法在三角化顶点中匹配")
                print(f"    原投影点: {p2d}")
                print(f"    最近距离: {dists[idx]:.6e}")
                print(f"    最近顶点: {tri_vertices_2d[idx]}")
            return None, []
        boundary_indices.append(idx)

    if len(set(boundary_indices)) != len(flat):
        if verbose:
            print("  [Seifert 初始圆盘] 失败: 边界顶点映射存在重复")
            print(f"    boundary_indices = {boundary_indices}")
        return None, []

    # 确保边界边按原始顺序存在，防止三角化打乱边界
    edge_set = set()
    for face in tri_faces:
        for j in range(3):
            a = int(face[j])
            b = int(face[(j + 1) % 3])
            edge_set.add((a, b))
            edge_set.add((b, a))

    for i in range(len(boundary_indices)):
        a = boundary_indices[i]
        b = boundary_indices[(i + 1) % len(boundary_indices)]
        if (a, b) not in edge_set:
            if verbose:
                print("  [Seifert 初始圆盘] 失败: 边界边顺序检查未通过")
                print(f"    segment {i}: ({a}, {b}) 不在三角化边集中")
            if _is_simple_planar_loop(mesh, loop_vertices):
                if verbose:
                    print("  [Seifert 初始圆盘] 尝试简单平面 fallback")
                return _generate_fallback_fan_disk(mesh, loop_vertices)
            return None, []

    v3d = centroid + tri_vertices_2d[:, 0:1] * u + tri_vertices_2d[:, 1:2] * v

    disk = trimesh.Trimesh(
        vertices=v3d,
        faces=tri_faces,
        process=False,
    )

    # 初始圆盘质量硬检查：不满足快速失败，不进入后续迭代
    if len(disk.faces) == 0:
        if verbose:
            print("  [Seifert 初始圆盘] 失败: 初始圆盘面片数为 0")
        return None, []

    areas = disk.area_faces
    if np.any(areas <= 1e-12):
        if verbose:
            print("  [Seifert 初始圆盘] 失败: 初始圆盘存在零面积面片")
            print(f"    zero_area_count = {int(np.sum(areas <= 1e-12))}")
        return None, []

    area_ratio = np.max(areas) / max(float(np.min(areas)), 1e-12)
    if area_ratio > 1000.0 and verbose:
        print(f"  [Seifert 初始圆盘] 警告: 面积比很大 ({area_ratio:.1f})，继续尝试优化")

    if not np.allclose(disk.vertices[boundary_indices], pts, atol=1e-8):
        if verbose:
            diff = disk.vertices[boundary_indices] - pts
            print("  [Seifert 初始圆盘] 失败: 初始圆盘边界与原始孔洞边界不一致")
            print(f"    max_abs_diff = {np.max(np.abs(diff)):.6e}")
        return None, []

    if verbose:
        print(f"  [Seifert 初始圆盘] 成功: "
              f"面片数={len(disk.faces)}, 顶点数={len(disk.vertices)}")

    return disk, boundary_indices


def _validate_seifert_patch(mesh, hole_vertex_indices, seifert_mesh, boundary_indices):
    if len(seifert_mesh.faces) == 0:
        return False

    areas = seifert_mesh.area_faces
    if np.any(areas <= 1e-12):
        return False

    try:
        boundary_positions = seifert_mesh.vertices[boundary_indices]
        original_positions = mesh.vertices[np.asarray(hole_vertex_indices, dtype=np.int64)]
    except Exception:
        return False

    if not np.allclose(boundary_positions, original_positions, atol=1e-8):
        return False

    return True


def generate_seifert_surface(mesh, hole_vertex_indices,
                             optimize_iterations=200,
                             step_size=1.0,
                             tol=1e-7,
                             verbose=False):
    hole_vertex_indices = [int(v) for v in hole_vertex_indices]
    if len(hole_vertex_indices) < 3:
        return {
            "success": False,
            "mesh": None,
            "boundary_indices": [],
            "message": "孔洞边界顶点数不足 3，无法生成 Seifert 曲面",
        }

    disk_mesh, boundary_indices = generate_initial_seifert_disk(
        mesh, hole_vertex_indices, verbose=verbose
    )
    if disk_mesh is None:
        return {
            "success": False,
            "mesh": None,
            "boundary_indices": [],
            "message": "当前孔洞边界过于复杂，未找到有效投影/初始圆盘，已快速失败",
        }

    seifert_mesh, opt_info = laplacian_smooth_fixed_boundary(
        disk_mesh,
        boundary_indices,
        iterations=optimize_iterations,
        step_size=step_size,
        tol=tol,
        verbose=verbose,
        return_info=True,
    )

    if seifert_mesh is None or not opt_info.get("success", False):
        msg = opt_info.get("message", "极小曲面优化失败")
        return {
            "success": False,
            "mesh": None,
            "boundary_indices": boundary_indices,
            "message": msg,
        }

    if not _validate_seifert_patch(
        mesh, hole_vertex_indices, seifert_mesh, boundary_indices
    ):
        return {
            "success": False,
            "mesh": None,
            "boundary_indices": boundary_indices,
            "message": "生成后的补丁未通过边界或几何验证",
        }

    return {
        "success": True,
        "mesh": seifert_mesh,
        "boundary_indices": boundary_indices,
        "message": "Seifert 曲面生成成功",
        "diagnostics": opt_info,
    }


def compute_seifert_fill_stats(mesh, comp, seifert_mesh,
                               hole_vertex_indices,
                               seifert_boundary_indices):
    """
    计算 Seifert 曲面填充前后，健康孔洞边界边及新增 Seifert 曲面边/面的属性统计。
    """
    seed_faces = list(map(int, comp.get("face_ids", [])))
    if not seed_faces:
        raise ValueError("无法获取组件种子面片")

    original_faces = np.asarray(mesh.faces, dtype=np.int64)
    face_idx_sub = np.array(sorted(seed_faces), dtype=np.int64)
    sub_faces = original_faces[face_idx_sub]

    unique_verts, inverse = np.unique(sub_faces.ravel(), return_inverse=True)
    local_vertices = mesh.vertices[unique_verts]
    local_faces = inverse.reshape(-1, 3)

    local_mesh = trimesh.Trimesh(
        vertices=local_vertices,
        faces=local_faces,
        process=False,
    )

    old_to_new = {
        int(old_v): int(new_v)
        for new_v, old_v in enumerate(unique_verts)
    }

    loop_local = []
    for v in hole_vertex_indices:
        if int(v) not in old_to_new:
            raise ValueError("孔洞边界顶点不在种子面片中")
        loop_local.append(old_to_new[int(v)])

    if len(loop_local) != len(seifert_boundary_indices):
        raise ValueError("Seifert 边界映射长度不一致")

    filled_mesh = _build_filled_mesh(
        local_mesh,
        seifert_mesh,
        loop_local,
        seifert_boundary_indices,
    )
    if filled_mesh is None:
        raise ValueError("无法构建填充网格")

    loop_edge_tuples = []
    for i in range(len(loop_local)):
        v0 = loop_local[i]
        v1 = loop_local[(i + 1) % len(loop_local)]
        loop_edge_tuples.append(_edge_tuple(v0, v1))

    local_edge_map = _build_edge_tuple_to_faces(local_mesh)
    filled_edge_map = _build_edge_tuple_to_faces(filled_mesh)

    before = {"open": 0, "manifold": 0, "nonmanifold": 0}
    after = {"open": 0, "manifold": 0, "nonmanifold": 0}

    for key in loop_edge_tuples:
        cnt_before = len(local_edge_map.get(key, []))
        cnt_after = len(filled_edge_map.get(key, []))
        before[_classify_edge_count(cnt_before)] += 1
        after[_classify_edge_count(cnt_after)] += 1

    new_face_start = len(local_mesh.faces)
    new_face_indices = list(range(new_face_start, len(filled_mesh.faces)))

    new_face_stats = {"open": 0, "manifold": 0, "nonmanifold": 0}
    new_edge_set = set()

    for fid in new_face_indices:
        face = filled_mesh.faces[fid]
        has_open = False
        has_nonmanifold = False

        for j in range(3):
            key = _edge_tuple(face[j], face[(j + 1) % 3])
            new_edge_set.add(key)

            cnt = len(filled_edge_map.get(key, []))
            if cnt == 1:
                has_open = True
            elif cnt >= 3:
                has_nonmanifold = True

        if has_open:
            new_face_stats["open"] += 1
        elif has_nonmanifold:
            new_face_stats["nonmanifold"] += 1
        else:
            new_face_stats["manifold"] += 1

    new_edge_stats = {"open": 0, "manifold": 0, "nonmanifold": 0}
    for key in new_edge_set:
        cnt = len(filled_edge_map.get(key, []))
        new_edge_stats[_classify_edge_count(cnt)] += 1

    return {
        "boundary_edges_before": before,
        "boundary_edges_after": after,
        "new_faces": len(new_face_indices),
        "new_faces_by_type": new_face_stats,
        "new_unique_edges": len(new_edge_set),
        "new_edges_by_type": new_edge_stats,
    }


def print_seifert_fill_stats(stats):
    """
    命令行打印 Seifert 填充统计。
    """
    before = stats["boundary_edges_before"]
    after = stats["boundary_edges_after"]

    print("  Seifert 曲面局部填充对比（0层邻域）:")
    print(
        f"    健康孔洞边界边: "
        f"开放={before['open']} -> {after['open']}, "
        f"流形={before['manifold']} -> {after['manifold']}, "
        f"非流形={before['nonmanifold']} -> {after['nonmanifold']}"
    )
    print(
        f"    新增 Seifert 面片: {stats['new_faces']} 个 "
        f"(开放={stats['new_faces_by_type']['open']}, "
        f"流形={stats['new_faces_by_type']['manifold']}, "
        f"非流形={stats['new_faces_by_type']['nonmanifold']})"
    )
    print(
        f"    新增 Seifert 唯一边: {stats['new_unique_edges']} 条 "
        f"(开放={stats['new_edges_by_type']['open']}, "
        f"流形={stats['new_edges_by_type']['manifold']}, "
        f"非流形={stats['new_edges_by_type']['nonmanifold']})"
    )


def compute_seifert_curvature_stats(seifert_mesh, boundary_vertex_indices):
    """
    计算 Seifert 曲面内部顶点的离散曲率统计。
    """
    return compute_curvature_statistics(seifert_mesh, boundary_vertex_indices)


def apply_seifert_patch_to_mesh(mesh, seifert_mesh,
                                hole_vertex_indices,
                                seifert_boundary_indices):
    """
    将 Seifert 曲面合并到原始网格中，实现真正的孔洞修补。
    边界顶点共享原网格索引，内部顶点追加到末尾。
    """
    hole_vertex_indices = [int(v) for v in hole_vertex_indices]
    seifert_boundary_indices = [int(v) for v in seifert_boundary_indices]

    if len(hole_vertex_indices) != len(seifert_boundary_indices):
        raise ValueError("边界顶点映射长度不一致")

    seifert_to_orig = {
        seifert_boundary_indices[i]: hole_vertex_indices[i]
        for i in range(len(hole_vertex_indices))
    }

    orig_vertices = np.asarray(mesh.vertices, dtype=np.float64)
    seifert_vertices = np.asarray(seifert_mesh.vertices, dtype=np.float64)
    seifert_faces = np.asarray(seifert_mesh.faces, dtype=np.int64)

    new_vertices_start = len(orig_vertices)
    remapped_faces = []
    for tri in seifert_faces:
        new_tri = []
        for vid in tri:
            vid = int(vid)
            if vid in seifert_to_orig:
                new_tri.append(seifert_to_orig[vid])
            else:
                new_tri.append(new_vertices_start + vid)
        remapped_faces.append(new_tri)

    combined_vertices = np.vstack([orig_vertices, seifert_vertices])
    combined_faces = np.vstack([
        mesh.faces,
        np.array(remapped_faces, dtype=np.int64),
    ])

    repaired = trimesh.Trimesh(
        vertices=combined_vertices,
        faces=combined_faces,
        process=False,
    )
    return repaired


def repair_healthy_hole(mesh, hole, seifert_options=None, verbose=False):
    """
    修补单个健康孔洞。

    hole: hole_diagnosis.json 中的 healthy_holes 元素
    """
    if seifert_options is None:
        seifert_options = {}

    hole_id = hole["hole_id"]
    loop = hole["vertex_indices"]

    if verbose:
        print(f"修补孔洞 {hole_id}（{len(loop)} 条边）...")

    result = generate_seifert_surface(mesh, loop, **seifert_options)
    if not result["success"]:
        if verbose:
            print(f"  [FAIL] {result['message']}")
        return None, result["message"]

    repaired = apply_seifert_patch_to_mesh(
        mesh,
        result["mesh"],
        loop,
        result["boundary_indices"],
    )

    if verbose:
        print(f"  [OK] 新增 {len(result['mesh'].faces)} 个面片")

    return repaired, "success"


def repair_all_healthy_holes(mesh, healthy_holes, seifert_options=None, verbose=False):
    """
    顺序修补所有健康孔洞。
    返回 (repaired_mesh, repaired_ids, failed_records)
    """
    if seifert_options is None:
        seifert_options = {}

    current_mesh = mesh
    repaired_ids = []
    failed_records = []

    for hole in healthy_holes:
        hole_id = hole["hole_id"]
        new_mesh, msg = repair_healthy_hole(
            current_mesh, hole, seifert_options, verbose
        )
        if new_mesh is None:
            failed_records.append({"hole_id": hole_id, "message": msg})
        else:
            current_mesh = new_mesh
            repaired_ids.append(hole_id)

    return current_mesh, repaired_ids, failed_records


# ---------------------------------------------------------------------------
# Seifert 内部辅助函数
# ---------------------------------------------------------------------------

def _build_filled_mesh(neighborhood_mesh, seifert_mesh, loop_original_indices,
                       seifert_boundary_indices):
    """
    将邻域网格与 Seifert 曲面合并，使孔洞边界共享同一组顶点。
    """
    if len(loop_original_indices) != len(seifert_boundary_indices):
        print("  [WARN] Seifert 边界映射长度不一致，跳过填充对比")
        return None

    seifert_to_orig = {
        int(seifert_boundary_indices[i]): int(loop_original_indices[i])
        for i in range(len(loop_original_indices))
    }

    orig_n = len(neighborhood_mesh.vertices)
    seifert_faces = np.asarray(seifert_mesh.faces, dtype=np.int64)
    remapped_faces = []

    for tri in seifert_faces:
        new_tri = []
        for vid in tri:
            vid = int(vid)
            if vid in seifert_to_orig:
                new_tri.append(seifert_to_orig[vid])
            else:
                new_tri.append(orig_n + vid)
        remapped_faces.append(new_tri)

    remapped_faces = np.array(remapped_faces, dtype=np.int64)
    combined_vertices = np.vstack([neighborhood_mesh.vertices, seifert_mesh.vertices])
    combined_faces = np.vstack([neighborhood_mesh.faces, remapped_faces])

    filled_mesh = trimesh.Trimesh(
        vertices=combined_vertices,
        faces=combined_faces,
        process=False,
    )
    return filled_mesh


def _edge_tuple(v0, v1):
    v0 = int(v0)
    v1 = int(v1)
    return (v0, v1) if v0 < v1 else (v1, v0)


def _classify_edge_count(cnt):
    if cnt == 1:
        return "open"
    if cnt == 2:
        return "manifold"
    return "nonmanifold"


def _build_edge_tuple_to_faces(mesh, face_indices=None):
    """构建 (min_v, max_v) -> 共享面片索引列表 的完整映射，包含开放边。"""
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if face_indices is None:
        face_indices = np.arange(len(faces), dtype=np.int64)

    edge_map = {}
    for fid in face_indices:
        fid = int(fid)
        face = faces[fid]
        for j in range(3):
            v0 = int(face[j])
            v1 = int(face[(j + 1) % 3])
            key = _edge_tuple(v0, v1)
            edge_map.setdefault(key, []).append(fid)
    return edge_map
