# src/toys3d/meshrepair.py
"""
网格修复工具集。

当前主要功能：
- 健康孔洞的 Seifert 极小曲面（固定边界）生成
- Seifert 填充前后局部拓扑统计
- Seifert 曲面曲率统计
"""

import numpy as np
import trimesh
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import json
import argparse
from pathlib import Path
from toys3d.geometrics import analyze_mesh_defects


def generate_seifert_surface(mesh, hole_vertex_indices,
                             optimize_iterations=200,
                             step_size=1.0,
                             tol=1e-7,
                             verbose=False):
    """
    为健康孔洞生成固定边界的 Seifert 极小曲面。
    """
    hole_vertex_indices = [int(v) for v in hole_vertex_indices]
    if len(hole_vertex_indices) < 3:
        return {
            "success": False,
            "mesh": None,
            "boundary_indices": [],
            "message": "孔洞边界顶点数不足 3，无法生成 Seifert 曲面",
        }

    disk_mesh, boundary_indices = _generate_initial_seifert_disk(
        mesh, hole_vertex_indices
    )
    if disk_mesh is None:
        return {
            "success": False,
            "mesh": None,
            "boundary_indices": [],
            "message": "无法生成初始 Seifert 圆盘",
        }

    seifert_mesh = _laplacian_smooth_fixed_boundary(
        disk_mesh,
        boundary_indices,
        iterations=optimize_iterations,
        step_size=step_size,
        tol=tol,
        verbose=verbose,
    )

    return {
        "success": True,
        "mesh": seifert_mesh,
        "boundary_indices": boundary_indices,
        "message": "Seifert 曲面生成成功",
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
    faces = np.asarray(seifert_mesh.faces, dtype=np.int64)
    vertices = np.asarray(seifert_mesh.vertices, dtype=np.float64)
    n_vertices = len(vertices)
    boundary_set = set(int(v) for v in boundary_vertex_indices)

    L = _build_cotangent_laplacian(seifert_mesh)

    area_faces = seifert_mesh.area_faces
    vertex_areas = np.bincount(
        faces.ravel(),
        weights=np.repeat(area_faces, 3),
        minlength=n_vertices,
    ) / 3.0
    vertex_areas[vertex_areas < 1e-12] = 1.0

    Hn = L @ vertices
    H_mag = np.linalg.norm(Hn, axis=1) / (2.0 * vertex_areas)

    angle_sum = np.zeros(n_vertices, dtype=np.float64)
    for face in faces:
        tri = vertices[face]
        for j in range(3):
            v_idx = int(face[j])
            p = tri[j]
            q = tri[(j + 1) % 3]
            r = tri[(j + 2) % 3]
            v1 = q - p
            v2 = r - p
            dot = np.dot(v1, v2)
            denom = np.linalg.norm(v1) * np.linalg.norm(v2)
            if denom < 1e-12:
                angle = 0.0
            else:
                cos_angle = np.clip(dot / denom, -1.0, 1.0)
                angle = float(np.arccos(cos_angle))
            angle_sum[v_idx] += angle
    K = (2.0 * np.pi - angle_sum) / vertex_areas

    interior_mask = np.array(
        [i not in boundary_set for i in range(n_vertices)],
        dtype=bool,
    )
    if not np.any(interior_mask):
        interior_mask = np.ones(n_vertices, dtype=bool)

    H_int = H_mag[interior_mask]
    K_int = K[interior_mask]

    return {
        "mean_abs_mean_curvature": float(np.mean(H_int)),
        "median_abs_mean_curvature": float(np.median(H_int)),
        "max_abs_mean_curvature": float(np.max(H_int)),
        "p95_abs_mean_curvature": float(np.percentile(H_int, 95)),
        "std_abs_mean_curvature": float(np.std(H_int)),
        "mean_gaussian_curvature": float(np.mean(K_int)),
        "median_gaussian_curvature": float(np.median(K_int)),
        "max_gaussian_curvature": float(np.max(K_int)),
        "min_gaussian_curvature": float(np.min(K_int)),
        "area": float(seifert_mesh.area),
        "perimeter": float(np.sum(np.linalg.norm(
            vertices[boundary_vertex_indices] -
            np.roll(vertices[boundary_vertex_indices], -1, axis=0), axis=1))),
    }


# ---------------------------------------------------------------------------
# 内部辅助函数
# ---------------------------------------------------------------------------

def _generate_initial_seifert_disk(mesh, loop_vertices):
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
            raise ValueError("invalid 2D vertices")
        if tri_faces.ndim != 2 or tri_faces.shape[1] != 3 or len(tri_faces) == 0:
            raise ValueError("empty or invalid faces")

        v3d = centroid + tri_vertices_2d[:, 0:1] * u + tri_vertices_2d[:, 1:2] * v

        boundary_indices = []
        for p2d in poly2d:
            dists = np.linalg.norm(tri_vertices_2d - p2d, axis=1)
            idx = int(np.argmin(dists))
            if dists[idx] > 1e-8:
                raise ValueError("boundary point not found")
            boundary_indices.append(idx)

        disk = trimesh.Trimesh(
            vertices=v3d,
            faces=tri_faces,
            process=False,
        )

        return disk, boundary_indices

    except Exception as e:
        print(f"  [WARN] 初始 Seifert 圆盘生成失败: {e}")
        return None, []


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


def _build_cotangent_laplacian(mesh):
    """
    构建当前网格的余切权重 Laplacian 矩阵。
    """
    faces = np.asarray(mesh.faces, dtype=np.int64)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    n_vertices = len(vertices)
    if n_vertices == 0:
        return csr_matrix((0, 0))

    row = []
    col = []
    data = []

    for fid, face in enumerate(faces):
        tri = vertices[face]
        a, b, c = tri[0], tri[1], tri[2]

        def angle_at(p, q, r):
            v1 = q - p
            v2 = r - p
            dot = np.dot(v1, v2)
            denom = np.linalg.norm(v1) * np.linalg.norm(v2)
            if denom < 1e-12:
                return 0.0
            cos_angle = np.clip(dot / denom, -1.0, 1.0)
            return float(np.arccos(cos_angle))

        alpha = angle_at(a, b, c)
        beta = angle_at(b, c, a)
        gamma = angle_at(c, a, b)

        def add_weight(e0, e1, w):
            if w == 0:
                return
            row.append(e0)
            col.append(e1)
            data.append(w)
            row.append(e1)
            col.append(e0)
            data.append(w)

        e0 = (int(face[1]), int(face[2])) if face[1] < face[2] else (int(face[2]), int(face[1]))
        e1 = (int(face[2]), int(face[0])) if face[2] < face[0] else (int(face[0]), int(face[2]))
        e2 = (int(face[0]), int(face[1])) if face[0] < face[1] else (int(face[1]), int(face[0]))

        cot_alpha = 1.0 / np.tan(alpha) if abs(np.tan(alpha)) > 1e-12 else 0.0
        cot_beta = 1.0 / np.tan(beta) if abs(np.tan(beta)) > 1e-12 else 0.0
        cot_gamma = 1.0 / np.tan(gamma) if abs(np.tan(gamma)) > 1e-12 else 0.0

        add_weight(e0[0], e0[1], cot_alpha)
        add_weight(e1[0], e1[1], cot_beta)
        add_weight(e2[0], e2[1], cot_gamma)

    if not row:
        return csr_matrix((n_vertices, n_vertices))

    L = csr_matrix(
        (np.array(data, dtype=np.float64),
         (np.array(row, dtype=np.int64), np.array(col, dtype=np.int64))),
        shape=(n_vertices, n_vertices),
    )

    row_sums = np.asarray(L.sum(axis=1)).ravel()
    L = L - csr_matrix(
        (row_sums, (np.arange(n_vertices), np.arange(n_vertices))),
        shape=(n_vertices, n_vertices),
    )

    return L


def _laplacian_smooth_fixed_boundary(mesh, boundary_vertex_indices,
                                     iterations=200, step_size=1.0,
                                     tol=1e-7, verbose=False):
    """
    固定边界顶点，内部顶点按离散 Plateau 问题迭代求解。
    """
    faces = np.asarray(mesh.faces, dtype=np.int64)
    vertices = mesh.vertices.copy()
    n_vertices = len(vertices)
    boundary_set = set(int(v) for v in boundary_vertex_indices)

    if n_vertices == 0 or len(faces) == 0:
        return mesh.copy()

    all_indices = np.arange(n_vertices, dtype=np.int64)
    interior_indices = np.array(
        [i for i in all_indices if int(i) not in boundary_set],
        dtype=np.int64,
    )
    boundary_indices = np.array(sorted(boundary_set), dtype=np.int64)

    if len(interior_indices) == 0:
        return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    step_size = float(np.clip(step_size, 0.0, 1.0))

    for it in range(iterations):
        current = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        L = _build_cotangent_laplacian(current)

        Lint = L[interior_indices, :][:, interior_indices].tocsr()
        Lbnd = L[interior_indices, :][:, boundary_indices].tocsr()

        n_int = len(interior_indices)
        eps_reg = 1e-10
        Lint = Lint + csr_matrix(np.eye(n_int, dtype=np.float64) * eps_reg)

        rhs = -Lbnd @ vertices[boundary_indices]
        sol = spsolve(Lint, rhs)

        new_vertices = vertices.copy()
        new_vertices[interior_indices] = (
            vertices[interior_indices] +
            step_size * (sol - vertices[interior_indices])
        )

        moves = np.linalg.norm(
            new_vertices[interior_indices] - vertices[interior_indices],
            axis=1,
        )
        max_move = float(np.max(moves)) if len(moves) > 0 else 0.0
        vertices = new_vertices

        if max_move < tol:
            if verbose:
                print(f"    Seifert 优化在第 {it+1} 次迭代收敛，最大位移 {max_move:.6e}")
            break

    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


# ---------------------------------------------------------------------------
# 新增：Seifert 修补整合，并支持命令行入口
# ---------------------------------------------------------------------------

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


def compute_global_mesh_stats(mesh):
    """
    统计网格全局边/面属性。
    """
    defect_stats, open_face_mask, nonmanifold_face_mask = analyze_mesh_defects(mesh)

    total_edges = len(mesh.edges_unique)
    open_edges = defect_stats["open_edges"]
    nonmanifold_edges = defect_stats["nonmanifold_edges"]
    manifold_edges = total_edges - open_edges - nonmanifold_edges

    total_faces = len(mesh.faces)
    open_faces = int(open_face_mask.sum())
    nonmanifold_faces = int(nonmanifold_face_mask.sum())
    union_mask = open_face_mask | nonmanifold_face_mask
    manifold_faces = total_faces - int(union_mask.sum())

    return {
        "total_edges": total_edges,
        "open_edges": open_edges,
        "manifold_edges": manifold_edges,
        "nonmanifold_edges": nonmanifold_edges,
        "total_faces": total_faces,
        "open_faces": open_faces,
        "manifold_faces": manifold_faces,
        "nonmanifold_faces": nonmanifold_faces,
    }


def print_global_mesh_stats(label, stats):
    print(f"  {label}:")
    print(f"    开放边:     {stats['open_edges']}")
    print(f"    流形边:     {stats['manifold_edges']}")
    print(f"    非流形边:   {stats['nonmanifold_edges']}")
    print(f"    总边数:     {stats['total_edges']}")
    print(f"    开放面片:   {stats['open_faces']}")
    print(f"    流形面片:   {stats['manifold_faces']}")
    print(f"    非流形面片: {stats['nonmanifold_faces']}")
    print(f"    总面片数:   {stats['total_faces']}")


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


def _main():
    parser = argparse.ArgumentParser(
        description="网格修复工具：对健康孔洞生成并合并 Seifert 极小曲面。"
    )
    parser.add_argument("input_file", help="输入网格文件 (ply/stl/obj)")
    parser.add_argument("output_file", help="输出修补后的网格文件")
    parser.add_argument(
        "--hole-diagnosis-dir",
        default="hole_diagnosis_report",
        help="hole diagnosis 输出目录（默认 hole_diagnosis_report）",
    )
    parser.add_argument(
        "--hole-id",
        type=int,
        default=None,
        help="指定修补的健康孔洞 ID；不指定则修补所有健康孔洞",
    )
    parser.add_argument(
        "--seifert-optimize-iterations",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--seifert-step-size",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--seifert-tolerance",
        type=float,
        default=1e-7,
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="打印详细过程",
    )

    args = parser.parse_args()

    print(f"加载网格: {args.input_file}")
    mesh = trimesh.load(args.input_file, force="mesh")

    diag_path = Path(args.hole_diagnosis_dir) / "hole_diagnosis.json"
    if not diag_path.exists():
        raise FileNotFoundError(f"未找到 {diag_path}")

    with open(diag_path, "r", encoding="utf-8") as f:
        diag = json.load(f)

    healthy_holes = diag.get("healthy_holes", [])
    if not healthy_holes:
        print("未找到健康孔洞，无需修补。")
        mesh.export(args.output_file)
        return

    seifert_options = {
        "optimize_iterations": args.seifert_optimize_iterations,
        "step_size": args.seifert_step_size,
        "tol": args.seifert_tolerance,
        "verbose": args.verbose,
    }

    print("\n修补前网格统计:")
    before_stats = compute_global_mesh_stats(mesh)
    print_global_mesh_stats("修补前", before_stats)
    print(f"  健康孔洞总数: {len(healthy_holes)}")

    if args.hole_id is not None:
        hole = next(
            (h for h in healthy_holes if h["hole_id"] == args.hole_id),
            None,
        )
        if hole is None:
            raise ValueError(f"未找到 hole_id={args.hole_id} 的健康孔洞")
        repaired_mesh, msg = repair_healthy_hole(
            mesh, hole, seifert_options, verbose=args.verbose
        )
        if repaired_mesh is None:
            raise RuntimeError(f"修补失败: {msg}")
        repaired_ids = [args.hole_id]
        failed_records = []
    else:
        repaired_mesh, repaired_ids, failed_records = repair_all_healthy_holes(
            mesh, healthy_holes, seifert_options, verbose=args.verbose
        )

    print("\n修补后网格统计:")
    after_stats = compute_global_mesh_stats(repaired_mesh)
    print_global_mesh_stats("修补后", after_stats)

    print(f"\n成功修补孔洞: {repaired_ids}")
    print(f"剩余健康孔洞: {len(healthy_holes) - len(repaired_ids)}")
    if failed_records:
        print("失败记录:")
        for rec in failed_records:
            print(f"  hole_id={rec['hole_id']}: {rec['message']}")

    repaired_mesh.export(args.output_file)
    print(f"\n修补后网格已保存: {args.output_file}")


if __name__ == "__main__":
    _main()
