# src/toys3d/geometrics/generation.py
"""生成/修复类工具。"""
import numpy as np
import trimesh


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
