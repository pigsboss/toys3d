"""Geometry utilities for mesh inspection and repair."""

import numpy as np
import trimesh


def compute_mesh_stats(mesh):
    """Compute basic statistics about a mesh."""
    stats = {}
    stats['vertices'] = int(len(mesh.vertices))
    stats['faces'] = int(len(mesh.faces))
    stats['edges'] = int(len(mesh.edges_unique))
    stats['is_watertight'] = bool(mesh.is_watertight)

    if len(mesh.edges_unique) > 0:
        v1 = mesh.vertices[mesh.edges_unique[:, 0]]
        v2 = mesh.vertices[mesh.edges_unique[:, 1]]
        edge_len = np.linalg.norm(v1 - v2, axis=1)
        stats['mean_edge_length'] = float(np.mean(edge_len))
        for p in [1, 5, 50, 95, 99]:
            stats[f'edge_length_p{p}'] = float(np.percentile(edge_len, p))
    else:
        stats['mean_edge_length'] = 0.0
        for p in [1, 5, 50, 95, 99]:
            stats[f'edge_length_p{p}'] = 0.0

    stats['area'] = float(mesh.area)
    return stats


def analyze_mesh_defects(mesh):
    """
    Analyze open and non‑manifold edges.

    Returns
    -------
    defect_stats : dict
    open_face_mask : np.ndarray, bool
    nonmanifold_face_mask : np.ndarray, bool
    """
    n_faces = len(mesh.faces)
    edges = np.asarray(mesh.edges, dtype=np.int64)
    edges_sorted = np.sort(edges, axis=1)

    edge_to_faces = {}
    for i, e in enumerate(edges_sorted):
        key = (int(e[0]), int(e[1]))
        face_idx = i // 3
        edge_to_faces.setdefault(key, []).append(face_idx)

    open_edges = 0
    nonmanifold_edges = 0
    open_face_mask = np.zeros(n_faces, dtype=bool)
    nonmanifold_face_mask = np.zeros(n_faces, dtype=bool)

    for faces in edge_to_faces.values():
        count = len(faces)
        if count == 1:
            open_edges += 1
            open_face_mask[faces] = True
        elif count > 2:
            nonmanifold_edges += 1
            nonmanifold_face_mask[faces] = True

    defect_stats = {
        'open_edges': open_edges,
        'nonmanifold_edges': nonmanifold_edges,
        'open_faces': int(open_face_mask.sum()),
        'nonmanifold_faces': int(nonmanifold_face_mask.sum()),
    }
    return defect_stats, open_face_mask, nonmanifold_face_mask


def repair_mesh_by_removing_duplicates(mesh):
    """Remove duplicate and degenerate faces."""
    if len(mesh.faces) == 0:
        return mesh.copy()

    faces = np.asarray(mesh.faces, dtype=np.int64)
    sorted_faces = np.sort(faces, axis=1)

    # First occurrence of each unique sorted face
    _, inverse = np.unique(sorted_faces, axis=0, return_inverse=True)
    first_occurrence = np.zeros(len(faces), dtype=bool)
    first_occurrence[np.unique(inverse, return_index=True)[1]] = True

    # Mark degenerate faces (zero area)
    verts = mesh.vertices
    tri_verts = verts[faces]
    cross = np.cross(tri_verts[:, 1] - tri_verts[:, 0],
                     tri_verts[:, 2] - tri_verts[:, 0], axis=1)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    valid_area = areas > 1e-12

    keep = first_occurrence & valid_area
    new_mesh = mesh.copy()
    new_mesh.update_faces(keep)
    new_mesh.remove_unreferenced_vertices()
    new_mesh.remove_degenerate_faces()
    return new_mesh


def repair_nonmanifold_edges(mesh, max_iterations=10, verbose=False):
    """Remove faces that are adjacent to non‑manifold edges."""
    repaired = mesh.copy()
    for _ in range(max_iterations):
        defects, _, nonmanifold_faces = analyze_mesh_defects(repaired)
        if defects['nonmanifold_edges'] == 0:
            break
        keep = ~nonmanifold_faces
        if not keep.any():
            break
        repaired.update_faces(keep)
        repaired.remove_unreferenced_vertices()
    return repaired


def _fill_holes_simple(mesh):
    """Fill holes using trimesh's built‑in repair."""
    from trimesh.repair import fill_holes
    fill_holes(mesh)
    return mesh


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
                        verbose=True):
    """Fill open boundary loops (adaptive parameters are accepted for API compatibility)."""
    return _fill_holes_simple(mesh.copy())


def extract_boundary_loops(mesh):
    """Extract closed boundary loops as lists of vertex indices."""
    edges = np.asarray(mesh.edges, dtype=np.int64)
    if len(edges) == 0:
        return []
    edges_sorted = np.sort(edges, axis=1)

    edge_counts = {}
    for e in edges_sorted:
        key = (int(e[0]), int(e[1]))
        edge_counts[key] = edge_counts.get(key, 0) + 1

    boundary = [key for key, cnt in edge_counts.items() if cnt == 1]
    if not boundary:
        return []

    adj = {}
    for a, b in boundary:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)

    loops = []
    visited = set()
    for start in list(adj.keys()):
        if start in visited:
            continue
        loop = []
        current = start
        prev = None
        while True:
            if current in visited:
                break
            loop.append(current)
            visited.add(current)
            nbrs = adj.get(current, [])
            nxt = None
            for nb in nbrs:
                if nb != prev:
                    nxt = nb
                    break
            if nxt is None:
                break
            prev = current
            current = nxt

        # Remove duplicate end if the loop is closed
        if loop and loop[0] == loop[-1]:
            loop = loop[:-1]

        if len(loop) > 1:
            loops.append(loop)

    return loops


def compute_loop_flatness(mesh, loop):
    """Return a flatness measure for a boundary loop (0 = perfect plane)."""
    pts = mesh.vertices[np.asarray(loop, dtype=np.int64)]
    if len(pts) < 3:
        return [0.0]

    centroid = pts.mean(axis=0)
    cov = (pts - centroid).T @ (pts - centroid)
    eigvals = np.linalg.eigvalsh(cov)
    max_eig = eigvals[-1]
    if max_eig <= 1e-12:
        return [0.0]
    flatness = eigvals[0] / max_eig
    return [float(flatness)]


def polygon_area_from_3d_ccw(pts):
    """Compute area of a 3D polygon using Newell's method."""
    if len(pts) < 3:
        return 0.0
    normal = np.zeros(3)
    n = len(pts)
    for i in range(n):
        v1 = pts[i]
        v2 = pts[(i + 1) % n]
        normal += np.cross(v1, v2)
    return 0.5 * np.linalg.norm(normal)


def repair_normals(mesh, verbose=False):
    """Fix inconsistent face normals."""
    mesh.fix_normals()
    return mesh


def remove_isolated_components(mesh,
                               min_faces=20,
                               min_area=None,
                               min_ratio=0.001,
                               verbose=False):
    """Remove small isolated components."""
    components = mesh.split(only_watertight=False)
    if len(components) <= 1:
        return mesh.copy()

    total_faces = len(mesh.faces)
    total_area = mesh.area
    keep = []
    for comp in components:
        nf = len(comp.faces)
        if nf < min_faces:
            continue
        if min_area is not None and comp.area < min_area:
            continue
        if nf / max(total_faces, 1) < min_ratio:
            continue
        keep.append(comp)

    if not keep:
        return mesh.copy()
    return trimesh.util.concatenate(keep)


def remove_quasi_isolated_components(mesh,
                                     radius=None,
                                     n_sample=2000,
                                     min_faces=30,
                                     max_ratio=0.05,
                                     remove_bridge=False,
                                     rng=None,
                                     verbose=False):
    """Remove components that are small and weakly connected (simplified)."""
    components = mesh.split(only_watertight=False)
    if len(components) <= 1:
        return mesh.copy()

    total_faces = len(mesh.faces)
    keep = []
    for comp in components:
        frac = len(comp.faces) / max(total_faces, 1)
        if frac > max_ratio:
            keep.append(comp)
        elif len(comp.faces) >= min_faces:
            keep.append(comp)
        # Otherwise discard

    if not keep:
        return mesh.copy()
    return trimesh.util.concatenate(keep)


def compute_hole_area_stats(mesh):
    """Compute statistics for boundary loops of a mesh."""
    loops = extract_boundary_loops(mesh)
    areas = []
    for loop in loops:
        pts = mesh.vertices[np.asarray(loop, dtype=np.int64)]
        areas.append(polygon_area_from_3d_ccw(pts))

    stats = {'count': len(loops)}
    if areas:
        areas = np.array(areas)
        stats['total_area'] = float(areas.sum())
        for p in [1, 5, 25, 50, 75, 90, 95, 99]:
            stats[f'p{p}_area'] = float(np.percentile(areas, p))
        stats['max_area'] = float(areas.max())
    else:
        stats['total_area'] = 0.0
        for p in [1, 5, 25, 50, 75, 90, 95, 99]:
            stats[f'p{p}_area'] = 0.0
        stats['max_area'] = 0.0
    return stats
