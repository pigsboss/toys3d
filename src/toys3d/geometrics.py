diff --git a/src/toys3d/geometrics.py b/src/toys3d/geometrics.py
--- a/src/toys3d/geometrics.py
+++ b/src/toys3d/geometrics.py
@@ -1635,7 +1635,214 @@ def _extract_boundary_loops_from_edge_keys(mesh, edge_keys):
     return loops
+
+
+def _expand_face_neighborhood(mesh, seed_faces, depth):
+    """
+    沿面邻接关系扩展种子面片集合，返回扩展后的面片索引集合。
+    depth=0 返回空集合；depth=1 返回种子面片本身。
+    """
+    if depth <= 0:
+        return set()
+    seed_faces = set(map(int, seed_faces))
+    if depth == 1:
+        return seed_faces.copy()
+
+    n_faces = len(mesh.faces)
+    adjacency = [[] for _ in range(n_faces)]
+    for f0, f1 in mesh.face_adjacency:
+        adjacency[int(f0)].append(int(f1))
+        adjacency[int(f1)].append(int(f0))
+
+    current = list(seed_faces)
+    visited = set(seed_faces)
+
+    for _ in range(depth - 1):
+        next_layer = []
+        for f in current:
+            for nb in adjacency[f]:
+                if nb not in visited:
+                    visited.add(nb)
+                    next_layer.append(nb)
+        current = next_layer
+        if not current:
+            break
+    return visited
+
+
+def _extract_component_point_cloud(mesh, component, neighborhood_depth):
+    """
+    从组件的邻域面片提取点云坐标与法向。
+
+    返回:
+        points : (N,3) float64
+        normals : (N,3) float64
+    """
+    seed_faces = set(map(int, component.get('face_ids', [])))
+    if not seed_faces:
+        return np.zeros((0, 3)), np.zeros((0, 3))
+
+    expanded = _expand_face_neighborhood(mesh, seed_faces, neighborhood_depth)
+    if not expanded:
+        expanded = seed_faces
+
+    faces_idx = np.asarray(sorted(expanded), dtype=np.int64)
+    submesh = mesh.submesh([faces_idx])[0]
+
+    verts = submesh.vertices
+    if hasattr(submesh, 'vertex_normals') and submesh.vertex_normals is not None:
+        vert_normals = submesh.vertex_normals
+    else:
+        face_normals = submesh.face_normals
+        vertex_face_count = np.bincount(submesh.faces.ravel(), minlength=len(verts))
+        vert_normals = np.zeros_like(verts)
+        for i, face in enumerate(submesh.faces):
+            for v in face:
+                vert_normals[v] += face_normals[i]
+        norms = np.linalg.norm(vert_normals, axis=1, keepdims=True)
+        norms[norms < 1e-12] = 1.0
+        vert_normals = vert_normals / norms
+
+    face_centers = submesh.triangles_center
+    face_normals = submesh.face_normals
+
+    points = np.vstack([verts, face_centers])
+    normals = np.vstack([vert_normals, face_normals])
+
+    centroid = points.mean(axis=0)
+    to_centroid = centroid - points
+    dot = np.sum(normals * to_centroid, axis=1)
+    flip = dot < 0
+    normals[flip] *= -1
+
+    return points, normals
+
+
+def _check_watertight_genus0(mesh):
+    """
+    检查网格是否水密且亏格为0（球面同胚）。
+    """
+    if not mesh.is_watertight:
+        return False
+    try:
+        euler = mesh.euler_number
+    except AttributeError:
+        V = len(mesh.vertices)
+        E = len(mesh.edges_unique)
+        F = len(mesh.faces)
+        euler = V - E + F
+    return abs(euler - 2) < 1e-6
+
+
+def _repair_to_watertight_mesh(mesh, voxel_size=None):
+    """
+    对非水密网格进行体素化+Marching Cubes修复，返回水密网格。
+    """
+    if mesh.is_watertight:
+        return mesh
+    if voxel_size is None:
+        bounds = mesh.bounds
+        diag = np.linalg.norm(bounds[1] - bounds[0])
+        voxel_size = diag / 128
+    vox = mesh.voxelized(voxel_size)
+    try:
+        vox = vox.fill()
+    except Exception:
+        pass
+    repaired = vox.marching_cubes
+    return repaired
+
+
+def fit_watertight_patch_from_component(
+    mesh,
+    component,
+    method='poisson',
+    neighborhood_depth=2,
+    poisson_depth=8,
+    density_quantile=0.2,
+    alpha=1.5,
+):
+    """
+    从边界组件的邻域点云生成水密包络曲面，并返回曲面及交线。
+
+    参数:
+        method : str, 可选 'poisson', 'convex_hull', 'concave_hull'
+        neighborhood_depth : int, 点云提取的邻域深度
+        poisson_depth : int, 泊松重建深度
+        density_quantile : float, 泊松密度过滤分位
+        alpha : float, 凹包算法的 alpha 参数
+
+    返回:
+        dict:
+            success : bool
+            message : str
+            watertight_mesh : trimesh.Trimesh 或 None
+            intersection_vertices : list of list of float
+            intersection_edges : list of list of int
+    """
+    points, normals = _extract_component_point_cloud(
+        mesh, component, neighborhood_depth
+    )
+    if len(points) < 4:
+        return {
+            'success': False,
+            'message': '点云点数不足，无法拟合曲面',
+            'watertight_mesh': None,
+            'intersection_vertices': [],
+            'intersection_edges': [],
+        }
+
+    watertight_mesh = None
+    message = ''
+    try:
+        if method == 'poisson':
+            try:
+                import open3d as o3d
+            except ImportError:
+                return {
+                    'success': False,
+                    'message': '未安装 Open3D，无法使用泊松重建',
+                    'watertight_mesh': None,
+                    'intersection_vertices': [],
+                    'intersection_edges': [],
+                }
+
+            pcd = o3d.geometry.PointCloud()
+            pcd.points = o3d.utility.Vector3dVector(points)
+            pcd.normals = o3d.utility.Vector3dVector(normals)
+
+            mesh_poisson, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
+                pcd, depth=poisson_depth, scale=1.1, linear_fit=False
+            )
+            densities = np.asarray(densities)
+            if density_quantile > 0 and len(densities) > 0:
+                threshold = np.quantile(densities, density_quantile)
+                mesh_poisson.remove_vertices_by_mask(densities < threshold)
+            watertight_mesh = trimesh.Trimesh(
+                vertices=np.asarray(mesh_poisson.vertices),
+                faces=np.asarray(mesh_poisson.triangles),
+                process=False,
+            )
+            if not _check_watertight_genus0(watertight_mesh):
+                watertight_mesh = _repair_to_watertight_mesh(watertight_mesh)
+                if not _check_watertight_genus0(watertight_mesh):
+                    return {
+                        'success': False,
+                        'message': '泊松重建结果未通过水密/亏格0检查',
+                        'watertight_mesh': None,
+                        'intersection_vertices': [],
+                        'intersection_edges': [],
+                    }
+            message = '泊松重建成功'
+
+        elif method == 'convex_hull':
+            from scipy.spatial import ConvexHull
+            hull = ConvexHull(points)
+            watertight_mesh = trimesh.Trimesh(
+                vertices=hull.points,
+                faces=hull.simplices,
+                process=False,
+            )
+            message = '凸包生成成功'
+
+        elif method == 'concave_hull':
+            import open3d as o3d
+            pcd = o3d.geometry.PointCloud()
+            pcd.points = o3d.utility.Vector3dVector(points)
+            pcd.normals = o3d.utility.Vector3dVector(normals)
+            mesh_alpha = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
+                pcd, alpha
+            )
+            watertight_mesh = trimesh.Trimesh(
+                vertices=np.asarray(mesh_alpha.vertices),
+                faces=np.asarray(mesh_alpha.triangles),
+                process=False,
+            )
+            if not _check_watertight_genus0(watertight_mesh):
+                watertight_mesh = _repair_to_watertight_mesh(watertight_mesh)
+                if not _check_watertight_genus0(watertight_mesh):
+                    return {
+                        'success': False,
+                        'message': '凹包结果未通过水密/亏格0检查',
+                        'watertight_mesh': None,
+                        'intersection_vertices': [],
+                        'intersection_edges': [],
+                    }
+            message = '凹包生成成功'
+        else:
+            return {
+                'success': False,
+                'message': f'未知算法: {method}',
+                'watertight_mesh': None,
+                'intersection_vertices': [],
+                'intersection_edges': [],
+            }
+    except Exception as e:
+        return {
+            'success': False,
+            'message': str(e),
+            'watertight_mesh': None,
+            'intersection_vertices': [],
+            'intersection_edges': [],
+        }
+
+    seed_faces = set(map(int, component.get('face_ids', [])))
+    expanded = _expand_face_neighborhood(mesh, seed_faces, neighborhood_depth)
+    if not expanded:
+        expanded = seed_faces
+    submesh = mesh.submesh([np.asarray(sorted(expanded), dtype=np.int64)])[0]
+    boundary_loops = extract_boundary_loops(submesh)
+
+    intersection_vertices = []
+    intersection_edges = []
+
+    for loop in boundary_loops:
+        loop_verts = np.asarray(loop, dtype=np.int64)
+        pts = submesh.vertices[loop_verts]
+        proj_pts, _, _ = watertight_mesh.nearest.on_surface(pts)
+        start_idx = len(intersection_vertices)
+        for p in proj_pts:
+            intersection_vertices.append(p.tolist())
+        for i in range(len(loop)):
+            v0 = start_idx + i
+            v1 = start_idx + (i + 1) % len(loop)
+            intersection_edges.append([v0, v1])
+
+    if not intersection_vertices:
+        return {
+            'success': False,
+            'message': '未能提取到交线',
+            'watertight_mesh': None,
+            'intersection_vertices': [],
+            'intersection_edges': [],
+        }
+
+    return {
+        'success': True,
+        'message': message,
+        'watertight_mesh': watertight_mesh,
+        'intersection_vertices': intersection_vertices,
+        'intersection_edges': intersection_edges,
+    }
