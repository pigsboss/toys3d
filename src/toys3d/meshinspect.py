diff --git a/src/toys3d/meshinspect.py b/src/toys3d/meshinspect.py
--- a/src/toys3d/meshinspect.py
+++ b/src/toys3d/meshinspect.py
@@ -1,8 +1,9 @@
 import sys
 import os
 
 # Ensure src directory is on the path so that 'toys3d' can be imported
 _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
 _src_parent = os.path.dirname(_project_root)
 if _src_parent not in sys.path:
     sys.path.insert(0, _src_parent)
@@ -50,7 +51,8 @@ from toys3d.geometrics import (
     build_manifold_face_adjacency,
     is_manifold_closed_boundary,
     find_minimal_enclosing_manifold_boundary_greedy,
+    fit_watertight_patch_from_component,
 )
 
 def compute_face_area_stats(mesh):
@@ -2072,6 +2074,21 @@ def _build_vertex_face_csr(mesh):
     return csr_matrix((data, (row_idx, col_idx)), shape=(n_vertices, n_faces))
 
+def _parse_color_string_flexible(s):
+    """将 'R,G,B' 或 'R,G,B,A' 字符串解析为整数列表。"""
+    parts = s.split(',')
+    if len(parts) not in (3, 4):
+        raise argparse.ArgumentTypeError(
+            f"Color must be 'R,G,B' or 'R,G,B,A', got '{s}'"
+        )
+    try:
+        return [int(p.strip()) for p in parts]
+    except ValueError:
+        raise argparse.ArgumentTypeError(
+            f"Color components must be integers, got '{s}'"
+        )
+
+
 def expand_face_neighborhood(mesh, seed_faces, depth):
     """
     从种子面片出发，返回拓扑邻域扩展 depth 层后的面片索引集合。
@@ -2324,6 +2341,19 @@ def visualize_boundary_component(mesh, args):
     if enclosing.get("success"):
         enclosing_vertices = enclosing.get("boundary_vertices", [])
         enclosing_radius = radius * 1.5   # 稍粗，更醒目
 
         for loop_verts in enclosing_vertices:
             for i in range(len(loop_verts) - 1):
                 v0 = loop_verts[i]
                 v1 = loop_verts[i + 1]
                 seg = trimesh.creation.cylinder(
                     radius=enclosing_radius,
                     segment=[mesh.vertices[v0], mesh.vertices[v1]],
                     sections=6,
                 )
                 seg.visual.face_colors = [255, 0, 255, 255]  # 洋红色
                 scene.add_geometry(seg)
+
+    # 拟合水密包络曲面并显示交线
+    if args.fit_watertight_patch:
+        print("拟合水密包络曲面...")
+        patch_result = fit_watertight_patch_from_component(
+            mesh,
+            comp,
+            method=args.patch_method,
+            neighborhood_depth=args.patch_neighborhood_depth,
+            poisson_depth=args.patch_poisson_depth,
+            density_quantile=args.patch_density_quantile,
+            alpha=args.patch_alpha,
+        )
+        if patch_result["success"]:
+            watertight_mesh = patch_result["watertight_mesh"]
+            intersection_vertices = patch_result["intersection_vertices"]
+            intersection_edges = patch_result["intersection_edges"]
+
+            # 显示拟合曲面（半透明青色）
+            watertight_mesh.visual.face_colors = [0, 200, 200, 80]
+            scene.add_geometry(watertight_mesh)
+
+            # 显示交线（洋红色圆柱）
+            for edge in intersection_edges:
+                p0 = intersection_vertices[edge[0]]
+                p1 = intersection_vertices[edge[1]]
+                seg = trimesh.creation.cylinder(
+                    radius=radius * 1.2,
+                    segment=[p0, p1],
+                    sections=5,
+                )
+                seg.visual.face_colors = [255, 0, 255, 255]
+                scene.add_geometry(seg)
+
+            print(f"  拟合成功：交线 {len(intersection_vertices)} 个顶点，"
+                  f"{len(intersection_edges)} 条边")
+        else:
+            print(f"  [WARN] 水密包络拟合失败: {patch_result['message']}")
@@ -2538,6 +2588,18 @@ def main():
     parser.add_argument("--boundary-show-original", action="store_true",
                         help="同时显示原始网格（半透明背景）")
     parser.add_argument("--hole-diagnosis-output", type=str, default="hole_diagnosis_report",
                         help="孔洞诊断输出目录（默认 hole_diagnosis_report）")
+    parser.add_argument("--fit-watertight-patch", action="store_true",
+                        help="拟合水密包络曲面并显示交线")
+    parser.add_argument("--patch-method",
+                        choices=["poisson", "convex_hull", "concave_hull"],
+                        default="poisson",
+                        help="水密包络曲面生成算法（默认 poisson）")
+    parser.add_argument("--patch-neighborhood-depth", type=int, default=2,
+                        help="点云邻域扩展深度，默认 2")
+    parser.add_argument("--patch-poisson-depth", type=int, default=8,
+                        help="泊松重建深度，默认 8")
+    parser.add_argument("--patch-density-quantile", type=float, default=0.2,
+                        help="泊松密度过滤分位，默认 0.2")
+    parser.add_argument("--patch-alpha", type=float, default=1.5,
+                        help="凹包算法的 alpha 参数（仅 --patch-method concave_hull 时有效）")
+    parser.add_argument("--patch-voxel-size", type=float, default=None,
+                        help="凹包体素重建时的体素大小（可选，自动计算）")
     parser.add_argument("--hole-diagnosis-output", type=str, default="hole_diagnosis_report",
                         help="孔洞诊断输出目录（默认 hole_diagnosis_report）")
