import sys
import os

# Ensure src directory is on the path so that 'toys3d' can be imported
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_src_parent = os.path.dirname(_project_root)
if _src_parent not in sys.path:
    sys.path.insert(0, _src_parent)

# If the current working directory contains an `inspect.py`, it would shadow
# the standard library module and cause circular imports (e.g., in NumPy).
if os.path.exists(os.path.join(os.getcwd(), 'inspect.py')):
    if '' in sys.path:
        sys.path.remove('')

import argparse
import colorsys
import numpy as np
import trimesh

from toys3d.geometrics import (
    compute_mesh_stats,
    analyze_mesh_defects,
    compute_hole_area_stats,
    extract_boundary_loops,
    polygon_area_from_3d_ccw,
)


def compute_face_area_stats(mesh):
    """
    计算三角面片面积的统计量。
    """
    areas = mesh.area_faces
    stats = {}
    if len(areas) == 0:
        stats['count'] = 0
        for key in ['mean', 'min', 'max', 'p1', 'p5', 'p10',
                    'p25', 'p50', 'p75', 'p90', 'p95', 'p99']:
            stats[key] = 0.0
        return stats

    stats['count'] = int(len(areas))
    stats['mean'] = float(np.mean(areas))
    stats['min'] = float(np.min(areas))
    stats['max'] = float(np.max(areas))
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        stats[f'p{p}'] = float(np.percentile(areas, p))
    return stats


def compute_bounding_box_stats(mesh):
    """
    计算包围盒相关统计。
    """
    bbox = mesh.bounding_box
    extents = bbox.extents
    return {
        'min': bbox.bounds[0],
        'max': bbox.bounds[1],
        'extents': extents,
        'diagonal': float(np.linalg.norm(extents)),
        'centroid': bbox.centroid,
    }


def compute_volume_if_closed(mesh):
    """
    若网格水密，返回体积；否则返回 NaN。
    """
    if mesh.is_watertight:
        return float(mesh.volume)
    return float(np.nan)


def build_defect_visualization(mesh, open_face_mask, nonmanifold_face_mask):
    """
    生成缺陷可视化网格：
    - 灰色：正常面片
    - 黄色：开放边界附近面片
    - 红色：非流形边附近面片
    - 橙色：同时具有两种缺陷的面片
    """
    vis = mesh.copy()
    N = len(vis.faces)

    colors = np.full((N, 4), 200, dtype=np.uint8)
    colors[:, 3] = 255

    open_only = open_face_mask & ~nonmanifold_face_mask
    nonmanifold_only = nonmanifold_face_mask & ~open_face_mask
    both = open_face_mask & nonmanifold_face_mask

    colors[open_only] = [255, 220, 0, 255]      # 黄
    colors[nonmanifold_only] = [255, 0, 0, 255]  # 红
    colors[both] = [255, 128, 0, 255]            # 橙

    vis.visual.face_colors = colors
    return vis


def make_double_sided(mesh, backface_color=None):
    """
    将网格渲染为双面几何，避免薄壳背面被背面剔除而显示为透明。

    Parameters
    ----------
    mesh : trimesh.Trimesh
    backface_color : list or None
        背面子颜色（RGBA）。
        若为 None，则根据每个正面颜色自动生成同色系暗色：
        保持 Hue 不变，Saturation * 0.8，Value * 0.5。
    """
    faces = np.asarray(mesh.faces, dtype=np.int64).reshape(-1, 3)
    if len(faces) == 0:
        return mesh.copy()

    double_faces = np.vstack([
        faces,
        faces[:, ::-1],
    ])

    vis = trimesh.Trimesh(
        vertices=mesh.vertices.copy(),
        faces=double_faces,
        process=False,
    )

    if hasattr(mesh.visual, 'face_colors') and mesh.visual.face_colors.shape[0] == len(faces):
        colors = np.asarray(mesh.visual.face_colors)
    else:
        colors = np.full((len(faces), 4), [200, 200, 200, 255], dtype=np.uint8)

    if backface_color is None:
        # 根据每个正面颜色自动生成同色系暗色背面颜色
        n = len(colors)
        back_colors = np.empty_like(colors)
        for i in range(n):
            r, g, b, a = colors[i].astype(np.float64) / 255.0
            h, s, v = colorsys.rgb_to_hsv(r, g, b)
            back_s = np.clip(s * 0.8, 0.0, 1.0)
            back_v = np.clip(v * 0.5, 0.0, 1.0)
            br, bg, bb = colorsys.hsv_to_rgb(h, back_s, back_v)
            back_colors[i] = np.array([
                br * 255.0,
                bg * 255.0,
                bb * 255.0,
                a * 255.0,
            ], dtype=np.uint8)
    else:
        back_colors = np.full_like(colors, np.asarray(backface_color, dtype=np.uint8))

    vis.visual.face_colors = np.vstack([colors, back_colors])
    return vis


def add_wireframe_to_scene(scene, mesh, color=None, radius=None):
    """
    将网格的边以圆柱线段形式加入场景，用于观察三角剖分。

    Parameters
    ----------
    scene : trimesh.Scene
    mesh : trimesh.Trimesh
    color : list or tuple or ndarray or None
        RGBA 颜色，默认纯黑不透明 [0, 0, 0, 255]。
    radius : float or None
        圆柱半径，默认基于包围盒对角线的 0.05%。
    """
    if color is None:
        color = np.array([0, 0, 0, 255], dtype=np.uint8)
    else:
        color = np.asarray(color, dtype=np.uint8)

    edges_unique = mesh.edges_unique
    if len(edges_unique) == 0:
        return

    if radius is None or radius <= 0:
        diag = float(np.linalg.norm(mesh.bounding_box.extents))
        radius = max(diag * 0.0005, 1e-6)

    for e in edges_unique:
        v0, v1 = mesh.vertices[e[0]], mesh.vertices[e[1]]
        seg = trimesh.creation.cylinder(
            radius=radius,
            segment=[v0, v1],
            sections=4,
        )
        seg.visual.face_colors = color
        scene.add_geometry(seg)


def _high_saturation_hole_palette():
    """
    返回一组高饱和度、且相互区分的 RGBA 颜色。
    """
    palette = [
        (255,   0,   0, 255),   # 红
        (  0, 255,   0, 255),   # 绿
        (  0, 128, 255, 255),   # 蓝
        (255, 255,   0, 255),   # 黄
        (255,   0, 255, 255),   # 品红
        (  0, 255, 255, 255),   # 青
        (255, 128,   0, 255),   # 橙
        (128,   0, 255, 255),   # 紫
        (  0, 255, 128, 255),   # 春绿
        (255,   0, 128, 255),   # 粉红
    ]
    return [np.array(c, dtype=np.uint8) for c in palette]


def _greedy_color_hole_loops(loops):
    """
    为孔洞边界环分配调色板颜色索引。

    如果两个孔洞共享顶点，则认为它们相邻，应使用不同颜色。
    使用贪心染色。返回 (color_indices, palette)。
    """
    n = len(loops)
    if n == 0:
        return [], _high_saturation_hole_palette()

    vertex_to_loops = {}
    for i, loop in enumerate(loops):
        for v in loop:
            vertex_to_loops.setdefault(int(v), []).append(i)

    adjacency = [set() for _ in range(n)]
    for loop_indices in vertex_to_loops.values():
        if len(loop_indices) <= 1:
            continue
        for i in loop_indices:
            for j in loop_indices:
                if i != j:
                    adjacency[i].add(j)
                    adjacency[j].add(i)

    palette = _high_saturation_hole_palette()
    color_indices = [None] * n

    for i in range(n):
        used = {color_indices[j] for j in adjacency[i]
                if color_indices[j] is not None}
        chosen = None
        for c in range(len(palette)):
            if c not in used:
                chosen = c
                break
        if chosen is None:
            chosen = i % len(palette)
        color_indices[i] = chosen

    return color_indices, palette


def add_hole_boundaries_to_scene(scene, mesh, radius=None,
                                 min_edges=3, min_area=0.0,
                                 verbose=False):
    """
    检测网格中的闭合孔洞边界环，并在场景中用高饱和度颜色绘制。

    只绘制边长和面积均满足阈值的闭合环。
    """
    loops = extract_boundary_loops(mesh)

    filtered_loops = []
    for loop in loops:
        if len(loop) < min_edges:
            continue
        pts = mesh.vertices[np.array(loop)]
        area = polygon_area_from_3d_ccw(pts)
        if area < min_area:
            continue
        filtered_loops.append(loop)

    if not filtered_loops:
        if verbose:
            print("  No closed hole boundary loops matching thresholds.")
        return

    color_indices, palette = _greedy_color_hole_loops(filtered_loops)

    if radius is None or radius <= 0:
        diag = float(np.linalg.norm(mesh.bounding_box.extents))
        radius = max(diag * 0.001, 1e-6)

    if verbose:
        print(f"  Drawing {len(filtered_loops)} hole boundary loops "
              f"(filtered from {len(loops)} total):")

    for loop_idx, loop in enumerate(filtered_loops):
        color = palette[color_indices[loop_idx]]
        if verbose:
            print(f"    hole {loop_idx}: {len(loop)} edges, color={color.tolist()}")

        for k in range(len(loop)):
            v0 = mesh.vertices[int(loop[k])]
            v1 = mesh.vertices[int(loop[(k + 1) % len(loop)])]
            seg = trimesh.creation.cylinder(
                radius=radius,
                segment=[v0, v1],
                sections=4,
            )
            seg.visual.face_colors = color
            scene.add_geometry(seg)


def print_separator(title=None):
    if title:
        print(f"\n{'=' * 60}")
        print(f" {title}")
        print(f"{'=' * 60}")
    else:
        print(f"\n{'=' * 60}")


def inspect_mesh(mesh, args):
    """
    主检查函数：输出统计信息并可选返回可视化场景。
    """
    stats = compute_mesh_stats(mesh)
    defect_stats, open_face_mask, nonmanifold_face_mask = analyze_mesh_defects(mesh)
    area_stats = compute_face_area_stats(mesh)
    bbox_stats = compute_bounding_box_stats(mesh)
    volume = compute_volume_if_closed(mesh)

    print_separator("Mesh Topology")
    print(f"  vertices:          {stats['vertices']}")
    print(f"  faces:             {stats['faces']}")
    print(f"  edges (unique):    {stats['edges']}")
    print(f"  watertight:        {stats['is_watertight']}")
    print(f"  open edges:        {defect_stats['open_edges']}")
    print(f"  nonmanifold edges: {defect_stats['nonmanifold_edges']}")
    print(f"  open faces:        {defect_stats['open_faces']}")
    print(f"  nonmanifold faces: {defect_stats['nonmanifold_faces']}")

    hole_stats = compute_hole_area_stats(mesh)
    print(f"  open boundary loops: {hole_stats['count']}")
    print(f"  total hole area:     {hole_stats['total_area']:.6f}")
    if hole_stats['count'] > 0:
        print(f"  hole area percentiles: "
              f"p1={hole_stats['p1_area']:.6f}, "
              f"p5={hole_stats['p5_area']:.6f}, "
              f"p25={hole_stats['p25_area']:.6f}, "
              f"p50={hole_stats['p50_area']:.6f}, "
              f"p75={hole_stats['p75_area']:.6f}, "
              f"p90={hole_stats['p90_area']:.6f}, "
              f"p95={hole_stats['p95_area']:.6f}, "
              f"p99={hole_stats['p99_area']:.6f}, "
              f"max={hole_stats['max_area']:.6f}")

    print_separator("Bounding Box")
    print(f"  min:      [{bbox_stats['min'][0]:.4f}, "
          f"{bbox_stats['min'][1]:.4f}, {bbox_stats['min'][2]:.4f}]")
    print(f"  max:      [{bbox_stats['max'][0]:.4f}, "
          f"{bbox_stats['max'][1]:.4f}, {bbox_stats['max'][2]:.4f}]")
    print(f"  extents:  [{bbox_stats['extents'][0]:.4f}, "
          f"{bbox_stats['extents'][1]:.4f}, {bbox_stats['extents'][2]:.4f}]")
    print(f"  diagonal: {bbox_stats['diagonal']:.4f}")
    print(f"  centroid: [{bbox_stats['centroid'][0]:.4f}, "
          f"{bbox_stats['centroid'][1]:.4f}, {bbox_stats['centroid'][2]:.4f}]")

    print_separator("Edge Length Statistics")
    print(f"  mean: {stats['mean_edge_length']:.6f}")
    print(f"  p1:   {stats['edge_length_p1']:.6f}")
    print(f"  p5:   {stats['edge_length_p5']:.6f}")
    print(f"  p50:  {stats['edge_length_p50']:.6f}")
    print(f"  p95:  {stats['edge_length_p95']:.6f}")
    print(f"  p99:  {stats['edge_length_p99']:.6f}")

    print_separator("Face Area Statistics")
    print(f"  count: {area_stats['count']}")
    print(f"  mean:  {area_stats['mean']:.6f}")
    print(f"  min:   {area_stats['min']:.6f}")
    print(f"  p1:    {area_stats['p1']:.6f}")
    print(f"  p5:    {area_stats['p5']:.6f}")
    print(f"  p10:   {area_stats['p10']:.6f}")
    print(f"  p25:   {area_stats['p25']:.6f}")
    print(f"  p50:   {area_stats['p50']:.6f}")
    print(f"  p75:   {area_stats['p75']:.6f}")
    print(f"  p90:   {area_stats['p90']:.6f}")
    print(f"  p95:   {area_stats['p95']:.6f}")
    print(f"  p99:   {area_stats['p99']:.6f}")
    print(f"  max:   {area_stats['max']:.6f}")

    print_separator("Surface & Volume")
    print(f"  total surface area: {mesh.area:.6f}")
    if np.isfinite(volume):
        print(f"  volume (watertight): {volume:.6f}")
    else:
        print(f"  volume: N/A (mesh is not watertight)")

    # 可视化
    scene = None
    if args.show or args.output:
        print_separator("Defect Visualization")
        print("  gray:    normal faces")
        print("  yellow:  faces adjacent to open edges")
        print("  red:     faces adjacent to non-manifold edges")
        print("  orange:  faces with both defects")

        vis = build_defect_visualization(mesh, open_face_mask, nonmanifold_face_mask)

        # 默认双面渲染：薄壳从任何一侧观察都可见
        if args.double_sided:
            vis = make_double_sided(vis, args.backface_color)

        scene = trimesh.Scene(vis)

        # 先导出着色的缺陷网格（不含线框）
        if args.output:
            vis.export(args.output)
            print(f"\nColored defect mesh saved to: {args.output}")

        # 再叠加线框到场景用于可视化
        if args.wireframe:
            add_wireframe_to_scene(
                scene, vis,
                color=args.wireframe_color,
                radius=args.wireframe_radius
            )

        # 高亮显示闭合孔洞边界
        if args.highlight_holes:
            add_hole_boundaries_to_scene(
                scene, mesh,
                radius=args.hole_radius,
                min_edges=args.min_hole_edges,
                min_area=args.min_hole_area,
                verbose=True,
            )

    return scene


def _parse_color_string(s):
    """
    将 'R,G,B,A' 字符串解析为整数列表。
    """
    parts = s.split(',')
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            f"Color must be 'R,G,B,A', got '{s}'"
        )
    try:
        return [int(p.strip()) for p in parts]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Color components must be integers, got '{s}'"
        )


def main():
    parser = argparse.ArgumentParser(
        description="检查并可视化网格模型，输出拓扑、边长、面积等统计信息。"
    )
    parser.add_argument("input_file", help="输入网格文件路径 (stl/ply/obj)")
    parser.add_argument("-o", "--output",
                        help="输出带缺陷着色的网格文件路径 (可选)")
    parser.add_argument("--show", action="store_true",
                        help="显示可视化窗口")
    parser.add_argument("--wireframe", action="store_true",
                        help="在可视化中叠加黑色线框，观察三角剖分")
    parser.add_argument("--wireframe-radius", type=float, default=None,
                        help="线框圆柱半径（默认按包围盒自动计算）")
    parser.add_argument("--wireframe-color", type=str, default="0,0,0,255",
                        help="线框 RGBA 颜色，默认 '0,0,0,255'")
    parser.add_argument("--highlight-holes", action="store_true",
                        help="高亮显示闭合孔洞边界，相邻孔洞使用不同高饱和度颜色")
    parser.add_argument("--hole-radius", type=float, default=None,
                        help="孔洞边界圆柱半径（默认自动计算，通常比普通线框略粗）")
    parser.add_argument("--double-sided", dest='double_sided',
                        action='store_true', default=True,
                        help="双面渲染薄壳网格（默认开启）")
    parser.add_argument("--no-double-sided", dest='double_sided',
                        action='store_false',
                        help="关闭双面渲染，恢复默认背面剔除")
    parser.add_argument("--backface-color", type=str, default=None,
                        help="双面渲染时背面子颜色，格式 'R,G,B,A'。"
                             "默认自动根据正面颜色生成同色系暗色")
    parser.add_argument("--min-hole-edges", type=int, default=3,
                        help="高亮孔洞的最小边界边数（默认 3）")
    parser.add_argument("--min-hole-area", type=float, default=0.0,
                        help="高亮孔洞的最小面积（默认 0，不过滤）")
    args = parser.parse_args()

    args.wireframe_color = _parse_color_string(args.wireframe_color)

    if args.backface_color is not None:
        args.backface_color = _parse_color_string(args.backface_color)

    print(f"Loading: {args.input_file}")
    mesh = trimesh.load(args.input_file, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump(concatenate=True)
        print("Multiple meshes detected, merged.")

    scene = inspect_mesh(mesh, args)

    if args.show and scene is not None:
        try:
            os.environ['TRIMESH_DEFAULT_VIEWER'] = 'vedo'
            scene.show()
        except Exception as e:
            print(f"\n[ERROR] Visualization failed: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
