# src/toys3d/reporting.py
"""
报告生成与诊断结果 I/O 工具。
"""
import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D

from .geometrics import hex_to_code


class DiagnosisBundle:
    """
    诊断目录中的数据缓存，避免批量提取时重复打开文件。
    """

    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self._npz = None
        self._diag_json = None
        self._comp_json = None

    @property
    def npz(self):
        if self._npz is None:
            path = self.data_dir / "hole_diagnosis_data.npz"
            if not path.exists():
                raise FileNotFoundError(f"未找到 {path}")
            self._npz = np.load(path)
        return self._npz

    @property
    def diag_json(self):
        if self._diag_json is None:
            path = self.data_dir / "hole_diagnosis.json"
            if not path.exists():
                raise FileNotFoundError(f"未找到 {path}")
            with open(path, "r") as f:
                self._diag_json = json.load(f)
        return self._diag_json

    @property
    def comp_json(self):
        if self._comp_json is None:
            path = self.data_dir / "uncovered_component_analysis.json"
            if not path.exists():
                raise FileNotFoundError(f"未找到 {path}")
            with open(path, "r") as f:
                self._comp_json = json.load(f)
        return self._comp_json


def load_uncovered_edge_data(data_dir):
    """
    从 hole diagnosis 输出目录加载未覆盖开放边数据。
    """
    data_dir = Path(data_dir)
    npz_path = data_dir / "hole_diagnosis_data.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"未找到 {npz_path}")

    npz = np.load(npz_path)
    uncovered_ids = npz["uncovered_edge_ids"]
    all_vertex_pairs = npz["open_edge_vertex_pairs"]
    categories = npz["uncovered_category"]
    return uncovered_ids, all_vertex_pairs, categories


def load_boundary_component_data(data_dir, boundary_id, boundary_type="uncovered",
                                  bundle=None):
    """
    从 hole diagnosis 输出目录加载指定边界组件或健康孔洞的数据。

    bundle 可选：传入 DiagnosisBundle 实例可复用已加载的 NPZ 与 JSON，
    避免批量提取时反复读取文件。
    """
    if bundle is None:
        bundle = DiagnosisBundle(data_dir)

    if boundary_type == "uncovered":
        comp_data = bundle.comp_json
        components = comp_data.get("components", [])
        if boundary_id < 0 or boundary_id >= len(components):
            raise ValueError(
                f"无效的未覆盖分量 ID: {boundary_id}，共 {len(components)} 个分量"
            )
        comp = components[boundary_id]
        comp.setdefault("endpoints", [])
        comp.setdefault("branch_vertices", [])
        comp.setdefault("candidate_breaks", [])
        return comp

    elif boundary_type == "healthy":
        npz = bundle.npz
        diag_json = bundle.diag_json

        healthy_holes = diag_json.get("healthy_holes", [])
        if boundary_id < 0 or boundary_id >= len(healthy_holes):
            raise ValueError(
                f"无效的健康孔洞 ID: {boundary_id}，共 {len(healthy_holes)} 个孔洞"
            )

        hole_vertex_list = healthy_holes[boundary_id]["vertex_indices"]
        hole_ids_per_edge = npz["hole_ids_per_edge"]
        open_edge_vertex_pairs = npz["open_edge_vertex_pairs"]
        open_edge_face_ids = npz["open_edge_face_ids"]

        edge_mask = hole_ids_per_edge == boundary_id
        edge_indices = np.where(edge_mask)[0]
        comp_edges = open_edge_vertex_pairs[edge_indices]
        comp_face_ids = np.unique(open_edge_face_ids[edge_indices])

        vertices_set = set()
        for v0, v1 in comp_edges:
            vertices_set.add(int(v0))
            vertices_set.add(int(v1))

        component = {
            "component_id": boundary_id,
            "num_edges": int(len(comp_edges)),
            "num_vertices": int(len(vertices_set)),
            "vertices": sorted(vertices_set),
            "edge_vertex_pairs": comp_edges.tolist(),
            "endpoints": [],
            "branch_vertices": [],
            "is_cycle": True,
            "face_ids": comp_face_ids.tolist(),
            "open_face_count": int(len(comp_face_ids)),
            "nonmanifold_face_count": 0,
            "candidate_breaks": [],
            "healthy_hole_vertex_indices": hole_vertex_list,
        }
        return component

    else:
        raise ValueError(f"未知的边界类型: {boundary_type}")


def _generate_topology_diagram(code, output_path):
    """
    根据拓扑编码生成三角形的点-线示意图。
    """
    if isinstance(code, bytes):
        fields = list(code)
    elif hasattr(code, 'tolist'):
        fields = [int(x) for x in code.tolist()]
    else:
        fields = [int(x) for x in code]

    if len(fields) != 6:
        raise ValueError("code must have exactly 6 fields")

    vA, eAB, vB, eBC, vC, eCA = fields

    fig, ax = plt.subplots(figsize=(2.0, 2.0), dpi=120)
    pts = {
        'A': (0, 0),
        'B': (1, 0),
        'C': (0.5, np.sqrt(3) / 2)
    }

    edge_styles = {
        1: ('blue', 'solid'),
        2: ('green', 'solid'),
        3: ('red', 'solid')
    }
    for (p1, p2, ecode) in [
        (pts['A'], pts['B'], eAB),
        (pts['B'], pts['C'], eBC),
        (pts['C'], pts['A'], eCA)
    ]:
        color, ls = edge_styles.get(ecode, ('black', 'dashed'))
        line = Line2D([p1[0], p2[0]], [p1[1], p2[1]],
                      color=color, linewidth=2, linestyle=ls)
        ax.add_line(line)

    for (pt, vcode) in [(pts['A'], vA), (pts['B'], vB), (pts['C'], vC)]:
        fill = (vcode == 1)
        circle = Circle(pt, radius=0.05, fill=fill,
                        color='black', linewidth=2)
        ax.add_patch(circle)

    height = np.sqrt(3) / 2
    margin_x = 0.2
    margin_y = 0.15
    ax.set_xlim(0 - margin_x, 1 + margin_x)
    ax.set_ylim(0 - margin_y, height + margin_y)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(output_path, format='svg', bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)


def _generate_component_3d_diagram(component, mesh, output_path):
    """
    为单个未覆盖开放边连通分量生成三维 SVG 图。
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    vertex_pairs = component.get('edge_vertex_pairs', [])
    if not vertex_pairs:
        return

    vertices = mesh.vertices
    involved_vertices = list(set(sum(vertex_pairs, [])))

    v_coords = vertices[involved_vertices]
    vmin = v_coords.min(axis=0)
    vmax = v_coords.max(axis=0)
    center = (vmin + vmax) / 2.0
    max_extent = (vmax - vmin).max()
    extra = max_extent * 0.1 + 1e-12

    fig = plt.figure(figsize=(3.0, 3.0), dpi=120)
    ax = fig.add_subplot(111, projection='3d')

    for v0, v1 in vertex_pairs:
        p0 = vertices[v0]
        p1 = vertices[v1]
        ax.plot(
            [p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
            color='blue', linewidth=0.8, alpha=0.7
        )

    endpoints = component.get('endpoints', [])
    if endpoints:
        ep = vertices[endpoints]
        ax.scatter(ep[:, 0], ep[:, 1], ep[:, 2],
                   c='green', marker='o', s=20, label='Endpoints')

    branch_vertices = component.get('branch_vertices', [])
    if branch_vertices:
        bv = vertices[branch_vertices]
        ax.scatter(bv[:, 0], bv[:, 1], bv[:, 2],
                   c='red', marker='s', s=30, label='Branch vertices')

    candidate_breaks = component.get('candidate_breaks', [])
    for cand in candidate_breaks:
        p0 = vertices[cand['v0']]
        p1 = vertices[cand['v1']]
        ax.plot(
            [p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
            '--', color='orange', linewidth=0.8, alpha=0.9
        )

    ax.set_xlim([center[0] - max_extent/2 - extra, center[0] + max_extent/2 + extra])
    ax.set_ylim([center[1] - max_extent/2 - extra, center[1] + max_extent/2 + extra])
    ax.set_zlim([center[2] - max_extent/2 - extra, center[2] + max_extent/2 + extra])

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_axis_off()

    if endpoints or branch_vertices:
        ax.legend(loc='upper right', fontsize=6)

    plt.tight_layout(pad=0)
    plt.savefig(output_path, format='svg', bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)


def generate_html_report(output_dir, id_to_code, results):
    """
    按 classes 生成 HTML 报告。
    """
    output_dir = Path(output_dir)
    html_path = output_dir / "report.html"

    html = [
        "<html><head><meta charset='utf-8'><title>Full Face Diagnosis</title>",
        "<style>",
        "body { font-family: sans-serif; margin: 20px; }",
        "table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }",
        "th, td { border: 1px solid #ccc; padding: 4px 8px; text-align: center; }",
        "th { background: #f0f0f0; }",
        ".class-block { margin-bottom: 30px; border: 1px solid #ddd; padding: 10px; }",
        "img { max-width: 300px; height: auto; }",
        ".diagram-container { display: inline-block; vertical-align: top; margin-right: 20px; }",
        ".diagram-container svg { width: 200px; height: 170px; }",
        ".legend { margin-bottom: 20px; padding: 10px; border: 1px solid #ccc; background: #fafafa; }",
        ".legend span.legend-dot { display: inline-block; width: 15px; height: 15px; border-radius: 50%; margin-right: 5px; }",
        ".legend span.solid { background: black; border: 1px solid black; }",
        ".legend span.hollow { background: white; border: 2px solid black; }",
        ".legend span.legend-line { display: inline-block; width: 30px; height: 0; border-top: 3px solid; margin-right: 5px; vertical-align: middle; }",
        ".legend span.blue { border-color: blue; }",
        ".legend span.green { border-color: green; }",
        ".legend span.red { border-color: red; }",
        "</style></head><body>",
        "<h1>Full Face Diagnosis Report</h1>"
    ]

    html.append("<div class='legend'>")
    html.append("<strong>图例：</strong><br>")
    html.append("<span class='legend-dot solid'></span> 独占顶点（仅被当前面引用）<br>")
    html.append("<span class='legend-dot hollow'></span> 共享顶点（被多个面引用）<br>")
    html.append("<span class='legend-line blue'></span> 开放边（仅属于当前面）<br>")
    html.append("<span class='legend-line green'></span> 流形边（被两面共享）<br>")
    html.append("<span class='legend-line red'></span> 非流形边（被三面或更多共享）<br>")
    html.append("</div>")

    html.append("<h2>Topology Classes</h2>")

    for class_id in sorted(results.keys()):
        res = results[class_id]
        code = res['code']
        area = res['area_stats']
        pc = res['point_counts']
        ec = res['edge_counts']
        v_stats = res.get('representative_vertex_stats')
        if not v_stats:
            v_stats = [{'normal': 0, 'open': 0, 'nonmanifold': 0} for _ in range(3)]
        e_stats = res.get('representative_edge_stats')
        if not e_stats:
            e_stats = [{'normal': 0, 'open': 0, 'nonmanifold': 0} for _ in range(3)]

        diagram_path = output_dir / "diagrams" / f"diagram_{class_id}.svg"
        if diagram_path.exists():
            svg_content = diagram_path.read_text(encoding="utf-8")
        else:
            svg_content = "<svg xmlns='http://www.w3.org/2000/svg' width='200' height='170'><text x='10' y='80'>Diagram not generated</text></svg>"

        html.append("<div class='class-block'>")
        html.append(f"<h3>Class {class_id}: {code}</h3>")
        html.append("<div class='diagram-container'>")
        html.append(svg_content)
        html.append("</div>")
        html.append("<div style='display: inline-block; vertical-align: top;'>")
        html.append("<table>")
        html.append("<tr><th>面积统计</th><th>值</th></tr>")
        rows = [
            ("count", "面片数"),
            ("mean", "平均"),
            ("min", "最小值"),
            ("p1", "p1"),
            ("p5", "p5"),
            ("p10", "p10"),
            ("p25", "p25"),
            ("p50", "p50"),
            ("p75", "p75"),
            ("p90", "p90"),
            ("p95", "p95"),
            ("p99", "p99"),
            ("max", "最大值"),
        ]
        for key, label in rows:
            html.append(f"<tr><td>{label}</td><td>{area[key]:.6f}</td></tr>")
        html.append("</table>")

        html.append("<table>")
        html.append("<tr><th></th><th>流形</th><th>开放</th><th>非流形</th></tr>")
        html.append("<tr><th>点邻（合计）</th>"
                    f"<td>{pc['normal']}</td><td>{pc['open']}</td><td>{pc['nonmanifold']}</td></tr>")
        html.append("<tr><th>边邻（合计）</th>"
                    f"<td>{ec['normal']}</td><td>{ec['open']}</td><td>{ec['nonmanifold']}</td></tr>")
        html.append("</table>")

        html.append("</div>")
        html.append("</div>")

    html.append("</body></html>")

    with open(html_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html))
    print(f"HTML 报告已保存: {html_path}")


def generate_latex_report(output_dir, id_to_code, results):
    """
    生成 LaTeX 报告。
    """
    output_dir = Path(output_dir)
    tex_path = output_dir / "report.tex"

    tex = [
        "\\documentclass{article}",
        "\\usepackage{graphicx}",
        "\\usepackage{booktabs}",
        "\\begin{document}",
        "\\section{Full Face Diagnosis Report}"
    ]

    for class_id in sorted(results.keys()):
        res = results[class_id]
        code = res['code']
        area = res['area_stats']
        pc = res['point_counts']
        ec = res['edge_counts']
        diagram_rel = f"diagrams/diagram_{class_id}.svg"

        tex.append(f"\\subsection{{Class {class_id}: {code}}}")
        tex.append("\\begin{figure}[h]")
        tex.append(f"\\includegraphics[width=0.25\\textwidth]{{{diagram_rel}}}")
        tex.append("\\end{figure}")

        tex.append("\\begin{tabular}{l r}")
        tex.append("\\toprule")
        tex.append("Area Metric & Value \\\\")
        tex.append("\\midrule")
        rows = [
            ("Count", area['count']),
            ("Mean", area['mean']),
            ("Min", area['min']),
            ("p1", area['p1']),
            ("p5", area['p5']),
            ("p10", area['p10']),
            ("p25", area['p25']),
            ("p50", area['p50']),
            ("p75", area['p75']),
            ("p90", area['p90']),
            ("p95", area['p95']),
            ("p99", area['p99']),
            ("Max", area['max']),
        ]
        for label, val in rows:
            tex.append(f"{label} & {val:.6f} \\\\")
        tex.append("\\bottomrule")
        tex.append("\\end{tabular}")

        tex.append("\\begin{tabular}{l c c c}")
        tex.append("\\toprule")
        tex.append("Neighbor & Manifold & Open & Nonmanifold \\\\")
        tex.append("\\midrule")
        tex.append(f"Point & {pc['normal']} & {pc['open']} & {pc['nonmanifold']} \\\\")
        tex.append(f"Edge & {ec['normal']} & {ec['open']} & {ec['nonmanifold']} \\\\")
        tex.append("\\bottomrule")
        tex.append("\\end{tabular}")

    tex.append("\\end{document}")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(tex))
    print(f"LaTeX 报告已保存: {tex_path}")


def generate_html_report_from_json(output_dir):
    """
    从 abnormal_truncated_classes.json 生成 HTML 报告。
    """
    output_dir = Path(output_dir)
    html_path = output_dir / "report.html"
    classes_json_path = output_dir / "abnormal_truncated_classes.json"
    diagrams_dir = output_dir / "diagrams"
    diagrams_dir.mkdir(exist_ok=True)

    if not classes_json_path.exists():
        print(f"[ERROR] {classes_json_path} not found.")
        return

    with open(classes_json_path, "r") as f:
        data = json.load(f)

    html = [
        "<html><head><meta charset='utf-8'><title>Full Face Diagnosis</title>",
        "<style>",
        "body { font-family: sans-serif; margin: 20px; }",
        "table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }",
        "th, td { border: 1px solid #ccc; padding: 4px 8px; text-align: center; }",
        "th { background: #f0f0f0; }",
        ".class-block { margin-bottom: 30px; border: 1px solid #ddd; padding: 10px; }",
        "img { max-width: 300px; height: auto; }",
        ".diagram-container { display: inline-block; vertical-align: top; margin-right: 20px; }",
        ".diagram-container svg { width: 200px; height: 170px; }",
        ".legend { margin-bottom: 20px; padding: 10px; border: 1px solid #ccc; background: #fafafa; }",
        ".legend span.legend-dot { display: inline-block; width: 15px; height: 15px; border-radius: 50%; margin-right: 5px; }",
        ".legend span.solid { background: black; border: 1px solid black; }",
        ".legend span.hollow { background: white; border: 2px solid black; }",
        ".legend span.legend-line { display: inline-block; width: 30px; height: 0; border-top: 3px solid; margin-right: 5px; vertical-align: middle; }",
        ".legend span.blue { border-color: blue; }",
        ".legend span.green { border-color: green; }",
        ".legend span.red { border-color: red; }",
        "</style></head><body>",
        "<h1>Full Face Diagnosis Report</h1>"
    ]

    html.append("<div class='legend'>")
    html.append("<strong>图例：</strong><br>")
    html.append("<span class='legend-dot solid'></span> 独占顶点（仅被当前面引用）<br>")
    html.append("<span class='legend-dot hollow'></span> 共享顶点（被多个面引用）<br>")
    html.append("<span class='legend-line blue'></span> 开放边（仅属于当前面）<br>")
    html.append("<span class='legend-line green'></span> 流形边（被两面共享）<br>")
    html.append("<span class='legend-line red'></span> 非流形边（被三面或更多共享）<br>")
    html.append("</div>")

    html.append("<h2>Topology Classes</h2>")

    classes = data.get("classes", {})
    if not classes:
        html.append("<p>No abnormal classes found.</p>")
    else:
        for hex_code, cls_data in classes.items():
            diagram_path = diagrams_dir / f"diagram_{hex_code}.svg"
            try:
                code_arr = hex_to_code(hex_code)
                _generate_topology_diagram(code_arr, str(diagram_path))
            except Exception as e:
                print(f"  [WARN] Diagram generation for {hex_code} failed: {e}")
                diagram_path = None

            svg_content = ""
            if diagram_path and diagram_path.exists():
                svg_content = diagram_path.read_text(encoding="utf-8")
            if not svg_content:
                svg_content = "<svg xmlns='http://www.w3.org/2000/svg' width='200' height='170'><text x='10' y='80'>Diagram not generated</text></svg>"

            html.append("<div class='class-block'>")
            html.append(f"<h3>Class {hex_code}</h3>")
            html.append("<div class='diagram-container'>")
            html.append(svg_content)
            html.append("</div>")
            html.append("<div style='display: inline-block; vertical-align: top;'>")

            area = cls_data.get("area_stats", {})
            if area:
                html.append("<table>")
                html.append("<tr><th>面积统计</th><th>值</th></tr>")
                rows = [
                    ("count", "面片数"),
                    ("mean", "平均"),
                    ("min", "最小值"),
                    ("p1", "p1"),
                    ("p5", "p5"),
                    ("p10", "p10"),
                    ("p25", "p25"),
                    ("p50", "p50"),
                    ("p75", "p75"),
                    ("p90", "p90"),
                    ("p95", "p95"),
                    ("p99", "p99"),
                    ("max", "最大值"),
                ]
                for key, label in rows:
                    if key in area:
                        html.append(f"<tr><td>{label}</td><td>{area[key]:.6f}</td></tr>")
                    else:
                        html.append(f"<tr><td>{label}</td><td>N/A</td></tr>")
                html.append("</table>")

            pc = cls_data.get("point_counts", {})
            ec = cls_data.get("edge_counts", {})
            if pc and ec:
                html.append("<table>")
                html.append("<tr><th></th><th>流形</th><th>开放</th><th>非流形</th></tr>")
                html.append("<tr><th>点邻（合计）</th>"
                            f"<td>{pc.get('normal', 0)}</td><td>{pc.get('open', 0)}</td><td>{pc.get('nonmanifold', 0)}</td></tr>")
                html.append("<tr><th>边邻（合计）</th>"
                            f"<td>{ec.get('normal', 0)}</td><td>{ec.get('open', 0)}</td><td>{ec.get('nonmanifold', 0)}</td></tr>")
                html.append("</table>")

            v_stats = cls_data.get("representative_vertex_stats")
            e_stats = cls_data.get("representative_edge_stats")
            if v_stats:
                html.append("<table>")
                html.append("<tr><th>代表面-顶点</th><th>流形</th><th>开放</th><th>非流形</th></tr>")
                for idx, vs in enumerate(v_stats):
                    html.append(f"<tr><td>V{idx}</td>"
                                f"<td>{vs.get('normal', 0)}</td><td>{vs.get('open', 0)}</td><td>{vs.get('nonmanifold', 0)}</td></tr>")
                html.append("</table>")
            if e_stats:
                html.append("<table>")
                html.append("<tr><th>代表面-边</th><th>流形</th><th>开放</th><th>非流形</th></tr>")
                for idx, es in enumerate(e_stats):
                    html.append(f"<tr><td>E{idx}</td>"
                                f"<td>{es.get('normal', 0)}</td><td>{es.get('open', 0)}</td><td>{es.get('nonmanifold', 0)}</td></tr>")
                html.append("</table>")

            vertex_dist = cls_data.get("truncated_vertex_valence_dist", {})
            edge_dist = cls_data.get("truncated_edge_valence_dist", {})

            if vertex_dist:
                html.append("<table>")
                html.append("<tr><th>顶点元截断分布</th><th>真实 valence</th><th>计数</th></tr>")
                for val_str, cnt in vertex_dist.items():
                    html.append(f"<tr><td>valence</td><td>{val_str}</td><td>{cnt}</td></tr>")
                html.append("</table>")

            if edge_dist:
                html.append("<table>")
                html.append("<tr><th>边元截断分布</th><th>真实共享数</th><th>计数</th></tr>")
                for val_str, cnt in edge_dist.items():
                    html.append(f"<tr><td>edge</td><td>{val_str}</td><td>{cnt}</td></tr>")
                html.append("</table>")

            html.append("</div>")
            html.append("</div>")

    html.append("</body></html>")
    html_path.write_text("\n".join(html), encoding="utf-8")
    print(f"HTML 报告已保存: {html_path}")
