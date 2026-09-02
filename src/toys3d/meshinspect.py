# src/toys3d/meshinspect.py
"""
网格检查工具，包含孔洞诊断、未覆盖组件分析、最小包络边界等。
"""

import sys
from pathlib import Path

# 支持直接运行脚本：将 src/ 加入路径
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

import numpy as np

from toys3d import geometrics


# 未覆盖开放边分类中文名
_UNCOVERED_CATEGORY_NAMES = {
    0: "孤立开放链",
    1: "悬空开放边",
    2: "分支内部开放边",
    4: "非流形关联开放边",
    5: "其他未覆盖开放边",
}


def _ensure_output_dir(out_dir):
    Path(out_dir).mkdir(parents=True, exist_ok=True)


def _build_hole_json_report(mesh, data):
    total_open_edges = len(data["open_edge_vertex_pairs"])
    total_healthy_holes = len(data["hole_vertex_lists"])
    uncovered_count = len(data["uncovered_edge_ids"])
    covered_open_edges = total_open_edges - uncovered_count

    healthy_holes = []
    for i, edges in enumerate(data["hole_edge_lists"]):
        healthy_holes.append({
            "hole_id": i,
            "num_edges": len(edges),
            "num_vertices": len(data["hole_vertex_lists"][i]),
        })

    # uncovered_categories_summary
    summary = {}
    cats = data["uncovered_category"]
    for cat_val in np.unique(cats):
        name = _UNCOVERED_CATEGORY_NAMES.get(int(cat_val), f"分类{int(cat_val)}")
        summary[name] = int(np.sum(cats == cat_val))

    report = {
        "total_open_edges": total_open_edges,
        "total_healthy_holes": total_healthy_holes,
        "covered_open_edges": covered_open_edges,
        "uncovered_open_edges": uncovered_count,
        "healthy_holes": healthy_holes,
        "uncovered_categories_summary": summary,
    }
    return report


def _build_hole_html(report):
    lines = [
        "<html><head><meta charset='utf-8'></head><body>",
        "<h1>孔洞诊断报告</h1>",
        f"<p>总开放边: {report['total_open_edges']}</p>",
        f"<p>健康孔洞数: {report['total_healthy_holes']}</p>",
        f"<p>覆盖开放边: {report['covered_open_edges']}</p>",
        f"<p>未覆盖开放边: {report['uncovered_open_edges']}</p>",
        "<h2>未覆盖分类</h2><ul>",
    ]
    for name, cnt in report["uncovered_categories_summary"].items():
        lines.append(f"<li>{name}: {cnt}</li>")
    lines.append("</ul>")
    lines.append("</body></html>")
    return "\n".join(lines)


def _write_hole_diagnosis_outputs(mesh, out_dir):
    data = geometrics.build_hole_diagnosis_data(mesh)

    npz_path = out_dir / "hole_diagnosis_data.npz"
    np.savez(
        npz_path,
        open_edge_vertex_pairs=data["open_edge_vertex_pairs"],
        open_edge_face_ids=data["open_edge_face_ids"],
        open_edge_keys=data["open_edge_keys"],
        hole_ids_per_edge=data["hole_ids_per_edge"],
        uncovered_edge_ids=data["uncovered_edge_ids"],
        uncovered_category=data["uncovered_category"],
    )

    report = _build_hole_json_report(mesh, data)
    json_path = out_dir / "hole_diagnosis.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    html = _build_hole_html(report)
    html_path = out_dir / "hole_report.html"
    html_path.write_text(html, encoding="utf-8")

    return data, report


def _write_uncovered_component_analysis(mesh, data, out_dir):
    components = geometrics.analyze_uncovered_open_edge_components(mesh, data)

    for comp in components:
        result = geometrics.find_minimal_enclosing_manifold_boundary_greedy(
            mesh, comp, max_depth=5
        )
        comp["minimal_enclosing_boundary"] = result

    payload = {"components": components}
    json_path = out_dir / "uncovered_component_analysis.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def perform_hole_diagnosis(mesh, args):
    """执行孔洞诊断，并将结果写入指定目录。

    参数:
        mesh: trimesh.Trimesh
        args: argparse.Namespace, 至少包含 hole_diagnosis_output。
              可选属性 compute_enclosing_boundaries: bool
    """
    out_dir = Path(args.hole_diagnosis_output)
    _ensure_output_dir(out_dir)

    data, _ = _write_hole_diagnosis_outputs(mesh, out_dir)

    if getattr(args, "compute_enclosing_boundaries", False):
        _write_uncovered_component_analysis(mesh, data, out_dir)
