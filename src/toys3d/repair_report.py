# src/toys3d/repair_report.py
"""
汇总批量修补结果，输出成功/失败统计与失败孔洞特征分析。
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path
from collections import Counter

import numpy as np

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_src_parent = os.path.dirname(_project_root)
if _src_parent not in sys.path:
    sys.path.insert(0, _src_parent)


def _find_status_files(output_dir):
    return sorted(Path(output_dir).glob("*.status.json"))


def _load_status(path):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        return {
            "status_json": str(path),
            "success": False,
            "message": f"读取状态文件失败: {e}",
            "failure_stage": "status_read_error",
        }
    data.setdefault("failure_stage", None)
    return data


def _percentile_summary(values, p_list=(5, 50, 75, 95)):
    clean = [v for v in values if isinstance(v, (int, float)) and np.isfinite(v)]
    if not clean:
        return {f"p{p}": None for p in p_list} | {"count": 0}
    arr = np.asarray(clean, dtype=np.float64)
    out = {f"p{p}": float(np.percentile(arr, p)) for p in p_list}
    out["count"] = int(len(arr))
    return out


def _aggregate(records):
    total = len(records)
    success_records = [r for r in records if r.get("success")]
    failed_records = [r for r in records if not r.get("success")]
    success = len(success_records)
    failed = len(failed_records)

    failure_stages = Counter(
        (r.get("failure_stage") or "unknown") for r in failed_records
    )

    new_face_counts = []
    patch_areas = []
    gaussian_means = []
    for r in success_records:
        patch = r.get("patch") or {}
        new_face_counts.append(patch.get("new_face_count"))
        patch_areas.append(patch.get("patch_area"))
        curv = patch.get("curvature") or {}
        gaussian_means.append(curv.get("gaussian_mean"))

    feature_keys = [
        "boundary_vertex_count",
        "planarity_ratio",
        "turn_angle_max_deg",
        "turn_angle_mean_deg",
        "area_ratio_3d_over_2d",
        "edge_length_ratio",
        "perimeter",
        "area_3d",
    ]

    def _feature_summary(records_subset, key):
        values = []
        for r in records_subset:
            f = r.get("input_features") or {}
            v = f.get(key)
            if isinstance(v, (int, float)) and np.isfinite(v):
                values.append(float(v))
        return _percentile_summary(values)

    return {
        "total": total,
        "success": success,
        "failed": failed,
        "success_ratio": success / total if total else 0.0,
        "failure_stages": dict(failure_stages),
        "success_patch_stats": {
            "new_face_count": _percentile_summary(new_face_counts),
            "patch_area": _percentile_summary(patch_areas),
            "gaussian_mean": _percentile_summary(gaussian_means),
        },
        "success_input_features": {
            k: _feature_summary(success_records, k) for k in feature_keys
        },
        "failure_input_features": {
            k: _feature_summary(failed_records, k) for k in feature_keys
        },
    }


def _format_percentiles(qs):
    parts = []
    for k, v in qs.items():
        if isinstance(v, float):
            parts.append(f"{k}={v:.4f}")
        else:
            parts.append(f"{k}={v}")
    return "  ".join(parts)


def _print_summary(summary):
    print("\n汇总")
    print(f"  总组件数:    {summary['total']}")
    print(f"  成功修补:    {summary['success']} ({summary['success_ratio']*100:.1f}%)")
    print(f"  失败:        {summary['failed']}")
    if summary["failure_stages"]:
        print("  失败阶段分布:")
        for stage, cnt in sorted(
            summary["failure_stages"].items(), key=lambda x: -x[1]
        ):
            print(f"    {stage}: {cnt}")

    if summary["success"] > 0:
        print("\n成功补丁的特性分位数:")
        for key, qs in summary["success_patch_stats"].items():
            print(f"  {key:18s} {_format_percentiles(qs)}")

    if summary["success"] > 0:
        print("\n成功修补的输入特征分位数:")
        for key, qs in summary["success_input_features"].items():
            print(f"  {key:22s} {_format_percentiles(qs)}")

    if summary["failed"] > 0:
        print("\n失败孔洞的输入特征分位数:")
        for key, qs in summary["failure_input_features"].items():
            print(f"  {key:22s} {_format_percentiles(qs)}")


def main():
    parser = argparse.ArgumentParser(
        description="汇总批量修补 status JSON，输出成功/失败统计与失败特征分析。"
    )
    parser.add_argument(
        "--output-dir",
        default="repaired_components",
        help="status JSON 所在目录（默认 repaired_components）",
    )
    parser.add_argument("--summary-json", default=None,
                        help="汇总 JSON 输出路径")
    parser.add_argument("--report-json", default=None,
                        help="完整记录 JSON 输出路径")
    parser.add_argument("--report-csv", default=None,
                        help="CSV 明细输出路径")
    parser.add_argument("--failed-features-json", default=None,
                        help="失败孔洞特征 JSON 输出路径")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        print(f"[ERROR] 目录不存在: {output_dir}")
        sys.exit(1)

    status_files = _find_status_files(output_dir)
    if not status_files:
        print(f"[WARN] 未找到任何 *.status.json 于 {output_dir}")
        sys.exit(0)

    print(f"发现 {len(status_files)} 个状态文件")

    records = []
    for p in status_files:
        rec = _load_status(p)
        rec["status_json_path"] = str(p)
        records.append(rec)

    summary = _aggregate(records)
    _print_summary(summary)

    if args.summary_json:
        Path(args.summary_json).write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\n汇总 JSON: {args.summary_json}")

    if args.report_json:
        Path(args.report_json).write_text(
            json.dumps(records, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"完整记录 JSON: {args.report_json}")

    if args.report_csv:
        csv_fields = [
            "input_stem", "boundary_type", "hole_id",
            "success", "failure_stage", "message",
        ]
        with open(args.report_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(csv_fields)
            for r in records:
                writer.writerow([
                    r.get("input_stem"),
                    r.get("boundary_type"),
                    r.get("hole_id"),
                    r.get("success"),
                    r.get("failure_stage"),
                    r.get("message"),
                ])
        print(f"CSV 明细: {args.report_csv}")

    if args.failed_features_json:
        failed = [r for r in records if not r.get("success")]
        Path(args.failed_features_json).write_text(
            json.dumps(failed, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"失败记录 JSON: {args.failed_features_json}")


if __name__ == "__main__":
    main()
