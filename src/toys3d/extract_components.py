# src/toys3d/extract_components.py
"""
批量提取健康孔洞或未覆盖开放边连通分量。

用法示例：
    python src/toys3d/extract_components.py input.stl \
        --boundary-type healthy \
        --output-dir extracted_components \
        --boundary-neighborhood-depth 2 \
        --ids 3,5,7 \
        --overwrite
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import trimesh

# 确保可以导入 toys3d 包
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_src_parent = os.path.dirname(_project_root)
if _src_parent not in sys.path:
    sys.path.insert(0, _src_parent)


def load_mesh(input_file):
    """加载网格并返回 trimesh.Trimesh（跳过慢速顶点合并）。"""
    mesh = trimesh.load(
        input_file,
        force="mesh",
        process=False,      # 跳过 merge_vertices，大幅加速千万级网格加载
        validate=False,
    )
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    return mesh


def load_healthy_holes(diag_dir):
    """
    从 hole diagnosis 目录加载健康孔洞列表。
    每个孔洞包含：
        hole_id, vertex_indices, num_edges, area, perimeter, face_ids
    """
    diag_dir = Path(diag_dir)
    diag_json = diag_dir / "hole_diagnosis.json"
    npz_path = diag_dir / "hole_diagnosis_data.npz"
    if not diag_json.exists() or not npz_path.exists():
        raise FileNotFoundError(
            f"缺少诊断文件: {diag_json} 或 {npz_path}"
        )

    with open(diag_json, "r", encoding="utf-8") as f:
        diagnosis = json.load(f)

    npz = np.load(npz_path)
    hole_ids_per_edge = npz["hole_ids_per_edge"]
    open_edge_face_ids = npz["open_edge_face_ids"]

    healthy_holes = []
    for hole in diagnosis.get("healthy_holes", []):
        hole_id = hole["hole_id"]
        # 反推种子面片
        edge_mask = hole_ids_per_edge == hole_id
        face_ids = list(set(open_edge_face_ids[edge_mask].tolist()))
        hole = dict(hole)
        hole["face_ids"] = face_ids
        healthy_holes.append(hole)

    return healthy_holes


def load_uncovered_components(diag_dir):
    """从 hole diagnosis 目录加载未覆盖开放边连通分量。"""
    diag_dir = Path(diag_dir)
    comp_json = diag_dir / "uncovered_component_analysis.json"
    if not comp_json.exists():
        raise FileNotFoundError(f"缺少未覆盖分量分析文件: {comp_json}")

    with open(comp_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    components = data.get("components", [])
    for comp in components:
        # 确保关键字段存在
        comp.setdefault("face_ids", [])
        comp.setdefault("endpoints", [])
        comp.setdefault("branch_vertices", [])
        comp.setdefault("candidate_breaks", [])
        comp.setdefault("edge_vertex_pairs", [])
    return components


def expand_face_neighborhood(mesh, seed_faces, depth):
    """
    从种子面片出发，返回拓扑邻域扩展 depth 层后的面片索引集合。
    depth<=0: 空集合
    depth==1: 种子面片本身
    depth>=2: 依次加入直接邻居等
    """
    if depth <= 0:
        return set()

    seed_faces = set(map(int, seed_faces))
    if depth == 1:
        return seed_faces.copy()

    adjacency = [[] for _ in range(len(mesh.faces))]
    for f0, f1 in mesh.face_adjacency:
        adjacency[int(f0)].append(int(f1))
        adjacency[int(f1)].append(int(f0))

    current = list(seed_faces)
    visited = set(seed_faces)

    for _ in range(depth - 1):
        next_layer = []
        for fid in current:
            for nb in adjacency[fid]:
                if nb not in visited:
                    visited.add(nb)
                    next_layer.append(nb)
        current = next_layer
        if not current:
            break

    return visited


def extract_component_by_id(
    mesh,
    comp_original,
    boundary_type,
    boundary_id,
    neighborhood_depth,
    output_dir,
    overwrite,
):
    """
    提取单个组件，保存 PLY 和 JSON。
    返回 (success: bool, local_face_count: int, local_vertex_count: int, message: str)
    """
    seed_faces = comp_original.get("face_ids", [])
    if not seed_faces:
        return False, 0, 0, "组件没有种子面片"

    # 扩展到指定深度
    expanded = expand_face_neighborhood(mesh, seed_faces, neighborhood_depth)
    if not expanded:
        return False, 0, 0, "扩展后无面片"

    faces_idx = np.array(sorted(expanded), dtype=np.int64)
    original_faces = np.asarray(mesh.faces, dtype=np.int64)[faces_idx]

    # 局部顶点重映射
    unique_old_vertices = np.unique(original_faces.ravel())
    old_to_new = {
        int(old_v): int(new_v)
        for new_v, old_v in enumerate(unique_old_vertices)
    }

    local_vertices = mesh.vertices[unique_old_vertices]
    local_faces = np.array(
        [
            [old_to_new[int(v)] for v in face]
            for face in original_faces
        ],
        dtype=np.int64,
    )

    local_mesh = trimesh.Trimesh(
        vertices=local_vertices,
        faces=local_faces,
        process=False,
    )

    # 重映射组件数据
    comp_new = comp_original.copy()

    # 面片索引映射
    face_idx_to_local = {
        int(old_fid): int(local_fid)
        for local_fid, old_fid in enumerate(faces_idx)
    }
    comp_new["face_ids"] = [
        face_idx_to_local[int(f)]
        for f in comp_original.get("face_ids", [])
        if int(f) in face_idx_to_local
    ]

    def remap_v(v):
        return old_to_new.get(int(v), -1)

    comp_new["vertices"] = [
        remap_v(v)
        for v in comp_original.get("vertices", [])
        if remap_v(v) >= 0
    ]

    comp_new["edge_vertex_pairs"] = [
        [remap_v(v0), remap_v(v1)]
        for v0, v1 in comp_original.get("edge_vertex_pairs", [])
        if remap_v(v0) >= 0 and remap_v(v1) >= 0
    ]

    comp_new["endpoints"] = [
        remap_v(v)
        for v in comp_original.get("endpoints", [])
        if remap_v(v) >= 0
    ]

    comp_new["branch_vertices"] = [
        remap_v(v)
        for v in comp_original.get("branch_vertices", [])
        if remap_v(v) >= 0
    ]

    comp_new["candidate_breaks"] = [
        {
            "v0": remap_v(c.get("v0", -1)),
            "v1": remap_v(c.get("v1", -1)),
            "distance": float(c.get("distance", 0.0)),
        }
        for c in comp_original.get("candidate_breaks", [])
        if remap_v(c.get("v0", -1)) >= 0 and remap_v(c.get("v1", -1)) >= 0
    ]

    if "healthy_hole_vertex_indices" in comp_original:
        comp_new["healthy_hole_vertex_indices"] = [
            remap_v(v)
            for v in comp_original.get("healthy_hole_vertex_indices", [])
            if remap_v(v) >= 0
        ]

    # 输出路径
    stem = f"{boundary_type}_{boundary_id}_depth{neighborhood_depth}"
    ply_path = output_dir / f"{stem}.ply"
    json_path = output_dir / f"{stem}.json"

    if not overwrite and (ply_path.exists() or json_path.exists()):
        return False, 0, 0, f"输出文件已存在（{stem}）"

    local_mesh.export(ply_path)

    package_data = {
        "source_file": str(mesh.metadata.get("file_name", "")),
        "boundary_type": boundary_type,
        "boundary_id": boundary_id,
        "neighborhood_depth": neighborhood_depth,
        "local_vertex_count": int(len(local_vertices)),
        "local_face_count": int(len(local_faces)),
        "component": comp_new,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(package_data, f, indent=2, ensure_ascii=False)

    return True, len(local_faces), len(local_vertices), ""


def main():
    parser = argparse.ArgumentParser(
        description="批量提取健康孔洞或未覆盖开放边连通分量。"
    )
    parser.add_argument("input_file", help="输入网格文件路径")
    parser.add_argument(
        "--boundary-type",
        choices=["healthy", "uncovered"],
        default="uncovered",
        help="组件类型：healthy（健康孔洞）或 uncovered（未覆盖开放边分量）",
    )
    parser.add_argument(
        "--output-dir",
        default="extracted_components",
        help="输出目录（默认 extracted_components）",
    )
    parser.add_argument(
        "--hole-diagnosis-dir",
        default="hole_diagnosis_report",
        help="hole diagnosis 输出目录（默认 hole_diagnosis_report）",
    )
    parser.add_argument(
        "--boundary-neighborhood-depth",
        type=int,
        default=1,
        help="邻域深度（默认 1，即仅组件面片本身）",
    )
    parser.add_argument(
        "--ids",
        type=str,
        default=None,
        help="逗号分隔的组件 ID 列表，例如 3,5,7；不指定则全部",
    )
    parser.add_argument(
        "--max-components",
        type=int,
        default=None,
        help="最多处理多少个组件（默认全部）",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已存在的输出文件",
    )

    args = parser.parse_args()

    # 输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Hey! Loading {args.input_file}")
    mesh = load_mesh(args.input_file)

    # 根据类型加载组件
    if args.boundary_type == "healthy":
        components = load_healthy_holes(args.hole_diagnosis_dir)
        # 为健康孔洞添加 component_id 等字段（hole_id 作为 component_id）
        for hole in components:
            hole["component_id"] = hole["hole_id"]
            hole["edge_vertex_pairs"] = []  # 健康孔洞暂时没有直接边对，后续可补
            hole["endpoints"] = []
            hole["branch_vertices"] = []
            hole["candidate_breaks"] = []
            # 加载网格时未合并顶点，全局顶点索引可能失效，这里不写入旧索引
            hole["vertices"] = []
            hole["healthy_hole_vertex_indices"] = []
    else:
        components = load_uncovered_components(args.hole_diagnosis_dir)
        # 未覆盖组件中已经有 component_id、face_ids 等
        for comp in components:
            comp.setdefault("healthy_hole_vertex_indices", [])

    if not components:
        print("没有找到任何组件。")
        return

    # 按 ID 过滤
    if args.ids is not None:
        id_set = set(int(x) for x in args.ids.split(",") if x.strip())
        components = [
            c for c in components if int(c.get("component_id", -1)) in id_set
        ]

    # 限制数量
    if args.max_components is not None:
        components = components[: args.max_components]

    success_count = 0
    fail_count = 0

    for comp in components:
        boundary_id = int(comp.get("component_id", -1))
        success, face_cnt, vert_cnt, msg = extract_component_by_id(
            mesh,
            comp,
            args.boundary_type,
            boundary_id,
            args.boundary_neighborhood_depth,
            output_dir,
            args.overwrite,
        )
        if success:
            success_count += 1
            print(
                f"[OK] {args.boundary_type}_{boundary_id}: "
                f"局部面片数 {face_cnt}, 局部顶点数 {vert_cnt}"
            )
        else:
            fail_count += 1
            print(f"[FAIL] {args.boundary_type}_{boundary_id}: {msg}")

    print("\n汇总:")
    print(f"成功提取: {success_count} 个组件")
    print(f"失败:     {fail_count} 个组件")
    print(f"输出目录: {output_dir}")


if __name__ == "__main__":
    main()
