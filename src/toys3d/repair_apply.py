# src/toys3d/repair_apply.py
"""
把成功的 Seifert 补丁应用回原始网格，并可选删除已处理的组件文件。
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import trimesh

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_src_parent = os.path.dirname(_project_root)
if _src_parent not in sys.path:
    sys.path.insert(0, _src_parent)


def _find_success_status(output_dir):
    files = sorted(Path(output_dir).glob("*.status.json"))
    success = []
    for p in files:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if data.get("success"):
            data["_status_path"] = str(p)
            success.append(data)
    return success


def _extract_patch_faces(input_mesh, repaired_mesh):
    before_keys = set()
    for face in input_mesh.faces:
        key = tuple(sorted(int(x) for x in face))
        before_keys.add(key)

    patch_face_indices = []
    for i, face in enumerate(repaired_mesh.faces):
        key = tuple(sorted(int(x) for x in face))
        if key not in before_keys:
            patch_face_indices.append(i)
    return patch_face_indices


def main():
    parser = argparse.ArgumentParser(
        description="把已成功修补的 Seifert 补丁应用回原始网格。"
    )
    parser.add_argument("input_mesh", help="原始输入网格 (stl/ply/obj)")
    parser.add_argument("output_mesh", help="输出应用补丁后的网格")
    parser.add_argument("--output-dir", default="repaired_components",
                        help="status JSON 所在目录（默认 repaired_components）")
    parser.add_argument("--delete-components", action="store_true",
                        help="应用成功后删除组件 PLY/JSON 与修补后 PLY/status JSON")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    success_status = _find_success_status(output_dir)
    if not success_status:
        print("没有找到成功修补的 status.json，直接复制输入网格。")
        mesh = trimesh.load(args.input_mesh, force="mesh")
        mesh.export(args.output_mesh)
        return

    print(f"发现 {len(success_status)} 个成功修补项")

    print(f"Hey! Loading original mesh: {args.input_mesh}")
    original = trimesh.load(args.input_mesh, force="mesh")
    original_vertices = np.asarray(original.vertices, dtype=np.float64)
    original_faces = np.asarray(original.faces, dtype=np.int64)

    new_vertices = []
    new_faces = []
    applied = []
    skipped = []

    for st in success_status:
        status_path = Path(st["_status_path"])
        comp_json_path = st.get("component_package")
        repaired_ply = st.get("output_file")
        input_ply = st.get("input_file")
        stem = st.get("input_stem", status_path.stem)

        if not comp_json_path or not Path(comp_json_path).exists():
            skipped.append((stem, "缺少组件 JSON"))
            continue
        if not repaired_ply or not Path(repaired_ply).exists():
            skipped.append((stem, "缺少修复后 PLY"))
            continue
        if not input_ply or not Path(input_ply).exists():
            skipped.append((stem, "缺少输入组件 PLY"))
            continue

        try:
            comp_data = json.loads(Path(comp_json_path).read_text(encoding="utf-8"))
        except Exception as e:
            skipped.append((stem, f"组件 JSON 读取失败: {e}"))
            continue

        source_vertex_indices = comp_data.get("source_vertex_indices")
        if not source_vertex_indices:
            skipped.append((stem, "组件 JSON 缺少 source_vertex_indices（需重新提取）"))
            continue

        try:
            input_mesh = trimesh.load(input_ply, force="mesh", process=False)
            repaired_mesh = trimesh.load(repaired_ply, force="mesh", process=False)
        except Exception as e:
            skipped.append((stem, f"局部网格读取失败: {e}"))
            continue

        local_M = len(source_vertex_indices)
        local_K = len(repaired_mesh.vertices) - local_M
        if local_K < 0:
            skipped.append((stem, "修复后网格顶点数小于组件 JSON 记录"))
            continue

        patch_face_indices = _extract_patch_faces(input_mesh, repaired_mesh)
        if not patch_face_indices:
            skipped.append((stem, "未检测到新增补丁面片"))
            continue

        base_new_global = len(original_vertices) + len(new_vertices)

        local_to_global = np.empty(len(repaired_mesh.vertices), dtype=np.int64)
        for i in range(local_M):
            local_to_global[i] = int(source_vertex_indices[i])
        for k in range(local_K):
            local_to_global[local_M + k] = base_new_global + k

        if local_K > 0:
            for k in range(local_K):
                new_vertices.append(repaired_mesh.vertices[local_M + k])

        for lf in patch_face_indices:
            tri = repaired_mesh.faces[lf]
            new_faces.append(local_to_global[tri])

        applied.append(stem)

    if not applied:
        print("没有可应用的补丁。")
        original.export(args.output_mesh)
        return

    print(f"\n已应用 {len(applied)} 个补丁，新增顶点 {len(new_vertices)}，新增面片 {len(new_faces)}")

    appended_vertices = (
        np.asarray(new_vertices, dtype=np.float64)
        if new_vertices
        else np.zeros((0, 3), dtype=np.float64)
    )
    appended_faces = (
        np.asarray(new_faces, dtype=np.int64)
        if new_faces
        else np.zeros((0, 3), dtype=np.int64)
    )

    combined_vertices = np.vstack([original_vertices, appended_vertices])
    combined_faces = np.vstack([original_faces, appended_faces])

    result = trimesh.Trimesh(
        vertices=combined_vertices,
        faces=combined_faces,
        process=True,
    )
    # trimesh 没有 remove_duplicate_faces，使用 unique_faces + update_faces
    result.update_faces(result.unique_faces())
    result.remove_unreferenced_vertices()
    result.export(args.output_mesh)
    print(f"\n应用补丁后的网格已保存: {args.output_mesh}")

    if skipped:
        print("\n跳过项:")
        for stem, reason in skipped:
            print(f"  {stem}: {reason}")

    if args.delete_components:
        deleted = 0
        for st in success_status:
            status_path = Path(st["_status_path"])
            comp_json = st.get("component_package")
            repaired_ply = st.get("output_file")
            input_ply = st.get("input_file")

            for p in [status_path, comp_json, repaired_ply, input_ply]:
                if p and Path(p).exists():
                    try:
                        Path(p).unlink()
                        deleted += 1
                    except Exception as e:
                        print(f"  [WARN] 删除失败 {p}: {e}")

            comp_json_path = Path(comp_json) if comp_json else None
            if comp_json_path:
                comp_ply_guess = comp_json_path.with_suffix(".ply")
                if comp_ply_guess.exists():
                    try:
                        comp_ply_guess.unlink()
                        deleted += 1
                    except Exception:
                        pass

        print(f"\n已删除 {deleted} 个文件（组件与修复输出）")

    print(f"\n成功应用: {len(applied)}")
    print(f"跳过:     {len(skipped)}")


if __name__ == "__main__":
    main()
