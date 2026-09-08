#!/usr/bin/env python3
"""
从 git 历史中查找 geometrics.py 内给定函数最后一次存在的 commit。

用法示例：
    python retrieve_functions.py plucker_design_matrix
    python retrieve_functions.py plucker_design_matrix axis_from_plucker
    python retrieve_functions.py --file src/toys3d/geometrics.py repair_nonmanifold_edges
"""

import argparse
import ast
import subprocess
from pathlib import Path


DEFAULT_FILE = "src/toys3d/geometrics.py"

# 当前 handlebar.py / brick.py / shell.py 缺失但历史上曾存在过的函数。
# 不传参数时，默认查找这些函数。
DEFAULT_SYMBOLS = [
    # handlebar
    "plucker_design_matrix",
    "axis_from_plucker",
    "orthogonalize_axes",
    "segment_tubular_regions",
    "line_line_distance_and_midpoint",
    "point_line_distance",
    "intersect_line_plane",
    "kmeans_1d",
    "average_antiparallel_directions",
    "estimate_symmetry_plane_voxel",
    "repair_nonmanifold_edges",
    "fill_small_holes",

    # brick
    "build_box_aligned_frame_voxel",
    "build_box_aligned_frame_mesh",

    # shell
    "build_face_adjacency",
    "detect_multiscale_edges",
    "segment_regions_by_edges",
    "estimate_shell_thickness",
    "segment_plates_by_smoothness",
    "detect_thin_regions",
    "compute_wall_thickness_statistics",
    "extract_plate_boundary_loops",
    "classify_edge_regularity",
    "build_proxy_mesh",
    "map_labels_from_proxy",
    "segment_plates_by_local_clustering",
    "segment_plates_by_plane_fitting",
]


def repo_root():
    """返回 git 仓库根目录。"""
    out = subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"],
        text=True,
        stderr=subprocess.DEVNULL,
    )
    return Path(out.strip())


def run_git(args):
    """在仓库根目录执行 git 命令，返回 stdout 字符串。"""
    return subprocess.check_output(
        ["git", *args],
        cwd=repo_root(),
        text=True,
        stderr=subprocess.DEVNULL,
    )


def list_commits_for_file(file_path):
    """
    返回从新到旧所有修改过 file_path 的 commit SHA。
    """
    out = run_git(["log", "--format=%H", "--", file_path])
    return [line.strip() for line in out.splitlines() if line.strip()]


def show_file_at_commit(commit, file_path):
    """
    返回指定 commit 中 file_path 的完整内容。
    如果文件不存在，返回空字符串。
    """
    try:
        return run_git(["show", f"{commit}:{file_path}"])
    except subprocess.CalledProcessError:
        return ""


def top_level_defs(source):
    """
    解析源码，返回所有顶层 def / class 名称集合。
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()

    defs = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defs.add(node.name)
    return defs


def find_latest_commit_with_all(commits, symbols, file_path):
    """
    从新到旧查找第一个同时包含所有 symbols 的 commit。
    返回 (commit, source) 或 (None, None)。
    """
    for commit in commits:
        source = show_file_at_commit(commit, file_path)
        if not source:
            continue

        defs = top_level_defs(source)
        if defs.issuperset(symbols):
            return commit, source

    return None, None


def find_latest_commit_per_symbol(commits, symbols, file_path):
    """
    逐个查找每个 symbol 最后一次存在的 commit。
    返回 dict: {symbol: commit 或 None}
    """
    result = {}
    for symbol in symbols:
        found = None
        for commit in commits:
            source = show_file_at_commit(commit, file_path)
            if not source:
                continue
            if symbol in top_level_defs(source):
                found = commit
                break
        result[symbol] = found
    return result


def main():
    parser = argparse.ArgumentParser(
        description="查找 geometrics.py 中给定函数最后一次存在的 commit。"
    )
    parser.add_argument(
        "symbols",
        nargs="*",
        help="要查找的函数名或类名。不传则使用内置的缺失函数列表。",
    )
    parser.add_argument(
        "--file",
        default=DEFAULT_FILE,
        help=f"要查询的文件路径，默认 {DEFAULT_FILE}",
    )
    args = parser.parse_args()

    symbols = list(dict.fromkeys(args.symbols or DEFAULT_SYMBOLS))
    file_path = args.file

    commits = list_commits_for_file(file_path)
    if not commits:
        print(f"No commits found for file: {file_path}")
        return

    print(f"查询文件: {file_path}")
    print(f"待查找 symbol 数: {len(symbols)}\n")

    # 1) 优先查找同时包含全部函数的 commit
    commit, _ = find_latest_commit_with_all(commits, symbols, file_path)

    if commit:
        print(f"包含全部给定 symbol 的最新 commit:\n{commit}")
        return

    # 2) 否则逐 symbol 输出
    print("未找到单个 commit 同时包含全部 symbol。")
    print("以下为每个 symbol 最后一次存在的 commit：\n")

    per_symbol = find_latest_commit_per_symbol(commits, symbols, file_path)
    for symbol, commit in per_symbol.items():
        if commit:
            print(f"{symbol:45s} -> {commit}")
        else:
            print(f"{symbol:45s} -> NOT FOUND")


if __name__ == "__main__":
    main()
