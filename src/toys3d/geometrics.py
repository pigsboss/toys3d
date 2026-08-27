# src/toys3d/geometrics.py
# 注意：此文件为占位实现，用于绕过 ImportError。
# 请通过 `git restore src/toys3d/geometrics.py` 恢复真正实现。


def _unavailable(name):
    def wrapper(*args, **kwargs):
        raise NotImplementedError(
            f"geometrics.{name} 尚未实现；请恢复原始 geometrics.py"
        )
    wrapper.__name__ = name
    return wrapper


_REQUIRED_NAMES = [
    "compute_mesh_stats",
    "analyze_mesh_defects",
    "compute_hole_area_stats",
    "extract_boundary_loops",
    "polygon_area_from_3d_ccw",
    "compute_reliable_face_mask",
    "repair_mesh_by_removing_duplicates",
    "project_vertices_to_shell",
    "weld_small_holes",
    "repair_nonmanifold_edges",
    "fill_holes_adaptive",
    "compute_loop_flatness",
    "repair_normals",
    "remove_small_open_edge_chains",
    "remove_pseudo_holes",
    "repair_to_watertight",
    "fuse_reliable_faces_with_shell",
]

for _name in _REQUIRED_NAMES:
    globals()[_name] = _unavailable(_name)

del _name, _REQUIRED_NAMES
