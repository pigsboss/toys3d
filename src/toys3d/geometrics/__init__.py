# src/toys3d/geometrics/__init__.py

from .euclidean import (
    polygon_area_from_3d_ccw,
    point_in_polygon_2d,
)

from .topology import (
    analyze_mesh_defects,
    extract_boundary_loops,
    compute_topological_reliable_face_mask,
    compute_vertex_face_counts,
    compute_face_edge_types,
    compute_edge_to_faces,
    compute_face_edge_keys,
    compute_face_edge_valences,
    compute_open_edge_data,
    compute_class_neighbor_stats,
    compute_single_face_neighbor_stats,
    compute_face_topology_codes,
    get_face_topology_code_and_order,
    group_faces_by_topology_codes,
    FaceTopologyCode6,
    canonicalize_single_code,
    canonicalize_faces,
    truncate_codes,
    code_to_bytes,
    bytes_to_code,
    code_to_hex,
    hex_to_code,
    save_codes,
    load_codes,
    validate_code,
    build_manifold_face_adjacency,
    is_manifold_closed_boundary,
    expand_face_neighborhood,
    compute_face_distances,
    reconstruct_loop_from_edges,
)

from .discrete import (
    compute_mesh_stats,
    repair_to_watertight,
    compute_face_area_stats,
    compute_bounding_box_stats,
    compute_volume_if_closed,
    build_cotangent_laplacian,
    laplacian_smooth_fixed_boundary,
    compute_curvature_statistics,
)

from .utilities import (
    compute_hole_area_stats,
    repair_mesh_by_removing_duplicates,
    project_vertices_to_shell,
    weld_small_holes,
    trim_isolated_faces,
    fix_winding_consistency,
    fit_watertight_patch_from_component,
    build_hole_diagnosis_data,
    analyze_uncovered_open_edge_components,
    find_minimal_enclosing_manifold_boundary_greedy,
    generate_initial_seifert_disk,
    extract_intersection_faces_by_vertex_state,
)
