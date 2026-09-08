# src/toys3d/geometrics/__init__.py

from .euclidean import (
    polygon_area_from_3d_ccw,
    point_in_polygon_2d,
    plucker_design_matrix,
    axis_from_plucker,
    orthogonalize_axes,
    line_line_distance_and_midpoint,
    point_line_distance,
    intersect_line_plane,
    kmeans_1d,
    average_antiparallel_directions,
    normalize,
    signed_distance_to_plane,
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
    extract_plate_boundary_loops,
    get_k_ring_neighbors,
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
    estimate_symmetry_plane_voxel,
)

from .analysis import (
    build_box_aligned_frame_voxel,
    build_box_aligned_frame_mesh,
    fit_line_3d,
    fit_circle_3d,
    ransac_plane_fitting,
    multi_ransac_planes,
    map_labels_from_proxy,
    segment_plates_by_plane_fitting,
)

from .generation import (
    repair_mesh_by_removing_duplicates,
    repair_nonmanifold_edges,
    fill_small_holes,
)

from .utilities import (
    compute_hole_area_stats,
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
    segment_tubular_regions,
)

# ------------------------------------------------------------------------------
# Legacy fallbacks for functions still used by shell.py / downstream tools but
# not yet moved into the split geometrics sub-package.
# ------------------------------------------------------------------------------
from toys3d.geometrics_older import (
    fit_spline_3d,
    classify_edge_regularity,
    detect_multiscale_edges,
    segment_regions_by_edges,
    estimate_shell_thickness,
    segment_plates_by_smoothness,
    detect_thin_regions,
    compute_wall_thickness_statistics,
    build_proxy_mesh,
    segment_plates_by_local_clustering,
)
