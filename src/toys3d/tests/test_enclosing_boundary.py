import argparse
import json
from pathlib import Path

import numpy as np
import pytest
import trimesh

from toys3d.meshinspect import perform_hole_diagnosis
from toys3d.geometrics import find_minimal_enclosing_manifold_boundary_greedy


def _add_closed_icosphere(radius=1.0, subdivisions=2):
    """Create a closed icosphere mesh."""
    mesh = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    # Ensure it's watertight (should be)
    assert mesh.is_watertight
    return mesh


def _get_disk_seed_faces(mesh, center_vertex=0, n_rings=1):
    """
    Return a set of faces forming a small disk around a given vertex.
    For an icosphere, faces are roughly evenly distributed.
    We simply take all faces that contain a given vertex, plus optionally
    their neighbours.
    """
    # Get all faces incident to center_vertex
    face_mask = np.any(mesh.faces == center_vertex, axis=1)
    seed_faces = set(np.where(face_mask)[0].tolist())
    # Optional: add adjacent faces to ensure a nice disk
    if n_rings > 1:
        # Simple expansion via face adjacency
        adj = {i: set() for i in range(len(mesh.faces))}
        for f0, f1 in mesh.face_adjacency:
            adj[int(f0)].add(int(f1))
            adj[int(f1)].add(int(f0))
        current = set(seed_faces)
        for _ in range(n_rings - 1):
            new = set()
            for f in current:
                new.update(adj[f])
            current = new
        seed_faces = current
    return seed_faces


def test_find_minimal_enclosing_boundary_greedy():
    """On a closed mesh, a disk of faces has a manifold closed boundary."""
    mesh = _add_closed_icosphere()
    seed_faces = _get_disk_seed_faces(mesh, center_vertex=0, n_rings=1)
    component = {"face_ids": sorted(seed_faces)}

    result = find_minimal_enclosing_manifold_boundary_greedy(mesh, component, max_depth=5)

    assert result['success'] is True
    assert seed_faces.issubset(set(result['enclosed_faces']))
    assert len(result['boundary_edges']) > 0
    assert len(result['boundary_vertices']) > 0


def test_hole_diagnosis_with_enclosing_boundaries(tmp_path):
    """
    Run hole diagnosis on a closed mesh (no open edges). It should complete
    without error and produce the component analysis JSON (even if empty).
    """
    mesh = _add_closed_icosphere()
    args = argparse.Namespace(
        hole_diagnosis_output=str(tmp_path),
        compute_enclosing_boundaries=True,
    )
    perform_hole_diagnosis(mesh, args)

    # The function should create the JSON file
    json_path = tmp_path / "uncovered_component_analysis.json"
    assert json_path.exists()
    with open(json_path) as f:
        data = json.load(f)
    # For a watertight mesh, there are no uncovered components
    assert data.get("components", []) == []
