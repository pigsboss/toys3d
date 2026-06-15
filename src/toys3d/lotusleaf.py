#!/usr/bin/env python
import numpy as np
import trimesh
import os
from scipy.spatial import Delaunay
from collections import Counter

leaf_diameter = 145.0
skirt_width = 30.0
skirt_angle = np.deg2rad(60.0)
skirt_radius = skirt_width / np.cos(skirt_angle)
stalk_top_diameter = 30.0
stalk_bottom_diameter = 8.0
stalk_height = 80.0
num_spokes = 12
thickness = 2.0
skirt_amplitude = 5.0
stalk_amplitude = 0.5

num_radius = 50
num_azimuth = 1000

agv = np.arange(num_azimuth) / num_azimuth * 2.0 * np.pi
rmin = leaf_diameter * 0.5
rmax = rmin + skirt_width
rgv = np.linspace(rmin, rmax, num_radius)
r, a = np.meshgrid(rgv, agv, indexing='xy')
z = np.cos(a * num_spokes) * skirt_amplitude * r / rmax + (skirt_radius**2.0 - (r-rmin)**2.0)**0.5 - skirt_radius
x = np.cos(a) * r
y = np.sin(a) * r
v_leaf_top = np.column_stack((x.ravel(), y.ravel(), z.ravel() + thickness/2.0))
v_leaf_bot = np.column_stack((x.ravel(), y.ravel(), z.ravel() - thickness/2.0))

if stalk_height > 0:
    # stalk vertices
    stalk_brim_radius = (stalk_top_diameter - stalk_bottom_diameter) / 2.0
    stalk_brim_radius_in  = stalk_brim_radius + thickness / 2.0
    stalk_brim_radius_out = stalk_brim_radius - thickness / 2.0
    zgv_in  = np.linspace(-stalk_brim_radius,  thickness/2.0, num_radius)
    zgv_out = np.linspace(-stalk_brim_radius, -thickness/2.0, num_radius)
    z_in,  _ = np.meshgrid(zgv_in,  agv, indexing='xy')
    z_out, _ = np.meshgrid(zgv_out, agv, indexing='xy')
    r_in  = stalk_bottom_diameter/2.0 + stalk_brim_radius - (stalk_brim_radius_in **2.0 - (z_in +stalk_brim_radius)**2.0)**0.5 + np.cos(a * num_spokes / 2.) * stalk_amplitude
    r_out = stalk_bottom_diameter/2.0 + stalk_brim_radius - (stalk_brim_radius_out**2.0 - (z_out+stalk_brim_radius)**2.0)**0.5 + np.cos(a * num_spokes / 2.) * stalk_amplitude
    x_in  = np.cos(a) * r_in
    y_in  = np.sin(a) * r_in
    x_out = np.cos(a) * r_out
    y_out = np.sin(a) * r_out
    v_top = np.column_stack((
        np.column_stack((x_in[:,0], x_in, x)).ravel(),
        np.column_stack((y_in[:,0], y_in, y)).ravel(),
        np.column_stack((np.full_like(z_in[:, 0], -stalk_height), z_in, z+thickness/2.0)).ravel()))
    v_bot = np.column_stack((
        np.column_stack((x_out[:,0], x_out, x)).ravel(),
        np.column_stack((y_out[:,0], y_out, y)).ravel(),
        np.column_stack((np.full_like(z_out[:, 0], -stalk_height), z_out, z-thickness/2.0)).ravel()))
    rows = num_azimuth
    cols = num_radius * 2 + 1
    offset = 0
    # stalk bottom ring wall
    r0 = np.arange(rows).ravel().astype('int32')
    r1 = np.mod(r0+1, rows)
    p0 = r0 * cols + offset
    p1 = r1 * cols + offset
    p2 = p0 + rows*cols + offset
    p3 = p1 + rows*cols + offset
    tri_ring_u = np.column_stack((p0.ravel(), p1.ravel(), p3.ravel()))
    tri_ring_d = np.column_stack((p0.ravel(), p3.ravel(), p2.ravel()))
    tri = np.vstack((tri_ring_u, tri_ring_d))
else:
    offset = 1
    rows = num_azimuth
    cols = num_radius
    v_top = np.vstack(([0., 0.,  thickness/2.], v_leaf_top))
    v_bot = np.vstack(([0., 0., -thickness/2.], v_leaf_bot))
    r1 = np.arange(rows).ravel().astype('int32')
    p1 = r1 * cols + offset
    p2 = np.mod(r1+1, rows) * cols + offset
    p0 = np.full_like(p1, 0)
    tri_top_center = np.column_stack((p0, p1, p2))
    tri_bot_center = np.column_stack((p0, p2, p1)) + rows*cols + offset
    tri = np.vstack((tri_top_center, tri_bot_center))

R, C = np.meshgrid(np.arange(rows).astype('int32'), np.arange(cols-1).astype('int32'), indexing='ij')
p0 = R * cols + C + offset
p1 = p0 + 1
p2 = np.mod(R+1, rows) * cols + C + offset
p3 = p2 + 1
tri_top_skirt_d = np.column_stack((p0.ravel(), p1.ravel(), p3.ravel()))
tri_top_skirt_u = np.column_stack((p0.ravel(), p3.ravel(), p2.ravel()))
tri_bot_skirt_d = np.column_stack((p0.ravel(), p2.ravel(), p1.ravel())) + rows*cols + offset
tri_bot_skirt_u = np.column_stack((p2.ravel(), p3.ravel(), p1.ravel())) + rows*cols + offset

# skirt side wall
r0 = np.arange(rows).ravel().astype('int32')
r1 = np.mod(r0+1, rows)
p0 = r0 * cols + cols - 1 + offset
p1 = r1 * cols + cols - 1 + offset
p2 = p0 + rows*cols + offset
p3 = p1 + rows*cols + offset
tri_side_u = np.column_stack((p0.ravel(), p2.ravel(), p1.ravel()))
tri_side_d = np.column_stack((p2.ravel(), p3.ravel(), p1.ravel()))

mesh = trimesh.Trimesh(
    np.vstack((v_top, v_bot)),
    np.vstack((
        tri,
        tri_top_skirt_u, tri_top_skirt_d,
        tri_bot_skirt_u, tri_bot_skirt_d,
        tri_side_u, tri_side_d
    )))
mesh.fix_normals()

if mesh.is_watertight:
    print(f'Watertight solid generated with {len(mesh.vertices)} vertices and {len(mesh.faces)} faces.')
else:
    print("Non-watertight mesh generated.")
    edge_counter = Counter()
    for face in mesh.faces:
        edge_counter[tuple(sorted([face[0], face[1]]))] += 1
        edge_counter[tuple(sorted([face[1], face[2]]))] += 1
        edge_counter[tuple(sorted([face[2], face[0]]))] += 1
    num_open_edges = 0
    num_complex_edges = 0
    num_kissing_edges = 0
    for e, cnt in edge_counter.items():
        if cnt < 2:
            num_open_edges += 1
        elif cnt > 2:
            num_complex_edges += 1
            if cnt == 4:
                num_kissing_edges += 1
            v1 = mesh.vertices[e[0]]
            v2 = mesh.vertices[e[1]]
    print(f"Diagnosis: {num_open_edges} open edges found, "
          f"{num_complex_edges} non-manifold edges found, "
          f"{num_kissing_edges} kissing edges found.")

mesh.export('lotusleaf.stl')
