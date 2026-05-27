import bpy
import math
import bmesh
from mathutils import Vector

def make_prism(obj_name, vertices, extrude_axis, bezel_width, segments):
    mesh = bpy.data.meshes.new(obj_name + "_Mesh")
    obj = bpy.data.objects.new(obj_name, mesh)
    bpy.context.collection.objects.link(obj)
    bm = bmesh.new()
    verts = [bm.verts.new(p) for p in vertices]
    face = bm.faces.new(verts)
    extruded = bmesh.ops.extrude_face_region(bm, geom=[face])
    new_verts = [v for v in extruded['geom'] if isinstance(v, bmesh.types.BMVert)]
    bmesh.ops.translate(
        bm, 
        vec=Vector(extrude_axis), 
        verts=new_verts
    )
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(mesh)
    bm.free()
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bevel_mod = obj.modifiers.new(name="Bevel", type='BEVEL')
    bevel_mod.width = bezel_width
    bevel_mod.segments = segments
    bpy.ops.object.modifier_apply(modifier=bevel_mod.name)
    return obj

def make_box(name, depth, width, height, location, bezel_width, segments=20):
    bpy.ops.mesh.primitive_cube_add(size=1, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.scale = (depth, width, height)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(scale=True)
    bevel_mod = obj.modifiers.new(name="Bevel", type='BEVEL')
    bevel_mod.width = bezel_width
    bevel_mod.segments = segments
    bpy.ops.object.modifier_apply(modifier=bevel_mod.name)
    return obj

def apply_boolean(target, cutter, op='DIFFERENCE'):
    bpy.context.view_layer.objects.active = target
    bool_mod = target.modifiers.new(name="Boolean", type='BOOLEAN')
    bool_mod.operation = op
    bool_mod.object = cutter
    bool_mod.solver = 'EXACT'
    bpy.ops.object.modifier_apply(modifier=bool_mod.name)
