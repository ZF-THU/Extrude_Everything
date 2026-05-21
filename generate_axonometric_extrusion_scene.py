import json
import math
import os

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector


ROOT = r"D:\26_THU\01_ZL\12_ForOwnUse_V4"
OUT_DIR = os.path.join(ROOT, "blender_axonometric_reference")
os.makedirs(OUT_DIR, exist_ok=True)


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def look_at(camera, target):
    direction = Vector(target) - camera.location
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def create_extruded_l_prism():
    # Source cap lies on local/world Z=0.  Copied cap lies on Z=depth.
    cap_2d = [
        (-1.8, -1.1),
        (0.6, -1.1),
        (0.6, -0.25),
        (1.65, -0.25),
        (1.65, 0.75),
        (-0.55, 0.75),
        (-0.55, 1.25),
        (-1.8, 1.25),
    ]
    depth = 1.35

    verts = [(x, y, 0.0) for x, y in cap_2d] + [(x, y, depth) for x, y in cap_2d]
    n = len(cap_2d)
    faces = []
    faces.append(tuple(range(n - 1, -1, -1)))  # source cap
    faces.append(tuple(range(n, 2 * n)))       # copied cap
    for i in range(n):
        j = (i + 1) % n
        faces.append((i, j, j + n, i + n))

    mesh = bpy.data.meshes.new("L_Cap_Extrusion_Mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()

    obj = bpy.data.objects.new("L_Cap_Extrusion", mesh)
    bpy.context.collection.objects.link(obj)

    mat = bpy.data.materials.new("mat_light_gray")
    mat.diffuse_color = (0.78, 0.80, 0.84, 1.0)
    obj.data.materials.append(mat)

    return obj, cap_2d, depth


def add_edge_overlay(obj):
    # Add bevel-free wire overlay so the rendered axonometric image has explicit edges.
    wire = obj.copy()
    wire.data = obj.data.copy()
    wire.name = "L_Cap_Extrusion_Black_Edges"
    bpy.context.collection.objects.link(wire)
    mat = bpy.data.materials.new("mat_black_edges")
    mat.diffuse_color = (0.0, 0.0, 0.0, 1.0)
    wire.data.materials.clear()
    wire.data.materials.append(mat)
    mod = wire.modifiers.new("WireframeEdges", "WIREFRAME")
    mod.thickness = 0.018
    mod.use_even_offset = True
    return wire


def make_material(name, color):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = color
    return mat


def add_cylinder_between(name, start, end, radius, material):
    start = Vector(start)
    end = Vector(end)
    mid = (start + end) * 0.5
    direction = end - start
    length = direction.length
    bpy.ops.mesh.primitive_cylinder_add(vertices=24, radius=radius, depth=length, location=mid)
    obj = bpy.context.object
    obj.name = name
    obj.data.materials.append(material)
    obj.rotation_euler = direction.to_track_quat("Z", "Y").to_euler()
    return obj


def add_sphere(name, location, radius, material):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=12, radius=radius, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.data.materials.append(material)
    return obj


def add_axis_calibration_objects():
    red = make_material("axis_x_red", (1.0, 0.05, 0.05, 1.0))
    green = make_material("axis_y_green", (0.05, 0.75, 0.05, 1.0))
    blue = make_material("axis_z_blue", (0.05, 0.20, 1.0, 1.0))
    black = make_material("axis_origin_black", (0.0, 0.0, 0.0, 1.0))

    objs = []
    objs.append(add_cylinder_between("Axis_O_to_X1_red", (0, 0, 0), (1, 0, 0), 0.025, red))
    objs.append(add_cylinder_between("Axis_O_to_Y1_green", (0, 0, 0), (0, 1, 0), 0.025, green))
    objs.append(add_cylinder_between("Axis_O_to_Z1_blue", (0, 0, 0), (0, 0, 1), 0.025, blue))
    objs.append(add_sphere("Axis_O_black", (0, 0, 0), 0.07, black))
    objs.append(add_sphere("Axis_X1_red", (1, 0, 0), 0.07, red))
    objs.append(add_sphere("Axis_Y1_green", (0, 1, 0), 0.07, green))
    objs.append(add_sphere("Axis_Z1_blue", (0, 0, 1), 0.07, blue))
    return objs


def create_camera():
    cam_data = bpy.data.cameras.new("Orthographic_Axonometric_Camera")
    cam = bpy.data.objects.new("Orthographic_Axonometric_Camera", cam_data)
    bpy.context.collection.objects.link(cam)

    # A stable orthographic axonometric view.  The view direction is not tied to
    # the extrusion axis; it is an independent camera parameter for projection.
    cam.location = Vector((5.0, -6.5, 4.2))
    look_at(cam, (0.0, 0.0, 0.55))
    cam.data.type = "ORTHO"
    cam.data.ortho_scale = 5.2
    cam.data.lens = 70
    cam.data.clip_end = 1000
    bpy.context.scene.camera = cam
    return cam


def setup_render():
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.eevee.taa_render_samples = 64
    scene.render.resolution_x = 1024
    scene.render.resolution_y = 1024
    scene.render.film_transparent = False
    scene.world = bpy.data.worlds.new("WhiteWorld") if scene.world is None else scene.world
    scene.world.color = (1.0, 1.0, 1.0)

    # Lighting for readable faces.
    light_data = bpy.data.lights.new("Key_Area_Light", "AREA")
    light = bpy.data.objects.new("Key_Area_Light", light_data)
    bpy.context.collection.objects.link(light)
    light.location = (2.0, -4.0, 7.0)
    light.data.energy = 350
    light.data.size = 5.0

    # Freestyle outlines in addition to wire overlay.
    scene.render.use_freestyle = True
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "Medium High Contrast"
    scene.view_settings.exposure = 0
    scene.view_settings.gamma = 1


def pixel_from_world(scene, cam, xyz):
    co = world_to_camera_view(scene, cam, Vector(xyz))
    return {
        "normalized": [float(co.x), float(co.y), float(co.z)],
        "pixel": [
            float(co.x * scene.render.resolution_x),
            float((1.0 - co.y) * scene.render.resolution_y),
        ],
    }


def write_camera_params(cam, cap_2d, depth):
    scene = bpy.context.scene
    bpy.context.view_layer.update()
    origin = pixel_from_world(scene, cam, (0, 0, 0))
    x_axis = pixel_from_world(scene, cam, (1, 0, 0))
    y_axis = pixel_from_world(scene, cam, (0, 1, 0))
    z_axis = pixel_from_world(scene, cam, (0, 0, 1))

    def delta(a, b):
        return [b["pixel"][0] - a["pixel"][0], b["pixel"][1] - a["pixel"][1]]

    params = {
        "camera": {
            "type": cam.data.type,
            "location": [float(v) for v in cam.location],
            "rotation_euler_radians": [float(v) for v in cam.rotation_euler],
            "rotation_euler_degrees": [float(math.degrees(v)) for v in cam.rotation_euler],
            "ortho_scale": float(cam.data.ortho_scale),
            "clip_end": float(cam.data.clip_end),
        },
        "render": {
            "resolution_x": int(scene.render.resolution_x),
            "resolution_y": int(scene.render.resolution_y),
            "engine": scene.render.engine,
        },
        "projection_reference": {
            "origin": origin,
            "unit_x": x_axis,
            "unit_y": y_axis,
            "unit_z": z_axis,
            "axis_projected_pixel_vectors": {
                "x": delta(origin, x_axis),
                "y": delta(origin, y_axis),
                "z": delta(origin, z_axis),
            },
            "projection_model": "orthographic axonometric",
            "image_pixel_origin": "top_left",
        },
        "object": {
            "name": "L_Cap_Extrusion",
            "source_cap_plane": "Z=0",
            "copied_cap_plane": f"Z={depth}",
            "extrusion_direction_3d": [0.0, 0.0, 1.0],
            "source_cap_vertices_xy": [[float(x), float(y)] for x, y in cap_2d],
            "depth": float(depth),
        },
    }

    with open(os.path.join(OUT_DIR, "camera_and_projection_params.json"), "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)

    calibration = {
        "image_pixel_origin": "top_left",
        "world_unit_points": {
            "O": [0.0, 0.0, 0.0],
            "X1": [1.0, 0.0, 0.0],
            "Y1": [0.0, 1.0, 0.0],
            "Z1": [0.0, 0.0, 1.0],
        },
        "image_points_pixels": {
            "O": origin["pixel"],
            "X1": x_axis["pixel"],
            "Y1": y_axis["pixel"],
            "Z1": z_axis["pixel"],
        },
        "basis_2d_pixels": {
            "u_x": delta(origin, x_axis),
            "u_y": delta(origin, y_axis),
            "u_z": delta(origin, z_axis),
        },
        "projection_equation": "pixel(P) = O + X*u_x + Y*u_y + Z*u_z",
        "camera": params["camera"],
        "render": params["render"],
    }
    with open(os.path.join(OUT_DIR, "axis_calibration_points.json"), "w", encoding="utf-8") as f:
        json.dump(calibration, f, indent=2)


def main():
    clear_scene()
    obj, cap_2d, depth = create_extruded_l_prism()
    edge_obj = add_edge_overlay(obj)
    cam = create_camera()
    setup_render()
    axis_objs = add_axis_calibration_objects()
    write_camera_params(cam, cap_2d, depth)

    blend_path = os.path.join(OUT_DIR, "axonometric_extrusion_reference.blend")
    render_path = os.path.join(OUT_DIR, "axonometric_extrusion_reference.png")
    axis_render_path = os.path.join(OUT_DIR, "axonometric_axis_calibration.png")
    bpy.ops.wm.save_as_mainfile(filepath=blend_path)
    bpy.context.scene.render.filepath = axis_render_path
    bpy.ops.render.render(write_still=True)

    for axis_obj in axis_objs:
        axis_obj.hide_render = True
        axis_obj.hide_viewport = True
    bpy.context.scene.render.filepath = render_path
    bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    main()
