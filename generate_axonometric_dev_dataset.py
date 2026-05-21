import json
import math
import os

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector


ROOT = r"D:\26_THU\01_ZL\12_ForOwnUse_V4"
OUT_DIR = os.path.join(ROOT, "blender_axonometric_dev_dataset")
os.makedirs(OUT_DIR, exist_ok=True)


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def look_at(camera, target):
    direction = Vector(target) - camera.location
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def make_material(name, color):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = color
    return mat


def create_test_extrusion():
    # A non-rectangular cap inside the known [0, 20]^3 workspace.
    cap_xy = [
        (4.5, 5.0),
        (11.0, 5.0),
        (11.0, 7.3),
        (15.0, 7.3),
        (15.0, 11.8),
        (8.6, 11.8),
        (8.6, 14.2),
        (4.5, 14.2),
    ]
    z0 = 7.0
    depth = 4.0
    verts = [(x, y, z0) for x, y in cap_xy] + [(x, y, z0 + depth) for x, y in cap_xy]
    n = len(cap_xy)
    faces = [tuple(range(n - 1, -1, -1)), tuple(range(n, 2 * n))]
    for i in range(n):
        j = (i + 1) % n
        faces.append((i, j, j + n, i + n))

    mesh = bpy.data.meshes.new("Dev_L_Extrusion_Mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()

    obj = bpy.data.objects.new("Dev_L_Extrusion", mesh)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(make_material("dev_object_gray", (0.78, 0.80, 0.84, 1.0)))

    wire = obj.copy()
    wire.data = obj.data.copy()
    wire.name = "Dev_L_Extrusion_Black_Edges"
    bpy.context.collection.objects.link(wire)
    wire.data.materials.clear()
    wire.data.materials.append(make_material("dev_edges_black", (0.0, 0.0, 0.0, 1.0)))
    mod = wire.modifiers.new("WireframeEdges", "WIREFRAME")
    mod.thickness = 0.035
    mod.use_even_offset = True

    return obj, {
        "source_cap_z": z0,
        "copied_cap_z": z0 + depth,
        "depth": depth,
        "source_cap_vertices_xyz": [[float(x), float(y), float(z0)] for x, y in cap_xy],
        "copied_cap_vertices_xyz": [[float(x), float(y), float(z0 + depth)] for x, y in cap_xy],
        "extrusion_direction_world": [0.0, 0.0, 1.0],
    }


def create_camera():
    cam_data = bpy.data.cameras.new("Dev_Orthographic_Axonometric_Camera")
    cam = bpy.data.objects.new("Dev_Orthographic_Axonometric_Camera", cam_data)
    bpy.context.collection.objects.link(cam)
    cam.location = Vector((35.0, -42.0, 31.0))
    look_at(cam, (10.0, 10.0, 9.0))
    cam.data.type = "ORTHO"
    cam.data.ortho_scale = 28.0
    cam.data.clip_end = 2000.0
    bpy.context.scene.camera = cam
    return cam


def setup_render():
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.eevee.taa_render_samples = 64
    scene.render.resolution_x = 1400
    scene.render.resolution_y = 1000
    scene.render.film_transparent = False
    scene.world = bpy.data.worlds.new("WhiteWorld") if scene.world is None else scene.world
    scene.world.color = (1.0, 1.0, 1.0)
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "Medium High Contrast"
    scene.render.use_freestyle = True

    light_data = bpy.data.lights.new("Dev_Key_Area_Light", "AREA")
    light = bpy.data.objects.new("Dev_Key_Area_Light", light_data)
    bpy.context.collection.objects.link(light)
    light.location = (16.0, -8.0, 30.0)
    light.data.energy = 500
    light.data.size = 8.0


def project_point(scene, cam, xyz):
    co = world_to_camera_view(scene, cam, Vector(xyz))
    return {
        "world": [float(xyz[0]), float(xyz[1]), float(xyz[2])],
        "uv_normalized": [float(co.x), float(co.y), float(co.z)],
        "pixel_top_left": [
            float(co.x * scene.render.resolution_x),
            float((1.0 - co.y) * scene.render.resolution_y),
        ],
        "visible_in_frame": bool(0.0 <= co.x <= 1.0 and 0.0 <= co.y <= 1.0),
    }


def point_near_object(point, object_info, margin=1.5):
    verts = object_info["source_cap_vertices_xyz"] + object_info["copied_cap_vertices_xyz"]
    mins = [min(v[i] for v in verts) - margin for i in range(3)]
    maxs = [max(v[i] for v in verts) + margin for i in range(3)]
    return all(mins[i] <= float(point[i]) <= maxs[i] for i in range(3))


def object_projected_bbox(scene, cam, object_info, margin_px=100.0):
    verts = object_info["source_cap_vertices_xyz"] + object_info["copied_cap_vertices_xyz"]
    pixels = [project_point(scene, cam, v)["pixel_top_left"] for v in verts]
    xs = [p[0] for p in pixels]
    ys = [p[1] for p in pixels]
    return (
        min(xs) - margin_px,
        min(ys) - margin_px,
        max(xs) + margin_px,
        max(ys) + margin_px,
    )


def pixel_inside_bbox(pixel, bbox):
    x, y = pixel
    x0, y0, x1, y1 = bbox
    return x0 <= x <= x1 and y0 <= y <= y1


def choose_visible_anchor(scene, cam, object_info):
    best = None
    best_score = float("inf")
    object_bbox = object_projected_bbox(scene, cam, object_info, margin_px=120.0)
    for x in range(0, 20):
        for y in range(0, 20):
            for z in range(0, 20):
                pts = [(x, y, z), (x + 1, y, z), (x, y + 1, z), (x, y, z + 1)]
                if any(point_near_object(p, object_info) for p in pts):
                    continue
                projs = [project_point(scene, cam, p) for p in pts]
                if not all(p["visible_in_frame"] for p in projs):
                    continue
                if any(pixel_inside_bbox(p["pixel_top_left"], object_bbox) for p in projs):
                    continue
                # Prefer a point near image center and not too close to the frame border.
                uv = Vector((projs[0]["uv_normalized"][0], projs[0]["uv_normalized"][1], 0.0))
                margin = min(uv.x, uv.y, 1.0 - uv.x, 1.0 - uv.y)
                if margin < 0.12:
                    continue
                # Prefer upper-left empty area for easy visual inspection.
                target = Vector((0.22, 0.78, 0.0))
                score = (uv - target).length
                if score < best_score:
                    best_score = score
                    best = (x, y, z)
    if best is None:
        raise RuntimeError("Could not find a visible anchor point with visible unit axes.")
    return best


def add_cylinder_between(name, start, end, radius, material):
    start = Vector(start)
    end = Vector(end)
    mid = (start + end) * 0.5
    direction = end - start
    bpy.ops.mesh.primitive_cylinder_add(vertices=24, radius=radius, depth=direction.length, location=mid)
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


def add_text(name, text, location, size, material):
    curve = bpy.data.curves.new(name, "FONT")
    curve.body = text
    curve.size = size
    curve.align_x = "CENTER"
    curve.align_y = "CENTER"
    obj = bpy.data.objects.new(name, curve)
    bpy.context.collection.objects.link(obj)
    obj.location = Vector(location)
    obj.data.materials.append(material)
    return obj


def add_anchor_visualization(anchor):
    x, y, z = anchor
    red = make_material("calib_x_red", (1.0, 0.05, 0.05, 1.0))
    green = make_material("calib_y_green", (0.05, 0.75, 0.05, 1.0))
    blue = make_material("calib_z_blue", (0.05, 0.20, 1.0, 1.0))
    black = make_material("calib_origin_black", (0.0, 0.0, 0.0, 1.0))
    objs = []
    O = (x, y, z)
    X1 = (x + 1, y, z)
    Y1 = (x, y + 1, z)
    Z1 = (x, y, z + 1)
    objs.append(add_cylinder_between("Anchor_O_to_X1_red", O, X1, 0.06, red))
    objs.append(add_cylinder_between("Anchor_O_to_Y1_green", O, Y1, 0.06, green))
    objs.append(add_cylinder_between("Anchor_O_to_Z1_blue", O, Z1, 0.06, blue))
    objs.append(add_sphere("Anchor_O_black", O, 0.16, black))
    objs.append(add_sphere("Anchor_X1_red", X1, 0.15, red))
    objs.append(add_sphere("Anchor_Y1_green", Y1, 0.15, green))
    objs.append(add_sphere("Anchor_Z1_blue", Z1, 0.15, blue))
    objs.append(add_text("Label_O", "O", (x - 0.25, y - 0.25, z), 0.38, black))
    objs.append(add_text("Label_X1", "X1", (x + 1.25, y, z), 0.34, red))
    objs.append(add_text("Label_Y1", "Y1", (x, y + 1.25, z), 0.34, green))
    objs.append(add_text("Label_Z1", "Z1", (x, y, z + 1.25), 0.34, blue))
    return objs


def calibration_json(scene, cam, anchor, object_info):
    x, y, z = anchor
    O = project_point(scene, cam, (x, y, z))
    X1 = project_point(scene, cam, (x + 1, y, z))
    Y1 = project_point(scene, cam, (x, y + 1, z))
    Z1 = project_point(scene, cam, (x, y, z + 1))

    def delta(a, b):
        return [
            float(b["pixel_top_left"][0] - a["pixel_top_left"][0]),
            float(b["pixel_top_left"][1] - a["pixel_top_left"][1]),
        ]

    return {
        "coordinate_space": "Blender world coordinates",
        "known_workspace_cube": {
            "min": [0.0, 0.0, 0.0],
            "max": [20.0, 20.0, 20.0],
        },
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
            "image_pixel_origin": "top_left",
        },
        "anchor": {
            "world": [float(x), float(y), float(z)],
            "O": O,
            "X1": X1,
            "Y1": Y1,
            "Z1": Z1,
            "unit_uv_vectors_pixels": {
                "u_x": delta(O, X1),
                "u_y": delta(O, Y1),
                "u_z": delta(O, Z1),
            },
            "projection_equation": "pixel(P) = O + (X-Xa)*u_x + (Y-Ya)*u_y + (Z-Za)*u_z",
        },
        "object": object_info,
    }


def main():
    clear_scene()
    obj, object_info = create_test_extrusion()
    cam = create_camera()
    setup_render()
    bpy.context.view_layer.update()

    anchor = choose_visible_anchor(bpy.context.scene, cam, object_info)
    calib_objs = add_anchor_visualization(anchor)
    bpy.context.view_layer.update()
    data = calibration_json(bpy.context.scene, cam, anchor, object_info)

    with open(os.path.join(OUT_DIR, "dev_camera_anchor_calibration.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT_DIR, "dev_axonometric_scene.blend"))

    bpy.context.scene.render.filepath = os.path.join(OUT_DIR, "dev_axonometric_with_anchor_axes.png")
    bpy.ops.render.render(write_still=True)

    for obj in calib_objs:
        obj.hide_render = True
        obj.hide_viewport = True
    bpy.context.scene.render.filepath = os.path.join(OUT_DIR, "dev_axonometric_render.png")
    bpy.ops.render.render(write_still=True)


if __name__ == "__main__":
    main()
