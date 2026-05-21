import argparse
import copy
import json
import math
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector


ROOT = Path(r"D:\26_THU\01_ZL\12_ForOwnUse_V4")
DEFAULT_OUT_DIR = ROOT / "blender_axonometric_dev_dataset"
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SKETCH_CONDA_ENV = "blender45torch"


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    for collection in (
        bpy.data.meshes,
        bpy.data.materials,
        bpy.data.curves,
        bpy.data.cameras,
        bpy.data.lights,
    ):
        for item in list(collection):
            collection.remove(item)


def look_at(camera, target):
    direction = Vector(target) - camera.location
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def make_material(name, color):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = color
    return mat


def random_plane_normal(rng):
    normal = Vector((rng.uniform(-0.55, 0.55), rng.uniform(-0.55, 0.55), rng.uniform(0.45, 1.0)))
    return normal.normalized()


def plane_basis(normal):
    ref = Vector((0.0, 0.0, 1.0))
    if abs(normal.dot(ref)) > 0.92:
        ref = Vector((1.0, 0.0, 0.0))
    u = ref.cross(normal).normalized()
    v = normal.cross(u).normalized()
    return u, v


def _polygon_is_concave_uv(loop_uv):
    """Ordered 2D polygon (u,v); concave iff turning direction changes."""
    n = len(loop_uv)
    if n < 4:
        return False
    signs = set()
    for i in range(n):
        p0 = loop_uv[i]
        p1 = loop_uv[(i + 1) % n]
        p2 = loop_uv[(i + 2) % n]
        v1 = (p1[0] - p0[0], p1[1] - p0[1])
        v2 = (p2[0] - p1[0], p2[1] - p1[1])
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        if abs(cross) < 1e-9:
            continue
        signs.add(1 if cross > 0 else -1)
        if len(signs) > 1:
            return True
    return False


def random_star_shaped_polygon_uv(rng, n_min, n_max):
    """
    Star-shaped simple polygon in plane coordinates (u,v): vertices are
    kernel + (r_i cos θ_i, r_i sin θ_i) after sorting θ_i around the kernel.
    Random vertex count n in [n_min, n_max].
    """
    n = rng.randint(n_min, n_max)
    concave_intent = rng.random() < 0.65
    angle_step = (2.0 * math.pi) / n
    angles = sorted(i * angle_step + rng.uniform(-0.22, 0.22) * angle_step for i in range(n))

    base_radius = rng.uniform(2.2, 3.8)
    radii = [base_radius * rng.uniform(0.78, 1.18) for _ in range(n)]
    if concave_intent:
        dent_count = rng.randint(1, max(1, n // 3))
        for idx in rng.sample(range(n), dent_count):
            radii[idx] *= rng.uniform(0.38, 0.58)

    loop_uv = [(radii[i] * math.cos(angles[i]), radii[i] * math.sin(angles[i])) for i in range(n)]
    is_concave = _polygon_is_concave_uv(loop_uv)
    return loop_uv, is_concave, n


def points_inside_workspace(points, margin=0.8):
    for point in points:
        if any(coord < margin or coord > 20.0 - margin for coord in point):
            return False
    return True


def create_random_extrusion(
    seed,
    cap_vertex_min=5,
    cap_vertex_max=12,
    extrusion_depth_min=2.8,
    extrusion_depth_max=4.8,
):
    rng = random.Random(seed)
    for _ in range(300):
        normal = random_plane_normal(rng)
        u, v = plane_basis(normal)
        plane_origin = Vector((rng.uniform(7.0, 13.0), rng.uniform(7.0, 13.0), rng.uniform(5.0, 10.0)))
        ku = rng.uniform(-0.45, 0.45)
        kv = rng.uniform(-0.45, 0.45)
        kernel = plane_origin + u * ku + v * kv
        loop_uv, is_concave, n_verts = random_star_shaped_polygon_uv(rng, cap_vertex_min, cap_vertex_max)
        depth = round(rng.uniform(extrusion_depth_min, extrusion_depth_max), 3)
        source = [kernel + u * x + v * y for x, y in loop_uv]
        copied = [p + normal * depth for p in source]
        if points_inside_workspace(source + copied):
            break
    else:
        raise RuntimeError("Could not generate a random extrusion inside the [0,20]^3 workspace.")

    verts = [tuple(p) for p in source] + [tuple(p) for p in copied]
    n = len(source)
    assert n == n_verts
    faces = [tuple(range(n - 1, -1, -1)), tuple(range(n, 2 * n))]
    for i in range(n):
        j = (i + 1) % n
        faces.append((i, j, j + n, i + n))

    mesh = bpy.data.meshes.new("Random_Extrusion_Mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update(calc_edges=True)

    obj = bpy.data.objects.new("Random_Extrusion", mesh)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(make_material("random_object_gray", (0.78, 0.80, 0.84, 1.0)))

    wire = obj.copy()
    wire.data = obj.data.copy()
    wire.name = "Random_Extrusion_Black_Edges"
    bpy.context.collection.objects.link(wire)
    wire.data.materials.clear()
    wire.data.materials.append(make_material("random_edges_black", (0.0, 0.0, 0.0, 1.0)))
    mod = wire.modifiers.new("WireframeEdges", "WIREFRAME")
    mod.thickness = 0.035
    mod.use_even_offset = True

    all_points = source + copied
    mins = [min(p[i] for p in all_points) for i in range(3)]
    maxs = [max(p[i] for p in all_points) for i in range(3)]
    source_center = sum(source, Vector((0.0, 0.0, 0.0))) / len(source)
    copied_center = source_center + normal * depth
    object_info = {
        "seed": int(seed),
        "generation": "star_shaped_random_plane_extrusion",
        "star_kernel_world": [float(c) for c in kernel],
        "star_plane_origin_world": [float(c) for c in plane_origin],
        "star_kernel_offset_uv": [float(ku), float(kv)],
        "cap_vertex_min": int(cap_vertex_min),
        "cap_vertex_max": int(cap_vertex_max),
        "source_cap_vertex_count": int(n),
        "source_cap_is_concave": bool(is_concave),
        "source_cap_z": float(source_center.z),
        "copied_cap_z": float(copied_center.z),
        "depth": float(depth),
        "source_cap_vertices_xyz": [[float(c) for c in p] for p in source],
        "copied_cap_vertices_xyz": [[float(c) for c in p] for p in copied],
        "source_cap_loop_uv_relative_to_kernel": [[float(x), float(y)] for x, y in loop_uv],
        "source_cap_plane": {
            "point": [float(c) for c in source_center],
            "normal": [float(c) for c in normal],
            "basis_u": [float(c) for c in u],
            "basis_v": [float(c) for c in v],
        },
        "copied_cap_plane": {
            "point": [float(c) for c in copied_center],
            "normal": [float(c) for c in normal],
            "basis_u": [float(c) for c in u],
            "basis_v": [float(c) for c in v],
        },
        "extrusion_direction_world": [float(c) for c in normal],
        "bbox_world": {
            "min": [float(c) for c in mins],
            "max": [float(c) for c in maxs],
        },
    }
    return obj, object_info


def support_plane_info(object_info, margin=1.4):
    plane = object_info["source_cap_plane"]
    origin = Vector(plane["point"])
    u = Vector(plane["basis_u"])
    v = Vector(plane["basis_v"])
    cap_points = [Vector(p) for p in object_info["source_cap_vertices_xyz"]]
    local = [((p - origin).dot(u), (p - origin).dot(v)) for p in cap_points]
    min_u = min(p[0] for p in local) - margin
    max_u = max(p[0] for p in local) + margin
    min_v = min(p[1] for p in local) - margin
    max_v = max(p[1] for p in local) + margin
    corners = [
        origin + u * min_u + v * min_v,
        origin + u * max_u + v * min_v,
        origin + u * max_u + v * max_v,
        origin + u * min_u + v * max_v,
    ]
    return {
        "name": "scene_50_support_plane",
        "world_vertices_xyz": [[float(c) for c in p] for p in corners],
        "world_normal": plane["normal"],
        "plane_point": plane["point"],
        "note": "This OBJ plane lies on the source cap plane and supports the extrusion object.",
    }


def add_support_plane_to_blend(plane_info):
    mesh = bpy.data.meshes.new("Support_Plane_Mesh")
    mesh.from_pydata([tuple(v) for v in plane_info["world_vertices_xyz"]], [], [(0, 1, 2, 3)])
    mesh.update(calc_edges=True)
    obj = bpy.data.objects.new("Support_Plane_scene_50", mesh)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(make_material("support_plane_light_gray", (0.58, 0.58, 0.58, 1.0)))
    return obj


def blender_world_to_obj_export_coords(world_xyz):
    # Blender's OBJ export/import convention for this project maps OBJ (x, y, z)
    # back to Blender world as (x, -z, y). Write the inverse so recovery reads it correctly.
    x, y, z = world_xyz
    return [x, z, -y]


def write_support_plane_obj(path, plane_info):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    obj_verts = [blender_world_to_obj_export_coords(v) for v in plane_info["world_vertices_xyz"]]
    lines = [
        "# Random extrusion support plane",
        "o scene_50_support_plane",
    ]
    for x, y, z in obj_verts:
        lines.append(f"v {x:.6f} {y:.6f} {z:.6f}")
    lines.append("s 0")
    lines.append("f 1 2 3 4")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
                uv = Vector((projs[0]["uv_normalized"][0], projs[0]["uv_normalized"][1], 0.0))
                margin = min(uv.x, uv.y, 1.0 - uv.x, 1.0 - uv.y)
                if margin < 0.12:
                    continue
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


def calibration_json(scene, cam, anchor, object_info, plane_info):
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
        "support_plane_obj": plane_info,
    }


def calibration_input_json(full_data):
    data = copy.deepcopy(full_data)
    data.pop("object", None)
    data.pop("support_plane_obj", None)
    return data


def conda_base_prefix():
    try:
        completed = subprocess.run(
            ["conda", "info", "--base"],
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        )
        base = completed.stdout.strip()
        return Path(base) if base else None
    except (FileNotFoundError, subprocess.CalledProcessError, TimeoutError, OSError):
        return None


def conda_env_python_exe(env_name):
    """Return python interpreter path for named conda env, or None if missing."""
    if not env_name:
        return None
    base = conda_base_prefix()
    if not base:
        return None
    if sys.platform == "win32":
        cand = base / "envs" / env_name / "python.exe"
    else:
        cand = base / "envs" / env_name / "bin" / "python"
    return str(cand.resolve()) if cand.is_file() else None


def resolve_sketch_python(explicit_exe, conda_env_name):
    if explicit_exe:
        return explicit_exe
    sk = os.environ.get("SKETCH_PYTHON")
    if sk:
        return sk
    py = conda_env_python_exe(conda_env_name)
    if py:
        return py
    return shutil.which("python") or shutil.which("python3") or shutil.which("py")


def write_handdraw_sketch_png(
    render_png_path,
    out_png_path,
    generation_seed,
    sketch_python_exe=None,
    sketch_conda_env=DEFAULT_SKETCH_CONDA_ENV,
    *,
    black_thr=40,
    close_iter=1,
    stroke_width=3,
    no_thinning=False,
    sketch_handdraw=True,
    wobble_amp=1.2,
    wobble_smooth=61,
    ink_rough=1,
    sketch_invert=True,
):
    """Post-process render PNG via render_sketch_cv.py (Python + OpenCV)."""
    py_exe = resolve_sketch_python(sketch_python_exe, sketch_conda_env)
    if not py_exe:
        raise RuntimeError(
            "Hand-draw sketch requires a Python with OpenCV installed. "
            "Use conda env (default --sketch-conda-env blender45torch), "
            "pass --sketch-python, set SKETCH_PYTHON, or use --no-handdraw to skip."
        )
    script = SCRIPT_DIR / "render_sketch_cv.py"
    if not script.is_file():
        raise FileNotFoundError(f"Missing sketch helper script: {script}")
    wobble_seed = (int(generation_seed) & 0x7FFFFFFF) or 1
    cmd = [
        py_exe,
        str(script),
        "--input",
        str(render_png_path),
        "--output",
        str(out_png_path),
        "--black-thr",
        str(black_thr),
        "--close-iter",
        str(close_iter),
        "--stroke-width",
        str(stroke_width),
        "--wobble-amp",
        str(wobble_amp),
        "--wobble-smooth",
        str(wobble_smooth),
        "--ink-rough",
        str(ink_rough),
        "--wobble-seed",
        str(wobble_seed),
        "--handdraw" if sketch_handdraw else "--no-handdraw",
        "--invert" if sketch_invert else "--no-invert",
    ]
    if no_thinning:
        cmd.append("--no-thinning")
    subprocess.run(cmd, check=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a random orthographic extrusion dataset for reconstruction development."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional deterministic seed. If omitted, a fresh random seed is used each run.",
    )
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--blend-name", default="dev_axonometric_scene.blend")
    parser.add_argument("--calibration-name", default="dev_camera_anchor_calibration.json")
    parser.add_argument("--calibration-input-name", default="dev_camera_anchor_calibrationInput.json")
    parser.add_argument("--obj-name", default="scene_50.obj")
    parser.add_argument("--render-name", default="dev_axonometric_render.png")
    parser.add_argument(
        "--handdraw-render-name",
        default="dev_axonometric_render_handdraw.png",
        help="Hand-drawn line PNG derived from --render-name (OpenCV post-process).",
    )
    parser.add_argument(
        "--no-handdraw",
        action="store_true",
        help="Skip generating the hand-draw sketch PNG.",
    )
    parser.add_argument(
        "--sketch-python",
        default=None,
        help="Python executable with OpenCV (cv2); overrides SKETCH_PYTHON and --sketch-conda-env.",
    )
    parser.add_argument(
        "--sketch-conda-env",
        default=DEFAULT_SKETCH_CONDA_ENV,
        help=(
            "Conda environment name for sketch post-process (default: blender45torch). "
            "Resolved via `conda info --base`. Use empty string to skip and fall back to PATH python."
        ),
    )
    parser.add_argument("--sketch-black-thr", type=int, default=40)
    parser.add_argument("--sketch-close-iter", type=int, default=1)
    parser.add_argument("--sketch-stroke-width", type=int, default=3)
    parser.add_argument(
        "--sketch-no-thinning",
        action="store_true",
        help="Pass --no-thinning to render_sketch_cv (matches Sketch_Own_Cur --no_thinning).",
    )
    parser.add_argument(
        "--sketch-handdraw",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable wobble + ink (Sketch_Own_Cur --handdraw). Use --no-sketch-handdraw to disable.",
    )
    parser.add_argument("--sketch-wobble-amp", type=float, default=1.2)
    parser.add_argument("--sketch-wobble-smooth", type=int, default=61)
    parser.add_argument("--sketch-ink-rough", type=int, default=1)
    parser.add_argument(
        "--sketch-invert",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Black lines on white (--no-sketch-invert: white on black). Matches Sketch --invert style.",
    )
    parser.add_argument("--anchor-render-name", default="dev_axonometric_with_anchor_axes.png")
    parser.add_argument(
        "--cap-vertex-min",
        type=int,
        default=5,
        help="Minimum random vertex count n for the star-shaped base polygon (>=3).",
    )
    parser.add_argument(
        "--cap-vertex-max",
        type=int,
        default=12,
        help="Maximum random vertex count n for the star-shaped base polygon.",
    )
    parser.add_argument(
        "--extrusion-depth-min",
        type=float,
        default=2.8,
        help="Minimum extrusion depth along the cap plane normal (scene units).",
    )
    parser.add_argument(
        "--extrusion-depth-max",
        type=float,
        default=4.8,
        help="Maximum extrusion depth along the cap plane normal (scene units).",
    )
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else None
    args, _ = parser.parse_known_args(argv)
    if args.cap_vertex_min < 3:
        parser.error("--cap-vertex-min must be >= 3")
    if args.cap_vertex_max < args.cap_vertex_min:
        parser.error("--cap-vertex-max must be >= --cap-vertex-min")
    if args.extrusion_depth_min <= 0:
        parser.error("--extrusion-depth-min must be > 0")
    if args.extrusion_depth_max < args.extrusion_depth_min:
        parser.error("--extrusion-depth-max must be >= --extrusion-depth-min")
    return args


def resolve_output_dir(out_dir_arg):
    """Relative paths are anchored to this script's directory (not Blender's cwd)."""
    p = Path(out_dir_arg).expanduser()
    if not p.is_absolute():
        p = (SCRIPT_DIR / p).resolve()
    else:
        p = p.resolve()
    return p


def main():
    args = parse_args()
    out_dir = resolve_output_dir(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    generation_seed = args.seed if args.seed is not None else random.SystemRandom().randrange(0, 2**32)
    obj_name = args.obj_name

    clear_scene()
    _, object_info = create_random_extrusion(
        generation_seed,
        cap_vertex_min=args.cap_vertex_min,
        cap_vertex_max=args.cap_vertex_max,
        extrusion_depth_min=args.extrusion_depth_min,
        extrusion_depth_max=args.extrusion_depth_max,
    )
    plane_info = support_plane_info(object_info)
    cam = create_camera()
    setup_render()
    bpy.context.view_layer.update()

    anchor = choose_visible_anchor(bpy.context.scene, cam, object_info)
    calib_objs = add_anchor_visualization(anchor)
    bpy.context.view_layer.update()
    data = calibration_json(bpy.context.scene, cam, anchor, object_info, plane_info)

    (out_dir / args.calibration_name).write_text(json.dumps(data, indent=2), encoding="utf-8")
    (out_dir / args.calibration_input_name).write_text(
        json.dumps(calibration_input_json(data), indent=2),
        encoding="utf-8",
    )
    write_support_plane_obj(out_dir / obj_name, plane_info)

    bpy.ops.wm.save_as_mainfile(filepath=str(out_dir / args.blend_name))

    bpy.context.scene.render.filepath = str(out_dir / args.anchor_render_name)
    bpy.ops.render.render(write_still=True)

    for obj in calib_objs:
        obj.hide_render = True
        obj.hide_viewport = True
    bpy.context.scene.render.filepath = str(out_dir / args.render_name)
    bpy.ops.render.render(write_still=True)

    render_png_path = out_dir / args.render_name
    handdraw_png_path = out_dir / args.handdraw_render_name
    if not args.no_handdraw:
        write_handdraw_sketch_png(
            render_png_path,
            handdraw_png_path,
            generation_seed,
            args.sketch_python,
            args.sketch_conda_env,
            black_thr=args.sketch_black_thr,
            close_iter=args.sketch_close_iter,
            stroke_width=args.sketch_stroke_width,
            no_thinning=args.sketch_no_thinning,
            sketch_handdraw=args.sketch_handdraw,
            wobble_amp=args.sketch_wobble_amp,
            wobble_smooth=args.sketch_wobble_smooth,
            ink_rough=args.sketch_ink_rough,
            sketch_invert=args.sketch_invert,
        )

    print(f"Wrote {out_dir / args.blend_name}")
    print(f"Wrote {out_dir / args.calibration_name}")
    print(f"Wrote {out_dir / args.calibration_input_name}")
    print(f"Wrote {out_dir / obj_name}")
    print(f"Wrote {out_dir / args.anchor_render_name}")
    print(f"Wrote {render_png_path}")
    if not args.no_handdraw:
        print(f"Wrote {handdraw_png_path}")


if __name__ == "__main__":
    main()
