import argparse
import json
import math
import re
import subprocess
import textwrap
from pathlib import Path

import numpy as np


STROKE_RE = re.compile(
    r"stroke\s+(\d+):.*?arc=([0-9.]+).*?chord=([0-9.]+).*?"
    r"p0=\(([0-9.+-]+),([0-9.+-]+)\),\s*p1=\(([0-9.+-]+),([0-9.+-]+)\)"
)

CLUSTER_RE = re.compile(
    r"rank\s+(\d+)\s+cluster\s+(\d+).*?side=\[([^\]]*)\].*?"
    r"best_cap_strokes=\[([^\]]*)\].*?best_cap_score=([0-9.+-]+)"
)

IOU_RE = re.compile(
    r"iou_rank\s+(\d+):.*?iou=([0-9.]+).*?source=rank_(\d+)_cluster_(\d+)_score_([A-Za-z0-9.+-]+)"
)

OVERLAY_NAME_RE = re.compile(
    r"iou_rank_(\d+)_iou_([0-9]+).*?rank_(\d+)_cluster_(\d+)_score_([A-Za-z0-9.+-]+)_side_bestcap_overlay"
)

CAP_GRAPH_CLUSTER_RE = re.compile(r"cluster\s+(\d+):")
CAP_GRAPH_COMPONENT_RE = re.compile(
    r"component\s+(\d+):.*?pruned_closed=(True|False).*?pruned_strokes=\[([^\]]*)\]"
)
CAP_GRAPH_ENDPOINT_RE = re.compile(
    r"endpoint\s+s(\d+):(start|end)=\(([0-9.+-]+),([0-9.+-]+)\).*?degree=(\d+).*?matches=\[([^\]]*)\]"
)


def parse_int_list(text):
    text = text.strip()
    if not text:
        return []
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_float_token(text):
    token = str(text).strip()
    lowered = token.lower()
    if lowered in {"neginf", "-inf", "-infinity"}:
        return float("-inf")
    if lowered in {"inf", "+inf", "infinity", "+infinity"}:
        return float("inf")
    return float(token)


def load_strokes(path):
    strokes = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        m = STROKE_RE.search(line)
        if not m:
            continue
        sid = int(m.group(1))
        strokes[sid] = {
            "id": sid,
            "arc": float(m.group(2)),
            "chord": float(m.group(3)),
            "p0": np.array([float(m.group(4)), float(m.group(5))], dtype=float),
            "p1": np.array([float(m.group(6)), float(m.group(7))], dtype=float),
        }
    if not strokes:
        raise RuntimeError(f"No strokes parsed from {path}")
    return strokes


def load_strokes_json(path):
    path = Path(path)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    strokes = {}
    for item in data.get("strokes", []):
        sid = int(item["index"])
        points = np.array(item.get("points", []), dtype=float)
        if len(points) < 2:
            continue
        strokes[sid] = {
            "id": sid,
            "arc": float(item.get("arc", 0.0)),
            "chord": float(item.get("chord", 0.0)),
            "straightness": float(item.get("straightness", 1.0)),
            "p0": points[0],
            "p1": points[-1],
            "points": points,
        }
    return strokes if strokes else None


def load_cluster_entries(path):
    entries = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        m = CLUSTER_RE.search(line)
        if not m:
            continue
        copy_side_m = re.search(r"copy_side_stroke=([0-9-]+)", line)
        copy_reason_m = re.search(r"copy_reason=([^\s]+)", line)
        copy_iou_m = re.search(r"copy_iou=([0-9.]+)", line)
        entries.append(
            {
                "rank": int(m.group(1)),
                "cluster": int(m.group(2)),
                "side": parse_int_list(m.group(3)),
                "best_cap_strokes": parse_int_list(m.group(4)),
                "best_cap_score": parse_float_token(m.group(5)),
                "copy_side_stroke": int(copy_side_m.group(1)) if copy_side_m else None,
                "copy_reason": copy_reason_m.group(1) if copy_reason_m else None,
                "copy_iou": float(copy_iou_m.group(1)) if copy_iou_m else None,
                "raw_line": line.strip(),
            }
        )
    if not entries:
        raise RuntimeError(f"No cluster entries parsed from {path}")
    return entries


def group_endpoint_members(endpoint_members, tol):
    groups = []
    for member in endpoint_members:
        point = member["pixel"]
        best_i = None
        best_d = float("inf")
        for i, group in enumerate(groups):
            d = norm(point - group["center"])
            if d < best_d:
                best_i = i
                best_d = d
        if best_i is not None and best_d <= tol:
            group = groups[best_i]
            group["members"].append(member)
            group["center"] = np.mean([m["pixel"] for m in group["members"]], axis=0)
        else:
            groups.append({"members": [member], "center": point.copy()})

    if groups:
        centroid = np.mean([g["center"] for g in groups], axis=0)
        groups.sort(key=lambda g: math.atan2(g["center"][1] - centroid[1], g["center"][0] - centroid[0]))
    return groups


def snap_nearby_endpoint_pairs(endpoint_members, tol):
    """One non-recursive snap pass: merge only disjoint close endpoint pairs."""
    pairs = []
    for i in range(len(endpoint_members)):
        for j in range(i + 1, len(endpoint_members)):
            d = norm(endpoint_members[i]["pixel"] - endpoint_members[j]["pixel"])
            if d <= tol:
                pairs.append((d, i, j))
    pairs.sort(key=lambda item: item[0])

    used = set()
    groups = []
    for d, i, j in pairs:
        if i in used or j in used:
            continue
        used.add(i)
        used.add(j)
        members = [endpoint_members[i], endpoint_members[j]]
        groups.append(
            {
                "members": members,
                "center": np.mean([m["pixel"] for m in members], axis=0),
                "one_pass_pair_snap_distance": float(d),
            }
        )

    for i, member in enumerate(endpoint_members):
        if i in used:
            continue
        groups.append({"members": [member], "center": member["pixel"].copy()})

    if groups:
        centroid = np.mean([g["center"] for g in groups], axis=0)
        groups.sort(key=lambda g: math.atan2(g["center"][1] - centroid[1], g["center"][0] - centroid[0]))
    return groups


def load_cap_endpoint_graph_groups(
    path,
    cluster_id,
    cap_ids,
    tol,
    strokes=None,
    straightness_threshold=0.9,
    resample_step=15.0,
    curve_min_chord_px=30.0,
    curve_min_p90_chord_dev_px=5.0,
    curve_min_dev_ratio=0.04,
    curve_min_pca_rms_px=2.0,
):
    path = Path(path)
    if not path.exists():
        return None, f"{path} does not exist"

    target_cap_ids = set(cap_ids)
    current_cluster = None
    current_component = None
    components = []

    for line in path.read_text(encoding="utf-8").splitlines():
        cluster_match = CAP_GRAPH_CLUSTER_RE.search(line)
        if cluster_match:
            current_cluster = int(cluster_match.group(1))
            current_component = None
            continue

        if current_cluster != cluster_id:
            continue

        component_match = CAP_GRAPH_COMPONENT_RE.search(line)
        if component_match:
            strokes_match = re.search(r"strokes=\[([^\]]*)\]", line)
            current_component = {
                "component": int(component_match.group(1)),
                "pruned_closed": component_match.group(2) == "True",
                "strokes_in_order": parse_int_list(strokes_match.group(1)) if strokes_match else [],
                "pruned_strokes": parse_int_list(component_match.group(3)),
                "endpoints": [],
            }
            components.append(current_component)
            continue

        endpoint_match = CAP_GRAPH_ENDPOINT_RE.search(line)
        if endpoint_match and current_component is not None:
            sid = int(endpoint_match.group(1))
            if sid not in target_cap_ids:
                continue
            pixel = np.array([float(endpoint_match.group(3)), float(endpoint_match.group(4))], dtype=float)
            if strokes is not None and sid in strokes:
                polyline_2d, _ = recovery_stroke_polyline(
                    strokes[sid],
                    straightness_threshold=straightness_threshold,
                    resample_step=resample_step,
                    curve_min_chord_px=curve_min_chord_px,
                    curve_min_p90_chord_dev_px=curve_min_p90_chord_dev_px,
                    curve_min_dev_ratio=curve_min_dev_ratio,
                    curve_min_pca_rms_px=curve_min_pca_rms_px,
                )
                pixel = polyline_2d[0] if endpoint_match.group(2) == "start" else polyline_2d[-1]
            current_component["endpoints"].append(
                {
                    "stroke": sid,
                    "endpoint": "p0" if endpoint_match.group(2) == "start" else "p1",
                    "graph_endpoint": endpoint_match.group(2),
                    "pixel": pixel,
                    "degree": int(endpoint_match.group(5)),
                    "matches": endpoint_match.group(6),
                }
            )

    if not components:
        return None, f"No cap endpoint graph components found for cluster {cluster_id}"

    exact = [
        c
        for c in components
        if c["pruned_closed"] and set(c["pruned_strokes"]) == target_cap_ids and c["endpoints"]
    ]
    subset = [
        c
        for c in components
        if c["pruned_closed"] and target_cap_ids.issubset(set(c["pruned_strokes"])) and c["endpoints"]
    ]
    fallback = [c for c in components if c["pruned_closed"] and c["endpoints"]]
    chosen = (exact or subset or fallback or [None])[0]
    if chosen is None:
        return None, f"No closed endpoint graph component matched cap strokes {sorted(target_cap_ids)}"

    raw_endpoint_count = len(chosen["endpoints"])
    groups = snap_nearby_endpoint_pairs(chosen["endpoints"], tol)
    member_to_group = {}
    snapped_pair_count = 0
    for group_idx, group in enumerate(groups):
        group["source"] = "cap_endpoint_graph_summary"
        group["component"] = chosen["component"]
        group["one_pass_pair_snap"] = len(group["members"]) == 2
        if group["one_pass_pair_snap"]:
            snapped_pair_count += 1
        for member in group["members"]:
            member_to_group[(member["stroke"], member["graph_endpoint"])] = group_idx

    graph_edges = set()
    for group_idx, group in enumerate(groups):
        for member in group["members"]:
            for match in re.findall(r"s(\d+):(start|end)", member.get("matches", "")):
                key = (int(match[0]), match[1])
                other_group_idx = member_to_group.get(key)
                if other_group_idx is None or other_group_idx == group_idx:
                    continue
                graph_edges.add(tuple(sorted((group_idx, other_group_idx))))

    # Stroke bodies are also graph edges: each cap stroke connects its start group to its end group.
    for sid in target_cap_ids:
        start_group = member_to_group.get((sid, "start"))
        end_group = member_to_group.get((sid, "end"))
        if start_group is not None and end_group is not None and start_group != end_group:
            graph_edges.add(tuple(sorted((start_group, end_group))))

    metadata = {
        "source": str(path),
        "cluster": int(cluster_id),
        "component": int(chosen["component"]),
        "component_strokes_in_order": [int(s) for s in chosen.get("strokes_in_order", []) if int(s) in target_cap_ids],
        "pruned_strokes": [int(s) for s in chosen["pruned_strokes"]],
        "exact_pruned_strokes_match": set(chosen["pruned_strokes"]) == target_cap_ids,
        "raw_endpoint_count": int(raw_endpoint_count),
        "post_snap_endpoint_group_count": int(len(groups)),
        "one_pass_pair_snap_tolerance_pixels": float(tol),
        "one_pass_pair_snap_count": int(snapped_pair_count),
        "graph_edges_group_indices": [[int(a), int(b)] for a, b in sorted(graph_edges)],
        "stroke_group_indices": {
            str(int(sid)): {
                "start": int(member_to_group[(sid, "start")]),
                "end": int(member_to_group[(sid, "end")]),
            }
            for sid in target_cap_ids
            if (sid, "start") in member_to_group and (sid, "end") in member_to_group
        },
    }
    return groups, metadata


def load_iou_rank(path, iou_rank):
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        m = IOU_RE.search(line)
        if m and int(m.group(1)) == iou_rank:
            return {
                "iou_rank": int(m.group(1)),
                "iou": float(m.group(2)),
                "cluster_rank": int(m.group(3)),
                "cluster": int(m.group(4)),
                "score_in_source_name": parse_float_token(m.group(5)),
                "raw_line": line.strip(),
            }
    raise RuntimeError(f"Could not find iou_rank {iou_rank} in {path}")


def parse_overlay_name(path_or_name):
    name = Path(path_or_name).name
    m = OVERLAY_NAME_RE.search(name)
    if not m:
        raise RuntimeError(
            "Overlay filename must look like "
            "iou_rank_00_iou_9657_inter_..._rank_00_cluster_00_score_78855_side_bestcap_overlay.png"
        )
    return {
        "iou_rank": int(m.group(1)),
        "iou_scaled_from_filename": int(m.group(2)),
        "cluster_rank": int(m.group(3)),
        "cluster": int(m.group(4)),
        "score_in_source_name": parse_float_token(m.group(5)),
        "raw_filename": name,
    }


def find_default_overlay_image(debug_dir, rank=0):
    debug_dir = Path(debug_dir)
    patterns = [
        f"cluster_side_caps_iou_ranked/iou_rank_{rank:02d}*side_bestcap_overlay*.png",
        f"**/iou_rank_{rank:02d}*side_bestcap_overlay*.png",
        f"**/rank_{rank:02d}*side_bestcap_overlay*.png",
    ]
    for pattern in patterns:
        matches = sorted(debug_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def color_mask_rgb(image_rgb, rgb, tolerance=2):
    target = np.array(rgb, dtype=np.int16)
    diff = np.abs(image_rgb.astype(np.int16) - target.reshape(1, 1, 3))
    return (np.max(diff, axis=2) <= tolerance).astype(np.uint8) * 255


def extract_mask_vertices(mask, max_vertices=12):
    import cv2

    if np.count_nonzero(mask) == 0:
        return []
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    closed = cv2.dilate(closed, kernel, iterations=1)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    contour = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(contour, True)
    best = None
    for eps_ratio in (0.012, 0.016, 0.022, 0.03, 0.045, 0.06):
        approx = cv2.approxPolyDP(contour, eps_ratio * peri, True)
        if len(approx) >= 3:
            best = approx[:, 0, :].astype(float)
            if len(best) <= max_vertices:
                break
    if best is None:
        return []
    if len(best) > max_vertices:
        idx = np.linspace(0, len(best) - 1, max_vertices).round().astype(int)
        best = best[idx]
    centroid = np.mean(best, axis=0)
    order = np.argsort(np.arctan2(best[:, 1] - centroid[1], best[:, 0] - centroid[0]))
    return [best[i] for i in order]


def pair_overlay_cap_vertices(source_vertices, copied_vertices):
    source_vertices = [np.array(v, dtype=float) for v in source_vertices]
    copied_vertices = [np.array(v, dtype=float) for v in copied_vertices]
    if not source_vertices or not copied_vertices:
        raise RuntimeError("Could not extract both original and copied cap vertices from overlay image.")
    offset = np.mean(copied_vertices, axis=0) - np.mean(source_vertices, axis=0)
    unused = set(range(len(copied_vertices)))
    groups = []
    for idx, src in enumerate(source_vertices):
        copied = src + offset
        if unused:
            best_i = min(unused, key=lambda i: norm((src + offset) - copied_vertices[i]))
            copied = copied_vertices[best_i]
            unused.remove(best_i)
        groups.append(
            {
                "center": src,
                "copied_center": copied,
                "members": [
                    {
                        "stroke": -1,
                        "endpoint": f"overlay_vertex_{idx}",
                        "pixel": src,
                    }
                ],
                "source": "side_bestcap_overlay_png",
            }
        )
    centroid = np.mean([g["center"] for g in groups], axis=0)
    groups.sort(key=lambda g: math.atan2(g["center"][1] - centroid[1], g["center"][0] - centroid[0]))
    paired_offsets = [g["copied_center"] - g["center"] for g in groups]
    if paired_offsets:
        offset = np.median(np.array(paired_offsets), axis=0)
    return groups, offset


def load_overlay_cap_groups(overlay_path):
    import cv2

    overlay_path = Path(overlay_path)
    bgr = cv2.imread(str(overlay_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Could not read overlay image: {overlay_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    green_mask = color_mask_rgb(rgb, (0, 255, 0))
    red_mask = color_mask_rgb(rgb, (255, 0, 0))
    blue_mask = color_mask_rgb(rgb, (0, 0, 255))
    source_vertices = extract_mask_vertices(green_mask)
    copied_vertices = extract_mask_vertices(red_mask)
    groups, offset = pair_overlay_cap_vertices(source_vertices, copied_vertices)
    return groups, offset, {
        "source": str(overlay_path),
        "image_shape_hw": [int(rgb.shape[0]), int(rgb.shape[1])],
        "green_cap_pixels": int(np.count_nonzero(green_mask)),
        "red_copied_cap_pixels": int(np.count_nonzero(red_mask)),
        "blue_side_pixels": int(np.count_nonzero(blue_mask)),
        "source_vertex_count": len(source_vertices),
        "copied_vertex_count": len(copied_vertices),
        "method": "exact RGB masks: green source cap, red copied cap, blue side strokes -> contour approximation -> centroid-offset vertex pairing",
    }


def load_anchor_calibration(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    anchor = data["anchor"]
    anchor_world = np.array(anchor["world"], dtype=float)
    origin_pixel = np.array(anchor["O"]["pixel_top_left"], dtype=float)

    if "unit_uv_vectors_pixels" in anchor:
        ux = np.array(anchor["unit_uv_vectors_pixels"]["u_x"], dtype=float)
        uy = np.array(anchor["unit_uv_vectors_pixels"]["u_y"], dtype=float)
        uz = np.array(anchor["unit_uv_vectors_pixels"]["u_z"], dtype=float)
    else:
        ux = np.array(anchor["X1"]["pixel_top_left"], dtype=float) - origin_pixel
        uy = np.array(anchor["Y1"]["pixel_top_left"], dtype=float) - origin_pixel
        uz = np.array(anchor["Z1"]["pixel_top_left"], dtype=float) - origin_pixel

    basis_2x3 = np.column_stack([ux, uy, uz])
    pseudo_inverse_3x2 = np.linalg.pinv(basis_2x3)
    return {
        "raw": data,
        "anchor_world": anchor_world,
        "origin_pixel": origin_pixel,
        "basis_2x3": basis_2x3,
        "pseudo_inverse_3x2": pseudo_inverse_3x2,
    }


def find_groundtruth_calibration(calibration_path):
    calibration_path = Path(calibration_path)
    candidates = [
        calibration_path,
        calibration_path.with_name("dev_camera_anchor_calibration.json"),
        Path("blender_axonometric_dev_dataset/dev_camera_anchor_calibration.json"),
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        data = json.loads(candidate.read_text(encoding="utf-8"))
        if "object" in data and "source_cap_vertices_xyz" in data["object"]:
            return candidate, data
    return None, None


def blender_obj_import_coords(coords):
    # Blender's default OBJ importer uses Forward=-Z, Up=Y, which maps OBJ coordinates
    # into Blender world approximately as (x, -z, y). Match that conversion here so
    # the software face-ID render agrees with importing the OBJ in Blender.
    x, y, z = coords
    return np.array([x, -z, y], dtype=float)


def parse_obj_faces(path, apply_blender_axis_conversion=True):
    vertices = []
    faces = []
    path = Path(path)
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("v "):
            coords = np.array([float(v) for v in line.split()[1:4]], dtype=float)
            if apply_blender_axis_conversion:
                coords = blender_obj_import_coords(coords)
            vertices.append(coords)
        elif line.startswith("f "):
            indices = [int(tok.split("/")[0]) - 1 for tok in line.split()[1:]]
            faces.append(indices)
    if not vertices or not faces:
        raise RuntimeError(f"No vertices/faces parsed from OBJ: {path}")
    return {"path": str(path), "vertices": vertices, "faces": faces}


def face_plane(vertices, face):
    pts = [vertices[i] for i in face]
    p0 = pts[0]
    normal = None
    for i in range(1, len(pts) - 1):
        n = np.cross(pts[i] - p0, pts[i + 1] - p0)
        if norm(n) > 1e-9:
            normal = unit(n)
            break
    if normal is None:
        raise RuntimeError("Degenerate OBJ face cannot define a plane.")
    return {"point": p0, "normal": normal}


def camera_view_direction_from_anchor_basis(cal):
    _, _, vh = np.linalg.svd(cal["basis_2x3"])
    view_dir = vh[-1, :]
    return unit(view_dir)


def ray_point_for_pixel(pixel, cal):
    return pixel_to_world_min_norm(pixel, cal)


def intersect_pixel_with_plane(pixel, plane_point, plane_normal, cal):
    ray_point = ray_point_for_pixel(pixel, cal)
    ray_dir = camera_view_direction_from_anchor_basis(cal)
    denom = float(np.dot(ray_dir, plane_normal))
    if abs(denom) < 1e-9:
        raise RuntimeError("Camera ray is parallel to selected cap plane; cannot intersect stably.")
    t = float(np.dot(plane_point - ray_point, plane_normal) / denom)
    return ray_point + t * ray_dir


def point_segment_distance_2d(point, a, b):
    point = np.array(point, dtype=float)
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom == 0:
        return norm(point - a)
    t = max(0.0, min(1.0, float(np.dot(point - a, ab) / denom)))
    return norm(point - (a + t * ab))


def point_in_polygon_2d(point, polygon):
    x, y = point
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            x_intersect = (x2 - x1) * (y - y1) / ((y2 - y1) or 1e-12) + x1
            if x < x_intersect:
                inside = not inside
    return inside


def point_polygon_distance_2d(point, polygon):
    if point_in_polygon_2d(point, polygon):
        return 0.0
    return min(point_segment_distance_2d(point, polygon[i], polygon[(i + 1) % len(polygon)]) for i in range(len(polygon)))


def bbox_of_points(points):
    arr = np.array(points, dtype=float)
    return np.array([arr[:, 0].min(), arr[:, 1].min(), arr[:, 0].max(), arr[:, 1].max()], dtype=float)


def bbox_center_size(bbox):
    return np.array([(bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5]), np.array(
        [bbox[2] - bbox[0], bbox[3] - bbox[1]]
    )


def face_cap_match_score(cap_pixels, face_pixels):
    cap_bbox = bbox_of_points(cap_pixels)
    face_bbox = bbox_of_points(face_pixels)
    cap_center, cap_size = bbox_center_size(cap_bbox)
    face_center, face_size = bbox_center_size(face_bbox)
    outside_dist = np.mean([point_polygon_distance_2d(p, face_pixels) for p in cap_pixels])
    center_dist = norm(cap_center - face_center)
    size_dist = norm(cap_size - face_size)
    # Endpoint-to-face containment is the strongest term; bbox terms disambiguate large projected side faces.
    return float(outside_dist + 0.05 * center_dist + 0.02 * size_dist)


def unique_face_color(face_id):
    value = int(face_id) + 1
    return np.array(
        [
            (value * 73) % 255,
            (value * 151) % 255,
            (value * 211) % 255,
        ],
        dtype=np.uint8,
    )


def rasterize_polygon_mask(shape, polygon):
    import cv2

    mask = np.zeros(shape, dtype=np.uint8)
    pts = np.round(np.array(polygon, dtype=float)).astype(np.int32)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def render_obj_face_id_map(obj, cal, id_render_output=None):
    import cv2

    render = cal["raw"].get("render", {})
    width = int(render.get("resolution_x", 1400))
    height = int(render.get("resolution_y", 1000))
    id_map = np.zeros((height, width), dtype=np.int32)
    color_img = np.zeros((height, width, 3), dtype=np.uint8)

    camera_location = np.array(cal["raw"]["camera"]["location"], dtype=float)
    face_infos = []
    for face_idx, face in enumerate(obj["faces"]):
        verts_world = [obj["vertices"][i] for i in face]
        verts_pixel = [project_world_with_anchor(v, cal) for v in verts_world]
        center = np.mean(verts_world, axis=0)
        distance_to_camera = norm(center - camera_location)
        face_infos.append(
            {
                "face_index": face_idx,
                "face": face,
                "verts_world": verts_world,
                "verts_pixel": verts_pixel,
                "center_world": center,
                "distance_to_camera": distance_to_camera,
            }
        )

    # Draw far faces first and near faces last, approximating an orthographic ID render.
    for info in sorted(face_infos, key=lambda item: item["distance_to_camera"], reverse=True):
        polygon = np.round(np.array(info["verts_pixel"], dtype=float)).astype(np.int32)
        face_id = info["face_index"] + 1
        cv2.fillPoly(id_map, [polygon], face_id)
        cv2.fillPoly(color_img, [polygon], unique_face_color(info["face_index"]).tolist())

    if id_render_output:
        output_path = Path(id_render_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR))

    return id_map, color_img, face_infos


def cap_id_area_stats(cap_pixels, id_map, face_infos):
    mask = rasterize_polygon_mask(id_map.shape, cap_pixels)
    values = id_map[mask > 0]
    counts = np.bincount(values, minlength=len(face_infos) + 1)
    counts[0] = 0
    total_area = int(counts.sum())
    max_area = int(counts.max()) if len(counts) else 0
    threshold = max(1, int(max_area * 0.20))
    large_ids = [int(i) for i, count in enumerate(counts) if i > 0 and count >= threshold]
    entries = []
    for face_id in range(1, len(counts)):
        count = int(counts[face_id])
        if count <= 0:
            continue
        info = face_infos[face_id - 1]
        entries.append(
            {
                "face_index": int(face_id - 1),
                "area_pixels": count,
                "distance_to_camera": float(info["distance_to_camera"]),
                "large_area_candidate": int(face_id) in large_ids,
            }
        )
    entries.sort(key=lambda item: (-item["area_pixels"], item["distance_to_camera"]))
    return {
        "total_id_area_pixels": total_area,
        "max_single_face_area_pixels": max_area,
        "large_area_threshold_pixels": threshold,
        "large_face_indices": [face_id - 1 for face_id in large_ids],
        "face_area_entries": entries,
    }


def write_debug_json(path, data):
    if path is None:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return str(path)


def segment_sample_points_2d(a, b, count=5):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    if count <= 1:
        return [a]
    return [a + (b - a) * (i / float(count - 1)) for i in range(count)]


def polygon_distance_stats_2d(points, polygon):
    distances = [point_polygon_distance_2d(p, polygon) for p in points]
    return {
        "distances": distances,
        "mean_distance": float(np.mean(distances)) if distances else float("inf"),
        "max_distance": float(max(distances)) if distances else float("inf"),
    }


def try_save_support_fallback_png(path, image_shape_hw, support_polygon, side_candidates, cap_candidates, selected_side=None, selected_anchor=None):
    if path is None:
        return None
    try:
        import cv2
    except ModuleNotFoundError:
        return None

    h, w = int(image_shape_hw[0]), int(image_shape_hw[1])
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    poly = np.round(np.array(support_polygon, dtype=float)).astype(np.int32)
    cv2.polylines(canvas, [poly], isClosed=True, color=(0, 180, 255), thickness=2, lineType=cv2.LINE_AA)

    for item in side_candidates:
        color = (180, 180, 180)
        thickness = 1
        if item.get("accepted"):
            color = (0, 0, 220)
            thickness = 2
        if selected_side is not None and item["stroke"] == selected_side["stroke"]:
            color = (0, 0, 255)
            thickness = 4
        p0 = tuple(np.round(item["near_pixel"]).astype(int))
        p1 = tuple(np.round(item["far_pixel"]).astype(int))
        cv2.line(canvas, p0, p1, color, thickness, cv2.LINE_AA)

    for item in cap_candidates:
        color = (120, 120, 120)
        radius = 4
        if item.get("accepted"):
            color = (0, 180, 0)
            radius = 5
        if selected_anchor is not None and item["group_index"] == selected_anchor["group_index"]:
            color = (255, 0, 0)
            radius = 7
        p = tuple(np.round(item["pixel"]).astype(int))
        cv2.circle(canvas, p, radius, color, -1, cv2.LINE_AA)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    return str(path)


def infer_support_plane_cap_assignment(
    best,
    cap_groups,
    copied_offset,
    cal,
    oriented_sides,
    support_debug_dir=None,
    polygon_tol=15.0,
):
    if not oriented_sides:
        raise RuntimeError("Support-plane fallback needs side strokes, but none are available.")

    support_debug_dir = Path(support_debug_dir) if support_debug_dir is not None else None
    support_plane_point = np.array(best["plane_point"], dtype=float)
    support_plane_normal = np.array(best["plane_normal"], dtype=float)
    support_polygon = [np.array(p, dtype=float) for p in best["face_vertices_pixel_top_left"]]
    render = cal["raw"].get("render", {})
    image_shape_hw = (int(render.get("resolution_y", 1000)), int(render.get("resolution_x", 1400)))

    debug = {
        "method": "support_plane_anchor_foreground_fallback",
        "support_face_index": int(best["face_index"]),
        "support_plane_point": to_list(support_plane_point),
        "support_plane_normal": to_list(support_plane_normal),
        "support_polygon_2d": [to_list(p) for p in support_polygon],
        "polygon_tolerance_pixels": float(polygon_tol),
        "copied_offset": to_list(copied_offset),
        "steps": {},
    }

    support_ray_dir = camera_view_direction_from_anchor_basis(cal)
    support_denom = float(np.dot(support_ray_dir, support_plane_normal))
    if abs(support_denom) < 1e-9:
        raise RuntimeError("Camera ray is parallel to support plane; support-plane fallback is unstable.")
    debug["steps"]["01_support_plane"] = {
        "ray_support_denom": support_denom,
    }

    side_debug = []
    accepted_sides = []
    for side in oriented_sides:
        samples = segment_sample_points_2d(side["near_pixel"], side["far_pixel"], count=5)
        stats = polygon_distance_stats_2d(samples, support_polygon)
        item = {
            "stroke": int(side["stroke"]),
            "near_endpoint": side["near_endpoint"],
            "far_endpoint": side["far_endpoint"],
            "near_pixel": to_list(side["near_pixel"]),
            "far_pixel": to_list(side["far_pixel"]),
            "length_2d": norm(side["far_pixel"] - side["near_pixel"]),
            "sample_pixels": [to_list(p) for p in samples],
            "sample_distances_to_support_polygon": [float(d) for d in stats["distances"]],
            "mean_distance_to_support_polygon": float(stats["mean_distance"]),
            "max_distance_to_support_polygon": float(stats["max_distance"]),
            "accepted": bool(stats["max_distance"] <= polygon_tol),
        }
        side_debug.append(item)
        if item["accepted"]:
            accepted_sides.append((item, side))
    debug["steps"]["02_side_candidates"] = side_debug
    if not accepted_sides:
        raise RuntimeError("No side stroke lies inside/near the support face 2D projection.")

    selected_side_debug, selected_side = max(accepted_sides, key=lambda pair: pair[0]["length_2d"])
    side_near_3d = intersect_pixel_with_plane(selected_side["near_pixel"], support_plane_point, support_plane_normal, cal)
    side_far_3d = intersect_pixel_with_plane(selected_side["far_pixel"], support_plane_point, support_plane_normal, cal)
    support_side_vector = side_far_3d - side_near_3d
    if norm(support_side_vector) < 1e-9:
        raise RuntimeError("Selected support side stroke has near-zero 3D direction vector.")
    cap_normal = unit(support_side_vector)
    if float(np.dot(cal["basis_2x3"] @ cap_normal, copied_offset)) < 0:
        cap_normal *= -1.0
        support_side_vector *= -1.0
        side_near_3d, side_far_3d = side_far_3d, side_near_3d

    projected_cap_normal = cal["basis_2x3"] @ cap_normal
    denom = float(np.dot(projected_cap_normal, projected_cap_normal))
    if denom < 1e-9:
        raise RuntimeError("Reconstructed cap-plane normal projects to near-zero; cannot infer extrusion length.")
    side_length_estimates = []
    for side in oriented_sides:
        length = float(np.dot(side["vector_pixel"], projected_cap_normal) / denom)
        if length < 0:
            length = -length
        side_length_estimates.append(
            {
                "stroke": int(side["stroke"]),
                "vector_pixel": to_list(side["vector_pixel"]),
                "length_world_along_cap_normal": length,
            }
        )
    positive_lengths = [x["length_world_along_cap_normal"] for x in side_length_estimates if x["length_world_along_cap_normal"] > 1e-9]
    if not positive_lengths:
        raise RuntimeError("Could not infer a positive extrusion length from side strokes.")
    extrusion_length = float(np.median(positive_lengths))
    extrusion_vector = cap_normal * extrusion_length
    debug["steps"]["03_selected_side"] = {
        **selected_side_debug,
        "near_world_on_support_plane": to_list(side_near_3d),
        "far_world_on_support_plane": to_list(side_far_3d),
        "support_side_vector_on_support_plane": to_list(support_side_vector),
        "support_side_length_on_support_plane": norm(support_side_vector),
        "cap_normal_from_support_side": to_list(cap_normal),
        "projected_cap_normal_2d": to_list(projected_cap_normal),
        "extrusion_length_source": "median_side_stroke_projection_onto_reconstructed_cap_normal",
        "side_stroke_length_estimates": side_length_estimates,
        "extrusion_length_from_side_strokes": extrusion_length,
        "extrusion_vector_perpendicular_to_cap_plane": to_list(extrusion_vector),
    }

    cap_candidate_debug = []
    anchor_candidates = []
    for group_idx, group in enumerate(cap_groups):
        pixel = np.array(group["center"], dtype=float)
        dist = point_polygon_distance_2d(pixel, support_polygon)
        item = {
            "group_index": int(group_idx),
            "pixel": to_list(pixel),
            "distance_to_support_polygon": float(dist),
            "accepted": bool(dist <= polygon_tol),
        }
        cap_candidate_debug.append(item)
        if item["accepted"]:
            q_support = intersect_pixel_with_plane(pixel, support_plane_point, support_plane_normal, cal)
            item["world_on_support_plane"] = to_list(q_support)
            item["world_on_support_plane_z"] = float(q_support[2])
            anchor_candidates.append((item, q_support))
    debug["steps"]["04_anchor_candidates"] = cap_candidate_debug
    if not anchor_candidates:
        raise RuntimeError("No cap endpoint lies inside/near the support face 2D projection.")

    camera_location = np.array(cal["raw"]["camera"]["location"], dtype=float)
    eval_debug = []
    all_foreground_candidates = []
    for candidate_order, (anchor_item, anchor_point) in enumerate(anchor_candidates):
        source_plane_point = anchor_point
        copied_plane_point = anchor_point + extrusion_vector
        per_endpoint = []
        positive_margins = []
        fail_count = 0
        for checked_item, q_support in anchor_candidates:
            pixel = np.array(checked_item["pixel"], dtype=float)
            p_cap = intersect_pixel_with_plane(pixel, source_plane_point, cap_normal, cal)
            dist_cap = norm(p_cap - camera_location)
            dist_support = norm(q_support - camera_location)
            margin = float(dist_cap - dist_support)
            is_anchor_endpoint = int(checked_item["group_index"]) == int(anchor_item["group_index"])
            passed = bool(abs(margin) <= 1e-6) if is_anchor_endpoint else bool(margin < 0.0)
            if not passed:
                fail_count += 1
                positive_margins.append(max(0.0, margin))
            per_endpoint.append(
                {
                    "group_index": int(checked_item["group_index"]),
                    "pixel": checked_item["pixel"],
                    "world_on_support_plane": to_list(q_support),
                    "world_on_candidate_cap_plane": to_list(p_cap),
                    "distance_to_camera_support": float(dist_support),
                    "distance_to_camera_cap": float(dist_cap),
                    "foreground_margin_cap_minus_support": margin,
                    "is_anchor_endpoint": bool(is_anchor_endpoint),
                    "foreground_pass": passed,
                }
            )
        item = {
            "candidate_order": int(candidate_order),
            "group_index": int(anchor_item["group_index"]),
            "anchor_pixel": anchor_item["pixel"],
            "anchor_world_on_support_plane": to_list(anchor_point),
            "checked_count": int(len(per_endpoint)),
            "pass_count": int(len(per_endpoint) - fail_count),
            "fail_count": int(fail_count),
            "all_foreground": bool(fail_count == 0),
            "distance_score_sum_positive_margin": float(sum(positive_margins)),
            "distance_score_max_margin": float(max([e["foreground_margin_cap_minus_support"] for e in per_endpoint], default=0.0)),
            "source_plane_point": to_list(source_plane_point),
            "copied_plane_point": to_list(copied_plane_point),
            "per_checked_endpoint": per_endpoint,
        }
        eval_debug.append(item)
        if item["all_foreground"]:
            all_foreground_candidates.append(item)
    debug["steps"]["05_anchor_foreground_eval"] = eval_debug

    if all_foreground_candidates:
        selected_anchor_eval = all_foreground_candidates[0]
        selection_reason = "first_all_checked_points_foreground"
    else:
        selected_anchor_eval = min(
            eval_debug,
            key=lambda item: (item["distance_score_sum_positive_margin"], item["distance_score_max_margin"], item["candidate_order"]),
        )
        selection_reason = "min_positive_foreground_margin"

    selected_anchor_point = np.array(selected_anchor_eval["anchor_world_on_support_plane"], dtype=float)
    source_plane_point = selected_anchor_point
    copied_plane_point = selected_anchor_point + extrusion_vector
    local_x = np.cross(support_plane_normal, cap_normal)
    if norm(local_x) < 1e-9:
        local_x = np.cross(camera_view_direction_from_anchor_basis(cal), cap_normal)
    local_x = unit(local_x)
    local_y = unit(np.cross(cap_normal, local_x))
    debug["steps"]["06_selected_anchor"] = {
        "selection_reason": selection_reason,
        "selected_anchor": selected_anchor_eval,
        "source_cap_plane_point": to_list(source_plane_point),
        "copied_cap_plane_point": to_list(copied_plane_point),
        "cap_plane_normal": to_list(cap_normal),
        "local_x": to_list(local_x),
        "local_y": to_list(local_y),
    }

    if support_debug_dir is not None:
        write_debug_json(support_debug_dir / "support_fallback_steps.json", debug)
        for step_name, step_data in debug["steps"].items():
            write_debug_json(support_debug_dir / f"{step_name}.json", step_data)
        png_path = try_save_support_fallback_png(
            support_debug_dir / "support_fallback_overview.png",
            image_shape_hw,
            support_polygon,
            side_debug,
            cap_candidate_debug,
            selected_side_debug,
            selected_anchor_eval,
        )
        debug["overview_png"] = png_path
        write_debug_json(support_debug_dir / "support_fallback_steps.json", debug)

    return {
        "source_plane_point": source_plane_point,
        "copied_plane_point": copied_plane_point,
        "plane_normal": cap_normal,
        "extrusion_vector": extrusion_vector,
        "extrusion_length": norm(extrusion_vector),
        "support_debug": debug,
    }


def infer_obj_cap_plane_assignment(
    cap_groups,
    copied_offset,
    obj_path,
    cal,
    id_render_output=None,
    normal_side_angle_tol_degrees=20.0,
    oriented_sides=None,
    support_debug_dir=None,
    support_polygon_tol=15.0,
):
    obj = parse_obj_faces(obj_path)
    source_pixels = [g["center"] for g in cap_groups]
    copied_pixels = [g.get("copied_center", g["center"] + copied_offset) for g in cap_groups]
    id_map, _, face_infos = render_obj_face_id_map(obj, cal, id_render_output)
    source_stats = cap_id_area_stats(source_pixels, id_map, face_infos)
    copied_stats = cap_id_area_stats(copied_pixels, id_map, face_infos)
    cap_stats = {"source_cap": source_stats, "copied_cap": copied_stats}

    matched_cap = (
        "source_cap"
        if source_stats["total_id_area_pixels"] >= copied_stats["total_id_area_pixels"]
        else "copied_cap"
    )
    selected_stats = cap_stats[matched_cap]
    large_entries = [
        entry for entry in selected_stats["face_area_entries"] if entry["large_area_candidate"]
    ]
    if not large_entries:
        large_entries = selected_stats["face_area_entries"]
    if not large_entries:
        raise RuntimeError(
            "No OBJ face ID overlaps either detected cap range. Check camera calibration and OBJ coordinate system."
        )
    # Among faces with substantial cap-area coverage, pick the one closest to the camera.
    best_entry = min(large_entries, key=lambda item: item["distance_to_camera"])
    best_face_idx = best_entry["face_index"]
    best_info = face_infos[best_face_idx]
    best_face = obj["faces"][best_face_idx]
    plane = face_plane(obj["vertices"], best_face)
    best = {
        "cap_name": matched_cap,
        "face_index": best_face_idx,
        "score": float(best_entry["area_pixels"]),
        "face_vertex_indices": best_face,
        "face_vertices_world": best_info["verts_world"],
        "face_vertices_pixel_top_left": best_info["verts_pixel"],
        "plane_point": plane["point"],
        "plane_normal": plane["normal"],
        "distance_to_camera": best_entry["distance_to_camera"],
    }
    top_candidates = selected_stats["face_area_entries"][:6]
    proj_normal = cal["basis_2x3"] @ best["plane_normal"]
    if norm(proj_normal) < 1e-9:
        raise RuntimeError("Selected OBJ cap plane normal projects to near-zero; side stroke length is unstable.")

    copied_offset_norm = norm(copied_offset)
    if copied_offset_norm < 1e-9:
        raise RuntimeError("Detected source->copied cap offset is near-zero; cannot infer extrusion.")

    proj_normal_unit = unit(proj_normal)
    copied_offset_unit = unit(copied_offset)
    normal_side_cos = max(-1.0, min(1.0, float(np.dot(proj_normal_unit, copied_offset_unit))))
    normal_side_angle_degrees = float(math.degrees(math.acos(normal_side_cos)))
    normal_side_axis_angle_degrees = min(normal_side_angle_degrees, 180.0 - normal_side_angle_degrees)
    normal_side_aligned = normal_side_axis_angle_degrees <= float(normal_side_angle_tol_degrees)

    signed_normal = best["plane_normal"].copy()
    if float(np.dot(cal["basis_2x3"] @ signed_normal, copied_offset)) < 0:
        signed_normal *= -1.0
    projected_signed_normal = cal["basis_2x3"] @ signed_normal
    if normal_side_aligned:
        # Orthogonal extrusion: the side-stroke offset is explained by moving along the cap normal.
        extrusion_length = float(
            np.dot(copied_offset, projected_signed_normal)
            / np.dot(projected_signed_normal, projected_signed_normal)
        )
        extrusion_vector = signed_normal * extrusion_length
        extrusion_method = "cap_normal_projection"
        support_fallback = None
    else:
        support_fallback = infer_support_plane_cap_assignment(
            best,
            cap_groups,
            copied_offset,
            cal,
            oriented_sides or [],
            support_debug_dir=support_debug_dir,
            polygon_tol=support_polygon_tol,
        )
        signed_normal = np.array(support_fallback["plane_normal"], dtype=float)
        projected_signed_normal = cal["basis_2x3"] @ signed_normal
        extrusion_vector = np.array(support_fallback["extrusion_vector"], dtype=float)
        extrusion_length = float(support_fallback["extrusion_length"])
        extrusion_method = "support_plane_anchor_foreground_fallback"

    face_plane_point = best["plane_point"]
    face_plane_normal = signed_normal
    if support_fallback is not None:
        source_plane_point = np.array(support_fallback["source_plane_point"], dtype=float)
        copied_plane_point = np.array(support_fallback["copied_plane_point"], dtype=float)
    elif best["cap_name"] == "source_cap":
        source_plane_point = face_plane_point
        copied_plane_point = face_plane_point + extrusion_vector
    else:
        copied_plane_point = face_plane_point
        source_plane_point = face_plane_point - extrusion_vector

    return {
        "obj_file": str(obj_path),
        "matched_cap": best["cap_name"],
        "matched_face_index": int(best["face_index"]),
        "selection_method": "single_face_id_render_cap_area_then_nearest_camera_face",
        "id_render_output": str(id_render_output) if id_render_output else None,
        "matched_face_area_pixels": int(best["score"]),
        "matched_face_distance_to_camera": float(best["distance_to_camera"]),
        "match_quality_warning": (
            "No rendered OBJ face ID overlaps the selected cap enough; check that OBJ and calibration "
            "use the same Blender world coordinate system."
            if best["score"] <= 0
            else (
                "Selected OBJ cap normal projection is not aligned with the detected side-stroke offset; "
                "using support-plane anchor foreground fallback instead of normal extrusion."
                if not normal_side_aligned
                else None
            )
        ),
        "cap_id_area_stats": cap_stats,
        "top_candidate_scores": [
            {
                "face_index": int(c["face_index"]),
                "area_pixels": int(c["area_pixels"]),
                "distance_to_camera": float(c["distance_to_camera"]),
                "large_area_candidate": bool(c["large_area_candidate"]),
            }
            for c in top_candidates
        ],
        "face_vertex_indices": [int(i) for i in best["face_vertex_indices"]],
        "face_vertices_world": [to_list(v) for v in best["face_vertices_world"]],
        "face_vertices_pixel_top_left": [to_list(v) for v in best["face_vertices_pixel_top_left"]],
        "face_plane_point": to_list(face_plane_point),
        "face_plane_normal_unoriented": to_list(best["plane_normal"]),
        "normal_projection_pixel": to_list(proj_normal),
        "normal_projection_oriented_pixel": to_list(projected_signed_normal),
        "side_offset_angle_to_face_normal_projection_degrees": normal_side_angle_degrees,
        "side_offset_axis_angle_to_face_normal_projection_degrees": normal_side_axis_angle_degrees,
        "side_offset_normal_alignment_tolerance_degrees": float(normal_side_angle_tol_degrees),
        "side_offset_aligned_with_face_normal_projection": bool(normal_side_aligned),
        "extrusion_inference_method": extrusion_method,
        "support_plane_fallback": support_fallback["support_debug"] if support_fallback is not None else None,
        "source_plane_point": to_list(source_plane_point),
        "copied_plane_point": to_list(copied_plane_point),
        "plane_normal_source_to_copied": to_list(face_plane_normal),
        "extrusion_length_source_to_copied": extrusion_length,
        "extrusion_plane_offset_source_to_copied": float(np.dot(extrusion_vector, face_plane_normal)),
        "extrusion_vector_source_to_copied": to_list(extrusion_vector),
        "unit_extrusion_direction_source_to_copied": to_list(unit(extrusion_vector)),
    }


def project_world_with_anchor(world, cal):
    return cal["origin_pixel"] + cal["basis_2x3"] @ (np.array(world, dtype=float) - cal["anchor_world"])


def pixel_to_world_on_z(pixel, z_value, cal):
    pixel = np.array(pixel, dtype=float)
    z_value = float(z_value)
    anchor = cal["anchor_world"]
    basis = cal["basis_2x3"]
    rhs = pixel - cal["origin_pixel"] - basis[:, 2] * (z_value - anchor[2])
    xy_delta = np.linalg.solve(basis[:, :2], rhs)
    return np.array([anchor[0] + xy_delta[0], anchor[1] + xy_delta[1], z_value], dtype=float)


def mean_nearest_distance(points, candidates):
    if not points or not candidates:
        return float("inf")
    distances = []
    candidate_arr = [np.array(c, dtype=float) for c in candidates]
    for point in points:
        p = np.array(point, dtype=float)
        distances.append(min(norm(p - c) for c in candidate_arr))
    return float(np.mean(distances))


def infer_gt_cap_assignment(cap_groups, gt_data, cal):
    obj = gt_data["object"]
    source_gt_pixels = [project_world_with_anchor(v, cal) for v in obj["source_cap_vertices_xyz"]]
    copied_gt_pixels = [project_world_with_anchor(v, cal) for v in obj["copied_cap_vertices_xyz"]]
    detected_source_pixels = [g["center"] for g in cap_groups]
    source_to_gt_source = mean_nearest_distance(detected_source_pixels, source_gt_pixels)
    source_to_gt_copied = mean_nearest_distance(detected_source_pixels, copied_gt_pixels)
    source_plane = obj.get("source_cap_plane")
    copied_plane = obj.get("copied_cap_plane")

    if source_to_gt_copied < source_to_gt_source:
        detected_source_z = obj["copied_cap_z"]
        detected_copied_z = obj["source_cap_z"]
        detected_source_plane = copied_plane
        detected_copied_plane = source_plane
        detected_source_label = "gt_copied_cap"
        detected_copied_label = "gt_source_cap"
    else:
        detected_source_z = obj["source_cap_z"]
        detected_copied_z = obj["copied_cap_z"]
        detected_source_plane = source_plane
        detected_copied_plane = copied_plane
        detected_source_label = "gt_source_cap"
        detected_copied_label = "gt_copied_cap"

    result = {
        "source_to_gt_source_mean_pixel_distance": source_to_gt_source,
        "source_to_gt_copied_mean_pixel_distance": source_to_gt_copied,
        "detected_source_cap_matches": detected_source_label,
        "detected_copied_cap_matches": detected_copied_label,
        "detected_source_z": float(detected_source_z),
        "detected_copied_z": float(detected_copied_z),
        "gt_extrusion_direction_world": obj.get("extrusion_direction_world"),
    }
    if detected_source_plane is not None and detected_copied_plane is not None:
        result["detected_source_plane_point"] = detected_source_plane["point"]
        result["detected_source_plane_normal"] = detected_source_plane["normal"]
        result["detected_copied_plane_point"] = detected_copied_plane["point"]
        result["detected_copied_plane_normal"] = detected_copied_plane["normal"]
    return result


def pixel_to_world_min_norm(pixel, cal):
    pixel = np.array(pixel, dtype=float)
    local_xyz = cal["pseudo_inverse_3x2"] @ (pixel - cal["origin_pixel"])
    return cal["anchor_world"] + local_xyz


def image_vector_to_world_min_norm(vector, cal):
    return cal["pseudo_inverse_3x2"] @ np.array(vector, dtype=float)


def norm(v):
    return float(np.linalg.norm(v))


def unit(v):
    length = norm(v)
    if length == 0:
        return np.array(v, dtype=float)
    return np.array(v, dtype=float) / length


def resample_polyline_by_step(points, step=15.0):
    pts = np.asarray(points, dtype=float)
    if len(pts) < 2:
        return pts.copy()
    step = float(step)
    if step <= 0:
        return pts.copy()
    segs = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    total = float(np.sum(segs))
    if total <= 1e-9:
        return np.vstack([pts[0], pts[-1]])
    target = np.arange(0.0, total + 1e-9, step)
    if target[-1] < total:
        target = np.append(target, total)
    out = [pts[0]]
    accumulated = 0.0
    seg_i = 0
    for dist in target[1:]:
        while seg_i < len(segs) and accumulated + segs[seg_i] < dist:
            accumulated += segs[seg_i]
            seg_i += 1
        if seg_i >= len(segs):
            out.append(pts[-1])
            continue
        seg_len = segs[seg_i]
        if seg_len <= 1e-9:
            out.append(pts[seg_i + 1])
            continue
        t = (dist - accumulated) / seg_len
        out.append(pts[seg_i] * (1.0 - t) + pts[seg_i + 1] * t)
    return np.asarray(out, dtype=float)


def stroke_curve_metrics(points, straightness=None):
    pts = np.asarray(points, dtype=float)
    if len(pts) < 2:
        return {
            "point_count": int(len(pts)),
            "arc": 0.0,
            "chord": 0.0,
            "straightness": 1.0 if straightness is None else float(straightness),
            "p90_chord_deviation": 0.0,
            "max_chord_deviation": 0.0,
            "deviation_ratio": 0.0,
            "pca_rms_error": 0.0,
        }

    segs = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    arc = float(np.sum(segs))
    chord_vec = pts[-1] - pts[0]
    chord = norm(chord_vec)
    measured_straightness = 1.0 if arc <= 1e-9 else float(chord / arc)
    if straightness is not None:
        measured_straightness = float(straightness)

    if chord <= 1e-9:
        chord_devs = np.linalg.norm(pts - pts[0], axis=1)
    else:
        chord_dir = chord_vec / chord
        rel = pts - pts[0]
        signed = rel[:, 0] * chord_dir[1] - rel[:, 1] * chord_dir[0]
        chord_devs = np.abs(signed)

    center = np.mean(pts, axis=0)
    X = pts - center
    if len(pts) >= 2:
        cov = X.T @ X / max(len(pts), 1)
        vals, vecs = np.linalg.eigh(cov)
        direction = unit(vecs[:, int(np.argmax(vals))])
        normal = np.array([-direction[1], direction[0]], dtype=float)
        pca_distances = X @ normal
        pca_rms = float(np.sqrt(np.mean(pca_distances * pca_distances)))
    else:
        pca_rms = 0.0

    p90 = float(np.percentile(chord_devs, 90)) if len(chord_devs) else 0.0
    max_dev = float(np.max(chord_devs)) if len(chord_devs) else 0.0
    ratio = 0.0 if chord <= 1e-9 else float(p90 / chord)
    return {
        "point_count": int(len(pts)),
        "arc": float(arc),
        "chord": float(chord),
        "straightness": float(measured_straightness),
        "p90_chord_deviation": p90,
        "max_chord_deviation": max_dev,
        "deviation_ratio": ratio,
        "pca_rms_error": pca_rms,
    }


def classify_stroke_curve(
    stroke,
    straightness_threshold=0.9,
    curve_min_chord_px=30.0,
    curve_min_p90_chord_dev_px=5.0,
    curve_min_dev_ratio=0.04,
    curve_min_pca_rms_px=2.0,
):
    points = np.asarray(stroke.get("points", [stroke["p0"], stroke["p1"]]), dtype=float)
    metrics = stroke_curve_metrics(points, straightness=stroke.get("straightness", None))
    metrics["straightness_threshold"] = float(straightness_threshold)
    metrics["curve_min_chord_px"] = float(curve_min_chord_px)
    metrics["curve_min_p90_chord_dev_px"] = float(curve_min_p90_chord_dev_px)
    metrics["curve_min_dev_ratio"] = float(curve_min_dev_ratio)
    metrics["curve_min_pca_rms_px"] = float(curve_min_pca_rms_px)

    checks = {
        "enough_points": len(points) >= 3,
        "chord_ge_min": metrics["chord"] >= float(curve_min_chord_px),
        "straightness_below_threshold": metrics["straightness"] < float(straightness_threshold),
        "p90_dev_ge_min": metrics["p90_chord_deviation"] >= float(curve_min_p90_chord_dev_px),
        "dev_ratio_ge_min": metrics["deviation_ratio"] >= float(curve_min_dev_ratio),
        "pca_rms_ge_min": metrics["pca_rms_error"] >= float(curve_min_pca_rms_px),
    }
    is_curve = all(checks.values())
    metrics["checks"] = checks
    metrics["decision"] = "curve_global_bend" if is_curve else "straight_or_hand_jitter"
    return bool(is_curve), metrics


def simplify_stroke_points(
    stroke,
    straightness_threshold=0.9,
    resample_step=15.0,
    curve_min_chord_px=30.0,
    curve_min_p90_chord_dev_px=5.0,
    curve_min_dev_ratio=0.04,
    curve_min_pca_rms_px=2.0,
    return_metrics=False,
):
    points = np.asarray(stroke.get("points", [stroke["p0"], stroke["p1"]]), dtype=float)
    is_curve, metrics = classify_stroke_curve(
        stroke,
        straightness_threshold=straightness_threshold,
        curve_min_chord_px=curve_min_chord_px,
        curve_min_p90_chord_dev_px=curve_min_p90_chord_dev_px,
        curve_min_dev_ratio=curve_min_dev_ratio,
        curve_min_pca_rms_px=curve_min_pca_rms_px,
    )
    if len(points) < 2:
        line = np.vstack([stroke["p0"], stroke["p1"]])
        return (line, False, metrics) if return_metrics else (line, False)
    if not is_curve:
        line = np.vstack([points[0], points[-1]])
        return (line, False, metrics) if return_metrics else (line, False)

    resampled = resample_polyline_by_step(points, step=resample_step)
    if len(resampled) < 2:
        line = np.vstack([points[0], points[-1]])
        return (line, False, metrics) if return_metrics else (line, False)

    center = np.mean(resampled, axis=0)
    X = resampled - center
    cov = X.T @ X / max(len(resampled), 1)
    vals, vecs = np.linalg.eigh(cov)
    direction = vecs[:, int(np.argmax(vals))]
    direction = unit(direction)
    if norm(direction) < 1e-9:
        line = np.vstack([points[0], points[-1]])
        return (line, False, metrics) if return_metrics else (line, False)
    proj = X @ direction
    line = np.vstack([center + direction * float(np.min(proj)), center + direction * float(np.max(proj))])
    return (line, True, metrics) if return_metrics else (line, True)


def recovery_stroke_endpoints(
    stroke,
    straightness_threshold=0.9,
    resample_step=15.0,
    curve_min_chord_px=30.0,
    curve_min_p90_chord_dev_px=5.0,
    curve_min_dev_ratio=0.04,
    curve_min_pca_rms_px=2.0,
    return_metrics=False,
):
    line_points, is_curved = simplify_stroke_points(
        stroke,
        straightness_threshold=straightness_threshold,
        resample_step=resample_step,
        curve_min_chord_px=curve_min_chord_px,
        curve_min_p90_chord_dev_px=curve_min_p90_chord_dev_px,
        curve_min_dev_ratio=curve_min_dev_ratio,
        curve_min_pca_rms_px=curve_min_pca_rms_px,
        return_metrics=False,
    )
    if return_metrics:
        _, metrics = classify_stroke_curve(
            stroke,
            straightness_threshold=straightness_threshold,
            curve_min_chord_px=curve_min_chord_px,
            curve_min_p90_chord_dev_px=curve_min_p90_chord_dev_px,
            curve_min_dev_ratio=curve_min_dev_ratio,
            curve_min_pca_rms_px=curve_min_pca_rms_px,
        )
        return line_points[0], line_points[1], is_curved, metrics
    return line_points[0], line_points[1], is_curved


def recovery_stroke_polyline(
    stroke,
    straightness_threshold=0.9,
    resample_step=15.0,
    curve_min_chord_px=30.0,
    curve_min_p90_chord_dev_px=5.0,
    curve_min_dev_ratio=0.04,
    curve_min_pca_rms_px=2.0,
    return_metrics=False,
):
    points = np.asarray(stroke.get("points", [stroke["p0"], stroke["p1"]]), dtype=float)
    if len(points) < 2:
        metrics = stroke_curve_metrics(points, straightness=stroke.get("straightness", None))
        line = np.vstack([stroke["p0"], stroke["p1"]])
        return (line, False, metrics) if return_metrics else (line, False)
    is_curve, metrics = classify_stroke_curve(
        stroke,
        straightness_threshold=straightness_threshold,
        curve_min_chord_px=curve_min_chord_px,
        curve_min_p90_chord_dev_px=curve_min_p90_chord_dev_px,
        curve_min_dev_ratio=curve_min_dev_ratio,
        curve_min_pca_rms_px=curve_min_pca_rms_px,
    )
    if not is_curve:
        line = np.vstack([points[0], points[-1]])
        return (line, False, metrics) if return_metrics else (line, False)
    polyline = resample_polyline_by_step(points, step=resample_step)
    if len(polyline) < 2:
        line = np.vstack([points[0], points[-1]])
        return (line, False, metrics) if return_metrics else (line, False)
    polyline[0] = points[0]
    polyline[-1] = points[-1]
    return (polyline, True, metrics) if return_metrics else (polyline, True)


def orient_cap_stroke_polyline(polyline, start_group, end_group, desired_start_group=None, previous_point=None):
    poly = np.asarray(polyline, dtype=float)
    if desired_start_group is not None:
        if start_group == desired_start_group and end_group != desired_start_group:
            return poly, start_group, end_group
        if end_group == desired_start_group and start_group != desired_start_group:
            return poly[::-1].copy(), end_group, start_group
    if previous_point is not None:
        previous_point = np.asarray(previous_point, dtype=float)
        if norm(poly[0] - previous_point) <= norm(poly[-1] - previous_point):
            return poly, start_group, end_group
        return poly[::-1].copy(), end_group, start_group
    return poly, start_group, end_group


def build_cap_stroke_loop_items(
    cap_endpoint_source,
    strokes,
    straightness_threshold=0.9,
    resample_step=15.0,
    curve_min_chord_px=30.0,
    curve_min_p90_chord_dev_px=5.0,
    curve_min_dev_ratio=0.04,
    curve_min_pca_rms_px=2.0,
):
    if not cap_endpoint_source or strokes is None:
        return []
    stroke_order = [
        int(s)
        for s in cap_endpoint_source.get("component_strokes_in_order", cap_endpoint_source.get("pruned_strokes", []))
        if int(s) in strokes
    ]
    if not stroke_order:
        return []

    stroke_groups = cap_endpoint_source.get("stroke_group_indices", {})
    items = []
    previous_end_group = None
    previous_end_point = None
    for idx, sid in enumerate(stroke_order):
        group_info = stroke_groups.get(str(int(sid)))
        if group_info is None:
            continue
        polyline_2d, is_curved, curve_metrics = recovery_stroke_polyline(
            strokes[sid],
            straightness_threshold=straightness_threshold,
            resample_step=resample_step,
            curve_min_chord_px=curve_min_chord_px,
            curve_min_p90_chord_dev_px=curve_min_p90_chord_dev_px,
            curve_min_dev_ratio=curve_min_dev_ratio,
            curve_min_pca_rms_px=curve_min_pca_rms_px,
            return_metrics=True,
        )
        start_group = int(group_info["start"])
        end_group = int(group_info["end"])
        if idx == 0 and len(stroke_order) > 1:
            next_group_info = stroke_groups.get(str(int(stroke_order[1])))
            if next_group_info is not None:
                next_groups = {int(next_group_info["start"]), int(next_group_info["end"])}
                if start_group in next_groups and end_group not in next_groups:
                    polyline_2d = polyline_2d[::-1].copy()
                    start_group, end_group = end_group, start_group
        polyline_2d, start_group, end_group = orient_cap_stroke_polyline(
            polyline_2d,
            start_group,
            end_group,
            desired_start_group=previous_end_group,
            previous_point=previous_end_point,
        )
        previous_end_group = end_group
        previous_end_point = polyline_2d[-1]
        items.append(
            {
                "stroke": int(sid),
                "straightness": float(strokes[sid].get("straightness", 1.0)),
                "is_curved": bool(is_curved),
                "curve_metrics": curve_metrics,
                "start_group_index": int(start_group),
                "end_group_index": int(end_group),
                "polyline_2d": polyline_2d,
            }
        )
    return items


def normalize_cap_stroke_loop_items(
    cap_stroke_loop_items,
    strokes,
    straightness_threshold=0.9,
    resample_step=15.0,
    curve_min_chord_px=30.0,
    curve_min_p90_chord_dev_px=5.0,
    curve_min_dev_ratio=0.04,
    curve_min_pca_rms_px=2.0,
):
    if not cap_stroke_loop_items or strokes is None:
        return cap_stroke_loop_items
    normalized = []
    for item in cap_stroke_loop_items:
        sid = int(item["stroke"])
        polyline_2d = np.asarray(item["polyline_2d"], dtype=float)
        normalized_item = dict(item)
        stroke = strokes.get(sid)
        if stroke is not None:
            recomputed_polyline, recomputed_is_curved, curve_metrics = recovery_stroke_polyline(
                stroke,
                straightness_threshold=straightness_threshold,
                resample_step=resample_step,
                curve_min_chord_px=curve_min_chord_px,
                curve_min_p90_chord_dev_px=curve_min_p90_chord_dev_px,
                curve_min_dev_ratio=curve_min_dev_ratio,
                curve_min_pca_rms_px=curve_min_pca_rms_px,
                return_metrics=True,
            )
            normalized_item["curve_metrics"] = curve_metrics
            recomputed_polyline = np.asarray(recomputed_polyline, dtype=float)
            if recomputed_is_curved and len(recomputed_polyline) > len(polyline_2d):
                if len(polyline_2d) >= 2:
                    same_dir = norm(recomputed_polyline[0] - polyline_2d[0]) + norm(recomputed_polyline[-1] - polyline_2d[-1])
                    flip_dir = norm(recomputed_polyline[0] - polyline_2d[-1]) + norm(recomputed_polyline[-1] - polyline_2d[0])
                    if flip_dir < same_dir:
                        recomputed_polyline = recomputed_polyline[::-1].copy()
                normalized_item["polyline_2d"] = recomputed_polyline
                normalized_item["is_curved"] = True
            elif not recomputed_is_curved:
                normalized_item["is_curved"] = False
        normalized.append(normalized_item)
    return normalized


def reverse_cap_stroke_loop_item(item):
    reversed_item = dict(item)
    reversed_item["polyline_2d"] = np.asarray(item["polyline_2d"], dtype=float)[::-1].copy()
    reversed_item["start_group_index"] = int(item["end_group_index"])
    reversed_item["end_group_index"] = int(item["start_group_index"])
    return reversed_item


def component_is_simple_closed_loop(cap_stroke_loop_items):
    if not cap_stroke_loop_items:
        return False
    degree = {}
    for item in cap_stroke_loop_items:
        a = int(item["start_group_index"])
        b = int(item["end_group_index"])
        if a == b:
            degree[a] = degree.get(a, 0) + 2
            continue
        degree[a] = degree.get(a, 0) + 1
        degree[b] = degree.get(b, 0) + 1
    return bool(degree) and all(v == 2 for v in degree.values())


def order_cap_stroke_loop_items(cap_stroke_loop_items):
    if len(cap_stroke_loop_items) <= 1:
        return cap_stroke_loop_items
    if not component_is_simple_closed_loop(cap_stroke_loop_items):
        return cap_stroke_loop_items

    indexed_items = [(idx, item) for idx, item in enumerate(cap_stroke_loop_items)]
    node_to_items = {}
    for idx, item in indexed_items:
        node_to_items.setdefault(int(item["start_group_index"]), []).append(idx)
        node_to_items.setdefault(int(item["end_group_index"]), []).append(idx)

    start_idx = 0
    start_item = cap_stroke_loop_items[start_idx]

    def attempt(first_item):
        ordered = [first_item]
        used = {start_idx}
        start_group = int(first_item["start_group_index"])
        current_group = int(first_item["end_group_index"])
        while len(ordered) < len(cap_stroke_loop_items):
            candidates = [idx for idx in node_to_items.get(current_group, []) if idx not in used]
            if not candidates:
                return None
            next_idx = min(candidates)
            next_item = cap_stroke_loop_items[next_idx]
            if int(next_item["start_group_index"]) == current_group:
                oriented = next_item
            elif int(next_item["end_group_index"]) == current_group:
                oriented = reverse_cap_stroke_loop_item(next_item)
            else:
                return None
            ordered.append(oriented)
            used.add(next_idx)
            current_group = int(oriented["end_group_index"])
        if current_group != start_group:
            return None
        return ordered

    ordered = attempt(start_item)
    if ordered is not None:
        return ordered
    reversed_start = reverse_cap_stroke_loop_item(start_item)
    ordered = attempt(reversed_start)
    return ordered if ordered is not None else cap_stroke_loop_items


def flatten_cap_stroke_loop_polyline_points(cap_stroke_loop_items):
    flat_points = []
    for item_idx, item in enumerate(cap_stroke_loop_items):
        poly = np.asarray(item["polyline_2d"], dtype=float)
        if len(poly) == 0:
            continue
        for point_idx, point in enumerate(poly):
            if flat_points and point_idx == 0 and points_close_2d(flat_points[-1], point):
                continue
            if (
                item_idx == len(cap_stroke_loop_items) - 1
                and point_idx == len(poly) - 1
                and flat_points
                and points_close_2d(flat_points[0], point)
            ):
                continue
            flat_points.append(np.asarray(point, dtype=float))
    return flat_points


def polygon_signed_area_2d(points):
    if len(points) < 3:
        return 0.0
    area2 = 0.0
    for idx in range(len(points)):
        p = np.asarray(points[idx], dtype=float)
        q = np.asarray(points[(idx + 1) % len(points)], dtype=float)
        area2 += float(p[0] * q[1] - q[0] * p[1])
    return 0.5 * area2


def orient2d(a, b, c):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def point_on_segment_2d(point, a, b, tol=1e-6):
    point = np.asarray(point, dtype=float)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if abs(orient2d(a, b, point)) > tol:
        return False
    min_x, max_x = sorted((float(a[0]), float(b[0])))
    min_y, max_y = sorted((float(a[1]), float(b[1])))
    return (
        min_x - tol <= float(point[0]) <= max_x + tol
        and min_y - tol <= float(point[1]) <= max_y + tol
    )


def segments_intersect_2d(a0, a1, b0, b1, tol=1e-6):
    if (
        points_close_2d(a0, b0, tol)
        or points_close_2d(a0, b1, tol)
        or points_close_2d(a1, b0, tol)
        or points_close_2d(a1, b1, tol)
    ):
        return False
    o1 = orient2d(a0, a1, b0)
    o2 = orient2d(a0, a1, b1)
    o3 = orient2d(b0, b1, a0)
    o4 = orient2d(b0, b1, a1)
    if ((o1 > tol and o2 < -tol) or (o1 < -tol and o2 > tol)) and ((o3 > tol and o4 < -tol) or (o3 < -tol and o4 > tol)):
        return True
    if abs(o1) <= tol and point_on_segment_2d(b0, a0, a1, tol):
        return True
    if abs(o2) <= tol and point_on_segment_2d(b1, a0, a1, tol):
        return True
    if abs(o3) <= tol and point_on_segment_2d(a0, b0, b1, tol):
        return True
    if abs(o4) <= tol and point_on_segment_2d(a1, b0, b1, tol):
        return True
    return False


def find_polyline_loop_self_intersections(points):
    count = len(points)
    if count < 4:
        return []
    intersections = []
    segments = [(points[idx], points[(idx + 1) % count]) for idx in range(count)]
    for i in range(count):
        a0, a1 = segments[i]
        for j in range(i + 1, count):
            if j == i:
                continue
            if j == (i + 1) % count or i == (j + 1) % count:
                continue
            if i == 0 and j == count - 1:
                continue
            b0, b1 = segments[j]
            if segments_intersect_2d(a0, a1, b0, b1):
                intersections.append([int(i), int(j)])
    return intersections


def evaluate_cap_stroke_loop_order(cap_stroke_loop_items):
    points = flatten_cap_stroke_loop_polyline_points(cap_stroke_loop_items)
    endpoint_gap_sum = 0.0
    max_endpoint_gap = 0.0
    if cap_stroke_loop_items:
        for idx, item in enumerate(cap_stroke_loop_items):
            current_end = np.asarray(item["polyline_2d"], dtype=float)[-1]
            next_start = np.asarray(cap_stroke_loop_items[(idx + 1) % len(cap_stroke_loop_items)]["polyline_2d"], dtype=float)[0]
            gap = norm(current_end - next_start)
            endpoint_gap_sum += gap
            max_endpoint_gap = max(max_endpoint_gap, float(gap))
    intersections = find_polyline_loop_self_intersections(points)
    area = polygon_signed_area_2d(points)
    return {
        "flat_point_count": int(len(points)),
        "self_intersection_pairs": intersections,
        "self_intersection_count": int(len(intersections)),
        "endpoint_gap_sum": float(endpoint_gap_sum),
        "max_endpoint_gap": float(max_endpoint_gap),
        "signed_area": float(area),
        "abs_area": float(abs(area)),
        "stroke_order": [int(item["stroke"]) for item in cap_stroke_loop_items],
    }


def apply_two_opt_to_cap_stroke_loop_items(cap_stroke_loop_items, i, j):
    if i < 0 or j >= len(cap_stroke_loop_items) or i >= j:
        return list(cap_stroke_loop_items)
    middle = [reverse_cap_stroke_loop_item(item) for item in reversed(cap_stroke_loop_items[i : j + 1])]
    return list(cap_stroke_loop_items[:i]) + middle + list(cap_stroke_loop_items[j + 1 :])


def cap_loop_order_score(metrics):
    return (
        int(metrics["self_intersection_count"]),
        int(round(metrics["max_endpoint_gap"] * 1000.0)),
        int(round(metrics["endpoint_gap_sum"] * 1000.0)),
        -int(round(metrics["abs_area"])),
    )


def repair_cap_stroke_loop_order(cap_stroke_loop_items, max_passes=32):
    current = list(cap_stroke_loop_items)
    before = evaluate_cap_stroke_loop_order(current)
    passes = 0
    changed = False
    while passes < max_passes:
        passes += 1
        current_metrics = evaluate_cap_stroke_loop_order(current)
        best_items = current
        best_metrics = current_metrics
        improved = False
        for i in range(1, max(len(current) - 1, 1)):
            for j in range(i + 1, len(current)):
                candidate = apply_two_opt_to_cap_stroke_loop_items(current, i, j)
                candidate_metrics = evaluate_cap_stroke_loop_order(candidate)
                if cap_loop_order_score(candidate_metrics) < cap_loop_order_score(best_metrics):
                    best_items = candidate
                    best_metrics = candidate_metrics
                    improved = True
        if not improved:
            break
        current = best_items
        changed = True
        if best_metrics["self_intersection_count"] == 0 and best_metrics["max_endpoint_gap"] <= 1e-5:
            break
    after = evaluate_cap_stroke_loop_order(current)
    return current, {
        "changed": bool(changed),
        "two_opt_passes": int(passes),
        "before": before,
        "after": after,
    }


def validate_cap_stroke_loop_order(cap_stroke_loop_items):
    ordered = order_cap_stroke_loop_items(cap_stroke_loop_items)
    repaired, repair_debug = repair_cap_stroke_loop_order(ordered)
    return repaired, repair_debug


def project_polyline_min_norm(polyline_2d, cal):
    return [pixel_to_world_min_norm(point, cal) for point in np.asarray(polyline_2d, dtype=float)]


def project_polyline_to_plane(polyline_2d, plane_point, plane_normal, cal):
    return [intersect_pixel_with_plane(point, plane_point, plane_normal, cal) for point in np.asarray(polyline_2d, dtype=float)]


def project_polyline_to_z(polyline_2d, z_value, cal):
    return [pixel_to_world_on_z(point, z_value, cal) for point in np.asarray(polyline_2d, dtype=float)]


def group_cap_endpoints(
    strokes,
    cap_ids,
    tol,
    straightness_threshold=0.9,
    resample_step=15.0,
    curve_min_chord_px=30.0,
    curve_min_p90_chord_dev_px=5.0,
    curve_min_dev_ratio=0.04,
    curve_min_pca_rms_px=2.0,
):
    groups = []
    for sid in cap_ids:
        if sid in strokes:
            polyline_2d, _ = recovery_stroke_polyline(
                strokes[sid],
                straightness_threshold=straightness_threshold,
                resample_step=resample_step,
                curve_min_chord_px=curve_min_chord_px,
                curve_min_p90_chord_dev_px=curve_min_p90_chord_dev_px,
                curve_min_dev_ratio=curve_min_dev_ratio,
                curve_min_pca_rms_px=curve_min_pca_rms_px,
            )
            p0, p1 = polyline_2d[0], polyline_2d[-1]
            endpoint_pixels = {"p0": p0, "p1": p1}
        else:
            endpoint_pixels = {"p0": strokes[sid]["p0"], "p1": strokes[sid]["p1"]}
        for endpoint_name in ("p0", "p1"):
            point = endpoint_pixels[endpoint_name]
            best_i = None
            best_d = float("inf")
            for i, group in enumerate(groups):
                d = norm(point - group["center"])
                if d < best_d:
                    best_i = i
                    best_d = d
            if best_i is not None and best_d <= tol:
                group = groups[best_i]
                group["members"].append({"stroke": sid, "endpoint": endpoint_name, "pixel": point})
                group["center"] = np.mean([m["pixel"] for m in group["members"]], axis=0)
            else:
                groups.append(
                    {
                        "members": [{"stroke": sid, "endpoint": endpoint_name, "pixel": point}],
                        "center": point.copy(),
                    }
                )

    centroid = np.mean([g["center"] for g in groups], axis=0)
    groups.sort(key=lambda g: math.atan2(g["center"][1] - centroid[1], g["center"][0] - centroid[0]))
    return groups


def cap_endpoint_member_role(member):
    endpoint = str(member.get("graph_endpoint", member.get("endpoint", ""))).lower()
    if endpoint in {"start", "p0"}:
        return "start"
    if endpoint in {"end", "p1"}:
        return "end"
    endpoint = str(member.get("endpoint", "")).lower()
    if endpoint in {"start", "p0"}:
        return "start"
    if endpoint in {"end", "p1"}:
        return "end"
    return None


def stroke_group_indices_from_cap_groups(cap_ids, cap_groups):
    member_to_group = {}
    for group_idx, group in enumerate(cap_groups):
        for member in group.get("members", []):
            role = cap_endpoint_member_role(member)
            if role is None:
                continue
            member_to_group[(int(member["stroke"]), role)] = int(group_idx)

    stroke_group_indices = {}
    missing_strokes = []
    for sid in cap_ids:
        sid = int(sid)
        start_group = member_to_group.get((sid, "start"))
        end_group = member_to_group.get((sid, "end"))
        if start_group is None or end_group is None:
            missing_strokes.append(sid)
            continue
        stroke_group_indices[str(sid)] = {
            "start": int(start_group),
            "end": int(end_group),
        }
    return stroke_group_indices, missing_strokes


def cap_stroke_order_from_group_graph(cap_ids, stroke_group_indices):
    """Order fallback cap strokes by walking their endpoint-group graph."""
    cap_ids = [int(sid) for sid in cap_ids if str(int(sid)) in stroke_group_indices]
    if not cap_ids:
        return [], {
            "method": "fallback_endpoint_group_euler_walk",
            "edge_count": 0,
            "ordered_count": 0,
            "missing_after_order": [],
            "odd_group_indices": [],
            "component_count": 0,
        }

    edges = []
    for sid in cap_ids:
        groups = stroke_group_indices[str(sid)]
        edges.append({
            "stroke": int(sid),
            "start": int(groups["start"]),
            "end": int(groups["end"]),
        })

    adjacency = {}
    degree = {}
    for edge_idx, edge in enumerate(edges):
        a = int(edge["start"])
        b = int(edge["end"])
        adjacency.setdefault(a, []).append(edge_idx)
        if a != b:
            adjacency.setdefault(b, []).append(edge_idx)
            degree[a] = degree.get(a, 0) + 1
            degree[b] = degree.get(b, 0) + 1
        else:
            degree[a] = degree.get(a, 0) + 2

    cap_order_pos = {sid: pos for pos, sid in enumerate(cap_ids)}
    for edge_list in adjacency.values():
        edge_list.sort(key=lambda idx: cap_order_pos.get(int(edges[idx]["stroke"]), idx))

    def choose_start_node(available_edges):
        available_nodes = sorted(
            {
                int(edges[idx]["start"])
                for idx in available_edges
            }
            | {
                int(edges[idx]["end"])
                for idx in available_edges
            }
        )
        odd_nodes = [node for node in available_nodes if degree.get(node, 0) % 2 == 1]
        if odd_nodes:
            return odd_nodes[0]
        first_edge = min(available_edges, key=lambda idx: cap_order_pos.get(int(edges[idx]["stroke"]), idx))
        return int(edges[first_edge]["start"])

    used = set()
    ordered_edge_indices = []
    component_count = 0

    while len(used) < len(edges):
        remaining_edges = [idx for idx in range(len(edges)) if idx not in used]
        start_node = choose_start_node(remaining_edges)
        stack = [(start_node, None)]
        component_edges = []
        component_count += 1

        while stack:
            node, incoming_edge_idx = stack[-1]
            next_edge_idx = None
            for edge_idx in adjacency.get(node, []):
                if edge_idx not in used:
                    next_edge_idx = edge_idx
                    break
            if next_edge_idx is None:
                _node, incoming = stack.pop()
                if incoming is not None:
                    component_edges.append(incoming)
                continue

            used.add(next_edge_idx)
            edge = edges[next_edge_idx]
            a = int(edge["start"])
            b = int(edge["end"])
            next_node = b if node == a else a
            stack.append((next_node, next_edge_idx))

        ordered_edge_indices.extend(reversed(component_edges))

    ordered = [int(edges[idx]["stroke"]) for idx in ordered_edge_indices]
    ordered_set = set(ordered)
    missing_after_order = [sid for sid in cap_ids if sid not in ordered_set]
    ordered.extend(missing_after_order)

    odd_group_indices = sorted([int(node) for node, deg in degree.items() if deg % 2 == 1])
    return ordered, {
        "method": "fallback_endpoint_group_euler_walk",
        "edge_count": int(len(edges)),
        "ordered_count": int(len(ordered)),
        "missing_after_order": [int(sid) for sid in missing_after_order],
        "odd_group_indices": odd_group_indices,
        "component_count": int(component_count),
    }


def build_fallback_cap_endpoint_source(source_path, fallback_reason, cap_ids, cap_groups, endpoint_tol):
    stroke_group_indices, missing_strokes = stroke_group_indices_from_cap_groups(cap_ids, cap_groups)
    ordered_strokes, order_debug = cap_stroke_order_from_group_graph(cap_ids, stroke_group_indices)

    graph_edges = set()
    stroke_edges = []
    for sid in cap_ids:
        group_info = stroke_group_indices.get(str(int(sid)))
        if group_info is None:
            continue
        start_group = int(group_info["start"])
        end_group = int(group_info["end"])
        stroke_edges.append({
            "stroke": int(sid),
            "start_group": start_group,
            "end_group": end_group,
        })
        if start_group != end_group:
            graph_edges.add(tuple(sorted((start_group, end_group))))

    return {
        "source": str(source_path),
        "fallback_reason": fallback_reason,
        "method": "fallback_endpoint_groups_from_stroke_directions",
        "endpoint_group_tolerance_pixels": float(endpoint_tol),
        "raw_endpoint_count": int(sum(len(g.get("members", [])) for g in cap_groups)),
        "post_snap_endpoint_group_count": int(len(cap_groups)),
        "pruned_strokes": [int(sid) for sid in cap_ids],
        "component_strokes_in_order": [int(sid) for sid in ordered_strokes],
        "fallback_missing_strokes": [int(sid) for sid in missing_strokes],
        "fallback_order_debug": order_debug,
        "graph_edges_group_indices": [[int(a), int(b)] for a, b in sorted(graph_edges)],
        "stroke_edges_group_indices": stroke_edges,
        "stroke_group_indices": stroke_group_indices,
    }


def min_distance_to_cap(point, cap_groups):
    return min(norm(point - g["center"]) for g in cap_groups)


def orient_side_stroke(
    stroke,
    cap_groups,
    straightness_threshold=0.9,
    resample_step=15.0,
    curve_min_chord_px=30.0,
    curve_min_p90_chord_dev_px=5.0,
    curve_min_dev_ratio=0.04,
    curve_min_pca_rms_px=2.0,
):
    polyline_2d, is_curved = recovery_stroke_polyline(
        stroke,
        straightness_threshold=straightness_threshold,
        resample_step=resample_step,
        curve_min_chord_px=curve_min_chord_px,
        curve_min_p90_chord_dev_px=curve_min_p90_chord_dev_px,
        curve_min_dev_ratio=curve_min_dev_ratio,
        curve_min_pca_rms_px=curve_min_pca_rms_px,
    )
    p0, p1 = polyline_2d[0], polyline_2d[-1]
    d0 = min_distance_to_cap(p0, cap_groups)
    d1 = min_distance_to_cap(p1, cap_groups)
    if d0 <= d1:
        near_name, far_name = "p0", "p1"
        near_pixel, far_pixel = p0, p1
    else:
        near_name, far_name = "p1", "p0"
        near_pixel, far_pixel = p1, p0
    return {
        "stroke": stroke["id"],
        "near_endpoint": near_name,
        "far_endpoint": far_name,
        "near_pixel": near_pixel,
        "far_pixel": far_pixel,
        "near_distance_to_source_cap": min(d0, d1),
        "far_distance_to_source_cap": max(d0, d1),
        "vector_pixel": far_pixel - near_pixel,
        "chord": norm(far_pixel - near_pixel),
        "arc": stroke["arc"],
        "straightness": float(stroke.get("straightness", 1.0)),
        "is_curved": bool(is_curved),
    }


def canonical_axis_direction_2d(vector):
    vector = np.array(vector, dtype=float)
    length = norm(vector)
    if length < 1e-9:
        return None
    direction = vector / length
    if direction[0] < 0 or (abs(direction[0]) < 1e-12 and direction[1] < 0):
        direction = -direction
    return direction


def angle_between_dirs_2d(a, b):
    da = canonical_axis_direction_2d(a)
    db = canonical_axis_direction_2d(b)
    if da is None or db is None:
        return 180.0
    c = abs(float(np.dot(da, db)))
    c = max(-1.0, min(1.0, c))
    return float(math.degrees(math.acos(c)))


def majority_side_direction_2d(oriented_sides):
    dirs = []
    for side in oriented_sides:
        direction = canonical_axis_direction_2d(side["vector_pixel"])
        if direction is not None:
            dirs.append(direction)
    if not dirs:
        return None
    ref = dirs[0]
    aligned = []
    for direction in dirs:
        if float(np.dot(direction, ref)) < 0:
            direction = -direction
        aligned.append(direction)
    mean = np.mean(aligned, axis=0)
    return canonical_axis_direction_2d(mean)


def choose_primary_side_by_majority(oriented_sides, angle_tol=25.0):
    if not oriented_sides:
        raise RuntimeError("No oriented side strokes available.")
    majority = majority_side_direction_2d(oriented_sides)
    ranked = sorted(oriented_sides, key=lambda s: float(s["chord"]), reverse=True)
    rejected = []
    for side in ranked:
        angle = 0.0 if majority is None else angle_between_dirs_2d(side["vector_pixel"], majority)
        if majority is not None and angle > float(angle_tol):
            rejected.append(
                {
                    "stroke": int(side["stroke"]),
                    "chord": float(side["chord"]),
                    "arc": float(side["arc"]),
                    "angle_to_majority": float(angle),
                    "reason": "angle_to_majority_gt_tolerance",
                }
            )
            continue
        selected = dict(side)
        selected["copy_selection_reason"] = "longest_direction_consistent_side"
        selected["angle_to_majority"] = float(angle)
        selected["copy_direction_angle_tol"] = float(angle_tol)
        selected["rejected_copy_side_candidates"] = rejected
        return selected

    selected = dict(ranked[0])
    selected["copy_selection_reason"] = "fallback_longest_no_direction_consistent_side"
    selected["angle_to_majority"] = angle_between_dirs_2d(selected["vector_pixel"], majority) if majority is not None else 0.0
    selected["copy_direction_angle_tol"] = float(angle_tol)
    selected["rejected_copy_side_candidates"] = rejected
    return selected


def choose_primary_side_by_preselected_stroke(oriented_sides, stroke_id, angle_tol=25.0, reason=None, copy_iou=None):
    if stroke_id is None:
        return None
    for side in oriented_sides:
        if int(side["stroke"]) != int(stroke_id):
            continue
        majority = majority_side_direction_2d(oriented_sides)
        selected = dict(side)
        selected["copy_selection_reason"] = reason or "preselected_copy_side_from_2d_sweep"
        selected["angle_to_majority"] = angle_between_dirs_2d(selected["vector_pixel"], majority) if majority is not None else 0.0
        selected["copy_direction_angle_tol"] = float(angle_tol)
        selected["rejected_copy_side_candidates"] = []
        if copy_iou is not None:
            selected["copy_iou"] = float(copy_iou)
        return selected
    return None


def to_list(value):
    arr = np.array(value, dtype=float)
    return [float(x) for x in arr.tolist()]


def cap_group_to_json(group, copied_offset, cal, gt_assignment=None, obj_assignment=None):
    source_pixel = group["center"]
    copied_pixel = group.get("copied_center", source_pixel + copied_offset)
    result = {
        "source_cap": {
            "pixel_top_left": to_list(source_pixel),
            "world_min_norm": to_list(pixel_to_world_min_norm(source_pixel, cal)),
        },
        "copied_cap": {
            "pixel_top_left": to_list(copied_pixel),
            "world_min_norm": to_list(pixel_to_world_min_norm(copied_pixel, cal)),
        },
        "members": [
            {
                "stroke": int(m["stroke"]),
                "endpoint": m["endpoint"],
                "pixel_top_left": to_list(m["pixel"]),
            }
            for m in group["members"]
        ],
    }
    if gt_assignment is not None:
        if "detected_source_plane_point" in gt_assignment:
            source_plane_point = np.array(gt_assignment["detected_source_plane_point"], dtype=float)
            source_plane_normal = np.array(gt_assignment["detected_source_plane_normal"], dtype=float)
            copied_plane_point = np.array(gt_assignment["detected_copied_plane_point"], dtype=float)
            copied_plane_normal = np.array(gt_assignment["detected_copied_plane_normal"], dtype=float)
            result["source_cap"]["world_on_gt_cap_plane"] = to_list(
                intersect_pixel_with_plane(source_pixel, source_plane_point, source_plane_normal, cal)
            )
            result["copied_cap"]["world_on_gt_cap_plane"] = to_list(
                intersect_pixel_with_plane(copied_pixel, copied_plane_point, copied_plane_normal, cal)
            )
        else:
            result["source_cap"]["world_on_gt_cap_plane"] = to_list(
                pixel_to_world_on_z(source_pixel, gt_assignment["detected_source_z"], cal)
            )
            result["copied_cap"]["world_on_gt_cap_plane"] = to_list(
                pixel_to_world_on_z(copied_pixel, gt_assignment["detected_copied_z"], cal)
            )
    if obj_assignment is not None:
        source_plane_point = np.array(obj_assignment["source_plane_point"], dtype=float)
        copied_plane_point = np.array(obj_assignment["copied_plane_point"], dtype=float)
        plane_normal = np.array(obj_assignment["plane_normal_source_to_copied"], dtype=float)
        result["source_cap"]["world_on_obj_cap_plane"] = to_list(
            intersect_pixel_with_plane(source_pixel, source_plane_point, plane_normal, cal)
        )
        result["copied_cap"]["world_on_obj_cap_plane"] = to_list(
            intersect_pixel_with_plane(copied_pixel, copied_plane_point, plane_normal, cal)
        )
    return result


def save_cap_endpoint_graph_debug_png(cap_groups, output_path, image_shape_hw, graph_edges=None):
    try:
        import cv2
    except ModuleNotFoundError:
        return None

    h, w = int(image_shape_hw[0]), int(image_shape_hw[1])
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    centers = [np.array(group["center"], dtype=float) for group in cap_groups]

    if graph_edges:
        for a, b in graph_edges:
            if a >= len(centers) or b >= len(centers):
                continue
            pa = tuple(np.round(centers[a]).astype(int))
            pb = tuple(np.round(centers[b]).astype(int))
            cv2.line(canvas, pa, pb, (0, 180, 0), 2, cv2.LINE_AA)
    elif len(centers) >= 2:
        pts = np.round(np.array(centers, dtype=float)).astype(np.int32)
        cv2.polylines(canvas, [pts.reshape(-1, 1, 2)], isClosed=True, color=(0, 180, 0), thickness=2)

    for group_idx, group in enumerate(cap_groups):
        center = np.round(group["center"]).astype(int)
        for member in group["members"]:
            p = np.round(member["pixel"]).astype(int)
            cv2.circle(canvas, tuple(p), 5, (160, 160, 160), -1)
            cv2.putText(
                canvas,
                f"s{member['stroke']}:{member.get('graph_endpoint', member['endpoint'])}",
                (int(p[0]) + 6, int(p[1]) - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (80, 80, 80),
                1,
                cv2.LINE_AA,
            )

        cv2.circle(canvas, tuple(center), 8, (0, 0, 255), -1)
        cv2.putText(
            canvas,
            f"G{group_idx} ({center[0]},{center[1]})",
            (int(center[0]) + 10, int(center[1]) + 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 180),
            1,
            cv2.LINE_AA,
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)


def points_close_2d(a, b, tol=1e-5):
    return norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)) <= tol


def flatten_stroke_loop_points_2d(stroke_loop, key):
    flat_points = []
    stroke_items = []
    total = len(stroke_loop)
    for item_idx, item in enumerate(stroke_loop):
        poly = np.asarray(item[f"{key}_polyline_2d"], dtype=float)
        if len(poly) == 0:
            continue
        point_indices = []
        for point_idx, point in enumerate(poly):
            if flat_points and point_idx == 0 and points_close_2d(flat_points[-1], point):
                point_indices.append(len(flat_points) - 1)
                continue
            if (
                item_idx == total - 1
                and point_idx == len(poly) - 1
                and flat_points
                and points_close_2d(flat_points[0], point)
            ):
                point_indices.append(0)
                continue
            flat_points.append(np.asarray(point, dtype=float))
            point_indices.append(len(flat_points) - 1)
        stroke_items.append(
            {
                "stroke": int(item["stroke"]),
                "is_curved": bool(item.get("is_curved", False) or len(poly) > 2),
                "straightness": float(item.get("straightness", 1.0)),
                "polyline_point_count": int(len(poly)),
                "point_indices": point_indices,
            }
        )
    return flat_points, stroke_items


def closed_loop_edges(point_count):
    if point_count < 2:
        return []
    return [[idx, (idx + 1) % point_count] for idx in range(point_count)]


def save_blender_cap_debug_outputs(
    result,
    png_path,
    json_path,
    image_shape_hw,
):
    cap_stroke_loop = result.get("cap_stroke_loop") or []
    source_groups = result.get("source_cap_endpoint_groups") or []
    copied_offset = np.array(
        result.get("copied_cap_offset_from_primary_side", {}).get("vector_pixel", [0.0, 0.0]),
        dtype=float,
    )

    mode = "endpoint_loop"
    source_strokes = []
    copied_strokes = []
    if cap_stroke_loop:
        mode = "polyline_loop"
        source_points, source_strokes = flatten_stroke_loop_points_2d(cap_stroke_loop, "source")
        copied_points, copied_strokes = flatten_stroke_loop_points_2d(cap_stroke_loop, "copied")
    else:
        source_points = [np.array(g["source_cap"]["pixel_top_left"], dtype=float) for g in source_groups]
        copied_points = [np.array(g["copied_cap"]["pixel_top_left"], dtype=float) for g in source_groups]

    source_edges = closed_loop_edges(len(source_points))
    copied_edges = closed_loop_edges(len(copied_points))

    payload = {
        "mode": mode,
        "source_loop_points_2d": [to_list(p) for p in source_points],
        "copied_loop_points_2d": [to_list(p) for p in copied_points],
        "source_loop_edges": source_edges,
        "copied_loop_edges": copied_edges,
        "source_loop_strokes": source_strokes,
        "copied_loop_strokes": copied_strokes,
        "copied_offset_vector_pixel": to_list(copied_offset),
        "cap_loop_order_debug": result.get("cap_loop_order_debug"),
    }
    write_debug_json(json_path, payload)

    try:
        import cv2
    except ModuleNotFoundError:
        return payload

    h, w = int(image_shape_hw[0]), int(image_shape_hw[1])
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)

    def draw_edges(points, edges, color):
        for a, b in edges:
            if a >= len(points) or b >= len(points):
                continue
            pa = tuple(np.round(points[a]).astype(int))
            pb = tuple(np.round(points[b]).astype(int))
            cv2.line(canvas, pa, pb, color, 2, cv2.LINE_AA)

    def draw_points(points, prefix, color):
        for idx, point in enumerate(points):
            p = tuple(np.round(point).astype(int))
            cv2.circle(canvas, p, 4, color, -1, cv2.LINE_AA)
            cv2.putText(
                canvas,
                f"{prefix}{idx}",
                (int(p[0]) + 4, int(p[1]) - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

    draw_edges(source_points, source_edges, (0, 160, 0))
    draw_edges(copied_points, copied_edges, (200, 80, 0))
    draw_points(source_points, "S", (0, 180, 0))
    draw_points(copied_points, "C", (255, 120, 0))

    cv2.putText(
        canvas,
        f"Blender cap input ({mode})",
        (15, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "green=source cap, orange=copied cap",
        (15, 54),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    png_path = Path(png_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(png_path), canvas)
    return payload


def write_blender_reconstruction_script(json_path, blend_output, solid_blend_output, render_output, script_output):
    script = f"""
import json
import math
from pathlib import Path

import bpy
from mathutils import Vector


JSON_PATH = Path({str(Path(json_path).resolve())!r})
BLEND_OUTPUT = Path({str(Path(blend_output).resolve())!r})
SOLID_BLEND_OUTPUT = Path({str(Path(solid_blend_output).resolve())!r})
RENDER_OUTPUT = Path({str(Path(render_output).resolve())!r}) if {str(render_output is not None)!r} == "True" else None


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def clear_non_object_data():
    for collection in (
        bpy.data.meshes,
        bpy.data.materials,
        bpy.data.curves,
        bpy.data.cameras,
        bpy.data.lights,
    ):
        for item in list(collection):
            collection.remove(item)


def make_material(name, color):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = color
    return mat


def add_sphere(name, location, radius, material):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=12, radius=radius, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.data.materials.append(material)
    return obj


def add_cylinder_between(name, start, end, radius, material):
    start = Vector(start)
    end = Vector(end)
    direction = end - start
    if direction.length == 0:
        return None
    mid = (start + end) * 0.5
    bpy.ops.mesh.primitive_cylinder_add(vertices=16, radius=radius, depth=direction.length, location=mid)
    obj = bpy.context.object
    obj.name = name
    obj.rotation_euler = direction.to_track_quat("Z", "Y").to_euler()
    obj.data.materials.append(material)
    return obj


def add_arrow(name, start, vector, shaft_radius, material):
    start = Vector(start)
    vector = Vector(vector)
    if vector.length == 0:
        return []
    end = start + vector
    shaft_end = start + vector * 0.82
    objs = [add_cylinder_between(name + "_shaft", start, shaft_end, shaft_radius, material)]
    bpy.ops.mesh.primitive_cone_add(
        vertices=24,
        radius1=shaft_radius * 3.0,
        radius2=0.0,
        depth=shaft_radius * 10.0,
        location=shaft_end + vector.normalized() * shaft_radius * 5.0,
    )
    cone = bpy.context.object
    cone.name = name + "_head"
    cone.rotation_euler = vector.to_track_quat("Z", "Y").to_euler()
    cone.data.materials.append(material)
    objs.append(cone)
    return objs


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


def world(item, key):
    return item[key].get("world_on_obj_cap_plane", item[key].get("world_on_gt_cap_plane", item[key]["world_min_norm"]))


def world_polyline(item, key):
    if key == "source":
        return item.get(
            "source_polyline_world_on_obj_cap_plane",
            item.get("source_polyline_world_on_gt_cap_plane", item["source_polyline_world_min_norm"]),
        )
    return item.get(
        "copied_polyline_world_on_obj_cap_plane",
        item.get("copied_polyline_world_on_gt_cap_plane", item["copied_polyline_world_min_norm"]),
    )


def first_vector(*values):
    for value in values:
        if value is not None:
            return value
    raise RuntimeError("No valid vector value was found.")


def fit_camera(points):
    pts = [Vector(p) for p in points]
    center = sum(pts, Vector((0, 0, 0))) / len(pts)
    span = max((p - center).length for p in pts)
    cam_data = bpy.data.cameras.new("Recovered_Endpoint_Camera")
    cam = bpy.data.objects.new("Recovered_Endpoint_Camera", cam_data)
    bpy.context.collection.objects.link(cam)
    cam.data.type = "ORTHO"
    cam.data.ortho_scale = max(8.0, span * 2.8)
    cam.location = center + Vector((12.0, -16.0, 10.0))
    direction = center - cam.location
    cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.camera = cam

    light_data = bpy.data.lights.new("Recovered_Key_Light", "AREA")
    light = bpy.data.objects.new("Recovered_Key_Light", light_data)
    bpy.context.collection.objects.link(light)
    light.location = center + Vector((0, -6, 10))
    light.data.energy = 450
    light.data.size = 5


def loop_order_from_edges(edges, point_count):
    if not edges:
        return list(range(point_count))
    adjacency = {{idx: set() for idx in range(point_count)}}
    for a, b in edges:
        if a == b or a >= point_count or b >= point_count:
            continue
        adjacency[a].add(b)
        adjacency[b].add(a)
    starts = [idx for idx, neighbors in adjacency.items() if neighbors]
    if not starts:
        return list(range(point_count))
    start = min(starts)
    order = [start]
    prev = None
    cur = start
    while True:
        candidates = sorted(n for n in adjacency[cur] if n != prev)
        if not candidates:
            break
        nxt = candidates[0]
        if nxt == start:
            return order
        if nxt in order:
            break
        order.append(nxt)
        prev, cur = cur, nxt
    raise RuntimeError(f"Could not trace a single cap loop from graph edges: {{edges}}")


def points_close(a, b, tol=1e-5):
    return (Vector(a) - Vector(b)).length <= tol


def flatten_stroke_loop_points(stroke_loop, key):
    out = []
    for item in stroke_loop:
        poly = [Vector(p)[:] for p in world_polyline(item, key)]
        if not poly:
            continue
        if out and points_close(out[-1], poly[0]):
            out.extend(poly[1:])
        else:
            out.extend(poly)
    if len(out) >= 2 and points_close(out[0], out[-1]):
        out.pop()
    return out


def add_extruded_volume(name, source_points, copied_points, graph_edges, material=None):
    loop_order = loop_order_from_edges(graph_edges, len(source_points))
    if len(loop_order) < 3:
        raise RuntimeError("Need at least three ordered cap endpoints to create a surface.")

    verts = [Vector(source_points[idx])[:] for idx in loop_order]
    verts += [Vector(copied_points[idx])[:] for idx in loop_order]
    n = len(loop_order)

    source_face = list(range(n))
    copied_face = list(range(2 * n - 1, n - 1, -1))
    side_faces = [
        [idx, (idx + 1) % n, n + ((idx + 1) % n), n + idx]
        for idx in range(n)
    ]

    mesh = bpy.data.meshes.new(name + "_mesh")
    mesh.from_pydata(verts, [], [source_face, copied_face] + side_faces)
    mesh.update(calc_edges=True)

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    if material is not None:
        obj.data.materials.append(material)
    return obj, loop_order


def add_extruded_volume_from_polyline_loop(name, source_loop_points, copied_loop_points, material=None):
    if len(source_loop_points) < 3 or len(copied_loop_points) < 3:
        raise RuntimeError("Need at least three polyline points to create a curved surface.")
    if len(source_loop_points) != len(copied_loop_points):
        raise RuntimeError("Source/copied polyline loops must have the same point count.")

    verts = [Vector(p)[:] for p in source_loop_points]
    verts += [Vector(p)[:] for p in copied_loop_points]
    n = len(source_loop_points)

    source_face = list(range(n))
    copied_face = list(range(2 * n - 1, n - 1, -1))
    side_faces = [
        [idx, (idx + 1) % n, n + ((idx + 1) % n), n + idx]
        for idx in range(n)
    ]

    mesh = bpy.data.meshes.new(name + "_mesh")
    mesh.from_pydata(verts, [], [source_face, copied_face] + side_faces)
    mesh.update(calc_edges=True)

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    if material is not None:
        obj.data.materials.append(material)
    return obj, list(range(n))


def main():
    data = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    clear_scene()

    green = make_material("source_cap_green", (0.0, 0.8, 0.0, 1.0))
    blue = make_material("copied_cap_blue", (0.0, 0.15, 1.0, 1.0))
    orange = make_material("side_stroke_orange", (1.0, 0.45, 0.0, 1.0))
    red = make_material("extrusion_direction_red", (1.0, 0.0, 0.0, 1.0))
    black = make_material("label_black", (0.0, 0.0, 0.0, 1.0))
    gray = make_material("cap_pair_gray", (0.35, 0.35, 0.35, 1.0))

    groups = data["source_cap_endpoint_groups"]
    cap_stroke_loop = data.get("cap_stroke_loop") or []
    source_points = [world(g, "source_cap") for g in groups]
    copied_points = [world(g, "copied_cap") for g in groups]
    source_loop_points = flatten_stroke_loop_points(cap_stroke_loop, "source") if cap_stroke_loop else source_points
    copied_loop_points = flatten_stroke_loop_points(cap_stroke_loop, "copied") if cap_stroke_loop else copied_points
    side_near = [
        s.get("near_world_on_obj_cap_plane", s.get("near_world_on_gt_cap_plane", s["near_world_min_norm"]))
        for s in data["side_strokes_oriented_near_to_far"]
    ]
    side_far = [
        s.get("far_world_on_obj_cap_plane", s.get("far_world_on_gt_cap_plane", s["far_world_min_norm"]))
        for s in data["side_strokes_oriented_near_to_far"]
    ]

    all_points = source_loop_points + copied_loop_points + side_near + side_far
    fit_camera(all_points)

    for idx, group in enumerate(groups):
        sp = world(group, "source_cap")
        cp = world(group, "copied_cap")
        add_sphere(f"source_cap_endpoint_{{idx:02d}}", sp, 0.10, green)
        add_sphere(f"copied_cap_endpoint_{{idx:02d}}", cp, 0.10, blue)
        add_cylinder_between(f"cap_pair_extrusion_link_{{idx:02d}}", sp, cp, 0.025, gray)
        add_text(f"source_label_{{idx:02d}}", f"S{{idx}}", Vector(sp) + Vector((0, 0, 0.22)), 0.18, black)
        add_text(f"copied_label_{{idx:02d}}", f"C{{idx}}", Vector(cp) + Vector((0, 0, 0.22)), 0.18, black)

    graph_edges = data.get("cap_endpoint_source", {{}}).get("graph_edges_group_indices") or []
    if cap_stroke_loop:
        for item in cap_stroke_loop:
            source_poly = world_polyline(item, "source")
            copied_poly = world_polyline(item, "copied")
            for edge_idx in range(len(source_poly) - 1):
                add_cylinder_between(
                    f"source_cap_polyline_{{item['stroke']}}_{{edge_idx:02d}}",
                    source_poly[edge_idx],
                    source_poly[edge_idx + 1],
                    0.03,
                    green,
                )
            for edge_idx in range(len(copied_poly) - 1):
                add_cylinder_between(
                    f"copied_cap_polyline_{{item['stroke']}}_{{edge_idx:02d}}",
                    copied_poly[edge_idx],
                    copied_poly[edge_idx + 1],
                    0.03,
                    blue,
                )
    elif graph_edges:
        for edge_idx, (a, b) in enumerate(graph_edges):
            if a >= len(source_points) or b >= len(source_points):
                continue
            add_cylinder_between(
                f"source_cap_graph_edge_{{edge_idx:02d}}",
                source_points[a],
                source_points[b],
                0.035,
                green,
            )
            add_cylinder_between(
                f"copied_cap_graph_edge_{{edge_idx:02d}}",
                copied_points[a],
                copied_points[b],
                0.035,
                blue,
            )
    else:
        for idx in range(len(source_points)):
            add_cylinder_between(
                f"source_cap_edge_{{idx:02d}}",
                source_points[idx],
                source_points[(idx + 1) % len(source_points)],
                0.035,
                green,
            )
            add_cylinder_between(
                f"copied_cap_edge_{{idx:02d}}",
                copied_points[idx],
                copied_points[(idx + 1) % len(copied_points)],
                0.035,
                blue,
            )

    for side in data["side_strokes_oriented_near_to_far"]:
        add_cylinder_between(
            f"side_stroke_{{side['stroke']}}_near_to_far",
            side.get("near_world_on_obj_cap_plane", side.get("near_world_on_gt_cap_plane", side["near_world_min_norm"])),
            side.get("far_world_on_obj_cap_plane", side.get("far_world_on_gt_cap_plane", side["far_world_min_norm"])),
            0.025,
            orange,
        )

    source_center = sum((Vector(p) for p in source_points), Vector((0, 0, 0))) / len(source_points)
    offset = data["copied_cap_offset_from_primary_side"]
    extrusion_vector = Vector(
        first_vector(
            offset.get("vector_world_on_obj_cap_planes"),
            offset.get("vector_world_on_gt_cap_planes"),
            offset.get("vector_world_min_norm"),
        )
    )
    add_arrow("primary_extrusion_direction", source_center, extrusion_vector, 0.045, red)
    add_text("title", "Recovered cap endpoints and extrusion direction", source_center + Vector((0, 0, 1.2)), 0.25, black)

    bpy.context.scene.render.engine = "CYCLES"
    bpy.context.scene.cycles.samples = 64
    bpy.context.scene.render.resolution_x = 1400
    bpy.context.scene.render.resolution_y = 1000
    bpy.context.scene.world.color = (0.78, 0.78, 0.78)

    BLEND_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(BLEND_OUTPUT))
    if RENDER_OUTPUT is not None:
        RENDER_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        bpy.context.scene.render.filepath = str(RENDER_OUTPUT)
        bpy.ops.render.render(write_still=True)

    clear_scene()
    clear_non_object_data()
    if cap_stroke_loop:
        solid_obj, loop_order = add_extruded_volume_from_polyline_loop(
            "reconstructed_cap_loop_extruded_solid",
            source_loop_points,
            copied_loop_points,
            None,
        )
    else:
        solid_obj, loop_order = add_extruded_volume(
            "reconstructed_cap_loop_extruded_solid",
            source_points,
            copied_points,
            graph_edges,
            None,
        )
    SOLID_BLEND_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(SOLID_BLEND_OUTPUT))


main()
"""
    script_output = Path(script_output)
    script_output.parent.mkdir(parents=True, exist_ok=True)
    script_output.write_text(textwrap.dedent(script).strip() + "\n", encoding="utf-8")


def run_blender_reconstruction(json_path, blend_output, solid_blend_output, render_output, blender_exe, script_output):
    write_blender_reconstruction_script(json_path, blend_output, solid_blend_output, render_output, script_output)
    subprocess.run(
        [
            str(blender_exe),
            "--background",
            "--python",
            str(script_output),
        ],
        check=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Recover rank cap endpoint 3D coordinates from extrusion debug output and anchor calibration."
    )
    parser.add_argument("--debug-dir", default="debug")
    parser.add_argument(
        "--calibration",
        default="blender_axonometric_dev_dataset/dev_camera_anchor_calibrationInput.json",
    )
    parser.add_argument("--output", default="debug/iou_rank00_cap_endpoints_3d.json")
    parser.add_argument("--rank-source", choices=["iou", "cluster"], default="iou")
    parser.add_argument(
        "--rank",
        type=int,
        default=0,
        help="Rank to recover. Defaults to IoU rank 0, i.e. the best IoU overlay.",
    )
    parser.add_argument("--overlay", default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--overlay-image",
        default=None,
        help="side_bestcap_overlay PNG. Defaults to iou_rank_00 side_bestcap_overlay under debug.",
    )
    parser.add_argument(
        "--endpoint-source",
        choices=["overlay", "graph"],
        default="graph",
        help="Read cap endpoints from cap_endpoint_graph_summary.txt by default.",
    )
    parser.add_argument(
        "--cap-endpoint-debug-output",
        default="debug/cap_endpoint_graph_points_debug.png",
        help="Debug PNG showing cap endpoint coordinates read from cap_endpoint_graph_summary.txt.",
    )
    parser.add_argument(
        "--blender-cap-debug-png",
        default="debug/blender_reconstruction_cap_2d.png",
        help="2D PNG debug view of the cap loop points/edges that are fed into Blender reconstruction.",
    )
    parser.add_argument(
        "--blender-cap-debug-json",
        default="debug/blender_reconstruction_cap_2d.json",
        help="JSON debug file with cap loop points and connectivity used for Blender reconstruction.",
    )
    parser.add_argument(
        "--endpoint-group-tol",
        type=float,
        default=15.0,
        help="Pixel tolerance for the single non-recursive nearby endpoint pair snap.",
    )
    parser.add_argument(
        "--obj",
        default="blender_axonometric_dev_dataset/scene_S0.obj",
        help="OBJ file used to find the cap plane by projecting its faces into the calibrated camera.",
    )
    parser.add_argument(
        "--obj-id-render-output",
        default="debug/scene_S0_face_id_render.png",
        help="Debug image for the single-pass OBJ face-ID render used to choose the cap plane.",
    )
    parser.add_argument(
        "--obj-normal-side-angle-tol",
        type=float,
        default=20.0,
        help=(
            "Maximum acute 2D angle, in degrees, between the projected OBJ cap normal and side-stroke "
            "offset to treat the extrusion as cap-normal aligned."
        ),
    )
    parser.add_argument(
        "--support-plane-polygon-tol",
        type=float,
        default=15.0,
        help="Pixel tolerance for support-plane fallback side-stroke and anchor containment tests.",
    )
    parser.add_argument(
        "--support-plane-debug-dir",
        default="debug/support_plane_fallback",
        help="Directory for step-by-step support-plane fallback debug JSON/PNG outputs.",
    )
    parser.add_argument(
        "--copy-side-angle-thresh",
        type=float,
        default=25.0,
        help="Maximum unoriented angle between a candidate copy side stroke and the side majority direction.",
    )
    parser.add_argument(
        "--curve-straightness-thresh",
        type=float,
        default=0.9,
        help="Treat strokes with straightness below this threshold as curved during 3D recovery.",
    )
    parser.add_argument(
        "--curve-resample-step-px",
        type=float,
        default=15.0,
        help="Arc-length step, in pixels, used to resample curved 2D strokes before straight-line simplification.",
    )
    parser.add_argument(
        "--curve-min-chord-px",
        type=float,
        default=30.0,
        help="Minimum stroke endpoint chord length required before a stroke can be treated as a curve.",
    )
    parser.add_argument(
        "--curve-min-p90-chord-dev-px",
        type=float,
        default=5.0,
        help="Minimum 90th-percentile distance from the endpoint chord required before a stroke can be treated as a curve.",
    )
    parser.add_argument(
        "--curve-min-dev-ratio",
        type=float,
        default=0.04,
        help="Minimum p90 chord-deviation divided by chord length required before a stroke can be treated as a curve.",
    )
    parser.add_argument(
        "--curve-min-pca-rms-px",
        type=float,
        default=2.0,
        help="Minimum RMS distance from the PCA fitted line required before a stroke can be treated as a curve.",
    )
    parser.add_argument(
        "--reconstruct-blender",
        action="store_true",
        help="Use Blender to rebuild a 3D scene from the generated endpoint JSON.",
    )
    parser.add_argument(
        "--blender-exe",
        default=r"C:\Program Files\Blender Foundation\Blender 3.6\blender.EXE",
        help="Blender executable used by --reconstruct-blender.",
    )
    parser.add_argument(
        "--blend-output",
        default="debug/iou_rank00_cap_endpoints_3d_reconstruction.blend",
        help="Output .blend file for --reconstruct-blender.",
    )
    parser.add_argument(
        "--solid-blend-output",
        default="debug/iou_rank00_cap_endpoints_3d_solid_reconstruction.blend",
        help="Output .blend file with a surface/solid reconstructed from the cap loop.",
    )
    parser.add_argument(
        "--render-output",
        default="debug/iou_rank00_cap_endpoints_3d_reconstruction.png",
        help="Optional render PNG for --reconstruct-blender. Use an empty string to skip rendering.",
    )
    parser.add_argument(
        "--blender-script-output",
        default="debug/reconstruct_rank_cap_endpoints_scene.py",
        help="Generated helper script passed to Blender.",
    )
    args = parser.parse_args()

    debug_dir = Path(args.debug_dir)
    cal = load_anchor_calibration(args.calibration)

    selected_iou = None
    selected = None
    primary_side = {"stroke": -1, "near_endpoint": "overlay_source_cap", "far_endpoint": "overlay_copied_cap"}
    strokes = None
    if args.endpoint_source == "overlay":
        overlay_image = Path(args.overlay_image) if args.overlay_image else find_default_overlay_image(debug_dir, args.rank)
        if overlay_image is None or not overlay_image.exists():
            raise RuntimeError(
                "Could not find iou_rank_00 side_bestcap_overlay PNG. "
                "Pass it explicitly with --overlay-image path\\to\\...side_bestcap_overlay.png"
            )
        cap_groups, copied_offset, cap_endpoint_source = load_overlay_cap_groups(overlay_image)
        if "iou_rank_" in overlay_image.name:
            try:
                selected_iou = parse_overlay_name(overlay_image.name)
            except RuntimeError:
                selected_iou = None
        oriented_sides = []
    else:
        strokes = load_strokes_json(debug_dir / "05a_stroke_directions.json")
        if strokes is None:
            strokes = load_strokes(debug_dir / "05a_stroke_directions.txt")
        clusters = load_cluster_entries(debug_dir / "cluster_side_caps" / "cluster_side_cap_summary.txt")
        if args.overlay:
            selected_iou = parse_overlay_name(args.overlay)
            selected = next(
                (
                    e
                    for e in clusters
                    if e["rank"] == selected_iou["cluster_rank"] and e["cluster"] == selected_iou["cluster"]
                ),
                None,
            )
            if selected is None:
                raise RuntimeError(f"No cluster entry matches overlay filename {selected_iou}")
        elif args.rank_source == "iou":
            selected_iou = load_iou_rank(debug_dir / "cluster_side_caps_iou_ranked" / "iou_similarity_summary.txt", args.rank)
            selected = next(
                (
                    e
                    for e in clusters
                    if e["rank"] == selected_iou["cluster_rank"] and e["cluster"] == selected_iou["cluster"]
                ),
                None,
            )
            if selected is None:
                raise RuntimeError(f"No cluster entry matches {selected_iou}")
        else:
            selected = next((e for e in clusters if e["rank"] == args.rank), None)
            if selected is None:
                raise RuntimeError(f"No cluster entry with rank {args.rank}")

        cap_groups, cap_endpoint_source = load_cap_endpoint_graph_groups(
            debug_dir / "cap_endpoint_graphs" / "cap_endpoint_graph_summary.txt",
            selected["cluster"],
            selected["best_cap_strokes"],
            args.endpoint_group_tol,
            strokes=strokes,
            straightness_threshold=args.curve_straightness_thresh,
            resample_step=args.curve_resample_step_px,
            curve_min_chord_px=args.curve_min_chord_px,
            curve_min_p90_chord_dev_px=args.curve_min_p90_chord_dev_px,
            curve_min_dev_ratio=args.curve_min_dev_ratio,
            curve_min_pca_rms_px=args.curve_min_pca_rms_px,
        )
        if cap_groups is None:
            fallback_reason = cap_endpoint_source
            cap_groups = group_cap_endpoints(
                strokes,
                selected["best_cap_strokes"],
                args.endpoint_group_tol,
                straightness_threshold=args.curve_straightness_thresh,
                resample_step=args.curve_resample_step_px,
                curve_min_chord_px=args.curve_min_chord_px,
                curve_min_p90_chord_dev_px=args.curve_min_p90_chord_dev_px,
                curve_min_dev_ratio=args.curve_min_dev_ratio,
                curve_min_pca_rms_px=args.curve_min_pca_rms_px,
            )
            fallback_source = debug_dir / "05a_stroke_directions.json"
            if not fallback_source.exists():
                fallback_source = debug_dir / "05a_stroke_directions.txt"
            cap_endpoint_source = build_fallback_cap_endpoint_source(
                fallback_source,
                fallback_reason,
                selected["best_cap_strokes"],
                cap_groups,
                args.endpoint_group_tol,
            )
        endpoint_debug_output = Path(args.cap_endpoint_debug_output)
        if not endpoint_debug_output.is_absolute():
            endpoint_debug_output = Path.cwd() / endpoint_debug_output
        render = cal["raw"].get("render", {})
        save_cap_endpoint_graph_debug_png(
            cap_groups,
            endpoint_debug_output,
            (
                int(render.get("resolution_y", 1000)),
                int(render.get("resolution_x", 1400)),
            ),
            cap_endpoint_source.get("graph_edges_group_indices"),
        )
        cap_endpoint_source["debug_png"] = str(endpoint_debug_output)
        oriented_sides = [
            orient_side_stroke(
                strokes[sid],
                cap_groups,
                straightness_threshold=args.curve_straightness_thresh,
                resample_step=args.curve_resample_step_px,
                curve_min_chord_px=args.curve_min_chord_px,
                curve_min_p90_chord_dev_px=args.curve_min_p90_chord_dev_px,
                curve_min_dev_ratio=args.curve_min_dev_ratio,
                curve_min_pca_rms_px=args.curve_min_pca_rms_px,
            )
            for sid in selected["side"]
        ]
        primary_side = choose_primary_side_by_preselected_stroke(
            oriented_sides,
            selected.get("copy_side_stroke"),
            angle_tol=args.copy_side_angle_thresh,
            reason=selected.get("copy_reason"),
            copy_iou=selected.get("copy_iou"),
        )
        if primary_side is None:
            primary_side = choose_primary_side_by_majority(
                oriented_sides,
                angle_tol=args.copy_side_angle_thresh,
            )
        copied_offset = primary_side["vector_pixel"]

    primary_world_vector = image_vector_to_world_min_norm(copied_offset, cal)
    all_world_vectors = [image_vector_to_world_min_norm(s["vector_pixel"], cal) for s in oriented_sides]
    mean_world_vector = np.mean(all_world_vectors, axis=0) if all_world_vectors else primary_world_vector
    gt_calibration_path, gt_data = find_groundtruth_calibration(args.calibration)
    gt_assignment = infer_gt_cap_assignment(cap_groups, gt_data, cal) if gt_data is not None else None
    obj_path = Path(args.obj)
    if not obj_path.is_absolute():
        obj_path = Path.cwd() / obj_path
    id_render_output = Path(args.obj_id_render_output)
    if not id_render_output.is_absolute():
        id_render_output = Path.cwd() / id_render_output
    obj_assignment = None
    obj_assignment_error = None
    if obj_path.exists():
        try:
            obj_assignment = infer_obj_cap_plane_assignment(
                cap_groups,
                copied_offset,
                obj_path,
                cal,
                id_render_output,
                args.obj_normal_side_angle_tol,
                oriented_sides,
                Path(args.support_plane_debug_dir),
                args.support_plane_polygon_tol,
            )
        except RuntimeError as exc:
            obj_assignment_error = str(exc)
    primary_gt_vector = None
    if gt_assignment is not None:
        if "detected_source_plane_point" in gt_assignment:
            primary_gt_vector = (
                np.array(gt_assignment["detected_copied_plane_point"], dtype=float)
                - np.array(gt_assignment["detected_source_plane_point"], dtype=float)
            )
        else:
            primary_gt_vector = np.array([0.0, 0.0, gt_assignment["detected_copied_z"] - gt_assignment["detected_source_z"]])
    primary_obj_vector = (
        np.array(obj_assignment["extrusion_vector_source_to_copied"], dtype=float) if obj_assignment is not None else None
    )

    side_json = []
    if args.endpoint_source == "overlay":
        overlay_side_items = []
        for idx, group in enumerate(cap_groups):
            source_pixel = group["center"]
            copied_pixel = group.get("copied_center", source_pixel + copied_offset)
            overlay_side_items.append(
                {
                    "stroke": -1,
                    "near_endpoint": f"source_cap_vertex_{idx}",
                    "far_endpoint": f"copied_cap_vertex_{idx}",
                    "near_pixel": source_pixel,
                    "far_pixel": copied_pixel,
                    "vector_pixel": copied_pixel - source_pixel,
                    "chord": norm(copied_pixel - source_pixel),
                    "arc": norm(copied_pixel - source_pixel),
                    "near_distance_to_source_cap": 0.0,
                    "far_distance_to_source_cap": norm(copied_pixel - source_pixel),
                }
            )
        oriented_sides = overlay_side_items

    for side in oriented_sides:
        vector_world = image_vector_to_world_min_norm(side["vector_pixel"], cal)
        item = {
            "stroke": int(side["stroke"]),
            "near_endpoint": side["near_endpoint"],
            "far_endpoint": side["far_endpoint"],
            "near_pixel_top_left": to_list(side["near_pixel"]),
            "far_pixel_top_left": to_list(side["far_pixel"]),
            "near_world_min_norm": to_list(pixel_to_world_min_norm(side["near_pixel"], cal)),
            "far_world_min_norm": to_list(pixel_to_world_min_norm(side["far_pixel"], cal)),
            "vector_pixel": to_list(side["vector_pixel"]),
            "vector_world_min_norm": to_list(vector_world),
            "unit_world_min_norm": to_list(unit(vector_world)),
            "chord": float(side["chord"]),
            "arc": float(side["arc"]),
            "near_distance_to_source_cap_pixels": float(side["near_distance_to_source_cap"]),
            "far_distance_to_source_cap_pixels": float(side["far_distance_to_source_cap"]),
        }
        if gt_assignment is not None:
            if "detected_source_plane_point" in gt_assignment:
                source_plane_point = np.array(gt_assignment["detected_source_plane_point"], dtype=float)
                source_plane_normal = np.array(gt_assignment["detected_source_plane_normal"], dtype=float)
                copied_plane_point = np.array(gt_assignment["detected_copied_plane_point"], dtype=float)
                copied_plane_normal = np.array(gt_assignment["detected_copied_plane_normal"], dtype=float)
                item["near_world_on_gt_cap_plane"] = to_list(
                    intersect_pixel_with_plane(side["near_pixel"], source_plane_point, source_plane_normal, cal)
                )
                item["far_world_on_gt_cap_plane"] = to_list(
                    intersect_pixel_with_plane(side["far_pixel"], copied_plane_point, copied_plane_normal, cal)
                )
            else:
                item["near_world_on_gt_cap_plane"] = to_list(
                    pixel_to_world_on_z(side["near_pixel"], gt_assignment["detected_source_z"], cal)
                )
                item["far_world_on_gt_cap_plane"] = to_list(
                    pixel_to_world_on_z(side["far_pixel"], gt_assignment["detected_copied_z"], cal)
                )
            item["vector_world_on_gt_cap_planes"] = to_list(
                np.array(item["far_world_on_gt_cap_plane"]) - np.array(item["near_world_on_gt_cap_plane"])
            )
            item["unit_world_on_gt_cap_planes"] = to_list(unit(item["vector_world_on_gt_cap_planes"]))
        if obj_assignment is not None:
            source_plane_point = np.array(obj_assignment["source_plane_point"], dtype=float)
            copied_plane_point = np.array(obj_assignment["copied_plane_point"], dtype=float)
            plane_normal = np.array(obj_assignment["plane_normal_source_to_copied"], dtype=float)
            near_obj = intersect_pixel_with_plane(side["near_pixel"], source_plane_point, plane_normal, cal)
            far_obj = intersect_pixel_with_plane(side["far_pixel"], copied_plane_point, plane_normal, cal)
            item["near_world_on_obj_cap_plane"] = to_list(near_obj)
            item["far_world_on_obj_cap_plane"] = to_list(far_obj)
            item["vector_world_on_obj_cap_planes"] = to_list(far_obj - near_obj)
            item["unit_world_on_obj_cap_planes"] = to_list(unit(far_obj - near_obj))
        side_json.append(item)

    cap_stroke_loop_json = []
    cap_stroke_loop_items = build_cap_stroke_loop_items(
        cap_endpoint_source,
        strokes,
        straightness_threshold=args.curve_straightness_thresh,
        resample_step=args.curve_resample_step_px,
        curve_min_chord_px=args.curve_min_chord_px,
        curve_min_p90_chord_dev_px=args.curve_min_p90_chord_dev_px,
        curve_min_dev_ratio=args.curve_min_dev_ratio,
        curve_min_pca_rms_px=args.curve_min_pca_rms_px,
    )
    cap_stroke_loop_items = normalize_cap_stroke_loop_items(
        cap_stroke_loop_items,
        strokes,
        straightness_threshold=args.curve_straightness_thresh,
        resample_step=args.curve_resample_step_px,
        curve_min_chord_px=args.curve_min_chord_px,
        curve_min_p90_chord_dev_px=args.curve_min_p90_chord_dev_px,
        curve_min_dev_ratio=args.curve_min_dev_ratio,
        curve_min_pca_rms_px=args.curve_min_pca_rms_px,
    )
    cap_stroke_loop_items, cap_loop_order_debug = validate_cap_stroke_loop_order(cap_stroke_loop_items)
    for item in cap_stroke_loop_items:
        source_polyline_2d = np.asarray(item["polyline_2d"], dtype=float)
        copied_polyline_2d = source_polyline_2d + np.asarray(copied_offset, dtype=float)
        loop_item = {
            "stroke": int(item["stroke"]),
            "straightness": float(item["straightness"]),
            "is_curved": bool(item["is_curved"]),
            "curve_metrics": item.get("curve_metrics", {}),
            "polyline_point_count": int(len(source_polyline_2d)),
            "start_group_index": int(item["start_group_index"]),
            "end_group_index": int(item["end_group_index"]),
            "source_polyline_2d": [to_list(p) for p in source_polyline_2d],
            "copied_polyline_2d": [to_list(p) for p in copied_polyline_2d],
            "source_polyline_world_min_norm": [to_list(p) for p in project_polyline_min_norm(source_polyline_2d, cal)],
            "copied_polyline_world_min_norm": [to_list(p) for p in project_polyline_min_norm(copied_polyline_2d, cal)],
        }
        if gt_assignment is not None:
            if "detected_source_plane_point" in gt_assignment:
                source_plane_point = np.array(gt_assignment["detected_source_plane_point"], dtype=float)
                source_plane_normal = np.array(gt_assignment["detected_source_plane_normal"], dtype=float)
                copied_plane_point = np.array(gt_assignment["detected_copied_plane_point"], dtype=float)
                copied_plane_normal = np.array(gt_assignment["detected_copied_plane_normal"], dtype=float)
                loop_item["source_polyline_world_on_gt_cap_plane"] = [
                    to_list(p) for p in project_polyline_to_plane(source_polyline_2d, source_plane_point, source_plane_normal, cal)
                ]
                loop_item["copied_polyline_world_on_gt_cap_plane"] = [
                    to_list(p) for p in project_polyline_to_plane(copied_polyline_2d, copied_plane_point, copied_plane_normal, cal)
                ]
            else:
                loop_item["source_polyline_world_on_gt_cap_plane"] = [
                    to_list(p) for p in project_polyline_to_z(source_polyline_2d, gt_assignment["detected_source_z"], cal)
                ]
                loop_item["copied_polyline_world_on_gt_cap_plane"] = [
                    to_list(p) for p in project_polyline_to_z(copied_polyline_2d, gt_assignment["detected_copied_z"], cal)
                ]
        if obj_assignment is not None:
            source_plane_point = np.array(obj_assignment["source_plane_point"], dtype=float)
            copied_plane_point = np.array(obj_assignment["copied_plane_point"], dtype=float)
            plane_normal = np.array(obj_assignment["plane_normal_source_to_copied"], dtype=float)
            loop_item["source_polyline_world_on_obj_cap_plane"] = [
                to_list(p) for p in project_polyline_to_plane(source_polyline_2d, source_plane_point, plane_normal, cal)
            ]
            loop_item["copied_polyline_world_on_obj_cap_plane"] = [
                to_list(p) for p in project_polyline_to_plane(copied_polyline_2d, copied_plane_point, plane_normal, cal)
            ]
        cap_stroke_loop_json.append(loop_item)

    result = {
        "note": (
            "2D orthographic back-projection is not unique. world_min_norm uses the Moore-Penrose "
            "minimum-norm solution in the calibrated X/Y/Z basis relative to the anchor point."
        ),
        "rank_source": args.rank_source,
        "requested_rank": args.rank,
        "overlay": args.overlay,
        "endpoint_source_mode": args.endpoint_source,
        "selected_iou_entry": selected_iou,
        "selected_cluster_entry": selected,
        "cap_endpoint_source": cap_endpoint_source,
        "calibration": {
            "file": str(args.calibration),
            "groundtruth_file": str(gt_calibration_path) if gt_calibration_path is not None else None,
            "anchor_world": to_list(cal["anchor_world"]),
            "anchor_pixel_top_left": to_list(cal["origin_pixel"]),
            "basis_2x3_columns_u_x_u_y_u_z": cal["basis_2x3"].tolist(),
        },
        "groundtruth_cap_assignment": gt_assignment,
        "obj_cap_plane_assignment": obj_assignment,
        "obj_cap_plane_assignment_error": obj_assignment_error,
        "source_cap_endpoint_groups": [
            cap_group_to_json(group, copied_offset, cal, gt_assignment, obj_assignment)
            for group in cap_groups
        ],
        "cap_stroke_loop": cap_stroke_loop_json,
        "cap_loop_order_debug": cap_loop_order_debug,
        "copied_cap_offset_from_primary_side": {
            "stroke": int(primary_side["stroke"]),
            "near_endpoint": primary_side["near_endpoint"],
            "far_endpoint": primary_side["far_endpoint"],
            "vector_pixel": to_list(copied_offset),
            "vector_world_min_norm": to_list(primary_world_vector),
            "unit_world_min_norm": to_list(unit(primary_world_vector)),
            "length_world_min_norm": norm(primary_world_vector),
            "vector_world_on_gt_cap_planes": to_list(primary_gt_vector) if primary_gt_vector is not None else None,
            "unit_world_on_gt_cap_planes": to_list(unit(primary_gt_vector)) if primary_gt_vector is not None else None,
            "vector_world_on_obj_cap_planes": to_list(primary_obj_vector) if primary_obj_vector is not None else None,
            "unit_world_on_obj_cap_planes": to_list(unit(primary_obj_vector)) if primary_obj_vector is not None else None,
            "copy_selection_reason": primary_side.get("copy_selection_reason"),
            "copy_iou": primary_side.get("copy_iou"),
            "angle_to_majority_degrees": float(primary_side.get("angle_to_majority", 0.0)),
            "copy_direction_angle_tolerance_degrees": float(primary_side.get("copy_direction_angle_tol", args.copy_side_angle_thresh)),
            "rejected_copy_side_candidates": primary_side.get("rejected_copy_side_candidates", []),
        },
        "extrusion_direction": {
            "primary_from_longest_side_stroke_unit_world_min_norm": to_list(unit(primary_world_vector)),
            "primary_from_longest_side_stroke_vector_world_min_norm": to_list(primary_world_vector),
            "mean_from_all_side_strokes_unit_world_min_norm": to_list(unit(mean_world_vector)),
            "mean_from_all_side_strokes_vector_world_min_norm": to_list(mean_world_vector),
            "primary_from_gt_cap_planes_unit_world": to_list(unit(primary_gt_vector)) if primary_gt_vector is not None else None,
            "primary_from_gt_cap_planes_vector_world": to_list(primary_gt_vector) if primary_gt_vector is not None else None,
            "primary_from_obj_cap_plane_unit_world": to_list(unit(primary_obj_vector)) if primary_obj_vector is not None else None,
            "primary_from_obj_cap_plane_vector_world": to_list(primary_obj_vector) if primary_obj_vector is not None else None,
        },
        "side_strokes_oriented_near_to_far": side_json,
    }

    render = cal["raw"].get("render", {})
    blender_cap_debug_png = Path(args.blender_cap_debug_png)
    if not blender_cap_debug_png.is_absolute():
        blender_cap_debug_png = Path.cwd() / blender_cap_debug_png
    blender_cap_debug_json = Path(args.blender_cap_debug_json)
    if not blender_cap_debug_json.is_absolute():
        blender_cap_debug_json = Path.cwd() / blender_cap_debug_json
    save_blender_cap_debug_outputs(
        result,
        blender_cap_debug_png,
        blender_cap_debug_json,
        (
            int(render.get("resolution_y", 1000)),
            int(render.get("resolution_x", 1400)),
        ),
    )
    result["blender_cap_debug"] = {
        "png": str(blender_cap_debug_png),
        "json": str(blender_cap_debug_json),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote {output_path}")
    if selected is not None:
        print(f"Selected cluster rank {selected['rank']} cluster {selected['cluster']}")
    print(f"Endpoint source mode={args.endpoint_source}")
    print(f"Endpoint source={cap_endpoint_source.get('source')}")
    print(f"Primary side stroke {primary_side['stroke']} vector_pixel={to_list(copied_offset)}")
    print(f"Primary extrusion unit world={to_list(unit(primary_world_vector))}")
    if gt_assignment is not None:
        print(
            "GT cap-plane assignment: "
            f"detected source={gt_assignment['detected_source_cap_matches']} "
            f"z={gt_assignment['detected_source_z']}, "
            f"detected copied={gt_assignment['detected_copied_cap_matches']} "
            f"z={gt_assignment['detected_copied_z']}"
        )
        print(f"GT-constrained extrusion unit world={to_list(unit(primary_gt_vector))}")
    if obj_assignment is not None:
        print(
            "OBJ cap-plane assignment: "
            f"matched {obj_assignment['matched_cap']} to face {obj_assignment['matched_face_index']} "
            f"area={obj_assignment['matched_face_area_pixels']} px "
            f"distance={obj_assignment['matched_face_distance_to_camera']:.3f}"
        )
        if obj_assignment.get("match_quality_warning"):
            print(f"WARNING: {obj_assignment['match_quality_warning']}")
        print(f"OBJ-constrained extrusion unit world={to_list(unit(primary_obj_vector))}")
    elif obj_assignment_error is not None:
        print(f"WARNING: OBJ cap-plane assignment failed: {obj_assignment_error}")

    if args.reconstruct_blender:
        render_output = args.render_output if args.render_output else None
        run_blender_reconstruction(
            output_path,
            args.blend_output,
            args.solid_blend_output,
            render_output,
            args.blender_exe,
            args.blender_script_output,
        )
        print(f"Wrote Blender reconstruction {args.blend_output}")
        print(f"Wrote Blender solid reconstruction {args.solid_blend_output}")
        if render_output:
            print(f"Wrote Blender reconstruction render {render_output}")


if __name__ == "__main__":
    main()
