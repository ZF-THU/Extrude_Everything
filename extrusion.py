# extrusion_debug_merge.py
import argparse
import math
import os

import cv2
import numpy as np


# ============================================================
# 1. Preprocess + thinning
# ============================================================

def zhang_suen_thinning(binary, max_iter=100):
    img = (binary > 0).astype(np.uint8)

    for _ in range(max_iter):
        changed = False

        for step in [0, 1]:
            P = np.pad(img, ((1, 1), (1, 1)), mode="constant")

            P2 = P[:-2, 1:-1]
            P3 = P[:-2, 2:]
            P4 = P[1:-1, 2:]
            P5 = P[2:, 2:]
            P6 = P[2:, 1:-1]
            P7 = P[2:, :-2]
            P8 = P[1:-1, :-2]
            P9 = P[:-2, :-2]

            B = P2 + P3 + P4 + P5 + P6 + P7 + P8 + P9

            A = (
                ((P2 == 0) & (P3 == 1)).astype(np.uint8)
                + ((P3 == 0) & (P4 == 1)).astype(np.uint8)
                + ((P4 == 0) & (P5 == 1)).astype(np.uint8)
                + ((P5 == 0) & (P6 == 1)).astype(np.uint8)
                + ((P6 == 0) & (P7 == 1)).astype(np.uint8)
                + ((P7 == 0) & (P8 == 1)).astype(np.uint8)
                + ((P8 == 0) & (P9 == 1)).astype(np.uint8)
                + ((P9 == 0) & (P2 == 1)).astype(np.uint8)
            )

            if step == 0:
                marker = (
                    (img == 1)
                    & (B >= 2)
                    & (B <= 6)
                    & (A == 1)
                    & ((P2 * P4 * P6) == 0)
                    & ((P4 * P6 * P8) == 0)
                )
            else:
                marker = (
                    (img == 1)
                    & (B >= 2)
                    & (B <= 6)
                    & (A == 1)
                    & ((P2 * P4 * P8) == 0)
                    & ((P2 * P6 * P8) == 0)
                )

            if np.any(marker):
                img[marker] = 0
                changed = True

        if not changed:
            break

    return (img * 255).astype(np.uint8)


def remove_small_components(mask, min_area=8):
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)

    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            out[labels == i] = 255

    return out


def preprocess(
    image_path,
    invert=True,
    close_kernel=3,
    close_iter=1,
    dilate_iter=1,
    min_component_area=12,
    min_skel_component_area=6,
):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)

    if invert:
        _, bw = cv2.threshold(
            gray_blur,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )
    else:
        _, bw = cv2.threshold(
            gray_blur,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )

    if close_kernel > 0 and close_iter > 0:
        k = np.ones((close_kernel, close_kernel), np.uint8)
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k, iterations=close_iter)

    if dilate_iter > 0:
        bw = cv2.dilate(bw, np.ones((2, 2), np.uint8), iterations=dilate_iter)

    bw = remove_small_components(bw, min_area=min_component_area)

    try:
        skel = cv2.ximgproc.thinning(bw)
    except Exception:
        skel = zhang_suen_thinning(bw)

    skel = remove_small_components(skel, min_area=min_skel_component_area)

    return img, bw, skel


# ============================================================
# 2. Skeleton graph tracing
# ============================================================

def get_neighbors(mask, p):
    x, y = p
    h, w = mask.shape
    out = []

    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            xx = x + dx
            yy = y + dy
            if 0 <= xx < w and 0 <= yy < h and mask[yy, xx] > 0:
                out.append((xx, yy))

    return out


def edge_key(a, b):
    return tuple(sorted((a, b)))


def trace_strokes(skel, min_pixels=3):
    ys, xs = np.where(skel > 0)
    pixels = set(zip(xs.tolist(), ys.tolist()))

    if not pixels:
        return []

    degree = {}
    for p in pixels:
        degree[p] = len(get_neighbors(skel, p))

    nodes = {p for p, d in degree.items() if d != 2}

    visited_edges = set()
    strokes = []

    for start in list(nodes):
        for nb in get_neighbors(skel, start):
            ek = edge_key(start, nb)
            if ek in visited_edges:
                continue

            path = [start]
            prev = start
            cur = nb
            visited_edges.add(ek)

            while True:
                path.append(cur)

                if cur in nodes and cur != start:
                    break

                nbs = get_neighbors(skel, cur)
                nbs = [q for q in nbs if q != prev]

                if len(nbs) == 0:
                    break

                if len(nbs) > 1:
                    break

                nxt = nbs[0]
                ek = edge_key(cur, nxt)

                if ek in visited_edges:
                    break

                visited_edges.add(ek)
                prev, cur = cur, nxt

            if len(path) >= min_pixels:
                strokes.append(np.array(path, dtype=np.float32))

    # Trace pure cycles
    for p in pixels:
        nbs = get_neighbors(skel, p)
        unvisited = [q for q in nbs if edge_key(p, q) not in visited_edges]
        if not unvisited:
            continue

        start = p
        prev = p
        cur = unvisited[0]
        path = [start]
        visited_edges.add(edge_key(start, cur))

        while True:
            path.append(cur)

            nbs = get_neighbors(skel, cur)
            nbs = [q for q in nbs if q != prev]

            if not nbs:
                break

            nxt = None
            for q in nbs:
                if edge_key(cur, q) not in visited_edges:
                    nxt = q
                    break

            if nxt is None:
                break

            if nxt == start:
                visited_edges.add(edge_key(cur, nxt))
                path.append(start)
                break

            visited_edges.add(edge_key(cur, nxt))
            prev, cur = cur, nxt

        if len(path) >= min_pixels:
            strokes.append(np.array(path, dtype=np.float32))

    return strokes


# ============================================================
# 3. Stroke merge
# ============================================================

def stroke_arc_length(points):
    if len(points) < 2:
        return 0.0
    d = np.diff(points, axis=0)
    return float(np.sum(np.sqrt(np.sum(d * d, axis=1))))


def stroke_chord_length(points):
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(points[-1] - points[0]))


def stroke_straightness(points):
    arc = stroke_arc_length(points)
    chord = stroke_chord_length(points)
    if arc < 1e-8:
        return 0.0
    return chord / arc


def endpoint_tangent(stroke, at_start=True, k=8):
    if len(stroke) < 2:
        return None

    k = min(k, len(stroke) - 1)

    if at_start:
        p0 = stroke[0]
        p1 = stroke[k]
        v = p1 - p0
    else:
        p0 = stroke[-1]
        p1 = stroke[-1 - k]
        v = p0 - p1

    n = np.linalg.norm(v)
    if n < 1e-8:
        return None

    return v / n


def angle_between_vectors(v1, v2):
    if v1 is None or v2 is None:
        return 180.0

    v1 = v1 / (np.linalg.norm(v1) + 1e-12)
    v2 = v2 / (np.linalg.norm(v2) + 1e-12)

    c = float(np.dot(v1, v2))
    c = np.clip(c, -1.0, 1.0)

    return math.degrees(math.acos(c))


def merge_two_strokes(s1, end1, s2, end2):
    a = s1.copy()
    b = s2.copy()

    # connected endpoint should be a[-1] and b[0]
    if end1 == 0:
        a = a[::-1]

    if end2 == 1:
        b = b[::-1]

    if np.linalg.norm(a[-1] - b[0]) < 1e-6:
        merged = np.vstack([a, b[1:]])
    else:
        merged = np.vstack([a, b])

    return merged.astype(np.float32)


def can_merge_strokes(s1, end1, s2, end2, max_gap=10.0, max_angle=40.0):
    p1 = s1[0] if end1 == 0 else s1[-1]
    p2 = s2[0] if end2 == 0 else s2[-1]

    gap = float(np.linalg.norm(p1 - p2))
    if gap > max_gap:
        return False, gap, 180.0

    t1 = endpoint_tangent(s1, at_start=(end1 == 0))
    t2 = endpoint_tangent(s2, at_start=(end2 == 0))

    if t1 is None or t2 is None:
        return False, gap, 180.0

    angle = angle_between_vectors(t1, t2)
    angle = min(angle, 180.0 - angle)

    if angle > max_angle:
        return False, gap, angle

    return True, gap, angle


def merge_strokes_by_endpoint(
    strokes,
    max_gap=10.0,
    max_angle=40.0,
    max_iters=80,
    min_length_after_merge=3,
):
    strokes = [s.copy() for s in strokes if len(s) >= min_length_after_merge]

    for _ in range(max_iters):
        best = None
        n = len(strokes)

        for i in range(n):
            for j in range(i + 1, n):
                for end_i in [0, 1]:
                    for end_j in [0, 1]:
                        ok, gap, angle = can_merge_strokes(
                            strokes[i],
                            end_i,
                            strokes[j],
                            end_j,
                            max_gap=max_gap,
                            max_angle=max_angle,
                        )

                        if not ok:
                            continue

                        cost = gap + 0.2 * angle

                        if best is None or cost < best["cost"]:
                            best = {
                                "i": i,
                                "j": j,
                                "end_i": end_i,
                                "end_j": end_j,
                                "cost": cost,
                                "gap": gap,
                                "angle": angle,
                            }

        if best is None:
            break

        i = best["i"]
        j = best["j"]

        merged = merge_two_strokes(
            strokes[i],
            best["end_i"],
            strokes[j],
            best["end_j"],
        )

        new_strokes = []
        for k, s in enumerate(strokes):
            if k not in (i, j):
                new_strokes.append(s)

        new_strokes.append(merged)
        strokes = new_strokes

    return strokes


# ============================================================
# 4. Stroke geometry
# ============================================================

def fit_line_to_points(points):
    pts = points.astype(np.float64)
    center = pts.mean(axis=0)

    X = pts - center
    cov = X.T @ X / max(len(points), 1)

    vals, vecs = np.linalg.eigh(cov)
    direction = vecs[:, np.argmax(vals)]
    direction = direction / (np.linalg.norm(direction) + 1e-12)

    a = -direction[1]
    b = direction[0]
    c = -(a * center[0] + b * center[1])

    n = math.hypot(a, b)
    line = np.array([a / n, b / n, c / n], dtype=np.float64)

    return line, direction, center


def build_stroke_infos(strokes):
    infos = []

    for i, pts in enumerate(strokes):
        if len(pts) < 2:
            continue

        arc = stroke_arc_length(pts)
        chord = stroke_chord_length(pts)
        straight = stroke_straightness(pts)
        line, direction, center = fit_line_to_points(pts)

        infos.append(
            {
                "index": i,
                "points": pts,
                "arc": arc,
                "chord": chord,
                "straightness": straight,
                "line": line,
                "direction": direction,
                "center": center,
            }
        )

    return infos


def intersect_lines(l1, l2):
    p = np.cross(l1, l2)

    if abs(p[2]) < 1e-8:
        return None

    return np.array([p[0] / p[2], p[1] / p[2]], dtype=np.float64)


def point_line_distance(line, point):
    a, b, c = line
    x, y = point
    return abs(a * x + b * y + c)


def angle_line_to_vp(direction, center, vp):
    v = vp - center
    nv = np.linalg.norm(v)

    if nv < 1e-8:
        return 90.0

    v = v / nv
    cosang = abs(float(np.dot(direction, v)))
    cosang = np.clip(cosang, -1.0, 1.0)

    return math.degrees(math.acos(cosang))


# ============================================================
# 5. VP / parallel extrusion direction estimation
# ============================================================

def estimate_vp_from_strokes(
    stroke_infos,
    image_shape,
    min_length=60,
    min_straightness=0.9,
    dist_thresh=8.0,
    angle_thresh=12.0,
):
    h, w = image_shape[:2]

    line_strokes = [
        s for s in stroke_infos
        if s["arc"] >= min_length and s["straightness"] >= min_straightness
    ]

    if len(line_strokes) < 2:
        return None, line_strokes

    candidates = []

    for i in range(len(line_strokes)):
        for j in range(i + 1, len(line_strokes)):
            l1 = line_strokes[i]["line"]
            l2 = line_strokes[j]["line"]

            normal_dot = abs(l1[0] * l2[0] + l1[1] * l2[1])
            if normal_dot > 0.995:
                continue

            p = intersect_lines(l1, l2)
            if p is None:
                continue

            if abs(p[0]) > 12 * w or abs(p[1]) > 12 * h:
                continue

            candidates.append(p)

    if not candidates:
        return None, line_strokes

    best = None

    for vp in candidates:
        inliers = []
        score = 0.0

        for s in line_strokes:
            d = point_line_distance(s["line"], vp)
            a = angle_line_to_vp(s["direction"], s["center"], vp)

            if d <= dist_thresh and a <= angle_thresh:
                inliers.append(s)
                score += s["arc"] * s["straightness"] / (1.0 + d)

        if len(inliers) < 2:
            continue

        if best is None or score > best["score"]:
            best = {
                "mode": "vp",
                "vp": vp,
                "score": float(score),
                "inliers": inliers,
            }

    return best, line_strokes


def angle_between_dirs(d1, d2):
    d1 = d1 / (np.linalg.norm(d1) + 1e-12)
    d2 = d2 / (np.linalg.norm(d2) + 1e-12)
    c = abs(float(np.dot(d1, d2)))
    c = np.clip(c, -1.0, 1.0)
    return math.degrees(math.acos(c))


def fallback_parallel_direction(line_strokes, angle_cluster_thresh=18.0):
    if len(line_strokes) < 1:
        return None

    best_group = []

    for s in line_strokes:
        base_dir = s["direction"]

        group = []
        for t in line_strokes:
            a = angle_between_dirs(base_dir, t["direction"])
            if a <= angle_cluster_thresh:
                group.append(t)

        if len(group) > len(best_group):
            best_group = group

    if len(best_group) < 1:
        return None

    dirs = []
    ref = best_group[0]["direction"]

    for s in best_group:
        d = s["direction"].copy()
        if np.dot(d, ref) < 0:
            d = -d
        dirs.append(d)

    v = np.mean(dirs, axis=0)
    v = v / (np.linalg.norm(v) + 1e-12)

    if v[1] < 0:
        v = -v

    score = sum(s["arc"] * s["straightness"] for s in best_group)

    return {
        "mode": "parallel",
        "direction": v,
        "score": float(score),
        "inliers": best_group,
    }


def foreground_bbox(skel, margin_ratio=0.35):
    ys, xs = np.where(skel > 0)

    if len(xs) == 0:
        return None

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()

    w = x1 - x0 + 1
    h = y1 - y0 + 1

    mx = int(w * margin_ratio)
    my = int(h * margin_ratio)

    return x0 - mx, y0 - my, x1 + mx, y1 + my


def point_inside_bbox(p, bbox):
    x0, y0, x1, y1 = bbox
    return x0 <= p[0] <= x1 and y0 <= p[1] <= y1


def choose_extrusion_model(args, infos, img_shape, skel):
    line_strokes = [
        s for s in infos
        if s["arc"] >= args.min_stroke_length
        and s["straightness"] >= args.straightness
    ]

    if args.force_parallel:
        model = fallback_parallel_direction(
            line_strokes,
            angle_cluster_thresh=args.parallel_angle_thresh,
        )
        return model, line_strokes

    if args.vp is not None:
        vp = np.array(args.vp, dtype=np.float64)

        inliers = []
        score = 0.0

        for s in line_strokes:
            d = point_line_distance(s["line"], vp)
            a = angle_line_to_vp(s["direction"], s["center"], vp)

            if d <= args.dist_thresh and a <= args.angle_thresh:
                inliers.append(s)
                score += s["arc"] * s["straightness"] / (1.0 + d)

        model = {
            "mode": "vp",
            "vp": vp,
            "score": float(score),
            "inliers": inliers,
        }
        return model, line_strokes

    vp_model, line_strokes = estimate_vp_from_strokes(
        infos,
        img_shape,
        min_length=args.min_stroke_length,
        min_straightness=args.straightness,
        dist_thresh=args.dist_thresh,
        angle_thresh=args.angle_thresh,
    )

    parallel_model = fallback_parallel_direction(
        line_strokes,
        angle_cluster_thresh=args.parallel_angle_thresh,
    )

    model = vp_model

    if model is not None and args.reject_vp_near_object:
        bbox = foreground_bbox(skel, margin_ratio=args.vp_reject_bbox_margin)
        if bbox is not None and point_inside_bbox(model["vp"], bbox):
            print("[info] Rejecting finite VP because it is inside/near object bbox.")
            model = parallel_model

    if model is not None and parallel_model is not None:
        if model["score"] < args.vp_score_ratio * parallel_model["score"]:
            print("[info] Parallel model is comparable to VP model; using parallel.")
            model = parallel_model

    if model is None:
        model = parallel_model

    return model, line_strokes


# ============================================================
# 6. Side stroke mask + cap candidate extraction
# ============================================================

def draw_polyline_mask(mask, points, thickness):
    pts = points.reshape(-1, 1, 2).astype(np.int32)
    cv2.polylines(mask, [pts], False, 255, thickness, cv2.LINE_AA)


def make_side_mask(skel, model, side_thickness=5):
    mask = np.zeros_like(skel)

    for s in model["inliers"]:
        draw_polyline_mask(mask, s["points"], side_thickness)

    mask = cv2.bitwise_and(mask, skel)
    mask = cv2.dilate(mask, np.ones((2, 2), np.uint8), iterations=1)

    return mask


def count_endpoints(component_mask):
    ys, xs = np.where(component_mask > 0)

    count = 0

    for x, y in zip(xs, ys):
        x0 = max(0, x - 1)
        x1 = min(component_mask.shape[1], x + 2)
        y0 = max(0, y - 1)
        y1 = min(component_mask.shape[0], y + 2)

        patch = component_mask[y0:y1, x0:x1]
        n = int(np.count_nonzero(patch)) - 1

        if n <= 1:
            count += 1

    return count


def component_center(mask):
    ys, xs = np.where(mask > 0)

    if len(xs) == 0:
        return None

    return np.array([xs.mean(), ys.mean()], dtype=np.float64)


def extract_cap_candidates(non_side_skel, min_pixels=40):
    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        non_side_skel,
        connectivity=8,
    )

    candidates = []

    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_pixels:
            continue

        comp = np.zeros_like(non_side_skel)
        comp[labels == i] = 255

        endpoints = count_endpoints(comp)
        center = component_center(comp)

        if endpoints == 0:
            closedness = 1.0
        elif endpoints <= 2:
            closedness = 0.75
        elif endpoints <= 6:
            closedness = 0.4
        else:
            closedness = 0.15

        score = area * (0.5 + closedness)

        candidates.append(
            {
                "mask": comp,
                "area": area,
                "endpoints": endpoints,
                "closedness": closedness,
                "score": score,
                "center": center,
            }
        )

    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates


# ============================================================
# 7. Visualization
# ============================================================

def colorize(out, mask, color):
    out[mask > 0] = color


def draw_model_arrow(out, model, candidates):
    h, w = out.shape[:2]

    if model["mode"] == "vp":
        vp = model["vp"]

        if 0 <= vp[0] < w and 0 <= vp[1] < h:
            cv2.circle(out, (int(vp[0]), int(vp[1])), 8, (0, 0, 255), -1)
            cv2.putText(
                out,
                "extrusion VP",
                (int(vp[0]) + 10, int(vp[1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
        else:
            cv2.putText(
                out,
                f"extrusion VP outside: ({vp[0]:.1f}, {vp[1]:.1f})",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        for c in candidates[:2]:
            ctr = c["center"]
            if ctr is None:
                continue

            d = vp - ctr
            n = np.linalg.norm(d)
            if n < 1e-8:
                continue

            d = d / n
            p1 = ctr
            p2 = ctr + d * min(160, n)

            cv2.arrowedLine(
                out,
                (int(p1[0]), int(p1[1])),
                (int(p2[0]), int(p2[1])),
                (0, 0, 255),
                2,
                tipLength=0.15,
            )

    else:
        d = model["direction"]

        for c in candidates[:2]:
            ctr = c["center"]
            if ctr is None:
                continue

            p1 = ctr
            p2 = ctr + d * 160

            cv2.arrowedLine(
                out,
                (int(p1[0]), int(p1[1])),
                (int(p2[0]), int(p2[1])),
                (0, 0, 255),
                2,
                tipLength=0.15,
            )


def draw_result(img, skel, model, side_mask, candidates, output):
    out = img.copy()
    out[skel > 0] = (0, 0, 0)

    colorize(out, side_mask, (0, 0, 255))

    colors = [
        (0, 180, 0),
        (255, 80, 0),
        (180, 0, 180),
        (0, 180, 180),
    ]

    for i, c in enumerate(candidates[:4]):
        thick = cv2.dilate(c["mask"], np.ones((3, 3), np.uint8), iterations=1)
        colorize(out, thick, colors[i])

        ctr = c["center"]
        if ctr is not None:
            label = f"cap_candidate_{i + 1}"
            if i == 0:
                label += " / base?"

            cv2.putText(
                out,
                label,
                (int(ctr[0]) + 8, int(ctr[1]) - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                colors[i],
                2,
                cv2.LINE_AA,
            )

    draw_model_arrow(out, model, candidates)

    h, _ = out.shape[:2]

    if model["mode"] == "vp":
        vp = model["vp"]
        info = (
            f"mode=VP, VP=({vp[0]:.1f},{vp[1]:.1f}), "
            f"side_strokes={len(model['inliers'])}, score={model['score']:.1f}"
        )
    else:
        d = model["direction"]
        info = (
            f"mode=parallel, dir=({d[0]:.2f},{d[1]:.2f}), "
            f"side_strokes={len(model['inliers'])}, score={model['score']:.1f}"
        )

    cv2.putText(
        out,
        info,
        (20, h - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.imwrite(output, out)
    return out


# ============================================================
# 8. Debug outputs
# ============================================================

def ensure_dir(path):
    if path is not None:
        os.makedirs(path, exist_ok=True)


def random_color(i):
    rng = np.random.default_rng(i + 12345)
    c = rng.integers(40, 230, size=3)
    return int(c[0]), int(c[1]), int(c[2])


def draw_strokes_image(shape, strokes, thickness=2, annotate=True):
    h, w = shape[:2]
    out = np.full((h, w, 3), 255, dtype=np.uint8)

    for i, pts in enumerate(strokes):
        color = random_color(i)
        pts_i = pts.reshape(-1, 1, 2).astype(np.int32)

        cv2.polylines(out, [pts_i], False, color, thickness, cv2.LINE_AA)

        if annotate and len(pts) > 0:
            ctr = pts.mean(axis=0)
            cv2.putText(
                out,
                str(i),
                (int(ctr[0]), int(ctr[1])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

    return out


def draw_stroke_infos_image(shape, infos, thickness=2):
    h, w = shape[:2]
    out = np.full((h, w, 3), 255, dtype=np.uint8)

    for s in infos:
        pts = s["points"].reshape(-1, 1, 2).astype(np.int32)
        straight = s["straightness"]

        if straight >= 0.95:
            color = (0, 180, 0)
        elif straight >= 0.88:
            color = (0, 180, 180)
        elif straight >= 0.75:
            color = (255, 120, 0)
        else:
            color = (180, 0, 180)

        cv2.polylines(out, [pts], False, color, thickness, cv2.LINE_AA)

        c = s["center"]
        label = f"{s['index']} s={straight:.2f}"
        cv2.putText(
            out,
            label,
            (int(c[0]), int(c[1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    legend = [
        "green: straightness >= 0.95",
        "yellow: >= 0.88",
        "blue: >= 0.75",
        "purple: curved",
    ]

    y = 22
    for t in legend:
        cv2.putText(
            out,
            t,
            (15, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        y += 22

    return out


def draw_line_stroke_candidates_image(shape, line_strokes, thickness=3):
    h, w = shape[:2]
    out = np.full((h, w, 3), 255, dtype=np.uint8)

    for i, s in enumerate(line_strokes):
        pts = s["points"].reshape(-1, 1, 2).astype(np.int32)
        color = random_color(i)

        cv2.polylines(out, [pts], False, color, thickness, cv2.LINE_AA)

        c = s["center"]
        label = f"{s['index']} len={s['arc']:.0f} str={s['straightness']:.2f}"
        cv2.putText(
            out,
            label,
            (int(c[0]), int(c[1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    cv2.putText(
        out,
        "Line stroke candidates used for direction estimation",
        (15, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )

    return out


def draw_selected_side_strokes_image(shape, model, thickness=4):
    h, w = shape[:2]
    out = np.full((h, w, 3), 255, dtype=np.uint8)

    for i, s in enumerate(model["inliers"]):
        pts = s["points"].reshape(-1, 1, 2).astype(np.int32)

        cv2.polylines(out, [pts], False, (0, 0, 255), thickness, cv2.LINE_AA)

        c = s["center"]
        cv2.putText(
            out,
            f"side {i}: stroke {s['index']}",
            (int(c[0]), int(c[1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    if model["mode"] == "vp":
        vp = model["vp"]
        cv2.putText(
            out,
            f"mode=VP, VP=({vp[0]:.1f}, {vp[1]:.1f})",
            (15, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        if 0 <= vp[0] < w and 0 <= vp[1] < h:
            cv2.circle(out, (int(vp[0]), int(vp[1])), 7, (0, 0, 255), -1)

    else:
        d = model["direction"]
        cv2.putText(
            out,
            f"mode=parallel, dir=({d[0]:.2f}, {d[1]:.2f})",
            (15, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        center = np.array([w * 0.5, h * 0.5])
        p1 = center - d * 100
        p2 = center + d * 100
        cv2.arrowedLine(
            out,
            (int(p1[0]), int(p1[1])),
            (int(p2[0]), int(p2[1])),
            (0, 0, 255),
            3,
            tipLength=0.15,
        )

    return out


def draw_non_side_skeleton_image(skel, side_mask, non_side):
    h, w = skel.shape
    out = np.full((h, w, 3), 255, dtype=np.uint8)

    out[skel > 0] = (180, 180, 180)
    out[side_mask > 0] = (0, 0, 255)
    out[non_side > 0] = (0, 0, 0)

    cv2.putText(
        out,
        "gray=original skeleton, red=removed side, black=remaining non-side",
        (15, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )

    return out


def draw_cap_candidates_debug(shape, candidates, max_draw=8):
    h, w = shape[:2]
    out = np.full((h, w, 3), 255, dtype=np.uint8)

    for i, c in enumerate(candidates[:max_draw]):
        color = random_color(i)
        mask = cv2.dilate(c["mask"], np.ones((3, 3), np.uint8), iterations=1)
        out[mask > 0] = color

        ctr = c["center"]
        if ctr is not None:
            label = (
                f"#{i} area={c['area']} "
                f"end={c['endpoints']} "
                f"closed={c['closedness']:.2f}"
            )
            cv2.putText(
                out,
                label,
                (int(ctr[0]), int(ctr[1])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

    cv2.putText(
        out,
        "Cap candidates after side-stroke removal",
        (15, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )

    return out


def draw_skeleton_nodes_debug(skel):
    h, w = skel.shape
    out = np.full((h, w, 3), 255, dtype=np.uint8)
    out[skel > 0] = (0, 0, 0)

    ys, xs = np.where(skel > 0)

    for x, y in zip(xs, ys):
        degree = len(get_neighbors(skel, (x, y)))

        if degree == 1:
            cv2.circle(out, (x, y), 3, (0, 0, 255), -1)
        elif degree != 2:
            cv2.circle(out, (x, y), 3, (255, 0, 0), -1)

    cv2.putText(
        out,
        "red=endpoints, blue=junction/degree!=2",
        (15, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )

    return out


def save_debug_report(
    path,
    args,
    raw_strokes,
    merged_strokes,
    infos,
    line_strokes,
    model,
    candidates,
):
    with open(path, "w", encoding="utf-8") as f:
        f.write("==== Debug Report ====\n\n")

        f.write("Arguments:\n")
        for k, v in vars(args).items():
            f.write(f"  {k}: {v}\n")

        f.write("\nCounts:\n")
        f.write(f"  raw traced strokes: {len(raw_strokes)}\n")
        f.write(f"  merged strokes: {len(merged_strokes)}\n")
        f.write(f"  stroke infos: {len(infos)}\n")
        f.write(f"  line stroke candidates: {len(line_strokes)}\n")
        f.write(f"  selected side strokes: {len(model['inliers'])}\n")
        f.write(f"  cap candidates: {len(candidates)}\n")

        f.write("\nMerged stroke infos:\n")
        for s in infos:
            c = s["center"]
            f.write(
                f"  stroke {s['index']:03d}: "
                f"arc={s['arc']:.1f}, "
                f"chord={s['chord']:.1f}, "
                f"straightness={s['straightness']:.3f}, "
                f"center=({c[0]:.1f},{c[1]:.1f})\n"
            )

        f.write("\nLine stroke candidates:\n")
        for s in line_strokes:
            c = s["center"]
            f.write(
                f"  stroke {s['index']:03d}: "
                f"arc={s['arc']:.1f}, "
                f"straightness={s['straightness']:.3f}, "
                f"center=({c[0]:.1f},{c[1]:.1f})\n"
            )

        f.write("\nSelected extrusion model:\n")
        f.write(f"  mode: {model['mode']}\n")
        f.write(f"  score: {model['score']:.3f}\n")
        f.write(f"  selected side strokes: {len(model['inliers'])}\n")

        if model["mode"] == "vp":
            vp = model["vp"]
            f.write(f"  VP: ({vp[0]:.2f}, {vp[1]:.2f})\n")
        else:
            d = model["direction"]
            f.write(f"  direction: ({d[0]:.4f}, {d[1]:.4f})\n")

        f.write("\nSelected side stroke inliers:\n")
        for s in model["inliers"]:
            c = s["center"]
            f.write(
                f"  stroke {s['index']:03d}: "
                f"arc={s['arc']:.1f}, "
                f"straightness={s['straightness']:.3f}, "
                f"center=({c[0]:.1f},{c[1]:.1f})\n"
            )

        f.write("\nCap candidates:\n")
        for i, c in enumerate(candidates):
            ctr = c["center"]
            ctr_txt = "None" if ctr is None else f"({ctr[0]:.1f},{ctr[1]:.1f})"

            f.write(
                f"  candidate {i:03d}: "
                f"area={c['area']}, "
                f"endpoints={c['endpoints']}, "
                f"closedness={c['closedness']:.2f}, "
                f"score={c['score']:.1f}, "
                f"center={ctr_txt}\n"
            )


def save_debug_outputs(
    debug_dir,
    args,
    img,
    bw,
    skel,
    raw_strokes,
    merged_strokes,
    infos,
    line_strokes,
    model,
    side_mask,
    non_side,
    candidates,
):
    if debug_dir is None:
        return

    ensure_dir(debug_dir)

    cv2.imwrite(os.path.join(debug_dir, "00_input.png"), img)
    cv2.imwrite(os.path.join(debug_dir, "01_binary.png"), bw)
    cv2.imwrite(os.path.join(debug_dir, "02_skeleton.png"), skel)

    nodes_img = draw_skeleton_nodes_debug(skel)
    cv2.imwrite(os.path.join(debug_dir, "02b_skeleton_nodes.png"), nodes_img)

    raw_img = draw_strokes_image(img.shape, raw_strokes, thickness=2, annotate=True)
    cv2.imwrite(os.path.join(debug_dir, "03a_raw_strokes.png"), raw_img)

    merged_img = draw_strokes_image(img.shape, merged_strokes, thickness=2, annotate=True)
    cv2.imwrite(os.path.join(debug_dir, "03b_merged_strokes.png"), merged_img)

    stroke_info_img = draw_stroke_infos_image(img.shape, infos, thickness=2)
    cv2.imwrite(os.path.join(debug_dir, "04_stroke_info.png"), stroke_info_img)

    line_candidates_img = draw_line_stroke_candidates_image(
        img.shape,
        line_strokes,
        thickness=3,
    )
    cv2.imwrite(
        os.path.join(debug_dir, "05_line_stroke_candidates.png"),
        line_candidates_img,
    )

    selected_side_img = draw_selected_side_strokes_image(
        img.shape,
        model,
        thickness=4,
    )
    cv2.imwrite(
        os.path.join(debug_dir, "06_selected_side_strokes.png"),
        selected_side_img,
    )

    non_side_img = draw_non_side_skeleton_image(skel, side_mask, non_side)
    cv2.imwrite(os.path.join(debug_dir, "07_non_side_skeleton.png"), non_side_img)

    cap_candidates_img = draw_cap_candidates_debug(
        img.shape,
        candidates,
        max_draw=8,
    )
    cv2.imwrite(
        os.path.join(debug_dir, "08_cap_candidates.png"),
        cap_candidates_img,
    )

    save_debug_report(
        os.path.join(debug_dir, "debug_report.txt"),
        args,
        raw_strokes,
        merged_strokes,
        infos,
        line_strokes,
        model,
        candidates,
    )


def print_debug(raw_strokes, merged_strokes, infos, line_strokes, model, candidates, output, debug_dir):
    print("==== Result ====")
    print(f"output: {output}")
    if debug_dir:
        print(f"debug dir: {debug_dir}")

    print(f"raw traced strokes: {len(raw_strokes)}")
    print(f"merged strokes: {len(merged_strokes)}")
    print(f"stroke infos: {len(infos)}")
    print(f"straight line stroke candidates: {len(line_strokes)}")
    print()

    print(f"mode: {model['mode']}")
    print(f"side strokes: {len(model['inliers'])}")
    print(f"score: {model['score']:.2f}")

    if model["mode"] == "vp":
        vp = model["vp"]
        print(f"vanishing point: ({vp[0]:.2f}, {vp[1]:.2f})")
    else:
        d = model["direction"]
        print(f"parallel direction: ({d[0]:.3f}, {d[1]:.3f})")

    print()
    print("Side stroke inliers:")
    for i, s in enumerate(model["inliers"]):
        c = s["center"]
        print(
            f"  #{i + 1}: stroke={s['index']}, "
            f"arc={s['arc']:.1f}, "
            f"straightness={s['straightness']:.3f}, "
            f"center=({c[0]:.1f},{c[1]:.1f})"
        )

    print()
    print("Cap candidates:")
    for i, c in enumerate(candidates[:8]):
        ctr = c["center"]
        ctr_text = "None" if ctr is None else f"({ctr[0]:.1f}, {ctr[1]:.1f})"
        print(
            f"  #{i + 1}: "
            f"area={c['area']}, "
            f"endpoints={c['endpoints']}, "
            f"closedness={c['closedness']:.2f}, "
            f"score={c['score']:.1f}, "
            f"center={ctr_text}"
        )


# ============================================================
# 9. Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("image")
    parser.add_argument("--output", default="result.png")
    parser.add_argument("--debug-dir", default=None)

    parser.add_argument(
        "--no-invert",
        action="store_true",
        help="Use this if input is white strokes on black background.",
    )

    # Preprocess controls
    parser.add_argument("--close-kernel", type=int, default=3)
    parser.add_argument("--close-iter", type=int, default=1)
    parser.add_argument("--dilate-iter", type=int, default=1)
    parser.add_argument("--min-component-area", type=int, default=12)
    parser.add_argument("--min-skel-component-area", type=int, default=6)

    # Raw tracing
    parser.add_argument("--trace-min-pixels", type=int, default=3)

    # Stroke merge
    parser.add_argument("--merge-gap", type=float, default=10.0)
    parser.add_argument("--merge-angle", type=float, default=40.0)
    parser.add_argument("--merge-iters", type=int, default=80)

    # Direction candidate filtering
    parser.add_argument("--min-stroke-length", type=float, default=30)
    parser.add_argument("--straightness", type=float, default=0.88)

    # VP fitting
    parser.add_argument("--dist-thresh", type=float, default=8.0)
    parser.add_argument("--angle-thresh", type=float, default=12.0)
    parser.add_argument("--vp", nargs=2, type=float, default=None)

    # Parallel fallback / preference
    parser.add_argument(
        "--force-parallel",
        action="store_true",
        help="Force weak-perspective / parallel extrusion direction.",
    )
    parser.add_argument(
        "--reject-vp-near-object",
        action="store_true",
        help="Reject VP if it lies inside or near the foreground bbox.",
    )
    parser.add_argument("--vp-reject-bbox-margin", type=float, default=0.35)
    parser.add_argument("--vp-score-ratio", type=float, default=1.25)
    parser.add_argument("--parallel-angle-thresh", type=float, default=18.0)

    # Masks and cap extraction
    parser.add_argument("--side-thickness", type=int, default=4)
    parser.add_argument("--min-cap-pixels", type=int, default=40)

    args = parser.parse_args()

    img, bw, skel = preprocess(
        args.image,
        invert=not args.no_invert,
        close_kernel=args.close_kernel,
        close_iter=args.close_iter,
        dilate_iter=args.dilate_iter,
        min_component_area=args.min_component_area,
        min_skel_component_area=args.min_skel_component_area,
    )

    raw_strokes = trace_strokes(skel, min_pixels=args.trace_min_pixels)

    merged_strokes = merge_strokes_by_endpoint(
        raw_strokes,
        max_gap=args.merge_gap,
        max_angle=args.merge_angle,
        max_iters=args.merge_iters,
    )

    infos = build_stroke_infos(merged_strokes)

    model, line_strokes = choose_extrusion_model(args, infos, img.shape, skel)

    if model is None:
        raise RuntimeError(
            "Could not estimate extrusion direction. "
            "Try --force-parallel, lower --straightness, lower --min-stroke-length, "
            "increase --merge-gap, or provide --vp manually."
        )

    if len(model["inliers"]) == 0:
        raise RuntimeError(
            "Extrusion direction model has no side stroke inliers. "
            "Try lowering --straightness or --min-stroke-length."
        )

    side_mask = make_side_mask(
        skel,
        model,
        side_thickness=args.side_thickness,
    )

    non_side = cv2.bitwise_and(skel, cv2.bitwise_not(side_mask))

    candidates = extract_cap_candidates(
        non_side,
        min_pixels=args.min_cap_pixels,
    )

    save_debug_outputs(
        args.debug_dir,
        args,
        img,
        bw,
        skel,
        raw_strokes,
        merged_strokes,
        infos,
        line_strokes,
        model,
        side_mask,
        non_side,
        candidates,
    )

    draw_result(
        img,
        skel,
        model,
        side_mask,
        candidates,
        args.output,
    )

    print_debug(
        raw_strokes,
        merged_strokes,
        infos,
        line_strokes,
        model,
        candidates,
        args.output,
        args.debug_dir,
    )


if __name__ == "__main__":
    main()