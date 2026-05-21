# extrusion_debug_caploop.py
# Robust debug version for sketch extrusion detection.
# Key fix: trace_strokes uses crossing number instead of raw degree!=2,
# avoiding false junctions caused by 8-neighbor stair-step skeleton artifacts.

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
        _, bw = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        _, bw = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

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
# 2. Skeleton graph utilities
# ============================================================

def get_neighbors(mask, p):
    """
    Safer skeleton neighbors.

    4-neighbor is always allowed.
    Diagonal neighbor is allowed only when it is not a corner-cutting shortcut.
    This reduces false junctions from 8-neighbor stair-step artifacts.
    """
    x, y = p
    h, w = mask.shape
    out = []

    # 4-neighbors
    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        xx, yy = x + dx, y + dy
        if 0 <= xx < w and 0 <= yy < h and mask[yy, xx] > 0:
            out.append((xx, yy))

    # diagonal neighbors
    for dx, dy in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
        xx, yy = x + dx, y + dy
        if not (0 <= xx < w and 0 <= yy < h):
            continue
        if mask[yy, xx] == 0:
            continue

        side1_x, side1_y = x + dx, y
        side2_x, side2_y = x, y + dy

        side1 = 0 <= side1_x < w and 0 <= side1_y < h and mask[side1_y, side1_x] > 0
        side2 = 0 <= side2_x < w and 0 <= side2_y < h and mask[side2_y, side2_x] > 0

        # If an orthogonal route exists, diagonal is likely only a shortcut.
        if not side1 and not side2:
            out.append((xx, yy))

    return out


def edge_key(a, b):
    return tuple(sorted((a, b)))


def raw_8_neighbor_values(mask, p):
    """
    Return P2...P9 around p:
      P2=N, P3=NE, P4=E, P5=SE, P6=S, P7=SW, P8=W, P9=NW.
    """
    x, y = p
    h, w = mask.shape

    coords = [
        (x, y - 1),
        (x + 1, y - 1),
        (x + 1, y),
        (x + 1, y + 1),
        (x, y + 1),
        (x - 1, y + 1),
        (x - 1, y),
        (x - 1, y - 1),
    ]

    vals = []
    for xx, yy in coords:
        vals.append(1 if 0 <= xx < w and 0 <= yy < h and mask[yy, xx] > 0 else 0)
    return vals


def crossing_number(mask, p):
    """
    Skeleton connectivity/crossing number.
    Endpoint usually CN=1, regular curve point CN=2, branch CN>=3.
    """
    vals = raw_8_neighbor_values(mask, p)
    transitions = 0
    for i in range(8):
        if vals[i] != vals[(i + 1) % 8]:
            transitions += 1
    return transitions // 2


def skeleton_node_type(mask, p):
    deg = len(get_neighbors(mask, p))
    cn = crossing_number(mask, p)

    if deg == 0:
        return "isolated"
    if deg == 1:
        return "endpoint"
    if cn >= 3:
        return "branch"
    return "regular"


def choose_best_continuation(prev, cur, candidates):
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    prev_v = np.array([cur[0] - prev[0], cur[1] - prev[1]], dtype=np.float64)
    n = np.linalg.norm(prev_v)
    if n < 1e-8:
        return candidates[0]
    prev_v /= n

    best = None
    best_score = -1e9
    for q in candidates:
        next_v = np.array([q[0] - cur[0], q[1] - cur[1]], dtype=np.float64)
        m = np.linalg.norm(next_v)
        if m < 1e-8:
            continue
        next_v /= m
        score = float(np.dot(prev_v, next_v))
        if score > best_score:
            best_score = score
            best = q
    return best


# ============================================================
# 3. Robust skeleton tracing
# ============================================================

def trace_strokes(skel, min_pixels=3):
    """
    Robust skeleton tracing.

    Instead of cutting at every raw degree!=2 pixel, this uses crossing number:
      endpoint = true endpoint
      branch = true topological branch
      regular = everything else, including many raw degree=3 stair-step artifacts.
    """
    ys, xs = np.where(skel > 0)
    pixels = set(zip(xs.tolist(), ys.tolist()))

    if not pixels:
        return []

    node_type = {}
    nodes = set()
    for p in pixels:
        t = skeleton_node_type(skel, p)
        node_type[p] = t
        if t in ("endpoint", "branch"):
            nodes.add(p)

    visited_edges = set()
    strokes = []

    # Trace paths from true endpoints / true branches
    for start in list(nodes):
        for nb in get_neighbors(skel, start):
            ek = edge_key(start, nb)
            if ek in visited_edges:
                continue

            path = [start]
            prev = start
            cur = nb
            visited_edges.add(ek)
            safety = 0

            while True:
                safety += 1
                if safety > 20000:
                    break

                path.append(cur)

                if cur in nodes and cur != start:
                    break

                nbs = [q for q in get_neighbors(skel, cur) if q != prev]
                nbs_unvisited = [q for q in nbs if edge_key(cur, q) not in visited_edges]
                if not nbs_unvisited:
                    break

                nxt = choose_best_continuation(prev, cur, nbs_unvisited)
                if nxt is None:
                    break

                visited_edges.add(edge_key(cur, nxt))
                prev, cur = cur, nxt

            if len(path) >= min_pixels:
                strokes.append(np.array(path, dtype=np.float32))

    # Trace remaining pure cycles / unvisited chains
    for p in pixels:
        for nb in get_neighbors(skel, p):
            if edge_key(p, nb) in visited_edges:
                continue

            start = p
            prev = p
            cur = nb
            path = [start]
            visited_edges.add(edge_key(start, cur))
            safety = 0

            while True:
                safety += 1
                if safety > 20000:
                    break

                path.append(cur)

                nbs = [q for q in get_neighbors(skel, cur) if q != prev]
                nbs_unvisited = [q for q in nbs if edge_key(cur, q) not in visited_edges]
                if not nbs_unvisited:
                    break

                nxt = choose_best_continuation(prev, cur, nbs_unvisited)
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
# 3b. Corner-based stroke splitting
# ============================================================

def polyline_arc_length(points):
    if points is None or len(points) < 2:
        return 0.0
    d = np.diff(points.astype(np.float64), axis=0)
    return float(np.sum(np.sqrt(np.sum(d * d, axis=1))))


def angle_between_oriented_vectors(v1, v2):
    """Turning angle between two oriented tangent vectors in degrees, range [0, 180]."""
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-8 or n2 < 1e-8:
        return 0.0
    u1 = v1 / n1
    u2 = v2 / n2
    c = float(np.dot(u1, u2))
    c = np.clip(c, -1.0, 1.0)
    return math.degrees(math.acos(c))


def stroke_corner_angles(points, window=8):
    """
    Estimate local turning angle along a traced stroke.

    For point i:
      before tangent = p[i] - p[i-window]
      after tangent  = p[i+window] - p[i]
    A sharp polyline corner gives a large angle; a near-straight continuation gives ~0.
    """
    pts = points.astype(np.float64)
    n = len(pts)
    angles = np.zeros(n, dtype=np.float64)
    if n < 2 * window + 3:
        return angles

    for i in range(window, n - window):
        v1 = pts[i] - pts[i - window]
        v2 = pts[i + window] - pts[i]
        angles[i] = angle_between_oriented_vectors(v1, v2)
    return angles


def find_corner_cut_indices(
    points,
    angle_thresh=15.0,
    window=8,
    min_segment_arc=25.0,
    nms_radius=6,
):
    """
    Return sorted cut indices where the stroke turns sharply.

    The detector uses non-maximum suppression so a single corner creates one cut,
    not a cluster of nearby cuts. Cuts that would create too-short fragments are ignored.
    """
    pts = points.astype(np.float64)
    n = len(pts)
    if n < 2 * window + 3:
        return []

    angles = stroke_corner_angles(pts, window=window)
    candidate_ids = np.where(angles >= angle_thresh)[0].tolist()
    if not candidate_ids:
        return []

    # Keep local maxima only.
    local_max = []
    for i in candidate_ids:
        lo = max(0, i - nms_radius)
        hi = min(n, i + nms_radius + 1)
        if angles[i] >= np.max(angles[lo:hi]) - 1e-9:
            local_max.append(i)

    # Sort strong corners first, then accept if sufficiently separated by arc length.
    local_max = sorted(set(local_max), key=lambda idx: float(angles[idx]), reverse=True)
    cuts = []

    def arc_between(a, b):
        if b <= a:
            return 0.0
        return polyline_arc_length(pts[a:b + 1])

    for idx in local_max:
        proposed = sorted(cuts + [idx])
        bounds = [0] + proposed + [n - 1]
        ok = True
        for a, b in zip(bounds[:-1], bounds[1:]):
            if arc_between(a, b) < min_segment_arc:
                ok = False
                break
        if ok:
            cuts.append(idx)

    return sorted(cuts)


def split_one_stroke_at_corners(
    points,
    angle_thresh=15.0,
    window=8,
    min_segment_arc=25.0,
    nms_radius=6,
):
    """Split one traced stroke into multiple strokes at sharp turns."""
    pts = points.astype(np.float32)
    cuts = find_corner_cut_indices(
        pts,
        angle_thresh=angle_thresh,
        window=window,
        min_segment_arc=min_segment_arc,
        nms_radius=nms_radius,
    )
    if not cuts:
        return [pts]

    out = []
    start = 0
    for cut in cuts:
        seg = pts[start:cut + 1]
        if len(seg) >= 2 and polyline_arc_length(seg) >= min_segment_arc:
            out.append(seg.astype(np.float32))
        start = cut
    seg = pts[start:]
    if len(seg) >= 2 and polyline_arc_length(seg) >= min_segment_arc:
        out.append(seg.astype(np.float32))

    return out if out else [pts]


def split_strokes_at_corners(
    strokes,
    angle_thresh=15.0,
    window=8,
    min_segment_arc=25.0,
    nms_radius=6,
):
    """
    Split all traced strokes at sharp corners.

    This is important for hand-drawn polyhedral sketches: trace_strokes() follows
    graph topology, so if a polyline changes direction without a topological branch,
    it can remain one long stroke. This post-pass cuts it when the local turning
    angle exceeds the threshold, e.g. 15 degrees.
    """
    result = []
    for s in strokes:
        parts = split_one_stroke_at_corners(
            s,
            angle_thresh=angle_thresh,
            window=window,
            min_segment_arc=min_segment_arc,
            nms_radius=nms_radius,
        )
        result.extend(parts)
    return result

# ============================================================
# 4. Stroke merge and geometry
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
        v = stroke[k] - stroke[0]
    else:
        v = stroke[-1] - stroke[-1 - k]
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

    # Put connected endpoints at a[-1] and b[0].
    if end1 == 0:
        a = a[::-1]
    if end2 == 1:
        b = b[::-1]

    if np.linalg.norm(a[-1] - b[0]) < 1e-6:
        merged = np.vstack([a, b[1:]])
    else:
        merged = np.vstack([a, b])
    return merged.astype(np.float32)

def get_branch_points(skel):
    """Return true branch pixels detected by crossing number.

    These are protected endpoints: merge is NOT allowed at or near them,
    because paths meeting at a true branch are different topology arcs, not a
    fake break.
    """
    ys, xs = np.where(skel > 0)
    pts = []
    for x, y in zip(xs, ys):
        if skeleton_node_type(skel, (x, y)) == "branch":
            pts.append((float(x), float(y)))
    if len(pts) == 0:
        return np.empty((0, 2), dtype=np.float32)
    return np.asarray(pts, dtype=np.float32)


def is_near_branch_endpoint(point, branch_points, radius=3.0):
    """True when point is at/near a protected true branch pixel."""
    if branch_points is None or len(branch_points) == 0:
        return False
    p = np.asarray(point, dtype=np.float32)
    d = branch_points - p[None, :]
    d2 = np.sum(d * d, axis=1)
    return bool(np.any(d2 <= radius * radius))



def can_merge_strokes(
    s1,
    end1,
    s2,
    end2,
    max_gap=10.0,
    max_angle=40.0,
    branch_points=None,
    branch_protect_radius=3.0,
):
    """Allow merging only fake endpoint-to-endpoint breaks.

    Critical rule:
      - If either candidate endpoint is at/near a true branch point, DO NOT merge.

    This prevents the three valid branch-to-branch paths of a cylinder-like
    drawing from being incorrectly glued into one large loop.
    """
    p1 = s1[0] if end1 == 0 else s1[-1]
    p2 = s2[0] if end2 == 0 else s2[-1]

    # Protect true topological branches. A merge at such a point would destroy
    # the graph structure by combining distinct paths that merely meet there.
    if is_near_branch_endpoint(p1, branch_points, branch_protect_radius):
        return False, float("inf"), 180.0
    if is_near_branch_endpoint(p2, branch_points, branch_protect_radius):
        return False, float("inf"), 180.0

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
    branch_points=None,
    branch_protect_radius=3.0,
):
    """Greedy merge for fake breaks only.

    It merges close, tangent-continuous endpoint pairs, but never merges at true
    branch endpoints. This lets broken open strokes reconnect while preserving
    real topology such as multiple arcs sharing the same two branch nodes.
    """
    strokes = [s.copy() for s in strokes if len(s) >= min_length_after_merge]

    for _ in range(max_iters):
        best = None
        n = len(strokes)

        for i in range(n):
            for j in range(i + 1, n):
                for end_i in [0, 1]:
                    for end_j in [0, 1]:
                        ok, gap, angle = can_merge_strokes(
                            strokes[i], end_i, strokes[j], end_j,
                            max_gap=max_gap,
                            max_angle=max_angle,
                            branch_points=branch_points,
                            branch_protect_radius=branch_protect_radius,
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

        i, j = best["i"], best["j"]
        merged = merge_two_strokes(strokes[i], best["end_i"], strokes[j], best["end_j"])
        strokes = [s for k, s in enumerate(strokes) if k not in (i, j)] + [merged]

    return strokes

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
        infos.append({
            "index": i,
            "points": pts,
            "arc": arc,
            "chord": chord,
            "straightness": straight,
            "line": line,
            "direction": direction,
            "center": center,
        })
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
    v /= nv
    cosang = abs(float(np.dot(direction, v)))
    cosang = np.clip(cosang, -1.0, 1.0)
    return math.degrees(math.acos(cosang))


# ============================================================
# 5. Direction model
# ============================================================

def estimate_vp_from_strokes(stroke_infos, image_shape, min_length=60, min_straightness=0.9, dist_thresh=8.0, angle_thresh=12.0):
    h, w = image_shape[:2]
    line_strokes = [s for s in stroke_infos if s["arc"] >= min_length and s["straightness"] >= min_straightness]

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
            best = {"mode": "vp", "vp": vp, "score": float(score), "inliers": inliers}
    return best, line_strokes


def angle_between_dirs(d1, d2):
    d1 = d1 / (np.linalg.norm(d1) + 1e-12)
    d2 = d2 / (np.linalg.norm(d2) + 1e-12)
    c = abs(float(np.dot(d1, d2)))
    c = np.clip(c, -1.0, 1.0)
    return math.degrees(math.acos(c))


def canonical_axis_direction(d):
    """
    Canonical representation of an UNORIENTED 2D direction axis.

    The clustering still uses angle_between_dirs(), which treats d and -d as
    the same direction.  This helper is only for debug printing so that every
    stroke's direction is displayed consistently.
    """
    d = np.asarray(d, dtype=np.float64)
    n = np.linalg.norm(d)
    if n < 1e-12:
        return np.array([0.0, 0.0], dtype=np.float64)
    d = d / n

    # Canonical sign for display only: prefer positive x; if vertical, positive y.
    if d[0] < 0 or (abs(d[0]) < 1e-12 and d[1] < 0):
        d = -d
    return d


def axis_angle_0_180(d):
    """Unoriented axis angle in degrees, range [0, 180)."""
    d = canonical_axis_direction(d)
    ang = math.degrees(math.atan2(d[1], d[0]))
    if ang < 0:
        ang += 180.0
    if ang >= 180.0:
        ang -= 180.0
    return float(ang)


def endpoint_axis_direction_from_points(points):
    """
    Endpoint chord direction as an unoriented debug axis.
    This is only for debugging; clustering still uses s["direction"] by default.
    """
    if len(points) < 2:
        return np.array([0.0, 0.0], dtype=np.float64)
    v = points[-1].astype(np.float64) - points[0].astype(np.float64)
    return canonical_axis_direction(v)


def stroke_direction_debug_values(s):
    """Return all direction-related debug values for one stroke-info dict."""
    pca_axis = canonical_axis_direction(s["direction"])
    endpoint_axis = endpoint_axis_direction_from_points(s["points"])
    return {
        "pca_axis": pca_axis,
        "pca_angle": axis_angle_0_180(pca_axis),
        "endpoint_axis": endpoint_axis,
        "endpoint_angle": axis_angle_0_180(endpoint_axis),
    }


def stroke_center(s):
    return s["points"].mean(axis=0)


def stroke_endpoints(s):
    return s["points"][0], s["points"][-1]


def endpoint_dist(a, b):
    return float(np.linalg.norm(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)))


def share_same_endpoint_pair(s1, s2, tol=12.0):
    """
    True if two strokes connect approximately the same two graph nodes.

    This is a strong cue that the two strokes are opposite arcs of the same
    loop/cap, not two separate side rails.
    """
    a0, a1 = stroke_endpoints(s1)
    b0, b1 = stroke_endpoints(s2)

    case1 = endpoint_dist(a0, b0) <= tol and endpoint_dist(a1, b1) <= tol
    case2 = endpoint_dist(a0, b1) <= tol and endpoint_dist(a1, b0) <= tol
    return case1 or case2


def mean_direction(strokes):
    ref = strokes[0]["direction"].copy()
    dirs = []

    for s in strokes:
        d = s["direction"].copy()
        if np.dot(d, ref) < 0:
            d = -d
        dirs.append(d)

    v = np.mean(dirs, axis=0)
    v = v / (np.linalg.norm(v) + 1e-12)

    # Display convention only: prefer downward direction in image coordinates.
    if v[1] < 0:
        v = -v

    return v


def perp_vector(direction):
    d = direction / (np.linalg.norm(direction) + 1e-12)
    return np.array([-d[1], d[0]], dtype=np.float64)


def cluster_perp_spread(strokes, direction):
    """
    Spread of stroke centers along the axis perpendicular to extrusion direction.

    A side-family cluster should occupy separated rails / side tracks.
    A cap-only cluster often stays close to one loop/arc region and has smaller
    useful perpendicular spread.
    """
    if not strokes:
        return 0.0

    n = perp_vector(direction)
    coords = [float(np.dot(stroke_center(s), n)) for s in strokes]
    return float(max(coords) - min(coords))


def cluster_same_loop_pairs(strokes, endpoint_tol=12.0):
    """Count pairwise same-loop evidence inside a direction cluster."""
    count = 0
    for i in range(len(strokes)):
        for j in range(i + 1, len(strokes)):
            if share_same_endpoint_pair(strokes[i], strokes[j], tol=endpoint_tol):
                count += 1
    return count


def share_any_endpoint(s1, s2, tol=12.0):
    """
    True if two strokes are connected by at least one endpoint.

    This is stricter than share_same_endpoint_pair():
      - share_same_endpoint_pair means two strokes connect the same two nodes;
      - share_any_endpoint means they touch at one endpoint, so they are adjacent
        in the stroke graph.

    For a side-direction cluster, adjacent/connected strokes usually indicate
    a cap/contour chain rather than independent parallel side rails. Therefore
    clusters with any connected pair are rejected by score_side_cluster().
    """
    a0, a1 = stroke_endpoints(s1)
    b0, b1 = stroke_endpoints(s2)
    return (
        endpoint_dist(a0, b0) <= tol
        or endpoint_dist(a0, b1) <= tol
        or endpoint_dist(a1, b0) <= tol
        or endpoint_dist(a1, b1) <= tol
    )


def cluster_connected_pairs(strokes, endpoint_tol=12.0):
    """Return all connected stroke pairs inside a cluster."""
    pairs = []
    for i in range(len(strokes)):
        for j in range(i + 1, len(strokes)):
            if share_any_endpoint(strokes[i], strokes[j], tol=endpoint_tol):
                pairs.append((int(strokes[i]["index"]), int(strokes[j]["index"])))
    return pairs


def pairwise_direction_consistent(strokes, angle_thresh=25.0):
    """True if every pair in strokes is within the unoriented angle threshold."""
    for i in range(len(strokes)):
        for j in range(i + 1, len(strokes)):
            if angle_between_dirs(strokes[i]["direction"], strokes[j]["direction"]) > angle_thresh:
                return False
    return True


def cluster_has_connected_pair(strokes, endpoint_tol=12.0):
    """True if any two strokes in the group share at least one endpoint."""
    return len(cluster_connected_pairs(strokes, endpoint_tol=endpoint_tol)) > 0


def build_direction_clusters(
    line_strokes,
    angle_thresh=25.0,
    endpoint_tol=12.0,
    enumerate_subsets=True,
    max_subset_size=12,
):
    """
    Build direction clusters by enumerating ALL pairwise-angle-consistent subsets.

    This version follows the pair-list logic directly:
      - Build an undirected graph where every line stroke is a node.
      - Add an edge when the pairwise UNORIENTED PCA angle <= angle_thresh.
      - Enumerate every clique/subset whose every pair satisfies the threshold.

    Therefore, if the pair list allows both:
      [2, 9, 13, 18]
      [9, 13, 18]
    both clusters are generated at the start.  The larger connected cluster is
    then hard-rejected later by score_side_cluster() if it contains connected
    stroke pairs; the valid subset remains available for scoring.

    Notes:
      - Clustering remains UNORIENTED: d and -d are treated as the same axis.
      - Single-stroke clusters are included as fallback/debug candidates.
      - endpoint_tol is not used here; connected-stroke rejection happens only
        in score_side_cluster(), so rejected parent clusters stay visible in
        the debug report.
    """
    import itertools

    n = len(line_strokes)
    if n == 0:
        return []

    # If there are too many candidates, exhaustive enumeration can explode.
    # For normal sketch cases n is small; if it is large, keep a safe fallback
    # using size-1 and size-2 valid clusters plus original seed/star clusters.
    exhaustive = n <= max_subset_size

    clusters = {}

    def add_cluster(group, source="angle_clique"):
        if not group:
            return
        key = tuple(sorted(int(x["index"]) for x in group))
        if key not in clusters:
            clusters[key] = {
                "strokes": list(group),
                "source": source,
            }

    def is_angle_clique(group):
        if len(group) <= 1:
            return True
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                if angle_between_dirs(group[i]["direction"], group[j]["direction"]) > angle_thresh:
                    return False
        return True

    if exhaustive:
        # Enumerate larger clusters first so debug ids roughly reflect the
        # intuitive cluster hierarchy.  Sorting by score still happens later.
        for r in range(n, 0, -1):
            for comb in itertools.combinations(line_strokes, r):
                comb = list(comb)
                if is_angle_clique(comb):
                    add_cluster(comb, source="angle_clique")
    else:
        # Fallback for unusually large candidate sets.
        for s in line_strokes:
            add_cluster([s], source="singleton_fallback")

        for a, b in itertools.combinations(line_strokes, 2):
            if angle_between_dirs(a["direction"], b["direction"]) <= angle_thresh:
                add_cluster([a, b], source="pair_fallback")

        # Also keep old star clusters visible for debugging.
        for s in line_strokes:
            group = []
            for t in line_strokes:
                if angle_between_dirs(s["direction"], t["direction"]) <= angle_thresh:
                    group.append(t)
            if group:
                add_cluster(group, source="star_fallback")

    # Stable order: larger groups first, then by stroke indices.
    out = list(clusters.values())
    out.sort(key=lambda e: (-len(e["strokes"]), tuple(sorted(int(st["index"]) for st in e["strokes"]))))
    return out


def cluster_length_similarity(strokes):
    """
    Measure how similar stroke lengths are inside a cluster.

    Returns a dict with:
      - length_mean / length_std
      - length_cv = std / mean
      - length_min / length_max / length_max_min_ratio
      - length_similarity_score in [0, 1], higher means more similar lengths.

    Single-stroke clusters get score 0.0 because there is no within-cluster
    length consistency evidence.
    """
    if not strokes:
        return {
            "length_mean": 0.0,
            "length_std": 0.0,
            "length_cv": 0.0,
            "length_min": 0.0,
            "length_max": 0.0,
            "length_max_min_ratio": 0.0,
            "length_similarity_score": 0.0,
        }

    lengths = np.array([float(s["arc"]) for s in strokes], dtype=np.float64)
    mean_len = float(np.mean(lengths))
    std_len = float(np.std(lengths))
    min_len = float(np.min(lengths))
    max_len = float(np.max(lengths))

    if mean_len <= 1e-8:
        cv = 0.0
    else:
        cv = std_len / mean_len

    if min_len <= 1e-8:
        max_min_ratio = float("inf")
    else:
        max_min_ratio = max_len / min_len

    # Similarity score: 1.0 when lengths are equal, decreases smoothly as CV grows.
    # Do not reward singleton clusters; there is no length-balance evidence.
    if len(strokes) <= 1:
        similarity = 0.0
    else:
        similarity = 1.0 / (1.0 + cv)

    return {
        "length_mean": mean_len,
        "length_std": std_len,
        "length_cv": float(cv),
        "length_min": min_len,
        "length_max": max_len,
        "length_max_min_ratio": float(max_min_ratio),
        "length_similarity_score": float(similarity),
    }


def score_side_cluster(
    cluster,
    direction,
    min_perp_spread=10.0,
    same_loop_endpoint_tol=12.0,
    count_weight=120.0,
    length_weight=0.7,
    straightness_weight=250.0,
    spread_weight=1.5,
    same_loop_penalty_weight=220.0,
    low_spread_penalty=250.0,
    length_similarity_weight=5000.0,
):
    """
    Side-likeness score for a direction cluster.

    Positive cues:
      - more strokes in the direction family;
      - longer total length;
      - higher mean straightness / lower curvature;
      - enough perpendicular spread between rails/tracks;
      - similar stroke lengths inside the cluster.

    Negative cues:
      - many pairs sharing the same two endpoints, likely a cap loop;
      - any pair of strokes connected by one endpoint: hard reject;
      - very low perpendicular spread for multi-stroke clusters.
    """
    if not cluster:
        return -1e18, {}

    total_len = float(sum(s["arc"] for s in cluster))
    mean_straight = float(np.mean([s["straightness"] for s in cluster]))
    perp_spread = cluster_perp_spread(cluster, direction)
    same_loop_pairs = cluster_same_loop_pairs(cluster, endpoint_tol=same_loop_endpoint_tol)
    connected_pairs = cluster_connected_pairs(cluster, endpoint_tol=same_loop_endpoint_tol)
    connected_pair_count = len(connected_pairs)
    length_stats = cluster_length_similarity(cluster)
    length_similarity_bonus = float(length_similarity_weight * length_stats["length_similarity_score"])

    spread_penalty = 0.0
    if len(cluster) >= 2 and perp_spread < min_perp_spread:
        spread_penalty = low_spread_penalty

    invalid_connected_cluster = connected_pair_count > 0

    if invalid_connected_cluster:
        # Hard reject: strokes in one direction cluster should be separate side
        # rails/fragments, not directly connected to each other. Connected
        # strokes are usually parts of the same cap/contour chain.
        score = -1e18
    else:
        score = (
            count_weight * len(cluster)
            + length_weight * total_len
            + straightness_weight * mean_straight
            + spread_weight * perp_spread
            + length_similarity_bonus
            - same_loop_penalty_weight * same_loop_pairs
            - spread_penalty
        )

    details = {
        "n": len(cluster),
        "total_len": total_len,
        "mean_straight": mean_straight,
        "perp_spread": perp_spread,
        "same_loop_pairs": same_loop_pairs,
        "connected_pair_count": connected_pair_count,
        "connected_pairs": connected_pairs,
        "invalid_connected_cluster": invalid_connected_cluster,
        "spread_penalty": spread_penalty,
        "length_mean": length_stats["length_mean"],
        "length_std": length_stats["length_std"],
        "length_cv": length_stats["length_cv"],
        "length_min": length_stats["length_min"],
        "length_max": length_stats["length_max"],
        "length_max_min_ratio": length_stats["length_max_min_ratio"],
        "length_similarity_score": length_stats["length_similarity_score"],
        "length_similarity_bonus": length_similarity_bonus,
    }
    return float(score), details


def fallback_parallel_direction(
    line_strokes,
    angle_cluster_thresh=25.0,
    min_perp_spread=10.0,
    same_loop_endpoint_tol=12.0,
    count_weight=120.0,
    length_weight=0.7,
    straightness_weight=250.0,
    spread_weight=1.5,
    same_loop_penalty_weight=220.0,
    low_spread_penalty=250.0,
    length_similarity_weight=5000.0,
):
    """
    Cluster-based parallel side selection with full debug information.

    Steps:
      1. build direction clusters from line_strokes;
      2. compute one side-likeness score per cluster;
      3. keep every cluster's score/details for debug output;
      4. select the highest score cluster as side inliers.
    """
    if len(line_strokes) < 1:
        return None

    cluster_entries = build_direction_clusters(line_strokes, angle_thresh=angle_cluster_thresh, endpoint_tol=same_loop_endpoint_tol)
    if not cluster_entries:
        return None

    scored_clusters = []
    best_i = None
    best_score = -1e18

    for ci, cluster_entry in enumerate(cluster_entries):
        cluster = cluster_entry["strokes"]
        direction = mean_direction(cluster)
        score, details = score_side_cluster(
            cluster,
            direction,
            min_perp_spread=min_perp_spread,
            same_loop_endpoint_tol=same_loop_endpoint_tol,
            count_weight=count_weight,
            length_weight=length_weight,
            straightness_weight=straightness_weight,
            spread_weight=spread_weight,
            same_loop_penalty_weight=same_loop_penalty_weight,
            low_spread_penalty=low_spread_penalty,
            length_similarity_weight=length_similarity_weight,
        )

        entry = {
            "cluster_id": ci,
            "strokes": cluster,
            "indices": [int(s["index"]) for s in cluster],
            "direction": direction,
            "score": float(score),
            "details": details,
            "source": cluster_entry.get("source", "unknown"),
        }
        scored_clusters.append(entry)

        if score > best_score:
            best_score = float(score)
            best_i = ci

    # Sort a copy for text/debug display, but keep cluster_id stable.
    ranked_clusters = sorted(scored_clusters, key=lambda x: x["score"], reverse=True)
    best = scored_clusters[best_i]
    cluster = best["strokes"]

    return {
        "mode": "parallel",
        "direction": best["direction"],
        "score": float(best["score"]),
        "inliers": cluster,
        "cluster_indices": [s["index"] for s in cluster],
        "cluster_details": best["details"],
        "selected_cluster_id": int(best["cluster_id"]),
        "cluster_debug": ranked_clusters,
        "cluster_angle_thresh": float(angle_cluster_thresh),
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
    line_strokes = [s for s in infos if s["arc"] >= args.min_stroke_length and s["straightness"] >= args.straightness]

    if args.force_parallel:
        return fallback_parallel_direction(
            line_strokes,
            angle_cluster_thresh=args.parallel_angle_thresh,
            min_perp_spread=args.cluster_min_perp_spread,
            same_loop_endpoint_tol=args.same_loop_endpoint_tol,
            count_weight=args.cluster_count_weight,
            length_weight=args.cluster_length_weight,
            straightness_weight=args.cluster_straightness_weight,
            spread_weight=args.cluster_spread_weight,
            same_loop_penalty_weight=args.cluster_same_loop_penalty,
            low_spread_penalty=args.cluster_low_spread_penalty,
            length_similarity_weight=args.cluster_length_similarity_weight,
        ), line_strokes

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
        return {"mode": "vp", "vp": vp, "score": float(score), "inliers": inliers}, line_strokes

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
        min_perp_spread=args.cluster_min_perp_spread,
        same_loop_endpoint_tol=args.same_loop_endpoint_tol,
        count_weight=args.cluster_count_weight,
        length_weight=args.cluster_length_weight,
        straightness_weight=args.cluster_straightness_weight,
        spread_weight=args.cluster_spread_weight,
        same_loop_penalty_weight=args.cluster_same_loop_penalty,
        low_spread_penalty=args.cluster_low_spread_penalty,
        length_similarity_weight=args.cluster_length_similarity_weight,
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
# 6. Masks + candidates
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
        x0, x1 = max(0, x - 1), min(component_mask.shape[1], x + 2)
        y0, y1 = max(0, y - 1), min(component_mask.shape[0], y + 2)
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
    num, labels, stats, _ = cv2.connectedComponentsWithStats(non_side_skel, connectivity=8)
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
        candidates.append({
            "mask": comp,
            "area": area,
            "endpoints": endpoints,
            "closedness": closedness,
            "score": score,
            "center": center,
        })
    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates




# ============================================================
# 6b. Stroke-loop based cap candidates
# ============================================================

def make_stroke_mask(shape, strokes, thickness=2):
    """Rasterize a list of stroke-info dicts to a mask."""
    mask = np.zeros(shape[:2], dtype=np.uint8)
    for s in strokes:
        pts = s["points"].reshape(-1, 1, 2).astype(np.int32)
        cv2.polylines(mask, [pts], False, 255, thickness, cv2.LINE_AA)
    mask = cv2.dilate(mask, np.ones((2, 2), np.uint8), iterations=1)
    return mask


def cluster_points_by_distance(points, tol=12.0):
    """Greedy endpoint clustering."""
    centers = []
    labels = []
    counts = []
    for p in points:
        p = np.asarray(p, dtype=np.float64)
        best_i = None
        best_d = float("inf")
        for i, c in enumerate(centers):
            d = float(np.linalg.norm(p - c))
            if d < best_d:
                best_d = d
                best_i = i
        if best_i is not None and best_d <= tol:
            labels.append(best_i)
            counts[best_i] += 1
            alpha = 1.0 / counts[best_i]
            centers[best_i] = (1.0 - alpha) * centers[best_i] + alpha * p
        else:
            labels.append(len(centers))
            centers.append(p.copy())
            counts.append(1)
    return labels, centers


def stroke_endpoint_node_ids(strokes, endpoint_tol=12.0):
    """Return endpoint node ids for each stroke based on endpoint clustering."""
    pts = []
    for s in strokes:
        pts.append(s["points"][0])
        pts.append(s["points"][-1])
    labels, centers = cluster_points_by_distance(pts, tol=endpoint_tol)
    endpoint_nodes = []
    for i in range(len(strokes)):
        endpoint_nodes.append((labels[2 * i], labels[2 * i + 1]))
    return endpoint_nodes, centers


def connected_components_of_stroke_graph(num_strokes, endpoint_nodes):
    """Connected components over strokes that share endpoint nodes."""
    node_to_strokes = {}
    for si, (a, b) in enumerate(endpoint_nodes):
        node_to_strokes.setdefault(a, []).append(si)
        node_to_strokes.setdefault(b, []).append(si)
    visited = set()
    comps = []
    for start in range(num_strokes):
        if start in visited:
            continue
        stack = [start]
        visited.add(start)
        comp = []
        while stack:
            sidx = stack.pop()
            comp.append(sidx)
            a, b = endpoint_nodes[sidx]
            for n in (a, b):
                for nb_sidx in node_to_strokes.get(n, []):
                    if nb_sidx not in visited:
                        visited.add(nb_sidx)
                        stack.append(nb_sidx)
        comps.append(comp)
    return comps


def is_closed_stroke_graph(comp, endpoint_nodes):
    """
    Strict closed-loop test.

    A cap candidate must form a loop in the endpoint graph:
      - every graph node touched by this component has degree exactly 2.

    This accepts two strokes sharing the same endpoint pair, and rejects
    open arcs / dangling fragments / branchy residuals.
    """
    if not comp:
        return False
    node_degree = {}
    for sidx in comp:
        a, b = endpoint_nodes[sidx]
        if a == b:
            node_degree[a] = node_degree.get(a, 0) + 2
        else:
            node_degree[a] = node_degree.get(a, 0) + 1
            node_degree[b] = node_degree.get(b, 0) + 1
    if not node_degree:
        return False
    return all(d == 2 for d in node_degree.values())


def candidate_from_stroke_loop(shape, strokes, comp, thickness=2):
    """Create a cap-candidate record from a closed stroke-loop component."""
    loop_strokes = [strokes[i] for i in comp]
    mask = make_stroke_mask(shape, loop_strokes, thickness=thickness)
    pts = np.vstack([s["points"] for s in loop_strokes])
    center = pts.mean(axis=0).astype(np.float64)
    area = int(np.count_nonzero(mask))
    total_arc = float(sum(s["arc"] for s in loop_strokes))
    return {
        "mask": mask,
        "area": area,
        "endpoints": 0,
        "closedness": 1.0,
        "score": float(area + total_arc),
        "center": center,
        "stroke_indices": [int(s["index"]) for s in loop_strokes],
        "stroke_count": len(loop_strokes),
        "total_arc": total_arc,
    }


def is_connected_stroke_graph(comp, endpoint_nodes):
    """Return True if a subset of stroke indices is connected in endpoint graph."""
    if not comp:
        return False
    if len(comp) == 1:
        return True

    comp_set = set(comp)
    node_to_strokes = {}
    for sidx in comp:
        a, b = endpoint_nodes[sidx]
        node_to_strokes.setdefault(a, []).append(sidx)
        node_to_strokes.setdefault(b, []).append(sidx)

    start = comp[0]
    visited = {start}
    stack = [start]
    while stack:
        sidx = stack.pop()
        a, b = endpoint_nodes[sidx]
        for n in (a, b):
            for nb in node_to_strokes.get(n, []):
                if nb in comp_set and nb not in visited:
                    visited.add(nb)
                    stack.append(nb)

    return len(visited) == len(comp)


def enumerate_closed_loop_subsets(component, endpoint_nodes, max_subset_size=14):
    """
    Enumerate connected closed-loop subsets inside one connected component.

    Why this is needed:
      A visible cap loop may be embedded in a larger connected component that
      also contains dangling strokes or branch attachments.  The old test
      required the WHOLE component to have all node degrees == 2, so a valid
      loop plus one dangling stroke was rejected.

    This function instead searches for loop subsets.  A subset is accepted when:
      - it is connected;
      - every endpoint node touched by the subset has degree exactly 2.
    """
    import itertools

    component = list(component)
    if not component:
        return []

    loops = []

    # Exhaustive search is fine for these sketch graphs.  Keep a guard to avoid
    # combinatorial blow-up on noisy images.
    if len(component) <= max_subset_size:
        for r in range(len(component), 0, -1):
            for subset in itertools.combinations(component, r):
                subset = list(subset)
                if not is_connected_stroke_graph(subset, endpoint_nodes):
                    continue
                if not is_closed_stroke_graph(subset, endpoint_nodes):
                    continue
                loops.append(subset)
    else:
        # Conservative fallback: try all small/medium subsets first.
        # Most cap loops in these sketches are 2-6 strokes.
        for r in range(min(8, len(component)), 0, -1):
            for subset in itertools.combinations(component, r):
                subset = list(subset)
                if not is_connected_stroke_graph(subset, endpoint_nodes):
                    continue
                if not is_closed_stroke_graph(subset, endpoint_nodes):
                    continue
                loops.append(subset)

    # Deduplicate identical subsets.
    seen = set()
    unique = []
    for loop in loops:
        key = tuple(sorted(loop))
        if key in seen:
            continue
        seen.add(key)
        unique.append(loop)
    return unique


def extract_cap_loop_candidates_from_strokes(
    image_shape,
    infos,
    side_inliers,
    endpoint_tol=12.0,
    min_pixels=40,
    thickness=2,
    max_loop_subset_size=14,
):
    """
    Extract cap candidates from non-side strokes that form closed loops.

    Important fix:
      Do NOT require the whole connected component to be a loop.  A real cap
      loop can be connected to dangling/branch strokes.  We therefore enumerate
      closed-loop subsets inside each non-side connected component.

    Open arcs / dangling residual strokes / fragments not forming a loop are
    rejected and are not cap candidates.
    """
    side_ids = {int(s["index"]) for s in side_inliers}
    non_side_infos = [s for s in infos if int(s["index"]) not in side_ids]
    if not non_side_infos:
        return []

    endpoint_nodes, _node_centers = stroke_endpoint_node_ids(non_side_infos, endpoint_tol=endpoint_tol)
    comps = connected_components_of_stroke_graph(len(non_side_infos), endpoint_nodes)

    candidates = []
    seen = set()

    for comp in comps:
        # Old behavior: whole component as loop.
        loop_subsets = []
        if is_closed_stroke_graph(comp, endpoint_nodes) and is_connected_stroke_graph(comp, endpoint_nodes):
            loop_subsets.append(list(comp))

        # New behavior: closed loop subsets within the component.
        loop_subsets.extend(
            enumerate_closed_loop_subsets(
                comp,
                endpoint_nodes,
                max_subset_size=max_loop_subset_size,
            )
        )

        for loop in loop_subsets:
            key = tuple(sorted(int(non_side_infos[i]["index"]) for i in loop))
            if key in seen:
                continue
            seen.add(key)

            cand = candidate_from_stroke_loop(image_shape, non_side_infos, loop, thickness=thickness)
            if cand["area"] < min_pixels:
                continue
            cand["loop_detection"] = "subset" if set(loop) != set(comp) else "component"
            cand["component_local_indices"] = list(map(int, comp))
            candidates.append(cand)

    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates




# ============================================================
# 6c. Cap-validated side cluster selection
# ============================================================

def make_trial_model_from_cluster_entry(base_model, entry, cluster_debug=None, cap_validated=False):
    """Create a model dict using one cluster entry as side inliers."""
    trial = dict(base_model)
    trial["mode"] = "parallel"
    trial["direction"] = entry["direction"]
    trial["score"] = float(entry["score"])
    trial["inliers"] = entry["strokes"]
    trial["cluster_indices"] = [int(s["index"]) for s in entry["strokes"]]
    trial["cluster_details"] = entry.get("details", {})
    trial["selected_cluster_id"] = int(entry.get("cluster_id", -1))
    if cluster_debug is not None:
        trial["cluster_debug"] = cluster_debug
    trial["cap_validated"] = bool(cap_validated)
    return trial


def largest_area_cap_candidate(candidates):
    """Return the largest-area cap candidate, or None."""
    if not candidates:
        return None
    return max(candidates, key=lambda c: (int(c.get("area", 0)), float(c.get("score", 0.0))))


def cluster_entry_index_set(entry):
    idxs = entry.get("indices", None)
    if idxs is not None:
        return set(int(x) for x in idxs)
    strokes = entry.get("strokes", [])
    return set(int(s.get("index", -1)) for s in strokes)


def validate_side_clusters_by_cap_candidates(
    model,
    infos,
    image_shape,
    endpoint_tol=12.0,
    min_pixels=40,
    thickness=2,
    max_loop_subset_size=14,
):
    """
    Compute cap candidates for ranked side clusters.

    New rule:
      If a higher-ranked side cluster already found a legal cap, then any lower-ranked
      side cluster whose stroke set is a subset of that higher-ranked successful side
      cluster will be skipped: no cap computation and no per-cluster visualization.
    """
    if model is None:
        return None, []

    cluster_debug = model.get("cluster_debug", None)

    if not cluster_debug:
        candidates = extract_cap_loop_candidates_from_strokes(
            image_shape,
            infos,
            model.get("inliers", []),
            endpoint_tol=endpoint_tol,
            min_pixels=min_pixels,
            thickness=thickness,
            max_loop_subset_size=max_loop_subset_size,
        )
        best_cap = largest_area_cap_candidate(candidates)
        model["cap_validated"] = best_cap is not None
        return model, ([] if best_cap is None else [best_cap])

    selected_entry = None
    selected_candidates = []
    successful_cap_clusters = []

    for rank, entry in enumerate(cluster_debug):
        details = entry.setdefault("details", {})
        n_strokes = int(details.get("n", len(entry.get("strokes", []))))
        side_set = cluster_entry_index_set(entry)

        details["cap_validation_rank"] = int(rank)
        details["cap_validation_checked"] = False
        details["cap_validation_skipped"] = False
        details["cap_validation_skip_reason"] = ""
        details["cap_validation_selectable"] = True
        details["cap_candidate_count"] = 0
        details["cap_candidate_strokes"] = []
        details["cap_candidate_scores"] = []
        details["cap_candidate_areas"] = []
        details["best_cap_strokes"] = []
        details["best_cap_score"] = 0.0
        details["best_cap_area"] = 0
        details["best_cap_center"] = None
        details["best_cap_total_arc"] = 0.0
        details["invalid_no_cap"] = False
        details["selected_by_cap_validation"] = False
        details["skip_percluster_output"] = False
        details["subset_of_higher_rank_cap_cluster"] = False
        details["subset_parent_rank"] = None
        details["subset_parent_cluster_id"] = None

        pre_rejected = (
            details.get("invalid_connected_cluster", False)
            or float(entry.get("score", -1e18)) <= -1e17
        )
        if pre_rejected:
            details["cap_validation_skipped"] = True
            details["cap_validation_skip_reason"] = "pre_rejected_cluster"
            details["cap_validation_selectable"] = False
            continue

        if n_strokes < 2:
            details["cap_validation_skipped"] = True
            details["cap_validation_skip_reason"] = "n_lt_2"
            details["cap_validation_selectable"] = False
            continue

        subset_parent = None
        for parent in successful_cap_clusters:
            if side_set and side_set.issubset(parent["side_set"]):
                subset_parent = parent
                break
        if subset_parent is not None:
            details["cap_validation_skipped"] = True
            details["cap_validation_skip_reason"] = "subset_of_higher_rank_cap_cluster"
            details["cap_validation_selectable"] = False
            details["skip_percluster_output"] = True
            details["subset_of_higher_rank_cap_cluster"] = True
            details["subset_parent_rank"] = int(subset_parent["rank"])
            details["subset_parent_cluster_id"] = int(subset_parent["cluster_id"])
            continue

        details["cap_validation_checked"] = True

        trial_candidates = extract_cap_loop_candidates_from_strokes(
            image_shape,
            infos,
            entry.get("strokes", []),
            endpoint_tol=endpoint_tol,
            min_pixels=min_pixels,
            thickness=thickness,
            max_loop_subset_size=max_loop_subset_size,
        )

        best_cap = largest_area_cap_candidate(trial_candidates)
        details["cap_candidate_count"] = int(len(trial_candidates))

        if best_cap is None:
            details["invalid_no_cap"] = True
            continue

        center = best_cap.get("center", None)
        center_tuple = None if center is None else (float(center[0]), float(center[1]))

        details["cap_candidate_strokes"] = [best_cap.get("stroke_indices", [])]
        details["cap_candidate_scores"] = [float(best_cap.get("score", 0.0))]
        details["cap_candidate_areas"] = [int(best_cap.get("area", 0))]
        details["best_cap_strokes"] = best_cap.get("stroke_indices", [])
        details["best_cap_score"] = float(best_cap.get("score", 0.0))
        details["best_cap_area"] = int(best_cap.get("area", 0))
        details["best_cap_center"] = center_tuple
        details["best_cap_total_arc"] = float(best_cap.get("total_arc", 0.0))

        successful_cap_clusters.append({
            "rank": int(rank),
            "cluster_id": int(entry.get("cluster_id", -1)),
            "side_set": set(side_set),
        })

        if selected_entry is None:
            selected_entry = entry
            selected_candidates = [best_cap]
            details["selected_by_cap_validation"] = True

    model["successful_cap_clusters"] = [
        {
            "rank": int(x["rank"]),
            "cluster_id": int(x["cluster_id"]),
            "side_indices": sorted(list(x["side_set"])),
        }
        for x in successful_cap_clusters
    ]

    if selected_entry is None:
        model["cap_validated"] = False
        model["cap_validation_failed"] = True
        model["cap_validation_message"] = "No ranked side cluster produced a legal closed-loop cap candidate."
        return model, []

    validated_model = make_trial_model_from_cluster_entry(
        model,
        selected_entry,
        cluster_debug=cluster_debug,
        cap_validated=True,
    )
    validated_model["cap_validation_failed"] = False
    validated_model["cap_validation_message"] = (
        "Computed best cap for ranked eligible side clusters; skipped lower-ranked side clusters "
        "whose side-stroke sets are subsets of higher-ranked clusters that already found a cap; "
        "selected the first ranked selectable side cluster that produced a legal cap."
    )
    validated_model["successful_cap_clusters"] = model.get("successful_cap_clusters", [])
    return validated_model, selected_candidates

# ============================================================
# 7. Visualization/debug
# ============================================================

def ensure_dir(path):
    if path is not None:
        os.makedirs(path, exist_ok=True)


def random_color(i):
    rng = np.random.default_rng(i + 12345)
    c = rng.integers(40, 230, size=3)
    return int(c[0]), int(c[1]), int(c[2])


def colorize(out, mask, color):
    out[mask > 0] = color


def draw_strokes_image(shape, strokes, thickness=2, annotate=True):
    h, w = shape[:2]
    out = np.full((h, w, 3), 255, dtype=np.uint8)
    for i, pts in enumerate(strokes):
        color = random_color(i)
        pts_i = pts.reshape(-1, 1, 2).astype(np.int32)
        cv2.polylines(out, [pts_i], False, color, thickness, cv2.LINE_AA)
        if annotate and len(pts) > 0:
            ctr = pts.mean(axis=0)
            cv2.putText(out, str(i), (int(ctr[0]), int(ctr[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def draw_skeleton_nodes_debug(skel):
    h, w = skel.shape
    out = np.full((h, w, 3), 255, dtype=np.uint8)
    out[skel > 0] = (0, 0, 0)
    ys, xs = np.where(skel > 0)

    for x, y in zip(xs, ys):
        p = (x, y)
        deg = len(get_neighbors(skel, p))
        t = skeleton_node_type(skel, p)
        if t == "endpoint":
            cv2.circle(out, (x, y), 3, (0, 0, 255), -1)
        elif t == "branch":
            cv2.circle(out, (x, y), 3, (255, 0, 0), -1)
        elif deg != 2:
            cv2.circle(out, (x, y), 2, (0, 180, 180), -1)

    cv2.putText(out, "red=endpoint, blue=true branch, yellow=ignored false junction", (15, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
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
        cv2.putText(out, f"{s['index']} s={straight:.2f}", (int(c[0]), int(c[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
    legend = ["green: straightness >= 0.95", "yellow: >= 0.88", "blue: >= 0.75", "purple: curved"]
    y = 22
    for t in legend:
        cv2.putText(out, t, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
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
        cv2.putText(out, f"{s['index']} len={s['arc']:.0f} str={s['straightness']:.2f}", (int(c[0]), int(c[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(out, "Line stroke candidates used for direction estimation", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)
    return out


def write_stroke_direction_debug_report(path, infos, line_strokes):
    """
    Write every stroke's direction for debugging.

    pca_axis is the unoriented PCA axis used by the current clustering logic.
    endpoint_axis is the unoriented chord axis from first point to last point.
    Both are canonicalized for display only; the algorithm remains unoriented.
    """
    line_ids = {int(s["index"]) for s in line_strokes}
    with open(path, "w", encoding="utf-8") as f:
        f.write("==== Stroke Direction Debug ====\n\n")
        f.write("Angles are UNORIENTED axes in [0, 180).\n")
        f.write("pca_axis is used by direction clustering; endpoint_axis is only a chord-direction debug reference.\n\n")
        f.write("All strokes:\n")
        for s in infos:
            dbg = stroke_direction_debug_values(s)
            c = s["center"]
            p0, p1 = s["points"][0], s["points"][-1]
            in_line = "YES" if int(s["index"]) in line_ids else "no"
            f.write(
                f"  stroke {int(s['index']):03d}: "
                f"line_candidate={in_line}, "
                f"arc={s['arc']:.1f}, chord={s['chord']:.1f}, straightness={s['straightness']:.3f}, "
                f"center=({c[0]:.1f},{c[1]:.1f}), "
                f"p0=({p0[0]:.1f},{p0[1]:.1f}), p1=({p1[0]:.1f},{p1[1]:.1f}), "
                f"pca_axis=({dbg['pca_axis'][0]:.6f},{dbg['pca_axis'][1]:.6f}), "
                f"pca_angle={dbg['pca_angle']:.2f}, "
                f"endpoint_axis=({dbg['endpoint_axis'][0]:.6f},{dbg['endpoint_axis'][1]:.6f}), "
                f"endpoint_angle={dbg['endpoint_angle']:.2f}\n"
            )

        f.write("\nLine stroke candidates only:\n")
        for s in line_strokes:
            dbg = stroke_direction_debug_values(s)
            f.write(
                f"  stroke {int(s['index']):03d}: "
                f"pca_axis=({dbg['pca_axis'][0]:.6f},{dbg['pca_axis'][1]:.6f}), "
                f"pca_angle={dbg['pca_angle']:.2f}, "
                f"endpoint_axis=({dbg['endpoint_axis'][0]:.6f},{dbg['endpoint_axis'][1]:.6f}), "
                f"endpoint_angle={dbg['endpoint_angle']:.2f}\n"
            )

        f.write("\nPairwise unoriented PCA angles among line stroke candidates:\n")
        for i in range(len(line_strokes)):
            for j in range(i + 1, len(line_strokes)):
                a = line_strokes[i]
                b = line_strokes[j]
                ang = angle_between_dirs(a["direction"], b["direction"])
                f.write(f"  ({int(a['index'])},{int(b['index'])}): angle={ang:.2f}\n")


def draw_stroke_directions_image(shape, infos, line_strokes, arrow_len=70):
    """Draw every stroke's PCA axis as an unoriented debug arrow/axis."""
    h, w = shape[:2]
    out = np.full((h, w, 3), 255, dtype=np.uint8)
    line_ids = {int(s["index"]) for s in line_strokes}

    for s in infos:
        pts = s["points"].reshape(-1, 1, 2).astype(np.int32)
        is_line = int(s["index"]) in line_ids
        stroke_color = (0, 120, 0) if is_line else (170, 170, 170)
        axis_color = (0, 0, 255) if is_line else (120, 120, 120)
        cv2.polylines(out, [pts], False, stroke_color, 2, cv2.LINE_AA)

        dbg = stroke_direction_debug_values(s)
        c = s["center"].astype(np.float64)
        d = dbg["pca_axis"]
        p1 = c - d * (arrow_len * 0.5)
        p2 = c + d * (arrow_len * 0.5)
        cv2.line(out, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), axis_color, 2, cv2.LINE_AA)
        cv2.circle(out, (int(c[0]), int(c[1])), 3, axis_color, -1)

        label = f"{int(s['index'])} a={dbg['pca_angle']:.1f}"
        if is_line:
            label += " L"
        cv2.putText(out, label, (int(c[0]) + 5, int(c[1]) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.43, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.putText(out, "Stroke PCA directions: green/red = line candidates, gray = filtered out",
                (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(out, "angle is unoriented axis angle in [0,180)",
                (15, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def draw_selected_side_strokes_image(shape, model, thickness=4):
    h, w = shape[:2]
    out = np.full((h, w, 3), 255, dtype=np.uint8)
    for i, s in enumerate(model["inliers"]):
        pts = s["points"].reshape(-1, 1, 2).astype(np.int32)
        cv2.polylines(out, [pts], False, (0, 0, 255), thickness, cv2.LINE_AA)
        c = s["center"]
        cv2.putText(out, f"side {i}: stroke {s['index']}", (int(c[0]), int(c[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

    if model["mode"] == "vp":
        vp = model["vp"]
        cv2.putText(out, f"mode=VP, VP=({vp[0]:.1f}, {vp[1]:.1f})", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)
        if 0 <= vp[0] < w and 0 <= vp[1] < h:
            cv2.circle(out, (int(vp[0]), int(vp[1])), 7, (0, 0, 255), -1)
    else:
        d = model["direction"]
        cv2.putText(out, f"mode=parallel, dir=({d[0]:.2f}, {d[1]:.2f})", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)
        center = np.array([w * 0.5, h * 0.5])
        p1 = center - d * 100
        p2 = center + d * 100
        cv2.arrowedLine(out, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0, 0, 255), 3, tipLength=0.15)
    return out


def draw_non_side_skeleton_image(skel, side_mask, non_side):
    h, w = skel.shape
    out = np.full((h, w, 3), 255, dtype=np.uint8)
    out[skel > 0] = (180, 180, 180)
    out[side_mask > 0] = (0, 0, 255)
    out[non_side > 0] = (0, 0, 0)
    cv2.putText(out, "gray=original skeleton, red=removed side, black=remaining non-side", (15, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
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
            stroke_txt = c.get("stroke_indices", None)
            if stroke_txt is None:
                label = f"#{i} area={c['area']} end={c['endpoints']} closed={c['closedness']:.2f}"
            else:
                label = f"#{i} strokes={stroke_txt} closed={c['closedness']:.2f}"
            cv2.putText(out, label, (int(ctr[0]), int(ctr[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(out, "Cap candidates after side-stroke removal", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)
    return out


def draw_model_arrow(out, model, candidates):
    h, w = out.shape[:2]
    if model["mode"] == "vp":
        vp = model["vp"]
        if 0 <= vp[0] < w and 0 <= vp[1] < h:
            cv2.circle(out, (int(vp[0]), int(vp[1])), 8, (0, 0, 255), -1)
        for c in candidates[:2]:
            ctr = c["center"]
            if ctr is None:
                continue
            d = vp - ctr
            n = np.linalg.norm(d)
            if n < 1e-8:
                continue
            d = d / n
            p2 = ctr + d * min(160, n)
            cv2.arrowedLine(out, (int(ctr[0]), int(ctr[1])), (int(p2[0]), int(p2[1])), (0, 0, 255), 2, tipLength=0.15)
    else:
        d = model["direction"]
        for c in candidates[:2]:
            ctr = c["center"]
            if ctr is None:
                continue
            p2 = ctr + d * 160
            cv2.arrowedLine(out, (int(ctr[0]), int(ctr[1])), (int(p2[0]), int(p2[1])), (0, 0, 255), 2, tipLength=0.15)


def draw_result(img, skel, model, side_mask, candidates, output):
    out = img.copy()
    out[skel > 0] = (0, 0, 0)
    colorize(out, side_mask, (0, 0, 255))
    colors = [(0, 180, 0), (255, 80, 0), (180, 0, 180), (0, 180, 180)]
    for i, c in enumerate(candidates[:4]):
        thick = cv2.dilate(c["mask"], np.ones((3, 3), np.uint8), iterations=1)
        colorize(out, thick, colors[i])
        ctr = c["center"]
        if ctr is not None:
            label = f"cap_candidate_{i + 1}" + (" / base?" if i == 0 else "")
            cv2.putText(out, label, (int(ctr[0]) + 8, int(ctr[1]) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colors[i], 2, cv2.LINE_AA)
    draw_model_arrow(out, model, candidates)
    h, _ = out.shape[:2]
    if model["mode"] == "vp":
        vp = model["vp"]
        info = f"mode=VP, VP=({vp[0]:.1f},{vp[1]:.1f}), side_strokes={len(model['inliers'])}, score={model['score']:.1f}"
    else:
        d = model["direction"]
        info = f"mode=parallel, dir=({d[0]:.2f},{d[1]:.2f}), side_strokes={len(model['inliers'])}, score={model['score']:.1f}"
    cv2.putText(out, info, (20, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imwrite(output, out)
    return out



def format_cluster_debug_entry(entry, selected_cluster_id=None):
    details = entry.get("details", {})
    direction = entry.get("direction", np.array([0.0, 0.0]))
    selected = " SELECTED" if entry.get("cluster_id") == selected_cluster_id else ""
    return (
        f"cluster {entry.get('cluster_id', -1):03d}{selected}: "
        f"score={entry.get('score', 0.0):.3f}, "
        f"source={entry.get('source', 'unknown')}, "
        f"strokes={entry.get('indices', [])}, "
        f"dir=({direction[0]:.4f},{direction[1]:.4f}), "
        f"n={details.get('n', 0)}, "
        f"total_len={details.get('total_len', 0.0):.1f}, "
        f"mean_straight={details.get('mean_straight', 0.0):.3f}, "
        f"perp_spread={details.get('perp_spread', 0.0):.2f}, "
        f"same_loop_pairs={details.get('same_loop_pairs', 0)}, "
        f"connected_pairs={details.get('connected_pair_count', 0)}, "
        f"invalid_connected={details.get('invalid_connected_cluster', False)}, "
        f"len_cv={details.get('length_cv', 0.0):.3f}, "
        f"len_sim={details.get('length_similarity_score', 0.0):.3f}, "
        f"len_bonus={details.get('length_similarity_bonus', 0.0):.1f}, "
        f"cap_count={details.get('cap_candidate_count', 0)}, "
        f"best_cap_area={details.get('best_cap_area', 0)}, "
        f"best_cap_strokes={details.get('best_cap_strokes', [])}, "
        f"invalid_no_cap={details.get('invalid_no_cap', False)}, "
        f"cap_selected={details.get('selected_by_cap_validation', False)}, "
        f"spread_penalty={details.get('spread_penalty', 0.0):.1f}"
    )


def write_cluster_debug_report(path, model):
    """Write full direction-cluster scores to a text file."""
    cluster_debug = model.get("cluster_debug", []) if model is not None else []
    selected_cluster_id = model.get("selected_cluster_id", None) if model is not None else None

    with open(path, "w", encoding="utf-8") as f:
        f.write("==== Direction Cluster Debug ====" + "\n\n")
        if model is None:
            f.write("No model.\n")
            return
        f.write(f"mode: {model.get('mode')}\n")
        f.write(f"selected_cluster_id: {selected_cluster_id}\n")
        f.write(f"selected_indices: {model.get('cluster_indices', [])}\n")
        f.write(f"selected_score: {model.get('score', 0.0):.3f}\n")
        if "direction" in model:
            d = model["direction"]
            f.write(f"selected_direction: ({d[0]:.6f}, {d[1]:.6f})\n")
        f.write(f"cluster_angle_thresh: {model.get('cluster_angle_thresh', None)}\n")
        f.write("\nRanked clusters:\n")
        for entry in cluster_debug:
            f.write("  " + format_cluster_debug_entry(entry, selected_cluster_id) + "\n")
            details = entry.get("details", {})
            f.write(
                f"    length_stats: mean={details.get('length_mean', 0.0):.2f}, "
                f"std={details.get('length_std', 0.0):.2f}, "
                f"cv={details.get('length_cv', 0.0):.3f}, "
                f"min={details.get('length_min', 0.0):.1f}, "
                f"max={details.get('length_max', 0.0):.1f}, "
                f"max/min={details.get('length_max_min_ratio', 0.0):.3f}, "
                f"similarity={details.get('length_similarity_score', 0.0):.3f}, "
                f"bonus={details.get('length_similarity_bonus', 0.0):.1f}\n"
            )
            if details.get("connected_pair_count", 0) > 0:
                f.write(f"    connected_pairs={details.get('connected_pairs', [])} -> REJECTED_CLUSTER\n")
            if details.get("cap_validation_checked", False):
                f.write(
                    f"    cap_validation: total_candidates={details.get('cap_candidate_count', 0)}, "
                    f"best_area={details.get('best_cap_area', 0)}, "
                    f"best_strokes={details.get('best_cap_strokes', [])}, "
                    f"best_score={details.get('best_cap_score', 0.0):.1f}, "
                    f"best_center={details.get('best_cap_center', None)}, "
                    f"best_total_arc={details.get('best_cap_total_arc', 0.0):.1f}, "
                    f"invalid_no_cap={details.get('invalid_no_cap', False)}, "
                    f"selectable={details.get('cap_validation_selectable', True)}, "
                    f"selected={details.get('selected_by_cap_validation', False)}\n"
                )
            elif details.get("cap_validation_skipped", False):
                f.write(f"    cap_validation: skipped reason={details.get('cap_validation_skip_reason', '')}\n")
            for s in entry.get("strokes", []):
                c = s["center"]
                f.write(
                    f"    stroke {int(s['index']):03d}: "
                    f"arc={s['arc']:.1f}, chord={s['chord']:.1f}, "
                    f"straightness={s['straightness']:.3f}, "
                    f"center=({c[0]:.1f},{c[1]:.1f}), "
                    f"dir=({s['direction'][0]:.4f},{s['direction'][1]:.4f}), "
                    f"axis_dir=({stroke_direction_debug_values(s)['pca_axis'][0]:.4f},{stroke_direction_debug_values(s)['pca_axis'][1]:.4f}), "
                    f"axis_angle={stroke_direction_debug_values(s)['pca_angle']:.2f}\n"
                )


def draw_cluster_overview_image(shape, model, max_clusters=12):
    """Draw a text-only overview of all direction clusters and their scores."""
    h, w = shape[:2]
    out = np.full((h, w, 3), 255, dtype=np.uint8)
    cluster_debug = model.get("cluster_debug", []) if model is not None else []
    selected_cluster_id = model.get("selected_cluster_id", None) if model is not None else None

    title = "Direction cluster scores, ranked high to low"
    cv2.putText(out, title, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)

    y = 58
    line_h = 24
    for rank, entry in enumerate(cluster_debug[:max_clusters]):
        details = entry.get("details", {})
        selected = entry.get("cluster_id") == selected_cluster_id
        color = (0, 0, 255) if selected else (0, 0, 0)
        prefix = "*" if selected else " "
        text = (
            f"{prefix}rank {rank:02d} C{entry.get('cluster_id', -1):02d} "
            f"score={entry.get('score', 0.0):.1f} "
            f"strokes={entry.get('indices', [])} "
            f"n={details.get('n', 0)} "
            f"len={details.get('total_len', 0.0):.0f} "
            f"str={details.get('mean_straight', 0.0):.2f} "
            f"spread={details.get('perp_spread', 0.0):.1f} "
            f"loop={details.get('same_loop_pairs', 0)} "
            f"conn={details.get('connected_pair_count', 0)} "
            f"cap={details.get('cap_candidate_count', 0)} "
            f"noCap={int(bool(details.get('invalid_no_cap', False)))} "
            f"bad={int(bool(details.get('invalid_connected_cluster', False)))} "
            f"pen={details.get('spread_penalty', 0.0):.0f}"
        )
        cv2.putText(out, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
        y += line_h
        if y > h - 20:
            break
    return out


def draw_single_cluster_image(shape, entry, selected=False, thickness=4):
    """Draw one direction cluster with its strokes and score."""
    h, w = shape[:2]
    out = np.full((h, w, 3), 255, dtype=np.uint8)
    color = (0, 0, 255) if selected else (30, 120, 220)

    for s in entry.get("strokes", []):
        pts = s["points"].reshape(-1, 1, 2).astype(np.int32)
        cv2.polylines(out, [pts], False, color, thickness, cv2.LINE_AA)
        c = s["center"]
        cv2.putText(out, f"s{int(s['index'])}", (int(c[0]), int(c[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    details = entry.get("details", {})
    direction = entry.get("direction", np.array([0.0, 0.0]))
    title = (
        f"C{entry.get('cluster_id', -1)} score={entry.get('score', 0.0):.1f} "
        f"src={entry.get('source', 'unknown')} strokes={entry.get('indices', [])} selected={selected}"
    )
    cv2.putText(out, title, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    metrics = (
        f"dir=({direction[0]:.2f},{direction[1]:.2f}) n={details.get('n', 0)} "
        f"len={details.get('total_len', 0.0):.1f} str={details.get('mean_straight', 0.0):.3f} "
        f"spread={details.get('perp_spread', 0.0):.1f} loop_pairs={details.get('same_loop_pairs', 0)} "
        f"connected_pairs={details.get('connected_pair_count', 0)} bad={int(bool(details.get('invalid_connected_cluster', False)))}"
    )
    cv2.putText(out, metrics, (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1, cv2.LINE_AA)

    center = np.array([w * 0.5, h * 0.5])
    p1 = center - direction * 80
    p2 = center + direction * 80
    cv2.arrowedLine(out, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color, 3, tipLength=0.15)
    return out


def save_cluster_debug_outputs(debug_dir, img_shape, model):
    """Save cluster score text + overview image + per-cluster images."""
    if debug_dir is None or model is None:
        return
    cluster_debug = model.get("cluster_debug", [])
    if not cluster_debug:
        return

    write_cluster_debug_report(os.path.join(debug_dir, "05b_direction_cluster_scores.txt"), model)
    overview = draw_cluster_overview_image(img_shape, model)
    cv2.imwrite(os.path.join(debug_dir, "05b_direction_cluster_scores.png"), overview)

    cluster_dir = os.path.join(debug_dir, "clusters")
    os.makedirs(cluster_dir, exist_ok=True)
    selected_cluster_id = model.get("selected_cluster_id", None)
    for rank, entry in enumerate(cluster_debug):
        selected = entry.get("cluster_id") == selected_cluster_id
        img = draw_single_cluster_image(img_shape, entry, selected=selected, thickness=4)
        safe_score = int(round(entry.get("score", 0.0)))
        filename = f"rank_{rank:02d}_cluster_{entry.get('cluster_id', -1):02d}_score_{safe_score}.png"
        cv2.imwrite(os.path.join(cluster_dir, filename), img)



def draw_cluster_best_cap_image(shape, entry, cap_candidate=None, selected=False):
    """Draw one cluster's largest-area cap candidate, or a skip/no-cap note."""
    h, w = shape[:2]
    out = np.full((h, w, 3), 255, dtype=np.uint8)
    details = entry.get("details", {})
    cluster_id = entry.get("cluster_id", -1)
    indices = entry.get("indices", [])

    title_color = (0, 0, 255) if selected else (0, 0, 0)
    cv2.putText(
        out,
        f"C{cluster_id} best cap for side={indices} selected={selected}",
        (15, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        title_color,
        1,
        cv2.LINE_AA,
    )

    if details.get("cap_validation_skipped", False):
        reason = details.get("cap_validation_skip_reason", "")
        cv2.putText(
            out,
            f"cap skipped: {reason}",
            (15, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
        return out

    if cap_candidate is None:
        cv2.putText(
            out,
            "no legal closed-loop cap candidate",
            (15, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
        return out

    mask = cv2.dilate(cap_candidate["mask"], np.ones((3, 3), np.uint8), iterations=1)
    out[mask > 0] = (0, 180, 0)
    ctr = cap_candidate.get("center", None)
    if ctr is not None:
        cv2.putText(
            out,
            f"best cap strokes={cap_candidate.get('stroke_indices', [])}",
            (int(ctr[0]) + 8, int(ctr[1]) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (0, 100, 0),
            1,
            cv2.LINE_AA,
        )

    cv2.putText(
        out,
        f"area={cap_candidate.get('area', 0)} score={cap_candidate.get('score', 0.0):.1f}",
        (15, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    return out


def draw_cluster_side_and_best_cap_overlay(shape, entry, cap_candidate=None, selected=False):
    """Draw side cluster strokes in red and that cluster's largest cap in green."""
    h, w = shape[:2]
    out = np.full((h, w, 3), 255, dtype=np.uint8)
    details = entry.get("details", {})
    side_color = (0, 0, 255) if selected else (30, 120, 220)

    if cap_candidate is not None:
        cap_mask = cv2.dilate(cap_candidate["mask"], np.ones((3, 3), np.uint8), iterations=1)
        out[cap_mask > 0] = (0, 180, 0)

    for s in entry.get("strokes", []):
        pts = s["points"].reshape(-1, 1, 2).astype(np.int32)
        cv2.polylines(out, [pts], False, side_color, 4, cv2.LINE_AA)
        c = s["center"]
        cv2.putText(
            out,
            f"s{int(s['index'])}",
            (int(c[0]), int(c[1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    cluster_id = entry.get("cluster_id", -1)
    cap_txt = "no_cap"
    if details.get("cap_validation_skipped", False):
        cap_txt = f"skipped:{details.get('cap_validation_skip_reason', '')}"
    elif cap_candidate is not None:
        cap_txt = f"cap={cap_candidate.get('stroke_indices', [])}, area={cap_candidate.get('area', 0)}"

    cv2.putText(
        out,
        f"C{cluster_id} side={entry.get('indices', [])} selected={selected}",
        (15, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        out,
        cap_txt,
        (15, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.54,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    return out


def sanitize_score_for_filename(score):
    if score <= -1e17:
        return "negInf"
    return str(int(round(score)))


def save_per_cluster_side_cap_outputs(debug_dir, img_shape, model, infos, args):
    """Save side/cap visualizations for ranked side clusters.

    Clusters skipped as subsets of higher-ranked successful-cap clusters are not output.
    """
    if debug_dir is None or model is None:
        return
    cluster_debug = model.get("cluster_debug", [])
    if not cluster_debug:
        return

    out_dir = os.path.join(debug_dir, "cluster_side_caps")
    os.makedirs(out_dir, exist_ok=True)
    selected_cluster_id = model.get("selected_cluster_id", None)

    summary_path = os.path.join(out_dir, "cluster_side_cap_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("==== Per-cluster Side/Cap Visualization Summary ====\n\n")
        f.write("Subset-skipped clusters are omitted from this directory.\n")
        f.write("Hard-rejected clusters do not compute cap; clusters with n<2 do not compute cap.\n")
        f.write("Each emitted cluster stores only the largest-area cap candidate.\n\n")

        for rank, entry in enumerate(cluster_debug):
            details = entry.get("details", {})
            if details.get("skip_percluster_output", False):
                continue

            selected = entry.get("cluster_id") == selected_cluster_id
            cluster_id = entry.get("cluster_id", -1)
            score_tag = sanitize_score_for_filename(float(entry.get("score", 0.0)))
            base = f"rank_{rank:02d}_cluster_{cluster_id:02d}_score_{score_tag}"

            side_img = draw_single_cluster_image(img_shape, entry, selected=selected, thickness=4)
            cv2.imwrite(os.path.join(out_dir, base + "_side.png"), side_img)

            cap_candidate = None
            skip_reason = ""

            pre_rejected = (
                details.get("invalid_connected_cluster", False)
                or float(entry.get("score", -1e18)) <= -1e17
            )
            n_strokes = int(details.get("n", len(entry.get("strokes", []))))

            if pre_rejected:
                skip_reason = "pre_rejected_cluster"
            elif n_strokes < 2:
                skip_reason = "n_lt_2"
            else:
                candidates = extract_cap_loop_candidates_from_strokes(
                    img_shape,
                    infos,
                    entry.get("strokes", []),
                    endpoint_tol=args.cap_loop_endpoint_tol,
                    min_pixels=args.min_cap_pixels,
                    thickness=args.cap_loop_thickness,
                    max_loop_subset_size=args.cap_loop_max_subset_size,
                )
                cap_candidate = largest_area_cap_candidate(candidates)
                if cap_candidate is None:
                    skip_reason = "no_legal_cap"

            cap_img = draw_cluster_best_cap_image(
                img_shape,
                entry,
                cap_candidate=cap_candidate,
                selected=selected,
            )
            cv2.imwrite(os.path.join(out_dir, base + "_bestcap.png"), cap_img)

            overlay_img = draw_cluster_side_and_best_cap_overlay(
                img_shape,
                entry,
                cap_candidate=cap_candidate,
                selected=selected,
            )
            cv2.imwrite(os.path.join(out_dir, base + "_side_bestcap_overlay.png"), overlay_img)

            if cap_candidate is not None:
                f.write(
                    f"rank {rank:02d} cluster {cluster_id:02d} selected={selected} "
                    f"side={entry.get('indices', [])} best_cap_strokes={cap_candidate.get('stroke_indices', [])} "
                    f"best_cap_area={cap_candidate.get('area', 0)} best_cap_score={cap_candidate.get('score', 0.0):.1f}\n"
                )
            else:
                f.write(
                    f"rank {rank:02d} cluster {cluster_id:02d} selected={selected} "
                    f"side={entry.get('indices', [])} cap=NONE reason={skip_reason}\n"
                )


def save_debug_report(path, args, raw_strokes, merged_strokes, infos, line_strokes, model, candidates):
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
            dbg = stroke_direction_debug_values(s)
            f.write(f"  stroke {s['index']:03d}: arc={s['arc']:.1f}, chord={s['chord']:.1f}, straightness={s['straightness']:.3f}, center=({c[0]:.1f},{c[1]:.1f}), pca_axis=({dbg['pca_axis'][0]:.4f},{dbg['pca_axis'][1]:.4f}), pca_angle={dbg['pca_angle']:.2f}\n")
        f.write("\nLine stroke candidates:\n")
        for s in line_strokes:
            c = s["center"]
            dbg = stroke_direction_debug_values(s)
            f.write(f"  stroke {s['index']:03d}: arc={s['arc']:.1f}, straightness={s['straightness']:.3f}, center=({c[0]:.1f},{c[1]:.1f}), pca_axis=({dbg['pca_axis'][0]:.4f},{dbg['pca_axis'][1]:.4f}), pca_angle={dbg['pca_angle']:.2f}\n")
        f.write("\nSelected extrusion model:\n")
        f.write(f"  mode: {model['mode']}\n")
        f.write(f"  score: {model['score']:.3f}\n")
        if model["mode"] == "vp":
            vp = model["vp"]
            f.write(f"  VP: ({vp[0]:.2f}, {vp[1]:.2f})\n")
        else:
            d = model["direction"]
            f.write(f"  direction: ({d[0]:.4f}, {d[1]:.4f})\n")
        if model.get("cluster_debug"):
            f.write("\nDirection cluster debug, ranked by score:\n")
            selected_cluster_id = model.get("selected_cluster_id", None)
            for entry in model.get("cluster_debug", []):
                f.write("  " + format_cluster_debug_entry(entry, selected_cluster_id) + "\n")

        f.write("\nSelected side stroke inliers:\n")
        for s in model["inliers"]:
            c = s["center"]
            dbg = stroke_direction_debug_values(s)
            f.write(f"  stroke {s['index']:03d}: arc={s['arc']:.1f}, straightness={s['straightness']:.3f}, center=({c[0]:.1f},{c[1]:.1f}), pca_axis=({dbg['pca_axis'][0]:.4f},{dbg['pca_axis'][1]:.4f}), pca_angle={dbg['pca_angle']:.2f}\n")
        f.write("\nCap candidates:\n")
        for i, c in enumerate(candidates):
            ctr = c["center"]
            ctr_txt = "None" if ctr is None else f"({ctr[0]:.1f},{ctr[1]:.1f})"
            stroke_txt = c.get("stroke_indices", None)
            stroke_txt = "None" if stroke_txt is None else str(stroke_txt)
            f.write(
                f"  candidate {i:03d}: area={c['area']}, "
                f"endpoints={c['endpoints']}, closedness={c['closedness']:.2f}, "
                f"score={c['score']:.1f}, center={ctr_txt}, strokes={stroke_txt}\n"
            )



def save_debug_outputs(debug_dir, args, img, bw, skel, raw_strokes, merged_strokes, infos, line_strokes, model, side_mask, non_side, candidates):
    if debug_dir is None:
        return
    ensure_dir(debug_dir)
    cv2.imwrite(os.path.join(debug_dir, "00_input.png"), img)
    cv2.imwrite(os.path.join(debug_dir, "01_binary.png"), bw)
    cv2.imwrite(os.path.join(debug_dir, "02_skeleton.png"), skel)
    cv2.imwrite(os.path.join(debug_dir, "02b_skeleton_nodes.png"), draw_skeleton_nodes_debug(skel))
    cv2.imwrite(os.path.join(debug_dir, "03a_raw_strokes.png"), draw_strokes_image(img.shape, raw_strokes, thickness=2, annotate=True))
    cv2.imwrite(os.path.join(debug_dir, "03b_merged_strokes.png"), draw_strokes_image(img.shape, merged_strokes, thickness=2, annotate=True))
    cv2.imwrite(os.path.join(debug_dir, "04_stroke_info.png"), draw_stroke_infos_image(img.shape, infos, thickness=2))
    cv2.imwrite(os.path.join(debug_dir, "05_line_stroke_candidates.png"), draw_line_stroke_candidates_image(img.shape, line_strokes, thickness=3))
    write_stroke_direction_debug_report(os.path.join(debug_dir, "05a_stroke_directions.txt"), infos, line_strokes)
    cv2.imwrite(os.path.join(debug_dir, "05a_stroke_directions.png"), draw_stroke_directions_image(img.shape, infos, line_strokes))
    save_cluster_debug_outputs(debug_dir, img.shape, model)
    save_per_cluster_side_cap_outputs(debug_dir, img.shape, model, infos, args)
    cv2.imwrite(os.path.join(debug_dir, "06_selected_side_strokes.png"), draw_selected_side_strokes_image(img.shape, model, thickness=4))
    cv2.imwrite(os.path.join(debug_dir, "07_non_side_skeleton.png"), draw_non_side_skeleton_image(skel, side_mask, non_side))
    cv2.imwrite(os.path.join(debug_dir, "08_cap_candidates.png"), draw_cap_candidates_debug(img.shape, candidates, max_draw=8))
    save_debug_report(os.path.join(debug_dir, "debug_report.txt"), args, raw_strokes, merged_strokes, infos, line_strokes, model, candidates)


def print_debug(raw_strokes, merged_strokes, infos, line_strokes, model, candidates, output, debug_dir):
    print("==== Result ====")
    print(f"output: {output}")
    if debug_dir:
        print(f"debug dir: {debug_dir}")
    print(f"raw traced strokes: {len(raw_strokes)}")
    print(f"merged strokes: {len(merged_strokes)}")
    print(f"stroke infos: {len(infos)}")
    print(f"straight line stroke candidates: {len(line_strokes)}")
    print(f"mode: {model['mode']}")
    print(f"side strokes: {len(model['inliers'])}")
    print(f"score: {model['score']:.2f}")
    if model["mode"] == "vp":
        vp = model["vp"]
        print(f"vanishing point: ({vp[0]:.2f}, {vp[1]:.2f})")
    else:
        d = model["direction"]
        print(f"parallel direction: ({d[0]:.3f}, {d[1]:.3f})")
    if model.get("cluster_debug"):
        print("\nDirection clusters, ranked by score:")
        selected_cluster_id = model.get("selected_cluster_id", None)
        for entry in model.get("cluster_debug", [])[:12]:
            print("  " + format_cluster_debug_entry(entry, selected_cluster_id))

    print("\nSide stroke inliers:")
    for i, s in enumerate(model["inliers"]):
        c = s["center"]
        dbg = stroke_direction_debug_values(s)
        print(f"  #{i+1}: stroke={s['index']}, arc={s['arc']:.1f}, straightness={s['straightness']:.3f}, center=({c[0]:.1f},{c[1]:.1f}), pca_axis=({dbg['pca_axis'][0]:.4f},{dbg['pca_axis'][1]:.4f}), pca_angle={dbg['pca_angle']:.2f}")
    print("\nCap candidates:")
    for i, c in enumerate(candidates[:8]):
        ctr = c["center"]
        ctr_text = "None" if ctr is None else f"({ctr[0]:.1f}, {ctr[1]:.1f})"
        stroke_txt = c.get("stroke_indices", None)
        stroke_txt = "None" if stroke_txt is None else str(stroke_txt)
        print(f"  #{i+1}: area={c['area']}, endpoints={c['endpoints']}, closedness={c['closedness']:.2f}, score={c['score']:.1f}, center={ctr_text}, strokes={stroke_txt}")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--output", default="result.png")
    parser.add_argument("--debug-dir", default=None)
    parser.add_argument("--no-invert", action="store_true", help="Use this if input is white strokes on black background.")

    parser.add_argument("--close-kernel", type=int, default=3)
    parser.add_argument("--close-iter", type=int, default=1)
    parser.add_argument("--dilate-iter", type=int, default=1)
    parser.add_argument("--min-component-area", type=int, default=12)
    parser.add_argument("--min-skel-component-area", type=int, default=6)

    parser.add_argument("--trace-min-pixels", type=int, default=3)
    parser.add_argument("--disable-corner-split", action="store_true",
                        help="Disable post-trace splitting at sharp polyline corners.")
    parser.add_argument("--split-corner-angle", type=float, default=15.0,
                        help="Split a traced stroke when local turning angle exceeds this many degrees.")
    parser.add_argument("--split-corner-window", type=int, default=8,
                        help="Point window used to estimate before/after tangents for corner splitting.")
    parser.add_argument("--split-corner-min-arc", type=float, default=25.0,
                        help="Minimum arc length of each piece after corner splitting.")
    parser.add_argument("--split-corner-nms-radius", type=int, default=6,
                        help="Non-maximum suppression radius for nearby corner cuts.")
    parser.add_argument("--enable-merge", action="store_true", help="Optional: merge fake endpoint breaks. Disabled by default; raw strokes are used directly.")
    parser.add_argument("--merge-gap", type=float, default=10.0)
    parser.add_argument("--merge-angle", type=float, default=40.0)
    parser.add_argument("--merge-iters", type=int, default=80)
    parser.add_argument("--merge-protect-branch-radius", type=float, default=3.0,
                        help="Do not merge endpoints within this radius of a true branch node.")

    parser.add_argument("--min-stroke-length", type=float, default=30)
    parser.add_argument("--straightness", type=float, default=0.88)
    parser.add_argument("--dist-thresh", type=float, default=8.0)
    parser.add_argument("--angle-thresh", type=float, default=12.0)
    parser.add_argument("--vp", nargs=2, type=float, default=None)

    parser.add_argument("--force-parallel", action="store_true", help="Force weak-perspective / parallel extrusion direction.")
    parser.add_argument("--reject-vp-near-object", action="store_true", help="Reject VP if it lies inside or near the foreground bbox.")
    parser.add_argument("--vp-reject-bbox-margin", type=float, default=0.35)
    parser.add_argument("--vp-score-ratio", type=float, default=1.25)
    parser.add_argument("--parallel-angle-thresh", type=float, default=25.0)

    # Cluster-based side selection controls
    parser.add_argument("--cluster-min-perp-spread", type=float, default=10.0)
    parser.add_argument("--same-loop-endpoint-tol", type=float, default=12.0)
    parser.add_argument("--cluster-count-weight", type=float, default=10000)
    parser.add_argument("--cluster-length-weight", type=float, default=0.7)
    parser.add_argument("--cluster-straightness-weight", type=float, default=10000)
    parser.add_argument("--cluster-spread-weight", type=float, default=1.5)
    parser.add_argument("--cluster-same-loop-penalty", type=float, default=10000)
    parser.add_argument("--cluster-low-spread-penalty", type=float, default=250.0)
    parser.add_argument("--cluster-length-similarity-weight", type=float, default=10000.0,
                        help="Reward clusters whose stroke lengths are similar. Score adds weight * (1 / (1 + length_cv)).")

    parser.add_argument("--side-thickness", type=int, default=4)
    parser.add_argument("--min-cap-pixels", type=int, default=40)
    parser.add_argument("--cap-loop-endpoint-tol", type=float, default=12.0,
                        help="Endpoint tolerance for deciding whether remaining strokes form a closed cap loop.")
    parser.add_argument("--cap-loop-thickness", type=int, default=2,
                        help="Raster thickness used to draw loop-based cap candidate masks.")
    parser.add_argument("--cap-loop-max-subset-size", type=int, default=14,
                        help="Max connected-component stroke count for exhaustive closed-loop subset search.")

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

    traced_strokes = trace_strokes(skel, min_pixels=args.trace_min_pixels)

    # Split graph-traced strokes at sharp geometric corners. trace_strokes() only
    # cuts at topology nodes; this pass cuts a continuous polyline when the local
    # direction changes abruptly, e.g. >15 degrees.
    if args.disable_corner_split:
        raw_strokes = traced_strokes
    else:
        raw_strokes = split_strokes_at_corners(
            traced_strokes,
            angle_thresh=args.split_corner_angle,
            window=args.split_corner_window,
            min_segment_arc=args.split_corner_min_arc,
            nms_radius=args.split_corner_nms_radius,
        )

    # Default behavior: do NOT merge. In these cylinder-like sketches the raw
    # strokes are already meaningful graph-topology primitives. Optional merge
    # is kept only for real broken endpoints, and protects true branch nodes.
    if args.enable_merge:
        branch_points = get_branch_points(skel)
        merged_strokes = merge_strokes_by_endpoint(
            raw_strokes,
            max_gap=args.merge_gap,
            max_angle=args.merge_angle,
            max_iters=args.merge_iters,
            branch_points=branch_points,
            branch_protect_radius=args.merge_protect_branch_radius,
        )
    else:
        merged_strokes = raw_strokes

    infos = build_stroke_infos(merged_strokes)
    model, line_strokes = choose_extrusion_model(args, infos, img.shape, skel)

    if model is None:
        raise RuntimeError(
            "Could not estimate extrusion direction. Try --force-parallel, lower --straightness, "
            "lower --min-stroke-length, increase --merge-gap, or provide --vp manually."
        )
    if len(model["inliers"]) == 0:
        raise RuntimeError("Extrusion direction model has no side stroke inliers.")

    # Cap-validated side cluster selection:
    # compute the largest-area cap candidate for every side cluster with n>=2;
    # select the first ranked selectable side cluster that has a legal cap.
    model, candidates = validate_side_clusters_by_cap_candidates(
        model,
        infos,
        skel.shape,
        endpoint_tol=args.cap_loop_endpoint_tol,
        min_pixels=args.min_cap_pixels,
        thickness=args.cap_loop_thickness,
        max_loop_subset_size=args.cap_loop_max_subset_size,
    )

    side_mask = make_side_mask(skel, model, side_thickness=args.side_thickness)
    non_side = cv2.bitwise_and(skel, cv2.bitwise_not(side_mask))

    save_debug_outputs(
        args.debug_dir, args, img, bw, skel, raw_strokes, merged_strokes,
        infos, line_strokes, model, side_mask, non_side, candidates
    )

    if model.get("cap_validation_failed", False):
        raise RuntimeError(
            "No side cluster produced a legal closed-loop cap candidate. "
            "Debug outputs were saved; check 05b_direction_cluster_scores.txt for cap_validation details."
        )

    draw_result(img, skel, model, side_mask, candidates, args.output)
    print_debug(raw_strokes, merged_strokes, infos, line_strokes, model, candidates, args.output, args.debug_dir)


if __name__ == "__main__":
    main()
