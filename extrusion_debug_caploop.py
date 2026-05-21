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


def build_direction_clusters(line_strokes, angle_thresh=25.0):
    """
    Greedy direction clustering.

    Each stroke is used as a seed; all strokes with similar PCA direction are
    collected. Clusters with identical index sets are deduplicated.
    """
    clusters = {}

    for s in line_strokes:
        group = []
        for t in line_strokes:
            if angle_between_dirs(s["direction"], t["direction"]) <= angle_thresh:
                group.append(t)

        if group:
            key = tuple(sorted(x["index"] for x in group))
            clusters[key] = group

    return list(clusters.values())


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
):
    """
    Side-likeness score for a direction cluster.

    Positive cues:
      - more strokes in the direction family;
      - longer total length;
      - higher mean straightness / lower curvature;
      - enough perpendicular spread between rails/tracks.

    Negative cues:
      - many pairs sharing the same two endpoints, likely a cap loop;
      - very low perpendicular spread for multi-stroke clusters.
    """
    if not cluster:
        return -1e18, {}

    total_len = float(sum(s["arc"] for s in cluster))
    mean_straight = float(np.mean([s["straightness"] for s in cluster]))
    perp_spread = cluster_perp_spread(cluster, direction)
    same_loop_pairs = cluster_same_loop_pairs(cluster, endpoint_tol=same_loop_endpoint_tol)

    spread_penalty = 0.0
    if len(cluster) >= 2 and perp_spread < min_perp_spread:
        spread_penalty = low_spread_penalty

    score = (
        count_weight * len(cluster)
        + length_weight * total_len
        + straightness_weight * mean_straight
        + spread_weight * perp_spread
        - same_loop_penalty_weight * same_loop_pairs
        - spread_penalty
    )

    details = {
        "n": len(cluster),
        "total_len": total_len,
        "mean_straight": mean_straight,
        "perp_spread": perp_spread,
        "same_loop_pairs": same_loop_pairs,
        "spread_penalty": spread_penalty,
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
):
    """
    Cluster-based parallel side selection.

    This keeps the reasonable idea of selecting a dominant direction cluster,
    but does not blindly choose the largest cluster. Instead every direction
    cluster is scored by side-likeness and cap-loop penalties.
    """
    if len(line_strokes) < 1:
        return None

    clusters = build_direction_clusters(line_strokes, angle_thresh=angle_cluster_thresh)
    if not clusters:
        return None

    best = None
    for cluster in clusters:
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
        )

        if best is None or score > best["score"]:
            best = {
                "cluster": cluster,
                "direction": direction,
                "score": score,
                "details": details,
            }

    cluster = best["cluster"]

    return {
        "mode": "parallel",
        "direction": best["direction"],
        "score": float(best["score"]),
        "inliers": cluster,
        "cluster_indices": [s["index"] for s in cluster],
        "cluster_details": best["details"],
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


def extract_cap_loop_candidates_from_strokes(
    image_shape,
    infos,
    side_inliers,
    endpoint_tol=12.0,
    min_pixels=40,
    thickness=2,
):
    """
    Extract cap candidates only from non-side strokes that form closed loops.

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
    for comp in comps:
        if not is_closed_stroke_graph(comp, endpoint_nodes):
            continue
        cand = candidate_from_stroke_loop(image_shape, non_side_infos, comp, thickness=thickness)
        if cand["area"] < min_pixels:
            continue
        candidates.append(cand)
    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates


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
            f.write(f"  stroke {s['index']:03d}: arc={s['arc']:.1f}, chord={s['chord']:.1f}, straightness={s['straightness']:.3f}, center=({c[0]:.1f},{c[1]:.1f})\n")
        f.write("\nLine stroke candidates:\n")
        for s in line_strokes:
            c = s["center"]
            f.write(f"  stroke {s['index']:03d}: arc={s['arc']:.1f}, straightness={s['straightness']:.3f}, center=({c[0]:.1f},{c[1]:.1f})\n")
        f.write("\nSelected extrusion model:\n")
        f.write(f"  mode: {model['mode']}\n")
        f.write(f"  score: {model['score']:.3f}\n")
        if model["mode"] == "vp":
            vp = model["vp"]
            f.write(f"  VP: ({vp[0]:.2f}, {vp[1]:.2f})\n")
        else:
            d = model["direction"]
            f.write(f"  direction: ({d[0]:.4f}, {d[1]:.4f})\n")
        f.write("\nSelected side stroke inliers:\n")
        for s in model["inliers"]:
            c = s["center"]
            f.write(f"  stroke {s['index']:03d}: arc={s['arc']:.1f}, straightness={s['straightness']:.3f}, center=({c[0]:.1f},{c[1]:.1f})\n")
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
    print("\nSide stroke inliers:")
    for i, s in enumerate(model["inliers"]):
        c = s["center"]
        print(f"  #{i+1}: stroke={s['index']}, arc={s['arc']:.1f}, straightness={s['straightness']:.3f}, center=({c[0]:.1f},{c[1]:.1f})")
    print("\nCap candidates:")
    for i, c in enumerate(candidates[:8]):
        ctr = c["center"]
        ctr_text = "None" if ctr is None else f"({ctr[0]:.1f}, {ctr[1]:.1f})"
        stroke_txt = c.get("stroke_indices", None)
        stroke_txt = "None" if stroke_txt is None else str(stroke_txt)
        print(f"  #{i+1}: area={c['area']}, endpoints={c['endpoints']}, closedness={c['closedness']:.2f}, score={c['score']:.1f}, center={ctr_text}, strokes={stroke_txt}")



# ============================================================
# 8. Main
# ============================================================

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
    parser.add_argument("--cluster-count-weight", type=float, default=120.0)
    parser.add_argument("--cluster-length-weight", type=float, default=0.7)
    parser.add_argument("--cluster-straightness-weight", type=float, default=500)
    parser.add_argument("--cluster-spread-weight", type=float, default=1.5)
    parser.add_argument("--cluster-same-loop-penalty", type=float, default=10000)
    parser.add_argument("--cluster-low-spread-penalty", type=float, default=250.0)

    parser.add_argument("--side-thickness", type=int, default=4)
    parser.add_argument("--min-cap-pixels", type=int, default=40)
    parser.add_argument("--cap-loop-endpoint-tol", type=float, default=12.0,
                        help="Endpoint tolerance for deciding whether remaining strokes form a closed cap loop.")
    parser.add_argument("--cap-loop-thickness", type=int, default=2,
                        help="Raster thickness used to draw loop-based cap candidate masks.")

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

    side_mask = make_side_mask(skel, model, side_thickness=args.side_thickness)
    non_side = cv2.bitwise_and(skel, cv2.bitwise_not(side_mask))

    # Cap candidates are now stroke-loop based:
    # only remaining non-side strokes that form a closed endpoint loop are caps.
    # Open arcs / dangling residual strokes are rejected.
    candidates = extract_cap_loop_candidates_from_strokes(
        skel.shape,
        infos,
        model["inliers"],
        endpoint_tol=args.cap_loop_endpoint_tol,
        min_pixels=args.min_cap_pixels,
        thickness=args.cap_loop_thickness,
    )

    save_debug_outputs(
        args.debug_dir, args, img, bw, skel, raw_strokes, merged_strokes,
        infos, line_strokes, model, side_mask, non_side, candidates
    )
    draw_result(img, skel, model, side_mask, candidates, args.output)
    print_debug(raw_strokes, merged_strokes, infos, line_strokes, model, candidates, args.output, args.debug_dir)


if __name__ == "__main__":
    main()
