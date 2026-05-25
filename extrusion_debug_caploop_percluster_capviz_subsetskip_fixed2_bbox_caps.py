# extrusion_debug_caploop.py
# Robust debug version for sketch extrusion detection.
# Key fix: trace_strokes uses crossing number instead of raw degree!=2,
# avoiding false junctions caused by 8-neighbor stair-step skeleton artifacts.

import argparse
import json
import math
import os
import re

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


def split_stroke_at_corners(stroke, angle_thresh=15.0, min_pixels=3, window=5, segment_arc=50.0, split_peak_min_distance=10.0, split_optimize_max_iters=5):
    """
    Split one traced stroke where left/right PCA segment directions change sharply.

    Every interior point is scored by the unoriented PCA axis angle between the
    left and right arc-length windows around that point. Points below
    angle_thresh are discarded. Adjacent high-score points are reduced to local
    peaks, with accepted peaks separated by split_peak_min_distance along the
    stroke arc before splitting.
    """
    pieces, _split_events, _candidate_events, _scan_events = split_stroke_at_corners_with_trace(
        stroke,
        angle_thresh=angle_thresh,
        min_pixels=min_pixels,
        window=window,
        segment_arc=segment_arc,
        split_peak_min_distance=split_peak_min_distance,
        split_optimize_max_iters=split_optimize_max_iters,
    )
    return pieces


def pca_direction_for_corner_segment(points):
    """Return PCA axis direction for a corner-split segment."""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 2:
        return None
    center = pts.mean(axis=0)
    X = pts - center
    cov = X.T @ X / max(len(pts), 1)
    vals, vecs = np.linalg.eigh(cov)
    direction = vecs[:, np.argmax(vals)]
    n = np.linalg.norm(direction)
    if n < 1e-12:
        return None
    return direction / n


def pca_segment_axis_angle(left, right):
    """Unoriented PCA axis angle between two candidate split segments."""
    d1 = pca_direction_for_corner_segment(left)
    d2 = pca_direction_for_corner_segment(right)
    if d1 is None or d2 is None:
        return 0.0
    c = abs(float(np.dot(d1, d2)))
    c = np.clip(c, -1.0, 1.0)
    return float(math.degrees(math.acos(c)))


def walk_left_by_arc(stroke, i, stop_i=0, max_arc=50.0):
    """Return left start index reached by walking backward up to max arc length."""
    start = int(i)
    total = 0.0
    for k in range(int(i), int(stop_i), -1):
        step = float(np.linalg.norm(stroke[k] - stroke[k - 1]))
        total += step
        start = k - 1
        if total >= max_arc:
            break
    return start


def walk_right_by_arc(stroke, i, stop_i=None, max_arc=50.0):
    """Return exclusive right end index reached by walking forward up to max arc length."""
    end = int(i) + 1
    total = 0.0
    if stop_i is None:
        stop_i = len(stroke) - 1
    stop_i = max(int(i), min(int(stop_i), len(stroke) - 1))
    for k in range(int(i), stop_i):
        step = float(np.linalg.norm(stroke[k + 1] - stroke[k]))
        total += step
        end = k + 2
        if total >= max_arc:
            break
    return end


def corner_segment_windows(stroke, i, last_split=0, next_split=None, min_pixels=3, segment_arc=50.0):
    """Return local left/right arc-length windows for segment-angle validation."""
    segment_arc = max(float(segment_arc), 1.0)
    left_start = walk_left_by_arc(stroke, i, stop_i=last_split, max_arc=segment_arc)
    right_end = walk_right_by_arc(stroke, i, stop_i=next_split, max_arc=segment_arc)
    left = stroke[left_start:int(i) + 1]
    right = stroke[int(i):right_end]
    return left, right


def split_stroke_at_corners_with_trace(stroke, angle_thresh=15.0, min_pixels=3, window=5, segment_arc=50.0, split_peak_min_distance=10.0, split_optimize_max_iters=5):
    """Split one stroke at corners and return both pieces and split metadata."""
    if angle_thresh is None:
        return [stroke], [], [], []
    if angle_thresh <= 0:
        return [stroke], [], [], []
    if len(stroke) < max(3, min_pixels * 2 + 1):
        return [stroke], [], [], []

    split_peak_min_distance = max(0.0, float(split_peak_min_distance))
    split_optimize_max_iters = max(1, int(split_optimize_max_iters))
    arc_prefix = np.zeros(len(stroke), dtype=np.float64)
    if len(stroke) > 1:
        steps = np.sqrt(np.sum(np.diff(stroke, axis=0) ** 2, axis=1))
        arc_prefix[1:] = np.cumsum(steps)

    def arc_distance_between_indices(i, j):
        i = max(0, min(int(i), len(stroke) - 1))
        j = max(0, min(int(j), len(stroke) - 1))
        return abs(float(arc_prefix[i] - arc_prefix[j]))

    candidate_events = []
    scan_events = []

    def base_event_for_index(i, pass_id=0, stage="scan", prev_split=None, next_split=None):
        i = int(i)
        p = stroke[i]
        event = {
            "index": int(i),
            "point": (float(p[0]), float(p[1])),
            "raw_angle": 0.0,
            "folded_angle": 0.0,
            "segment_angle": 0.0,
            "segment_left_len": 0,
            "segment_right_len": 0,
            "accepted": False,
            "candidate": False,
            "high_score": False,
            "local_max": False,
            "reject_reason": "",
            "optimization_pass": int(pass_id),
            "optimization_stage": stage,
            "prev_split_index": None if prev_split is None else int(prev_split),
            "next_split_index": None if next_split is None else int(next_split),
        }
        # Debug-only local tangent angle. It no longer gates candidate creation.
        v1 = stroke[i] - stroke[i - 1]
        v2 = stroke[i + 1] - stroke[i]
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 >= 1e-8 and n2 >= 1e-8:
            v1 = v1 / n1
            v2 = v2 / n2
            c = float(np.dot(v1, v2))
            c = np.clip(c, -1.0, 1.0)
            raw_angle = math.degrees(math.acos(c))
            folded_angle = min(raw_angle, 180.0 - raw_angle)
            event["raw_angle"] = float(raw_angle)
            event["folded_angle"] = float(folded_angle)
        return event

    def peak_rank_key(event):
        return (
            float(event.get("segment_angle", 0.0)),
            float(event.get("folded_angle", 0.0)),
            -int(event.get("index", 0)),
        )

    def local_peaks_in_high_score_group(group):
        if len(group) <= 1:
            return list(group)
        peaks = []
        for j, event in enumerate(group):
            score = float(event.get("segment_angle", 0.0))
            left_score = float(group[j - 1].get("segment_angle", 0.0)) if j > 0 else None
            right_score = float(group[j + 1].get("segment_angle", 0.0)) if j + 1 < len(group) else None
            if (left_score is None or score >= left_score) and (right_score is None or score >= right_score):
                peaks.append(event)
        return peaks if peaks else [max(group, key=peak_rank_key)]

    def neighbor_bounds(selected_indices, split_i):
        left = [idx for idx in selected_indices if int(idx) < int(split_i)]
        right = [idx for idx in selected_indices if int(idx) > int(split_i)]
        has_prev_split = bool(left)
        has_next_split = bool(right)
        prev_split = max(left) if has_prev_split else 0
        next_split = min(right) if has_next_split else None
        return prev_split, next_split, has_prev_split, has_next_split

    def evaluate_split_event_between_neighbors(event, selected_indices, pass_id=0, stage="optimize"):
        split_i = int(event.get("index", 0))
        prev_split, next_split, has_prev_split, _has_next_split = neighbor_bounds(selected_indices, split_i)
        updated = dict(event)
        updated["accepted"] = False
        updated["candidate"] = True
        updated["local_max"] = True
        updated["optimization_pass"] = int(pass_id)
        updated["optimization_stage"] = stage
        updated["prev_split_index"] = None if not has_prev_split else int(prev_split)
        updated["next_split_index"] = None if next_split is None else int(next_split)

        if split_i - int(prev_split) < min_pixels:
            updated["reject_reason"] = "near_previous_split"
            updated["optimized_score"] = 0.0
            return updated, False
        if next_split is None:
            if len(stroke) - split_i < min_pixels:
                updated["reject_reason"] = "right_segment_too_short"
                updated["optimized_score"] = 0.0
                return updated, False
        elif int(next_split) - split_i < min_pixels:
            updated["reject_reason"] = "right_segment_too_short_before_next_split"
            updated["optimized_score"] = 0.0
            return updated, False

        if has_prev_split and split_peak_min_distance > 0.0 and arc_distance_between_indices(split_i, int(prev_split)) < split_peak_min_distance:
            updated["reject_reason"] = "near_previous_split_distance"
            updated["optimized_score"] = 0.0
            return updated, False
        if next_split is not None and split_peak_min_distance > 0.0 and arc_distance_between_indices(split_i, int(next_split)) < split_peak_min_distance:
            updated["reject_reason"] = "near_next_split_distance"
            updated["optimized_score"] = 0.0
            return updated, False

        left, right = corner_segment_windows(
            stroke,
            split_i,
            last_split=prev_split,
            next_split=next_split,
            min_pixels=min_pixels,
            segment_arc=segment_arc,
        )
        final_segment_angle = pca_segment_axis_angle(left, right)
        updated["segment_angle"] = float(final_segment_angle)
        updated["segment_left_len"] = int(len(left))
        updated["segment_right_len"] = int(len(right))
        updated["segment_left_points"] = [
            (float(q[0]), float(q[1])) for q in left
        ]
        updated["segment_right_points"] = [
            (float(q[0]), float(q[1])) for q in right
        ]

        if len(left) < 2 or len(right) < 2:
            updated["reject_reason"] = "segment_window_too_short_between_splits"
            updated["optimized_score"] = 0.0
            return updated, False
        if final_segment_angle < angle_thresh:
            updated["reject_reason"] = "segment_angle_below_threshold_between_splits"
            updated["optimized_score"] = float(final_segment_angle)
            return updated, False

        updated["accepted"] = True
        updated["reject_reason"] = ""
        updated["optimized_score"] = float(final_segment_angle)
        return updated, True

    def conflict_rank_key(event):
        return (
            float(event.get("optimized_score", 0.0)),
            float(event.get("raw_segment_angle", event.get("segment_angle", 0.0))),
            float(event.get("folded_angle", 0.0)),
            -int(event.get("index", 0)),
        )

    def resolve_close_split_conflicts(proposed_indices, evaluated, pass_id):
        kept = []
        for idx in sorted(proposed_indices, key=lambda x: conflict_rank_key(evaluated[int(x)]), reverse=True):
            if split_peak_min_distance > 0.0:
                too_close_to = None
                for kept_idx in kept:
                    if arc_distance_between_indices(int(idx), int(kept_idx)) < split_peak_min_distance:
                        too_close_to = int(kept_idx)
                        break
                if too_close_to is not None:
                    rejected = dict(evaluated[int(idx)])
                    rejected["accepted"] = False
                    rejected["reject_reason"] = "rejected_min_distance_to_stronger_split"
                    rejected["conflict_split_index"] = int(too_close_to)
                    rejected["optimization_pass"] = int(pass_id)
                    evaluated[int(idx)] = rejected
                    continue
            kept.append(int(idx))
        return set(kept)

    def split_segments(selected_indices):
        interior = sorted(
            {
                int(idx)
                for idx in selected_indices
                if 0 < int(idx) < len(stroke) - 1
            }
        )
        bounds = [0] + interior + [len(stroke) - 1]
        return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]

    def scan_event_between_neighbors(i, prev_split, next_split, pass_id=0, stage="scan"):
        debug_next_split = None
        if next_split is not None and int(next_split) < len(stroke) - 1:
            debug_next_split = int(next_split)
        event = base_event_for_index(
            int(i),
            pass_id=pass_id,
            stage=stage,
            prev_split=None if int(prev_split) == 0 else int(prev_split),
            next_split=debug_next_split,
        )
        left, right = corner_segment_windows(
            stroke,
            int(i),
            last_split=int(prev_split),
            next_split=next_split,
            min_pixels=min_pixels,
            segment_arc=segment_arc,
        )
        segment_angle = pca_segment_axis_angle(left, right)
        event["segment_angle"] = float(segment_angle)
        event["segment_left_len"] = int(len(left))
        event["segment_right_len"] = int(len(right))

        if int(i) - int(prev_split) < min_pixels:
            event["reject_reason"] = "near_previous_split"
        elif next_split is None:
            if len(stroke) - int(i) < min_pixels:
                event["reject_reason"] = "right_segment_too_short"
            elif len(left) < 2 or len(right) < 2:
                event["reject_reason"] = "segment_window_too_short"
            elif segment_angle < angle_thresh:
                event["reject_reason"] = "segment_angle_below_threshold"
            else:
                event["high_score"] = True
                event["reject_reason"] = "non_maximum_suppressed"
        elif int(next_split) - int(i) < min_pixels:
            event["reject_reason"] = "right_segment_too_short_before_next_split"
        elif int(prev_split) > 0 and split_peak_min_distance > 0.0 and arc_distance_between_indices(int(i), int(prev_split)) < split_peak_min_distance:
            event["reject_reason"] = "near_previous_split_distance"
        elif next_split is not None and int(next_split) < len(stroke) - 1 and split_peak_min_distance > 0.0 and arc_distance_between_indices(int(i), int(next_split)) < split_peak_min_distance:
            event["reject_reason"] = "near_next_split_distance"
        elif len(left) < 2 or len(right) < 2:
            event["reject_reason"] = "segment_window_too_short"
        elif segment_angle < angle_thresh:
            event["reject_reason"] = "segment_angle_below_threshold"
        else:
            event["high_score"] = True
            event["reject_reason"] = "non_maximum_suppressed"
        return event

    def scan_candidates_for_selected_splits(selected_indices, pass_id=0, stage="rescan"):
        round_scan_events = []
        candidate_peaks = []
        group_id = 0

        def flush_group(group):
            nonlocal group_id
            if not group:
                return
            for peak in local_peaks_in_high_score_group(group):
                peak["candidate"] = True
                peak["local_max"] = True
                peak["accepted"] = False
                peak["reject_reason"] = "not_selected"
                peak["candidate_group"] = int(group_id)
                peak["raw_segment_angle"] = float(peak.get("segment_angle", 0.0))
                peak["raw_segment_left_len"] = int(peak.get("segment_left_len", 0))
                peak["raw_segment_right_len"] = int(peak.get("segment_right_len", 0))
                candidate_peaks.append(dict(peak))
            group_id += 1

        for prev_split, next_split in split_segments(selected_indices):
            group = []
            for i in range(int(prev_split) + 1, int(next_split)):
                event = scan_event_between_neighbors(
                    i,
                    prev_split,
                    next_split,
                    pass_id=pass_id,
                    stage=stage,
                )
                round_scan_events.append(event)
                if event.get("high_score", False):
                    if group and int(event["index"]) == int(group[-1]["index"]) + 1:
                        group.append(event)
                    else:
                        flush_group(group)
                        group = [event]
                else:
                    flush_group(group)
                    group = []
            flush_group(group)

        return round_scan_events, candidate_peaks

    selected_indices = set()
    final_evaluated = {}
    seen_selected_sets = {tuple()}
    max_opt_iters = split_optimize_max_iters

    for pass_id in range(max_opt_iters):
        old_indices = set(selected_indices)
        round_scan_events, candidate_peaks = scan_candidates_for_selected_splits(
            selected_indices,
            pass_id=pass_id,
            stage="rescan",
        )
        scan_events.extend(round_scan_events)

        candidate_pool = {
            int(event.get("index", 0)): dict(event)
            for event in candidate_peaks
        }
        for split_i in selected_indices:
            split_i = int(split_i)
            if split_i not in candidate_pool and 0 < split_i < len(stroke) - 1:
                carried = base_event_for_index(
                    split_i,
                    pass_id=pass_id,
                    stage="carry_selected",
                )
                carried["candidate"] = True
                carried["carried_split"] = True
                carried["reject_reason"] = "carried_from_previous_pass"
                candidate_pool[split_i] = carried

        evaluated = {}
        proposed_indices = set()
        for split_i, event in sorted(candidate_pool.items()):
            context_indices = set(selected_indices)
            context_indices.discard(int(split_i))
            updated, ok = evaluate_split_event_between_neighbors(
                event,
                context_indices,
                pass_id=pass_id,
                stage="evaluate",
            )
            evaluated[int(split_i)] = updated
            if ok:
                proposed_indices.add(int(split_i))

        selected_indices = resolve_close_split_conflicts(proposed_indices, evaluated, pass_id)
        final_evaluated = evaluated
        for event in round_scan_events:
            if not event.get("candidate", False):
                continue
            split_i = int(event.get("index", 0))
            updated = evaluated.get(split_i)
            if updated is None:
                continue
            event["accepted"] = bool(split_i in selected_indices and updated.get("accepted", False))
            event["reject_reason"] = "" if event["accepted"] else updated.get("reject_reason", "")
            event["segment_angle"] = float(updated.get("segment_angle", event.get("segment_angle", 0.0)))
            event["segment_left_len"] = int(updated.get("segment_left_len", event.get("segment_left_len", 0)))
            event["segment_right_len"] = int(updated.get("segment_right_len", event.get("segment_right_len", 0)))
        selected_key = tuple(sorted(selected_indices))
        if selected_indices == old_indices or selected_key in seen_selected_sets:
            break
        seen_selected_sets.add(selected_key)

    final_scan_events, final_candidate_peaks = scan_candidates_for_selected_splits(
        selected_indices,
        pass_id=max_opt_iters,
        stage="final_rescan",
    )
    scan_events.extend(final_scan_events)
    final_pool = {
        int(event.get("index", 0)): dict(event)
        for event in final_candidate_peaks
    }
    for split_i in selected_indices:
        split_i = int(split_i)
        if split_i not in final_pool and 0 < split_i < len(stroke) - 1:
            carried = base_event_for_index(
                split_i,
                pass_id=max_opt_iters,
                stage="final_carry_selected",
            )
            carried["candidate"] = True
            carried["carried_split"] = True
            carried["reject_reason"] = "carried_from_previous_pass"
            final_pool[split_i] = carried

    final_evaluated = {}
    for split_i, event in sorted(final_pool.items()):
        context_indices = set(selected_indices)
        context_indices.discard(int(split_i))
        updated, _ok = evaluate_split_event_between_neighbors(
            event,
            context_indices,
            pass_id=max_opt_iters,
            stage="final",
        )
        final_evaluated[int(split_i)] = updated

    split_events = []
    for event in final_scan_events:
        if not event.get("candidate", False):
            continue
        split_i = int(event.get("index", 0))
        updated = final_evaluated.get(split_i)
        if updated is None:
            continue
        event["accepted"] = bool(split_i in selected_indices and updated.get("accepted", False))
        event["reject_reason"] = "" if event["accepted"] else updated.get("reject_reason", "")
        event["segment_angle"] = float(updated.get("segment_angle", event.get("segment_angle", 0.0)))
        event["segment_left_len"] = int(updated.get("segment_left_len", event.get("segment_left_len", 0)))
        event["segment_right_len"] = int(updated.get("segment_right_len", event.get("segment_right_len", 0)))

    for event in sorted(final_evaluated.values(), key=lambda e: int(e.get("index", 0))):
        split_i = int(event.get("index", 0))
        final_event = dict(event)
        final_event["accepted"] = bool(split_i in selected_indices and final_event.get("accepted", False))
        if split_i not in selected_indices:
            final_event["accepted"] = False
            if not final_event.get("reject_reason", ""):
                final_event["reject_reason"] = "not_selected"
        candidate_events.append(dict(final_event))
        if bool(final_event.get("accepted", False)):
            split_events.append(dict(final_event))

    if not split_events:
        return [stroke], [], candidate_events, scan_events

    pieces = []
    start = 0
    for event in split_events:
        split_i = int(event["index"])
        piece = stroke[start:split_i + 1]
        if len(piece) >= min_pixels:
            pieces.append(piece.astype(np.float32))
        start = split_i

    tail = stroke[start:]
    if len(tail) >= min_pixels:
        pieces.append(tail.astype(np.float32))

    return (pieces if pieces else [stroke]), split_events, candidate_events, scan_events


def split_strokes_at_corners(strokes, angle_thresh=None, min_pixels=3, window=5, segment_arc=50.0, split_peak_min_distance=10.0, split_optimize_max_iters=5):
    """Apply corner splitting to every traced stroke."""
    if angle_thresh is None:
        return strokes

    out = []
    for stroke in strokes:
        out.extend(
            split_stroke_at_corners(
                stroke,
                angle_thresh=angle_thresh,
                min_pixels=min_pixels,
                window=window,
                segment_arc=segment_arc,
                split_peak_min_distance=split_peak_min_distance,
                split_optimize_max_iters=split_optimize_max_iters,
            )
        )
    return out


def split_strokes_at_corners_with_trace(strokes, angle_thresh=None, min_pixels=3, window=5, segment_arc=50.0, split_peak_min_distance=10.0, split_optimize_max_iters=5):
    """Apply corner splitting and record input-to-output stroke mapping."""
    if angle_thresh is None:
        trace = []
        for i, stroke in enumerate(strokes):
            trace.append({
                "input_index": int(i),
                "input_len": int(len(stroke)),
                "split": False,
                "split_events": [],
                "candidate_events": [],
                "scan_events": [],
                "output_indices": [int(i)],
                "output_lengths": [int(len(stroke))],
            })
        return strokes, trace

    out = []
    trace = []
    for i, stroke in enumerate(strokes):
        pieces, split_events, candidate_events, scan_events = split_stroke_at_corners_with_trace(
            stroke,
            angle_thresh=angle_thresh,
            min_pixels=min_pixels,
            window=window,
            segment_arc=segment_arc,
            split_peak_min_distance=split_peak_min_distance,
            split_optimize_max_iters=split_optimize_max_iters,
        )
        output_indices = list(range(len(out), len(out) + len(pieces)))
        out.extend(pieces)
        trace.append({
            "input_index": int(i),
            "input_len": int(len(stroke)),
            "split": len(split_events) > 0,
            "split_events": split_events,
            "candidate_events": candidate_events,
            "scan_events": scan_events,
            "output_indices": [int(x) for x in output_indices],
            "output_lengths": [int(len(p)) for p in pieces],
        })
    return out, trace


def endpoint_candidates_for_merge(stroke):
    """Return both endpoints with their endpoint index."""
    return [(0, stroke[0]), (1, stroke[-1])]


def stroke_axis_for_merge(stroke):
    """Unoriented PCA axis for a raw stroke polyline."""
    if len(stroke) < 2:
        return None
    _line, direction, _center = fit_line_to_points(stroke)
    return canonical_axis_direction(direction)


def stroke_endpoint_axis_for_merge(stroke):
    """Unoriented chord axis between a stroke's two endpoints."""
    if len(stroke) < 2:
        return None
    direction = np.asarray(stroke[-1], dtype=np.float64) - np.asarray(stroke[0], dtype=np.float64)
    if np.linalg.norm(direction) < 1e-8:
        return None
    return canonical_axis_direction(direction)


def merge_polyline_by_endpoints(s1, end1, s2, end2):
    """Merge two raw polylines by placing matching endpoints together."""
    a = s1.copy()
    b = s2.copy()
    if end1 == 0:
        a = a[::-1]
    if end2 == 1:
        b = b[::-1]
    if np.linalg.norm(a[-1] - b[0]) < 1e-6:
        return np.vstack([a, b[1:]]).astype(np.float32)
    return np.vstack([a, b]).astype(np.float32)


def can_post_split_merge(s1, s2, max_gap=3.0, max_angle=12.0):
    """Return merge metadata if two post-split strokes look like one line."""
    d1 = stroke_axis_for_merge(s1)
    d2 = stroke_axis_for_merge(s2)
    if d1 is None or d2 is None:
        return None

    angle = angle_between_dirs(d1, d2)
    if angle > max_angle:
        return None

    chord_d1 = stroke_endpoint_axis_for_merge(s1)
    chord_d2 = stroke_endpoint_axis_for_merge(s2)
    if chord_d1 is None or chord_d2 is None:
        return None

    best = None
    for end1, p1 in endpoint_candidates_for_merge(s1):
        for end2, p2 in endpoint_candidates_for_merge(s2):
            gap = float(np.linalg.norm(np.asarray(p1, dtype=np.float64) - np.asarray(p2, dtype=np.float64)))
            if gap > max_gap:
                continue
            merged = merge_polyline_by_endpoints(s1, end1, s2, end2)
            merged_chord = stroke_endpoint_axis_for_merge(merged)
            if merged_chord is None:
                continue
            merged_angle_1 = angle_between_dirs(merged_chord, chord_d1)
            merged_angle_2 = angle_between_dirs(merged_chord, chord_d2)
            merged_endpoint_angle = max(merged_angle_1, merged_angle_2)
            if merged_endpoint_angle > max_angle:
                continue
            if best is None or gap < best["gap"]:
                merge_point = 0.5 * (
                    np.asarray(p1, dtype=np.float64)
                    + np.asarray(p2, dtype=np.float64)
                )
                best = {
                    "end1": int(end1),
                    "end2": int(end2),
                    "gap": float(gap),
                    "angle": float(angle),
                    "merged_endpoint_angle": float(merged_endpoint_angle),
                    "merged_endpoint_angle_1": float(merged_angle_1),
                    "merged_endpoint_angle_2": float(merged_angle_2),
                    "merge_point": (float(merge_point[0]), float(merge_point[1])),
                }
    return best


def third_stroke_endpoint_near_merge_point(strokes, i, j, info, radius=3.0):
    """Return nearby third-stroke endpoint metadata when a merge point is a junction."""
    if radius is None or float(radius) <= 0.0:
        return None
    merge_point = np.asarray(info.get("merge_point", (0.0, 0.0)), dtype=np.float64)
    radius = float(radius)
    best = None
    for k, stroke in enumerate(strokes):
        if k in (i, j):
            continue
        for endpoint_i, p in endpoint_candidates_for_merge(stroke):
            dist = float(np.linalg.norm(np.asarray(p, dtype=np.float64) - merge_point))
            if dist > radius:
                continue
            if best is None or dist < best["distance"]:
                best = {
                    "stroke_index": int(k),
                    "endpoint": int(endpoint_i),
                    "distance": float(dist),
                    "point": (float(p[0]), float(p[1])),
                    "merge_point": (float(merge_point[0]), float(merge_point[1])),
                }
    return best


def merge_post_corner_split_strokes(strokes, max_gap=3.0, max_angle=12.0, max_iters=80, protect_junction_radius=3.0):
    """
    Merge accidental corner-split fragments that still share a nearly collinear axis.

    This is intentionally narrower than the optional topology merge: it only
    uses endpoint distance and PCA axis angle after corner splitting.  A merge
    endpoint is protected when any third stroke endpoint, even a short one, is
    already attached there.
    """
    strokes = [s.copy() for s in strokes]
    trace = []

    for _ in range(max_iters):
        best = None
        for i in range(len(strokes)):
            for j in range(i + 1, len(strokes)):
                info = can_post_split_merge(strokes[i], strokes[j], max_gap=max_gap, max_angle=max_angle)
                if info is None:
                    continue
                protected = third_stroke_endpoint_near_merge_point(
                    strokes,
                    i,
                    j,
                    info,
                    radius=protect_junction_radius,
                )
                if protected is not None:
                    trace.append({
                        "action": "skip_junction_protected",
                        "left_index": int(i),
                        "right_index": int(j),
                        "gap": float(info["gap"]),
                        "angle": float(info["angle"]),
                        "merged_endpoint_angle": float(info.get("merged_endpoint_angle", 0.0)),
                        "merge_point": info.get("merge_point", None),
                        "protected_by_stroke": int(protected["stroke_index"]),
                        "protected_by_endpoint": int(protected["endpoint"]),
                        "protected_by_distance": float(protected["distance"]),
                        "protected_by_point": protected["point"],
                    })
                    continue
                cost = info["gap"] + 0.1 * info["angle"]
                if best is None or cost < best["cost"]:
                    best = {
                        "i": i,
                        "j": j,
                        "cost": float(cost),
                        **info,
                    }

        if best is None:
            break

        i, j = best["i"], best["j"]
        merged = merge_polyline_by_endpoints(strokes[i], best["end1"], strokes[j], best["end2"])
        trace.append({
            "action": "merge",
            "left_index": int(i),
            "right_index": int(j),
            "gap": float(best["gap"]),
            "angle": float(best["angle"]),
            "merged_endpoint_angle": float(best.get("merged_endpoint_angle", 0.0)),
            "merge_point": best.get("merge_point", None),
            "merged_len": int(len(merged)),
        })
        strokes = [s for k, s in enumerate(strokes) if k not in (i, j)] + [merged]

    return strokes, trace


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


def axis_angle_diff(a, b):
    """Unoriented angle difference between two axis angles in [0, 180)."""
    d = abs(float(a) - float(b))
    return float(min(d, 180.0 - d))


def mean_axis_angle_from_angles(angles):
    """Mean unoriented axis angle using double-angle circular averaging."""
    if not angles:
        return 0.0
    vals = np.asarray(angles, dtype=np.float64)
    radians = np.deg2rad(vals * 2.0)
    x = float(np.mean(np.cos(radians)))
    y = float(np.mean(np.sin(radians)))
    if abs(x) < 1e-12 and abs(y) < 1e-12:
        return float(vals[0] % 180.0)
    ang = math.degrees(math.atan2(y, x)) * 0.5
    if ang < 0:
        ang += 180.0
    if ang >= 180.0:
        ang -= 180.0
    return float(ang)


def find_largest_axis_angle_gap(sorted_items):
    """Return index of the largest gap in sorted angle items on [0, 180)."""
    n = len(sorted_items)
    if n <= 1:
        return 0
    best_i = 0
    best_gap = -1.0
    for i in range(n):
        a = float(sorted_items[i]["angle"])
        b = float(sorted_items[(i + 1) % n]["angle"])
        if i == n - 1:
            b += 180.0
        gap = b - a
        if gap > best_gap:
            best_gap = gap
            best_i = i
    return best_i


def unwrap_axis_angle_items(items):
    """
    Sort axis-angle items and cut at the largest empty gap.

    This handles the 0/180 wrap so angles such as 179 and 1 are adjacent in the
    resulting linear order.
    """
    if not items:
        return []
    sorted_items = sorted(items, key=lambda x: x["angle"])
    gap_i = find_largest_axis_angle_gap(sorted_items)
    ordered = sorted_items[gap_i + 1:] + sorted_items[:gap_i + 1]
    if not ordered:
        return []

    base = float(ordered[0]["angle"])
    out = []
    for item in ordered:
        unwrapped = float(item["angle"])
        if unwrapped < base:
            unwrapped += 180.0
        e = dict(item)
        e["unwrapped_angle"] = unwrapped
        out.append(e)
    return out


def group_mean_angle_is_valid(items, angle_thresh):
    """True if every item is within threshold of the group's mean axis angle."""
    if not items:
        return False, 0.0, 0.0
    angles = [float(x["angle"]) for x in items]
    mean_angle = mean_axis_angle_from_angles(angles)
    max_diff = max(axis_angle_diff(a, mean_angle) for a in angles)
    return max_diff <= angle_thresh, mean_angle, float(max_diff)


def axis_angle_mode_candidates(items, angle_thresh, max_iters=16):
    """
    Return compact center-based direction candidates for unoriented axis angles.

    Each seed performs a small mean-shift search on the circular 0/180 axis
    domain.  A candidate is valid only when every member is within
    angle_thresh of the candidate mean.
    """
    candidates = {}
    for seed in items:
        center = float(seed["angle"])
        prev_key = None
        support = [seed]
        for _ in range(max_iters):
            support = [x for x in items if axis_angle_diff(x["angle"], center) <= angle_thresh]
            if not support:
                support = [seed]
            key = tuple(sorted(int(x["stroke"]["index"]) for x in support))
            new_center = mean_axis_angle_from_angles([x["angle"] for x in support])
            if key == prev_key and axis_angle_diff(new_center, center) < 1e-6:
                break
            prev_key = key
            center = new_center

        ok, mean_angle, max_diff = group_mean_angle_is_valid(support, angle_thresh)
        if not ok:
            continue
        key = tuple(sorted(int(x["stroke"]["index"]) for x in support))
        existing = candidates.get(key)
        total_arc = float(sum(x["stroke"].get("arc", 0.0) for x in support))
        candidate = {
            "items": support,
            "mean_angle": float(mean_angle),
            "max_mean_angle_diff": float(max_diff),
            "total_arc": total_arc,
        }
        if existing is None or (candidate["max_mean_angle_diff"], -candidate["total_arc"]) < (
            existing["max_mean_angle_diff"],
            -existing["total_arc"],
        ):
            candidates[key] = candidate
    return list(candidates.values())


def build_centered_axis_angle_groups(items, angle_thresh):
    """
    Partition strokes by repeatedly taking the strongest compact angle mode.

    This avoids the old boundary-pair failure where a stroke near a dense mode
    was consumed by an earlier weak pair during left-to-right angle scanning.
    """
    remaining = list(items)
    groups = []
    while remaining:
        candidates = axis_angle_mode_candidates(remaining, angle_thresh)
        if not candidates:
            x = remaining.pop(0)
            groups.append({
                "items": [x],
                "mean_angle": float(x["angle"]),
                "max_mean_angle_diff": 0.0,
            })
            continue

        candidates.sort(
            key=lambda c: (
                -len(c["items"]),
                c["max_mean_angle_diff"],
                -c["total_arc"],
                tuple(sorted(int(x["stroke"]["index"]) for x in c["items"])),
            )
        )
        best = candidates[0]
        groups.append(best)
        used = {int(x["stroke"]["index"]) for x in best["items"]}
        remaining = [x for x in remaining if int(x["stroke"]["index"]) not in used]

    return groups


def build_direction_clusters(
    line_strokes,
    angle_thresh=25.0,
    endpoint_tol=12.0,
    enumerate_subsets=True,
    max_subset_size=12,
):
    """
    Build direction groups by clustering PCA unoriented axis angles.

    Current rule:
      1. every input stroke already has a PCA direction in s["direction"];
      2. convert it to an unoriented axis angle in [0, 180);
      3. find compact center/mode candidates using double-angle circular means;
      4. repeatedly take the strongest mode and remove its members.

    Notes:
      - Direction is UNORIENTED: d and -d are treated as the same axis.
      - This avoids chain merging from graph connected components and avoids
        greedy boundary-pair errors from sorted angle scans.
      - endpoint_tol, enumerate_subsets, and max_subset_size are kept in the
        signature for CLI/backward compatibility, but are not used here.
    """
    n = len(line_strokes)
    if n == 0:
        return []

    items = []
    for s in line_strokes:
        dbg = stroke_direction_debug_values(s)
        items.append({
            "stroke": s,
            "angle": float(dbg["pca_angle"]),
        })

    grouped_items = build_centered_axis_angle_groups(items, angle_thresh)
    out = []
    for group in grouped_items:
        comp = [x["stroke"] for x in group["items"]]
        out.append({
            "strokes": comp,
            "source": "direction_mean_angle_cluster",
            "mean_angle": float(group["mean_angle"]),
            "max_mean_angle_diff": float(group["max_mean_angle_diff"]),
        })

    # Stable order: larger groups first, then by stroke indices.
    out.sort(key=lambda e: (-len(e["strokes"]), tuple(sorted(int(st["index"]) for st in e["strokes"]))))
    return out


def refine_direction_clusters_by_nearest_mean(cluster_entries, angle_thresh, max_iters=8):
    if not cluster_entries:
        return []

    strokes = []
    for entry in cluster_entries:
        strokes.extend(entry["strokes"])
    if not strokes:
        return []

    means = [float(entry.get("mean_angle", axis_angle_0_180(mean_direction(entry["strokes"])))) for entry in cluster_entries]
    stroke_angles = {
        int(s["index"]): float(stroke_direction_debug_values(s)["pca_angle"])
        for s in strokes
    }

    assignments = None
    for _ in range(max_iters):
        groups = [[] for _ in means]
        extra_groups = []
        new_assignments = {}
        for s in strokes:
            sid = int(s["index"])
            angle = stroke_angles[sid]
            candidates = [
                (axis_angle_diff(angle, mean), gi)
                for gi, mean in enumerate(means)
                if axis_angle_diff(angle, mean) <= angle_thresh
            ]
            if candidates:
                _diff, gi = min(candidates, key=lambda item: (item[0], item[1]))
                groups[gi].append(s)
                new_assignments[sid] = gi
            else:
                gi = len(groups) + len(extra_groups)
                extra_groups.append([s])
                new_assignments[sid] = gi

        groups.extend(extra_groups)
        groups = [g for g in groups if g]
        if assignments == new_assignments:
            break
        assignments = new_assignments
        means = [
            mean_axis_angle_from_angles([stroke_angles[int(s["index"])] for s in group])
            for group in groups
        ]

    refined = []
    for group in groups:
        angles = [stroke_angles[int(s["index"])] for s in group]
        mean_angle = mean_axis_angle_from_angles(angles)
        max_diff = max(axis_angle_diff(a, mean_angle) for a in angles) if angles else 0.0
        refined.append({
            "strokes": group,
            "source": "direction_mean_angle_cluster",
            "mean_angle": float(mean_angle),
            "max_mean_angle_diff": float(max_diff),
        })
    return refined


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
            "mean_angle": cluster_entry.get("mean_angle", axis_angle_0_180(direction)),
            "max_mean_angle_diff": cluster_entry.get("max_mean_angle_diff", 0.0),
        }
        scored_clusters.append(entry)

        if best_i is None or score > best_score:
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
    direction_group_strokes = [s for s in infos if s["arc"] >= args.min_stroke_length]

    if args.force_parallel:
        return fallback_parallel_direction(
            direction_group_strokes,
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
        direction_group_strokes,
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


def estimate_enclosed_area_from_loop_mask(mask, close_kernel=5):
    """
    Estimate the filled area enclosed by a loop stroke mask.

    The returned area excludes the stroke pixels themselves. Small noisy loops
    therefore get a small enclosed_area even if their rasterized line has enough
    pixels to pass min_cap_pixels.
    """
    if mask is None or mask.size == 0:
        return 0, np.zeros_like(mask)

    work = (mask > 0).astype(np.uint8) * 255
    if close_kernel and close_kernel > 1:
        k = np.ones((int(close_kernel), int(close_kernel)), np.uint8)
        work = cv2.morphologyEx(work, cv2.MORPH_CLOSE, k, iterations=1)

    h, w = work.shape
    flood = work.copy()
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)

    interior = cv2.bitwise_not(flood)
    interior[work > 0] = 0
    enclosed_area = int(np.count_nonzero(interior))
    return enclosed_area, interior


def cluster_points_by_distance(points, tol=12.0):
    """
    Endpoint clustering by pairwise distance graph.

    Any two real endpoints within tol are connected, then connected components
    of that endpoint graph become endpoint nodes. This avoids the old greedy
    center-update behavior where cluster assignment depended on point order and
    a drifting center could miss visually close endpoints.
    """
    pts = [np.asarray(p, dtype=np.float64) for p in points]
    n = len(pts)
    if n == 0:
        return [], []

    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        ra = find(a)
        rb = find(b)
        if ra == rb:
            return
        if rb < ra:
            ra, rb = rb, ra
        parent[rb] = ra

    tol2 = float(tol) * float(tol)
    for i in range(n):
        for j in range(i + 1, n):
            d = pts[i] - pts[j]
            if float(np.dot(d, d)) <= tol2:
                union(i, j)

    root_to_label = {}
    labels = []
    grouped = []
    for i in range(n):
        root = find(i)
        if root not in root_to_label:
            root_to_label[root] = len(grouped)
            grouped.append([])
        label = root_to_label[root]
        labels.append(label)
        grouped[label].append(pts[i])

    centers = [np.mean(group, axis=0) for group in grouped]
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


def stroke_endpoint_points(stroke_info):
    """Return the two real endpoints of one stroke-info dict."""
    return [stroke_info["points"][0], stroke_info["points"][-1]]


def endpoint_points_close(a, b, tol):
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return float(np.dot(d, d)) <= float(tol) * float(tol)


def strokes_connected_by_endpoint_tol(s1, s2, endpoint_tol=12.0):
    """True when any real endpoint pair between two strokes is within tolerance."""
    for p in stroke_endpoint_points(s1):
        for q in stroke_endpoint_points(s2):
            if endpoint_points_close(p, q, endpoint_tol):
                return True
    return False


def build_nearest_endpoint_matches(strokes, endpoint_tol=12.0):
    """
    For every real endpoint, keep only the nearest other endpoint within tol.

    Keys and values are (stroke_local_index, endpoint_index).  If an endpoint
    has no neighbor within tol, it has no entry in the returned dict.
    """
    endpoints = []
    for si, s in enumerate(strokes):
        for ei, p in enumerate(stroke_endpoint_points(s)):
            endpoints.append({
                "key": (int(si), int(ei)),
                "point": np.asarray(p, dtype=np.float64),
            })

    matches = {}
    tol2 = float(endpoint_tol) * float(endpoint_tol)
    for i, item in enumerate(endpoints):
        best_key = None
        best_d2 = float("inf")
        for j, other in enumerate(endpoints):
            if i == j:
                continue
            if item["key"][0] == other["key"][0]:
                continue
            d = item["point"] - other["point"]
            d2 = float(np.dot(d, d))
            if d2 > tol2:
                continue
            if d2 < best_d2:
                best_d2 = d2
                best_key = other["key"]
        if best_key is not None:
            matches[item["key"]] = best_key
    return matches


def connected_components_by_endpoint_proximity(strokes, endpoint_tol=12.0):
    """
    Connected components over strokes using direct endpoint proximity.

    This does not merge endpoints into graph nodes first.  It follows the user's
    intended search: start from a non-side stroke, add any non-side stroke whose
    real endpoint is within endpoint_tol, continue until no new connected stroke
    exists, then start again from the remaining strokes.
    """
    n = len(strokes)
    visited = set()
    comps = []
    nearest_matches = build_nearest_endpoint_matches(strokes, endpoint_tol=endpoint_tol)
    stroke_neighbors = {i: set() for i in range(n)}
    for (si, _ei), (sj, _ej) in nearest_matches.items():
        if si == sj:
            continue
        stroke_neighbors[si].add(sj)
        stroke_neighbors[sj].add(si)

    for start in range(n):
        if start in visited:
            continue
        comp = []
        stack = [start]
        visited.add(start)

        while stack:
            i = stack.pop()
            comp.append(i)
            for j in sorted(stroke_neighbors.get(i, [])):
                if j not in visited:
                    visited.add(j)
                    stack.append(j)

        comps.append(comp)

    return comps


def endpoint_connection_degree_in_component(strokes, comp, stroke_local_i, endpoint_i, endpoint_tol=12.0):
    """
    Count the nearest-neighbor endpoint connection for one endpoint in a component.

    Each endpoint can contribute at most one connection: its nearest other
    endpoint within endpoint_tol.  If that nearest endpoint is outside this
    component, the degree is treated as 0 for this component.
    """
    if len(comp) == 1:
        p0, p1 = stroke_endpoint_points(strokes[stroke_local_i])
        if endpoint_i == 0 and endpoint_points_close(p0, p1, endpoint_tol):
            return 1, [(int(strokes[stroke_local_i]["index"]), 1)]
        if endpoint_i == 1 and endpoint_points_close(p1, p0, endpoint_tol):
            return 1, [(int(strokes[stroke_local_i]["index"]), 0)]
        return 0, []

    nearest_matches = build_nearest_endpoint_matches(strokes, endpoint_tol=endpoint_tol)
    match = nearest_matches.get((int(stroke_local_i), int(endpoint_i)), None)
    if match is None:
        return 0, []
    if match[0] not in set(comp):
        return 0, []
    return 1, [(int(strokes[match[0]]["index"]), int(match[1]))]


def is_closed_stroke_component_by_endpoint_proximity(strokes, comp, endpoint_tol=12.0):
    """
    Closed-loop test without endpoint-node merging.

    A component is a closed loop when every real stroke endpoint has exactly one
    endpoint-proximity connection inside the component.  Endpoints with zero
    connections are open; endpoints with multiple connections are branches.
    """
    if not comp:
        return False
    if len(comp) == 1:
        s = strokes[comp[0]]
        return endpoint_points_close(s["points"][0], s["points"][-1], endpoint_tol)

    for stroke_local_i in comp:
        for endpoint_i in (0, 1):
            degree, _matches = endpoint_connection_degree_in_component(
                strokes,
                comp,
                stroke_local_i,
                endpoint_i,
                endpoint_tol=endpoint_tol,
            )
            if degree != 1:
                return False
    return True


def prune_open_branches_from_component(strokes, comp, endpoint_tol=12.0):
    """
    Remove dangling branches from a grown endpoint-proximity component.

    The component is allowed to grow through all nearby endpoints first.  Then any
    stroke with an unmatched endpoint is removed; this is repeated so a branch is
    peeled back from its open tip until only closed-loop strokes remain.
    """
    remaining = set(int(i) for i in comp)
    removed = []

    changed = True
    while changed and remaining:
        changed = False
        current = sorted(remaining)
        to_remove = []

        for stroke_local_i in current:
            endpoint_degrees = []
            for endpoint_i in (0, 1):
                degree, _matches = endpoint_connection_degree_in_component(
                    strokes,
                    current,
                    stroke_local_i,
                    endpoint_i,
                    endpoint_tol=endpoint_tol,
                )
                endpoint_degrees.append(degree)

            if any(degree == 0 for degree in endpoint_degrees):
                to_remove.append(stroke_local_i)

        if to_remove:
            changed = True
            for stroke_local_i in to_remove:
                if stroke_local_i in remaining:
                    remaining.remove(stroke_local_i)
                    removed.append(stroke_local_i)

    return sorted(remaining), removed


def self_loop_positions_in_component(strokes, comp, endpoint_tol=12.0):
    """
    Return stroke positions inside comp whose two endpoints match the same external endpoint.

    This is applied only after a loop candidate has already been formed, so the
    goal is to remove obvious one-stroke self-loops without altering component
    growth or open-branch pruning.
    """
    if not comp:
        return []
    comp = list(comp)
    comp_strokes = [strokes[i] for i in comp]
    matches = build_nearest_endpoint_matches(comp_strokes, endpoint_tol=endpoint_tol)
    out = []
    for pos in range(len(comp_strokes)):
        m0 = matches.get((int(pos), 0), None)
        m1 = matches.get((int(pos), 1), None)
        if m0 is None or m1 is None:
            continue
        if m0[0] == pos or m1[0] == pos:
            continue
        if m0 == m1:
            out.append(int(pos))
    return out


def remove_post_loop_self_loops(strokes, comp, endpoint_tol=12.0):
    """
    Remove self-loop strokes from an already closed-loop component, then re-check closure.
    """
    current = list(comp)
    removed = []

    changed = True
    while changed and current:
        changed = False
        positions = self_loop_positions_in_component(strokes, current, endpoint_tol=endpoint_tol)
        if not positions:
            break
        changed = True
        for pos in sorted(positions, reverse=True):
            removed.append(int(current[pos]))
            del current[pos]

    return current, removed


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
    enclosed_area, enclosed_mask = estimate_enclosed_area_from_loop_mask(mask)
    pts = np.vstack([s["points"] for s in loop_strokes])
    center = pts.mean(axis=0).astype(np.float64)
    area = int(np.count_nonzero(mask))
    total_arc = float(sum(s["arc"] for s in loop_strokes))
    return {
        "mask": mask,
        "enclosed_mask": enclosed_mask,
        "area": area,
        "enclosed_area": int(enclosed_area),
        "endpoints": 0,
        "closedness": 1.0,
        "score": float(enclosed_area + area + total_arc),
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

    max_subset_size means the maximum number of strokes allowed in a candidate
    loop, regardless of the total size of the containing connected component.
    """
    import itertools

    component = list(component)
    if not component:
        return []

    loops = []
    max_r = min(int(max_subset_size), len(component))
    if max_r <= 0:
        return []

    for r in range(max_r, 0, -1):
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
    min_enclosed_area=0,
    min_total_arc=0.0,
    thickness=2,
    max_loop_subset_size=14,
):
    """
    Extract cap candidates from non-side connected components that are closed loops.

    Current rule:
      - The caller passes the cap/search pool, normally the same strokes used
        as direction-group candidates.
      - Remove the side strokes from that pool to get non-side strokes.
      - Starting from each remaining stroke, grow a connected component by
        direct real-endpoint proximity: any stroke endpoint within endpoint_tol
        is connected.
      - Before closed-loop validation, prune dangling open branches from the
        grown component by repeatedly removing strokes with degree-0 endpoints.
      - Any stroke whose two endpoints collapse into the same endpoint node is
        treated as a single-stroke self-loop and removed from cap membership.
      - Accept a component as a cap only when every real stroke endpoint has
        exactly one endpoint-proximity connection inside that component.
    """
    _ = max_loop_subset_size  # Kept only for CLI/API compatibility.
    side_ids = {int(s["index"]) for s in side_inliers}
    non_side_infos = [s for s in infos if int(s["index"]) not in side_ids]
    if not non_side_infos:
        return []

    comps = connected_components_by_endpoint_proximity(non_side_infos, endpoint_tol=endpoint_tol)

    candidates = []
    seen = set()

    for comp in comps:
        pruned_comp, removed_branch = prune_open_branches_from_component(
            non_side_infos,
            comp,
            endpoint_tol=endpoint_tol,
        )
        if not pruned_comp:
            continue
        if not is_closed_stroke_component_by_endpoint_proximity(non_side_infos, pruned_comp, endpoint_tol=endpoint_tol):
            continue
        post_loop_comp, removed_self_loops = remove_post_loop_self_loops(
            non_side_infos,
            pruned_comp,
            endpoint_tol=endpoint_tol,
        )
        if not post_loop_comp:
            continue
        if removed_self_loops:
            if not is_closed_stroke_component_by_endpoint_proximity(
                non_side_infos,
                post_loop_comp,
                endpoint_tol=endpoint_tol,
            ):
                continue
        final_comp = list(map(int, post_loop_comp))

        key = tuple(sorted(int(non_side_infos[i]["index"]) for i in final_comp))
        if key in seen:
            continue
        seen.add(key)

        cand = candidate_from_stroke_loop(image_shape, non_side_infos, final_comp, thickness=thickness)
        if cand["area"] < min_pixels:
            continue
        if cand.get("enclosed_area", 0) < min_enclosed_area:
            continue
        if cand.get("total_arc", 0.0) < min_total_arc:
            continue
        if removed_self_loops:
            cand["loop_detection"] = "component_postloop_self_loop_removed"
        else:
            cand["loop_detection"] = "component_pruned_open_branches" if removed_branch else "component"
        cand["component_local_indices"] = list(map(int, final_comp))
        cand["pruned_branch_strokes"] = [int(non_side_infos[i]["index"]) for i in removed_branch]
        cand["removed_post_loop_self_strokes"] = [int(non_side_infos[i]["index"]) for i in removed_self_loops]
        candidates.append(cand)

    candidates.sort(key=lambda c: (c.get("enclosed_area", 0), c["score"]), reverse=True)
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


def init_cap_validation_details(entry, rank):
    """Initialize cap-validation debug fields for a direction group entry."""
    details = entry.setdefault("details", {})
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
    details["best_cap_enclosed_area"] = 0
    details["best_cap_center"] = None
    details["best_cap_total_arc"] = 0.0
    details["invalid_no_cap"] = False
    details["selected_by_cap_validation"] = False
    details["skip_percluster_output"] = False
    details["subset_of_higher_rank_cap_cluster"] = False
    details["subset_parent_rank"] = None
    details["subset_parent_cluster_id"] = None
    details["is_remove_one_subgroup"] = bool(entry.get("is_remove_one_subgroup", False))
    details["is_removed_subgroup"] = bool(entry.get("is_removed_subgroup", False))
    details["parent_cluster_id"] = entry.get("parent_cluster_id", None)
    details["removed_stroke_index"] = entry.get("removed_stroke_index", None)
    details["removed_stroke_indices"] = entry.get("removed_stroke_indices", [])
    details["removal_depth"] = int(entry.get("removal_depth", 0))
    return details


def fill_best_cap_details(details, best_cap, candidate_count):
    """Store a cap candidate summary in an entry details dict."""
    details["cap_candidate_count"] = int(candidate_count)
    if best_cap is None:
        details["invalid_no_cap"] = True
        return

    center = best_cap.get("center", None)
    center_tuple = None if center is None else (float(center[0]), float(center[1]))
    details["cap_candidate_strokes"] = [best_cap.get("stroke_indices", [])]
    details["cap_candidate_scores"] = [float(best_cap.get("score", 0.0))]
    details["cap_candidate_areas"] = [int(best_cap.get("area", 0))]
    details["best_cap_strokes"] = best_cap.get("stroke_indices", [])
    details["best_cap_score"] = float(best_cap.get("score", 0.0))
    details["best_cap_area"] = int(best_cap.get("area", 0))
    details["best_cap_enclosed_area"] = int(best_cap.get("enclosed_area", 0))
    details["best_cap_center"] = center_tuple
    details["best_cap_total_arc"] = float(best_cap.get("total_arc", 0.0))


def make_removed_subgroup_entry(parent_entry, removed_strokes, subgroup_strokes, cluster_id):
    """Create a direction-group subgroup by removing one or more strokes."""
    removed_strokes = list(removed_strokes)
    removal_depth = len(removed_strokes)
    removed_indices = [int(s["index"]) for s in removed_strokes]
    direction = mean_direction(subgroup_strokes)
    score, details = score_side_cluster(subgroup_strokes, direction)
    details["n"] = len(subgroup_strokes)
    details["is_remove_one_subgroup"] = removal_depth == 1
    details["is_removed_subgroup"] = True
    details["parent_cluster_id"] = int(parent_entry.get("cluster_id", -1))
    details["removed_stroke_index"] = removed_indices[0] if removal_depth == 1 else None
    details["removed_stroke_indices"] = removed_indices
    details["removal_depth"] = int(removal_depth)
    return {
        "cluster_id": int(cluster_id),
        "parent_cluster_id": int(parent_entry.get("cluster_id", -1)),
        "removed_stroke_index": removed_indices[0] if removal_depth == 1 else None,
        "removed_stroke_indices": removed_indices,
        "removal_depth": int(removal_depth),
        "is_remove_one_subgroup": removal_depth == 1,
        "is_removed_subgroup": True,
        "strokes": list(subgroup_strokes),
        "indices": [int(s["index"]) for s in subgroup_strokes],
        "direction": direction,
        "score": float(score),
        "details": details,
        "source": f"direction_component_minus_{removal_depth}",
    }


def compute_best_cap_for_side_entry(
    entry,
    infos,
    image_shape,
    endpoint_tol=12.0,
    min_pixels=40,
    min_enclosed_area=0,
    min_total_arc=0.0,
    thickness=2,
    max_loop_subset_size=14,
    cap_pool_infos=None,
):
    """Try one direction group as side strokes and return its largest-area cap."""
    pool_infos = infos if cap_pool_infos is None else cap_pool_infos
    candidates = extract_cap_loop_candidates_from_strokes(
        image_shape,
        pool_infos,
        entry.get("strokes", []),
        endpoint_tol=endpoint_tol,
        min_pixels=min_pixels,
        min_enclosed_area=min_enclosed_area,
        min_total_arc=min_total_arc,
        thickness=thickness,
        max_loop_subset_size=max_loop_subset_size,
    )
    return largest_area_cap_candidate(candidates), candidates


def validate_side_clusters_by_cap_candidates(
    model,
    infos,
    image_shape,
    endpoint_tol=12.0,
    min_pixels=40,
    min_enclosed_area=0,
    min_total_arc=0.0,
    thickness=2,
    max_loop_subset_size=14,
    max_subgroup_removals=-1,
    cap_pool_infos=None,
):
    """
    Compute cap candidates for every direction group.

    For each direction group:
      - use the whole group as side strokes and compute its best cap;
      - after all full groups are checked, groups with no cap are expanded into
        remove-one-stroke subgroups and checked;
      - groups still without cap are expanded into deeper remove-k-stroke
        subgroups until a valid cap is found or only one stroke remains.

    All original groups and generated subgroups remain in cluster_debug so the
    normal side/cap debug images can be written for each case.
    """
    if model is None:
        return None, []

    pool_infos = infos if cap_pool_infos is None else cap_pool_infos
    cluster_debug = model.get("cluster_debug", None)

    if not cluster_debug:
        candidates = extract_cap_loop_candidates_from_strokes(
            image_shape,
            pool_infos,
            model.get("inliers", []),
            endpoint_tol=endpoint_tol,
            min_pixels=min_pixels,
            min_enclosed_area=min_enclosed_area,
            min_total_arc=min_total_arc,
            thickness=thickness,
            max_loop_subset_size=max_loop_subset_size,
        )
        best_cap = largest_area_cap_candidate(candidates)
        model["cap_validated"] = best_cap is not None
        return model, ([] if best_cap is None else [best_cap])

    original_cluster_debug = list(cluster_debug)
    expanded_cluster_debug = []
    next_cluster_id = max([int(e.get("cluster_id", -1)) for e in original_cluster_debug] + [-1]) + 1
    selected_entry = None
    selected_candidates = []
    successful_cap_clusters = []
    cap_search_trace = []

    def evaluate_entry(entry, rank):
        details = init_cap_validation_details(entry, rank)
        n_strokes = int(details.get("n", len(entry.get("strokes", []))))
        best_cap = None
        if n_strokes < 2:
            details["cap_validation_skipped"] = True
            details["cap_validation_skip_reason"] = "n_lt_2"
            details["cap_validation_selectable"] = False
            trial_candidates = []
        else:
            details["cap_validation_checked"] = True
            best_cap, trial_candidates = compute_best_cap_for_side_entry(
                entry,
                infos,
                image_shape,
                endpoint_tol=endpoint_tol,
                min_pixels=min_pixels,
                min_enclosed_area=min_enclosed_area,
                min_total_arc=min_total_arc,
                thickness=thickness,
                max_loop_subset_size=max_loop_subset_size,
                cap_pool_infos=pool_infos,
            )
            fill_best_cap_details(details, best_cap, len(trial_candidates))

        cap_search_trace.append({
            "order": int(len(cap_search_trace)),
            "rank": int(rank),
            "cluster_id": int(entry.get("cluster_id", -1)),
            "source": entry.get("source", "unknown"),
            "parent_cluster_id": entry.get("parent_cluster_id", None),
            "removal_depth": int(entry.get("removal_depth", 0)),
            "removed_stroke_indices": list(entry.get("removed_stroke_indices", [])),
            "side_indices": [int(s["index"]) for s in entry.get("strokes", [])],
            "n": int(n_strokes),
            "checked": bool(details.get("cap_validation_checked", False)),
            "skipped": bool(details.get("cap_validation_skipped", False)),
            "skip_reason": details.get("cap_validation_skip_reason", ""),
            "cap_found": best_cap is not None,
            "cap_candidate_count": int(details.get("cap_candidate_count", 0)),
            "best_cap_area": int(details.get("best_cap_area", 0)),
            "best_cap_enclosed_area": int(details.get("best_cap_enclosed_area", 0)),
            "best_cap_total_arc": float(details.get("best_cap_total_arc", 0.0)),
            "best_cap_score": float(details.get("best_cap_score", 0.0)),
            "best_cap_strokes": list(details.get("best_cap_strokes", [])),
        })
        return best_cap

    def record_success(entry, rank, best_cap, parent_cluster_id=None, removed_stroke_indices=None):
        nonlocal selected_entry, selected_candidates
        side_set = cluster_entry_index_set(entry)
        removed_stroke_indices = [] if removed_stroke_indices is None else list(removed_stroke_indices)
        successful_cap_clusters.append({
            "rank": int(rank),
            "cluster_id": int(entry.get("cluster_id", -1)),
            "side_set": set(side_set),
            "parent_cluster_id": parent_cluster_id,
            "removed_stroke_index": removed_stroke_indices[0] if len(removed_stroke_indices) == 1 else None,
            "removed_stroke_indices": removed_stroke_indices,
            "removal_depth": int(len(removed_stroke_indices)),
        })

        if selected_entry is None:
            selected_entry = entry
            selected_candidates = [best_cap]
            entry["details"]["selected_by_cap_validation"] = True

    failed_original_groups = []

    # Level 0: check every full direction group first.
    for entry in original_cluster_debug:
        rank = len(expanded_cluster_debug)
        expanded_cluster_debug.append(entry)
        best_cap = evaluate_entry(entry, rank)

        if best_cap is None:
            strokes = list(entry.get("strokes", []))
            if len(strokes) > 1:
                failed_original_groups.append(entry)
            continue

        record_success(entry, rank, best_cap)

    # Levels 1..N: for original groups that still have no cap, check all
    # remove-k subgroups before moving to remove-(k+1).
    import itertools

    max_possible_removals = 0
    if failed_original_groups:
        max_possible_removals = max(len(entry.get("strokes", [])) - 1 for entry in failed_original_groups)
    if max_subgroup_removals is None or int(max_subgroup_removals) < 0:
        max_subgroup_removals = max_possible_removals
    else:
        max_subgroup_removals = min(max_possible_removals, max(0, int(max_subgroup_removals)))

    for removal_depth in range(1, max_subgroup_removals + 1):
        if not failed_original_groups:
            break

        still_failed = []
        for parent_entry in failed_original_groups:
            parent_strokes = list(parent_entry.get("strokes", []))
            if len(parent_strokes) - removal_depth < 1:
                continue

            found_for_parent = False
            for removed_positions in itertools.combinations(range(len(parent_strokes)), removal_depth):
                removed_positions = set(removed_positions)
                removed_strokes = [s for i, s in enumerate(parent_strokes) if i in removed_positions]
                subgroup_strokes = [s for i, s in enumerate(parent_strokes) if i not in removed_positions]

                subgroup = make_removed_subgroup_entry(
                    parent_entry,
                    removed_strokes,
                    subgroup_strokes,
                    cluster_id=next_cluster_id,
                )
                next_cluster_id += 1

                subgroup_rank = len(expanded_cluster_debug)
                expanded_cluster_debug.append(subgroup)
                subgroup_cap = evaluate_entry(subgroup, subgroup_rank)
                if subgroup_cap is None:
                    continue

                found_for_parent = True
                record_success(
                    subgroup,
                    subgroup_rank,
                    subgroup_cap,
                    parent_cluster_id=int(parent_entry.get("cluster_id", -1)),
                    removed_stroke_indices=[int(s["index"]) for s in removed_strokes],
                )
                break

            if not found_for_parent:
                still_failed.append(parent_entry)

        failed_original_groups = still_failed

    model["cluster_debug"] = expanded_cluster_debug
    model["cap_search_trace"] = cap_search_trace
    model["successful_cap_clusters"] = [
        {
            "rank": int(x["rank"]),
            "cluster_id": int(x["cluster_id"]),
            "side_indices": sorted(list(x["side_set"])),
            "parent_cluster_id": x.get("parent_cluster_id", None),
            "removed_stroke_index": x.get("removed_stroke_index", None),
            "removed_stroke_indices": x.get("removed_stroke_indices", []),
            "removal_depth": x.get("removal_depth", 0),
        }
        for x in successful_cap_clusters
    ]

    if selected_entry is None:
        model["cap_validated"] = False
        model["cap_validation_failed"] = True
        model["cap_validation_message"] = "No ranked side cluster produced a legal closed-loop cap candidate."
        model["cap_search_trace"] = cap_search_trace
        return model, []

    validated_model = make_trial_model_from_cluster_entry(
        model,
        selected_entry,
        cluster_debug=expanded_cluster_debug,
        cap_validated=True,
    )
    validated_model["cap_validation_failed"] = False
    validated_model["cap_validation_message"] = (
        "Computed best cap for every full direction group first.  For groups with no legal cap, "
        "computed remove-k-stroke subgroups level by level and selected the first entry that produced a legal cap."
    )
    validated_model["successful_cap_clusters"] = model.get("successful_cap_clusters", [])
    validated_model["cap_search_trace"] = cap_search_trace
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


def trace_branch_path_from_endpoint(skel, start, stop_nodes=None):
    """
    Trace one dangling path from an endpoint until the first branch/endpoint.

    Returns a list of pixel coordinates including start and the terminal node.
    """
    stop_nodes = set() if stop_nodes is None else set(stop_nodes)
    path = [start]
    prev = None
    cur = start
    safety = 0

    while True:
        safety += 1
        if safety > 20000:
            break

        nbs = [q for q in get_neighbors(skel, cur) if q != prev]
        if not nbs:
            break

        if len(path) > 1 and skeleton_node_type(skel, cur) in ("endpoint", "branch"):
            break

        nxt = choose_best_continuation(prev if prev is not None else cur, cur, nbs)
        if nxt is None:
            break

        prev, cur = cur, nxt
        path.append(cur)

        if cur in stop_nodes:
            break
        if skeleton_node_type(skel, cur) in ("endpoint", "branch"):
            break

    return path


def endpoint_pairs_mutual_nearest_within_tol(endpoints, tol):
    """Pair endpoints by mutual nearest-neighbor within tol."""
    if len(endpoints) < 2:
        return [], list(range(len(endpoints)))

    pts = [np.asarray(p, dtype=np.float64) for p in endpoints]
    tol2 = float(tol) * float(tol)
    nearest = {}

    for i, p in enumerate(pts):
        best_j = None
        best_d2 = float("inf")
        for j, q in enumerate(pts):
            if i == j:
                continue
            d2 = float(np.dot(p - q, p - q))
            if d2 > tol2:
                continue
            if d2 < best_d2:
                best_d2 = d2
                best_j = j
        if best_j is not None:
            nearest[i] = (best_j, best_d2)

    pairs = []
    used = set()
    for i, (j, d2) in nearest.items():
        if i in used or j in used:
            continue
        other = nearest.get(j, None)
        if other is None or other[0] != i:
            continue
        a, b = sorted((i, j))
        pairs.append((a, b, math.sqrt(d2)))
        used.add(a)
        used.add(b)

    unpaired = [i for i in range(len(endpoints)) if i not in used]
    return pairs, unpaired


def rasterize_connection_line(shape, p0, p1, thickness=1):
    """Rasterize one gap-connection line into its own mask."""
    mask = np.zeros(shape[:2], dtype=np.uint8)
    cv2.line(
        mask,
        tuple(map(int, p0)),
        tuple(map(int, p1)),
        255,
        int(max(1, thickness)),
        cv2.LINE_AA,
    )
    return mask


def shortest_skeleton_path(mask, start, goal):
    """Return a shortest pixel path between two skeleton pixels, or None."""
    start = tuple(map(int, start))
    goal = tuple(map(int, goal))
    h, w = mask.shape[:2]
    if not (0 <= start[0] < w and 0 <= start[1] < h):
        return None
    if not (0 <= goal[0] < w and 0 <= goal[1] < h):
        return None
    if mask[start[1], start[0]] == 0 or mask[goal[1], goal[0]] == 0:
        return None

    queue = [start]
    head = 0
    parent = {start: None}
    while head < len(queue):
        cur = queue[head]
        head += 1
        if cur == goal:
            break
        for nb in get_neighbors(mask, cur):
            if nb in parent:
                continue
            parent[nb] = cur
            queue.append(nb)

    if goal not in parent:
        return None

    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path


def bbox_for_points(points):
    """Return bbox metadata for a non-empty list of (x, y) points."""
    if not points:
        return None
    xs = [int(p[0]) for p in points]
    ys = [int(p[1]) for p in points]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    return {
        "bbox": (int(x0), int(y0), int(x1), int(y1)),
        "bbox_area": int((x1 - x0 + 1) * (y1 - y0 + 1)),
    }


def prune_added_connection_small_loops(original_skel, connected_skel, connections, bbox_area_thresh=0.0, connect_thickness=1):
    """
    Remove only newly drawn gap-connection pixels when that connection closes a small loop.

    Each added connection is tested by temporarily removing just its new pixels and
    checking whether its endpoints are still connected through the existing graph.
    If so, that alternate path plus the added connection is a loop; small loops are
    filtered by bbox area instead of exact filled area.
    """
    current = (connected_skel > 0).astype(np.uint8) * 255
    original = (original_skel > 0).astype(np.uint8) * 255
    thresh = float(bbox_area_thresh or 0.0)
    records = []

    for ci, item in enumerate(connections):
        p0 = tuple(map(int, item["p0"]))
        p1 = tuple(map(int, item["p1"]))
        line_mask = rasterize_connection_line(current.shape, p0, p1, thickness=connect_thickness)
        added_mask = ((line_mask > 0) & (original == 0)).astype(np.uint8) * 255
        active_added = (added_mask > 0) & (current > 0)
        ys, xs = np.where(active_added)
        added_points = list(zip(xs.tolist(), ys.tolist()))

        record = {
            "connection_index": int(ci),
            "p0": p0,
            "p1": p1,
            "added_pixel_count": int(len(added_points)),
            "forms_loop": False,
            "removed": False,
            "skip_reason": "",
            "bbox": None,
            "bbox_area": 0,
            "alternate_path_pixels": 0,
        }

        if thresh <= 0.0:
            record["skip_reason"] = "threshold_disabled"
            records.append(record)
            continue
        if not added_points:
            record["skip_reason"] = "no_new_pixels"
            records.append(record)
            continue

        trial = current.copy()
        trial[active_added] = 0
        path = shortest_skeleton_path(trial, p0, p1)
        if path is None:
            record["skip_reason"] = "no_alternate_path"
            records.append(record)
            continue

        loop_points = path + added_points
        bbox = bbox_for_points(loop_points)
        record["forms_loop"] = True
        record["alternate_path_pixels"] = int(len(path))
        if bbox is not None:
            record["bbox"] = bbox["bbox"]
            record["bbox_area"] = int(bbox["bbox_area"])

        if record["bbox_area"] < thresh:
            current[active_added] = 0
            record["removed"] = True
            record["skip_reason"] = ""
        else:
            record["skip_reason"] = "bbox_area_ge_threshold"
        records.append(record)

    current = remove_small_components(current, min_area=1)
    return current, records


def skeleton_branch_stop_points(skel):
    """Return branch points from a fixed skeleton snapshot."""
    stops = set()
    ys, xs = np.where(skel > 0)
    for x, y in zip(xs, ys):
        p = (int(x), int(y))
        if skeleton_node_type(skel, p) == "branch":
            stops.add(p)
    return stops


def prune_dangling_branches_from_endpoints(skel, endpoints, max_pixels=None, stop_nodes=None):
    """Remove endpoint-started dangling paths from the current skeleton."""
    cleaned = (skel > 0).astype(np.uint8) * 255
    removed_paths = []
    skipped_paths = []
    stop_nodes = set() if stop_nodes is None else set(stop_nodes)
    max_pixels = None if max_pixels is None or float(max_pixels) <= 0.0 else float(max_pixels)

    for p in endpoints:
        p = tuple(map(int, p))
        if not (0 <= p[0] < cleaned.shape[1] and 0 <= p[1] < cleaned.shape[0]):
            continue
        if cleaned[p[1], p[0]] == 0:
            continue
        if skeleton_node_type(cleaned, p) != "endpoint":
            continue
        path = trace_branch_path_from_endpoint(cleaned, p, stop_nodes=stop_nodes)
        if not path:
            continue
        if max_pixels is not None and len(path) > max_pixels:
            skipped_paths.append({
                "path": [tuple(map(int, q)) for q in path],
                "reason": "longer_than_max_pixels",
                "max_pixels": float(max_pixels),
            })
            continue
        removed_paths.append(path)
        for x, y in path[:-1]:
            cleaned[y, x] = 0
        end_x, end_y = path[-1]
        if (end_x, end_y) not in stop_nodes and skeleton_node_type(cleaned, (end_x, end_y)) == "endpoint":
            cleaned[end_y, end_x] = 0

    cleaned = remove_small_components(cleaned, min_area=1)
    return cleaned, removed_paths, skipped_paths


def cleanup_skeleton_endpoints(
    skel,
    gap_tol=0.0,
    connect_thickness=1,
    small_loop_bbox_area_thresh=0.0,
    branch_prune_max_pixels=0.0,
):
    """
    Connect mutual-nearest endpoint pairs within gap_tol, then drop dead branches.

    Returns cleaned skeleton plus debug metadata.
    """
    work = (skel > 0).astype(np.uint8) * 255
    endpoints_before = skeleton_endpoints_for_closure(work)

    if gap_tol is None or float(gap_tol) <= 0.0 or len(endpoints_before) < 2:
        return work, work.copy(), work.copy(), {
            "endpoint_count_before": int(len(endpoints_before)),
            "endpoint_count_after_connect": int(len(endpoints_before)),
            "endpoint_count_after_small_loop_prune": int(len(endpoints_before)),
            "endpoint_count_final": int(len(endpoints_before)),
            "gap_tol": float(gap_tol or 0.0),
            "small_loop_bbox_area_thresh": float(small_loop_bbox_area_thresh or 0.0),
            "connections": [],
            "small_loop_added_edge_candidates": [],
            "removed_branches": [],
            "skipped_branches": [],
            "removed_endpoint_count": 0,
            "branch_prune_endpoint_count": 0,
            "branch_prune_endpoint_source": "not_run",
            "branch_prune_max_pixels": 0.0,
        }

    pairs, unpaired = endpoint_pairs_mutual_nearest_within_tol(endpoints_before, gap_tol)
    connected = work.copy()
    connections = []
    for i, j, _dist in pairs:
        p0 = tuple(map(int, endpoints_before[i]))
        p1 = tuple(map(int, endpoints_before[j]))
        cv2.line(
            connected,
            p0,
            p1,
            255,
            int(max(1, connect_thickness)),
            cv2.LINE_AA,
        )
        connections.append({
            "connection_index": int(len(connections)),
            "p0": p0,
            "p1": p1,
            "dist": float(_dist),
        })

    endpoints_after_connect = skeleton_endpoints_for_closure(connected)
    small_loop_pruned, small_loop_records = prune_added_connection_small_loops(
        work,
        connected,
        connections,
        bbox_area_thresh=small_loop_bbox_area_thresh,
        connect_thickness=connect_thickness,
    )
    endpoints_after_small_loop_prune = skeleton_endpoints_for_closure(small_loop_pruned)

    if float(small_loop_bbox_area_thresh or 0.0) > 0.0:
        branch_prune_points = endpoints_after_small_loop_prune
        branch_prune_source = "all_02c2_endpoints_after_small_loop_prune"
    else:
        branch_prune_points = [endpoints_before[i] for i in unpaired]
        branch_prune_source = "unpaired_original_endpoints"

    effective_branch_prune_max_pixels = None
    if branch_prune_max_pixels is not None and float(branch_prune_max_pixels) > 0.0:
        effective_branch_prune_max_pixels = float(branch_prune_max_pixels)
    elif float(small_loop_bbox_area_thresh or 0.0) > 0.0:
        effective_branch_prune_max_pixels = float(max(30.0, 3.0 * float(gap_tol)))

    branch_stop_nodes = skeleton_branch_stop_points(small_loop_pruned)
    cleaned, removed_paths, skipped_paths = prune_dangling_branches_from_endpoints(
        small_loop_pruned,
        branch_prune_points,
        max_pixels=effective_branch_prune_max_pixels,
        stop_nodes=branch_stop_nodes,
    )
    endpoints_final = skeleton_endpoints_for_closure(cleaned)

    return connected, small_loop_pruned, cleaned, {
        "endpoint_count_before": int(len(endpoints_before)),
        "endpoint_count_after_connect": int(len(endpoints_after_connect)),
        "endpoint_count_after_small_loop_prune": int(len(endpoints_after_small_loop_prune)),
        "endpoint_count_final": int(len(endpoints_final)),
        "endpoints_before": [tuple(map(int, p)) for p in endpoints_before],
        "endpoints_after_connect": [tuple(map(int, p)) for p in endpoints_after_connect],
        "endpoints_after_small_loop_prune": [tuple(map(int, p)) for p in endpoints_after_small_loop_prune],
        "endpoints_final": [tuple(map(int, p)) for p in endpoints_final],
        "gap_tol": float(gap_tol),
        "small_loop_bbox_area_thresh": float(small_loop_bbox_area_thresh or 0.0),
        "connections": connections,
        "small_loop_added_edge_candidates": small_loop_records,
        "removed_branches": [
            [tuple(map(int, q)) for q in path]
            for path in removed_paths
        ],
        "skipped_branches": skipped_paths,
        "removed_endpoint_count": int(len(branch_prune_points)),
        "branch_prune_endpoint_count": int(len(branch_prune_points)),
        "branch_prune_points": [tuple(map(int, p)) for p in branch_prune_points],
        "branch_prune_endpoint_source": branch_prune_source,
        "branch_prune_stop_nodes": sorted([tuple(map(int, p)) for p in branch_stop_nodes]),
        "branch_prune_stop_node_count": int(len(branch_stop_nodes)),
        "branch_prune_max_pixels": (
            0.0 if effective_branch_prune_max_pixels is None else float(effective_branch_prune_max_pixels)
        ),
    }


def draw_skeleton_cleanup_debug(shape, skel_before, skel_after_connect, skel_after_prune, cleanup_info):
    """Visualize skeleton endpoint connection and dangling-branch removal."""
    h, w = shape[:2]
    out = np.full((h, w, 3), 255, dtype=np.uint8)
    out[skel_after_prune > 0] = (0, 0, 0)

    removed_connection_indices = {
        int(item.get("connection_index", -1))
        for item in cleanup_info.get("small_loop_added_edge_candidates", [])
        if item.get("removed", False)
    }
    for item in cleanup_info.get("connections", []):
        ci = int(item.get("connection_index", len(removed_connection_indices)))
        p0 = tuple(item["p0"])
        p1 = tuple(item["p1"])
        color = (180, 0, 180) if ci in removed_connection_indices else (0, 170, 0)
        cv2.line(out, p0, p1, color, 1, cv2.LINE_AA)
        cv2.circle(out, p0, 3, color, -1, cv2.LINE_AA)
        cv2.circle(out, p1, 3, color, -1, cv2.LINE_AA)

    for item in cleanup_info.get("small_loop_added_edge_candidates", []):
        if not item.get("forms_loop", False):
            continue
        bbox = item.get("bbox", None)
        if bbox is None:
            continue
        x0, y0, x1, y1 = map(int, bbox)
        color = (180, 0, 180) if item.get("removed", False) else (0, 180, 180)
        cv2.rectangle(out, (x0, y0), (x1, y1), color, 1, cv2.LINE_AA)
        cv2.putText(
            out,
            f"a={int(item.get('bbox_area', 0))}",
            (x0, max(12, y0 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            color,
            1,
            cv2.LINE_AA,
        )

    for path in cleanup_info.get("removed_branches", []):
        if len(path) < 2:
            continue
        pts = np.asarray(path, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], False, (0, 140, 255), 2, cv2.LINE_AA)

    for item in cleanup_info.get("skipped_branches", []):
        path = item.get("path", [])
        if len(path) < 2:
            continue
        pts = np.asarray(path, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], False, (180, 180, 0), 2, cv2.LINE_AA)

    cv2.putText(
        out,
        f"endpoint cleanup: tol={cleanup_info.get('gap_tol', 0.0):.1f}, "
        f"connected={len(cleanup_info.get('connections', []))}, "
        f"small_loop_links={len(removed_connection_indices)}, "
        f"branches={len(cleanup_info.get('removed_branches', []))}",
        (15, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        out,
        "green=kept endpoint links, purple=small-loop links removed, orange=removed dangling branches, cyan=skipped long branches",
        (15, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    return out


def save_skeleton_cleanup_debug_outputs(debug_dir, skel_before, skel_after_connect, skel_after_small_loop_prune, skel_after_prune, cleanup_info):
    """Save debug images/text for skeleton endpoint cleanup."""
    if debug_dir is None:
        return
    cv2.imwrite(os.path.join(debug_dir, "02c_skeleton_after_gap_connect.png"), skel_after_connect)
    cv2.imwrite(os.path.join(debug_dir, "02c1_skeleton_after_gap_connect_nodes.png"), draw_skeleton_nodes_debug(skel_after_connect))
    cv2.imwrite(os.path.join(debug_dir, "02c2_skeleton_after_small_loop_prune.png"), skel_after_small_loop_prune)
    cv2.imwrite(os.path.join(debug_dir, "02c3_skeleton_after_small_loop_prune_nodes.png"), draw_skeleton_nodes_debug(skel_after_small_loop_prune))
    cv2.imwrite(os.path.join(debug_dir, "02d_skeleton_after_branch_prune.png"), skel_after_prune)
    cv2.imwrite(os.path.join(debug_dir, "02d1_skeleton_after_branch_prune_nodes.png"), draw_skeleton_nodes_debug(skel_after_prune))
    cv2.imwrite(
        os.path.join(debug_dir, "02d2_skeleton_endpoint_cleanup_overlay.png"),
        draw_skeleton_cleanup_debug(skel_before.shape, skel_before, skel_after_connect, skel_after_prune, cleanup_info),
    )
    with open(os.path.join(debug_dir, "02d_skeleton_endpoint_cleanup.json"), "w", encoding="utf-8") as jf:
        json.dump(cleanup_info, jf, indent=2)
    with open(os.path.join(debug_dir, "02d_skeleton_endpoint_cleanup.txt"), "w", encoding="utf-8") as f:
        f.write("==== Skeleton Endpoint Cleanup ====\n\n")
        f.write(f"gap_tol: {float(cleanup_info.get('gap_tol', 0.0)):.1f}\n")
        f.write(f"small_loop_bbox_area_thresh: {float(cleanup_info.get('small_loop_bbox_area_thresh', 0.0)):.1f}\n")
        f.write(f"branch_prune_max_pixels: {float(cleanup_info.get('branch_prune_max_pixels', 0.0)):.1f}\n")
        f.write(f"endpoint_count_before: {int(cleanup_info.get('endpoint_count_before', 0))}\n")
        f.write(f"endpoint_count_after_connect: {int(cleanup_info.get('endpoint_count_after_connect', 0))}\n")
        f.write(f"endpoint_count_after_small_loop_prune: {int(cleanup_info.get('endpoint_count_after_small_loop_prune', 0))}\n")
        f.write(f"endpoint_count_final: {int(cleanup_info.get('endpoint_count_final', 0))}\n")
        f.write(f"connections: {len(cleanup_info.get('connections', []))}\n")
        for i, item in enumerate(cleanup_info.get("connections", [])):
            f.write(
                f"  connect {i:03d}: p0={item['p0']} p1={item['p1']} dist={float(item['dist']):.2f}\n"
            )
        f.write(f"endpoints_on_02c2_after_small_loop_prune: {len(cleanup_info.get('endpoints_after_small_loop_prune', []))}\n")
        for i, p in enumerate(cleanup_info.get("endpoints_after_small_loop_prune", [])):
            f.write(f"  02c2 endpoint {i:03d}: {p}\n")
        records = cleanup_info.get("small_loop_added_edge_candidates", [])
        f.write(f"small_loop_added_edge_candidates: {len(records)}\n")
        for item in records:
            f.write(
                f"  edge {int(item.get('connection_index', -1)):03d}: "
                f"p0={item.get('p0')} p1={item.get('p1')} "
                f"added_pixels={int(item.get('added_pixel_count', 0))} "
                f"forms_loop={bool(item.get('forms_loop', False))} "
                f"bbox={item.get('bbox')} bbox_area={int(item.get('bbox_area', 0))} "
                f"alternate_path_pixels={int(item.get('alternate_path_pixels', 0))} "
                f"removed={bool(item.get('removed', False))} "
                f"reason={item.get('skip_reason', '')}\n"
            )
        f.write(f"branch_prune_endpoint_source: {cleanup_info.get('branch_prune_endpoint_source', '')}\n")
        f.write(f"branch_prune_endpoint_count: {int(cleanup_info.get('branch_prune_endpoint_count', 0))}\n")
        f.write(f"branch_prune_stop_node_count: {int(cleanup_info.get('branch_prune_stop_node_count', 0))}\n")
        for i, p in enumerate(cleanup_info.get("branch_prune_points", [])):
            f.write(f"  branch prune endpoint {i:03d}: {p}\n")
        for i, p in enumerate(cleanup_info.get("branch_prune_stop_nodes", [])):
            f.write(f"  frozen 02c2 branch stop {i:03d}: {p}\n")
        f.write(f"removed_branches: {len(cleanup_info.get('removed_branches', []))}\n")
        for i, path in enumerate(cleanup_info.get("removed_branches", [])):
            start = path[0] if path else None
            end = path[-1] if path else None
            f.write(
                f"  remove {i:03d}: pixels={len(path)} start={start} end={end} path={path}\n"
            )
        f.write(f"skipped_branches: {len(cleanup_info.get('skipped_branches', []))}\n")
        for i, item in enumerate(cleanup_info.get("skipped_branches", [])):
            path = item.get("path", [])
            start = path[0] if path else None
            end = path[-1] if path else None
            f.write(
                f"  skip {i:03d}: pixels={len(path)} start={start} end={end} "
                f"reason={item.get('reason', '')} max_pixels={float(item.get('max_pixels', 0.0)):.1f} path={path}\n"
            )


def write_corner_split_trace_report(path, trace, angle_thresh=None, peak_min_distance=None):
    """Write raw trace stroke to post-corner-split stroke mapping."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("==== Corner Split Trace ====\n\n")
        f.write(f"split_corner_angle: {angle_thresh}\n")
        f.write(f"split_peak_min_distance: {peak_min_distance}\n")
        f.write("Angles use folded unoriented axis angle: min(raw_angle, 180 - raw_angle).\n")
        f.write("output_indices are stroke ids after corner splitting, before optional merge.\n\n")

        if not trace:
            f.write("No traced strokes.\n")
            return

        for item in trace:
            f.write(
                f"input {item.get('input_index', -1):03d}: "
                f"input_len={item.get('input_len', 0)}, "
                f"split={item.get('split', False)}, "
                f"outputs={item.get('output_indices', [])}, "
                f"output_lengths={item.get('output_lengths', [])}\n"
            )
            for event in item.get("split_events", []):
                p = event.get("point", (0.0, 0.0))
                f.write(
                    f"  split_at_index={event.get('index', -1)}, "
                    f"point=({p[0]:.1f},{p[1]:.1f}), "
                    f"raw_angle={event.get('raw_angle', 0.0):.2f}, "
                    f"folded_angle={event.get('folded_angle', 0.0):.2f}, "
                    f"segment_angle={event.get('segment_angle', 0.0):.2f}, "
                    f"segment_left_len={event.get('segment_left_len', 0)}, "
                    f"segment_right_len={event.get('segment_right_len', 0)}\n"
                )


def write_corner_split_candidates_report(path, trace, angle_thresh=None, peak_min_distance=None):
    """Write accepted and rejected corner split candidates."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("==== Corner Split Candidates ====\n\n")
        f.write(f"split_corner_angle: {angle_thresh}\n")
        f.write(f"split_peak_min_distance: {peak_min_distance}\n")
        f.write("Candidates are local maxima among adjacent points whose PCA segment_angle >= split_corner_angle.\n")
        f.write("Multiple candidates inside one high-score run may be accepted when they are far enough apart.\n")
        f.write("folded_angle is the immediate-neighbor local angle and is reported only for debugging.\n\n")

        any_candidate = False
        for item in trace:
            candidates = item.get("candidate_events", [])
            if not candidates:
                continue
            any_candidate = True
            f.write(
                f"input {item.get('input_index', -1):03d}: "
                f"input_len={item.get('input_len', 0)}, "
                f"outputs={item.get('output_indices', [])}\n"
            )
            for event in candidates:
                p = event.get("point", (0.0, 0.0))
                status = "ACCEPT" if event.get("accepted", False) else "reject"
                reason = event.get("reject_reason", "")
                opt_txt = ""
                if "optimization_pass" in event:
                    opt_txt = (
                        f", opt_pass={event.get('optimization_pass')}, "
                        f"opt_stage={event.get('optimization_stage', '')}, "
                        f"prev={event.get('prev_split_index', None)}, "
                        f"next={event.get('next_split_index', None)}"
                    )
                f.write(
                    f"  {status}: index={event.get('index', -1)}, "
                    f"point=({p[0]:.1f},{p[1]:.1f}), "
                    f"raw_angle={event.get('raw_angle', 0.0):.2f}, "
                    f"folded_angle={event.get('folded_angle', 0.0):.2f}, "
                    f"segment_angle={event.get('segment_angle', 0.0):.2f}, "
                    f"segment_left_len={event.get('segment_left_len', 0)}, "
                    f"segment_right_len={event.get('segment_right_len', 0)}, "
                    f"reason={reason}{opt_txt}\n"
                )

        if not any_candidate:
            f.write("No local-max corner candidates met the PCA segment_angle threshold.\n")


def write_corner_split_scan_points_report(path, trace, angle_thresh=None, peak_min_distance=None):
    """Write every point scanned while searching for corner split candidates."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("==== Corner Split Scanned Points ====\n\n")
        f.write(f"split_corner_angle: {angle_thresh}\n")
        f.write(f"split_peak_min_distance: {peak_min_distance}\n")
        f.write("Every listed point is an interior index visited by the corner-split scan; iterative rescans can list the same index in multiple passes.\n")
        f.write("candidate=True means the point survived segment_angle thresholding and local-maximum suppression.\n")
        f.write("folded_angle is an immediate-neighbor local angle for debugging only; segment_angle drives candidate selection.\n\n")

        any_scan = False
        for item in trace:
            scans = item.get("scan_events", [])
            if not scans:
                continue
            any_scan = True
            candidates = [event for event in scans if event.get("candidate", False)]
            accepted = [event for event in scans if event.get("accepted", False)]
            passes = sorted({event.get("optimization_pass", None) for event in scans})
            with_angle = [event for event in scans if event.get("raw_angle", 0.0) or event.get("folded_angle", 0.0)]
            with_segment = [event for event in scans if event.get("segment_angle", 0.0)]
            if with_angle:
                max_event = max(with_angle, key=lambda event: float(event.get("folded_angle", 0.0)))
                max_p = max_event.get("point", (0.0, 0.0))
                max_folded_text = (
                    f"max_folded={max_event.get('folded_angle', 0.0):.2f} "
                    f"at index={max_event.get('index', -1)} "
                    f"point=({max_p[0]:.1f},{max_p[1]:.1f})"
                )
            else:
                max_folded_text = "max_folded=n/a"
            if with_segment:
                max_segment_event = max(with_segment, key=lambda event: float(event.get("segment_angle", 0.0)))
                max_segment_p = max_segment_event.get("point", (0.0, 0.0))
                max_segment_text = (
                    f"max_segment={max_segment_event.get('segment_angle', 0.0):.2f} "
                    f"at index={max_segment_event.get('index', -1)} "
                    f"point=({max_segment_p[0]:.1f},{max_segment_p[1]:.1f})"
                )
            else:
                max_segment_text = "max_segment=n/a"

            f.write(
                f"input {item.get('input_index', -1):03d}: "
                f"input_len={item.get('input_len', 0)}, "
                f"outputs={item.get('output_indices', [])}, "
                f"scanned={len(scans)}, "
                f"passes={passes}, "
                f"candidates={len(candidates)}, "
                f"accepted={len(accepted)}, "
                f"{max_segment_text}, "
                f"{max_folded_text}\n"
            )
            for event in scans:
                p = event.get("point", (0.0, 0.0))
                if event.get("accepted", False):
                    status = "ACCEPT"
                elif event.get("candidate", False):
                    status = "candidate_reject"
                else:
                    status = "scan"
                f.write(
                    f"  {status}: index={event.get('index', -1)}, "
                    f"point=({p[0]:.1f},{p[1]:.1f}), "
                    f"raw_angle={event.get('raw_angle', 0.0):.2f}, "
                    f"folded_angle={event.get('folded_angle', 0.0):.2f}, "
                    f"segment_angle={event.get('segment_angle', 0.0):.2f}, "
                    f"candidate={event.get('candidate', False)}, "
                    f"high_score={event.get('high_score', False)}, "
                    f"local_max={event.get('local_max', False)}, "
                    f"pass={event.get('optimization_pass', None)}, "
                    f"stage={event.get('optimization_stage', '')}, "
                    f"prev={event.get('prev_split_index', None)}, "
                    f"next={event.get('next_split_index', None)}, "
                    f"reason={event.get('reject_reason', '')}\n"
                )
            f.write("\n")

        if not any_scan:
            f.write("No scan points were recorded.\n")


def draw_corner_split_candidates_image(shape, trace):
    """Draw every corner candidate and the left/right segment windows used for PCA angle."""
    h, w = shape[:2]
    out = np.full((h, w, 3), 255, dtype=np.uint8)

    def draw_points_polyline(points, color, thickness=2):
        if not points:
            return
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2).astype(np.int32)
        if len(pts) >= 2:
            cv2.polylines(out, [pts], False, color, thickness, cv2.LINE_AA)
        else:
            p = pts[0, 0]
            cv2.circle(out, (int(p[0]), int(p[1])), 2, color, -1, cv2.LINE_AA)

    y = 24
    legend = [
        ("blue=left segment window for PCA", (255, 90, 0)),
        ("orange=right segment window for PCA", (0, 150, 255)),
        ("green filled=candidate accepted", (0, 170, 0)),
        ("red hollow=candidate rejected", (0, 0, 255)),
    ]
    for text, color in legend:
        cv2.putText(out, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)
        y += 22

    for item in trace:
        input_index = int(item.get("input_index", -1))
        for event in item.get("candidate_events", []):
            left = event.get("segment_left_points", [])
            right = event.get("segment_right_points", [])
            draw_points_polyline(left, (255, 90, 0), thickness=2)
            draw_points_polyline(right, (0, 150, 255), thickness=2)

            p = event.get("point", None)
            if p is None:
                continue
            x, y = int(round(p[0])), int(round(p[1]))
            accepted = bool(event.get("accepted", False))
            if accepted:
                cv2.circle(out, (x, y), 5, (0, 170, 0), -1, cv2.LINE_AA)
                cv2.circle(out, (x, y), 7, (0, 110, 0), 1, cv2.LINE_AA)
            else:
                cv2.circle(out, (x, y), 6, (0, 0, 255), 2, cv2.LINE_AA)

            label = (
                f"in{input_index}:{event.get('index', -1)} "
                f"loc={event.get('folded_angle', 0.0):.1f} "
                f"seg={event.get('segment_angle', 0.0):.1f}"
            )
            cv2.putText(
                out,
                label,
                (x + 7, y - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

    cv2.putText(
        out,
        "Corner split candidates: local maxima of PCA segment_angle plus their left/right PCA windows",
        (15, h - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    return out


def draw_corner_split_scan_points_image(shape, trace):
    """Draw every point scanned while searching for corner split candidates."""
    h, w = shape[:2]
    out = np.full((h, w, 3), 255, dtype=np.uint8)

    legend = [
        ("gray=PCA segment angle below threshold", (130, 130, 130)),
        ("blue=high segment angle suppressed by local max", (255, 80, 0)),
        ("yellow=skipped near a previous accepted split", (0, 190, 230)),
        ("red hollow=local-max candidate rejected by final checks", (0, 0, 255)),
        ("green filled=candidate accepted", (0, 170, 0)),
    ]
    y = 24
    for text, color in legend:
        cv2.putText(out, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 1, cv2.LINE_AA)
        y += 22

    for item in trace:
        input_index = int(item.get("input_index", -1))
        for event in item.get("scan_events", []):
            p = event.get("point", None)
            if p is None:
                continue
            x, y = int(round(p[0])), int(round(p[1]))
            reason = event.get("reject_reason", "")
            candidate = bool(event.get("candidate", False))
            accepted = bool(event.get("accepted", False))

            if accepted:
                cv2.circle(out, (x, y), 5, (0, 170, 0), -1, cv2.LINE_AA)
                cv2.circle(out, (x, y), 7, (0, 110, 0), 1, cv2.LINE_AA)
            elif candidate:
                cv2.circle(out, (x, y), 5, (0, 0, 255), 1, cv2.LINE_AA)
            elif reason == "non_maximum_suppressed":
                cv2.circle(out, (x, y), 2, (255, 80, 0), -1, cv2.LINE_AA)
            elif reason == "near_previous_split":
                cv2.circle(out, (x, y), 2, (0, 190, 230), -1, cv2.LINE_AA)
            elif reason == "segment_window_too_short":
                cv2.circle(out, (x, y), 2, (180, 0, 180), -1, cv2.LINE_AA)
            else:
                cv2.circle(out, (x, y), 1, (130, 130, 130), -1, cv2.LINE_AA)

            if accepted or candidate:
                label = (
                    f"in{input_index}:{event.get('index', -1)} "
                    f"loc={event.get('folded_angle', 0.0):.1f} "
                    f"seg={event.get('segment_angle', 0.0):.1f}"
                )
                cv2.putText(
                    out,
                    label,
                    (x + 7, y - 7),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )

    cv2.putText(
        out,
        "All scanned corner-split points; candidates are local maxima of PCA segment_angle",
        (15, h - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    return out


def write_post_split_merge_trace_report(path, trace, max_gap=None, max_angle=None, protect_junction_radius=None):
    """Write post-corner-split merge operations."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("==== Post Corner Split Merge Trace ====\n\n")
        f.write(f"post_split_merge_gap: {max_gap}\n")
        f.write(f"post_split_merge_angle: {max_angle}\n")
        f.write(f"post_split_merge_protect_junction_radius: {protect_junction_radius}\n")
        f.write("Merges use endpoint gap, PCA axis angle, and merged endpoint-chord angle.\n\n")

        if not trace:
            f.write("No post-split merges.\n")
            return

        for i, item in enumerate(trace):
            action = item.get("action", "merge")
            if action == "skip_junction_protected":
                f.write(
                    f"{i:03d}: skip junction-protected current_stroke[{item.get('left_index')}] "
                    f"+ current_stroke[{item.get('right_index')}], "
                    f"gap={item.get('gap', 0.0):.2f}, "
                    f"angle={item.get('angle', 0.0):.2f}, "
                    f"merged_endpoint_angle={item.get('merged_endpoint_angle', 0.0):.2f}, "
                    f"merge_point={item.get('merge_point', None)}, "
                    f"protected_by=current_stroke[{item.get('protected_by_stroke', -1)}]:"
                    f"{'start' if int(item.get('protected_by_endpoint', 0)) == 0 else 'end'}, "
                    f"protected_dist={float(item.get('protected_by_distance', 0.0)):.2f}, "
                    f"protected_point={item.get('protected_by_point', None)}\n"
                )
            else:
                f.write(
                    f"{i:03d}: merge current_stroke[{item.get('left_index')}] "
                    f"+ current_stroke[{item.get('right_index')}], "
                    f"gap={item.get('gap', 0.0):.2f}, "
                    f"angle={item.get('angle', 0.0):.2f}, "
                    f"merged_endpoint_angle={item.get('merged_endpoint_angle', 0.0):.2f}, "
                    f"merge_point={item.get('merge_point', None)}, "
                    f"merged_len={item.get('merged_len', 0)}\n"
                )


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


def write_stroke_direction_debug_json(path, infos, line_strokes):
    """Write stroke geometry with full sampled points for downstream recovery."""
    if path is None:
        return
    line_ids = {int(s["index"]) for s in line_strokes}
    payload = {
        "strokes": [
            {
                "index": int(s["index"]),
                "line_candidate": int(s["index"]) in line_ids,
                "arc": float(s["arc"]),
                "chord": float(s["chord"]),
                "straightness": float(s["straightness"]),
                "center": [float(x) for x in np.asarray(s["center"], dtype=float).tolist()],
                "p0": [float(x) for x in np.asarray(s["points"][0], dtype=float).tolist()],
                "p1": [float(x) for x in np.asarray(s["points"][-1], dtype=float).tolist()],
                "points": [[float(x), float(y)] for x, y in np.asarray(s["points"], dtype=float).tolist()],
            }
            for s in infos
        ]
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


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


def direction_group_angle(group):
    """Return the displayed unoriented average angle for one direction group."""
    if not group:
        return 0.0
    return axis_angle_0_180(mean_direction(group))


def direction_group_center(group):
    """Return the mean point of all stroke centers in a direction group."""
    if not group:
        return np.array([0.0, 0.0], dtype=np.float64)
    centers = [s["center"].astype(np.float64) for s in group]
    return np.mean(centers, axis=0)


def write_direction_groups_debug_report(path, strokes, angle_thresh=25.0, min_stroke_length=None):
    """Write length-filtered direction groups produced by unoriented angle similarity."""
    groups = build_direction_clusters(strokes, angle_thresh=angle_thresh)
    with open(path, "w", encoding="utf-8") as f:
        f.write("==== Length-filtered Stroke Direction Groups ====\n\n")
        if min_stroke_length is not None:
            f.write(f"min_stroke_length: {float(min_stroke_length):.2f}\n")
        f.write(f"angle_thresh: {angle_thresh:.2f}\n")
        f.write("Grouping rule: strokes are connected when unoriented PCA angle <= threshold.\n")
        f.write("Same direction and opposite direction are treated as the same axis.\n\n")

        for gi, entry in enumerate(groups):
            group = entry["strokes"]
            angle = float(entry.get("mean_angle", direction_group_angle(group)))
            indices = [int(s["index"]) for s in group]
            f.write(
                f"group {gi:03d}: angle={angle:.2f}, "
                f"max_mean_diff={entry.get('max_mean_angle_diff', 0.0):.2f}, "
                f"source={entry.get('source', 'unknown')}, "
                f"strokes={indices}, n={len(group)}\n"
            )
            for s in group:
                dbg = stroke_direction_debug_values(s)
                f.write(
                    f"  stroke {int(s['index']):03d}: "
                    f"pca_angle={dbg['pca_angle']:.2f}, "
                    f"axis=({dbg['pca_axis'][0]:.4f},{dbg['pca_axis'][1]:.4f}), "
                    f"arc={s['arc']:.1f}, straightness={s['straightness']:.3f}\n"
                )


def draw_direction_groups_image(shape, strokes, angle_thresh=25.0, thickness=4, min_stroke_length=None):
    """Draw length-filtered direction groups with distinct colors and group angles."""
    h, w = shape[:2]
    out = np.full((h, w, 3), 255, dtype=np.uint8)
    groups = build_direction_clusters(strokes, angle_thresh=angle_thresh)

    for gi, entry in enumerate(groups):
        group = entry["strokes"]
        color = random_color(gi)
        for s in group:
            pts = s["points"].reshape(-1, 1, 2).astype(np.int32)
            cv2.polylines(out, [pts], False, color, thickness, cv2.LINE_AA)
            c = s["center"]
            cv2.putText(
                out,
                f"s{int(s['index'])}",
                (int(c[0]), int(c[1])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

        angle = float(entry.get("mean_angle", direction_group_angle(group)))
        ctr = direction_group_center(group)
        cv2.putText(
            out,
            f"G{gi} angle={angle:.1f} n={len(group)}",
            (int(ctr[0]) + 8, int(ctr[1]) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            color,
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        out,
        f"Length-filtered direction groups, threshold={angle_thresh:.1f} deg",
        (15, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        out,
        "Unoriented PCA angle: same and opposite directions are grouped together",
        (15, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    if min_stroke_length is not None:
        cv2.putText(
            out,
            f"Only strokes with arc >= {float(min_stroke_length):.1f} are grouped",
            (15, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (0, 0, 0),
            1,
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
                label = f"#{i} area={c['area']} enclosed={c.get('enclosed_area', 0)} end={c['endpoints']}"
            else:
                label = f"#{i} strokes={stroke_txt} enclosed={c.get('enclosed_area', 0)}"
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
    subgroup = ""
    if entry.get("is_removed_subgroup", False):
        subgroup = (
            f"parent={entry.get('parent_cluster_id', None)}, "
            f"removed={entry.get('removed_stroke_indices', [])}, "
            f"removal_depth={entry.get('removal_depth', 0)}, "
        )
    return (
        f"cluster {entry.get('cluster_id', -1):03d}{selected}: "
        f"score={entry.get('score', 0.0):.3f}, "
        f"source={entry.get('source', 'unknown')}, "
        f"{subgroup}"
        f"strokes={entry.get('indices', [])}, "
        f"dir=({direction[0]:.4f},{direction[1]:.4f}), "
        f"mean_angle={entry.get('mean_angle', axis_angle_0_180(direction)):.2f}, "
        f"max_mean_diff={entry.get('max_mean_angle_diff', 0.0):.2f}, "
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
            f"best_cap_enclosed={details.get('best_cap_enclosed_area', 0)}, "
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
                    f"best_enclosed_area={details.get('best_cap_enclosed_area', 0)}, "
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


def write_cap_search_trace_report(path, model):
    """Write cap-search attempts in the exact order they were executed."""
    trace = model.get("cap_search_trace", []) if model is not None else []
    with open(path, "w", encoding="utf-8") as f:
        f.write("==== Cap Search Trace ====\n\n")
        f.write("Order is the actual execution order.\n")
        f.write("Full direction groups are checked first.  Failed groups are then expanded level by level: remove-1, remove-2, ... until success or one stroke remains.\n")
        f.write("A direction group stops expanding as soon as its full group or one subgroup finds a valid cap.\n\n")

        if not trace:
            f.write("No cap search trace recorded.\n")
            return

        for item in trace:
            status = "CAP" if item.get("cap_found", False) else "no_cap"
            if item.get("skipped", False):
                status = f"skipped:{item.get('skip_reason', '')}"
            f.write(
                f"{int(item.get('order', -1)):04d}: "
                f"cluster={int(item.get('cluster_id', -1)):03d}, "
                f"source={item.get('source', 'unknown')}, "
                f"parent={item.get('parent_cluster_id', None)}, "
                f"depth={int(item.get('removal_depth', 0))}, "
                f"removed={item.get('removed_stroke_indices', [])}, "
                f"side={item.get('side_indices', [])}, "
                f"n={int(item.get('n', 0))}, "
                f"checked={item.get('checked', False)}, "
                f"result={status}, "
                f"cap_count={int(item.get('cap_candidate_count', 0))}, "
                f"best_area={int(item.get('best_cap_area', 0))}, "
                f"best_enclosed_area={int(item.get('best_cap_enclosed_area', 0))}, "
                f"best_total_arc={float(item.get('best_cap_total_arc', 0.0)):.1f}, "
                f"best_score={float(item.get('best_cap_score', 0.0)):.1f}, "
                f"best_cap_strokes={item.get('best_cap_strokes', [])}\n"
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
        f"mean_angle={entry.get('mean_angle', axis_angle_0_180(direction)):.1f} "
        f"max_diff={entry.get('max_mean_angle_diff', 0.0):.1f} "
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
    write_cap_search_trace_report(os.path.join(debug_dir, "05d_cap_search_trace.txt"), model)
    overview = draw_cluster_overview_image(img_shape, model)
    cv2.imwrite(os.path.join(debug_dir, "05b_direction_cluster_scores.png"), overview)

    cluster_dir = os.path.join(debug_dir, "clusters")
    os.makedirs(cluster_dir, exist_ok=True)
    selected_cluster_id = model.get("selected_cluster_id", None)
    for rank, entry in enumerate(cluster_debug):
        selected = entry.get("cluster_id") == selected_cluster_id
        img = draw_single_cluster_image(img_shape, entry, selected=selected, thickness=4)
        safe_score = sanitize_score_for_filename(float(entry.get("score", 0.0)))
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
    cv2.putText(
        out,
        f"enclosed_area={cap_candidate.get('enclosed_area', 0)}",
        (15, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        out,
        f"total_arc={cap_candidate.get('total_arc', 0.0):.1f}",
        (15, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    return out


def nearest_mask_point(point, mask_points):
    """Return the closest mask point to a 2D point."""
    if mask_points is None or len(mask_points) == 0:
        return None
    p = np.asarray(point, dtype=np.float64)
    d = mask_points - p
    idx = int(np.argmin(np.sum(d * d, axis=1)))
    return mask_points[idx]


def side_copy_majority_direction(entry):
    """Return the side-cluster majority direction used to validate copy vectors."""
    direction = entry.get("direction", None)
    if direction is not None:
        direction = np.asarray(direction, dtype=np.float64)
        if np.linalg.norm(direction) > 1e-8:
            return canonical_axis_direction(direction)
    strokes = entry.get("strokes", [])
    if strokes:
        return canonical_axis_direction(mean_direction(strokes))
    return None


def side_copy_geometry_from_candidate(candidate, cap_points, angle_tol, selection_reason, rejected=None, iou_comparison=None):
    """Build near/far endpoint copy geometry for one side-stroke candidate."""
    p0, p1 = candidate["p0"], candidate["p1"]
    axis = p1 - p0
    axis_norm = np.linalg.norm(axis)
    if axis_norm <= 1e-6:
        return None

    p0_cap = nearest_mask_point(p0, cap_points)
    p1_cap = nearest_mask_point(p1, cap_points)
    if p0_cap is None or p1_cap is None:
        return None
    endpoint0_distance = float(np.linalg.norm(p0 - p0_cap))
    endpoint1_distance = float(np.linalg.norm(p1 - p1_cap))
    if endpoint0_distance <= endpoint1_distance:
        near_endpoint, far_endpoint = p0, p1
    else:
        near_endpoint, far_endpoint = p1, p0

    result = {
        "near_endpoint": near_endpoint,
        "far_endpoint": far_endpoint,
        "vector": far_endpoint - near_endpoint,
        "selected_stroke": int(candidate["stroke"].get("index", -1)),
        "longest_length": float(candidate["stroke_length"]),
        "selected_chord": float(candidate["chord_length"]),
        "angle_to_majority": float(candidate["angle_to_majority"]),
        "copy_direction_angle_tol": float(angle_tol),
        "copy_selection_reason": selection_reason,
        "rejected_copy_side_candidates": rejected or [],
        "endpoint0": p0,
        "endpoint1": p1,
        "endpoint0_distance_to_cap": float(endpoint0_distance),
        "endpoint1_distance_to_cap": float(endpoint1_distance),
    }
    if iou_comparison is not None:
        result["copy_iou_comparison"] = iou_comparison
    return result


def evaluate_side_copy_candidate_iou(candidate, cap_mask, cap_points, sketch_mask, angle_tol):
    """Sweep a cap by one side-stroke vector and compute IoU with the sketch mask."""
    geometry = side_copy_geometry_from_candidate(
        candidate,
        cap_points,
        angle_tol,
        selection_reason="iou_trial",
        rejected=[],
    )
    if geometry is None or sketch_mask is None:
        return None
    if cap_mask.shape[:2] != sketch_mask.shape[:2]:
        return None

    swept_cap = fill_binary_mask(sweep_mask_along_vector(cap_mask, geometry["vector"]))
    occupied = fill_binary_mask(swept_cap)
    iou, intersection, union = binary_mask_iou(occupied, sketch_mask)
    return {
        "candidate": candidate,
        "geometry": geometry,
        "iou": float(iou),
        "intersection": int(intersection),
        "union": int(union),
        "sweep_area": int(np.count_nonzero(occupied > 0)),
    }


def side_copy_iou_compare_count(total_candidates, compare_percent):
    """Return how many longest side strokes to compare by IoU."""
    total = int(total_candidates)
    if total < 2:
        return 0
    if compare_percent is None:
        return 2
    percent = float(compare_percent)
    percent = max(0.0, min(100.0, percent))
    if percent <= 0.0:
        return 0
    count = int(math.ceil(total * percent / 100.0))
    count = max(2, count)
    return min(total, count)


def longest_side_stroke_copy_geometry(
    entry,
    cap_candidate,
    cap_mask_override=None,
    direction_angle_tol=None,
    sketch_mask=None,
    iou_compare_percent=None,
):
    """Return the longest direction-consistent side stroke and its near-to-far copy vector."""
    if cap_candidate is None:
        return None

    cap_mask = cap_mask_override if cap_mask_override is not None else cap_candidate.get("mask", None)
    if cap_mask is None:
        return None
    ys, xs = np.where(cap_mask > 0)
    if len(xs) == 0:
        return None
    cap_points = np.column_stack([xs, ys]).astype(np.float64)
    iou_cap_mask = cap_candidate.get("mask", None)
    if iou_cap_mask is None or np.count_nonzero(iou_cap_mask > 0) == 0:
        iou_cap_mask = cap_mask
    iou_ys, iou_xs = np.where(iou_cap_mask > 0)
    iou_cap_points = np.column_stack([iou_xs, iou_ys]).astype(np.float64) if len(iou_xs) > 0 else cap_points

    side_strokes = entry.get("strokes", [])
    if not side_strokes:
        return None

    majority_direction = side_copy_majority_direction(entry)
    angle_tol = 25.0 if direction_angle_tol is None else float(direction_angle_tol)
    candidates = []
    for s in side_strokes:
        p0, p1 = stroke_endpoint_points(s)
        p0 = np.asarray(p0, dtype=np.float64)
        p1 = np.asarray(p1, dtype=np.float64)
        stroke_length = float(s.get("arc", np.linalg.norm(p1 - p0)))
        chord = p1 - p0
        chord_length = float(np.linalg.norm(chord))
        if chord_length <= 1e-6:
            continue
        chord_direction = canonical_axis_direction(chord)
        angle_to_majority = 0.0
        if majority_direction is not None:
            angle_to_majority = angle_between_dirs(chord_direction, majority_direction)
        candidates.append({
            "stroke": s,
            "p0": p0,
            "p1": p1,
            "stroke_length": stroke_length,
            "chord_length": chord_length,
            "angle_to_majority": float(angle_to_majority),
        })

    if not candidates:
        return None

    candidates.sort(key=lambda item: item["stroke_length"], reverse=True)
    rejected = []
    selected = None
    selection_reason = None
    iou_comparison = None
    compare_count = side_copy_iou_compare_count(len(candidates), iou_compare_percent)

    if (
        compare_count >= 2
        and majority_direction is not None
        and sketch_mask is not None
        and candidates[0]["angle_to_majority"] > candidates[1]["angle_to_majority"] + 1e-6
    ):
        trials = []
        for candidate in candidates[:compare_count]:
            trial = evaluate_side_copy_candidate_iou(
                candidate,
                iou_cap_mask,
                iou_cap_points,
                sketch_mask,
                angle_tol,
            )
            if trial is not None:
                trials.append(trial)
        if len(trials) >= 2:
            trials.sort(
                key=lambda item: (
                    item["iou"],
                    item["candidate"]["stroke_length"],
                ),
                reverse=True,
            )
            selected = trials[0]["candidate"]
            selection_reason = f"iou_compare_top_{compare_count}_angle_deviation"
            iou_comparison = [
                {
                    "stroke": int(trial["candidate"]["stroke"].get("index", -1)),
                    "length": float(trial["candidate"]["stroke_length"]),
                    "chord": float(trial["candidate"]["chord_length"]),
                    "angle_to_majority": float(trial["candidate"]["angle_to_majority"]),
                    "iou": float(trial["iou"]),
                    "intersection": int(trial["intersection"]),
                    "union": int(trial["union"]),
                    "sweep_area": int(trial["sweep_area"]),
                    "compare_rank": int(i),
                }
                for i, trial in enumerate(trials)
            ]

    for candidate in candidates:
        if selected is not None:
            break
        if majority_direction is not None and candidate["angle_to_majority"] > angle_tol:
            rejected.append({
                "stroke": int(candidate["stroke"].get("index", -1)),
                "length": float(candidate["stroke_length"]),
                "chord": float(candidate["chord_length"]),
                "angle_to_majority": float(candidate["angle_to_majority"]),
                "reason": "angle_to_majority_gt_tolerance",
            })
            continue
        selected = candidate
        selection_reason = "longest_direction_consistent_side"
        break

    if selected is None:
        selected = candidates[0]
        selection_reason = "fallback_longest_no_direction_consistent_side"

    return side_copy_geometry_from_candidate(
        selected,
        cap_points,
        angle_tol,
        selection_reason,
        rejected=rejected,
        iou_comparison=iou_comparison,
    )


def estimate_cap_to_far_side_vector(
    entry,
    cap_candidate,
    cap_mask_override=None,
    direction_angle_tol=None,
    sketch_mask=None,
    iou_compare_percent=None,
):
    """Use the longest direction-consistent side stroke to copy the cap."""
    geometry = longest_side_stroke_copy_geometry(
        entry,
        cap_candidate,
        cap_mask_override=cap_mask_override,
        direction_angle_tol=direction_angle_tol,
        sketch_mask=sketch_mask,
        iou_compare_percent=iou_compare_percent,
    )
    if geometry is None:
        return None
    return geometry["vector"]


def translate_mask(mask, vector):
    """Translate a binary mask by a floating-point vector."""
    h, w = mask.shape[:2]
    dx, dy = float(vector[0]), float(vector[1])
    transform = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    return cv2.warpAffine(mask, transform, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)


def draw_cluster_side_and_best_cap_overlay(
    shape,
    entry,
    cap_candidate=None,
    selected=False,
    copy_direction_angle_tol=None,
    sketch_mask=None,
    copy_geometry_override=None,
    iou_compare_percent=None,
):
    """Draw side strokes, the detected cap, and the cap copied to the far side."""
    h, w = shape[:2]
    out = np.full((h, w, 3), 255, dtype=np.uint8)
    details = entry.get("details", {})
    side_color = (255, 0, 0)
    copied_cap_vector = None
    copy_geometry = None

    if cap_candidate is not None:
        cap_mask = cv2.dilate(cap_candidate["mask"], np.ones((3, 3), np.uint8), iterations=1)
        out[cap_mask > 0] = (0, 255, 0)
        copy_geometry = copy_geometry_override
        if copy_geometry is None:
            copy_geometry = longest_side_stroke_copy_geometry(
                entry,
                cap_candidate,
                direction_angle_tol=copy_direction_angle_tol,
                sketch_mask=sketch_mask,
                iou_compare_percent=iou_compare_percent,
            )
        copied_cap_vector = None if copy_geometry is None else copy_geometry["vector"]
        if copied_cap_vector is not None:
            copied_cap_mask = translate_mask(cap_mask, copied_cap_vector)
            out[copied_cap_mask > 0] = (0, 0, 255)

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

    if copy_geometry is not None:
        near = copy_geometry["near_endpoint"]
        far = copy_geometry["far_endpoint"]
        cv2.line(
            out,
            (int(round(near[0])), int(round(near[1]))),
            (int(round(far[0])), int(round(far[1]))),
            (0, 0, 0),
            6,
            cv2.LINE_AA,
        )
        cv2.circle(out, (int(round(near[0])), int(round(near[1]))), 10, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(out, (int(round(far[0])), int(round(far[1]))), 10, (255, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(
            out,
            "near",
            (int(round(near[0])) + 12, int(round(near[1])) - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            "far",
            (int(round(far[0])) + 12, int(round(far[1])) - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            (
                f"copy side=s{copy_geometry['selected_stroke']}, len={copy_geometry['longest_length']:.1f}, "
                f"ang={copy_geometry['angle_to_majority']:.1f}/{copy_geometry['copy_direction_angle_tol']:.1f}, "
                f"d0={copy_geometry['endpoint0_distance_to_cap']:.1f}, "
                f"d1={copy_geometry['endpoint1_distance_to_cap']:.1f}"
            ),
            (15, 78),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            f"copy reason={copy_geometry.get('copy_selection_reason', 'unknown')}",
            (15, 98),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
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
        if copied_cap_vector is not None:
            cap_txt += f", copied=({copied_cap_vector[0]:.1f},{copied_cap_vector[1]:.1f})"

            ctr = cap_candidate.get("center", None)
            if ctr is not None:
                ctr = np.asarray(ctr, dtype=np.float64)
                dst = ctr + copied_cap_vector
                cv2.arrowedLine(
                    out,
                    (int(round(ctr[0])), int(round(ctr[1]))),
                    (int(round(dst[0])), int(round(dst[1]))),
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA,
                    tipLength=0.12,
                )
                if 0 <= dst[0] < w and 0 <= dst[1] < h:
                    cv2.putText(
                        out,
                        "copied base cap",
                        (int(dst[0]) + 8, int(dst[1]) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.50,
                        (0, 0, 0),
                        1,
                        cv2.LINE_AA,
                    )

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


def draw_cap_endpoint_graph_image(shape, entry, cap_pool_infos, endpoint_tol=12.0):
    """Draw the non-side endpoint graph used by cap component validation."""
    h, w = shape[:2]
    out = np.full((h, w, 3), 255, dtype=np.uint8)
    side_ids = {int(s["index"]) for s in entry.get("strokes", [])}
    non_side_infos = [s for s in cap_pool_infos if int(s["index"]) not in side_ids]

    for s in entry.get("strokes", []):
        pts = s["points"].reshape(-1, 1, 2).astype(np.int32)
        cv2.polylines(out, [pts], False, (0, 0, 255), 4, cv2.LINE_AA)
        c = s["center"]
        cv2.putText(out, f"s{int(s['index'])}", (int(c[0]), int(c[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 120), 1, cv2.LINE_AA)

    comps = connected_components_by_endpoint_proximity(non_side_infos, endpoint_tol=endpoint_tol)

    for ci, comp in enumerate(comps):
        color = random_color(ci + 50)
        closed = is_closed_stroke_component_by_endpoint_proximity(non_side_infos, comp, endpoint_tol=endpoint_tol)
        pruned_comp, removed_branch = prune_open_branches_from_component(non_side_infos, comp, endpoint_tol=endpoint_tol)
        pruned_closed = is_closed_stroke_component_by_endpoint_proximity(non_side_infos, pruned_comp, endpoint_tol=endpoint_tol)
        for li in comp:
            s = non_side_infos[li]
            pts = s["points"].reshape(-1, 1, 2).astype(np.int32)
            cv2.polylines(out, [pts], False, color, 3, cv2.LINE_AA)
            c = s["center"]
            cv2.putText(out, f"s{int(s['index'])}", (int(c[0]), int(c[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.43, (0, 0, 0), 1, cv2.LINE_AA)

        all_pts = []
        for li in comp:
            for p in stroke_endpoint_points(non_side_infos[li]):
                all_pts.append(np.asarray(p, dtype=np.float64))
        if all_pts:
            label_pos = np.mean(np.asarray(all_pts), axis=0)
            status = "closed" if closed else "open"
            pruned_status = "closed" if pruned_closed else "open"
            cv2.putText(out, f"comp {ci}: {status}, prune={pruned_status}, n={len(pruned_comp)}/{len(comp)}",
                        (int(label_pos[0]) + 8, int(label_pos[1]) - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)

        # Draw real endpoint-to-endpoint connections used for component growth
        # and closed-loop degree checks.
        for li in comp:
            s = non_side_infos[li]
            for endpoint_i, p in enumerate(stroke_endpoint_points(s)):
                degree, matches = endpoint_connection_degree_in_component(
                    non_side_infos,
                    comp,
                    li,
                    endpoint_i,
                    endpoint_tol=endpoint_tol,
                )
                px, py = int(round(float(p[0]))), int(round(float(p[1])))
                ok = degree == 1
                endpoint_color = (0, 150, 0) if ok else (0, 0, 255)
                cv2.circle(out, (px, py), 6 if ok else 8, endpoint_color, 2, cv2.LINE_AA)
                cv2.putText(out, f"{int(s['index'])}{'s' if endpoint_i == 0 else 'e'}:d{degree}",
                            (px + 5, py - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.34, endpoint_color, 1, cv2.LINE_AA)

                for match_stroke_id, match_endpoint_i in matches:
                    match_stroke = next((x for x in non_side_infos if int(x["index"]) == match_stroke_id), None)
                    if match_stroke is None:
                        continue
                    q = stroke_endpoint_points(match_stroke)[match_endpoint_i]
                    qx, qy = int(round(float(q[0]))), int(round(float(q[1])))
                    if (px, py) <= (qx, qy):
                        cv2.line(out, (px, py), (qx, qy), (120, 120, 120), 1, cv2.LINE_AA)

    cluster_id = entry.get("cluster_id", -1)
    cv2.putText(out, f"cluster {cluster_id} cap endpoint graph, endpoint_tol={endpoint_tol}",
                (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(out, "red strokes=side, colored strokes=non-side components, red endpoints=degree != 1",
                (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(out, "gray lines=real endpoint-to-endpoint tolerance links; no endpoint centers are merged",
                (15, 73), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def save_cap_endpoint_graph_debug_outputs(debug_dir, img_shape, model, infos, args):
    """Save endpoint-graph visualizations for original direction groups."""
    if debug_dir is None or model is None:
        return
    cluster_debug = model.get("cluster_debug", [])
    if not cluster_debug:
        return

    out_dir = os.path.join(debug_dir, "cap_endpoint_graphs")
    os.makedirs(out_dir, exist_ok=True)
    #cap_pool_infos = [s for s in infos if s["arc"] >= args.min_stroke_length]
    cap_pool_infos = [s for s in infos if s["arc"] >= 10]
    summary_path = os.path.join(out_dir, "cap_endpoint_graph_summary.txt")

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("==== Cap Endpoint Graph Summary ====\n\n")
        f.write(f"cap_pool: strokes with arc >= {float(args.min_stroke_length):.1f}\n")
        f.write(f"endpoint_tol: {float(args.cap_loop_endpoint_tol):.1f}\n")
        f.write("Only original direction groups are listed. Subgroups are omitted.\n\n")

        for entry in cluster_debug:
            if int(entry.get("removal_depth", 0)) != 0:
                continue
            if entry.get("parent_cluster_id", None) is not None:
                continue
            cluster_id = int(entry.get("cluster_id", -1))
            img = draw_cap_endpoint_graph_image(
                img_shape,
                entry,
                cap_pool_infos,
                endpoint_tol=args.cap_loop_endpoint_tol,
            )
            cv2.imwrite(os.path.join(out_dir, f"cluster_{cluster_id:03d}_endpoint_graph.png"), img)

            side_ids = {int(s["index"]) for s in entry.get("strokes", [])}
            non_side_infos = [s for s in cap_pool_infos if int(s["index"]) not in side_ids]
            comps = connected_components_by_endpoint_proximity(non_side_infos, endpoint_tol=args.cap_loop_endpoint_tol)
            f.write(f"cluster {cluster_id:03d}: side={sorted(side_ids)}, non_side={[int(s['index']) for s in non_side_infos]}\n")
            for ci, comp in enumerate(comps):
                gids = [int(non_side_infos[i]["index"]) for i in comp]
                closed = is_closed_stroke_component_by_endpoint_proximity(
                    non_side_infos,
                    comp,
                    endpoint_tol=args.cap_loop_endpoint_tol,
                )
                pruned_comp, removed_branch = prune_open_branches_from_component(
                    non_side_infos,
                    comp,
                    endpoint_tol=args.cap_loop_endpoint_tol,
                )
                pruned_gids = [int(non_side_infos[i]["index"]) for i in pruned_comp]
                removed_gids = [int(non_side_infos[i]["index"]) for i in removed_branch]
                pruned_closed = is_closed_stroke_component_by_endpoint_proximity(
                    non_side_infos,
                    pruned_comp,
                    endpoint_tol=args.cap_loop_endpoint_tol,
                )
                f.write(
                    f"  component {ci:03d}: closed={closed}, strokes={gids}, "
                    f"pruned_closed={pruned_closed}, pruned_strokes={pruned_gids}, "
                    f"removed_open_branch_strokes={removed_gids}\n"
                )
                for li in comp:
                    sid = int(non_side_infos[li]["index"])
                    for endpoint_i, p in enumerate(stroke_endpoint_points(non_side_infos[li])):
                        degree, matches = endpoint_connection_degree_in_component(
                            non_side_infos,
                            comp,
                            li,
                            endpoint_i,
                            endpoint_tol=args.cap_loop_endpoint_tol,
                        )
                        end_name = "start" if endpoint_i == 0 else "end"
                        bad = " BAD" if degree != 1 else ""
                        match_txt = [f"s{m_sid}:{'start' if m_ep == 0 else 'end'}" for m_sid, m_ep in matches]
                        f.write(
                            f"    endpoint s{sid}:{end_name}=({p[0]:.1f},{p[1]:.1f}), "
                            f"degree={degree}, matches={match_txt}{bad}\n"
                        )
            f.write("\n")


def successful_direction_parent_ids(model):
    """Return original direction group ids that eventually found a valid cap."""
    ids = set()
    if model is None:
        return ids
    for item in model.get("cap_search_trace", []):
        if not item.get("cap_found", False):
            continue
        parent_id = item.get("parent_cluster_id", None)
        if parent_id is None:
            parent_id = item.get("cluster_id", None)
        if parent_id is not None:
            ids.add(int(parent_id))
    return ids


def entry_original_parent_id(entry):
    """Return the original direction group id for a full group or subgroup entry."""
    parent_id = entry.get("parent_cluster_id", None)
    if parent_id is None:
        parent_id = entry.get("cluster_id", None)
    return None if parent_id is None else int(parent_id)


def exact_color_mask(image, color):
    """Extract pixels that exactly match one BGR color in a rendered debug image."""
    color = np.asarray(color, dtype=np.uint8)
    return (np.all(image == color, axis=2).astype(np.uint8) * 255)


def centroid_from_mask(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return np.array([float(xs.mean()), float(ys.mean())], dtype=np.float64)


def sweep_mask_along_vector(mask, vector, steps=48):
    """Union repeated translations of a cap mask along the side-stroke vector."""
    out = np.zeros_like(mask)
    for t in np.linspace(0.0, 1.0, int(steps)):
        shifted = translate_mask(mask, np.asarray(vector, dtype=np.float64) * float(t))
        out[shifted > 0] = 255
    return out


def fill_binary_mask(mask, close_kernel=5):
    """Fill enclosed regions in a binary mask so swept caps become solid areas."""
    if np.count_nonzero(mask) == 0:
        return mask.copy()

    work = (mask > 0).astype(np.uint8) * 255
    if close_kernel and close_kernel > 1:
        k = np.ones((int(close_kernel), int(close_kernel)), np.uint8)
        work = cv2.morphologyEx(work, cv2.MORPH_CLOSE, k, iterations=1)

    contours, _ = cv2.findContours(work, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(work)
    if contours:
        cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return filled


def _parse_csv_ints(s):
    s = (s or "").strip()
    if not s:
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_cap_endpoint_graph_summary(summary_path):
    """
    Parse debug/cap_endpoint_graphs/cap_endpoint_graph_summary.txt into:
      { cluster_id: { "side": [...], "non_side": [...], "components": [ {...}, ... ] } }
    Each component has endpoints list with stroke, role, point, matches [(sid, role), ...].
    """
    if not summary_path or not os.path.isfile(summary_path):
        return {}
    with open(summary_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    clusters = {}
    current_cluster = None
    current_comp = None

    comp_re = re.compile(
        r"^  component (\d+): closed=(True|False), strokes=\[(.*?)\], "
        r"pruned_closed=(True|False), pruned_strokes=\[(.*?)\], removed_open_branch_strokes=\[(.*?)\]\s*$"
    )
    ep_re = re.compile(
        r"^    endpoint s(\d+):(start|end)=\(([\d.]+),([\d.]+)\), degree=(\d+), matches=\[(.*?)\]\s*$"
    )
    cluster_re = re.compile(r"^cluster (\d+): side=\[(.*?)\], non_side=\[(.*?)\]\s*$")

    for line in lines:
        mc = cluster_re.match(line)
        if mc:
            cid = int(mc.group(1))
            current_cluster = cid
            clusters[cid] = {
                "side": _parse_csv_ints(mc.group(2)),
                "non_side": _parse_csv_ints(mc.group(3)),
                "components": [],
            }
            current_comp = None
            continue

        mo = comp_re.match(line)
        if mo and current_cluster is not None:
            current_comp = {
                "id": int(mo.group(1)),
                "closed": mo.group(2) == "True",
                "strokes": _parse_csv_ints(mo.group(3)),
                "pruned_closed": mo.group(4) == "True",
                "pruned_strokes": _parse_csv_ints(mo.group(5)),
                "removed": _parse_csv_ints(mo.group(6)),
                "endpoints": [],
            }
            clusters[current_cluster]["components"].append(current_comp)
            continue

        me = ep_re.match(line)
        if me and current_comp is not None:
            sid = int(me.group(1))
            role = me.group(2)
            x, y = float(me.group(3)), float(me.group(4))
            matches_raw = (me.group(6) or "").strip()
            matches = [(int(m.group(1)), m.group(2)) for m in re.finditer(r"s(\d+):(start|end)", matches_raw)]
            current_comp["endpoints"].append(
                {"stroke": sid, "role": role, "point": (x, y), "matches": matches}
            )
            continue

    return clusters


def parse_stroke_directions_p0p1(directions_path):
    """
    Parse 05a_stroke_directions.txt "All strokes:" section:
      stroke NNN: ... p0=(x,y), p1=(x,y), ...
    Returns dict stroke_id -> (p0_tuple, p1_tuple).
    """
    result = {}
    if not directions_path or not os.path.isfile(directions_path):
        return result
    stroke_line_re = re.compile(
        r"^\s*stroke (\d+):.*?p0=\(([\d.-]+)\s*,\s*([\d.-]+)\)\s*,\s*p1=\(([\d.-]+)\s*,\s*([\d.-]+)\)"
    )
    in_all = False
    with open(directions_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("All strokes:"):
                in_all = True
                continue
            if line.startswith("Line stroke candidates only:"):
                break
            if not in_all:
                continue
            m = stroke_line_re.match(line)
            if not m:
                continue
            sid = int(m.group(1))
            p0 = (float(m.group(2)), float(m.group(3)))
            p1 = (float(m.group(4)), float(m.group(5)))
            result[sid] = (p0, p1)
    return result


def entry_uses_cap_endpoint_summary(entry):
    """Summary file lists only original direction groups (same filter as when it is written)."""
    if entry is None:
        return False
    if int(entry.get("removal_depth", 0)) != 0:
        return False
    if entry.get("parent_cluster_id", None) is not None:
        return False
    return True


def _pick_component_for_cap_strokes(cluster_data, stroke_indices):
    """Choose the summary component that corresponds to cap_candidate stroke_indices."""
    if not cluster_data or not stroke_indices:
        return None
    wanted = {int(i) for i in stroke_indices}

    def score_comp(comp):
        ps = set(comp.get("pruned_strokes", []))
        ss = set(comp.get("strokes", []))
        if comp.get("pruned_closed") and ps == wanted:
            return (0, len(ps))
        if comp.get("closed") and ss == wanted:
            return (1, len(ss))
        return None

    best = None
    best_key = None
    for comp in cluster_data.get("components", []):
        key = score_comp(comp)
        if key is None:
            continue
        if best_key is None or key < best_key:
            best_key = key
            best = comp
    return best


def mask_source_cap_from_summary_file(
    shape,
    cluster_id,
    stroke_indices,
    summary_path,
    *,
    stroke_directions_path=None,
    line_thickness=3,
    fill_interior=True,
    gap_connector_px=2.0,
):
    """
    Cap footprint: topology (matches) from cap_endpoint_graph_summary.txt plus stroke chords.

    For every stroke id in cap_candidate stroke_indices, draw the straight chord p0–p1 from
    05a_stroke_directions.txt (same line thickness as the primary loop strokes).

    Also draw each summary match as a segment between the matched endpoints (05a positions);
    when endpoints are farther apart than gap_connector_px, use a thinner bridge stroke.

    No summary-file coordinate fallback — missing 05a data yields None.
    """
    if not stroke_directions_path or not os.path.isfile(stroke_directions_path):
        return None
    data = parse_cap_endpoint_graph_summary(summary_path)
    if cluster_id not in data:
        return None
    comp = _pick_component_for_cap_strokes(data[cluster_id], stroke_indices)
    if comp is None:
        return None

    p0p1 = parse_stroke_directions_p0p1(stroke_directions_path)

    key_to_point = {}
    for ep in comp["endpoints"]:
        sid, role = ep["stroke"], ep["role"]
        if sid not in p0p1:
            return None
        p0, p1 = p0p1[sid]
        key_to_point[(sid, role)] = p0 if role == "start" else p1

    for ep in comp["endpoints"]:
        for msid, mrole in ep["matches"]:
            if (msid, mrole) not in key_to_point:
                return None

    h, w = int(shape[0]), int(shape[1])
    mask = np.zeros((h, w), dtype=np.uint8)
    drawn_edges = set()
    lt = max(1, int(line_thickness))
    bridge_t = max(1, min(lt, lt // 2 or 1))

    cap_sid_set = {int(i) for i in stroke_indices}
    for sid in sorted(cap_sid_set):
        if sid not in p0p1:
            return None
        p0f, p1f = p0p1[sid]
        dist = math.hypot(p0f[0] - p1f[0], p0f[1] - p1f[1])
        if dist < 1e-9:
            continue
        a = (int(round(p0f[0])), int(round(p0f[1])))
        b = (int(round(p1f[0])), int(round(p1f[1])))
        if not (0 <= a[0] < w and 0 <= a[1] < h):
            continue
        if not (0 <= b[0] < w and 0 <= b[1] < h):
            continue
        edge = tuple(sorted((a, b)))
        if edge in drawn_edges:
            continue
        drawn_edges.add(edge)
        cv2.line(mask, a, b, 255, int(lt), cv2.LINE_AA)

    for ep in comp["endpoints"]:
        k = (ep["stroke"], ep["role"])
        pf = key_to_point[k]
        for msid, mrole in ep["matches"]:
            qf = key_to_point[(msid, mrole)]
            dist = math.hypot(pf[0] - qf[0], pf[1] - qf[1])
            if dist < 1e-9:
                continue
            a = (int(round(pf[0])), int(round(pf[1])))
            b = (int(round(qf[0])), int(round(qf[1])))
            if not (0 <= a[0] < w and 0 <= a[1] < h):
                continue
            if not (0 <= b[0] < w and 0 <= b[1] < h):
                continue
            edge = tuple(sorted((a, b)))
            if edge in drawn_edges:
                continue
            drawn_edges.add(edge)
            thick = bridge_t if dist > float(gap_connector_px) else lt
            cv2.line(mask, a, b, 255, int(thick), cv2.LINE_AA)

    if np.count_nonzero(mask) == 0:
        return None
    if fill_interior:
        mask = fill_binary_mask(mask)
    return mask


def collect_strokes_by_indices(infos, stroke_indices):
    if not stroke_indices:
        return []
    wanted = {int(i) for i in stroke_indices}
    return [s for s in infos if int(s["index"]) in wanted]


def rasterize_side_strokes(shape, side_strokes, thickness=4):
    mask = np.zeros(shape[:2], dtype=np.uint8)
    for s in side_strokes:
        pts = s["points"].reshape(-1, 1, 2).astype(np.int32)
        cv2.polylines(mask, [pts], False, 255, thickness, cv2.LINE_AA)
    return mask


def build_bbox_masks_geometric(
    shape,
    entry,
    cap_candidate,
    infos,
    endpoint_tol,
    cap_loop_thickness,
    debug_dir=None,
    copy_direction_angle_tol=None,
    sketch_mask=None,
    iou_compare_percent=None,
):
    """
    IoU / sweep masks use the original curved cap stroke pixels from cap_candidate["mask"].

    Sweep vector = longest side stroke by arc length, oriented near→far vs source cap mask.
    """
    if cap_candidate is None:
        return None
    stroke_indices = cap_candidate.get("stroke_indices", [])
    wanted = {int(i) for i in stroke_indices}
    loop_strokes = collect_strokes_by_indices(infos, stroke_indices)
    if len(loop_strokes) != len(wanted):
        return None
    bt = max(2, int(cap_loop_thickness))

    raw_source = cap_candidate.get("mask", None)
    if raw_source is None or np.count_nonzero(raw_source > 0) == 0:
        return None
    raw_source = (raw_source > 0).astype(np.uint8) * 255
    mask_source = "cap_candidate_original_curved_stroke_pixels"

    geometry = longest_side_stroke_copy_geometry(
        entry,
        cap_candidate,
        cap_mask_override=raw_source,
        direction_angle_tol=copy_direction_angle_tol,
        sketch_mask=sketch_mask,
        iou_compare_percent=iou_compare_percent,
    )
    if geometry is None:
        return None
    vec = geometry["vector"]
    copied_cap = translate_mask(raw_source, vec)
    side_bin = rasterize_side_strokes(shape, entry.get("strokes", []), thickness=4)
    swept_cap = fill_binary_mask(sweep_mask_along_vector(raw_source, vec))
    occupied = fill_binary_mask(swept_cap)
    return {
        "source_cap": raw_source,
        "copied_cap": copied_cap,
        "side_strokes": side_bin,
        "swept_cap": swept_cap,
        "occupied": occupied,
        "sweep_vector": vec,
        "geometry": geometry,
        "mask_source": mask_source,
    }


def save_bbox_masks_geometric(
    out_dir,
    base,
    shape,
    entry,
    cap_candidate,
    infos,
    endpoint_tol,
    cap_loop_thickness,
    debug_dir=None,
    copy_direction_angle_tol=None,
    sketch_mask=None,
    iou_compare_percent=None,
):
    masks = build_bbox_masks_geometric(
        shape,
        entry,
        cap_candidate,
        infos,
        endpoint_tol,
        cap_loop_thickness,
        debug_dir=debug_dir,
        copy_direction_angle_tol=copy_direction_angle_tol,
        sketch_mask=sketch_mask,
        iou_compare_percent=iou_compare_percent,
    )
    if masks is None:
        return None
    cv2.imwrite(os.path.join(out_dir, base + "_mask_source_cap.png"), masks["source_cap"])
    cv2.imwrite(os.path.join(out_dir, base + "_mask_copied_cap.png"), masks["copied_cap"])
    cv2.imwrite(os.path.join(out_dir, base + "_mask_side_strokes.png"), masks["side_strokes"])
    cv2.imwrite(os.path.join(out_dir, base + "_mask_swept_cap.png"), masks["swept_cap"])
    cv2.imwrite(os.path.join(out_dir, base + "_mask_extrusion_occupied.png"), masks["occupied"])
    return masks


def draw_cap_sweep_visualization(bbox_masks):
    """Same legend as draw_bbox_from_bestcap_overlay but driven by geometric masks."""
    green_cap = bbox_masks["source_cap"]
    red_cap = bbox_masks["copied_cap"]
    blue_side = bbox_masks["side_strokes"]
    sweep = bbox_masks["swept_cap"]

    h, w = green_cap.shape[:2]
    out = np.full((h, w, 3), 255, dtype=np.uint8)

    if np.count_nonzero(sweep) > 0:
        out[sweep > 0] = (220, 220, 220)

    out[blue_side > 0] = (255, 0, 0)
    out[green_cap > 0] = (0, 255, 0)
    out[red_cap > 0] = (0, 0, 255)

    g_ctr = centroid_from_mask(green_cap)
    r_ctr = centroid_from_mask(red_cap)
    if g_ctr is not None and r_ctr is not None:
        cv2.arrowedLine(
            out,
            (int(round(g_ctr[0])), int(round(g_ctr[1]))),
            (int(round(r_ctr[0])), int(round(r_ctr[1]))),
            (0, 0, 0),
            2,
            cv2.LINE_AA,
            tipLength=0.12,
        )

    cv2.putText(
        out,
        "cap sweep (geometry): gray=swept, green=filled source cap, red=copied cap, blue=sides",
        (15, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    return out


def build_bbox_masks_from_overlay(overlay_img, direction_vector):
    """Build binary masks from the rendered overlay colors (legacy RGB scrape fallback)."""
    green_cap = fill_binary_mask(exact_color_mask(overlay_img, (0, 255, 0)))
    red_cap = fill_binary_mask(exact_color_mask(overlay_img, (0, 0, 255)))
    blue_side = exact_color_mask(overlay_img, (255, 0, 0))
    swept_cap = np.zeros_like(green_cap)

    if direction_vector is not None and np.count_nonzero(green_cap) > 0:
        swept_cap = fill_binary_mask(sweep_mask_along_vector(green_cap, direction_vector))

    # The final extrusion occupancy is the cap volume swept from one cap to the other.
    occupied = fill_binary_mask(swept_cap)

    return {
        "source_cap": green_cap,
        "copied_cap": red_cap,
        "side_strokes": blue_side,
        "swept_cap": swept_cap,
        "occupied": occupied,
    }


def save_bbox_masks_from_overlay(out_dir, base, overlay_img, direction_vector):
    """Save black/white masks for caps, side strokes, and occupied extrusion pixels."""
    masks = build_bbox_masks_from_overlay(overlay_img, direction_vector)
    cv2.imwrite(os.path.join(out_dir, base + "_mask_source_cap.png"), masks["source_cap"])
    cv2.imwrite(os.path.join(out_dir, base + "_mask_copied_cap.png"), masks["copied_cap"])
    cv2.imwrite(os.path.join(out_dir, base + "_mask_side_strokes.png"), masks["side_strokes"])
    cv2.imwrite(os.path.join(out_dir, base + "_mask_swept_cap.png"), masks["swept_cap"])
    cv2.imwrite(os.path.join(out_dir, base + "_mask_extrusion_occupied.png"), masks["occupied"])
    return masks


def binary_mask_area(mask):
    return int(np.count_nonzero(mask > 0))


def binary_mask_iou(mask_a, mask_b):
    a = mask_a > 0
    b = mask_b > 0
    intersection = int(np.count_nonzero(a & b))
    union = int(np.count_nonzero(a | b))
    if union == 0:
        return 0.0, intersection, union
    return float(intersection) / float(union), intersection, union


def clean_ranked_output_dir(path):
    os.makedirs(path, exist_ok=True)
    for name in os.listdir(path):
        if name.lower().endswith((".png", ".txt")):
            os.remove(os.path.join(path, name))


def save_iou_ranked_side_bestcap_overlays(debug_dir, records, sketch_area):
    """Save side/cap overlays ordered by IoU with the sketch enclosed mask."""
    if not records:
        return

    ranked_dir = os.path.join(debug_dir, "cluster_side_caps_iou_ranked")
    clean_ranked_output_dir(ranked_dir)

    sorted_records = sorted(records, key=lambda r: r["iou"], reverse=True)
    summary_path = os.path.join(ranked_dir, "iou_similarity_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("==== Cap Sweep IoU Similarity Ranking ====\n\n")
        f.write(f"sketch_mask_area: {int(sketch_area)}\n")
        f.write("iou = intersection(mask_extrusion_occupied, sketch_enclosed_mask) / union(...)\n\n")

        for sorted_rank, record in enumerate(sorted_records):
            iou_tag = int(round(record["iou"] * 10000))
            filename = (
                f"iou_rank_{sorted_rank:02d}_iou_{iou_tag:04d}_"
                f"inter_{int(record['intersection'])}_union_{int(record['union'])}_"
                f"{record['base']}_side_bestcap_overlay.png"
            )
            cv2.imwrite(os.path.join(ranked_dir, filename), record["overlay_img"])
            f.write(
                f"iou_rank {sorted_rank:02d}: iou={record['iou']:.6f}, "
                f"intersection={int(record['intersection'])}, union={int(record['union'])}, "
                f"sweep_area={int(record['sweep_area'])}, sketch_area={int(sketch_area)}, "
                f"copy_side_stroke={record.get('copy_side_stroke', -1)}, "
                f"copy_reason={record.get('copy_reason', 'unknown')}, "
                f"source={record['base']}\n"
            )


def draw_bbox_from_bestcap_overlay(overlay_img, direction_vector):
    """Visualize the direct cap sweep mask from source cap to copied cap."""
    masks = build_bbox_masks_from_overlay(overlay_img, direction_vector)
    green_cap = masks["source_cap"]
    red_cap = masks["copied_cap"]
    blue_side = masks["side_strokes"]
    sweep = masks["swept_cap"]

    h, w = overlay_img.shape[:2]
    out = np.full((h, w, 3), 255, dtype=np.uint8)

    if np.count_nonzero(sweep) > 0:
        out[sweep > 0] = (220, 220, 220)

    out[blue_side > 0] = (255, 0, 0)
    out[green_cap > 0] = (0, 255, 0)
    out[red_cap > 0] = (0, 0, 255)

    g_ctr = centroid_from_mask(green_cap)
    r_ctr = centroid_from_mask(red_cap)
    if g_ctr is not None and r_ctr is not None:
        cv2.arrowedLine(
            out,
            (int(round(g_ctr[0])), int(round(g_ctr[1]))),
            (int(round(r_ctr[0])), int(round(r_ctr[1]))),
            (0, 0, 0),
            2,
            cv2.LINE_AA,
            tipLength=0.12,
        )

    cv2.putText(
        out,
        "direct cap sweep mask: gray=swept occupancy, green=source cap, red=copied cap, blue=side strokes",
        (15, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    return out


def save_per_cluster_side_cap_outputs(debug_dir, img_shape, model, infos, args):
    """Save side/cap visualizations for ranked side clusters.

    Only original direction-group search paths that eventually found a valid cap
    are output. Fully failed direction groups and their failed subgroups are not
    written to this directory.
    """
    if debug_dir is None or model is None:
        return
    cluster_debug = model.get("cluster_debug", [])
    if not cluster_debug:
        return

    out_dir = os.path.join(debug_dir, "cluster_side_caps")
    os.makedirs(out_dir, exist_ok=True)
    selected_cluster_id = model.get("selected_cluster_id", None)
    successful_parent_ids = successful_direction_parent_ids(model)
    #cap_pool_infos = [s for s in infos if s["arc"] >= args.min_stroke_length]
    cap_pool_infos = [s for s in infos if s["arc"] >= 10]
    sketch_mask_path = os.path.join(debug_dir, "00c_input_enclosed_mask.png")
    sketch_mask = cv2.imread(sketch_mask_path, cv2.IMREAD_GRAYSCALE)
    sketch_area = binary_mask_area(sketch_mask) if sketch_mask is not None else 0
    iou_rank_records = []

    summary_path = os.path.join(out_dir, "cluster_side_cap_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("==== Per-cluster Side/Cap Visualization Summary ====\n\n")
        f.write("Every direction group is tried as side strokes.\n")
        f.write("When a direction group has no valid cap, remove-k-stroke subgroups are tried level by level until success or one stroke remains.\n")
        f.write("Clusters with n<2 are recorded but do not compute cap.\n")
        f.write("Only direction-group search paths that eventually found a valid cap are emitted here.\n")
        f.write("Each emitted cluster stores only the largest-area cap candidate.\n\n")

        for rank, entry in enumerate(cluster_debug):
            details = entry.get("details", {})
            if details.get("skip_percluster_output", False):
                continue
            parent_id = entry_original_parent_id(entry)
            if parent_id not in successful_parent_ids:
                continue

            selected = entry.get("cluster_id") == selected_cluster_id
            cluster_id = entry.get("cluster_id", -1)
            score_tag = sanitize_score_for_filename(float(entry.get("score", 0.0)))
            base = f"rank_{rank:02d}_cluster_{cluster_id:02d}_score_{score_tag}"

            side_img = draw_single_cluster_image(img_shape, entry, selected=selected, thickness=4)
            cv2.imwrite(os.path.join(out_dir, base + "_side.png"), side_img)

            cap_candidate = None
            skip_reason = ""

            n_strokes = int(details.get("n", len(entry.get("strokes", []))))

            if n_strokes < 2:
                skip_reason = "n_lt_2"
            else:
                candidates = extract_cap_loop_candidates_from_strokes(
                    img_shape,
                    cap_pool_infos,
                    entry.get("strokes", []),
                    endpoint_tol=args.cap_loop_endpoint_tol,
                    min_pixels=args.min_cap_pixels,
                    min_enclosed_area=args.min_cap_enclosed_area,
                    min_total_arc=args.min_cap_total_arc,
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

            bbox_masks = None
            bbox_img = None
            if cap_candidate is not None:
                bbox_masks = save_bbox_masks_geometric(
                    out_dir,
                    base,
                    img_shape,
                    entry,
                    cap_candidate,
                    infos,
                    endpoint_tol=args.cap_loop_endpoint_tol,
                    cap_loop_thickness=args.cap_loop_thickness,
                    debug_dir=debug_dir,
                    copy_direction_angle_tol=args.parallel_angle_thresh,
                    sketch_mask=sketch_mask,
                    iou_compare_percent=args.copy_side_iou_compare_percent,
                )
            overlay_geometry = bbox_masks.get("geometry") if bbox_masks is not None else None
            overlay_img = draw_cluster_side_and_best_cap_overlay(
                img_shape,
                entry,
                cap_candidate=cap_candidate,
                selected=selected,
                copy_direction_angle_tol=args.parallel_angle_thresh,
                sketch_mask=sketch_mask,
                copy_geometry_override=overlay_geometry,
                iou_compare_percent=args.copy_side_iou_compare_percent,
            )
            cv2.imwrite(os.path.join(out_dir, base + "_side_bestcap_overlay.png"), overlay_img)

            if bbox_masks is None:
                bbox_vector = estimate_cap_to_far_side_vector(
                    entry,
                    cap_candidate,
                    direction_angle_tol=args.parallel_angle_thresh,
                    sketch_mask=sketch_mask,
                    iou_compare_percent=args.copy_side_iou_compare_percent,
                )
                if bbox_vector is not None:
                    bbox_masks = save_bbox_masks_from_overlay(out_dir, base, overlay_img, bbox_vector)
                    bbox_img = draw_bbox_from_bestcap_overlay(overlay_img, bbox_vector)
            else:
                bbox_img = draw_cap_sweep_visualization(bbox_masks)

            if bbox_masks is not None and bbox_img is not None:
                cv2.imwrite(os.path.join(out_dir, base + "_cap_sweep.png"), bbox_img)
                if sketch_area > 0 and sketch_mask is not None:
                    sweep_area = binary_mask_area(bbox_masks["occupied"])
                    iou, intersection, union = binary_mask_iou(bbox_masks["occupied"], sketch_mask)
                    iou_rank_records.append({
                        "base": base,
                        "overlay_img": overlay_img.copy(),
                        "sweep_area": sweep_area,
                        "sketch_area": sketch_area,
                        "iou": iou,
                        "intersection": intersection,
                        "union": union,
                        "copy_side_stroke": (
                            int(bbox_masks.get("geometry", {}).get("selected_stroke", -1))
                            if bbox_masks is not None
                            else -1
                        ),
                        "copy_reason": (
                            bbox_masks.get("geometry", {}).get("copy_selection_reason", "unknown")
                            if bbox_masks is not None
                            else "unknown"
                        ),
                    })

            if cap_candidate is not None:
                copy_geometry = bbox_masks.get("geometry") if bbox_masks is not None else overlay_geometry
                copy_side_txt = ""
                if copy_geometry is not None:
                    copy_iou = ""
                    comparison = copy_geometry.get("copy_iou_comparison") or []
                    if comparison:
                        selected_copy_iou = next(
                            (
                                item.get("iou", 0.0)
                                for item in comparison
                                if int(item.get("stroke", -1)) == int(copy_geometry.get("selected_stroke", -1))
                            ),
                            0.0,
                        )
                        copy_iou = f" copy_iou={float(selected_copy_iou):.6f}"
                    copy_side_txt = (
                        f" copy_side_stroke={int(copy_geometry.get('selected_stroke', -1))} "
                        f"copy_reason={copy_geometry.get('copy_selection_reason', 'unknown')} "
                        f"copy_angle_to_mean={float(copy_geometry.get('angle_to_majority', 0.0)):.2f}"
                        f"{copy_iou}"
                    )
                f.write(
                    f"rank {rank:02d} cluster {cluster_id:02d} selected={selected} "
                    f"side={entry.get('indices', [])} best_cap_strokes={cap_candidate.get('stroke_indices', [])} "
                    f"best_cap_area={cap_candidate.get('area', 0)} best_cap_enclosed_area={cap_candidate.get('enclosed_area', 0)} "
                    f"best_cap_total_arc={cap_candidate.get('total_arc', 0.0):.1f} "
                    f"best_cap_score={cap_candidate.get('score', 0.0):.1f} "
                    f"removed_post_loop_self_strokes={cap_candidate.get('removed_post_loop_self_strokes', [])} "
                    f"{copy_side_txt} "
                    f"source={entry.get('source', 'unknown')} parent={entry.get('parent_cluster_id', None)} "
                    f"removed={entry.get('removed_stroke_indices', [])} depth={entry.get('removal_depth', 0)}\n"
                )
            else:
                f.write(
                    f"rank {rank:02d} cluster {cluster_id:02d} selected={selected} "
                    f"side={entry.get('indices', [])} cap=NONE reason={skip_reason} "
                    f"source={entry.get('source', 'unknown')} parent={entry.get('parent_cluster_id', None)} "
                    f"removed={entry.get('removed_stroke_indices', [])} depth={entry.get('removal_depth', 0)}\n"
                )

    save_iou_ranked_side_bestcap_overlays(debug_dir, iou_rank_records, sketch_area)


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
                f"enclosed_area={c.get('enclosed_area', 0)}, "
                f"endpoints={c['endpoints']}, closedness={c['closedness']:.2f}, "
                f"total_arc={c.get('total_arc', 0.0):.1f}, "
                f"score={c['score']:.1f}, center={ctr_txt}, strokes={stroke_txt}, "
                f"removed_post_loop_self_strokes={c.get('removed_post_loop_self_strokes', [])}\n"
            )


def flood_enclosed_regions_from_barrier(barrier):
    """Return enclosed background regions inside a closed stroke barrier."""
    background = np.where(barrier > 0, 0, 255).astype(np.uint8)
    flood = background.copy()
    h, w = flood.shape
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)

    border_points = []
    for x in range(w):
        border_points.append((x, 0))
        border_points.append((x, h - 1))
    for y in range(h):
        border_points.append((0, y))
        border_points.append((w - 1, y))

    for x, y in border_points:
        if flood[y, x] == 255:
            cv2.floodFill(flood, flood_mask, (int(x), int(y)), 128)

    return flood == 255


def close_stroke_barrier(stroke_mask, close_kernel):
    barrier = (stroke_mask > 0).astype(np.uint8) * 255
    if close_kernel and close_kernel > 1:
        k = np.ones((int(close_kernel), int(close_kernel)), np.uint8)
        barrier = cv2.morphologyEx(barrier, cv2.MORPH_CLOSE, k, iterations=1)
    return barrier


def convex_hull_mask_from_strokes(stroke_mask):
    """Fallback closed region when the sketch has gaps too large for morphology."""
    ys, xs = np.where(stroke_mask > 0)
    hull_mask = np.zeros_like(stroke_mask, dtype=np.uint8)
    hull_barrier = np.zeros_like(stroke_mask, dtype=np.uint8)
    if len(xs) < 3:
        return hull_mask, hull_barrier

    pts = np.column_stack([xs, ys]).astype(np.int32)
    hull = cv2.convexHull(pts.reshape(-1, 1, 2))
    cv2.drawContours(hull_mask, [hull], -1, 255, thickness=cv2.FILLED)
    cv2.polylines(hull_barrier, [hull], True, 255, thickness=3, lineType=cv2.LINE_AA)
    return hull_mask, hull_barrier


def binary_component_stats(mask):
    binary = (mask > 0).astype(np.uint8)
    num, _labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    areas = []
    for i in range(1, num):
        areas.append(int(stats[i, cv2.CC_STAT_AREA]))
    areas.sort(reverse=True)
    total = int(np.count_nonzero(binary))
    largest = areas[0] if areas else 0
    return {
        "component_count": int(len(areas)),
        "component_areas": areas,
        "total_area": total,
        "largest_area": int(largest),
        "largest_ratio": float(largest / max(total, 1)),
    }


def is_single_connected_mask(mask, min_largest_ratio=0.98):
    stats = binary_component_stats(mask)
    if stats["component_count"] <= 1:
        return True
    return stats["largest_ratio"] >= float(min_largest_ratio)


def skeleton_endpoints_for_closure(skel):
    """Find real skeleton endpoints used to close small drawing gaps."""
    endpoints = []
    ys, xs = np.where(skel > 0)
    for x, y in zip(xs, ys):
        p = (int(x), int(y))
        if skeleton_node_type(skel, p) == "endpoint":
            endpoints.append(p)
    return endpoints


def connect_nearby_endpoints(barrier, skel, endpoint_tol=50.0, thickness=5):
    """Connect all skeleton endpoint pairs within endpoint_tol."""
    out = barrier.copy()
    endpoints = skeleton_endpoints_for_closure(skel)
    connections = []
    tol2 = float(endpoint_tol) * float(endpoint_tol)

    for i, p in enumerate(endpoints):
        p_arr = np.asarray(p, dtype=np.float64)
        for j in range(i + 1, len(endpoints)):
            q = endpoints[j]
            q_arr = np.asarray(q, dtype=np.float64)
            d2 = float(np.dot(p_arr - q_arr, p_arr - q_arr))
            if d2 > tol2:
                continue
            cv2.line(out, p, q, 255, int(thickness), cv2.LINE_AA)
            connections.append((p, q, math.sqrt(d2)))

    return out, endpoints, connections


def connect_each_endpoint_to_nearest(barrier, skel, thickness=5):
    """Connect every endpoint to its nearest other endpoint so no endpoint remains dead."""
    out = barrier.copy()
    endpoints = skeleton_endpoints_for_closure(skel)
    if len(endpoints) < 2:
        return out, endpoints, [], len(endpoints)

    connections_by_pair = {}
    for i, p in enumerate(endpoints):
        p_arr = np.asarray(p, dtype=np.float64)
        best_j = None
        best_d2 = float("inf")
        for j, q in enumerate(endpoints):
            if i == j:
                continue
            q_arr = np.asarray(q, dtype=np.float64)
            d2 = float(np.dot(p_arr - q_arr, p_arr - q_arr))
            if d2 < best_d2:
                best_d2 = d2
                best_j = j
        if best_j is None:
            continue
        a, b = sorted((i, best_j))
        connections_by_pair[(a, b)] = (endpoints[a], endpoints[b], math.sqrt(best_d2))

    for p, q, _dist in connections_by_pair.values():
        cv2.line(out, p, q, 255, int(thickness), cv2.LINE_AA)

    connected_endpoint_ids = set()
    for a, b in connections_by_pair:
        connected_endpoint_ids.add(a)
        connected_endpoint_ids.add(b)
    dead_endpoint_count = len(endpoints) - len(connected_endpoint_ids)

    return out, endpoints, list(connections_by_pair.values()), dead_endpoint_count


def stroke_info_endpoints_for_closure(stroke_infos):
    """Endpoints from split/merged stroke infos, matching cap endpoint graph inputs."""
    endpoints = []
    for s in stroke_infos:
        for p in stroke_endpoint_points(s):
            p = np.asarray(p, dtype=np.float64)
            endpoints.append((int(round(float(p[0]))), int(round(float(p[1])))))
    return endpoints


def connect_endpoint_points_to_nearest(barrier, endpoints, thickness=5):
    """Connect every supplied endpoint point to its nearest other endpoint."""
    out = barrier.copy()
    if len(endpoints) < 2:
        return out, [], len(endpoints)

    connections_by_pair = {}
    for i, p in enumerate(endpoints):
        p_arr = np.asarray(p, dtype=np.float64)
        best_j = None
        best_d2 = float("inf")
        for j, q in enumerate(endpoints):
            if i == j:
                continue
            q_arr = np.asarray(q, dtype=np.float64)
            d2 = float(np.dot(p_arr - q_arr, p_arr - q_arr))
            if d2 < best_d2:
                best_d2 = d2
                best_j = j
        if best_j is None:
            continue
        a, b = sorted((i, best_j))
        connections_by_pair[(a, b)] = (endpoints[a], endpoints[b], math.sqrt(best_d2))

    for p, q, _dist in connections_by_pair.values():
        cv2.line(out, p, q, 255, int(thickness), cv2.LINE_AA)

    connected_endpoint_ids = set()
    for a, b in connections_by_pair:
        connected_endpoint_ids.add(a)
        connected_endpoint_ids.add(b)
    dead_endpoint_count = len(endpoints) - len(connected_endpoint_ids)

    return out, list(connections_by_pair.values()), dead_endpoint_count


def endpoint_threshold_schedule(endpoint_tol, image_shape):
    """Thresholds tried when endpoint closure is not yet one connected region."""
    h, w = image_shape[:2]
    diag = float(math.hypot(w, h))
    base = max(1.0, float(endpoint_tol))
    values = [
        base,
        base * 1.25,
        base * 1.5,
        base * 2.0,
        base * 3.0,
        base * 4.0,
        base * 6.0,
        diag * 0.25,
        diag * 0.35,
    ]
    out = []
    for value in values:
        value = min(float(value), diag)
        if not any(abs(value - old) < 1e-6 for old in out):
            out.append(value)
    return out


def make_input_enclosed_region_mask(stroke_mask, skel_mask=None, stroke_infos=None, endpoint_tol=50.0, close_kernel=5, return_debug=False):
    """Fill areas enclosed by the input sketch; close small gaps before filling."""
    base = (stroke_mask > 0).astype(np.uint8) * 255
    skel = (skel_mask > 0).astype(np.uint8) * 255 if skel_mask is not None else None
    h, w = base.shape
    min_enclosed_area = max(64, int(0.001 * h * w))
    kernels = []
    for k in [close_kernel, 9, 15, 25, 35, 51, 75]:
        k = int(k)
        if k > 1 and k not in kernels:
            kernels.append(k)

    candidates = []
    endpoint_debug = {
        "endpoint_count": 0,
        "endpoint_connection_count": 0,
        "endpoint_tol": float(endpoint_tol),
        "endpoint_tol_used": None,
        "endpoint_thresholds_tried": [],
        "nearest_endpoint_connection_count": 0,
        "dead_endpoint_count": 0,
        "endpoint_source": "skeleton" if stroke_infos is None else "split_stroke_infos",
    }

    if stroke_infos is not None:
        info_endpoints = stroke_info_endpoints_for_closure(stroke_infos)
        nearest_barrier, nearest_connections, dead_endpoint_count = connect_endpoint_points_to_nearest(
            base,
            info_endpoints,
            thickness=max(3, int(close_kernel)),
        )
        enclosed_background = flood_enclosed_regions_from_barrier(nearest_barrier)
        enclosed_area = int(np.count_nonzero(enclosed_background))
        candidates.append({
            "barrier": nearest_barrier,
            "enclosed_background": enclosed_background,
            "enclosed_area": enclosed_area,
            "kernel": None,
            "method": "split_endpoint_nearest_connect",
            "endpoint_tol_used": None,
            "endpoint_connection_count": int(len(nearest_connections)),
            "dead_endpoint_count": int(dead_endpoint_count),
        })
        endpoint_debug["endpoint_count"] = int(len(info_endpoints))
        endpoint_debug["nearest_endpoint_connection_count"] = int(len(nearest_connections))
        endpoint_debug["endpoint_connection_count"] = int(len(nearest_connections))
        endpoint_debug["dead_endpoint_count"] = int(dead_endpoint_count)

    if skel is not None:
        nearest_barrier, endpoints, nearest_connections, dead_endpoint_count = connect_each_endpoint_to_nearest(
            base,
            skel,
            thickness=max(3, int(close_kernel)),
        )
        enclosed_background = flood_enclosed_regions_from_barrier(nearest_barrier)
        enclosed_area = int(np.count_nonzero(enclosed_background))
        candidates.append({
            "barrier": nearest_barrier,
            "enclosed_background": enclosed_background,
            "enclosed_area": enclosed_area,
            "kernel": None,
            "method": "endpoint_nearest_connect",
            "endpoint_tol_used": None,
            "endpoint_connection_count": int(len(nearest_connections)),
            "dead_endpoint_count": int(dead_endpoint_count),
        })

        endpoint_thresholds = endpoint_threshold_schedule(endpoint_tol, base.shape)
        endpoint_debug["endpoint_thresholds_tried"] = [float(x) for x in endpoint_thresholds]
        endpoint_debug["endpoint_count"] = int(len(endpoints))
        endpoint_debug["nearest_endpoint_connection_count"] = int(len(nearest_connections))
        endpoint_debug["dead_endpoint_count"] = int(dead_endpoint_count)

        for tol in endpoint_thresholds:
            barrier, _endpoints, connections = connect_nearby_endpoints(
                base,
                skel,
                endpoint_tol=tol,
                thickness=max(3, int(close_kernel)),
            )
            enclosed_background = flood_enclosed_regions_from_barrier(barrier)
            enclosed_area = int(np.count_nonzero(enclosed_background))
            candidate = {
                "barrier": barrier,
                "enclosed_background": enclosed_background,
                "enclosed_area": enclosed_area,
                "kernel": None,
                "method": "endpoint_connect",
                "endpoint_tol_used": float(tol),
                "endpoint_connection_count": int(len(connections)),
            }
            candidates.append(candidate)

            enclosed_mask = np.zeros_like(barrier)
            enclosed_mask[enclosed_background] = 255
            enclosed_mask[barrier > 0] = 255
            filled = fill_binary_mask(enclosed_mask)
            if enclosed_area >= min_enclosed_area and is_single_connected_mask(filled):
                endpoint_debug["endpoint_tol_used"] = float(tol)
                endpoint_debug["endpoint_connection_count"] = int(len(connections))
                break

    for kernel in kernels:
        barrier = close_stroke_barrier(base, kernel)
        enclosed_background = flood_enclosed_regions_from_barrier(barrier)
        enclosed_area = int(np.count_nonzero(enclosed_background))
        candidates.append({
            "barrier": barrier,
            "enclosed_background": enclosed_background,
            "enclosed_area": enclosed_area,
            "kernel": kernel,
            "method": "already_closed" if kernel == int(close_kernel) else "morph_close",
        })

    valid_candidates = []
    for candidate in candidates:
        enclosed_mask = np.zeros_like(candidate["barrier"])
        enclosed_mask[candidate["enclosed_background"]] = 255
        enclosed_mask[candidate["barrier"] > 0] = 255
        filled = fill_binary_mask(enclosed_mask)
        stats = binary_component_stats(filled)
        candidate["filled"] = filled
        candidate["component_stats"] = stats
        candidate["is_single_connected"] = (
            candidate["enclosed_area"] >= min_enclosed_area
            and is_single_connected_mask(filled)
        )
        if candidate["is_single_connected"]:
            valid_candidates.append(candidate)

    if valid_candidates:
        endpoint_valid = [c for c in valid_candidates if c["method"] == "endpoint_connect"]
        split_nearest_valid = [c for c in valid_candidates if c["method"] == "split_endpoint_nearest_connect" and c.get("dead_endpoint_count", 0) == 0]
        nearest_valid = [c for c in valid_candidates if c["method"] == "endpoint_nearest_connect" and c.get("dead_endpoint_count", 0) == 0]
        if split_nearest_valid:
            best = split_nearest_valid[0]
            endpoint_debug["endpoint_connection_count"] = int(best.get("endpoint_connection_count", 0))
            endpoint_debug["dead_endpoint_count"] = int(best.get("dead_endpoint_count", 0))
        elif nearest_valid:
            # Prefer the closure that explicitly connects every endpoint to its nearest endpoint.
            best = nearest_valid[0]
            endpoint_debug["endpoint_connection_count"] = int(best.get("endpoint_connection_count", 0))
            endpoint_debug["dead_endpoint_count"] = int(best.get("dead_endpoint_count", 0))
        elif endpoint_valid:
            # Use the first endpoint threshold that forms one connected mask.
            best = endpoint_valid[0]
            endpoint_debug["endpoint_tol_used"] = float(best.get("endpoint_tol_used", endpoint_tol))
            endpoint_debug["endpoint_connection_count"] = int(best.get("endpoint_connection_count", 0))
        else:
            best = max(valid_candidates, key=lambda c: c["enclosed_area"])
    elif candidates:
        # If no candidate is single-connected even after increasing thresholds,
        # keep the most connected/filled candidate and report the failure.
        best = max(candidates, key=lambda c: c.get("enclosed_area", 0))
        best["method"] = f"{best['method']}_not_single_connected"
    else:
        hull_mask, hull_barrier = convex_hull_mask_from_strokes(base)
        best = {
            "barrier": hull_barrier,
            "enclosed_background": hull_mask > 0,
            "enclosed_area": int(np.count_nonzero(hull_mask)),
            "kernel": None,
            "method": "convex_hull_fallback",
        }
        best["filled"] = fill_binary_mask(hull_mask)
        best["component_stats"] = binary_component_stats(best["filled"])

    barrier = best["barrier"]
    filled = best["filled"]
    if return_debug:
        return filled, barrier, {
            "method": best["method"],
            "kernel": best["kernel"],
            "enclosed_area": int(np.count_nonzero(filled > 0)),
            "interior_area": int(best["enclosed_area"]),
            "min_enclosed_area": int(min_enclosed_area),
            "component_count": best["component_stats"]["component_count"],
            "component_areas": best["component_stats"]["component_areas"][:10],
            "largest_component_ratio": best["component_stats"]["largest_ratio"],
            "endpoint_source": endpoint_debug["endpoint_source"],
            "endpoint_tol": endpoint_debug["endpoint_tol"],
            "endpoint_tol_used": endpoint_debug["endpoint_tol_used"],
            "endpoint_thresholds_tried": endpoint_debug["endpoint_thresholds_tried"],
            "endpoint_count": endpoint_debug["endpoint_count"],
            "endpoint_connection_count": endpoint_debug["endpoint_connection_count"],
            "nearest_endpoint_connection_count": endpoint_debug["nearest_endpoint_connection_count"],
            "dead_endpoint_count": endpoint_debug["dead_endpoint_count"],
        }
    return filled


def save_preprocess_debug_outputs(debug_dir, img, bw, skel):
    """Stepwise debug output for preprocessing/skeletonization."""
    if debug_dir is None:
        return
    ensure_dir(debug_dir)
    cv2.imwrite(os.path.join(debug_dir, "00_input.png"), img)
    cv2.imwrite(os.path.join(debug_dir, "01_binary.png"), bw)
    cv2.imwrite(os.path.join(debug_dir, "02_skeleton.png"), skel)
    cv2.imwrite(os.path.join(debug_dir, "02b_skeleton_nodes.png"), draw_skeleton_nodes_debug(skel))


def save_input_enclosed_mask_debug_outputs(debug_dir, bw, infos, endpoint_tol=50.0):
    """Save input enclosed mask using split/post-merge stroke endpoints."""
    if debug_dir is None:
        return
    ensure_dir(debug_dir)
    enclosed_mask, closed_strokes, close_info = make_input_enclosed_region_mask(
        bw,
        stroke_infos=infos,
        endpoint_tol=endpoint_tol,
        return_debug=True,
    )
    cv2.imwrite(os.path.join(debug_dir, "00b_input_closed_strokes.png"), closed_strokes)
    cv2.imwrite(os.path.join(debug_dir, "00c_input_enclosed_mask.png"), enclosed_mask)
    with open(os.path.join(debug_dir, "00c_input_enclosed_mask_info.txt"), "w", encoding="utf-8") as f:
        for k, v in close_info.items():
            f.write(f"{k}: {v}\n")


def save_trace_debug_outputs(debug_dir, img_shape, traced_strokes):
    """Stepwise debug output immediately after raw skeleton tracing."""
    if debug_dir is None:
        return
    ensure_dir(debug_dir)
    cv2.imwrite(
        os.path.join(debug_dir, "03a_raw_traced_strokes_before_corner_split.png"),
        draw_strokes_image(img_shape, traced_strokes, thickness=2, annotate=True),
    )


def save_corner_split_debug_outputs(debug_dir, args, img_shape, corner_split_strokes, corner_split_trace):
    """Stepwise debug output immediately after corner splitting."""
    if debug_dir is None:
        return
    ensure_dir(debug_dir)
    cv2.imwrite(
        os.path.join(debug_dir, "03a1_corner_split_before_post_merge.png"),
        draw_strokes_image(img_shape, corner_split_strokes, thickness=2, annotate=True),
    )
    write_corner_split_trace_report(
        os.path.join(debug_dir, "03c_corner_split_trace.txt"),
        corner_split_trace,
        angle_thresh=args.split_corner_angle,
        peak_min_distance=args.split_peak_min_distance,
    )
    write_corner_split_candidates_report(
        os.path.join(debug_dir, "03d_corner_split_candidates.txt"),
        corner_split_trace,
        angle_thresh=args.split_corner_angle,
        peak_min_distance=args.split_peak_min_distance,
    )
    write_corner_split_scan_points_report(
        os.path.join(debug_dir, "03d0_corner_split_scanned_points.txt"),
        corner_split_trace,
        angle_thresh=args.split_corner_angle,
        peak_min_distance=args.split_peak_min_distance,
    )
    cv2.imwrite(
        os.path.join(debug_dir, "03d0_corner_split_scanned_points.png"),
        draw_corner_split_scan_points_image(img_shape, corner_split_trace),
    )
    cv2.imwrite(
        os.path.join(debug_dir, "03d_corner_split_candidates.png"),
        draw_corner_split_candidates_image(img_shape, corner_split_trace),
    )


def save_post_split_merge_debug_outputs(debug_dir, args, img_shape, raw_strokes, post_split_merge_trace):
    """Stepwise debug output immediately after optional post-corner-split merge."""
    if debug_dir is None:
        return
    ensure_dir(debug_dir)
    write_post_split_merge_trace_report(
        os.path.join(debug_dir, "03e_post_split_merge_trace.txt"),
        post_split_merge_trace,
        max_gap=args.post_split_merge_gap,
        max_angle=args.post_split_merge_angle,
        protect_junction_radius=args.post_split_merge_protect_junction_radius,
    )
    cv2.imwrite(
        os.path.join(debug_dir, "03a2_after_post_split_merge.png"),
        draw_strokes_image(img_shape, raw_strokes, thickness=2, annotate=True),
    )


def save_merged_strokes_debug_outputs(debug_dir, img_shape, merged_strokes):
    """Stepwise debug output after optional endpoint merge."""
    if debug_dir is None:
        return
    ensure_dir(debug_dir)
    cv2.imwrite(
        os.path.join(debug_dir, "03b_merged_strokes.png"),
        draw_strokes_image(img_shape, merged_strokes, thickness=2, annotate=True),
    )


def save_stroke_info_debug_outputs(debug_dir, args, img_shape, infos, line_strokes):
    """Stepwise debug output after stroke geometry and line-candidate filtering."""
    if debug_dir is None:
        return
    ensure_dir(debug_dir)
    cv2.imwrite(os.path.join(debug_dir, "04_stroke_info.png"), draw_stroke_infos_image(img_shape, infos, thickness=2))
    cv2.imwrite(os.path.join(debug_dir, "05_line_stroke_candidates.png"), draw_line_stroke_candidates_image(img_shape, line_strokes, thickness=3))
    write_stroke_direction_debug_report(os.path.join(debug_dir, "05a_stroke_directions.txt"), infos, line_strokes)
    write_stroke_direction_debug_json(os.path.join(debug_dir, "05a_stroke_directions.json"), infos, line_strokes)
    cv2.imwrite(os.path.join(debug_dir, "05a_stroke_directions.png"), draw_stroke_directions_image(img_shape, infos, line_strokes))

    direction_group_strokes = [s for s in infos if s["arc"] >= args.min_stroke_length]
    write_direction_groups_debug_report(
        os.path.join(debug_dir, "05c_all_stroke_direction_groups.txt"),
        direction_group_strokes,
        angle_thresh=args.parallel_angle_thresh,
        min_stroke_length=args.min_stroke_length,
    )
    cv2.imwrite(
        os.path.join(debug_dir, "05c_all_stroke_direction_groups.png"),
        draw_direction_groups_image(
            img_shape,
            direction_group_strokes,
            angle_thresh=args.parallel_angle_thresh,
            min_stroke_length=args.min_stroke_length,
        ),
    )


def save_model_debug_outputs(debug_dir, img_shape, model, infos, args):
    """Stepwise debug output after direction model/cap-search details are available."""
    if debug_dir is None:
        return
    ensure_dir(debug_dir)
    save_cluster_debug_outputs(debug_dir, img_shape, model)
    save_cap_endpoint_graph_debug_outputs(debug_dir, img_shape, model, infos, args)
    save_per_cluster_side_cap_outputs(debug_dir, img_shape, model, infos, args)


def save_final_masks_debug_outputs(debug_dir, img_shape, skel, model, side_mask, non_side, candidates):
    """Stepwise debug output after final side/non-side masks and cap candidates exist."""
    if debug_dir is None:
        return
    ensure_dir(debug_dir)
    cv2.imwrite(os.path.join(debug_dir, "06_selected_side_strokes.png"), draw_selected_side_strokes_image(img_shape, model, thickness=4))
    cv2.imwrite(os.path.join(debug_dir, "07_non_side_skeleton.png"), draw_non_side_skeleton_image(skel, side_mask, non_side))
    cv2.imwrite(os.path.join(debug_dir, "08_cap_candidates.png"), draw_cap_candidates_debug(img_shape, candidates, max_draw=8))



def save_debug_outputs(debug_dir, args, img, bw, skel, pre_post_split_merge_strokes, raw_strokes, corner_split_trace, post_split_merge_trace, merged_strokes, infos, line_strokes, model, side_mask, non_side, candidates):
    if debug_dir is None:
        return
    ensure_dir(debug_dir)
    cv2.imwrite(os.path.join(debug_dir, "00_input.png"), img)
    cv2.imwrite(os.path.join(debug_dir, "01_binary.png"), bw)
    cv2.imwrite(os.path.join(debug_dir, "02_skeleton.png"), skel)
    cv2.imwrite(os.path.join(debug_dir, "02b_skeleton_nodes.png"), draw_skeleton_nodes_debug(skel))
    cv2.imwrite(os.path.join(debug_dir, "03a_raw_strokes.png"), draw_strokes_image(img.shape, raw_strokes, thickness=2, annotate=True))
    cv2.imwrite(os.path.join(debug_dir, "03a1_corner_split_before_post_merge.png"), draw_strokes_image(img.shape, pre_post_split_merge_strokes, thickness=2, annotate=True))
    write_corner_split_trace_report(os.path.join(debug_dir, "03c_corner_split_trace.txt"), corner_split_trace, angle_thresh=args.split_corner_angle, peak_min_distance=args.split_peak_min_distance)
    write_corner_split_candidates_report(os.path.join(debug_dir, "03d_corner_split_candidates.txt"), corner_split_trace, angle_thresh=args.split_corner_angle, peak_min_distance=args.split_peak_min_distance)
    write_corner_split_scan_points_report(os.path.join(debug_dir, "03d0_corner_split_scanned_points.txt"), corner_split_trace, angle_thresh=args.split_corner_angle, peak_min_distance=args.split_peak_min_distance)
    cv2.imwrite(
        os.path.join(debug_dir, "03d0_corner_split_scanned_points.png"),
        draw_corner_split_scan_points_image(img.shape, corner_split_trace),
    )
    cv2.imwrite(
        os.path.join(debug_dir, "03d_corner_split_candidates.png"),
        draw_corner_split_candidates_image(img.shape, corner_split_trace),
    )
    write_post_split_merge_trace_report(
        os.path.join(debug_dir, "03e_post_split_merge_trace.txt"),
        post_split_merge_trace,
        max_gap=args.post_split_merge_gap,
        max_angle=args.post_split_merge_angle,
        protect_junction_radius=args.post_split_merge_protect_junction_radius,
    )
    cv2.imwrite(os.path.join(debug_dir, "03b_merged_strokes.png"), draw_strokes_image(img.shape, merged_strokes, thickness=2, annotate=True))
    cv2.imwrite(os.path.join(debug_dir, "04_stroke_info.png"), draw_stroke_infos_image(img.shape, infos, thickness=2))
    cv2.imwrite(os.path.join(debug_dir, "05_line_stroke_candidates.png"), draw_line_stroke_candidates_image(img.shape, line_strokes, thickness=3))
    write_stroke_direction_debug_report(os.path.join(debug_dir, "05a_stroke_directions.txt"), infos, line_strokes)
    write_stroke_direction_debug_json(os.path.join(debug_dir, "05a_stroke_directions.json"), infos, line_strokes)
    cv2.imwrite(os.path.join(debug_dir, "05a_stroke_directions.png"), draw_stroke_directions_image(img.shape, infos, line_strokes))
    direction_group_strokes = [s for s in infos if s["arc"] >= args.min_stroke_length]
    write_direction_groups_debug_report(
        os.path.join(debug_dir, "05c_all_stroke_direction_groups.txt"),
        direction_group_strokes,
        angle_thresh=args.parallel_angle_thresh,
        min_stroke_length=args.min_stroke_length,
    )
    cv2.imwrite(
        os.path.join(debug_dir, "05c_all_stroke_direction_groups.png"),
        draw_direction_groups_image(
            img.shape,
            direction_group_strokes,
            angle_thresh=args.parallel_angle_thresh,
            min_stroke_length=args.min_stroke_length,
        ),
    )
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
    parser.add_argument("--split-corner-angle", type=float, default=None,
                        help="Accepted for command compatibility. Corner-based stroke splitting is not applied in this version.")
    parser.add_argument("--split-segment-arc", type=float, default=50.0,
                        help="Max arc length sampled on each side for PCA segment-angle validation in corner splitting.")
    parser.add_argument("--split-peak-min-distance", type=float, default=10.0,
                        help="Minimum stroke-arc pixel distance between multiple accepted corner split peaks.")
    parser.add_argument("--split-optimize-max-iters", type=int, default=5,
                        help="Maximum optimization iterations for selecting corner split candidates.")
    parser.add_argument("--split-segment-window", type=int, default=None,
                        help="Deprecated compatibility option; use --split-segment-arc instead.")
    parser.add_argument("--disable-post-split-merge", action="store_true",
                        help="Disable automatic merge of nearly collinear fragments created by corner splitting.")
    parser.add_argument("--post-split-merge-gap", type=float, default=3.0,
                        help="Max endpoint gap for merging accidental post-corner-split fragments.")
    parser.add_argument("--post-split-merge-angle", type=float, default=12.0,
                        help="Max PCA axis angle difference for merging accidental post-corner-split fragments.")
    parser.add_argument("--post-split-merge-protect-junction-radius", type=float, default=3.0,
                        help="Reject a post-corner-split merge when any third stroke endpoint, including short strokes, lies within this radius of the proposed merge point. 0 disables this protection.")
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
    parser.add_argument(
        "--copy-side-iou-compare-percent",
        type=float,
        default=None,
        help=(
            "Percent of longest side strokes to compare by cap-sweep IoU when the longest side "
            "is more angle-deviated from the cluster mean than the second longest. "
            "The count is ceil(n * percent / 100), with a minimum of 2 when enabled. "
            "Omit to keep the default behavior of comparing the two longest strokes; 0 disables this comparison."
        ),
    )

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

    parser.add_argument("--skeleton-gap-tol", type=float, default=0.0,
                        help="Before stroke tracing, connect mutual-nearest skeleton endpoints within this gap tolerance and remove unmatched dangling branches. 0 disables skeleton cleanup.")
    parser.add_argument("--skeleton-small-loop-bbox-area-thresh", type=float, default=0.0,
                        help="After skeleton gap connection, remove only newly added gap edges that close a loop whose loop-pixel bbox area is below this threshold. 0 disables this small-loop cleanup.")
    parser.add_argument("--skeleton-branch-prune-max-pixels", type=float, default=0.0,
                        help="Maximum traced pixels for deleting a 02c3 endpoint-started dangling branch. 0 uses an automatic max(30, 3*skeleton-gap-tol) when small-loop cleanup is enabled; use a large value to disable this guard.")
    parser.add_argument("--side-thickness", type=int, default=4)
    parser.add_argument("--min-cap-pixels", type=int, default=40)
    parser.add_argument("--min-cap-enclosed-area", type=int, default=0,
                        help="Reject cap candidates whose estimated filled/enclosed area is smaller than this value.")
    parser.add_argument("--min-cap-total-arc", type=float, default=0.0,
                        help="Reject cap candidates whose total stroke arc length is smaller than this value.")
    parser.add_argument("--cap-loop-endpoint-tol", type=float, default=12.0,
                        help="Endpoint tolerance for deciding whether remaining strokes form a closed cap loop.")
    parser.add_argument("--cap-loop-thickness", type=int, default=2,
                        help="Raster thickness used to draw loop-based cap candidate masks.")
    parser.add_argument("--cap-loop-max-subset-size", type=int, default=14,
                        help="Deprecated compatibility option. Cap detection now checks whole connected components only; no loop subset enumeration is used.")
    parser.add_argument("--cap-subgroup-max-removals", type=int, default=-1,
                        help="Max remove-k subgroup depth for direction groups that have no valid cap. Use -1 to continue until one stroke remains.")

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
    save_preprocess_debug_outputs(args.debug_dir, img, bw, skel)

    if args.skeleton_gap_tol > 0.0:
        skel_before_cleanup = skel.copy()
        skel_after_connect, skel_after_small_loop_prune, skel_after_prune, cleanup_info = cleanup_skeleton_endpoints(
            skel,
            gap_tol=args.skeleton_gap_tol,
            connect_thickness=1,
            small_loop_bbox_area_thresh=args.skeleton_small_loop_bbox_area_thresh,
            branch_prune_max_pixels=args.skeleton_branch_prune_max_pixels,
        )
        skel = skel_after_prune
        save_skeleton_cleanup_debug_outputs(
            args.debug_dir,
            skel_before_cleanup,
            skel_after_connect,
            skel_after_small_loop_prune,
            skel_after_prune,
            cleanup_info,
        )

    traced_strokes = trace_strokes(skel, min_pixels=args.trace_min_pixels)
    save_trace_debug_outputs(args.debug_dir, img.shape, traced_strokes)

    raw_strokes, corner_split_trace = split_strokes_at_corners_with_trace(
        traced_strokes,
        angle_thresh=args.split_corner_angle,
        min_pixels=args.trace_min_pixels,
        segment_arc=args.split_segment_arc,
        split_peak_min_distance=args.split_peak_min_distance,
        split_optimize_max_iters=args.split_optimize_max_iters,
    )
    pre_post_split_merge_strokes = [s.copy() for s in raw_strokes]
    save_corner_split_debug_outputs(
        args.debug_dir,
        args,
        img.shape,
        pre_post_split_merge_strokes,
        corner_split_trace,
    )

    if args.disable_post_split_merge:
        post_split_merge_trace = []
    else:
        raw_strokes, post_split_merge_trace = merge_post_corner_split_strokes(
            raw_strokes,
            max_gap=args.post_split_merge_gap,
            max_angle=args.post_split_merge_angle,
            protect_junction_radius=args.post_split_merge_protect_junction_radius,
        )
    save_post_split_merge_debug_outputs(
        args.debug_dir,
        args,
        img.shape,
        raw_strokes,
        post_split_merge_trace,
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
    save_merged_strokes_debug_outputs(args.debug_dir, img.shape, merged_strokes)

    infos = build_stroke_infos(merged_strokes)
    save_input_enclosed_mask_debug_outputs(args.debug_dir, bw, infos, endpoint_tol=args.cap_loop_endpoint_tol)
    step_line_strokes = [
        s for s in infos
        if s["arc"] >= args.min_stroke_length and s["straightness"] >= args.straightness
    ]
    save_stroke_info_debug_outputs(args.debug_dir, args, img.shape, infos, step_line_strokes)
    model, line_strokes = choose_extrusion_model(args, infos, img.shape, skel)
    save_stroke_info_debug_outputs(args.debug_dir, args, img.shape, infos, line_strokes)

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
    #cap_pool_infos = [s for s in infos if s["arc"] >= args.min_stroke_length]
    cap_pool_infos = [s for s in infos if s["arc"] >= 10]
    model, candidates = validate_side_clusters_by_cap_candidates(
        model,
        infos,
        skel.shape,
        endpoint_tol=args.cap_loop_endpoint_tol,
        min_pixels=args.min_cap_pixels,
        min_enclosed_area=args.min_cap_enclosed_area,
        min_total_arc=args.min_cap_total_arc,
        thickness=args.cap_loop_thickness,
        max_loop_subset_size=args.cap_loop_max_subset_size,
        max_subgroup_removals=args.cap_subgroup_max_removals,
        cap_pool_infos=cap_pool_infos,
    )
    save_model_debug_outputs(args.debug_dir, img.shape, model, infos, args)

    side_mask = make_side_mask(skel, model, side_thickness=args.side_thickness)
    non_side = cv2.bitwise_and(skel, cv2.bitwise_not(side_mask))
    save_final_masks_debug_outputs(args.debug_dir, img.shape, skel, model, side_mask, non_side, candidates)

    save_debug_outputs(
        args.debug_dir, args, img, bw, skel, pre_post_split_merge_strokes, raw_strokes, corner_split_trace, post_split_merge_trace, merged_strokes,
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
