# extrusion_debug_caploop.py
# Robust debug version for sketch extrusion detection.
# Key fix: trace_strokes uses crossing number instead of raw degree!=2,
# avoiding false junctions caused by 8-neighbor stair-step skeleton artifacts.

import argparse
import base64
from datetime import datetime
import json
import math
import os
import re
import shutil
import time

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


def line_distance_values(points, line):
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) == 0:
        return np.zeros(0, dtype=np.float64)
    a, b, c = np.asarray(line, dtype=np.float64)
    return np.abs(a * pts[:, 0] + b * pts[:, 1] + c)


def chord_line_from_points(points):
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 2:
        return None
    p0 = pts[0]
    p1 = pts[-1]
    v = p1 - p0
    n = np.linalg.norm(v)
    if n < 1e-8:
        return None
    normal = np.array([-v[1], v[0]], dtype=np.float64) / n
    c = -float(np.dot(normal, p0))
    return np.array([normal[0], normal[1], c], dtype=np.float64)


def stroke_linearity_metrics(points, pca_line=None):
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) == 0:
        return {
            "p90_pca_line_error": 0.0,
            "pca_rms_error": 0.0,
            "p90_chord_deviation": 0.0,
            "chord_deviation_ratio": float("inf"),
        }

    if pca_line is None:
        pca_line, _, _ = fit_line_to_points(pts)

    pca_dist = line_distance_values(pts, pca_line)
    p90_pca = float(np.percentile(pca_dist, 90)) if len(pca_dist) else 0.0
    pca_rms = float(np.sqrt(np.mean(pca_dist * pca_dist))) if len(pca_dist) else 0.0

    chord = stroke_chord_length(pts)
    chord_line = chord_line_from_points(pts)
    if chord_line is None or chord < 1e-8:
        p90_chord = 0.0
        chord_ratio = float("inf")
    else:
        chord_dist = line_distance_values(pts, chord_line)
        p90_chord = float(np.percentile(chord_dist, 90)) if len(chord_dist) else 0.0
        chord_ratio = float(p90_chord / chord)

    return {
        "p90_pca_line_error": p90_pca,
        "pca_rms_error": pca_rms,
        "p90_chord_deviation": p90_chord,
        "chord_deviation_ratio": chord_ratio,
    }


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
        linearity = stroke_linearity_metrics(pts, pca_line=line)
        info = {
            "index": i,
            "points": pts,
            "arc": arc,
            "chord": chord,
            "straightness": straight,
            "line": line,
            "direction": direction,
            "center": center,
        }
        info.update(linearity)
        infos.append(info)
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

    For a side-direction cluster, adjacent/connected strokes can be useful debug
    evidence, but current scoring does not reject or penalize connected pairs.
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

    Active scoring is intentionally minimal:
      - reward only the number of strokes in the direction family;
      - penalize same-loop endpoint pairs.

    Connected-pair, length, straightness, perpendicular spread, and
    length-similarity metrics are still recorded for debug output, but they do
    not affect the score.
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
    length_similarity_bonus = 0.0
    spread_penalty = 0.0
    count_term = float(count_weight * len(cluster))
    same_loop_penalty_term = float(same_loop_penalty_weight * same_loop_pairs)

    invalid_connected_cluster = False
    score = count_term - same_loop_penalty_term

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
        "score_count_weight": float(count_weight),
        "score_same_loop_penalty_weight": float(same_loop_penalty_weight),
        "score_same_loop_endpoint_tol": float(same_loop_endpoint_tol),
        "score_count_term": count_term,
        "score_same_loop_penalty_term": same_loop_penalty_term,
        "score_uses_only_count_same_loop": True,
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


def side_line_error_limit(chord, abs_px, ratio):
    limits = []
    if abs_px is not None and float(abs_px) > 0.0:
        limits.append(float(abs_px))
    if ratio is not None and float(ratio) > 0.0:
        limits.append(float(chord) * float(ratio))
    if not limits:
        return None
    return float(max(limits))


def side_direction_group_filter_result(
    s,
    min_length,
    min_straightness,
    min_chord=0.0,
    line_p90_error_px=4.0,
    line_p90_error_ratio=0.035,
    line_rms_error_px=2.5,
    line_rms_error_ratio=0.025,
    chord_dev_ratio_max=0.08,
):
    """Return side-line prefilter diagnostics for one stroke."""
    arc = float(s.get("arc", 0.0))
    chord = float(s.get("chord", 0.0))
    straightness = float(s.get("straightness", 0.0))
    p90_pca = float(s.get("p90_pca_line_error", float("inf")))
    pca_rms = float(s.get("pca_rms_error", float("inf")))
    chord_ratio = float(s.get("chord_deviation_ratio", float("inf")))
    p90_limit = side_line_error_limit(chord, line_p90_error_px, line_p90_error_ratio)
    rms_limit = side_line_error_limit(chord, line_rms_error_px, line_rms_error_ratio)

    reasons = []
    if arc < float(min_length):
        reasons.append(f"arc {arc:.2f} < {float(min_length):.2f}")
    if chord < float(min_chord):
        reasons.append(f"chord {chord:.2f} < {float(min_chord):.2f}")
    if straightness < float(min_straightness):
        reasons.append(f"straightness {straightness:.3f} < {float(min_straightness):.3f}")
    if p90_limit is not None and p90_pca > p90_limit:
        reasons.append(f"p90_pca_line_error {p90_pca:.2f} > {p90_limit:.2f}")
    if rms_limit is not None and pca_rms > rms_limit:
        reasons.append(f"pca_rms_error {pca_rms:.2f} > {rms_limit:.2f}")
    if chord_dev_ratio_max is not None and float(chord_dev_ratio_max) > 0.0:
        if chord_ratio > float(chord_dev_ratio_max):
            reasons.append(f"chord_dev_ratio {chord_ratio:.4f} > {float(chord_dev_ratio_max):.4f}")

    failed_checks = {
        "arc": arc < float(min_length),
        "chord": chord < float(min_chord),
        "straightness": straightness < float(min_straightness),
        "p90_pca_line_error": p90_limit is not None and p90_pca > p90_limit,
        "pca_rms_error": rms_limit is not None and pca_rms > rms_limit,
        "chord_deviation_ratio": (
            chord_dev_ratio_max is not None
            and float(chord_dev_ratio_max) > 0.0
            and chord_ratio > float(chord_dev_ratio_max)
        ),
    }

    return {
        "accepted": len(reasons) == 0,
        "reasons": reasons,
        "p90_pca_limit": p90_limit,
        "pca_rms_limit": rms_limit,
        "failed_checks": failed_checks,
    }


def filter_side_direction_group_strokes(
    stroke_infos,
    min_length,
    min_straightness,
    min_chord=0.0,
    line_p90_error_px=4.0,
    line_p90_error_ratio=0.035,
    line_rms_error_px=2.5,
    line_rms_error_ratio=0.025,
    chord_dev_ratio_max=0.08,
):
    """Return strokes eligible for side-direction clustering."""
    kept = []
    for s in stroke_infos:
        result = side_direction_group_filter_result(
            s,
            min_length=min_length,
            min_straightness=min_straightness,
            min_chord=min_chord,
            line_p90_error_px=line_p90_error_px,
            line_p90_error_ratio=line_p90_error_ratio,
            line_rms_error_px=line_rms_error_px,
            line_rms_error_ratio=line_rms_error_ratio,
            chord_dev_ratio_max=chord_dev_ratio_max,
        )
        if result["accepted"]:
            kept.append(s)
    return kept


def filter_side_direction_group_strokes_for_args(stroke_infos, args):
    return filter_side_direction_group_strokes(
        stroke_infos,
        min_length=args.min_stroke_length,
        min_straightness=args.side_straightness,
        min_chord=args.side_min_chord_px,
        line_p90_error_px=args.side_line_p90_error_px,
        line_p90_error_ratio=args.side_line_p90_error_ratio,
        line_rms_error_px=args.side_line_rms_error_px,
        line_rms_error_ratio=args.side_line_rms_error_ratio,
        chord_dev_ratio_max=args.side_chord_dev_ratio_max,
    )


def choose_extrusion_model(args, infos, img_shape, skel):
    line_strokes = [s for s in infos if s["arc"] >= args.min_stroke_length and s["straightness"] >= args.straightness]
    direction_group_strokes = filter_side_direction_group_strokes_for_args(infos, args)

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


CAP_POOL_MIN_ARC = 10.0
CAP_POOL_SHORT_STROKE_KEEP_CONNECTED_GT = 2


def endpoint_neighbor_stroke_indices(
    strokes,
    stroke_local_i,
    endpoint_i,
    endpoint_tol=12.0,
    nearest_matches=None,
    reverse_matches=None,
):
    """
    Return distinct other stroke ids touching one endpoint in the nearest-match graph.

    This is intentionally narrower than counting every endpoint within endpoint_tol.
    We count only endpoint pairs that participate in the current nearest-neighbor
    endpoint matching used elsewhere in cap-loop component construction.
    """
    stroke_local_i = int(stroke_local_i)
    endpoint_i = int(endpoint_i)
    if stroke_local_i < 0 or stroke_local_i >= len(strokes):
        return []

    if nearest_matches is None:
        nearest_matches = build_nearest_endpoint_matches(strokes, endpoint_tol=endpoint_tol)
    if reverse_matches is None:
        reverse_matches = {}
        for src_key, dst_key in nearest_matches.items():
            reverse_matches.setdefault(dst_key, []).append(src_key)

    target_key = (stroke_local_i, endpoint_i)
    neighbor_ids = set()
    touched_keys = []
    match_key = nearest_matches.get(target_key, None)
    if match_key is not None:
        touched_keys.append(match_key)
    touched_keys.extend(reverse_matches.get(target_key, []))

    for other_local_i, _other_endpoint_i in touched_keys:
        other_local_i = int(other_local_i)
        if other_local_i == stroke_local_i:
            continue
        neighbor_ids.add(int(strokes[other_local_i].get("index", other_local_i)))
    return sorted(neighbor_ids)


def short_stroke_connected_neighbor_info(strokes, stroke_local_i, endpoint_tol=12.0):
    """Return per-endpoint and union neighbor-stroke ids for one short stroke."""
    nearest_matches = build_nearest_endpoint_matches(strokes, endpoint_tol=endpoint_tol)
    reverse_matches = {}
    for src_key, dst_key in nearest_matches.items():
        reverse_matches.setdefault(dst_key, []).append(src_key)

    start_neighbors = endpoint_neighbor_stroke_indices(
        strokes,
        stroke_local_i,
        0,
        endpoint_tol=endpoint_tol,
        nearest_matches=nearest_matches,
        reverse_matches=reverse_matches,
    )
    end_neighbors = endpoint_neighbor_stroke_indices(
        strokes,
        stroke_local_i,
        1,
        endpoint_tol=endpoint_tol,
        nearest_matches=nearest_matches,
        reverse_matches=reverse_matches,
    )
    connected_ids = sorted(set(start_neighbors) | set(end_neighbors))
    return {
        "start_neighbors": start_neighbors,
        "end_neighbors": end_neighbors,
        "connected_stroke_indices": connected_ids,
        "connected_stroke_count": int(len(connected_ids)),
    }


def build_cap_pool_infos(
    strokes,
    min_arc=CAP_POOL_MIN_ARC,
    endpoint_tol=12.0,
    keep_short_if_connected_gt=CAP_POOL_SHORT_STROKE_KEEP_CONNECTED_GT,
):
    """
    Keep all strokes with arc >= min_arc.

    For shorter strokes, keep them only when the union of other strokes touching
    either endpoint in the nearest-match graph within endpoint_tol contains more than
    keep_short_if_connected_gt strokes. This preserves short junction bridges in
    the cap pool while still filtering isolated tiny fragments.
    """
    cap_pool_infos = []
    for stroke_local_i, stroke in enumerate(strokes):
        arc = float(stroke.get("arc", 0.0))
        if arc >= float(min_arc):
            cap_pool_infos.append(stroke)
            continue
        neighbor_info = short_stroke_connected_neighbor_info(
            strokes,
            stroke_local_i,
            endpoint_tol=endpoint_tol,
        )
        if int(neighbor_info.get("connected_stroke_count", 0)) > int(keep_short_if_connected_gt):
            cap_pool_infos.append(stroke)
    return cap_pool_infos

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
    endpoint within endpoint_tol.  The nearest-neighbor search is recomputed
    inside the current component, so pruning an open branch can expose the next
    nearest endpoint and preserve a valid closed sub-loop.
    """
    comp = [int(i) for i in comp]
    stroke_local_i = int(stroke_local_i)
    endpoint_i = int(endpoint_i)
    comp_set = set(comp)
    if stroke_local_i not in comp_set:
        return 0, []

    if len(comp) == 1:
        p0, p1 = stroke_endpoint_points(strokes[stroke_local_i])
        if endpoint_i == 0 and endpoint_points_close(p0, p1, endpoint_tol):
            return 1, [(int(strokes[stroke_local_i]["index"]), 1)]
        if endpoint_i == 1 and endpoint_points_close(p1, p0, endpoint_tol):
            return 1, [(int(strokes[stroke_local_i]["index"]), 0)]
        return 0, []

    global_to_component_i = {global_i: component_i for component_i, global_i in enumerate(comp)}
    component_i = global_to_component_i[stroke_local_i]
    component_strokes = [strokes[global_i] for global_i in comp]
    nearest_matches = build_nearest_endpoint_matches(component_strokes, endpoint_tol=endpoint_tol)
    match = nearest_matches.get((component_i, endpoint_i), None)
    if match is None:
        return 0, []
    matched_global_i = comp[int(match[0])]
    return 1, [(int(strokes[matched_global_i]["index"]), int(match[1]))]


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


def normalize_cap_loop_component(strokes, comp, endpoint_tol=12.0):
    """
    Alternate open-branch pruning and self-loop removal until the component stabilizes.

    Removing a self-loop can expose new open ends, so one prune pass is not enough.
    This helper keeps iterating until neither phase removes any more strokes.
    """
    current = sorted(set(int(i) for i in comp))
    removed_open_local = []
    removed_self_local = []
    trace = []

    while current:
        pruned_comp, removed_branch = prune_open_branches_from_component(
            strokes,
            current,
            endpoint_tol=endpoint_tol,
        )
        current = list(map(int, pruned_comp))
        removed_branch = list(map(int, removed_branch))
        if removed_branch:
            removed_open_local.extend(removed_branch)

        closed_after_prune = bool(current) and is_closed_stroke_component_by_endpoint_proximity(
            strokes,
            current,
            endpoint_tol=endpoint_tol,
        )
        trace.append({
            "phase": "prune_open_branches",
            "remaining_strokes": [int(strokes[i]["index"]) for i in current],
            "removed_open_branch_strokes": [int(strokes[i]["index"]) for i in removed_branch],
            "closed": bool(closed_after_prune),
        })
        if not current or not closed_after_prune:
            break

        post_loop_comp, removed_self_loops = remove_post_loop_self_loops(
            strokes,
            current,
            endpoint_tol=endpoint_tol,
        )
        current = list(map(int, post_loop_comp))
        removed_self_loops = list(map(int, removed_self_loops))
        if removed_self_loops:
            removed_self_local.extend(removed_self_loops)

        closed_after_self_loop = bool(current) and is_closed_stroke_component_by_endpoint_proximity(
            strokes,
            current,
            endpoint_tol=endpoint_tol,
        )
        trace.append({
            "phase": "remove_post_loop_self_loops",
            "remaining_strokes": [int(strokes[i]["index"]) for i in current],
            "removed_post_loop_self_strokes": [int(strokes[i]["index"]) for i in removed_self_loops],
            "closed": bool(closed_after_self_loop),
        })
        if not removed_self_loops:
            break

    final_closed = bool(current) and is_closed_stroke_component_by_endpoint_proximity(
        strokes,
        current,
        endpoint_tol=endpoint_tol,
    )
    return {
        "component_local_indices": list(map(int, current)),
        "component_stroke_indices": [int(strokes[i]["index"]) for i in current],
        "closed": bool(final_closed),
        "removed_open_branch_local_indices": removed_open_local,
        "removed_open_branch_strokes": [int(strokes[i]["index"]) for i in removed_open_local],
        "removed_post_loop_self_local_indices": removed_self_local,
        "removed_post_loop_self_strokes": [int(strokes[i]["index"]) for i in removed_self_local],
        "trace": trace,
    }


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


def build_cap_component_endpoint_graph(strokes, comp, endpoint_tol=12.0):
    """Build an endpoint-node graph for a normalized cap component."""
    comp = list(map(int, comp))
    comp_strokes = [strokes[i] for i in comp]
    endpoint_nodes, centers = stroke_endpoint_node_ids(comp_strokes, endpoint_tol=endpoint_tol)

    edges = []
    nonself_edges = []
    self_loop_strokes = []
    for pos, (a, b) in enumerate(endpoint_nodes):
        edge = {
            "edge_index": int(pos),
            "stroke_local_i": int(comp[pos]),
            "stroke_index": int(strokes[comp[pos]]["index"]),
            "start_node": int(a),
            "end_node": int(b),
            "is_self_loop": int(a) == int(b),
        }
        edges.append(edge)
        if edge["is_self_loop"]:
            self_loop_strokes.append(edge["stroke_index"])
        else:
            nonself_edges.append(edge)

    return {
        "component_local_indices": comp,
        "endpoint_centers": centers,
        "edges": edges,
        "nonself_edges": nonself_edges,
        "self_loop_strokes": self_loop_strokes,
    }


def cap_graph_edge_components(nonself_edges):
    """Connected components over graph edges."""
    node_to_edges = {}
    for ei, edge in enumerate(nonself_edges):
        node_to_edges.setdefault(int(edge["start_node"]), []).append(ei)
        node_to_edges.setdefault(int(edge["end_node"]), []).append(ei)

    visited = set()
    comps = []
    for start in range(len(nonself_edges)):
        if start in visited:
            continue
        stack = [start]
        visited.add(start)
        comp = []
        while stack:
            ei = stack.pop()
            comp.append(ei)
            edge = nonself_edges[ei]
            for node in (int(edge["start_node"]), int(edge["end_node"])):
                for nb in node_to_edges.get(node, []):
                    if nb not in visited:
                        visited.add(nb)
                        stack.append(nb)
        comps.append(sorted(comp))
    return comps


def cap_graph_component_stats(nonself_edges, edge_indices):
    node_degree = {}
    nodes = set()
    for ei in edge_indices:
        edge = nonself_edges[int(ei)]
        a = int(edge["start_node"])
        b = int(edge["end_node"])
        nodes.add(a)
        nodes.add(b)
        node_degree[a] = node_degree.get(a, 0) + 1
        node_degree[b] = node_degree.get(b, 0) + 1
    edge_count = int(len(edge_indices))
    node_count = int(len(nodes))
    cycle_count = max(0, edge_count - node_count + 1) if edge_count > 0 else 0
    return {
        "edge_count": edge_count,
        "node_count": node_count,
        "cycle_count": int(cycle_count),
        "node_degrees": {int(k): int(v) for k, v in sorted(node_degree.items())},
        "all_degree_two": bool(node_degree) and all(int(v) == 2 for v in node_degree.values()),
        "is_simple_cycle": edge_count > 0 and edge_count == node_count and all(int(v) == 2 for v in node_degree.values()),
    }


def enumerate_simple_edge_cycles(nonself_edges, max_cycle_edges=None, max_cycles=512):
    """Enumerate simple cycles as edge-index sets in a small undirected multigraph."""
    if not nonself_edges:
        return []

    max_cycle_edges = len(nonself_edges) if not max_cycle_edges else int(max_cycle_edges)
    max_cycle_edges = max(2, min(max_cycle_edges, len(nonself_edges)))

    adjacency = {}
    nodes = set()
    for ei, edge in enumerate(nonself_edges):
        a = int(edge["start_node"])
        b = int(edge["end_node"])
        nodes.add(a)
        nodes.add(b)
        adjacency.setdefault(a, []).append((ei, b))
        adjacency.setdefault(b, []).append((ei, a))
    for items in adjacency.values():
        items.sort(key=lambda item: (item[1], item[0]))

    seen = set()
    cycles = []

    def dfs(start, node, visited_nodes, path_edges):
        if len(cycles) >= int(max_cycles):
            return
        if len(path_edges) >= max_cycle_edges:
            return

        for edge_idx, nb in adjacency.get(node, []):
            if edge_idx in path_edges:
                continue
            if nb == start:
                if len(path_edges) >= 1:
                    key = tuple(sorted(path_edges + [edge_idx]))
                    if len(key) >= 2 and key not in seen:
                        seen.add(key)
                        cycles.append(list(key))
                continue
            if nb < start or nb in visited_nodes:
                continue
            dfs(start, nb, visited_nodes | {nb}, path_edges + [edge_idx])

    for start in sorted(nodes):
        dfs(start, start, {start}, [])
        if len(cycles) >= int(max_cycles):
            break

    cycles.sort(key=lambda c: (-len(c), c))
    return cycles


def find_edge_disjoint_cycle_cover(nonself_edges, cycles):
    """Return a set of cycles covering every edge exactly once, if one exists."""
    universe = set(range(len(nonself_edges)))
    if not universe:
        return []

    cycle_sets = [set(map(int, c)) for c in cycles]
    edge_to_cycles = {}
    for ci, cycle in enumerate(cycle_sets):
        for edge_idx in cycle:
            edge_to_cycles.setdefault(edge_idx, []).append(ci)

    for edge_idx in universe:
        if edge_idx not in edge_to_cycles:
            return None

    def search(remaining, chosen):
        if not remaining:
            return chosen
        edge_idx = min(remaining, key=lambda e: len(edge_to_cycles.get(e, [])))
        for cycle_idx in edge_to_cycles.get(edge_idx, []):
            cycle = cycle_sets[cycle_idx]
            if not cycle.issubset(remaining):
                continue
            result = search(remaining - cycle, chosen + [cycle_idx])
            if result is not None:
                return result
        return None

    chosen_indices = search(universe, [])
    if chosen_indices is None:
        return None
    return [cycles[i] for i in chosen_indices]


def cap_loop_components_from_topology(strokes, comp, endpoint_tol=12.0, max_loop_subset_size=14):
    """
    Return one or more cap loop components after endpoint-graph topology checks.

    Multiple simple loops may be kept together when their edges can be covered by
    edge-disjoint simple cycles. If the graph contains shared-edge cycles, return
    the individual simple cycles instead of the whole component.
    """
    graph = build_cap_component_endpoint_graph(strokes, comp, endpoint_tol=endpoint_tol)
    nonself_edges = graph["nonself_edges"]
    edge_components = cap_graph_edge_components(nonself_edges)
    component_stats = [cap_graph_component_stats(nonself_edges, c) for c in edge_components]
    cycle_count = int(sum(item["cycle_count"] for item in component_stats))
    max_cycle_edges = len(nonself_edges)
    cycles = enumerate_simple_edge_cycles(nonself_edges, max_cycle_edges=max_cycle_edges)
    cycle_cover = None
    if not graph["self_loop_strokes"] and cycles:
        cycle_cover = find_edge_disjoint_cycle_cover(nonself_edges, cycles)

    topology = {
        "nonself_edge_count": int(len(nonself_edges)),
        "self_loop_strokes": list(graph["self_loop_strokes"]),
        "edge_component_count": int(len(edge_components)),
        "cycle_count": int(cycle_count),
        "simple_cycle_count": int(len(cycles)),
        "edge_disjoint_cycle_cover_found": cycle_cover is not None,
        "edge_disjoint_cycle_cover_count": 0 if cycle_cover is None else int(len(cycle_cover)),
        "component_stats": component_stats,
    }

    if cycle_cover is not None:
        kind = "simple_loop" if len(cycle_cover) == 1 else "multi_simple_loop_edge_disjoint"
        return [{
            "component_local_indices": list(map(int, comp)),
            "topology_kind": kind,
            "topology": topology,
        }]

    out = []
    seen = set()
    max_subset_size = int(max_loop_subset_size) if max_loop_subset_size is not None else 0
    for cycle in cycles:
        loop_local_indices = [int(nonself_edges[ei]["stroke_local_i"]) for ei in cycle]
        if max_subset_size > 0 and len(loop_local_indices) > max_subset_size:
            continue
        key = tuple(sorted(loop_local_indices))
        if key in seen:
            continue
        seen.add(key)
        cycle_strokes = [int(strokes[i]["index"]) for i in loop_local_indices]
        cycle_topology = dict(topology)
        cycle_topology["decomposed_from_shared_edge_graph"] = True
        cycle_topology["cycle_strokes"] = cycle_strokes
        out.append({
            "component_local_indices": loop_local_indices,
            "topology_kind": "decomposed_simple_loop",
            "topology": cycle_topology,
        })
    return out


def candidate_from_stroke_loop(shape, strokes, comp, thickness=2):
    """Create a cap-candidate record from a closed stroke-loop component."""
    loop_strokes = [strokes[i] for i in comp]
    mask = make_stroke_mask(shape, loop_strokes, thickness=thickness)
    enclosed_area, enclosed_mask = estimate_enclosed_area_from_loop_mask(mask)
    pts = np.vstack([s["points"] for s in loop_strokes])
    cap_pixels_mask = ((mask > 0) | (enclosed_mask > 0)).astype(np.uint8)
    ys, xs = np.where(cap_pixels_mask > 0)
    bbox_meta = bbox_for_points(list(zip(xs.tolist(), ys.tolist())))
    center = pts.mean(axis=0).astype(np.float64)
    area = int(np.count_nonzero(mask))
    total_arc = float(sum(s["arc"] for s in loop_strokes))
    return {
        "mask": mask,
        "enclosed_mask": enclosed_mask,
        "area": area,
        "enclosed_area": int(enclosed_area),
        "bbox": None if bbox_meta is None else bbox_meta.get("bbox"),
        "bbox_area": 0 if bbox_meta is None else int(bbox_meta.get("bbox_area", 0)),
        "endpoints": 0,
        "closedness": 1.0,
        "score": float(enclosed_area + area + total_arc),
        "center": center,
        "stroke_indices": [int(s["index"]) for s in loop_strokes],
        "stroke_count": len(loop_strokes),
        "total_arc": total_arc,
    }


def canonical_cycle_sequence(seq):
    """Canonicalize a stroke-index cycle up to rotation and reversal."""
    seq = [int(x) for x in seq]
    if not seq:
        return tuple()
    variants = []
    n = len(seq)
    for items in (seq, list(reversed(seq))):
        for i in range(n):
            variants.append(tuple(items[i:] + items[:i]))
    return min(variants)


def endpoint_port_point(strokes, stroke_local_i, endpoint_i):
    pts = np.asarray(strokes[int(stroke_local_i)]["points"], dtype=np.float64).reshape(-1, 2)
    return pts[0] if int(endpoint_i) == 0 else pts[-1]


def build_endpoint_port_connectors(strokes, comp, endpoint_tol=12.0, max_connectors_per_port=10):
    """Build non-transitive endpoint-to-endpoint connector choices for a component."""
    comp = [int(i) for i in comp]
    ports = [(int(si), 0) for si in comp] + [(int(si), 1) for si in comp]
    port_points = {p: endpoint_port_point(strokes, p[0], p[1]) for p in ports}
    connectors = {p: [] for p in ports}
    tol2 = float(endpoint_tol) * float(endpoint_tol)
    for i, p in enumerate(ports):
        for q in ports[i + 1:]:
            if int(p[0]) == int(q[0]):
                continue
            d = port_points[p] - port_points[q]
            d2 = float(np.dot(d, d))
            if d2 > tol2:
                continue
            dist = float(math.sqrt(d2))
            connectors[p].append({"to": q, "distance": dist})
            connectors[q].append({"to": p, "distance": dist})

    for p, items in connectors.items():
        items.sort(key=lambda item: (
            float(item["distance"]),
            int(strokes[int(item["to"][0])]["index"]),
            int(item["to"][1]),
        ))
        if max_connectors_per_port is not None and int(max_connectors_per_port) > 0:
            connectors[p] = items[:int(max_connectors_per_port)]
    return connectors, port_points


def ordered_loop_points_from_port_cycle(strokes, oriented_cycle):
    """Convert oriented strokes into one ordered closed-loop point sequence."""
    chunks = []
    for item in oriented_cycle:
        si = int(item["stroke_local_i"])
        entry_endpoint = int(item["entry_endpoint"])
        pts = np.asarray(strokes[si]["points"], dtype=np.float64).reshape(-1, 2)
        chunks.append(pts if entry_endpoint == 0 else pts[::-1])

        next_entry = item.get("next_entry_point", None)
        if next_entry is not None:
            next_entry = np.asarray(next_entry, dtype=np.float64).reshape(1, 2)
            if len(chunks[-1]) == 0 or np.linalg.norm(chunks[-1][-1] - next_entry[0]) > 1e-6:
                chunks.append(next_entry)

    if not chunks:
        return np.empty((0, 2), dtype=np.float64)
    return np.vstack([chunk for chunk in chunks if len(chunk) > 0])


def remove_consecutive_duplicate_points(points, tol=1e-6):
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(pts) == 0:
        return pts
    out = [pts[0]]
    tol2 = float(tol) * float(tol)
    for p in pts[1:]:
        d = p - out[-1]
        if float(np.dot(d, d)) > tol2:
            out.append(p)
    if len(out) > 1:
        d = out[0] - out[-1]
        if float(np.dot(d, d)) <= tol2:
            out.pop()
    return np.asarray(out, dtype=np.float64).reshape(-1, 2)


def remove_collinear_closed_polyline_vertices(points, tol=1e-6):
    pts = remove_consecutive_duplicate_points(points, tol=tol)
    if len(pts) < 4:
        return pts
    keep = []
    for i, p in enumerate(pts):
        prev_p = pts[(i - 1) % len(pts)]
        next_p = pts[(i + 1) % len(pts)]
        a = p - prev_p
        b = next_p - p
        la = float(np.linalg.norm(a))
        lb = float(np.linalg.norm(b))
        if la <= tol or lb <= tol:
            continue
        cross = abs(float(a[0] * b[1] - a[1] * b[0]))
        dot = float(np.dot(a, b))
        if cross <= float(tol) * max(1.0, la * lb) and dot >= -float(tol):
            continue
        keep.append(p)
    if len(keep) < 3:
        return pts
    return np.asarray(keep, dtype=np.float64).reshape(-1, 2)


def simplify_loop_points_for_intersection(points, approx_epsilon=1.0):
    """Simplify ordered loop samples to a closed polyline used for intersection checks."""
    pts = remove_consecutive_duplicate_points(points)
    if len(pts) < 3:
        return pts
    epsilon = max(0.0, float(approx_epsilon))
    if epsilon > 0.0 and len(pts) >= 4:
        approx = cv2.approxPolyDP(
            pts.astype(np.float32).reshape(-1, 1, 2),
            epsilon,
            True,
        )
        pts = approx.reshape(-1, 2).astype(np.float64)
        pts = remove_consecutive_duplicate_points(pts)
    pts = remove_collinear_closed_polyline_vertices(pts)
    return pts


def orientation_value(a, b, c):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def point_on_segment(p, a, b, eps=1e-6):
    p = np.asarray(p, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if abs(orientation_value(a, b, p)) > float(eps):
        return False
    return (
        min(a[0], b[0]) - float(eps) <= p[0] <= max(a[0], b[0]) + float(eps)
        and min(a[1], b[1]) - float(eps) <= p[1] <= max(a[1], b[1]) + float(eps)
    )


def segments_intersect_2d(a, b, c, d, eps=1e-6):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    d = np.asarray(d, dtype=np.float64)
    if np.linalg.norm(a - b) <= float(eps) or np.linalg.norm(c - d) <= float(eps):
        return False
    if (
        max(min(a[0], b[0]), min(c[0], d[0])) > min(max(a[0], b[0]), max(c[0], d[0])) + float(eps)
        or max(min(a[1], b[1]), min(c[1], d[1])) > min(max(a[1], b[1]), max(c[1], d[1])) + float(eps)
    ):
        return False

    o1 = orientation_value(a, b, c)
    o2 = orientation_value(a, b, d)
    o3 = orientation_value(c, d, a)
    o4 = orientation_value(c, d, b)

    if (
        (o1 > eps and o2 < -eps or o1 < -eps and o2 > eps)
        and (o3 > eps and o4 < -eps or o3 < -eps and o4 > eps)
    ):
        return True
    if abs(o1) <= eps and point_on_segment(c, a, b, eps=eps):
        return True
    if abs(o2) <= eps and point_on_segment(d, a, b, eps=eps):
        return True
    if abs(o3) <= eps and point_on_segment(a, c, d, eps=eps):
        return True
    if abs(o4) <= eps and point_on_segment(b, c, d, eps=eps):
        return True
    return False


def loop_self_intersection_info(points):
    """Detect intersections between non-adjacent segments of a closed polyline."""
    pts = simplify_loop_points_for_intersection(points)
    info = {
        "self_intersecting": False,
        "reason": "",
        "simplified_vertex_count": int(len(pts)),
        "intersections": [],
    }
    n = int(len(pts))
    if n < 3:
        info["self_intersecting"] = True
        info["reason"] = "loop_too_short_after_simplification"
        return info

    seg_starts = pts
    seg_ends = np.roll(pts, -1, axis=0)
    min_x = np.minimum(seg_starts[:, 0], seg_ends[:, 0])
    max_x = np.maximum(seg_starts[:, 0], seg_ends[:, 0])
    min_y = np.minimum(seg_starts[:, 1], seg_ends[:, 1])
    max_y = np.maximum(seg_starts[:, 1], seg_ends[:, 1])
    eps = 1e-6

    for i in range(n):
        a = seg_starts[i]
        b = seg_ends[i]
        for j in range(i + 1, n):
            if j == i:
                continue
            if abs(i - j) == 1:
                continue
            if i == 0 and j == n - 1:
                continue
            if (
                max(min_x[i], min_x[j]) > min(max_x[i], max_x[j]) + eps
                or max(min_y[i], min_y[j]) > min(max_y[i], max_y[j]) + eps
            ):
                continue
            c = seg_starts[j]
            d = seg_ends[j]
            if segments_intersect_2d(a, b, c, d):
                info["self_intersecting"] = True
                info["reason"] = "self_intersecting_loop"
                info["intersections"].append({
                    "segment_a": int(i),
                    "segment_b": int(j),
                    "a0": [float(a[0]), float(a[1])],
                    "a1": [float(b[0]), float(b[1])],
                    "b0": [float(c[0]), float(c[1])],
                    "b1": [float(d[0]), float(d[1])],
                })
                return info
    return info


def candidate_from_ordered_loop(shape, strokes, loop_info, thickness=2):
    """Create a cap candidate from an explicitly ordered endpoint-port cycle."""
    loop_comp = [int(i) for i in loop_info.get("component_local_indices", [])]
    loop_strokes = [strokes[i] for i in loop_comp]
    loop_points = np.asarray(loop_info.get("ordered_loop_points", []), dtype=np.float64).reshape(-1, 2)
    if len(loop_points) < 3:
        return None
    self_intersection = loop_self_intersection_info(loop_points)
    loop_info["self_intersection"] = json_safe_debug_value(self_intersection)
    if self_intersection.get("self_intersecting", False):
        loop_info["reject_reason"] = "self_intersecting_loop"
        return None

    mask = np.zeros(shape[:2], dtype=np.uint8)
    pts_i = np.rint(loop_points).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(mask, [pts_i], True, 255, max(1, int(thickness)), cv2.LINE_AA)
    mask = cv2.dilate(mask, np.ones((2, 2), np.uint8), iterations=1)
    enclosed_area, enclosed_mask = estimate_enclosed_area_from_loop_mask(mask)

    cap_pixels_mask = ((mask > 0) | (enclosed_mask > 0)).astype(np.uint8)
    ys, xs = np.where(cap_pixels_mask > 0)
    bbox_meta = bbox_for_points(list(zip(xs.tolist(), ys.tolist())))
    stroke_pts = np.vstack([s["points"] for s in loop_strokes]) if loop_strokes else loop_points
    center = stroke_pts.mean(axis=0).astype(np.float64)
    area = int(np.count_nonzero(mask))
    total_arc = float(sum(s["arc"] for s in loop_strokes))
    return {
        "mask": mask,
        "enclosed_mask": enclosed_mask,
        "area": area,
        "enclosed_area": int(enclosed_area),
        "bbox": None if bbox_meta is None else bbox_meta.get("bbox"),
        "bbox_area": 0 if bbox_meta is None else int(bbox_meta.get("bbox_area", 0)),
        "endpoints": 0,
        "closedness": 1.0,
        "score": float(enclosed_area + area + total_arc),
        "center": center,
        "stroke_indices": [int(s["index"]) for s in loop_strokes],
        "stroke_count": len(loop_strokes),
        "total_arc": total_arc,
        "ordered_loop_points": [[float(x), float(y)] for x, y in loop_points.tolist()],
        "ordered_cycle_strokes": [int(s["index"]) for s in loop_strokes],
        "ordered_cycle": json_safe_debug_value(loop_info.get("ordered_cycle", [])),
        "connector_edges": json_safe_debug_value(loop_info.get("connector_edges", [])),
        "connector_total_gap": float(loop_info.get("connector_total_gap", 0.0)),
        "connector_max_gap": float(loop_info.get("connector_max_gap", 0.0)),
        "self_intersection": json_safe_debug_value(self_intersection),
    }


def signed_polyline_area(points):
    """Signed polygon area for an ordered closed polyline."""
    pts = remove_consecutive_duplicate_points(points)
    if len(pts) < 3:
        return 0.0
    x = pts[:, 0]
    y = pts[:, 1]
    return float(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))



def rasterize_planar_face_masks(faces, pad=2):
    """Rasterize ordered face loops onto a shared canvas for pixel containment tests."""
    if not faces:
        return [], None

    point_sets = []
    for face in faces:
        pts = np.asarray(face.get("ordered_loop_points", []), dtype=np.float64).reshape(-1, 2)
        if len(pts) >= 3:
            point_sets.append(pts)
    if not point_sets:
        return [], None

    all_pts = np.vstack(point_sets)
    min_x = int(math.floor(float(np.min(all_pts[:, 0])))) - int(pad)
    min_y = int(math.floor(float(np.min(all_pts[:, 1])))) - int(pad)
    max_x = int(math.ceil(float(np.max(all_pts[:, 0])))) + int(pad)
    max_y = int(math.ceil(float(np.max(all_pts[:, 1])))) + int(pad)
    width = max(1, int(max_x - min_x + 1))
    height = max(1, int(max_y - min_y + 1))

    kernel = np.ones((3, 3), np.uint8)
    masks = []
    for face in faces:
        pts = np.asarray(face.get("ordered_loop_points", []), dtype=np.float64).reshape(-1, 2)
        if len(pts) < 3:
            masks.append(None)
            continue
        shifted = np.rint(pts - np.array([min_x, min_y], dtype=np.float64)).astype(np.int32).reshape(-1, 1, 2)
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [shifted], 255)
        interior = cv2.erode(mask, kernel, iterations=1)
        if int(np.count_nonzero(interior)) == 0:
            interior = mask.copy()
        masks.append({
            "mask": mask,
            "interior_mask": interior,
            "pixel_area": int(np.count_nonzero(mask)),
            "interior_pixel_area": int(np.count_nonzero(interior)),
        })
    return masks, {
        "origin": [int(min_x), int(min_y)],
        "shape": [int(height), int(width)],
    }


def select_outer_face_index_by_pixel_containment(faces, contain_ratio=0.98, min_overlap_pixels=16):
    """
    Drop the largest-area face as outer only when pixels show it contains others.

    Why this is needed:
      Largest polygon area alone is too aggressive for this project. A large raw
      face can still be a valid bounded loop candidate. Before dropping one as
      the outer face, require pixel evidence that it actually contains at least
      one other face.
    """
    masks, canvas = rasterize_planar_face_masks(faces)
    meta = {
        "selected_index": None,
        "selected_reason": "no_largest_face_contains_other_faces_by_pixels",
        "canvas": canvas,
        "contain_ratio": float(contain_ratio),
        "min_overlap_pixels": int(min_overlap_pixels),
        "largest_area_face_index": None,
        "faces": [],
    }
    if not faces or not masks:
        return None, meta

    def face_sort_key(face_index):
        face_mask_info = masks[face_index] if face_index < len(masks) else None
        return (
            float(faces[face_index].get("abs_area", 0.0)),
            0 if face_mask_info is None else int(face_mask_info.get("pixel_area", 0)),
            -int(face_index),
        )

    largest_face_i = max(range(len(faces)), key=face_sort_key)
    meta["largest_area_face_index"] = int(largest_face_i)

    for i, face in enumerate(faces):
        face_mask_info = masks[i] if i < len(masks) else None
        meta["faces"].append({
            "face_index": int(i),
            "abs_area": float(face.get("abs_area", 0.0)),
            "pixel_area": 0 if face_mask_info is None else int(face_mask_info.get("pixel_area", 0)),
            "tested_as_outer_candidate": bool(i == largest_face_i),
            "contained_faces": [],
        })

    largest_mask_info = masks[largest_face_i] if largest_face_i < len(masks) else None
    if largest_mask_info is None:
        meta["selected_reason"] = "largest_face_has_no_pixel_mask"
        return None, meta

    outer_mask = np.asarray(largest_mask_info["mask"], dtype=np.uint8) > 0
    contained = []
    for j, _other in enumerate(faces):
        if largest_face_i == j:
            continue
        other_mask_info = masks[j] if j < len(masks) else None
        if other_mask_info is None:
            continue
        inner_mask = np.asarray(other_mask_info["interior_mask"], dtype=np.uint8) > 0
        inner_pixels = int(np.count_nonzero(inner_mask))
        if inner_pixels <= 0:
            inner_mask = np.asarray(other_mask_info["mask"], dtype=np.uint8) > 0
            inner_pixels = int(np.count_nonzero(inner_mask))
        if inner_pixels <= 0:
            continue
        overlap = int(np.count_nonzero(outer_mask & inner_mask))
        ratio = 0.0 if inner_pixels <= 0 else float(overlap) / float(inner_pixels)
        required_overlap = max(1, min(int(min_overlap_pixels), int(inner_pixels)))
        if overlap >= required_overlap and ratio >= float(contain_ratio):
            contained.append({
                "face_index": int(j),
                "overlap_pixels": int(overlap),
                "required_overlap_pixels": int(required_overlap),
                "inner_pixels": int(inner_pixels),
                "contained_ratio": float(ratio),
            })

    meta["faces"][largest_face_i]["contained_faces"] = contained
    if contained:
        meta["selected_index"] = int(largest_face_i)
        meta["selected_reason"] = "largest_face_pixel_contains_other_faces"
        return int(largest_face_i), meta
    return None, meta


def first_nonzero_halfedge_vector(points, fallback, eps=1e-6):
    """Return the first usable tangent vector from the start of a halfedge."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(pts) >= 2:
        start = pts[0]
        for p in pts[1:]:
            v = p - start
            if float(np.dot(v, v)) > float(eps) * float(eps):
                return v
    return np.asarray(fallback, dtype=np.float64)


def ordered_points_from_halfedges(halfedges, face_halfedge_ids):
    """Concatenate halfedge geometry into one ordered closed boundary."""
    chunks = []
    for he_id in face_halfedge_ids:
        pts = np.asarray(halfedges[int(he_id)]["points"], dtype=np.float64).reshape(-1, 2)
        if len(pts) == 0:
            continue
        if not chunks:
            chunks.append(pts)
            continue
        prev = chunks[-1][-1]
        if float(np.dot(prev - pts[0], prev - pts[0])) <= 1e-12:
            chunks.append(pts[1:])
        else:
            chunks.append(pts)
    if not chunks:
        return np.empty((0, 2), dtype=np.float64)
    return remove_consecutive_duplicate_points(np.vstack([chunk for chunk in chunks if len(chunk) > 0]))


def canonical_face_halfedge_signature(face_halfedge_ids):
    """Canonical directed halfedge cycle signature up to rotation."""
    seq = [int(x) for x in face_halfedge_ids]
    if not seq:
        return tuple()
    variants = []
    for i in range(len(seq)):
        variants.append(tuple(seq[i:] + seq[:i]))
    return min(variants)


def graph_cycle_edge_points(strokes, graph, edge, from_node, to_node, centers):
    """Return geometry for one graph edge oriented from_node -> to_node."""
    from_node = int(from_node)
    to_node = int(to_node)
    if edge.get("edge_kind") == "connector":
        return np.asarray([centers[from_node], centers[to_node]], dtype=np.float64).reshape(-1, 2)

    pts = np.asarray(strokes[int(edge["stroke_local_i"])]["points"], dtype=np.float64).reshape(-1, 2)
    if int(edge.get("start_node", -1)) == from_node and int(edge.get("end_node", -1)) == to_node:
        return pts.copy()
    return pts[::-1].copy()


def ordered_points_from_graph_cycle(strokes, graph, edge_indices, node_cycle, centers):
    """Concatenate oriented graph-edge geometry into one ordered closed loop."""
    graph_edges = list(graph.get("edges", []))
    chunks = []
    for i, edge_index in enumerate(edge_indices):
        edge = graph_edges[int(edge_index)]
        from_node = int(node_cycle[i])
        to_node = int(node_cycle[i + 1])
        pts = graph_cycle_edge_points(strokes, graph, edge, from_node, to_node, centers)
        if len(pts) == 0:
            continue
        if not chunks:
            chunks.append(pts)
            continue
        prev = chunks[-1][-1]
        if float(np.dot(prev - pts[0], prev - pts[0])) <= 1e-12:
            chunks.append(pts[1:])
        else:
            chunks.append(pts)
    if not chunks:
        return np.empty((0, 2), dtype=np.float64)
    return remove_consecutive_duplicate_points(np.vstack([chunk for chunk in chunks if len(chunk) > 0]))


def enumerate_planar_graph_simple_cycle_loop_infos(
    strokes,
    comp,
    endpoint_tol=12.0,
    min_abs_area=1.0,
    max_cycle_edges=None,
    max_cycles=1024,
    progress_callback=None,
    progress_context=None,
    stop_event=None,
):
    """
    Enumerate simple cycles in the stroke-first planar graph.

    This is a fallback/complement to half-edge face walking. Shared-node cap
    components can produce degenerate zero-area face walks when several
    zero-length connector choices meet at the same drawn junction; simple-cycle
    enumeration still exposes the individual loop boundaries so the later cover
    selection can verify whether their stroke union explains the component.
    """
    comp = list(map(int, comp))
    context = {} if progress_context is None else dict(progress_context)
    if cap_search_stop_requested(stop_event):
        if progress_callback is not None:
            progress_callback(
                "planar_graph_cycle_extraction_cancelled",
                **context,
                component_local_indices=comp,
            )
        return []

    graph = build_stroke_first_planar_graph(strokes, comp, endpoint_tol=endpoint_tol)
    graph_edges = list(graph.get("edges", []))
    centers = [np.asarray(c, dtype=np.float64).reshape(2) for c in graph.get("endpoint_centers", [])]
    if not graph_edges or not centers:
        if progress_callback is not None:
            progress_callback(
                "planar_graph_cycle_extraction_finished",
                **context,
                raw_cycle_count=0,
                loop_info_count=0,
                reason="empty_graph",
            )
        return []

    if max_cycle_edges is None or int(max_cycle_edges) <= 0:
        max_cycle_edges = len(graph_edges)
    max_cycle_edges = max(2, min(int(max_cycle_edges), len(graph_edges)))
    max_cycles = max(1, int(max_cycles))

    adjacency = {}
    for edge in graph_edges:
        edge_index = int(edge["edge_index"])
        a = int(edge["start_node"])
        b = int(edge["end_node"])
        if a < 0 or b < 0 or a >= len(centers) or b >= len(centers) or a == b:
            continue
        adjacency.setdefault(a, []).append((edge_index, b))
        adjacency.setdefault(b, []).append((edge_index, a))
    for items in adjacency.values():
        items.sort(key=lambda item: (int(item[1]), int(item[0])))

    if progress_callback is not None:
        progress_callback(
            "planar_graph_cycle_extraction_started",
            **context,
            component_local_indices=comp,
            component_stroke_indices=[int(strokes[int(i)]["index"]) for i in comp],
            graph_summary={
                "node_count": int(len(centers)),
                "edge_count": int(len(graph_edges)),
                "real_edge_count": int(len(graph.get("real_edges", []))),
                "connector_edge_count": int(len(graph.get("connector_edges", []))),
                "unmatched_endpoint_count": int(len(graph.get("unmatched_endpoint_nodes", []))),
                "strategy": graph.get("strategy", ""),
            },
            max_cycle_edges=int(max_cycle_edges),
            max_cycles=int(max_cycles),
        )

    raw_cycles = []
    seen_edge_sets = set()

    def add_cycle(edge_indices, node_cycle):
        key = tuple(sorted(int(ei) for ei in edge_indices))
        if len(key) < 2 or key in seen_edge_sets:
            return
        seen_edge_sets.add(key)
        raw_cycles.append({
            "edge_indices": [int(ei) for ei in edge_indices],
            "node_cycle": [int(n) for n in node_cycle],
        })

    def dfs(start_node, node, visited_nodes, path_edges, path_nodes):
        if len(raw_cycles) >= max_cycles:
            return
        if cap_search_stop_requested(stop_event):
            return
        if len(path_edges) >= max_cycle_edges:
            return
        for edge_index, nb in adjacency.get(int(node), []):
            edge_index = int(edge_index)
            nb = int(nb)
            if edge_index in path_edges:
                continue
            if nb == start_node:
                if len(path_edges) >= 1:
                    add_cycle(path_edges + [edge_index], path_nodes + [start_node])
                continue
            # Only let the minimum node in a cycle start that cycle.
            if nb < start_node or nb in visited_nodes:
                continue
            dfs(
                start_node,
                nb,
                visited_nodes | {nb},
                path_edges + [edge_index],
                path_nodes + [nb],
            )
            if len(raw_cycles) >= max_cycles or cap_search_stop_requested(stop_event):
                return

    for start_node in sorted(adjacency):
        dfs(int(start_node), int(start_node), {int(start_node)}, [], [int(start_node)])
        if len(raw_cycles) >= max_cycles or cap_search_stop_requested(stop_event):
            break

    loop_infos = []
    seen_loop_keys = set()
    for cycle in raw_cycles:
        edge_indices = [int(ei) for ei in cycle.get("edge_indices", [])]
        node_cycle = [int(n) for n in cycle.get("node_cycle", [])]
        if len(edge_indices) < 2 or len(node_cycle) != len(edge_indices) + 1:
            continue

        ordered_points = ordered_points_from_graph_cycle(strokes, graph, edge_indices, node_cycle, centers)
        if len(ordered_points) < 3:
            continue
        signed_area = signed_polyline_area(ordered_points)
        abs_area = abs(float(signed_area))
        if abs_area < float(min_abs_area):
            continue

        local_indices = []
        stroke_ids = []
        connector_edges = []
        connector_gaps = []
        ordered_cycle = []
        for i, edge_index in enumerate(edge_indices):
            edge = graph_edges[int(edge_index)]
            from_node = int(node_cycle[i])
            to_node = int(node_cycle[i + 1])
            if edge.get("edge_kind") == "connector":
                gap = float(edge.get("connector_distance", 0.0))
                connector_gaps.append(gap)
                connector_edges.append({
                    "from_stroke": int(graph["nodes"][from_node]["stroke_index"]),
                    "from_endpoint": int(graph["nodes"][from_node]["endpoint"]),
                    "to_stroke": int(graph["nodes"][to_node]["stroke_index"]),
                    "to_endpoint": int(graph["nodes"][to_node]["endpoint"]),
                    "from_node": int(from_node),
                    "to_node": int(to_node),
                    "distance": gap,
                })
                ordered_cycle.append({
                    "edge_kind": "connector",
                    "edge_index": int(edge_index),
                    "from_node": int(from_node),
                    "to_node": int(to_node),
                    "stroke_local_i": -1,
                    "stroke_index": -1,
                })
                continue

            local_i = int(edge["stroke_local_i"])
            stroke_id = int(edge["stroke_index"])
            local_indices.append(local_i)
            stroke_ids.append(stroke_id)
            ordered_cycle.append({
                "edge_kind": "stroke",
                "edge_index": int(edge_index),
                "from_node": int(from_node),
                "to_node": int(to_node),
                "stroke_local_i": local_i,
                "stroke_index": stroke_id,
            })

        if len(local_indices) < 2:
            continue
        stroke_key = canonical_cycle_sequence(stroke_ids)
        loop_key = (
            stroke_key,
            round(abs_area, 3),
            round(float(sum(connector_gaps)), 3),
        )
        if loop_key in seen_loop_keys:
            continue
        seen_loop_keys.add(loop_key)

        loop_infos.append({
            "component_local_indices": local_indices,
            "topology_kind": "planar_graph_simple_cycle",
            "topology": {
                "from_planar_graph_simple_cycle": True,
                "from_planar_stroke_first_graph": True,
                "cycle_strokes": stroke_ids,
                "cycle_edge_indices": edge_indices,
                "cycle_node_ids": node_cycle,
                "signed_area": float(signed_area),
                "abs_area": float(abs_area),
                "endpoint_node_count": int(len(centers)),
                "real_edge_count": int(len(graph.get("real_edges", []))),
                "connector_edge_count": int(len(graph.get("connector_edges", []))),
                "raw_cycle_count": int(len(raw_cycles)),
                "unmatched_endpoint_nodes": list(graph.get("unmatched_endpoint_nodes", [])),
                "strategy": graph.get("strategy", ""),
            },
            "ordered_cycle": ordered_cycle,
            "ordered_loop_points": [[float(x), float(y)] for x, y in ordered_points.tolist()],
            "connector_edges": json_safe_debug_value(connector_edges),
            "connector_total_gap": float(sum(connector_gaps)),
            "connector_max_gap": 0.0 if not connector_gaps else float(max(connector_gaps)),
        })

    loop_infos.sort(key=lambda item: (
        -float(item.get("topology", {}).get("abs_area", 0.0)),
        float(item.get("connector_total_gap", 0.0)),
        item.get("topology", {}).get("cycle_strokes", []),
    ))
    if progress_callback is not None:
        progress_callback(
            "planar_graph_cycle_extraction_finished",
            **context,
            raw_cycle_count=int(len(raw_cycles)),
            loop_info_count=int(len(loop_infos)),
            hit_max_cycles=bool(len(raw_cycles) >= max_cycles),
            graph_summary={
                "node_count": int(len(centers)),
                "edge_count": int(len(graph_edges)),
                "real_edge_count": int(len(graph.get("real_edges", []))),
                "connector_edge_count": int(len(graph.get("connector_edges", []))),
                "unmatched_endpoint_count": int(len(graph.get("unmatched_endpoint_nodes", []))),
                "strategy": graph.get("strategy", ""),
            },
            cycle_strokes=[item.get("topology", {}).get("cycle_strokes", []) for item in loop_infos[:20]],
        )
    return loop_infos


def build_stroke_first_planar_graph(strokes, comp, endpoint_tol=12.0, max_connectors_per_endpoint=2):
    """
    Build a planar graph without collapsing short strokes.

    Every real stroke endpoint first becomes its own graph node, and every
    stroke becomes a real graph edge. Short gap connectors are added afterwards
    between nearby endpoints, so short strokes cannot disappear into one
    endpoint cluster before face extraction.
    """
    comp = list(map(int, comp))
    nodes = []
    port_to_node = {}
    for si in comp:
        pts = np.asarray(strokes[int(si)]["points"], dtype=np.float64).reshape(-1, 2)
        if len(pts) == 0:
            continue
        for endpoint_i, point in ((0, pts[0]), (1, pts[-1])):
            node_id = len(nodes)
            node = {
                "node_id": int(node_id),
                "point": np.asarray(point, dtype=np.float64).reshape(2),
                "stroke_local_i": int(si),
                "stroke_index": int(strokes[int(si)]["index"]),
                "endpoint": int(endpoint_i),
            }
            nodes.append(node)
            port_to_node[(int(si), int(endpoint_i))] = int(node_id)

    edges = []

    def add_edge(edge_kind, start_node, end_node, stroke_local_i=-1, connector_distance=0.0):
        edge_index = len(edges)
        start_node = int(start_node)
        end_node = int(end_node)
        start_meta = nodes[start_node]
        end_meta = nodes[end_node]
        if edge_kind == "stroke":
            stroke_index = int(strokes[int(stroke_local_i)]["index"])
            start_endpoint = 0
            end_endpoint = 1
        else:
            stroke_index = -1
            start_endpoint = int(start_meta["endpoint"])
            end_endpoint = int(end_meta["endpoint"])
        edge = {
            "edge_index": int(edge_index),
            "edge_kind": str(edge_kind),
            "stroke_local_i": int(stroke_local_i),
            "stroke_index": int(stroke_index),
            "start_node": int(start_node),
            "end_node": int(end_node),
            "start_stroke": int(start_meta["stroke_index"]),
            "start_endpoint": int(start_endpoint),
            "end_stroke": int(end_meta["stroke_index"]),
            "end_endpoint": int(end_endpoint),
            "connector_distance": float(connector_distance),
            "is_connector": edge_kind == "connector",
        }
        edges.append(edge)
        return edge

    real_edges = []
    for si in comp:
        start_node = port_to_node.get((int(si), 0), None)
        end_node = port_to_node.get((int(si), 1), None)
        if start_node is None or end_node is None:
            continue
        real_edges.append(add_edge("stroke", start_node, end_node, stroke_local_i=int(si)))

    connector_candidates = []
    tol2 = float(endpoint_tol) * float(endpoint_tol)
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if int(nodes[i]["stroke_local_i"]) == int(nodes[j]["stroke_local_i"]):
                continue
            d = nodes[i]["point"] - nodes[j]["point"]
            d2 = float(np.dot(d, d))
            if d2 > tol2:
                continue
            connector_candidates.append({
                "a": int(i),
                "b": int(j),
                "distance": float(math.sqrt(d2)),
            })
    connector_candidates.sort(key=lambda item: (
        float(item["distance"]),
        int(nodes[int(item["a"])]["stroke_index"]),
        int(nodes[int(item["a"])]["endpoint"]),
        int(nodes[int(item["b"])]["stroke_index"]),
        int(nodes[int(item["b"])]["endpoint"]),
    ))

    connector_degree = {int(node["node_id"]): 0 for node in nodes}
    connector_pairs = set()
    connector_edges = []
    max_connectors_per_endpoint = max(1, int(max_connectors_per_endpoint))

    def add_connector(candidate):
        a = int(candidate["a"])
        b = int(candidate["b"])
        key = tuple(sorted((a, b)))
        if key in connector_pairs:
            return False
        connector_pairs.add(key)
        connector_degree[a] = int(connector_degree.get(a, 0)) + 1
        connector_degree[b] = int(connector_degree.get(b, 0)) + 1
        connector_edges.append(add_edge(
            "connector",
            a,
            b,
            stroke_local_i=-1,
            connector_distance=float(candidate["distance"]),
        ))
        return True

    # First pass: greedily match nearest open endpoints. This is the normal
    # stroke-first closure path and avoids creating dense connector cliques.
    for candidate in connector_candidates:
        a = int(candidate["a"])
        b = int(candidate["b"])
        if connector_degree.get(a, 0) == 0 and connector_degree.get(b, 0) == 0:
            add_connector(candidate)

    # Second pass: if a remaining endpoint still has no connector, attach it to
    # its nearest available endpoint. This handles small junction ambiguities
    # without connecting every endpoint pair within the tolerance.
    for candidate in connector_candidates:
        a = int(candidate["a"])
        b = int(candidate["b"])
        deg_a = int(connector_degree.get(a, 0))
        deg_b = int(connector_degree.get(b, 0))
        if deg_a > 0 and deg_b > 0:
            continue
        if deg_a >= max_connectors_per_endpoint or deg_b >= max_connectors_per_endpoint:
            continue
        add_connector(candidate)

    unmatched_nodes = [
        int(node_id)
        for node_id, degree in sorted(connector_degree.items())
        if int(degree) == 0
    ]
    return {
        "component_local_indices": list(map(int, comp)),
        "nodes": nodes,
        "endpoint_centers": [node["point"] for node in nodes],
        "edges": edges,
        "real_edges": real_edges,
        "connector_edges": connector_edges,
        "connector_candidates": connector_candidates,
        "connector_degree": {int(k): int(v) for k, v in connector_degree.items()},
        "unmatched_endpoint_nodes": unmatched_nodes,
        "self_loop_strokes": [],
        "strategy": "stroke_first_nearest_gap_connectors",
    }


def component_mask_canvas_bounds(strokes, comp, pad=8):
    """Return a padded canvas that covers every point in comp."""
    point_sets = []
    for si in comp:
        pts = np.asarray(strokes[int(si)]["points"], dtype=np.float64).reshape(-1, 2)
        if len(pts) > 0:
            point_sets.append(pts)
    if not point_sets:
        return None

    all_pts = np.vstack(point_sets)
    min_x = int(math.floor(float(np.min(all_pts[:, 0])))) - int(pad)
    min_y = int(math.floor(float(np.min(all_pts[:, 1])))) - int(pad)
    max_x = int(math.ceil(float(np.max(all_pts[:, 0])))) + int(pad)
    max_y = int(math.ceil(float(np.max(all_pts[:, 1])))) + int(pad)
    width = max(1, int(max_x - min_x + 1))
    height = max(1, int(max_y - min_y + 1))
    return {
        "origin": np.asarray([float(min_x), float(min_y)], dtype=np.float64),
        "shape": (int(height), int(width)),
    }


def draw_component_stroke_masks(strokes, comp, origin, shape, thickness=1):
    """Rasterize every stroke in comp onto one shared canvas."""
    mask = np.zeros(shape, dtype=np.uint8)
    stroke_masks = {}
    for si in comp:
        pts = np.asarray(strokes[int(si)]["points"], dtype=np.float64).reshape(-1, 2)
        stroke_mask = np.zeros(shape, dtype=np.uint8)
        if len(pts) == 1:
            px = int(round(float(pts[0][0] - origin[0])))
            py = int(round(float(pts[0][1] - origin[1])))
            if 0 <= px < shape[1] and 0 <= py < shape[0]:
                cv2.circle(stroke_mask, (px, py), max(1, int(thickness)), 255, -1, cv2.LINE_8)
        elif len(pts) >= 2:
            shifted = np.rint(pts - origin[None, :]).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(stroke_mask, [shifted], False, 255, max(1, int(thickness)), cv2.LINE_8)
        if int(np.count_nonzero(stroke_mask)) <= 0:
            continue
        stroke_masks[int(si)] = stroke_mask
        mask = cv2.bitwise_or(mask, stroke_mask)
    return mask, stroke_masks


def connected_labels_near_point(labels, point_xy, radius=1):
    """Return connected-component labels present near one point."""
    h, w = labels.shape[:2]
    x = int(round(float(point_xy[0])))
    y = int(round(float(point_xy[1])))
    x0 = max(0, x - int(radius))
    y0 = max(0, y - int(radius))
    x1 = min(w, x + int(radius) + 1)
    y1 = min(h, y + int(radius) + 1)
    if x0 >= x1 or y0 >= y1:
        return set()
    return {int(v) for v in np.unique(labels[y0:y1, x0:x1]).tolist()}


def connector_supported_mask_from_local_closing(mask, p0, p1, gap_distance, max_kernel_size=9):
    """Use a local morphology close to support a tiny endpoint gap without drawing a direct line."""
    if gap_distance <= 0.0:
        return None

    max_kernel_size = max(3, int(max_kernel_size))
    if max_kernel_size % 2 == 0:
        max_kernel_size += 1
    half = max(1, int((max_kernel_size - 1) // 2))
    radius = min(half, max(1, int(math.ceil(float(gap_distance) / 2.0))))
    kernel_size = 2 * int(radius) + 1
    margin = int(kernel_size) + 2

    x0 = max(0, int(math.floor(min(float(p0[0]), float(p1[0])))) - margin)
    y0 = max(0, int(math.floor(min(float(p0[1]), float(p1[1])))) - margin)
    x1 = min(mask.shape[1], int(math.ceil(max(float(p0[0]), float(p1[0])))) + margin + 1)
    y1 = min(mask.shape[0], int(math.ceil(max(float(p0[1]), float(p1[1])))) + margin + 1)
    if x0 >= x1 or y0 >= y1:
        return None

    roi = np.asarray(mask[y0:y1, x0:x1], dtype=np.uint8)
    if roi.size == 0:
        return None
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(kernel_size), int(kernel_size)))
    closed = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel)
    closed = ((closed > 0).astype(np.uint8)) * 255
    added = np.where((closed > 0) & (roi == 0), 255, 0).astype(np.uint8)
    added_pixels = int(np.count_nonzero(added))
    if added_pixels <= 0:
        return None

    _count, labels = cv2.connectedComponents((closed > 0).astype(np.uint8), connectivity=8)
    local_p0 = np.asarray([float(p0[0]) - float(x0), float(p0[1]) - float(y0)], dtype=np.float64)
    local_p1 = np.asarray([float(p1[0]) - float(x0), float(p1[1]) - float(y0)], dtype=np.float64)
    label0 = connected_labels_near_point(labels, local_p0, radius=max(1, int(kernel_size // 3)))
    label1 = connected_labels_near_point(labels, local_p1, radius=max(1, int(kernel_size // 3)))
    common = sorted(int(v) for v in (label0 & label1) if int(v) > 0)
    if not common:
        return None

    added_limit = max(24, int(math.ceil(max(1.0, float(gap_distance)) * float(kernel_size) * 2.5)))
    if added_pixels > added_limit:
        return None
    return {
        "x0": int(x0),
        "y0": int(y0),
        "mask": added,
        "kernel_size": int(kernel_size),
        "added_pixels": int(added_pixels),
    }


def build_geometry_supported_component_masks(strokes, comp, graph, endpoint_tol=12.0):
    """Build a geometry-first mask from real strokes plus the graph connector segments themselves."""
    canvas = component_mask_canvas_bounds(strokes, comp, pad=max(8, int(math.ceil(float(endpoint_tol))) + 2))
    if canvas is None:
        return None

    origin = np.asarray(canvas["origin"], dtype=np.float64).reshape(2)
    shape = tuple(canvas["shape"])
    real_mask, stroke_masks = draw_component_stroke_masks(strokes, comp, origin, shape, thickness=1)
    support_mask = np.zeros(shape, dtype=np.uint8)
    if graph is None:
        return {
            "origin": origin,
            "shape": shape,
            "real_mask": real_mask,
            "support_mask": support_mask,
            "final_mask": real_mask.copy(),
            "stroke_masks": stroke_masks,
            "support_records": [],
        }

    support_records = []
    connector_edges = sorted(
        list(graph.get("connector_edges", [])),
        key=lambda item: (
            float(item.get("connector_distance", 0.0)),
            int(item.get("start_node", -1)),
            int(item.get("end_node", -1)),
        ),
    )
    nodes = list(graph.get("nodes", []))
    current_mask = real_mask.copy()
    for edge in connector_edges:
        gap = float(edge.get("connector_distance", 0.0))
        if gap <= 0.0:
            continue
        start_node = int(edge.get("start_node", -1))
        end_node = int(edge.get("end_node", -1))
        if start_node < 0 or end_node < 0 or start_node >= len(nodes) or end_node >= len(nodes):
            continue

        p0 = np.asarray(nodes[start_node]["point"], dtype=np.float64).reshape(2) - origin
        p1 = np.asarray(nodes[end_node]["point"], dtype=np.float64).reshape(2) - origin
        a = tuple(np.rint(p0).astype(np.int32).tolist())
        b = tuple(np.rint(p1).astype(np.int32).tolist())

        line_mask = np.zeros(shape, dtype=np.uint8)
        cv2.line(line_mask, a, b, 255, 1, cv2.LINE_8)
        if int(np.count_nonzero(line_mask)) <= 0:
            continue

        added = np.where((line_mask > 0) & (current_mask == 0), 255, 0).astype(np.uint8)
        support_mask = cv2.bitwise_or(support_mask, line_mask)
        current_mask = cv2.bitwise_or(real_mask, support_mask)

        x0 = max(0, min(a[0], b[0]) - 1)
        y0 = max(0, min(a[1], b[1]) - 1)
        x1 = min(shape[1], max(a[0], b[0]) + 2)
        y1 = min(shape[0], max(a[1], b[1]) + 2)
        support_records.append({
            "start_stroke": int(edge.get("start_stroke", -1)),
            "start_endpoint": int(edge.get("start_endpoint", -1)),
            "end_stroke": int(edge.get("end_stroke", -1)),
            "end_endpoint": int(edge.get("end_endpoint", -1)),
            "start_point": [float(nodes[start_node]["point"][0]), float(nodes[start_node]["point"][1])],
            "end_point": [float(nodes[end_node]["point"][0]), float(nodes[end_node]["point"][1])],
            "distance": float(gap),
            "kernel_size": 0,
            "added_pixels": int(np.count_nonzero(added)),
            "roi": [int(x0), int(y0), int(max(0, x1 - x0)), int(max(0, y1 - y0))],
        })

    return {
        "origin": origin,
        "shape": shape,
        "real_mask": real_mask,
        "support_mask": support_mask,
        "final_mask": cv2.bitwise_or(real_mask, support_mask),
        "stroke_masks": stroke_masks,
        "support_records": support_records,
    }

def encode_debug_mask_png(mask):
    if mask is None:
        return None
    ok, encoded = cv2.imencode(".png", np.asarray(mask, dtype=np.uint8))
    if not ok:
        return None
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def decode_debug_mask_png(encoded):
    if not encoded:
        return None
    try:
        raw = base64.b64decode(encoded.encode("ascii"))
    except Exception:
        return None
    arr = np.frombuffer(raw, dtype=np.uint8)
    if arr.size <= 0:
        return None
    return cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)


def build_geometry_face_debug_payload(masks, face_infos, component_index=None):
    if masks is None:
        return None

    payload_faces = []
    for face_i, face in enumerate(face_infos):
        pts = np.asarray(face.get("ordered_loop_points", []), dtype=np.float64).reshape(-1, 2)
        payload_faces.append({
            "face_index": int(face_i),
            "abs_area": float(face.get("topology", {}).get("abs_area", 0.0)),
            "pixel_area": int(face.get("topology", {}).get("pixel_area", 0)),
            "stroke_indices": [int(x) for x in face.get("topology", {}).get("cycle_strokes", [])],
            "ordered_loop_points": [[float(x), float(y)] for x, y in pts.tolist()],
        })

    return {
        "component_index": None if component_index is None else int(component_index),
        "origin": [float(x) for x in np.asarray(masks.get("origin", [0.0, 0.0]), dtype=np.float64).reshape(2).tolist()],
        "shape": [int(x) for x in tuple(masks.get("shape", (0, 0)))],
        "real_mask_png": encode_debug_mask_png(masks.get("real_mask", None)),
        "support_mask_png": encode_debug_mask_png(masks.get("support_mask", None)),
        "final_mask_png": encode_debug_mask_png(masks.get("final_mask", None)),
        "support_records": json_safe_debug_value(masks.get("support_records", [])),
        "real_mask_pixels": int(np.count_nonzero(masks.get("real_mask", None))),
        "support_mask_pixels": int(np.count_nonzero(masks.get("support_mask", None))),
        "final_mask_pixels": int(np.count_nonzero(masks.get("final_mask", None))),
        "face_count": int(len(payload_faces)),
        "faces": payload_faces,
    }

def extract_geometry_supported_face_loop_infos(
    strokes,
    comp,
    endpoint_tol=12.0,
    min_abs_area=1.0,
    progress_callback=None,
    progress_context=None,
    stop_event=None,
):
    """Extract bounded faces from a rasterized real-stroke mask instead of direct graph connectors."""
    comp = list(map(int, comp))
    context = {} if progress_context is None else dict(progress_context)
    if cap_search_stop_requested(stop_event):
        return []

    graph = build_stroke_first_planar_graph(strokes, comp, endpoint_tol=endpoint_tol)
    masks = build_geometry_supported_component_masks(strokes, comp, graph, endpoint_tol=endpoint_tol)
    if masks is None:
        if progress_callback is not None:
            progress_callback(
                "geometry_face_extraction_finished",
                **context,
                face_count=0,
                reason="empty_component_mask",
            )
        return []

    final_mask = np.asarray(masks["final_mask"], dtype=np.uint8)
    if int(np.count_nonzero(final_mask)) <= 0:
        if progress_callback is not None:
            progress_callback(
                "geometry_face_extraction_finished",
                **context,
                face_count=0,
                reason="empty_geometry_mask",
            )
        return []

    binary = (final_mask > 0).astype(np.uint8)
    background = (binary == 0).astype(np.uint8)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(background, connectivity=4)
    origin = np.asarray(masks["origin"], dtype=np.float64).reshape(2)
    stroke_masks = masks.get("stroke_masks", {})
    boundary_kernel = np.ones((3, 3), np.uint8)
    face_infos = []
    min_pixel_area = max(1, int(math.ceil(float(min_abs_area))))

    for label_id in range(1, int(count)):
        if cap_search_stop_requested(stop_event):
            break
        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        w = int(stats[label_id, cv2.CC_STAT_WIDTH])
        h = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area < min_pixel_area:
            continue
        if x <= 0 or y <= 0 or x + w >= final_mask.shape[1] or y + h >= final_mask.shape[0]:
            continue

        region_mask = np.where(labels == int(label_id), 255, 0).astype(np.uint8)
        contours, _hier = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        if len(contour) < 3:
            continue

        ordered_points_local = contour.reshape(-1, 2).astype(np.float64)
        ordered_points = ordered_points_local + origin[None, :]
        signed_area = signed_polyline_area(ordered_points)
        abs_area = abs(float(signed_area))
        if abs_area < float(min_abs_area):
            continue

        boundary_mask = np.zeros_like(region_mask)
        cv2.drawContours(boundary_mask, [contour], -1, 255, 2, cv2.LINE_8)
        boundary_mask = cv2.dilate(boundary_mask, boundary_kernel, iterations=1)
        local_indices = []
        stroke_ids = []
        for local_i in comp:
            stroke_mask = stroke_masks.get(int(local_i), None)
            if stroke_mask is None:
                continue
            if int(np.count_nonzero(cv2.bitwise_and(stroke_mask, boundary_mask))) <= 0:
                continue
            local_indices.append(int(local_i))
            stroke_ids.append(int(strokes[int(local_i)]["index"]))
        if len(local_indices) < 2:
            continue

        support_records = [dict(record) for record in masks.get("support_records", [])]
        face_infos.append({
            "component_local_indices": local_indices,
            "topology_kind": "geometry_bounded_face",
            "topology": {
                "from_geometry_supported_mask": True,
                "cycle_strokes": [int(x) for x in stroke_ids],
                "signed_area": float(signed_area),
                "abs_area": float(abs_area),
                "pixel_area": int(area),
                "real_mask_pixels": int(np.count_nonzero(masks.get("real_mask", None))),
                "support_mask_pixels": int(np.count_nonzero(masks.get("support_mask", None))),
                "support_connector_count": int(len(masks.get("support_records", []))),
            },
            "ordered_cycle": [
                {
                    "stroke_local_i": int(local_i),
                    "stroke_index": int(strokes[int(local_i)]["index"]),
                    "edge_kind": "stroke",
                }
                for local_i in local_indices
            ],
            "ordered_loop_points": [[float(x), float(y)] for x, y in ordered_points.tolist()],
            "connector_edges": json_safe_debug_value(support_records),
            "connector_total_gap": 0.0,
            "connector_max_gap": 0.0,
        })

    face_infos.sort(key=lambda item: (
        -float(item.get("topology", {}).get("abs_area", 0.0)),
        item.get("topology", {}).get("cycle_strokes", []),
    ))
    if progress_callback is not None:
        progress_callback(
            "geometry_face_extraction_finished",
            **context,
            face_count=int(len(face_infos)),
            support_connector_count=int(len(masks.get("support_records", []))),
            real_mask_pixels=int(np.count_nonzero(masks.get("real_mask", None))),
            support_mask_pixels=int(np.count_nonzero(masks.get("support_mask", None))),
            face_strokes=[item.get("topology", {}).get("cycle_strokes", []) for item in face_infos[:20]],
            geometry_face_debug=build_geometry_face_debug_payload(
                masks,
                face_infos,
                component_index=context.get("component_index", None),
            ),
        )
    return face_infos

def extract_planar_face_loop_infos(
    strokes,
    comp,
    endpoint_tol=12.0,
    min_abs_area=1.0,
    progress_callback=None,
    progress_context=None,
    stop_event=None,
):
    """
    Extract bounded faces from a stroke-first planar graph.

    The normalized component is already closed in endpoint proximity. This pass
    keeps every stroke as a real edge first, then adds short virtual connector
    edges for remaining endpoint gaps before walking the planar half-edge
    embedding.
    """
    comp = list(map(int, comp))
    context = {} if progress_context is None else dict(progress_context)
    if cap_search_stop_requested(stop_event):
        if progress_callback is not None:
            progress_callback(
                "planar_face_extraction_cancelled",
                **context,
                component_local_indices=comp,
            )
        return []
    if progress_callback is not None:
        progress_callback(
            "planar_face_extraction_started",
            **context,
            component_local_indices=comp,
            component_stroke_indices=[int(strokes[int(i)]["index"]) for i in comp],
            endpoint_tol=float(endpoint_tol),
        )

    graph = build_stroke_first_planar_graph(strokes, comp, endpoint_tol=endpoint_tol)
    if cap_search_stop_requested(stop_event):
        if progress_callback is not None:
            progress_callback(
                "planar_face_extraction_cancelled",
                **context,
                component_local_indices=comp,
            )
        return []
    nodes = list(graph.get("nodes", []))
    centers = [np.asarray(c, dtype=np.float64).reshape(2) for c in graph.get("endpoint_centers", [])]
    graph_edges = list(graph.get("edges", []))
    real_edges = list(graph.get("real_edges", []))
    connector_edges = list(graph.get("connector_edges", []))
    if not centers or not graph_edges:
        if progress_callback is not None:
            progress_callback(
                "planar_face_extraction_finished",
                **context,
                bounded_face_count=0,
                raw_face_count=0,
                reason="empty_graph",
                graph_summary={
                    "node_count": int(len(centers)),
                    "real_edge_count": int(len(real_edges)),
                    "connector_edge_count": int(len(connector_edges)),
                    "unmatched_endpoint_count": int(len(graph.get("unmatched_endpoint_nodes", []))),
                    "strategy": graph.get("strategy", ""),
                },
            )
        return []

    halfedges = []
    outgoing = {}

    def add_halfedge(edge, from_node, to_node, points, twin_id=None):
        he_id = len(halfedges)
        fallback = centers[int(to_node)] - centers[int(from_node)]
        tangent = first_nonzero_halfedge_vector(points, fallback)
        angle = float(math.atan2(float(tangent[1]), float(tangent[0])))
        halfedges.append({
            "id": int(he_id),
            "twin": None if twin_id is None else int(twin_id),
            "edge_index": int(edge["edge_index"]),
            "edge_kind": edge.get("edge_kind", "stroke"),
            "stroke_local_i": int(edge["stroke_local_i"]),
            "stroke_index": int(edge["stroke_index"]),
            "from_node": int(from_node),
            "to_node": int(to_node),
            "points": np.asarray(points, dtype=np.float64).reshape(-1, 2),
            "angle": angle,
            "connector_distance": float(edge.get("connector_distance", 0.0)),
        })
        outgoing.setdefault(int(from_node), []).append(he_id)
        return he_id

    for edge in graph_edges:
        start_node = int(edge["start_node"])
        end_node = int(edge["end_node"])
        if start_node < 0 or end_node < 0 or start_node >= len(centers) or end_node >= len(centers):
            continue
        if edge.get("edge_kind") == "connector":
            pts = np.asarray([centers[start_node], centers[end_node]], dtype=np.float64).reshape(-1, 2)
        else:
            pts = np.asarray(strokes[int(edge["stroke_local_i"])]["points"], dtype=np.float64).reshape(-1, 2)
        if len(pts) < 2:
            continue
        pts_uv = pts.copy()
        pts_uv[0] = centers[start_node]
        pts_uv[-1] = centers[end_node]
        pts_vu = pts_uv[::-1].copy()
        he_uv = add_halfedge(edge, start_node, end_node, pts_uv)
        he_vu = add_halfedge(edge, end_node, start_node, pts_vu, twin_id=he_uv)
        halfedges[he_uv]["twin"] = int(he_vu)

    if not halfedges:
        if progress_callback is not None:
            progress_callback(
                "planar_face_extraction_finished",
                **context,
                bounded_face_count=0,
                raw_face_count=0,
                reason="empty_halfedge_graph",
                graph_summary={
                    "node_count": int(len(centers)),
                    "real_edge_count": int(len(real_edges)),
                    "connector_edge_count": int(len(connector_edges)),
                    "unmatched_endpoint_count": int(len(graph.get("unmatched_endpoint_nodes", []))),
                    "strategy": graph.get("strategy", ""),
                },
            )
        return []

    halfedge_pos_at_node = {}
    for node, items in outgoing.items():
        items.sort(key=lambda he_id: (
            float(halfedges[int(he_id)]["angle"]),
            int(halfedges[int(he_id)]["stroke_index"]),
            int(halfedges[int(he_id)]["edge_index"]),
            int(halfedges[int(he_id)]["to_node"]),
        ))
        for pos, he_id in enumerate(items):
            halfedge_pos_at_node[int(he_id)] = int(pos)

    visited = set()
    raw_faces = []
    max_steps = max(1, len(halfedges) + 1)
    for start_id in range(len(halfedges)):
        if start_id % 32 == 0 and cap_search_stop_requested(stop_event):
            if progress_callback is not None:
                progress_callback(
                    "planar_face_extraction_cancelled",
                    **context,
                    raw_face_count=int(len(raw_faces)),
                )
            return []
        if start_id in visited:
            continue
        cur = int(start_id)
        face_halfedges = []
        closed = False
        for _step in range(max_steps):
            if _step % 64 == 0 and cap_search_stop_requested(stop_event):
                if progress_callback is not None:
                    progress_callback(
                        "planar_face_extraction_cancelled",
                        **context,
                        raw_face_count=int(len(raw_faces)),
                    )
                return []
            if cur in visited:
                closed = cur == start_id
                break
            visited.add(cur)
            face_halfedges.append(cur)
            he = halfedges[cur]
            twin = int(he["twin"])
            at_node = int(he["to_node"])
            ring = outgoing.get(at_node, [])
            if not ring:
                break
            twin_pos = halfedge_pos_at_node.get(twin, None)
            if twin_pos is None:
                break
            # Previous in CCW order traces the next face adjacent to this directed edge.
            cur = int(ring[(int(twin_pos) - 1) % len(ring)])
            if cur == start_id:
                closed = True
                break
        if not closed or len(face_halfedges) < 2:
            continue
        signature = canonical_face_halfedge_signature(face_halfedges)
        if any(signature == item.get("signature") for item in raw_faces):
            continue
        points = ordered_points_from_halfedges(halfedges, face_halfedges)
        area = signed_polyline_area(points)
        stroke_ids = []
        local_indices = []
        face_connectors = []
        connector_gaps = []
        for he_id in face_halfedges:
            he = halfedges[int(he_id)]
            if he.get("edge_kind") == "connector":
                edge = graph_edges[int(he["edge_index"])]
                gap = float(edge.get("connector_distance", 0.0))
                connector_gaps.append(gap)
                face_connectors.append({
                    "from_stroke": int(nodes[int(he["from_node"])]["stroke_index"]),
                    "from_endpoint": int(nodes[int(he["from_node"])]["endpoint"]),
                    "to_stroke": int(nodes[int(he["to_node"])]["stroke_index"]),
                    "to_endpoint": int(nodes[int(he["to_node"])]["endpoint"]),
                    "from_node": int(he["from_node"]),
                    "to_node": int(he["to_node"]),
                    "distance": gap,
                })
                continue
            stroke_ids.append(int(he["stroke_index"]))
            local_indices.append(int(he["stroke_local_i"]))
        raw_faces.append({
            "signature": signature,
            "halfedge_ids": list(map(int, face_halfedges)),
            "ordered_loop_points": points,
            "signed_area": float(area),
            "abs_area": float(abs(area)),
            "stroke_indices": stroke_ids,
            "component_local_indices": local_indices,
            "connector_edges": face_connectors,
            "connector_total_gap": float(sum(connector_gaps)),
            "connector_max_gap": 0.0 if not connector_gaps else float(max(connector_gaps)),
        })

    area_faces = [face for face in raw_faces if float(face.get("abs_area", 0.0)) >= float(min_abs_area)]
    outer_face_i, outer_face_selection = select_outer_face_index_by_pixel_containment(area_faces)

    bounded_faces = []
    seen_face_keys = set()
    for i, face in enumerate(area_faces):
        if outer_face_i is not None and int(i) == int(outer_face_i) and len(area_faces) > 1:
            continue
        local_indices = [int(x) for x in face.get("component_local_indices", [])]
        ordered_points = np.asarray(face.get("ordered_loop_points", []), dtype=np.float64).reshape(-1, 2)
        if len(local_indices) < 2 or len(ordered_points) < 3:
            continue
        stroke_key = canonical_cycle_sequence(face.get("stroke_indices", []))
        area_key = round(float(face.get("abs_area", 0.0)), 3)
        face_key = (stroke_key, area_key)
        if face_key in seen_face_keys:
            continue
        seen_face_keys.add(face_key)
        bounded_faces.append({
            "component_local_indices": local_indices,
            "topology_kind": "planar_bounded_face",
            "topology": {
                "from_planar_endpoint_node_graph": True,
                "cycle_strokes": [int(x) for x in face.get("stroke_indices", [])],
                "face_halfedge_ids": list(map(int, face.get("halfedge_ids", []))),
                "signed_area": float(face.get("signed_area", 0.0)),
                "abs_area": float(face.get("abs_area", 0.0)),
                "endpoint_node_count": int(len(centers)),
                "real_edge_count": int(len(real_edges)),
                "connector_edge_count": int(len(connector_edges)),
                "halfedge_count": int(len(halfedges)),
                "raw_face_count": int(len(raw_faces)),
                "bounded_face_count": int(len(area_faces) - (1 if outer_face_i is not None and len(area_faces) > 1 else 0)),
                "unmatched_endpoint_nodes": list(graph.get("unmatched_endpoint_nodes", [])),
                "from_planar_stroke_first_graph": True,
                "from_planar_endpoint_node_graph": False,
            },
            "ordered_cycle": [
                {
                    "stroke_local_i": int(halfedges[int(he_id)]["stroke_local_i"]),
                    "stroke_index": int(halfedges[int(he_id)]["stroke_index"]),
                    "from_node": int(halfedges[int(he_id)]["from_node"]),
                    "to_node": int(halfedges[int(he_id)]["to_node"]),
                    "edge_kind": halfedges[int(he_id)].get("edge_kind", "stroke"),
                }
                for he_id in face.get("halfedge_ids", [])
            ],
            "ordered_loop_points": [[float(x), float(y)] for x, y in ordered_points.tolist()],
            "connector_edges": json_safe_debug_value(face.get("connector_edges", [])),
            "connector_total_gap": float(face.get("connector_total_gap", 0.0)),
            "connector_max_gap": float(face.get("connector_max_gap", 0.0)),
        })

    bounded_faces.sort(key=lambda item: (
        -float(item.get("topology", {}).get("abs_area", 0.0)),
        item.get("topology", {}).get("cycle_strokes", []),
    ))
    if progress_callback is not None:
        progress_callback(
            "planar_face_extraction_finished",
            **context,
            bounded_face_count=int(len(bounded_faces)),
            raw_face_count=int(len(raw_faces)),
            area_face_count=int(len(area_faces)),
            dropped_outer_face_index=None if outer_face_i is None else int(outer_face_i),
            outer_face_selection=json_safe_debug_value(outer_face_selection),
            outer_face_pixel_debug=build_outer_face_pixel_debug_payload(
                area_faces,
                outer_face_i,
                outer_face_selection,
                component_index=context.get("component_index", None),
            ),
            graph_summary={
                "node_count": int(len(centers)),
                "real_edge_count": int(len(real_edges)),
                "connector_edge_count": int(len(connector_edges)),
                "halfedge_count": int(len(halfedges)),
                "unmatched_endpoint_count": int(len(graph.get("unmatched_endpoint_nodes", []))),
                "strategy": graph.get("strategy", ""),
            },
            face_strokes=[item.get("topology", {}).get("cycle_strokes", []) for item in bounded_faces],
        )
    return bounded_faces


def enumerate_endpoint_port_cycles(
    strokes,
    comp,
    endpoint_tol=12.0,
    max_loop_subset_size=14,
    max_cycles=4096,
    max_connectors_per_port=10,
    max_dfs_steps=250000,
    progress_callback=None,
    progress_context=None,
):
    """
    Enumerate closed boundary cycles without transitive endpoint-node merging.

    Each real stroke endpoint is a graph port. Boundary walks alternate between a
    stroke edge and a short virtual connector edge. This keeps nearby alternative
    connections available, which is required for shared-edge or shared-node loops.
    """
    comp = sorted(set(int(i) for i in comp))
    if len(comp) < 2:
        return []
    connectors, _port_points = build_endpoint_port_connectors(
        strokes,
        comp,
        endpoint_tol=endpoint_tol,
        max_connectors_per_port=max_connectors_per_port,
    )
    max_cycle_strokes = len(comp)
    if max_loop_subset_size is not None and int(max_loop_subset_size) > 0:
        # Treat the CLI value as a search guard, not as a hard cap that can hide
        # long visible cap loops split into many short traced strokes.
        max_cycle_strokes = min(len(comp), max(int(max_loop_subset_size), 32))

    context = {} if progress_context is None else dict(progress_context)
    if progress_callback is not None:
        progress_callback(
            "endpoint_cycle_dfs_started",
            **context,
            component_local_indices=list(comp),
            component_stroke_indices=[int(strokes[int(i)]["index"]) for i in comp],
            connector_port_count=int(len(connectors)),
            connector_edge_count=int(sum(len(v) for v in connectors.values())),
            max_cycle_strokes=int(max_cycle_strokes),
            max_cycles=int(max_cycles),
            max_dfs_steps=int(max_dfs_steps),
        )

    cycles = []
    seen = set()
    dfs_steps = 0
    last_progress_cycle_count = 0

    def add_cycle(path, closing_connector):
        nonlocal last_progress_cycle_count
        oriented = []
        connector_edges = []
        for idx, item in enumerate(path):
            si, entry_endpoint, _incoming = item
            next_si, next_entry, next_incoming = path[(idx + 1) % len(path)]
            connector = closing_connector if idx == len(path) - 1 else next_incoming
            next_entry_point = endpoint_port_point(strokes, next_si, next_entry)
            oriented.append({
                "stroke_local_i": int(si),
                "stroke_index": int(strokes[int(si)]["index"]),
                "entry_endpoint": int(entry_endpoint),
                "exit_endpoint": int(1 - entry_endpoint),
                "next_entry_point": [float(next_entry_point[0]), float(next_entry_point[1])],
            })
            from_port = (int(si), int(1 - entry_endpoint))
            to_port = (int(next_si), int(next_entry))
            connector_edges.append({
                "from_stroke": int(strokes[from_port[0]]["index"]),
                "from_endpoint": int(from_port[1]),
                "to_stroke": int(strokes[to_port[0]]["index"]),
                "to_endpoint": int(to_port[1]),
                "distance": float(connector["distance"]),
            })

        stroke_seq = [item["stroke_index"] for item in oriented]
        signature = canonical_cycle_sequence(stroke_seq)
        if signature in seen:
            return
        seen.add(signature)
        loop_points = ordered_loop_points_from_port_cycle(strokes, oriented)
        if len(loop_points) < 3:
            return
        gaps = [float(item["distance"]) for item in connector_edges]
        cycles.append({
            "component_local_indices": [int(si) for si, _entry, _conn in path],
            "topology_kind": "endpoint_port_cycle",
            "topology": {
                "from_endpoint_port_graph": True,
                "cycle_strokes": stroke_seq,
                "connector_total_gap": float(sum(gaps)),
                "connector_max_gap": 0.0 if not gaps else float(max(gaps)),
                "connector_count": int(len(connector_edges)),
            },
            "ordered_cycle": oriented,
            "ordered_loop_points": [[float(x), float(y)] for x, y in loop_points.tolist()],
            "connector_edges": connector_edges,
            "connector_total_gap": float(sum(gaps)),
            "connector_max_gap": 0.0 if not gaps else float(max(gaps)),
        })
        if progress_callback is not None and len(cycles) - last_progress_cycle_count >= 100:
            last_progress_cycle_count = len(cycles)
            progress_callback(
                "endpoint_cycle_dfs_progress",
                **context,
                dfs_steps=int(dfs_steps),
                cycles_found=int(len(cycles)),
                seen_signatures=int(len(seen)),
            )

    def dfs(start_port, path, used_strokes):
        nonlocal dfs_steps
        if len(cycles) >= int(max_cycles) or dfs_steps >= int(max_dfs_steps):
            return
        dfs_steps += 1
        if progress_callback is not None and dfs_steps % 5000 == 0:
            progress_callback(
                "endpoint_cycle_dfs_progress",
                **context,
                dfs_steps=int(dfs_steps),
                cycles_found=int(len(cycles)),
                seen_signatures=int(len(seen)),
                current_start_port=list(start_port),
                current_path_len=int(len(path)),
            )
        current_si, current_entry, _incoming = path[-1]
        exit_port = (int(current_si), int(1 - current_entry))
        for connector in connectors.get(exit_port, []):
            next_port = connector["to"]
            next_si, next_entry = int(next_port[0]), int(next_port[1])
            if next_port == start_port:
                if len(path) >= 2:
                    add_cycle(path, connector)
                continue
            if next_si in used_strokes:
                continue
            if len(path) >= int(max_cycle_strokes):
                continue
            dfs(start_port, path + [(next_si, next_entry, connector)], used_strokes | {next_si})

    for start_si in comp:
        for start_entry in (0, 1):
            start_port = (int(start_si), int(start_entry))
            dfs(start_port, [(int(start_si), int(start_entry), None)], {int(start_si)})
            if len(cycles) >= int(max_cycles) or dfs_steps >= int(max_dfs_steps):
                break
        if len(cycles) >= int(max_cycles) or dfs_steps >= int(max_dfs_steps):
            break

    cycles.sort(key=lambda item: (
        -len(item.get("component_local_indices", [])),
        float(item.get("connector_total_gap", 0.0)),
        item.get("topology", {}).get("cycle_strokes", []),
    ))
    if progress_callback is not None:
        progress_callback(
            "endpoint_cycle_dfs_finished",
            **context,
            dfs_steps=int(dfs_steps),
            cycles_found=int(len(cycles)),
            seen_signatures=int(len(seen)),
            hit_max_cycles=bool(len(cycles) >= int(max_cycles)),
            hit_max_dfs_steps=bool(dfs_steps >= int(max_dfs_steps)),
        )
    return cycles


def cap_loop_record_quality_key(record):
    """Quality key for choosing one geometry when the stroke set is identical."""
    cand = record.get("candidate", {}) or {}
    stroke_set = record.get("stroke_set", set())
    return (
        float(cand.get("connector_total_gap", 0.0)),
        float(cand.get("connector_max_gap", 0.0)),
        -int(cand.get("enclosed_area", 0)),
        -int(cand.get("area", 0)),
        -len(stroke_set),
        tuple(int(x) for x in sorted(stroke_set)),
    )


def deduplicate_cap_loop_records_by_stroke_set(records):
    """Keep the best ordered loop for each exact stroke set."""
    best_by_set = {}
    for record in records:
        stroke_set = frozenset(int(x) for x in record.get("stroke_set", set()))
        if not stroke_set:
            continue
        record = dict(record)
        record["stroke_set"] = stroke_set
        key = tuple(sorted(stroke_set))
        quality = cap_loop_record_quality_key(record)
        old = best_by_set.get(key)
        if old is None or quality < old[0]:
            best_by_set[key] = (quality, record)
    return [item[1] for item in best_by_set.values()]


def cap_loop_cover_stats(selected_records, universe_strokes):
    universe = set(int(x) for x in universe_strokes)
    stroke_counts = {}
    union = set()
    total_gap = 0.0
    max_gap = 0.0
    total_area = 0
    total_enclosed_area = 0
    total_stroke_memberships = 0
    interior_occupied = None
    interior_overlap_area = 0

    for record in selected_records:
        stroke_set = set(int(x) for x in record.get("stroke_set", set()))
        cand = record.get("candidate", {}) or {}
        union |= stroke_set
        total_stroke_memberships += len(stroke_set)
        total_gap += float(cand.get("connector_total_gap", 0.0))
        max_gap = max(max_gap, float(cand.get("connector_max_gap", 0.0)))
        total_area += int(cand.get("area", 0))
        total_enclosed_area += int(cand.get("enclosed_area", 0))
        for sid in stroke_set:
            stroke_counts[sid] = stroke_counts.get(sid, 0) + 1

        interior_mask = cand.get("enclosed_mask", None)
        if interior_mask is None:
            interior_mask = record.get("cover_source_mask", None)
        if interior_mask is not None:
            current = np.asarray(interior_mask) > 0
            if interior_occupied is None:
                interior_occupied = np.zeros_like(current, dtype=bool)
            if interior_occupied.shape == current.shape:
                interior_overlap_area += int(np.count_nonzero(interior_occupied & current))
                interior_occupied |= current

    shared_strokes = sorted(int(sid) for sid, count in stroke_counts.items() if int(count) > 1)
    repeated_count = int(max(0, total_stroke_memberships - len(union)))
    missing = sorted(int(sid) for sid in universe - union)
    extra = sorted(int(sid) for sid in union - universe)
    return {
        "union_strokes": sorted(int(sid) for sid in union),
        "uses_all_component_strokes": bool(union == universe),
        "missing_component_strokes": missing,
        "extra_strokes": extra,
        "shared_strokes": shared_strokes,
        "shared_stroke_count": int(len(shared_strokes)),
        "repeated_stroke_membership_count": repeated_count,
        "total_connector_gap": float(total_gap),
        "max_connector_gap": float(max_gap),
        "total_area": int(total_area),
        "total_enclosed_area": int(total_enclosed_area),
        "interior_overlap_area": int(interior_overlap_area),
        "loop_count": int(len(selected_records)),
    }


def cap_loop_cover_objective(selected_records, universe_strokes):
    stats = cap_loop_cover_stats(selected_records, universe_strokes)
    return (
        int(stats["interior_overlap_area"]),
        float(stats["total_connector_gap"]),
        float(stats["max_connector_gap"]),
        int(stats["loop_count"]),
        -int(stats["shared_stroke_count"]),
        int(stats["repeated_stroke_membership_count"]),
        -int(stats["total_enclosed_area"]),
        tuple(tuple(sorted(int(x) for x in record.get("stroke_set", set()))) for record in selected_records),
    )


def cap_loop_selection_objective_from_meta(meta):
    if not meta or not meta.get("selected", False):
        return (999999999, float("inf"), float("inf"), 999999, 0, 999999, 0, tuple())
    objective = meta.get("cover_objective", None)
    if objective is None:
        return (999999999, float("inf"), float("inf"), 999999, 0, 999999, 0, tuple())
    return tuple(objective)


def select_full_component_loop_records(
    records,
    universe_strokes,
    endpoint_tol=12.0,
    max_cover_count=12,
    max_records_per_stroke=64,
    max_search_states=200000,
    stop_event=None,
):
    """
    Select only loops that explain the whole normalized component.

    Preference order:
      1. a single non-self-intersecting simple loop that uses every stroke;
      2. otherwise, a small family of simple loops whose union uses every stroke.
    """
    universe = set(int(x) for x in universe_strokes)
    meta = {
        "selected": False,
        "mode": "no_full_component_loop_cover",
        "universe_strokes": sorted(universe),
        "record_count_raw": int(len(records)),
        "record_count_unique": 0,
        "missing_component_strokes": sorted(universe),
        "search_states": 0,
        "search_truncated": False,
        "search_cancelled": False,
        "max_cover_count": int(max_cover_count),
        "max_records_per_stroke": int(max_records_per_stroke),
        "max_search_states": int(max_search_states),
    }
    if cap_search_stop_requested(stop_event):
        meta["search_cancelled"] = True
        return [], meta
    if not universe:
        return [], meta

    unique_records = deduplicate_cap_loop_records_by_stroke_set(records)
    unique_records = [
        record for record in unique_records
        if set(record.get("stroke_set", set())).issubset(universe)
    ]
    unique_records.sort(key=cap_loop_record_quality_key)
    meta["record_count_unique"] = int(len(unique_records))
    if not unique_records:
        return [], meta

    full_records = [
        record for record in unique_records
        if set(record.get("stroke_set", set())) == universe
    ]
    if full_records:
        selected = [min(full_records, key=cap_loop_record_quality_key)]
        stats = cap_loop_cover_stats(selected, universe)
        objective = cap_loop_cover_objective(selected, universe)
        meta.update(stats)
        meta.update({
            "selected": True,
            "mode": "single_simple_loop_uses_all_component_strokes",
            "cover_objective": json_safe_debug_value(objective),
        })
        return selected, meta

    stroke_to_records = {sid: [] for sid in universe}
    for record in unique_records:
        for sid in set(record.get("stroke_set", set())):
            if int(sid) in stroke_to_records:
                stroke_to_records[int(sid)].append(record)

    missing = sorted(int(sid) for sid, items in stroke_to_records.items() if not items)
    meta["missing_component_strokes"] = missing
    if missing:
        return [], meta

    def choice_key_for_stroke(record):
        cand = record.get("candidate", {}) or {}
        stroke_set = set(record.get("stroke_set", set()))
        return (
            float(cand.get("connector_total_gap", 0.0)),
            float(cand.get("connector_max_gap", 0.0)),
            -len(stroke_set),
            -int(cand.get("enclosed_area", 0)),
            tuple(sorted(int(x) for x in stroke_set)),
        )

    for sid, items in stroke_to_records.items():
        items.sort(key=choice_key_for_stroke)
        if max_records_per_stroke is not None and int(max_records_per_stroke) > 0:
            stroke_to_records[sid] = items[:int(max_records_per_stroke)]

    record_ids = {id(record): i for i, record in enumerate(unique_records)}
    best_selected = None
    best_objective = None
    search_states = 0
    truncated = False
    visited_states = set()
    max_cover_count = max(1, int(max_cover_count))
    max_search_states = max(1, int(max_search_states))

    def dfs(selected, selected_ids, covered):
        nonlocal best_selected, best_objective, search_states, truncated
        if cap_search_stop_requested(stop_event):
            meta["search_cancelled"] = True
            return
        if search_states >= max_search_states:
            truncated = True
            return
        search_states += 1

        if covered == universe:
            objective = cap_loop_cover_objective(selected, universe)
            if best_objective is None or objective < best_objective:
                best_objective = objective
                best_selected = list(selected)
            return

        if len(selected) >= max_cover_count:
            return

        if best_objective is not None:
            partial_gap = sum(
                float((record.get("candidate", {}) or {}).get("connector_total_gap", 0.0))
                for record in selected
            )
            if partial_gap > float(best_objective[1]):
                return

        state_key = (
            tuple(sorted(selected_ids)),
            tuple(sorted(int(sid) for sid in covered)),
        )
        if state_key in visited_states:
            return
        visited_states.add(state_key)

        uncovered = universe - covered
        pivot = min(
            uncovered,
            key=lambda sid: sum(
                1 for record in stroke_to_records.get(int(sid), [])
                if record_ids[id(record)] not in selected_ids
                and bool(set(record.get("stroke_set", set())) - covered)
            ),
        )

        choices = []
        for record in stroke_to_records.get(int(pivot), []):
            rid = record_ids[id(record)]
            if rid in selected_ids:
                continue
            stroke_set = set(record.get("stroke_set", set()))
            new_strokes = stroke_set - covered
            if not new_strokes:
                continue
            cand = record.get("candidate", {}) or {}
            overlap = len(stroke_set & covered)
            choices.append((
                (
                    float(cand.get("connector_total_gap", 0.0)),
                    float(cand.get("connector_max_gap", 0.0)),
                    -overlap,
                    -len(new_strokes),
                    -int(cand.get("enclosed_area", 0)),
                    tuple(sorted(int(x) for x in stroke_set)),
                ),
                rid,
                record,
            ))
        choices.sort(key=lambda item: item[0])

        for _key, rid, record in choices:
            if cap_search_stop_requested(stop_event):
                meta["search_cancelled"] = True
                break
            stroke_set = set(record.get("stroke_set", set()))
            dfs(
                selected + [record],
                selected_ids | {rid},
                covered | stroke_set,
            )
            if truncated:
                break

    dfs([], set(), set())
    meta["search_states"] = int(search_states)
    meta["search_truncated"] = bool(truncated)
    if meta.get("search_cancelled", False):
        return [], meta

    if best_selected is None:
        covered = set()
        for record in unique_records:
            covered |= set(record.get("stroke_set", set()))
        meta["missing_component_strokes"] = sorted(int(sid) for sid in universe - covered)
        return [], meta

    best_selected.sort(key=cap_loop_record_quality_key)
    stats = cap_loop_cover_stats(best_selected, universe)
    meta.update(stats)
    meta.update({
        "selected": True,
        "mode": "shared_stroke_simple_loop_cover",
        "cover_objective": json_safe_debug_value(best_objective),
    })
    return best_selected, meta


def cap_loop_selection_needs_extended_search(meta, endpoint_tol=12.0):
    if not meta or not meta.get("selected", False):
        return True
    if meta.get("mode") == "single_simple_loop_uses_all_component_strokes":
        return False
    if int(meta.get("interior_overlap_area", 0)) > 0:
        return True
    total_gap = float(meta.get("total_connector_gap", 0.0))
    gap_limit = max(float(endpoint_tol or 0.0) * 3.0, 1.0)
    return bool(meta.get("search_truncated", False)) or total_gap > gap_limit


def cap_loop_selection_has_full_component_cover(meta):
    """Return True when selected loops explain every normalized component stroke."""
    if not meta or not meta.get("selected", False):
        return False
    if bool(meta.get("search_cancelled", False)):
        return False
    if meta.get("missing_component_strokes", []):
        return False
    if meta.get("extra_strokes", []):
        return False
    if "uses_all_component_strokes" in meta:
        return bool(meta.get("uses_all_component_strokes", False))
    universe = set(int(x) for x in meta.get("universe_strokes", []))
    union = set(int(x) for x in meta.get("union_strokes", []))
    return bool(universe) and union == universe


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
    min_bbox_area=0,
    min_total_arc=0.0,
    thickness=2,
    max_loop_subset_size=14,
    progress_callback=None,
    stop_event=None,
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
      - Rebuild the existing endpoint-node graph for the normalized component.
      - Traverse its planar half-edge embedding to get bounded faces directly.
      - If those bounded faces cannot cover the normalized component without
        interior overlap, this cluster contributes no cap candidate.
    """
    side_ids = {int(s["index"]) for s in side_inliers}
    non_side_infos = [s for s in infos if int(s["index"]) not in side_ids]
    if cap_search_stop_requested(stop_event):
        return []
    if progress_callback is not None:
        progress_callback(
            "cap_candidate_extraction_started",
            side_indices=sorted(side_ids),
            input_pool_count=int(len(infos)),
            non_side_count=int(len(non_side_infos)),
            endpoint_tol=float(endpoint_tol),
            min_pixels=int(min_pixels),
            min_enclosed_area=int(min_enclosed_area),
            min_bbox_area=int(min_bbox_area),
            min_total_arc=float(min_total_arc),
            max_loop_subset_size=int(max_loop_subset_size),
        )
    if not non_side_infos:
        if progress_callback is not None:
            progress_callback("cap_candidate_extraction_finished", candidate_count=0, reason="empty_non_side_pool")
        return []

    comps = connected_components_by_endpoint_proximity(non_side_infos, endpoint_tol=endpoint_tol)
    if progress_callback is not None:
        progress_callback(
            "connected_components_built",
            component_count=int(len(comps)),
            components=[
                {
                    "component_index": int(i),
                    "local_indices": [int(x) for x in comp],
                    "stroke_indices": [int(non_side_infos[int(x)]["index"]) for x in comp],
                    "stroke_count": int(len(comp)),
                }
                for i, comp in enumerate(comps)
            ],
        )

    candidates = []
    seen = set()

    for component_i, comp in enumerate(comps):
        if cap_search_stop_requested(stop_event):
            if progress_callback is not None:
                progress_callback(
                    "cap_candidate_extraction_cancelled",
                    component_index=int(component_i),
                    candidate_count=int(len(candidates)),
                )
            return candidates
        comp = list(map(int, comp))
        if progress_callback is not None:
            progress_callback(
                "component_started",
                component_index=int(component_i),
                local_indices=comp,
                stroke_indices=[int(non_side_infos[i]["index"]) for i in comp],
                stroke_count=int(len(comp)),
            )
        normalized = normalize_cap_loop_component(
            non_side_infos,
            comp,
            endpoint_tol=endpoint_tol,
        )
        final_comp = list(map(int, normalized["component_local_indices"]))
        if not final_comp:
            if progress_callback is not None:
                progress_callback(
                    "component_skipped",
                    component_index=int(component_i),
                    reason="empty_after_normalization",
                    trace=list(normalized.get("trace", [])),
                )
            continue
        if not normalized.get("closed", False):
            if progress_callback is not None:
                progress_callback(
                    "component_skipped",
                    component_index=int(component_i),
                    reason="not_closed_after_normalization",
                    normalized_local_indices=final_comp,
                    normalized_stroke_indices=[int(non_side_infos[i]["index"]) for i in final_comp],
                    removed_open_branch_strokes=list(normalized.get("removed_open_branch_strokes", [])),
                    removed_post_loop_self_strokes=list(normalized.get("removed_post_loop_self_strokes", [])),
                    trace=list(normalized.get("trace", [])),
                )
            continue

        removed_branch = normalized.get("removed_open_branch_strokes", [])
        removed_self_loops = normalized.get("removed_post_loop_self_strokes", [])
        component_strokes = [int(non_side_infos[i]["index"]) for i in final_comp]
        component_universe = set(component_strokes)
        if progress_callback is not None:
            progress_callback(
                "component_normalized_closed",
                component_index=int(component_i),
                normalized_local_indices=final_comp,
                normalized_stroke_indices=component_strokes,
                removed_open_branch_strokes=list(removed_branch),
                removed_post_loop_self_strokes=list(removed_self_loops),
                trace=list(normalized.get("trace", [])),
            )

        def build_cap_record(loop_info):
            loop_comp = list(map(int, loop_info.get("component_local_indices", [])))
            if not loop_comp:
                return None

            if loop_info.get("ordered_loop_points", None) is not None:
                cand = candidate_from_ordered_loop(image_shape, non_side_infos, loop_info, thickness=thickness)
                if cand is None:
                    return None
            else:
                cand = candidate_from_stroke_loop(image_shape, non_side_infos, loop_comp, thickness=thickness)
            if cand["area"] < min_pixels:
                return None
            if cand.get("enclosed_area", 0) < min_enclosed_area:
                return None
            if cand.get("bbox_area", 0) < min_bbox_area:
                return None
            if cand.get("total_arc", 0.0) < min_total_arc:
                return None
            _sweep_source, sweep_source_method = cap_ordered_filled_sweep_source_mask(
                cand,
                non_side_infos,
                endpoint_tol=endpoint_tol,
                thickness=thickness,
            )
            if _sweep_source is None:
                return None
            cand["sweep_source_mask_method"] = sweep_source_method

            topology_kind = loop_info.get("topology_kind", "component")
            if topology_kind == "decomposed_simple_loop":
                cand["loop_detection"] = "component_shared_edge_decomposed_simple_loop"
            elif topology_kind == "endpoint_port_cycle":
                cand["loop_detection"] = "component_endpoint_port_cycle"
            elif topology_kind == "geometry_bounded_face":
                cand["loop_detection"] = "component_geometry_bounded_face"
            elif topology_kind == "planar_bounded_face":
                cand["loop_detection"] = "component_planar_bounded_face"
            elif topology_kind == "planar_graph_simple_cycle":
                cand["loop_detection"] = "component_planar_graph_simple_cycle"
            elif removed_self_loops and removed_branch:
                cand["loop_detection"] = "component_iterative_prune_and_self_loop_removal"
            elif removed_self_loops:
                cand["loop_detection"] = "component_postloop_self_loop_removed"
            else:
                cand["loop_detection"] = "component_pruned_open_branches" if removed_branch else "component"
            if topology_kind == "multi_simple_loop_edge_disjoint":
                cand["loop_detection"] = "component_multi_simple_loop_edge_disjoint"
            elif topology_kind == "endpoint_proximity_closed_component":
                cand["loop_detection"] = "component_endpoint_proximity_closed"
            elif topology_kind == "planar_bounded_face":
                cand["loop_detection"] = "component_planar_bounded_face"
            elif topology_kind == "planar_graph_simple_cycle":
                cand["loop_detection"] = "component_planar_graph_simple_cycle"
            elif topology_kind == "simple_loop" and cand["loop_detection"] == "component":
                cand["loop_detection"] = "component_simple_loop"

            cand["component_local_indices"] = list(map(int, loop_comp))
            cand["source_normalized_component_local_indices"] = list(map(int, final_comp))
            cand["source_normalized_component_strokes"] = list(component_strokes)
            cand["component_normalization_trace"] = list(normalized.get("trace", []))
            cand["component_topology"] = dict(loop_info.get("topology", {}))
            cand["topology_kind"] = topology_kind
            cand["pruned_branch_strokes"] = list(removed_branch)
            cand["removed_post_loop_self_strokes"] = list(removed_self_loops)

            stroke_set = frozenset(int(sid) for sid in cand.get("stroke_indices", []))
            if not stroke_set:
                return None
            return {
                "candidate": cand,
                "loop_info": loop_info,
                "stroke_set": stroke_set,
                "cover_source_mask": _sweep_source,
            }

        def collect_geometry_face_records():
            loop_infos = extract_geometry_supported_face_loop_infos(
                non_side_infos,
                final_comp,
                endpoint_tol=endpoint_tol,
                progress_callback=progress_callback,
                progress_context={
                    "component_index": int(component_i),
                    "component_strokes": component_strokes,
                },
                stop_event=stop_event,
            )
            loop_infos.sort(key=lambda item: (
                -float(item.get("topology", {}).get("abs_area", 0.0)),
                item.get("topology", {}).get("cycle_strokes", []),
            ))
            records = []
            processed_loop_count = 0
            for loop_info in loop_infos:
                if cap_search_stop_requested(stop_event):
                    break
                processed_loop_count += 1
                record = build_cap_record(loop_info)
                if record is not None:
                    records.append(record)
                if progress_callback is not None and (
                    processed_loop_count == 1
                    or processed_loop_count == len(loop_infos)
                    or processed_loop_count % 25 == 0
                ):
                    progress_callback(
                        "geometry_face_record_collection_progress",
                        component_index=int(component_i),
                        processed_loop_count=int(processed_loop_count),
                        raw_loop_count=int(len(loop_infos)),
                        valid_loop_record_count=int(len(records)),
                        last_loop_strokes=loop_info.get("topology", {}).get("cycle_strokes", []),
                        last_record_valid=bool(record is not None),
                    )
            if progress_callback is not None:
                progress_callback(
                    "geometry_face_record_collection_finished",
                    component_index=int(component_i),
                    processed_loop_count=int(processed_loop_count),
                    raw_loop_count=int(len(loop_infos)),
                    valid_loop_record_count=int(len(records)),
                )
            return records, {
                "source": "geometry_bounded_faces",
                "raw_loop_count": int(len(loop_infos)),
                "processed_loop_count": int(processed_loop_count),
                "valid_loop_record_count": int(len(records)),
            }

        def infer_cover_mode(selected_records, fallback_mode):
            topology_kinds = {
                str((record.get("loop_info", {}) or {}).get("topology_kind", ""))
                for record in selected_records
            }
            if "geometry_bounded_face" in topology_kinds:
                return "geometry_bounded_face_cover"
            if "planar_graph_simple_cycle" in topology_kinds:
                return "planar_graph_cycle_cover"
            if "planar_bounded_face" in topology_kinds:
                return "planar_bounded_face_cover"
            return fallback_mode

        def collect_planar_face_records():
            loop_infos = extract_planar_face_loop_infos(
                non_side_infos,
                final_comp,
                endpoint_tol=endpoint_tol,
                progress_callback=progress_callback,
                progress_context={
                    "component_index": int(component_i),
                    "component_strokes": component_strokes,
                },
                stop_event=stop_event,
            )
            loop_infos.sort(key=lambda item: (
                -float(item.get("topology", {}).get("abs_area", 0.0)),
                item.get("topology", {}).get("cycle_strokes", []),
            ))
            records = []
            processed_loop_count = 0
            for loop_info in loop_infos:
                if cap_search_stop_requested(stop_event):
                    break
                processed_loop_count += 1
                record = build_cap_record(loop_info)
                if record is not None:
                    records.append(record)
                if progress_callback is not None and (
                    processed_loop_count == 1
                    or processed_loop_count == len(loop_infos)
                    or processed_loop_count % 25 == 0
                ):
                    progress_callback(
                        "planar_face_record_collection_progress",
                        component_index=int(component_i),
                        processed_loop_count=int(processed_loop_count),
                        raw_loop_count=int(len(loop_infos)),
                        valid_loop_record_count=int(len(records)),
                        last_loop_strokes=loop_info.get("topology", {}).get("cycle_strokes", []),
                        last_record_valid=bool(record is not None),
                    )
            if progress_callback is not None:
                progress_callback(
                    "planar_face_record_collection_finished",
                    component_index=int(component_i),
                    processed_loop_count=int(processed_loop_count),
                    raw_loop_count=int(len(loop_infos)),
                    valid_loop_record_count=int(len(records)),
                )
            return records, {
                "source": "planar_bounded_faces",
                "raw_loop_count": int(len(loop_infos)),
                "processed_loop_count": int(processed_loop_count),
                "valid_loop_record_count": int(len(records)),
            }

        def collect_planar_graph_cycle_records():
            cycle_edge_guard = None
            if max_loop_subset_size is not None and int(max_loop_subset_size) > 0:
                cycle_edge_guard = max(2 * int(max_loop_subset_size) + 4, 32)
            loop_infos = enumerate_planar_graph_simple_cycle_loop_infos(
                non_side_infos,
                final_comp,
                endpoint_tol=endpoint_tol,
                max_cycle_edges=cycle_edge_guard,
                progress_callback=progress_callback,
                progress_context={
                    "component_index": int(component_i),
                    "component_strokes": component_strokes,
                },
                stop_event=stop_event,
            )
            loop_infos.sort(key=lambda item: (
                -float(item.get("topology", {}).get("abs_area", 0.0)),
                float(item.get("connector_total_gap", 0.0)),
                item.get("topology", {}).get("cycle_strokes", []),
            ))
            records = []
            processed_loop_count = 0
            for loop_info in loop_infos:
                if cap_search_stop_requested(stop_event):
                    break
                processed_loop_count += 1
                record = build_cap_record(loop_info)
                if record is not None:
                    records.append(record)
                if progress_callback is not None and (
                    processed_loop_count == 1
                    or processed_loop_count == len(loop_infos)
                    or processed_loop_count % 25 == 0
                ):
                    progress_callback(
                        "planar_graph_cycle_record_collection_progress",
                        component_index=int(component_i),
                        processed_loop_count=int(processed_loop_count),
                        raw_loop_count=int(len(loop_infos)),
                        valid_loop_record_count=int(len(records)),
                        last_loop_strokes=loop_info.get("topology", {}).get("cycle_strokes", []),
                        last_record_valid=bool(record is not None),
                    )
            if progress_callback is not None:
                progress_callback(
                    "planar_graph_cycle_record_collection_finished",
                    component_index=int(component_i),
                    processed_loop_count=int(processed_loop_count),
                    raw_loop_count=int(len(loop_infos)),
                    valid_loop_record_count=int(len(records)),
                )
            return records, {
                "source": "planar_graph_simple_cycles",
                "raw_loop_count": int(len(loop_infos)),
                "processed_loop_count": int(processed_loop_count),
                "valid_loop_record_count": int(len(records)),
            }

        geometry_records, geometry_search_meta = collect_geometry_face_records()
        search_rounds = [geometry_search_meta]
        if not geometry_records:
            if progress_callback is not None:
                progress_callback(
                    "component_skipped",
                    component_index=int(component_i),
                    reason="geometry_faces_not_found",
                    geometry_record_count=0,
                    selection_meta={
                        "selected": False,
                        "mode": "no_geometry_face_records",
                        "search_rounds": list(search_rounds),
                    },
                )
            continue

        selected_records, selection_meta = select_full_component_loop_records(
            geometry_records,
            component_universe,
            endpoint_tol=endpoint_tol,
            max_cover_count=max(12, len(geometry_records)),
            stop_event=stop_event,
        )
        selection_meta["search_rounds"] = list(search_rounds)
        if selected_records:
            selection_meta["mode"] = infer_cover_mode(selected_records, selection_meta.get("mode", "geometry_bounded_face_cover"))
        if progress_callback is not None:
            progress_callback(
                "component_geometry_loop_selection_finished",
                component_index=int(component_i),
                selected_record_count=int(len(selected_records)),
                geometry_record_count=int(len(geometry_records)),
                selection_meta=json_safe_debug_value(selection_meta),
            )

        cover_is_full = cap_loop_selection_has_full_component_cover(selection_meta)
        if not cover_is_full:
            partial_geometry_records = deduplicate_cap_loop_records_by_stroke_set(geometry_records)
            partial_geometry_records.sort(key=cap_loop_record_quality_key)
            partial_union = set()
            for record in partial_geometry_records:
                partial_union |= set(int(x) for x in record.get("stroke_set", set()))
            if partial_geometry_records:
                selection_meta = dict(selection_meta)
                selection_meta.update(cap_loop_cover_stats(partial_geometry_records, component_universe))
                selection_meta.update({
                    "selected": True,
                    "selected_partial": True,
                    "mode": "geometry_bounded_face_partial_component",
                    "search_rounds": list(search_rounds),
                    "union_strokes": sorted(int(x) for x in partial_union),
                    "universe_strokes": sorted(int(x) for x in component_universe),
                    "missing_component_strokes": sorted(int(x) for x in (component_universe - partial_union)),
                    "extra_strokes": [],
                    "uses_all_component_strokes": bool(partial_union == component_universe),
                    "selected_record_count": int(len(partial_geometry_records)),
                })
                selected_records = partial_geometry_records
                if progress_callback is not None:
                    progress_callback(
                        "component_geometry_partial_faces_selected",
                        component_index=int(component_i),
                        selected_record_count=int(len(selected_records)),
                        geometry_record_count=int(len(geometry_records)),
                        covered_strokes=sorted(int(x) for x in partial_union),
                        missing_component_strokes=selection_meta.get("missing_component_strokes", []),
                        selection_meta=json_safe_debug_value(selection_meta),
                    )
            else:
                if progress_callback is not None:
                    progress_callback(
                        "component_skipped",
                        component_index=int(component_i),
                        reason="geometry_faces_do_not_form_required_cover",
                        geometry_record_count=int(len(geometry_records)),
                        selection_meta=json_safe_debug_value(selection_meta),
                    )
                continue

        for cover_index, record in enumerate(selected_records):
            cand = record.get("candidate")
            if cand is None:
                continue
            key = tuple(sorted(int(sid) for sid in cand.get("stroke_indices", [])))
            if key in seen:
                if progress_callback is not None:
                    progress_callback(
                        "candidate_skipped_duplicate",
                        component_index=int(component_i),
                        cover_index=int(cover_index),
                        stroke_indices=list(key),
                    )
                continue
            seen.add(key)

            cover_stats = cap_loop_cover_stats(selected_records, component_universe)
            selection_payload = dict(selection_meta)
            selection_payload.update(cover_stats)
            cand["component_loop_cover_selected"] = True
            cand["component_loop_cover_complete"] = bool(cap_loop_selection_has_full_component_cover(selection_meta))
            cand["component_loop_cover_partial"] = not bool(cand["component_loop_cover_complete"])
            cand["component_loop_cover_index"] = int(cover_index)
            cand["component_loop_cover_count"] = int(len(selected_records))
            cand["component_loop_selection"] = json_safe_debug_value(selection_payload)
            cand["component_loop_cover_union_strokes"] = sorted(
                int(x) for x in selection_payload.get("union_strokes", sorted(component_universe))
            )
            cand["component_loop_cover_shared_strokes"] = list(cover_stats.get("shared_strokes", []))

            mode = str(selection_meta.get("mode", ""))
            if mode == "single_simple_loop_uses_all_component_strokes":
                cand["loop_detection"] = "component_single_simple_loop_uses_all_strokes"
            elif mode == "shared_stroke_simple_loop_cover":
                cand["loop_detection"] = "component_shared_stroke_loop_cover_simple_loop"
            elif mode == "geometry_bounded_face_cover":
                cand["loop_detection"] = "component_geometry_bounded_face_cover"
            elif mode == "geometry_bounded_face_partial_component":
                cand["loop_detection"] = "component_geometry_bounded_face_partial_component"
            elif mode == "planar_bounded_face_cover":
                cand["loop_detection"] = "component_planar_bounded_face_cover"
            elif mode == "planar_graph_cycle_cover":
                cand["loop_detection"] = "component_planar_graph_cycle_cover"
            elif mode == "single_full_component_fallback_loop":
                cand["loop_detection"] = "component_single_full_component_fallback_loop"
            candidates.append(cand)
            if progress_callback is not None:
                progress_callback(
                    "candidate_added",
                    component_index=int(component_i),
                    cover_index=int(cover_index),
                    candidate_count=int(len(candidates)),
                    candidate=summarize_cap_candidate_for_status(cand),
                )

    candidates.sort(key=lambda c: (c.get("enclosed_area", 0), c["score"]), reverse=True)
    if progress_callback is not None:
        progress_callback(
            "cap_candidate_extraction_finished",
            candidate_count=int(len(candidates)),
            candidates=[summarize_cap_candidate_for_status(c) for c in candidates[:20]],
        )
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


def cap_search_stop_requested(stop_event=None):
    if stop_event is None:
        return False
    try:
        return bool(stop_event.is_set())
    except Exception:
        return False


def cap_search_request_stop(stop_event=None):
    if stop_event is None:
        return
    try:
        stop_event.set()
    except Exception:
        pass


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
    details["best_cap_bbox"] = None
    details["best_cap_bbox_area"] = 0
    details["best_cap_center"] = None
    details["best_cap_total_arc"] = 0.0
    details["best_cap_topology_kind"] = ""
    details["best_cap_loop_detection"] = ""
    details["best_cap_topology_cycle_count"] = 0
    details["best_cap_topology_simple_cycle_count"] = 0
    details["best_cap_topology_edge_disjoint_cover"] = False
    details["best_cap_topology_edge_disjoint_cover_count"] = 0
    details["best_cap_topology_self_loop_strokes"] = []
    details["best_cap_component_loop_selection_mode"] = ""
    details["best_cap_component_loop_cover_count"] = 0
    details["best_cap_component_loop_cover_union_strokes"] = []
    details["best_cap_component_loop_cover_shared_strokes"] = []
    details["best_cap_component_loop_cover_total_gap"] = 0.0
    details["best_cap_component_loop_cover_interior_overlap_area"] = 0
    details["cap_candidate_evaluated_count"] = 0
    details["sweep_gate_enabled"] = False
    details["best_sweep_valid"] = False
    details["best_sweep_iou"] = 0.0
    details["best_sweep_intersection"] = 0
    details["best_sweep_union"] = 0
    details["best_sweep_area"] = 0
    details["best_sweep_copy_side_stroke"] = None
    details["best_sweep_copy_reason"] = ""
    details["best_sweep_mask_source"] = ""
    details["best_sweep_passed"] = False
    details["cap_found_but_sweep_rejected"] = False
    details["side_cap_connect_enabled"] = False
    details["side_cap_connect_tol"] = 0.0
    details["side_cap_connect_passed"] = True
    details["side_cap_connected_count"] = 0
    details["side_cap_side_count"] = 0
    details["side_cap_disconnected_strokes"] = []
    details["side_cap_ignored_disconnected_strokes"] = []
    details["side_cap_range_checked_count"] = 0
    details["side_cap_range_method"] = ""
    details["side_cap_connection_details"] = []
    details["cap_found_but_side_cap_disconnected"] = False
    details["selection_passed"] = False
    details["invalid_no_cap"] = False
    details["selected_by_cap_validation"] = False
    details["selected_by_iou_fallback"] = False
    details["cap_validation_fallback_reason"] = ""
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
    details["best_cap_bbox"] = best_cap.get("bbox", None)
    details["best_cap_bbox_area"] = int(best_cap.get("bbox_area", 0))
    details["best_cap_center"] = center_tuple
    details["best_cap_total_arc"] = float(best_cap.get("total_arc", 0.0))
    topology = best_cap.get("component_topology", {}) or {}
    details["best_cap_topology_kind"] = str(best_cap.get("topology_kind", ""))
    details["best_cap_loop_detection"] = str(best_cap.get("loop_detection", ""))
    details["best_cap_topology_cycle_count"] = int(topology.get("cycle_count", 0))
    details["best_cap_topology_simple_cycle_count"] = int(topology.get("simple_cycle_count", 0))
    details["best_cap_topology_edge_disjoint_cover"] = bool(topology.get("edge_disjoint_cycle_cover_found", False))
    details["best_cap_topology_edge_disjoint_cover_count"] = int(topology.get("edge_disjoint_cycle_cover_count", 0))
    details["best_cap_topology_self_loop_strokes"] = list(topology.get("self_loop_strokes", []))
    selection = best_cap.get("component_loop_selection", {}) or {}
    details["best_cap_component_loop_selection_mode"] = str(selection.get("mode", ""))
    details["best_cap_component_loop_cover_count"] = int(selection.get("loop_count", best_cap.get("component_loop_cover_count", 0)))
    details["best_cap_component_loop_cover_union_strokes"] = list(selection.get("union_strokes", best_cap.get("component_loop_cover_union_strokes", [])))
    details["best_cap_component_loop_cover_shared_strokes"] = list(selection.get("shared_strokes", best_cap.get("component_loop_cover_shared_strokes", [])))
    details["best_cap_component_loop_cover_total_gap"] = float(selection.get("total_connector_gap", 0.0))
    details["best_cap_component_loop_cover_interior_overlap_area"] = int(selection.get("interior_overlap_area", 0))


def fill_best_cap_sweep_details(details, sweep_info, *, gate_enabled=False, stop_thresh=0.0):
    """Store cap-sweep similarity metadata for one entry."""
    details["sweep_gate_enabled"] = bool(gate_enabled)
    if not sweep_info:
        return

    details["best_sweep_valid"] = bool(sweep_info.get("valid", False))
    details["best_sweep_iou"] = float(sweep_info.get("iou", 0.0))
    details["best_sweep_intersection"] = int(sweep_info.get("intersection", 0))
    details["best_sweep_union"] = int(sweep_info.get("union", 0))
    details["best_sweep_area"] = int(sweep_info.get("sweep_area", 0))
    details["best_sweep_copy_side_stroke"] = sweep_info.get("copy_side_stroke", None)
    details["best_sweep_copy_reason"] = str(sweep_info.get("copy_reason", ""))
    details["best_sweep_mask_source"] = str(sweep_info.get("mask_source", ""))
    passed = bool(sweep_info.get("valid", False)) and float(sweep_info.get("iou", 0.0)) >= float(stop_thresh)
    details["best_sweep_passed"] = bool(passed)


def min_distance_to_points(point, points):
    if points is None or len(points) == 0:
        return float("inf")
    p = np.asarray(point, dtype=np.float64)
    pts = np.asarray(points, dtype=np.float64)
    d = pts - p[None, :]
    return float(np.sqrt(np.min(np.sum(d * d, axis=1))))


def cap_candidate_point_cloud(cap_candidate, infos):
    if cap_candidate is None:
        return np.empty((0, 2), dtype=np.float64)
    cap_ids = {int(i) for i in cap_candidate.get("stroke_indices", [])}
    chunks = []
    for s in infos:
        if int(s.get("index", -1)) in cap_ids:
            chunks.append(np.asarray(s["points"], dtype=np.float64).reshape(-1, 2))
    if chunks:
        return np.vstack(chunks)

    mask = cap_candidate.get("mask", None)
    if mask is not None:
        ys, xs = np.where(mask > 0)
        if len(xs) > 0:
            return np.column_stack([xs, ys]).astype(np.float64)
    return np.empty((0, 2), dtype=np.float64)


def odd_kernel_size_from_tolerance(tol, min_size=5, max_size=51):
    size = int(round(float(tol or 0.0) * 2.0 + 1.0))
    size = max(int(min_size), min(int(max_size), size))
    if size % 2 == 0:
        size += 1
    return size


def cap_candidate_stroke_infos(cap_candidate, infos):
    if cap_candidate is None or infos is None:
        return []
    lookup = {int(s.get("index", -1)): s for s in infos}
    return [
        lookup[int(sid)]
        for sid in cap_candidate.get("stroke_indices", [])
        if int(sid) in lookup
    ]


def ordered_loop_polylines_from_strokes(
    strokes,
    endpoint_tol=12.0,
    allow_open_endpoint_close=False,
    validate_gap_lengths=True,
):
    """Order stroke graph components into closed or near-closed cap polylines."""
    strokes = list(strokes or [])
    if not strokes:
        return []

    endpoint_nodes, _centers = stroke_endpoint_node_ids(strokes, endpoint_tol=endpoint_tol)
    node_to_edges = {}
    for edge_idx, (a, b) in enumerate(endpoint_nodes):
        if int(a) == int(b):
            return []
        node_to_edges.setdefault(int(a), []).append(edge_idx)
        node_to_edges.setdefault(int(b), []).append(edge_idx)

    loops = []
    edge_components = connected_components_of_stroke_graph(len(strokes), endpoint_nodes)
    for comp_edges in edge_components:
        comp_edges = set(int(e) for e in comp_edges)
        comp_nodes = set()
        comp_node_degree = {}
        for edge_idx in comp_edges:
            a, b = endpoint_nodes[edge_idx]
            a = int(a)
            b = int(b)
            comp_nodes.add(a)
            comp_nodes.add(b)
            comp_node_degree[a] = comp_node_degree.get(a, 0) + 1
            comp_node_degree[b] = comp_node_degree.get(b, 0) + 1

        degree_one_nodes = [n for n, d in comp_node_degree.items() if int(d) == 1]
        if not all(int(d) in (1, 2) for d in comp_node_degree.values()):
            return []
        if len(degree_one_nodes) == 0:
            start_edge = min(comp_edges)
            current_node = int(endpoint_nodes[start_edge][0])
            stop_node = current_node
            closed_cycle = True
        elif allow_open_endpoint_close and len(degree_one_nodes) == 2:
            current_node = int(degree_one_nodes[0])
            stop_node = int(degree_one_nodes[1])
            closed_cycle = False
        else:
            return []

        previous_edge = None
        used_this_loop = set()
        ordered_chunks = []

        for _ in range(len(strokes) + 1):
            candidate_edges = [
                int(e)
                for e in node_to_edges.get(current_node, [])
                if int(e) in comp_edges and int(e) != previous_edge and int(e) not in used_this_loop
            ]
            if previous_edge is None and closed_cycle and start_edge in candidate_edges:
                current_edge = int(start_edge)
            elif candidate_edges:
                current_edge = min(candidate_edges)
            elif not closed_cycle and current_node == stop_node and used_this_loop == comp_edges:
                break
            elif closed_cycle and current_node == stop_node and used_this_loop == comp_edges:
                break
            else:
                return []

            a, b = endpoint_nodes[current_edge]
            a = int(a)
            b = int(b)
            pts = np.asarray(strokes[current_edge]["points"], dtype=np.float64).reshape(-1, 2)
            if current_node == a:
                to_node = b
                oriented = pts
            elif current_node == b:
                to_node = a
                oriented = pts[::-1]
            else:
                return []

            if ordered_chunks:
                if validate_gap_lengths:
                    gap = float(np.linalg.norm(ordered_chunks[-1][-1] - oriented[0]))
                    if gap > float(endpoint_tol):
                        return []
                ordered_chunks.append(oriented)
            else:
                ordered_chunks.append(oriented)

            used_this_loop.add(current_edge)
            previous_edge = current_edge
            current_node = to_node
            if current_node == stop_node and used_this_loop == comp_edges:
                break
        else:
            return []

        if used_this_loop != comp_edges:
            return []

        if ordered_chunks:
            if validate_gap_lengths:
                closing_gap = float(np.linalg.norm(ordered_chunks[-1][-1] - ordered_chunks[0][0]))
                if closing_gap > float(endpoint_tol):
                    return []
            polyline = np.vstack([chunk for chunk in ordered_chunks if len(chunk) > 0])
            if len(polyline) >= 3:
                loops.append(polyline)

    return loops


def ordered_loop_polylines_from_nearest_endpoint_matches(strokes, endpoint_tol=12.0):
    """Order closed cap loop(s) using real nearest-endpoint connections."""
    strokes = list(strokes or [])
    if not strokes:
        return []

    matches = build_nearest_endpoint_matches(strokes, endpoint_tol=endpoint_tol)
    endpoint_keys = [(si, ei) for si in range(len(strokes)) for ei in (0, 1)]
    for key in endpoint_keys:
        other = matches.get(key, None)
        if other is None:
            return []
        if int(other[0]) == int(key[0]):
            return []
        if matches.get(other, None) != key:
            return []

    unused_strokes = set(range(len(strokes)))
    loops = []
    while unused_strokes:
        start_si = min(unused_strokes)
        start_key = (int(start_si), 0)
        current_key = start_key
        used_this_loop = set()
        chunks = []

        for _ in range(len(strokes) + 1):
            si, ei = int(current_key[0]), int(current_key[1])
            if si in used_this_loop:
                return []
            pts = np.asarray(strokes[si]["points"], dtype=np.float64).reshape(-1, 2)
            oriented = pts if ei == 0 else pts[::-1]
            chunks.append(oriented)
            used_this_loop.add(si)

            exit_key = (si, 1 - ei)
            next_key = matches.get(exit_key, None)
            if next_key is None:
                return []
            next_key = (int(next_key[0]), int(next_key[1]))
            if next_key == start_key:
                break
            current_key = next_key
        else:
            return []

        if not used_this_loop:
            return []
        unused_strokes -= used_this_loop
        polyline = np.vstack([chunk for chunk in chunks if len(chunk) > 0])
        if len(polyline) < 3:
            return []
        loops.append(polyline)

    return loops


def ordered_loop_polylines_from_cap_candidate(cap_candidate):
    """Return precomputed loop points for endpoint-port cycle candidates."""
    if cap_candidate is None:
        return []
    points = cap_candidate.get("ordered_loop_points", None)
    if points is None:
        return []
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 3:
        return []
    return [pts]


def cap_ordered_loop_range_mask(cap_candidate, infos, endpoint_tol=12.0):
    """Fill the actual closed polygon(s) implied by cap stroke endpoint topology."""
    stroke_mask = None if cap_candidate is None else cap_candidate.get("mask", None)
    if stroke_mask is None:
        return None, "no_cap_mask"

    loops = ordered_loop_polylines_from_cap_candidate(cap_candidate)
    method = "cap_endpoint_port_cycle_loop_mask"
    if not loops:
        cap_strokes = cap_candidate_stroke_infos(cap_candidate, infos)
        loops = ordered_loop_polylines_from_nearest_endpoint_matches(
            cap_strokes,
            endpoint_tol=endpoint_tol,
        )
        method = "cap_ordered_nearest_endpoint_loop_mask"
        if not loops:
            loops = ordered_loop_polylines_from_strokes(
                cap_strokes,
                endpoint_tol=endpoint_tol,
                allow_open_endpoint_close=False,
                validate_gap_lengths=True,
            )
            method = "cap_ordered_stroke_loop_mask"
    if not loops:
        return None, "cap_ordered_loop_unavailable"

    range_mask = np.zeros_like(stroke_mask, dtype=np.uint8)
    for loop in loops:
        pts = np.rint(loop).astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(range_mask, [pts], 255, lineType=cv2.LINE_AA)
    range_mask[stroke_mask > 0] = 255
    if np.count_nonzero(range_mask > 0) == 0:
        return None, "cap_ordered_loop_empty"
    return range_mask, method


def cap_ordered_filled_sweep_source_mask(cap_candidate, infos, endpoint_tol=12.0, thickness=2):
    """Build the strict filled cap mask used as the source face for sweep IoU."""
    stroke_mask = None if cap_candidate is None else cap_candidate.get("mask", None)
    if stroke_mask is None or np.count_nonzero(stroke_mask > 0) == 0:
        return None, "no_cap_mask"

    loops = ordered_loop_polylines_from_cap_candidate(cap_candidate)
    method = "cap_endpoint_port_cycle_bridged_filled_mask"
    if not loops:
        cap_strokes = cap_candidate_stroke_infos(cap_candidate, infos)
        if len(cap_strokes) != len(set(int(i) for i in cap_candidate.get("stroke_indices", []))):
            return None, "missing_cap_strokes"
        loops = ordered_loop_polylines_from_nearest_endpoint_matches(
            cap_strokes,
            endpoint_tol=endpoint_tol,
        )
        method = "cap_ordered_nearest_endpoint_bridged_filled_mask"
        if not loops:
            loops = ordered_loop_polylines_from_strokes(
                cap_strokes,
                endpoint_tol=endpoint_tol,
                allow_open_endpoint_close=False,
                validate_gap_lengths=True,
            )
            method = "cap_ordered_endpoint_bridged_filled_mask"
    if not loops:
        return None, "closed_ordered_loop_unavailable"

    boundary = np.zeros_like(stroke_mask, dtype=np.uint8)
    filled = np.zeros_like(stroke_mask, dtype=np.uint8)
    line_thickness = max(1, int(thickness))
    for loop in loops:
        pts = np.rint(loop).astype(np.int32).reshape(-1, 1, 2)
        if len(pts) < 3:
            return None, "closed_ordered_loop_too_short"
        cv2.polylines(boundary, [pts], True, 255, line_thickness, cv2.LINE_AA)
        cv2.fillPoly(filled, [pts], 255, lineType=cv2.LINE_AA)

    source = (filled > 0).astype(np.uint8) * 255
    source[boundary > 0] = 255
    if np.count_nonzero(source > 0) == 0:
        return None, "closed_filled_cap_empty"
    if np.count_nonzero(source > 0) <= np.count_nonzero(boundary > 0):
        return None, "closed_filled_cap_has_no_interior"
    return source, method


def cap_ordered_loop_range_mask_legacy(cap_candidate, infos, endpoint_tol=12.0):
    """Legacy endpoint-node loop fill; kept for debugging comparisons."""
    stroke_mask = None if cap_candidate is None else cap_candidate.get("mask", None)
    if stroke_mask is None:
        return None, "no_cap_mask"

    cap_strokes = cap_candidate_stroke_infos(cap_candidate, infos)
    loops = ordered_loop_polylines_from_strokes(
        cap_strokes,
        endpoint_tol=endpoint_tol,
        allow_open_endpoint_close=False,
        validate_gap_lengths=True,
    )
    if not loops:
        return None, "cap_ordered_loop_unavailable"

    range_mask = np.zeros_like(stroke_mask, dtype=np.uint8)
    for loop in loops:
        pts = np.rint(loop).astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(range_mask, [pts], 255, lineType=cv2.LINE_AA)
    range_mask[stroke_mask > 0] = 255
    if np.count_nonzero(range_mask > 0) == 0:
        return None, "cap_ordered_loop_empty"
    return range_mask, "cap_ordered_stroke_loop_mask"


def cap_candidate_enclosed_range_mask(cap_candidate, infos=None, connect_tol=0.0):
    """Return the cap-stroke enclosed range mask used by side-cap connection gating."""
    if cap_candidate is None:
        return None, "no_cap"

    ordered_mask, ordered_method = cap_ordered_loop_range_mask(
        cap_candidate,
        infos,
        endpoint_tol=max(1.0, float(connect_tol or 0.0)),
    )
    if ordered_mask is not None:
        return ordered_mask, ordered_method

    stroke_mask = cap_candidate.get("mask", None)
    enclosed_mask = cap_candidate.get("enclosed_mask", None)
    if enclosed_mask is not None and np.count_nonzero(enclosed_mask > 0) > 0:
        range_mask = (enclosed_mask > 0).astype(np.uint8) * 255
        if stroke_mask is not None:
            range_mask[stroke_mask > 0] = 255
        method = f"cap_enclosed_mask_fallback_after_{ordered_method}"
    elif stroke_mask is not None and np.count_nonzero(stroke_mask > 0) > 0:
        close_kernel = odd_kernel_size_from_tolerance(connect_tol)
        range_mask = fill_binary_mask(stroke_mask, close_kernel=close_kernel)
        if np.count_nonzero(range_mask > 0) <= np.count_nonzero(stroke_mask > 0):
            range_mask = (stroke_mask > 0).astype(np.uint8) * 255
            method = f"cap_stroke_mask_fallback_after_{ordered_method}"
        else:
            method = f"cap_filled_stroke_mask_k{close_kernel}_fallback_after_{ordered_method}"
    else:
        return None, "no_cap_mask"

    return range_mask, method


def stroke_points_overlap_mask(stroke, mask):
    """Return True when any sampled point of a stroke lies in a binary mask."""
    if mask is None:
        return True
    pts = np.asarray(stroke.get("points", []), dtype=np.float64).reshape(-1, 2)
    if len(pts) == 0:
        pts = np.asarray(stroke_endpoint_points(stroke), dtype=np.float64).reshape(-1, 2)
    h, w = mask.shape[:2]
    xs = np.rint(pts[:, 0]).astype(np.int32)
    ys = np.rint(pts[:, 1]).astype(np.int32)
    valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    if not np.any(valid):
        return False
    return bool(np.any(mask[ys[valid], xs[valid]] > 0))


def side_cap_connection_info(entry, cap_candidate, infos, connect_tol=0.0):
    """Check side-to-cap connection, enforcing only strokes inside the cap range."""
    tol = 0.0 if connect_tol is None else float(connect_tol)
    enabled = tol > 0.0
    side_strokes = list(entry.get("strokes", []))
    info = {
        "enabled": bool(enabled),
        "tol": float(tol),
        "passed": True,
        "side_count": int(len(side_strokes)),
        "connected_count": int(len(side_strokes)),
        "range_checked_count": int(len(side_strokes)),
        "disconnected_strokes": [],
        "ignored_disconnected_strokes": [],
        "cap_range_method": "",
        "details": [],
    }
    if not enabled:
        return info

    cap_points = cap_candidate_point_cloud(cap_candidate, infos)
    if len(cap_points) == 0:
        info["passed"] = False
        info["connected_count"] = 0
        info["range_checked_count"] = int(len(side_strokes))
        info["disconnected_strokes"] = [int(s.get("index", -1)) for s in side_strokes]
        return info

    cap_range_mask, cap_range_method = cap_candidate_enclosed_range_mask(
        cap_candidate,
        infos=infos,
        connect_tol=tol,
    )
    info["cap_range_method"] = cap_range_method

    connected_count = 0
    range_checked_count = 0
    disconnected = []
    ignored_disconnected = []
    details = []
    for s in side_strokes:
        p0, p1 = stroke_endpoint_points(s)
        d0 = min_distance_to_points(p0, cap_points)
        d1 = min_distance_to_points(p1, cap_points)
        min_d = min(d0, d1)
        nearest_endpoint = "start" if d0 <= d1 else "end"
        connected = bool(min_d <= tol)
        in_cap_range = stroke_points_overlap_mask(s, cap_range_mask)
        if connected:
            connected_count += 1
        elif in_cap_range:
            range_checked_count += 1
            disconnected.append(int(s.get("index", -1)))
        else:
            ignored_disconnected.append(int(s.get("index", -1)))
        if connected and in_cap_range:
            range_checked_count += 1
        details.append({
            "stroke": int(s.get("index", -1)),
            "start_distance": float(d0),
            "end_distance": float(d1),
            "min_distance": float(min_d),
            "nearest_endpoint": nearest_endpoint,
            "connected": bool(connected),
            "in_cap_range": bool(in_cap_range),
            "ignored_disconnected": bool((not connected) and (not in_cap_range)),
        })

    info["passed"] = len(disconnected) == 0
    info["connected_count"] = int(connected_count)
    info["range_checked_count"] = int(range_checked_count)
    info["disconnected_strokes"] = disconnected
    info["ignored_disconnected_strokes"] = ignored_disconnected
    info["details"] = details
    return info


def fill_side_cap_connection_details(details, connection_info):
    if not connection_info:
        return
    details["side_cap_connect_enabled"] = bool(connection_info.get("enabled", False))
    details["side_cap_connect_tol"] = float(connection_info.get("tol", 0.0))
    details["side_cap_connect_passed"] = bool(connection_info.get("passed", True))
    details["side_cap_connected_count"] = int(connection_info.get("connected_count", 0))
    details["side_cap_side_count"] = int(connection_info.get("side_count", 0))
    details["side_cap_range_checked_count"] = int(connection_info.get("range_checked_count", 0))
    details["side_cap_disconnected_strokes"] = list(connection_info.get("disconnected_strokes", []))
    details["side_cap_ignored_disconnected_strokes"] = list(connection_info.get("ignored_disconnected_strokes", []))
    details["side_cap_range_method"] = connection_info.get("cap_range_method", "")
    details["side_cap_connection_details"] = list(connection_info.get("details", []))


def make_removed_subgroup_entry(parent_entry, removed_strokes, subgroup_strokes, cluster_id):
    """Create a direction-group subgroup by removing one or more strokes."""
    removed_strokes = list(removed_strokes)
    removal_depth = len(removed_strokes)
    removed_indices = [int(s["index"]) for s in removed_strokes]
    direction = mean_direction(subgroup_strokes)
    parent_details = parent_entry.get("details", {}) or {}
    score, details = score_side_cluster(
        subgroup_strokes,
        direction,
        same_loop_endpoint_tol=parent_details.get("score_same_loop_endpoint_tol", 12.0),
        count_weight=parent_details.get("score_count_weight", 120.0),
        same_loop_penalty_weight=parent_details.get("score_same_loop_penalty_weight", 220.0),
    )
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
    min_bbox_area=0,
    min_total_arc=0.0,
    thickness=2,
    max_loop_subset_size=14,
    cap_pool_infos=None,
    progress_debug_dir=None,
    progress_rank=None,
    stop_event=None,
):
    """Try one direction group as side strokes and return its largest-area cap."""
    pool_infos = infos if cap_pool_infos is None else cap_pool_infos

    def progress(event, **payload):
        if (
            progress_debug_dir is None
            or progress_rank is None
            or cap_search_stop_requested(stop_event)
        ):
            return
        write_cluster_cap_search_step(progress_debug_dir, entry, progress_rank, event, **payload)

    if cap_search_stop_requested(stop_event):
        return None, []

    progress(
        "compute_best_cap_for_side_entry_started",
        cap_pool_count=int(len(pool_infos)),
        side_indices=[int(s["index"]) for s in entry.get("strokes", [])],
        endpoint_tol=float(endpoint_tol),
        min_pixels=int(min_pixels),
        min_enclosed_area=int(min_enclosed_area),
        min_bbox_area=int(min_bbox_area),
        min_total_arc=float(min_total_arc),
        max_loop_subset_size=int(max_loop_subset_size),
    )
    candidates = extract_cap_loop_candidates_from_strokes(
        image_shape,
        pool_infos,
        entry.get("strokes", []),
        endpoint_tol=endpoint_tol,
        min_pixels=min_pixels,
        min_enclosed_area=min_enclosed_area,
        min_bbox_area=min_bbox_area,
        min_total_arc=min_total_arc,
        thickness=thickness,
        max_loop_subset_size=max_loop_subset_size,
        progress_callback=progress if progress_debug_dir is not None and progress_rank is not None else None,
        stop_event=stop_event,
    )
    if cap_search_stop_requested(stop_event):
        return None, []
    best = largest_area_cap_candidate(candidates)
    progress(
        "compute_best_cap_for_side_entry_finished",
        candidate_count=int(len(candidates)),
        best_cap=summarize_cap_candidate_for_status(best),
    )
    return best, candidates


def evaluate_cap_candidate_for_selection(
    entry,
    cap_candidate,
    infos,
    image_shape,
    endpoint_tol,
    thickness,
    valid_input_enclosed_mask=None,
    sweep_gate_enabled=False,
    sweep_iou_stop_thresh=0.0,
    copy_direction_angle_tol=None,
    copy_iou_compare_percent=None,
    side_cap_connect_tol=0.0,
    progress_debug_dir=None,
    stop_event=None,
):
    """Evaluate one cap candidate with sweep IoU and side-cap connection gates."""
    if cap_search_stop_requested(stop_event):
        return {
            "cap_candidate": cap_candidate,
            "sweep_info": None,
            "connection_info": None,
            "selection_passed": False,
        }
    sweep_info = None
    if cap_candidate is not None and valid_input_enclosed_mask is not None:
        sweep_info = compute_cap_sweep_similarity(
            image_shape,
            entry,
            cap_candidate,
            infos,
            endpoint_tol,
            thickness,
            sketch_mask=valid_input_enclosed_mask,
            copy_direction_angle_tol=copy_direction_angle_tol,
            iou_compare_percent=copy_iou_compare_percent,
        )

    connection_info = side_cap_connection_info(
        entry,
        cap_candidate,
        infos,
        connect_tol=side_cap_connect_tol,
    )

    selection_passed = cap_candidate is not None
    if sweep_gate_enabled:
        selection_passed = (
            sweep_info is not None
            and bool(sweep_info.get("valid", False))
            and float(sweep_info.get("iou", 0.0)) >= float(sweep_iou_stop_thresh)
        )
    if connection_info is not None:
        selection_passed = bool(selection_passed) and bool(connection_info.get("passed", True))

    return {
        "cap_candidate": cap_candidate,
        "sweep_info": sweep_info,
        "connection_info": connection_info,
        "selection_passed": bool(selection_passed),
    }


def choose_best_cap_candidate_for_selection(
    entry,
    candidates,
    infos,
    image_shape,
    endpoint_tol,
    thickness,
    valid_input_enclosed_mask=None,
    sweep_gate_enabled=False,
    sweep_iou_stop_thresh=0.0,
    copy_direction_angle_tol=None,
    copy_iou_compare_percent=None,
    side_cap_connect_tol=0.0,
    progress_debug_dir=None,
    stop_event=None,
):
    """Choose the best cap candidate after evaluating every candidate's IoU."""
    if not candidates or cap_search_stop_requested(stop_event):
        return None, None, None, False, []

    evaluated = []
    for cap_candidate in candidates:
        if cap_search_stop_requested(stop_event):
            break
        evaluated.append(evaluate_cap_candidate_for_selection(
            entry,
            cap_candidate,
            infos,
            image_shape,
            endpoint_tol,
            thickness,
            valid_input_enclosed_mask=valid_input_enclosed_mask,
            sweep_gate_enabled=sweep_gate_enabled,
            sweep_iou_stop_thresh=sweep_iou_stop_thresh,
            copy_direction_angle_tol=copy_direction_angle_tol,
            copy_iou_compare_percent=copy_iou_compare_percent,
            side_cap_connect_tol=side_cap_connect_tol,
            stop_event=stop_event,
        ))
    if not evaluated:
        return None, None, None, False, []

    def sort_key(item):
        cap = item.get("cap_candidate") or {}
        sweep_info = item.get("sweep_info") or {}
        connection_info = item.get("connection_info") or {}
        return (
            1 if item.get("selection_passed", False) else 0,
            1 if connection_info.get("passed", True) else 0,
            float(sweep_info.get("iou", 0.0)),
            int(cap.get("enclosed_area", 0)),
            int(cap.get("area", 0)),
            float(cap.get("score", 0.0)),
        )

    best = max(evaluated, key=sort_key)
    return (
        best.get("cap_candidate"),
        best.get("sweep_info"),
        best.get("connection_info"),
        bool(best.get("selection_passed", False)),
        evaluated,
    )


def is_iou_only_rejected_result(result, *, sweep_gate_enabled=False, sweep_iou_stop_thresh=0.0):
    """Return True when an entry failed selection only because its IoU missed the threshold."""
    if not sweep_gate_enabled or result is None:
        return False
    if result.get("selection_passed", False):
        return False
    if result.get("best_cap", None) is None:
        return False

    trace_item = result.get("trace_item", {}) or {}
    if not bool(trace_item.get("best_sweep_valid", False)):
        return False
    if float(trace_item.get("best_sweep_iou", 0.0)) >= float(sweep_iou_stop_thresh):
        return False
    if not bool(trace_item.get("side_cap_passed", True)):
        return False
    return True


def evaluate_cap_cluster_entry_worker(payload):
    """Pickle-friendly wrapper used by ProcessPoolExecutor."""
    if payload.get("limit_cv_threads", False):
        try:
            cv2.setNumThreads(1)
        except Exception:
            pass
    return evaluate_cap_cluster_entry(**payload)


def evaluate_cap_cluster_entry(
    entry,
    rank,
    infos,
    image_shape,
    endpoint_tol=12.0,
    min_pixels=40,
    min_enclosed_area=0,
    min_bbox_area=0,
    min_total_arc=0.0,
    thickness=2,
    max_loop_subset_size=14,
    pool_infos=None,
    progress_debug_dir=None,
    valid_input_enclosed_mask=None,
    sweep_gate_enabled=False,
    sweep_iou_stop_thresh=0.0,
    copy_direction_angle_tol=None,
    copy_iou_compare_percent=None,
    side_cap_connect_tol=0.0,
    stop_event=None,
    limit_cv_threads=False,
):
    """Evaluate one side cluster/subgroup and return its cap-search result."""
    if limit_cv_threads:
        try:
            cv2.setNumThreads(1)
        except Exception:
            pass

    details = init_cap_validation_details(entry, rank)
    details["sweep_gate_enabled"] = bool(sweep_gate_enabled)
    n_strokes = int(details.get("n", len(entry.get("strokes", []))))
    best_cap = None
    sweep_info = None
    connection_info = None
    trial_candidates = []
    cap_evaluations = []
    selection_passed = False
    cancelled = False

    def stopped():
        return cap_search_stop_requested(stop_event)

    def mark_cancelled(phase):
        nonlocal cancelled, selection_passed
        cancelled = True
        selection_passed = False
        details["cap_validation_cancelled"] = True
        details["cap_validation_cancel_phase"] = str(phase)
        details["cap_validation_selectable"] = False
        details["selection_passed"] = False

    try:
        if stopped():
            mark_cancelled("before_started")
        else:
            write_cluster_progress_status(
                progress_debug_dir,
                entry,
                rank,
                "started",
                selection_passed=False,
                sweep_gate_enabled=bool(sweep_gate_enabled),
                sweep_iou_stop_thresh=float(sweep_iou_stop_thresh or 0.0),
                side_cap_connect_tol=float(side_cap_connect_tol or 0.0),
            )

        if not cancelled and stopped():
            mark_cancelled("after_started")

        if not cancelled:
            save_cluster_progress_preview_outputs(
                progress_debug_dir,
                image_shape,
                entry,
                rank,
                infos,
                pool_infos,
                endpoint_tol,
                stop_event=stop_event,
            )

        if not cancelled and stopped():
            mark_cancelled("after_preview")

        if not cancelled:
            if n_strokes < 2:
                details["cap_validation_skipped"] = True
                details["cap_validation_skip_reason"] = "n_lt_2"
                details["cap_validation_selectable"] = False
                if not stopped():
                    write_cluster_progress_status(
                        progress_debug_dir,
                        entry,
                        rank,
                        "skipped_n_lt_2",
                        selection_passed=False,
                    )
            else:
                details["cap_validation_checked"] = True
                if not stopped():
                    write_cluster_progress_status(
                        progress_debug_dir,
                        entry,
                        rank,
                        "searching_cap_candidates",
                        selection_passed=False,
                    )
                if stopped():
                    mark_cancelled("before_cap_search")
                else:
                    largest_cap_preview, trial_candidates = compute_best_cap_for_side_entry(
                        entry,
                        infos,
                        image_shape,
                        endpoint_tol=endpoint_tol,
                        min_pixels=min_pixels,
                        min_enclosed_area=min_enclosed_area,
                        min_bbox_area=min_bbox_area,
                        min_total_arc=min_total_arc,
                        thickness=thickness,
                        max_loop_subset_size=max_loop_subset_size,
                        cap_pool_infos=pool_infos,
                        progress_debug_dir=progress_debug_dir,
                        progress_rank=rank,
                        stop_event=stop_event,
                    )
                    if stopped():
                        mark_cancelled("after_cap_search")
                    else:
                        write_cluster_progress_status(
                            progress_debug_dir,
                            entry,
                            rank,
                            "evaluating_cap_candidates",
                            selection_passed=False,
                            cap_candidate_count=int(len(trial_candidates)),
                            best_cap_preview=summarize_cap_candidate_for_status(largest_cap_preview),
                        )
                        best_cap, sweep_info, connection_info, selection_passed, cap_evaluations = choose_best_cap_candidate_for_selection(
                            entry,
                            trial_candidates,
                            infos,
                            image_shape,
                            endpoint_tol,
                            thickness,
                            valid_input_enclosed_mask=valid_input_enclosed_mask,
                            sweep_gate_enabled=sweep_gate_enabled,
                            sweep_iou_stop_thresh=sweep_iou_stop_thresh,
                            copy_direction_angle_tol=copy_direction_angle_tol,
                            copy_iou_compare_percent=copy_iou_compare_percent,
                            side_cap_connect_tol=side_cap_connect_tol,
                            stop_event=stop_event,
                        )
                        if stopped():
                            mark_cancelled("after_candidate_evaluation")
                        else:
                            details["cap_candidate_evaluated_count"] = int(len(cap_evaluations))
                            fill_best_cap_details(details, best_cap, len(trial_candidates))
                            fill_best_cap_sweep_details(
                                details,
                                sweep_info,
                                gate_enabled=sweep_gate_enabled,
                                stop_thresh=sweep_iou_stop_thresh,
                            )
                            if connection_info is not None:
                                fill_side_cap_connection_details(details, connection_info)

        if not cancelled and best_cap is not None:
            if sweep_gate_enabled:
                details["cap_found_but_sweep_rejected"] = not bool(details.get("best_sweep_passed", False))
            if connection_info is not None:
                details["cap_found_but_side_cap_disconnected"] = not bool(connection_info.get("passed", True))
        if not cancelled:
            details["selection_passed"] = bool(selection_passed)
            details["cap_validation_selectable"] = bool(selection_passed)

        if not cancelled and stopped():
            mark_cancelled("before_debug_outputs")

        if not cancelled:
            write_cluster_progress_status(
                progress_debug_dir,
                entry,
                rank,
                "writing_debug_outputs",
                selection_passed=bool(selection_passed),
                cap_candidate_count=int(len(trial_candidates)),
                cap_candidate_evaluated_count=int(len(cap_evaluations)),
                best_cap=summarize_cap_candidate_for_status(best_cap),
                best_sweep_iou=0.0 if sweep_info is None else float(sweep_info.get("iou", 0.0)),
                best_sweep_valid=False if sweep_info is None else bool(sweep_info.get("valid", False)),
                side_cap_passed=True if connection_info is None else bool(connection_info.get("passed", True)),
            )

            save_single_cluster_candidate_debug_output(
                progress_debug_dir,
                image_shape,
                entry,
                rank,
                infos,
                pool_infos,
                best_cap,
                cap_evaluations,
                endpoint_tol,
                thickness,
                valid_input_enclosed_mask=valid_input_enclosed_mask,
                sweep_gate_enabled=sweep_gate_enabled,
                sweep_iou_stop_thresh=sweep_iou_stop_thresh,
                copy_direction_angle_tol=copy_direction_angle_tol,
                copy_iou_compare_percent=copy_iou_compare_percent,
                stop_event=stop_event,
            )

        if not cancelled and stopped():
            mark_cancelled("after_debug_outputs")

        if not cancelled:
            write_cluster_progress_status(
                progress_debug_dir,
                entry,
                rank,
                "completed",
                selection_passed=bool(selection_passed),
                cap_candidate_count=int(len(trial_candidates)),
                cap_candidate_evaluated_count=int(len(cap_evaluations)),
                best_cap=summarize_cap_candidate_for_status(best_cap),
                best_sweep_iou=0.0 if sweep_info is None else float(sweep_info.get("iou", 0.0)),
                best_sweep_valid=False if sweep_info is None else bool(sweep_info.get("valid", False)),
                side_cap_passed=True if connection_info is None else bool(connection_info.get("passed", True)),
                side_cap_connected_count=0 if connection_info is None else int(connection_info.get("connected_count", 0)),
                side_cap_side_count=0 if connection_info is None else int(connection_info.get("side_count", 0)),
            )
    except Exception as exc:
        if not stopped():
            write_cluster_progress_status(
                progress_debug_dir,
                entry,
                rank,
                "failed",
                selection_passed=False,
                error=str(exc),
                cap_candidate_count=int(len(trial_candidates)),
                cap_candidate_evaluated_count=int(len(cap_evaluations)),
            )
        raise

    trace_item = {
        "order": -1,
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
        "cancelled": bool(cancelled or details.get("cap_validation_cancelled", False)),
        "cancel_phase": details.get("cap_validation_cancel_phase", ""),
        "cap_found": best_cap is not None,
        "cap_candidate_count": int(details.get("cap_candidate_count", 0)),
        "cap_candidate_evaluated_count": int(details.get("cap_candidate_evaluated_count", 0)),
        "best_cap_area": int(details.get("best_cap_area", 0)),
        "best_cap_enclosed_area": int(details.get("best_cap_enclosed_area", 0)),
        "best_cap_bbox_area": int(details.get("best_cap_bbox_area", 0)),
        "best_cap_total_arc": float(details.get("best_cap_total_arc", 0.0)),
        "best_cap_score": float(details.get("best_cap_score", 0.0)),
        "best_cap_strokes": list(details.get("best_cap_strokes", [])),
        "best_cap_topology_kind": details.get("best_cap_topology_kind", ""),
        "best_cap_loop_detection": details.get("best_cap_loop_detection", ""),
        "best_cap_topology_cycle_count": int(details.get("best_cap_topology_cycle_count", 0)),
        "best_cap_topology_simple_cycle_count": int(details.get("best_cap_topology_simple_cycle_count", 0)),
        "best_cap_topology_edge_disjoint_cover": bool(details.get("best_cap_topology_edge_disjoint_cover", False)),
        "best_cap_topology_edge_disjoint_cover_count": int(details.get("best_cap_topology_edge_disjoint_cover_count", 0)),
        "best_sweep_valid": bool(details.get("best_sweep_valid", False)),
        "best_sweep_iou": float(details.get("best_sweep_iou", 0.0)),
        "best_sweep_intersection": int(details.get("best_sweep_intersection", 0)),
        "best_sweep_union": int(details.get("best_sweep_union", 0)),
        "best_sweep_area": int(details.get("best_sweep_area", 0)),
        "best_sweep_copy_side_stroke": details.get("best_sweep_copy_side_stroke", None),
        "best_sweep_copy_reason": details.get("best_sweep_copy_reason", ""),
        "best_sweep_mask_source": details.get("best_sweep_mask_source", ""),
        "side_cap_connect_enabled": bool(details.get("side_cap_connect_enabled", False)),
        "side_cap_connect_tol": float(details.get("side_cap_connect_tol", 0.0)),
        "side_cap_connect_passed": bool(details.get("side_cap_connect_passed", True)),
        "side_cap_connected_count": int(details.get("side_cap_connected_count", 0)),
        "side_cap_side_count": int(details.get("side_cap_side_count", 0)),
        "side_cap_range_checked_count": int(details.get("side_cap_range_checked_count", 0)),
        "side_cap_disconnected_strokes": list(details.get("side_cap_disconnected_strokes", [])),
        "side_cap_ignored_disconnected_strokes": list(details.get("side_cap_ignored_disconnected_strokes", [])),
        "side_cap_range_method": details.get("side_cap_range_method", ""),
        "side_cap_connection_details": list(details.get("side_cap_connection_details", [])),
        "cap_found_but_side_cap_disconnected": bool(details.get("cap_found_but_side_cap_disconnected", False)),
        "cap_found_but_sweep_rejected": bool(details.get("cap_found_but_sweep_rejected", False)),
        "selection_passed": bool(selection_passed),
    }
    return {
        "entry": entry,
        "rank": int(rank),
        "best_cap": best_cap,
        "trial_candidates": [],
        "selection_passed": bool(selection_passed),
        "cancelled": bool(cancelled),
        "trace_item": trace_item,
    }


def validate_side_clusters_by_cap_candidates(
    model,
    infos,
    image_shape,
    endpoint_tol=12.0,
    min_pixels=40,
    min_enclosed_area=0,
    min_bbox_area=0,
    min_total_arc=0.0,
    thickness=2,
    max_loop_subset_size=14,
    max_subgroup_removals=-1,
    cap_pool_infos=None,
    input_enclosed_mask=None,
    sweep_iou_stop_thresh=0.0,
    copy_direction_angle_tol=None,
    copy_iou_compare_percent=None,
    side_cap_connect_tol=0.0,
    progress_debug_dir=None,
    cap_round_workers=1,
    cap_worker_backend="thread",
    cap_search_time_limit_sec=0.0,
):
    """
    Compute cap candidates for every direction group.

    With cap_round_workers=1, search is executed round by round.
    With cap_round_workers>1, search uses pipelined speculative scheduling.

    Round 0 evaluates all full direction groups.
    Round k evaluates all remove-k subgroups for every still-unresolved parent.

    A round stops the search only when at least one entry produces a legal cap
    and, when enabled, its cap-sweep occupancy IoU against the input enclosed
    mask passes the configured threshold.  Otherwise the whole round is treated
    as unresolved and the search continues to the next removal depth.
    """
    if model is None:
        return None, []

    try:
        cap_round_workers = max(1, int(cap_round_workers or 1))
    except Exception:
        cap_round_workers = 1
    cap_worker_backend = str(cap_worker_backend or "thread").strip().lower()
    if cap_worker_backend not in {"thread", "process"}:
        cap_worker_backend = "thread"

    pool_infos = infos if cap_pool_infos is None else cap_pool_infos
    cluster_debug = model.get("cluster_debug", None)
    valid_input_enclosed_mask = None
    if input_enclosed_mask is not None and np.count_nonzero(input_enclosed_mask > 0) > 0:
        valid_input_enclosed_mask = (input_enclosed_mask > 0).astype(np.uint8) * 255
    sweep_gate_enabled = bool(valid_input_enclosed_mask is not None and float(sweep_iou_stop_thresh or 0.0) > 0.0)
    search_start_time = time.monotonic()
    try:
        cap_search_time_limit_sec = float(cap_search_time_limit_sec or 0.0)
    except Exception:
        cap_search_time_limit_sec = 0.0
    cap_search_time_limit_sec = max(0.0, cap_search_time_limit_sec)
    time_limit_triggered = False
    search_stop_reason = ""

    def elapsed_sec():
        return float(time.monotonic() - search_start_time)

    def remaining_time_sec():
        if cap_search_time_limit_sec <= 0.0:
            return None
        return max(0.0, cap_search_time_limit_sec - elapsed_sec())

    def mark_time_limit_if_needed():
        nonlocal time_limit_triggered, search_stop_reason
        if cap_search_time_limit_sec <= 0.0:
            return False
        if elapsed_sec() < cap_search_time_limit_sec:
            return False
        time_limit_triggered = True
        if not search_stop_reason:
            search_stop_reason = "time_limit"
        cap_search_request_stop(cap_stop_event)
        return True

    if not cluster_debug:
        candidates = extract_cap_loop_candidates_from_strokes(
            image_shape,
            pool_infos,
            model.get("inliers", []),
            endpoint_tol=endpoint_tol,
            min_pixels=min_pixels,
            min_enclosed_area=min_enclosed_area,
            min_bbox_area=min_bbox_area,
            min_total_arc=min_total_arc,
            thickness=thickness,
            max_loop_subset_size=max_loop_subset_size,
        )
        pseudo_entry = {
            "strokes": list(model.get("inliers", [])),
            "indices": [int(s["index"]) for s in model.get("inliers", [])],
            "direction": model.get("direction", None),
        }
        best_cap, sweep_info, connection_info, selection_passed, _cap_evaluations = choose_best_cap_candidate_for_selection(
            pseudo_entry,
            candidates,
            infos,
            image_shape,
            endpoint_tol,
            thickness,
            valid_input_enclosed_mask=valid_input_enclosed_mask,
            sweep_gate_enabled=sweep_gate_enabled,
            sweep_iou_stop_thresh=sweep_iou_stop_thresh,
            copy_direction_angle_tol=copy_direction_angle_tol,
            copy_iou_compare_percent=copy_iou_compare_percent,
            side_cap_connect_tol=side_cap_connect_tol,
        )
        if connection_info is None:
            connection_info = side_cap_connection_info(
                pseudo_entry,
                best_cap,
                infos,
                connect_tol=side_cap_connect_tol,
            )

        model["cap_sweep_gate_enabled"] = bool(sweep_gate_enabled)
        model["side_cap_connect_enabled"] = bool(connection_info.get("enabled", False))
        model["side_cap_connect_tol"] = float(connection_info.get("tol", 0.0))
        model["side_cap_connect_passed"] = bool(connection_info.get("passed", True))
        model["side_cap_disconnected_strokes"] = list(connection_info.get("disconnected_strokes", []))
        model["side_cap_ignored_disconnected_strokes"] = list(connection_info.get("ignored_disconnected_strokes", []))
        model["side_cap_range_checked_count"] = int(connection_info.get("range_checked_count", 0))
        model["side_cap_range_method"] = connection_info.get("cap_range_method", "")
        model["cap_validated"] = bool(selection_passed)
        model["cap_validation_failed"] = not bool(selection_passed)
        model["cap_worker_backend"] = cap_worker_backend
        model["cap_round_workers"] = int(cap_round_workers)
        model["cap_search_time_limit_sec"] = float(cap_search_time_limit_sec)
        model["cap_search_elapsed_sec"] = elapsed_sec()
        model["cap_search_time_limit_reached"] = bool(time_limit_triggered)
        model["cap_search_stop_reason"] = search_stop_reason
        if sweep_info is not None:
            model["best_sweep_iou"] = float(sweep_info.get("iou", 0.0))
            model["best_sweep_valid"] = bool(sweep_info.get("valid", False))
        return model, ([] if not selection_passed else [best_cap])

    original_cluster_debug = list(cluster_debug)
    expanded_cluster_debug = []
    next_cluster_id = max([int(e.get("cluster_id", -1)) for e in original_cluster_debug] + [-1]) + 1
    successful_cap_clusters = []
    cap_search_trace = []
    cap_search_rounds = []
    selected_result = None
    selected_via_iou_fallback = False
    iou_only_rejected_results = []

    prepare_cluster_progress_debug_dir(progress_debug_dir)

    def flush_round_progress(current_winner=None):
        """Persist the current round state so long searches expose progress incrementally."""
        if progress_debug_dir is None:
            return
        progress_model = dict(model)
        progress_model["cluster_debug"] = expanded_cluster_debug
        progress_model["cap_search_trace"] = cap_search_trace
        progress_model["cap_search_rounds"] = cap_search_rounds
        progress_model["cap_sweep_gate_enabled"] = bool(sweep_gate_enabled)
        progress_model["side_cap_connect_enabled"] = bool(float(side_cap_connect_tol or 0.0) > 0.0)
        progress_model["side_cap_connect_tol"] = float(side_cap_connect_tol or 0.0)
        progress_model["cap_round_workers"] = int(cap_round_workers)
        progress_model["cap_worker_backend"] = cap_worker_backend
        progress_model["cap_search_time_limit_sec"] = float(cap_search_time_limit_sec)
        progress_model["cap_search_elapsed_sec"] = elapsed_sec()
        progress_model["cap_search_time_limit_reached"] = bool(time_limit_triggered)
        progress_model["cap_search_stop_reason"] = search_stop_reason
        progress_model["selected_cluster_id"] = (
            None
            if current_winner is None
            else int(current_winner["entry"].get("cluster_id", -1))
        )
        save_cluster_debug_outputs(progress_debug_dir, image_shape, progress_model)

    cap_stop_event = None

    def build_entry_eval_payload(entry, rank, limit_cv_threads=False):
        return {
            "entry": entry,
            "rank": int(rank),
            "infos": infos,
            "image_shape": image_shape,
            "endpoint_tol": endpoint_tol,
            "min_pixels": min_pixels,
            "min_enclosed_area": min_enclosed_area,
            "min_bbox_area": min_bbox_area,
            "min_total_arc": min_total_arc,
            "thickness": thickness,
            "max_loop_subset_size": max_loop_subset_size,
            "pool_infos": pool_infos,
            "progress_debug_dir": progress_debug_dir,
            "valid_input_enclosed_mask": valid_input_enclosed_mask,
            "sweep_gate_enabled": sweep_gate_enabled,
            "sweep_iou_stop_thresh": sweep_iou_stop_thresh,
            "copy_direction_angle_tol": copy_direction_angle_tol,
            "copy_iou_compare_percent": copy_iou_compare_percent,
            "side_cap_connect_tol": side_cap_connect_tol,
            "stop_event": cap_stop_event,
            "limit_cv_threads": bool(limit_cv_threads),
        }

    def merge_result_entry(result):
        rank = int(result.get("rank", -1))
        if 0 <= rank < len(expanded_cluster_debug):
            expanded_cluster_debug[rank] = result.get("entry", expanded_cluster_debug[rank])

    def evaluate_entry(entry, rank):
        result = evaluate_cap_cluster_entry(**build_entry_eval_payload(entry, rank))
        merge_result_entry(result)
        return result

    def merge_round_trace(round_results):
        for result in round_results:
            trace_item = result.get("trace_item", {})
            trace_item["order"] = int(len(cap_search_trace))
            cap_search_trace.append(trace_item)

    def evaluate_round(entry_rank_pairs):
        entry_rank_pairs = list(entry_rank_pairs)
        if not entry_rank_pairs:
            return []
        if cap_round_workers <= 1 or len(entry_rank_pairs) <= 1:
            results = []
            for entry, rank in entry_rank_pairs:
                if mark_time_limit_if_needed():
                    break
                results.append(evaluate_entry(entry, rank))
                mark_time_limit_if_needed()
            merge_round_trace(results)
            return results

        from concurrent.futures import ThreadPoolExecutor, as_completed

        result_by_rank = {}
        max_workers = min(int(cap_round_workers), len(entry_rank_pairs))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_rank = {
                executor.submit(evaluate_entry, entry, rank): int(rank)
                for entry, rank in entry_rank_pairs
            }
            for future in as_completed(future_to_rank):
                result = future.result()
                result_by_rank[int(result["rank"])] = result

        results = [result_by_rank[int(rank)] for _entry, rank in entry_rank_pairs]
        merge_round_trace(results)
        return results

    def winner_sort_key(result):
        cap = result.get("best_cap") or {}
        trace_item = result.get("trace_item", {})
        return (
            float(trace_item.get("best_sweep_iou", 0.0)),
            int(cap.get("area", 0)),
            float(cap.get("score", 0.0)),
            -int(result.get("rank", 0)),
        )

    def choose_round_winner(round_results):
        if sweep_gate_enabled:
            passing = [r for r in round_results if r.get("selection_passed", False)]
            if not passing:
                return None
            return max(passing, key=winner_sort_key)
        for result in round_results:
            if result.get("selection_passed", False):
                return result
        return None

    def record_successful_results(round_results):
        for result in round_results:
            if not result.get("selection_passed", False):
                continue
            entry = result["entry"]
            removed_stroke_indices = list(entry.get("removed_stroke_indices", []))
            side_set = cluster_entry_index_set(entry)
            successful_cap_clusters.append({
                "rank": int(result.get("rank", -1)),
                "cluster_id": int(entry.get("cluster_id", -1)),
                "side_set": set(side_set),
                "parent_cluster_id": entry.get("parent_cluster_id", None),
                "removed_stroke_index": removed_stroke_indices[0] if len(removed_stroke_indices) == 1 else None,
                "removed_stroke_indices": removed_stroke_indices,
                "removal_depth": int(entry.get("removal_depth", 0)),
            })

    def append_fallback_success_result(result):
        if result is None:
            return
        entry = result["entry"]
        cluster_id = int(entry.get("cluster_id", -1))
        for item in successful_cap_clusters:
            if int(item.get("cluster_id", -1)) == cluster_id:
                return
        removed_stroke_indices = list(entry.get("removed_stroke_indices", []))
        side_set = cluster_entry_index_set(entry)
        successful_cap_clusters.append({
            "rank": int(result.get("rank", -1)),
            "cluster_id": int(entry.get("cluster_id", -1)),
            "side_set": set(side_set),
            "parent_cluster_id": entry.get("parent_cluster_id", None),
            "removed_stroke_index": removed_stroke_indices[0] if len(removed_stroke_indices) == 1 else None,
            "removed_stroke_indices": removed_stroke_indices,
            "removal_depth": int(entry.get("removal_depth", 0)),
        })

    def record_iou_only_rejected_results(round_results):
        for result in round_results:
            if is_iou_only_rejected_result(
                result,
                sweep_gate_enabled=sweep_gate_enabled,
                sweep_iou_stop_thresh=sweep_iou_stop_thresh,
            ):
                iou_only_rejected_results.append(result)

    def summarize_round(removal_depth, round_results, winner):
        cap_results = [r for r in round_results if r.get("best_cap") is not None]
        best_iou_result = None
        valid_iou_results = [r for r in cap_results if r.get("trace_item", {}).get("best_sweep_valid", False)]
        if valid_iou_results:
            best_iou_result = max(valid_iou_results, key=winner_sort_key)
        cap_search_rounds.append({
            "removal_depth": int(removal_depth),
            "entry_count": int(len(round_results)),
            "cap_count": int(len(cap_results)),
            "passed_count": int(sum(1 for r in round_results if r.get("selection_passed", False))),
            "best_iou_cluster_id": None if best_iou_result is None else int(best_iou_result["entry"].get("cluster_id", -1)),
            "best_iou": 0.0 if best_iou_result is None else float(best_iou_result.get("trace_item", {}).get("best_sweep_iou", 0.0)),
            "winner_cluster_id": None if winner is None else int(winner["entry"].get("cluster_id", -1)),
            "stopped": bool(winner is not None),
            "elapsed_sec": elapsed_sec(),
            "time_limit_reached": bool(time_limit_triggered),
            "stop_reason": search_stop_reason,
        })

    def parent_unresolved_after_round(parent_entry, parent_round_results, removal_depth):
        remaining_after_removal = len(parent_entry.get("strokes", [])) - int(removal_depth)
        if remaining_after_removal <= 1:
            return False
        return not any(r.get("selection_passed", False) for r in parent_round_results)

    import itertools

    def run_pipelined_cluster_search():
        """
        Keep workers full by scheduling the next subgroup depth for each
        parent as soon as that parent's current depth finishes without a pass.

        Final selection is still finalized in increasing removal_depth order, so
        deeper speculative results cannot beat a shallower valid round.
        """
        nonlocal next_cluster_id, selected_result, cap_stop_event
        from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait

        max_possible = max([len(entry.get("strokes", [])) - 1 for entry in original_cluster_debug] + [0])
        if max_subgroup_removals is None or int(max_subgroup_removals) < 0:
            effective_max_removals = max_possible
        else:
            effective_max_removals = min(max_possible, max(0, int(max_subgroup_removals)))

        parent_states = []
        ready_tasks = []
        results_by_depth = {}
        finalized_depth = 0
        stop_scheduling = False

        def parent_layer_complete(state, depth):
            expected = state["expected"].get(int(depth), None)
            if expected is None:
                return False
            return len(state["results"].get(int(depth), [])) >= int(expected)

        def parent_layer_passed(state, depth):
            return any(
                result.get("selection_passed", False)
                for result in state["results"].get(int(depth), [])
            )

        def parent_relevant_at_depth(state, depth):
            depth = int(depth)
            if depth == 0:
                return True
            if depth > int(state["max_depth"]):
                return False
            for prev_depth in range(depth):
                if prev_depth > int(state["max_depth"]):
                    return False
                if not parent_layer_complete(state, prev_depth):
                    return None
                if parent_layer_passed(state, prev_depth):
                    return False
            return True

        def depth_complete(depth):
            depth = int(depth)
            any_relevant = False
            for state in parent_states:
                relevant = parent_relevant_at_depth(state, depth)
                if relevant is None:
                    return False
                if not relevant:
                    continue
                any_relevant = True
                if not parent_layer_complete(state, depth):
                    return False
            return any_relevant

        def enqueue_entry(state, entry, depth):
            rank = len(expanded_cluster_debug)
            expanded_cluster_debug.append(entry)
            depth = int(depth)
            state["expected"][depth] = int(state["expected"].get(depth, 0)) + 1
            ready_tasks.append({
                "entry": entry,
                "rank": int(rank),
                "depth": depth,
                "state": state,
            })

        def schedule_parent_depth(state, depth):
            nonlocal next_cluster_id
            depth = int(depth)
            if depth in state["scheduled_depths"]:
                return
            if depth > int(state["max_depth"]):
                return
            parent_entry = state["entry"]
            parent_strokes = list(parent_entry.get("strokes", []))
            if len(parent_strokes) - depth < 1:
                return
            state["scheduled_depths"].add(depth)
            state["results"].setdefault(depth, [])
            state["expected"].setdefault(depth, 0)
            if depth == 0:
                enqueue_entry(state, parent_entry, depth)
                return

            for removed_positions in itertools.combinations(range(len(parent_strokes)), depth):
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

                enqueue_entry(state, subgroup, depth)

        def finalize_available_depths():
            nonlocal finalized_depth, selected_result, stop_scheduling
            max_depth = max([int(state["max_depth"]) for state in parent_states] + [0])
            while finalized_depth <= max_depth:
                round_results = sorted(
                    results_by_depth.get(int(finalized_depth), []),
                    key=lambda result: int(result.get("rank", 0)),
                )
                if not depth_complete(finalized_depth):
                    break

                current_winner = choose_round_winner(round_results)
                if current_winner is not None:
                    merge_round_trace(round_results)
                    record_successful_results(round_results)
                    record_iou_only_rejected_results(round_results)
                    selected_result = current_winner
                    cap_search_request_stop(cap_stop_event)
                    summarize_round(finalized_depth, round_results, selected_result)
                    flush_round_progress(selected_result)
                    finalized_depth += 1
                    stop_scheduling = True
                    ready_tasks.clear()
                    break

                merge_round_trace(round_results)
                record_successful_results(round_results)
                record_iou_only_rejected_results(round_results)
                summarize_round(finalized_depth, round_results, None)
                flush_round_progress(None)
                finalized_depth += 1

        for parent_i, entry in enumerate(original_cluster_debug):
            max_depth = min(int(effective_max_removals), max(0, len(entry.get("strokes", [])) - 1))
            state = {
                "parent_index": int(parent_i),
                "entry": entry,
                "max_depth": int(max_depth),
                "scheduled_depths": set(),
                "expected": {},
                "results": {},
            }
            parent_states.append(state)
            schedule_parent_depth(state, 0)

        def submit_ready(executor, future_to_task):
            while ready_tasks and len(future_to_task) < int(cap_round_workers) and not stop_scheduling:
                if mark_time_limit_if_needed():
                    break
                task = ready_tasks.pop(0)
                if cap_worker_backend == "process":
                    payload = build_entry_eval_payload(
                        task["entry"],
                        task["rank"],
                        limit_cv_threads=True,
                    )
                    future = executor.submit(evaluate_cap_cluster_entry_worker, payload)
                else:
                    future = executor.submit(evaluate_entry, task["entry"], task["rank"])
                future_to_task[future] = task

        def run_executor_loop(executor):
            future_to_task = {}
            submit_ready(executor, future_to_task)
            while future_to_task or (ready_tasks and not stop_scheduling):
                if mark_time_limit_if_needed():
                    ready_tasks.clear()
                    for pending_future in list(future_to_task.keys()):
                        pending_future.cancel()
                    future_to_task.clear()
                    break
                submit_ready(executor, future_to_task)
                if not future_to_task:
                    break
                wait_timeout = remaining_time_sec()
                done, _pending = wait(
                    list(future_to_task.keys()),
                    return_when=FIRST_COMPLETED,
                    timeout=wait_timeout,
                )
                if not done:
                    if mark_time_limit_if_needed():
                        ready_tasks.clear()
                        for pending_future in list(future_to_task.keys()):
                            pending_future.cancel()
                        future_to_task.clear()
                        break
                    continue
                for future in done:
                    task = future_to_task.pop(future)
                    result = future.result()
                    merge_result_entry(result)
                    depth = int(task["depth"])
                    state = task["state"]
                    state["results"].setdefault(depth, []).append(result)
                    results_by_depth.setdefault(depth, []).append(result)

                    if (
                        not stop_scheduling
                        and parent_layer_complete(state, depth)
                        and not parent_layer_passed(state, depth)
                    ):
                        schedule_parent_depth(state, depth + 1)

                finalize_available_depths()
                if mark_time_limit_if_needed():
                    ready_tasks.clear()
                    for pending_future in list(future_to_task.keys()):
                        pending_future.cancel()
                    future_to_task.clear()
                    break
                if stop_scheduling:
                    ready_tasks.clear()
                    for pending_future in list(future_to_task.keys()):
                        pending_future.cancel()
                    future_to_task.clear()
                    break

        if cap_worker_backend == "process":
            import multiprocessing

            with multiprocessing.Manager() as manager:
                cap_stop_event = manager.Event()
                mp_context = multiprocessing.get_context("spawn")
                with ProcessPoolExecutor(
                    max_workers=max(1, int(cap_round_workers)),
                    mp_context=mp_context,
                ) as executor:
                    run_executor_loop(executor)
        else:
            import threading

            cap_stop_event = threading.Event()
            with ThreadPoolExecutor(max_workers=max(1, int(cap_round_workers))) as executor:
                run_executor_loop(executor)

        finalize_available_depths()

    if cap_round_workers > 1:
        run_pipelined_cluster_search()
    else:
        import threading

        cap_stop_event = threading.Event()
        original_entry_rank_pairs = []
        for entry in original_cluster_debug:
            rank = len(expanded_cluster_debug)
            expanded_cluster_debug.append(entry)
            original_entry_rank_pairs.append((entry, rank))
        original_round_results = evaluate_round(original_entry_rank_pairs)

        record_successful_results(original_round_results)
        record_iou_only_rejected_results(original_round_results)
        selected_result = choose_round_winner(original_round_results)
        summarize_round(0, original_round_results, selected_result)
        flush_round_progress(selected_result)

        failed_original_groups = []
        if selected_result is None and not time_limit_triggered:
            for result in original_round_results:
                entry = result["entry"]
                if len(entry.get("strokes", [])) <= 1:
                    continue
                if not result.get("selection_passed", False):
                    failed_original_groups.append(entry)

        max_possible_removals = 0
        if failed_original_groups:
            max_possible_removals = max(len(entry.get("strokes", [])) - 1 for entry in failed_original_groups)
        if max_subgroup_removals is None or int(max_subgroup_removals) < 0:
            max_subgroup_removals = max_possible_removals
        else:
            max_subgroup_removals = min(max_possible_removals, max(0, int(max_subgroup_removals)))

        for removal_depth in range(1, max_subgroup_removals + 1):
            if selected_result is not None or not failed_original_groups or time_limit_triggered:
                break

            round_entry_rank_pairs = []
            parent_round_pairs = []
            still_failed = []
            for parent_entry in failed_original_groups:
                parent_strokes = list(parent_entry.get("strokes", []))
                if len(parent_strokes) - removal_depth < 1:
                    continue

                current_parent_pairs = []
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
                    pair = (subgroup, subgroup_rank)
                    round_entry_rank_pairs.append(pair)
                    current_parent_pairs.append(pair)

                parent_round_pairs.append((parent_entry, current_parent_pairs))

            round_results = evaluate_round(round_entry_rank_pairs)
            result_by_rank = {int(result["rank"]): result for result in round_results}
            for parent_entry, current_parent_pairs in parent_round_pairs:
                parent_round_results = [
                    result_by_rank[int(rank)]
                    for _entry, rank in current_parent_pairs
                    if int(rank) in result_by_rank
                ]
                if parent_unresolved_after_round(parent_entry, parent_round_results, removal_depth):
                    still_failed.append(parent_entry)

            record_successful_results(round_results)
            record_iou_only_rejected_results(round_results)
            selected_result = choose_round_winner(round_results)
            summarize_round(removal_depth, round_results, selected_result)
            flush_round_progress(selected_result)
            if selected_result is not None:
                break
            if time_limit_triggered:
                break

            failed_original_groups = still_failed

    if selected_result is None and iou_only_rejected_results:
        selected_result = max(iou_only_rejected_results, key=winner_sort_key)
        selected_via_iou_fallback = True
        append_fallback_success_result(selected_result)

    model["cluster_debug"] = expanded_cluster_debug
    model["cap_search_trace"] = cap_search_trace
    model["cap_search_rounds"] = cap_search_rounds
    model["cap_sweep_gate_enabled"] = bool(sweep_gate_enabled)
    model["side_cap_connect_enabled"] = bool(float(side_cap_connect_tol or 0.0) > 0.0)
    model["side_cap_connect_tol"] = float(side_cap_connect_tol or 0.0)
    model["cap_round_workers"] = int(cap_round_workers)
    model["cap_worker_backend"] = cap_worker_backend
    model["cap_validation_used_iou_fallback"] = bool(selected_via_iou_fallback)
    model["cap_search_time_limit_sec"] = float(cap_search_time_limit_sec)
    model["cap_search_elapsed_sec"] = elapsed_sec()
    model["cap_search_time_limit_reached"] = bool(time_limit_triggered)
    model["cap_search_stop_reason"] = search_stop_reason
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

    if selected_result is None:
        model["cap_validated"] = False
        model["cap_validation_failed"] = True
        model["cap_validation_used_iou_fallback"] = False
        if sweep_gate_enabled:
            model["cap_validation_message"] = (
                "No search round produced a cap whose swept extrusion occupancy IoU passed the input enclosed-mask threshold."
            )
        else:
            model["cap_validation_message"] = "No ranked side cluster produced a legal closed-loop cap candidate."
        model["cap_search_trace"] = cap_search_trace
        return model, []

    selected_entry = selected_result["entry"]
    selected_entry["details"]["selected_by_cap_validation"] = True
    selected_entry["details"]["selected_by_iou_fallback"] = bool(selected_via_iou_fallback)
    if selected_via_iou_fallback:
        selected_entry["details"]["cap_validation_fallback_reason"] = "highest_iou_rejected_only_by_sweep_threshold"
    selected_candidates = [selected_result["best_cap"]]

    validated_model = make_trial_model_from_cluster_entry(
        model,
        selected_entry,
        cluster_debug=expanded_cluster_debug,
        cap_validated=True,
    )
    validated_model["cap_validation_failed"] = False
    if sweep_gate_enabled:
        validated_model["cap_validation_message"] = (
            "Evaluated cap-search clusters/subgroups with the configured worker backend and stopped when the shallowest "
            "finalized removal depth produced a cap whose swept extrusion occupancy IoU passed the input enclosed-mask "
            "threshold and whose side strokes passed the side-cap connection gate."
        )
    else:
        validated_model["cap_validation_message"] = (
            "Evaluated cap-search clusters/subgroups with the configured worker backend. If no full direction group had a "
            "legal cap, remove-k subgroups were evaluated until the shallowest finalized depth produced a legal cap."
        )
    validated_model["successful_cap_clusters"] = model.get("successful_cap_clusters", [])
    validated_model["cap_search_trace"] = cap_search_trace
    validated_model["cap_search_rounds"] = cap_search_rounds
    validated_model["cap_sweep_gate_enabled"] = bool(sweep_gate_enabled)
    validated_model["side_cap_connect_enabled"] = bool(float(side_cap_connect_tol or 0.0) > 0.0)
    validated_model["side_cap_connect_tol"] = float(side_cap_connect_tol or 0.0)
    validated_model["cap_round_workers"] = int(cap_round_workers)
    validated_model["cap_worker_backend"] = cap_worker_backend
    validated_model["cap_validation_used_iou_fallback"] = bool(selected_via_iou_fallback)
    validated_model["side_cap_connect_passed"] = bool(
        selected_entry.get("details", {}).get("side_cap_connect_passed", True)
    )
    validated_model["side_cap_disconnected_strokes"] = list(
        selected_entry.get("details", {}).get("side_cap_disconnected_strokes", [])
    )
    validated_model["side_cap_ignored_disconnected_strokes"] = list(
        selected_entry.get("details", {}).get("side_cap_ignored_disconnected_strokes", [])
    )
    validated_model["side_cap_range_checked_count"] = int(
        selected_entry.get("details", {}).get("side_cap_range_checked_count", 0)
    )
    validated_model["side_cap_range_method"] = selected_entry.get("details", {}).get("side_cap_range_method", "")
    if selected_via_iou_fallback:
        validated_model["cap_validation_message"] = (
            "No cap passed the sweep IoU threshold, so the highest-IoU candidate whose only rejection reason was that IoU threshold "
            "was selected as a fallback."
        )
    if time_limit_triggered:
        validated_model["cap_validation_message"] = (
            f"Cap search stopped after reaching the time limit ({cap_search_time_limit_sec:.1f}s); "
            "the best available IoU candidate was selected from completed evaluations."
        )
    validated_model["cap_search_time_limit_sec"] = float(cap_search_time_limit_sec)
    validated_model["cap_search_elapsed_sec"] = elapsed_sec()
    validated_model["cap_search_time_limit_reached"] = bool(time_limit_triggered)
    validated_model["cap_search_stop_reason"] = search_stop_reason
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
        1,
        cv2.LINE_8,
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
            1,
            cv2.LINE_8,
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


def draw_label_segments(out, x, y, segments, font_scale=0.36):
    cursor_x = int(x)
    base_y = int(y)
    for text, color, text_thickness in segments:
        text_thickness = int(text_thickness)
        # OpenCV thickness=2 gets chunky on small debug labels; use a lighter
        # two-pass horizontal emphasis for failed threshold values.
        if text_thickness >= 2:
            cv2.putText(
                out,
                text,
                (cursor_x, base_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                out,
                text,
                (cursor_x + 1, base_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                1,
                cv2.LINE_AA,
            )
            size, _ = cv2.getTextSize(
                text,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                1,
            )
            cursor_x += size[0] + 3
            continue
        cv2.putText(
            out,
            text,
            (cursor_x, base_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            text_thickness,
            cv2.LINE_AA,
        )
        size, _ = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_thickness,
        )
        cursor_x += size[0] + 2


def side_prefilter_result_for_draw(s, args):
    if args is None:
        return {
            "failed_checks": {
                "arc": False,
                "chord": False,
                "straightness": False,
                "p90_pca_line_error": False,
                "pca_rms_error": False,
                "chord_deviation_ratio": False,
            }
        }
    return side_direction_group_filter_result(
        s,
        min_length=args.min_stroke_length,
        min_straightness=args.side_straightness,
        min_chord=args.side_min_chord_px,
        line_p90_error_px=args.side_line_p90_error_px,
        line_p90_error_ratio=args.side_line_p90_error_ratio,
        line_rms_error_px=args.side_line_rms_error_px,
        line_rms_error_ratio=args.side_line_rms_error_ratio,
        chord_dev_ratio_max=args.side_chord_dev_ratio_max,
    )


def draw_side_prefilter_candidates_image(shape, infos, side_candidates, args=None, thickness=4):
    """Draw strokes that pass the pre-cluster side-line filters."""
    h, w = shape[:2]
    out = np.full((h, w, 3), 255, dtype=np.uint8)
    side_ids = {int(s["index"]) for s in side_candidates}
    red = (0, 0, 220)
    black = (0, 0, 0)

    for s in infos:
        if int(s["index"]) in side_ids:
            continue
        pts = s["points"].reshape(-1, 1, 2).astype(np.int32)
        cv2.polylines(out, [pts], False, (205, 205, 205), 1, cv2.LINE_AA)
        result = side_prefilter_result_for_draw(s, args)
        failed = result.get("failed_checks", {})
        if failed.get("arc") or failed.get("chord"):
            continue
        c = s["center"]
        x = int(c[0]) + 4
        y = int(c[1]) - 4
        draw_label_segments(
            out,
            x,
            y,
            [
                (f"{int(s['index'])} ", red, 1),
                ("s=", red, 1),
                (f"{float(s.get('straightness', 0.0)):.2f} ", red, 2 if failed.get("straightness") else 1),
                ("p90=", red, 1),
                (f"{float(s.get('p90_pca_line_error', 0.0)):.1f}", red, 2 if failed.get("p90_pca_line_error") else 1),
            ],
            font_scale=0.34,
        )
        draw_label_segments(
            out,
            x,
            y + 14,
            [
                ("rms=", red, 1),
                (f"{float(s.get('pca_rms_error', 0.0)):.1f} ", red, 2 if failed.get("pca_rms_error") else 1),
                ("d/c=", red, 1),
                (f"{float(s.get('chord_deviation_ratio', 0.0)):.3f}", red, 2 if failed.get("chord_deviation_ratio") else 1),
            ],
            font_scale=0.34,
        )

    for i, s in enumerate(side_candidates):
        pts = s["points"].reshape(-1, 1, 2).astype(np.int32)
        color = random_color(i)
        cv2.polylines(out, [pts], False, color, thickness, cv2.LINE_AA)
        c = s["center"]
        label1 = f"{int(s['index'])} s={float(s.get('straightness', 0.0)):.2f}"
        label2 = (
            f"p90={float(s.get('p90_pca_line_error', 0.0)):.1f} "
            f"rms={float(s.get('pca_rms_error', 0.0)):.1f} "
            f"d/c={float(s.get('chord_deviation_ratio', 0.0)):.3f}"
        )
        cv2.putText(
            out,
            label1,
            (int(c[0]) + 4, int(c[1]) - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            black,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            label2,
            (int(c[0]) + 4, int(c[1]) + 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            black,
            1,
            cv2.LINE_AA,
        )

    cv2.putText(
        out,
        "Side prefilter candidates entering PCA direction clustering",
        (15, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        out,
        "gray = rejected before clustering, colored = possible side strokes",
        (15, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        red,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        out,
        "red labels = non-length rejected strokes, slightly bolder values = failed checks",
        (15, 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        red,
        1,
        cv2.LINE_AA,
    )
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
                f"p90_pca_line_error={s.get('p90_pca_line_error', 0.0):.2f}, "
                f"pca_rms_error={s.get('pca_rms_error', 0.0):.2f}, "
                f"chord_dev_ratio={s.get('chord_deviation_ratio', 0.0):.4f}, "
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
                "p90_pca_line_error": float(s.get("p90_pca_line_error", 0.0)),
                "pca_rms_error": float(s.get("pca_rms_error", 0.0)),
                "p90_chord_deviation": float(s.get("p90_chord_deviation", 0.0)),
                "chord_deviation_ratio": float(s.get("chord_deviation_ratio", 0.0)),
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


def write_side_direction_prefilter_report(path, infos, args):
    """Write all side-direction prefilter decisions before PCA clustering."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("==== Side Direction Prefilter ====\n\n")
        f.write("These filters run before PCA direction clustering for side-stroke selection.\n\n")
        f.write(f"min_stroke_length/arc: {float(args.min_stroke_length):.2f}\n")
        f.write(f"side_min_chord_px: {float(args.side_min_chord_px):.2f}\n")
        f.write(f"side_straightness: {float(args.side_straightness):.3f}\n")
        f.write(f"side_line_p90_error_px: {float(args.side_line_p90_error_px):.2f}\n")
        f.write(f"side_line_p90_error_ratio: {float(args.side_line_p90_error_ratio):.4f}\n")
        f.write(f"side_line_rms_error_px: {float(args.side_line_rms_error_px):.2f}\n")
        f.write(f"side_line_rms_error_ratio: {float(args.side_line_rms_error_ratio):.4f}\n")
        f.write(f"side_chord_dev_ratio_max: {float(args.side_chord_dev_ratio_max):.4f}\n\n")

        accepted_count = 0
        rejected_count = 0
        for s in infos:
            result = side_direction_group_filter_result(
                s,
                min_length=args.min_stroke_length,
                min_straightness=args.side_straightness,
                min_chord=args.side_min_chord_px,
                line_p90_error_px=args.side_line_p90_error_px,
                line_p90_error_ratio=args.side_line_p90_error_ratio,
                line_rms_error_px=args.side_line_rms_error_px,
                line_rms_error_ratio=args.side_line_rms_error_ratio,
                chord_dev_ratio_max=args.side_chord_dev_ratio_max,
            )
            if result["accepted"]:
                accepted_count += 1
            else:
                rejected_count += 1

            p90_limit = result["p90_pca_limit"]
            rms_limit = result["pca_rms_limit"]
            p90_limit_txt = "disabled" if p90_limit is None else f"{p90_limit:.2f}"
            rms_limit_txt = "disabled" if rms_limit is None else f"{rms_limit:.2f}"
            reasons = "OK" if result["accepted"] else "; ".join(result["reasons"])
            f.write(
                f"stroke {int(s['index']):03d}: "
                f"{'ACCEPT' if result['accepted'] else 'reject'}; "
                f"arc={float(s.get('arc', 0.0)):.1f}, "
                f"chord={float(s.get('chord', 0.0)):.1f}, "
                f"straightness={float(s.get('straightness', 0.0)):.3f}, "
                f"p90_pca_line_error={float(s.get('p90_pca_line_error', 0.0)):.2f}/{p90_limit_txt}, "
                f"pca_rms_error={float(s.get('pca_rms_error', 0.0)):.2f}/{rms_limit_txt}, "
                f"p90_chord_deviation={float(s.get('p90_chord_deviation', 0.0)):.2f}, "
                f"chord_dev_ratio={float(s.get('chord_deviation_ratio', 0.0)):.4f}; "
                f"{reasons}\n"
            )

        f.write(f"\naccepted: {accepted_count}\n")
        f.write(f"rejected: {rejected_count}\n")


def write_direction_groups_debug_report(path, strokes, angle_thresh=25.0, min_stroke_length=None, min_straightness=None, args=None):
    """Write side-prefiltered direction groups produced by unoriented angle similarity."""
    groups = build_direction_clusters(strokes, angle_thresh=angle_thresh)
    with open(path, "w", encoding="utf-8") as f:
        f.write("==== Side-Prefiltered Stroke Direction Groups ====\n\n")
        if min_stroke_length is not None:
            f.write(f"min_stroke_length: {float(min_stroke_length):.2f}\n")
        if min_straightness is not None:
            f.write(f"side_straightness: {float(min_straightness):.2f}\n")
        if args is not None:
            f.write(f"side_min_chord_px: {float(args.side_min_chord_px):.2f}\n")
            f.write(
                "side line p90 limit: "
                f"max({float(args.side_line_p90_error_px):.2f}, chord * {float(args.side_line_p90_error_ratio):.4f})\n"
            )
            f.write(
                "side line rms limit: "
                f"max({float(args.side_line_rms_error_px):.2f}, chord * {float(args.side_line_rms_error_ratio):.4f})\n"
            )
            f.write(f"side_chord_dev_ratio_max: {float(args.side_chord_dev_ratio_max):.4f}\n")
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
                    f"arc={s['arc']:.1f}, chord={s['chord']:.1f}, "
                    f"straightness={s['straightness']:.3f}, "
                    f"p90_pca_line_error={s.get('p90_pca_line_error', 0.0):.2f}, "
                    f"pca_rms_error={s.get('pca_rms_error', 0.0):.2f}, "
                    f"chord_dev_ratio={s.get('chord_deviation_ratio', 0.0):.4f}\n"
                )


def draw_direction_groups_image(shape, strokes, angle_thresh=25.0, thickness=4, min_stroke_length=None, min_straightness=None):
    """Draw side-prefiltered direction groups with distinct colors and group angles."""
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
        f"Side-prefiltered direction groups, threshold={angle_thresh:.1f} deg",
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
    if min_straightness is not None:
        cv2.putText(
            out,
            f"Only strokes with straightness >= {float(min_straightness):.2f} are grouped for side clustering",
            (15, 100),
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
        f"count_term={details.get('score_count_term', 0.0):.1f}, "
        f"same_loop_penalty_term={details.get('score_same_loop_penalty_term', 0.0):.1f}, "
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
            f"best_cap_bbox_area={details.get('best_cap_bbox_area', 0)}, "
            f"best_sweep_iou={details.get('best_sweep_iou', 0.0):.4f}, "
            f"side_cap_connected={details.get('side_cap_connected_count', 0)}/{details.get('side_cap_side_count', 0)}, "
            f"side_cap_passed={details.get('side_cap_connect_passed', True)}, "
            f"side_cap_ignored={details.get('side_cap_ignored_disconnected_strokes', [])}, "
            f"best_cap_strokes={details.get('best_cap_strokes', [])}, "
            f"best_cap_topology={details.get('best_cap_topology_kind', '')}, "
        f"invalid_no_cap={details.get('invalid_no_cap', False)}, "
        f"selection_passed={details.get('selection_passed', False)}, "
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
        f.write(
            "active_score_formula: count_weight * n - same_loop_penalty * same_loop_pairs\n"
        )
        f.write(
            "connected-pair/length/straightness/spread/length-similarity fields below are diagnostics only.\n"
        )
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
                f.write(f"    connected_pairs={details.get('connected_pairs', [])} -> diagnostic_only\n")
            if details.get("cap_validation_checked", False):
                f.write(
                f"    cap_validation: total_candidates={details.get('cap_candidate_count', 0)}, "
                f"best_area={details.get('best_cap_area', 0)}, "
                f"best_enclosed_area={details.get('best_cap_enclosed_area', 0)}, "
                f"best_bbox_area={details.get('best_cap_bbox_area', 0)}, "
                f"best_sweep_iou={details.get('best_sweep_iou', 0.0):.4f}, "
                f"best_sweep_valid={details.get('best_sweep_valid', False)}, "
                f"best_sweep_passed={details.get('best_sweep_passed', False)}, "
                f"best_sweep_mask_source={details.get('best_sweep_mask_source', '')}, "
                f"side_cap_connect_enabled={details.get('side_cap_connect_enabled', False)}, "
                f"side_cap_connect_tol={details.get('side_cap_connect_tol', 0.0):.1f}, "
                f"side_cap_connected={details.get('side_cap_connected_count', 0)}/{details.get('side_cap_side_count', 0)}, "
                f"side_cap_range_checked={details.get('side_cap_range_checked_count', 0)}, "
                f"side_cap_range_method={details.get('side_cap_range_method', '')}, "
                f"side_cap_passed={details.get('side_cap_connect_passed', True)}, "
                f"side_cap_disconnected={details.get('side_cap_disconnected_strokes', [])}, "
                f"side_cap_ignored={details.get('side_cap_ignored_disconnected_strokes', [])}, "
                f"best_strokes={details.get('best_cap_strokes', [])}, "
                f"best_score={details.get('best_cap_score', 0.0):.1f}, "
                f"best_center={details.get('best_cap_center', None)}, "
                f"best_total_arc={details.get('best_cap_total_arc', 0.0):.1f}, "
                f"best_topology={details.get('best_cap_topology_kind', '')}, "
                f"loop_detection={details.get('best_cap_loop_detection', '')}, "
                f"cycles={details.get('best_cap_topology_cycle_count', 0)}, "
                f"simple_cycles={details.get('best_cap_topology_simple_cycle_count', 0)}, "
                f"edge_disjoint_cover={details.get('best_cap_topology_edge_disjoint_cover', False)}, "
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
        f.write("Full direction groups are checked first. Failed groups are then expanded level by level: remove-1, remove-2, ... until success or one stroke remains.\n")
        f.write("Every subgroup in the same removal-depth round is evaluated before deciding whether that round can stop the search.\n\n")

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
                f"cap_evaluated={int(item.get('cap_candidate_evaluated_count', 0))}, "
                f"best_area={int(item.get('best_cap_area', 0))}, "
                f"best_enclosed_area={int(item.get('best_cap_enclosed_area', 0))}, "
                f"best_bbox_area={int(item.get('best_cap_bbox_area', 0))}, "
                f"best_sweep_iou={float(item.get('best_sweep_iou', 0.0)):.4f}, "
                f"best_sweep_valid={item.get('best_sweep_valid', False)}, "
                f"best_sweep_mask_source={item.get('best_sweep_mask_source', '')}, "
                f"side_cap_connected={int(item.get('side_cap_connected_count', 0))}/{int(item.get('side_cap_side_count', 0))}, "
                f"side_cap_range_checked={int(item.get('side_cap_range_checked_count', 0))}, "
                f"side_cap_range_method={item.get('side_cap_range_method', '')}, "
                f"side_cap_passed={item.get('side_cap_connect_passed', True)}, "
                f"side_cap_disconnected={item.get('side_cap_disconnected_strokes', [])}, "
                f"side_cap_ignored={item.get('side_cap_ignored_disconnected_strokes', [])}, "
                f"selection_passed={item.get('selection_passed', False)}, "
                f"best_topology={item.get('best_cap_topology_kind', '')}, "
                f"cycles={int(item.get('best_cap_topology_cycle_count', 0))}, "
                f"simple_cycles={int(item.get('best_cap_topology_simple_cycle_count', 0))}, "
                f"edge_disjoint_cover={item.get('best_cap_topology_edge_disjoint_cover', False)}, "
                f"best_total_arc={float(item.get('best_cap_total_arc', 0.0)):.1f}, "
                f"best_score={float(item.get('best_cap_score', 0.0)):.1f}, "
                f"best_cap_strokes={item.get('best_cap_strokes', [])}\n"
            )


def write_cap_search_round_report(path, model):
    """Write one line per removal-depth round for sweep-gated cap search."""
    rounds = model.get("cap_search_rounds", []) if model is not None else []
    with open(path, "w", encoding="utf-8") as f:
        f.write("==== Cap Search Rounds ====\n\n")
        if model is not None:
            f.write(f"sweep_gate_enabled: {bool(model.get('cap_sweep_gate_enabled', False))}\n")
            f.write(f"side_cap_connect_enabled: {bool(model.get('side_cap_connect_enabled', False))}\n")
            f.write(f"side_cap_connect_tol: {float(model.get('side_cap_connect_tol', 0.0)):.2f}\n")
            f.write(f"cap_search_time_limit_sec: {float(model.get('cap_search_time_limit_sec', 0.0)):.2f}\n")
            f.write(f"cap_search_elapsed_sec: {float(model.get('cap_search_elapsed_sec', 0.0)):.2f}\n")
            f.write(f"cap_search_time_limit_reached: {bool(model.get('cap_search_time_limit_reached', False))}\n")
            f.write(f"cap_search_stop_reason: {model.get('cap_search_stop_reason', '')}\n")
        if not rounds:
            f.write("No round summary recorded.\n")
            return
        for item in rounds:
            f.write(
                f"round depth={int(item.get('removal_depth', 0))}: "
                f"entries={int(item.get('entry_count', 0))}, "
                f"cap_count={int(item.get('cap_count', 0))}, "
                f"passed_count={int(item.get('passed_count', 0))}, "
                f"best_iou_cluster={item.get('best_iou_cluster_id', None)}, "
                f"best_iou={float(item.get('best_iou', 0.0)):.4f}, "
                f"winner_cluster={item.get('winner_cluster_id', None)}, "
                f"stopped={bool(item.get('stopped', False))}, "
                f"elapsed_sec={float(item.get('elapsed_sec', 0.0)):.2f}, "
                f"time_limit_reached={bool(item.get('time_limit_reached', False))}, "
                f"stop_reason={item.get('stop_reason', '')}\n"
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
    write_cap_search_round_report(os.path.join(debug_dir, "05e_cap_search_rounds.txt"), model)
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


def serialize_stroke_info_for_cap_pool(s):
    """Serialize one stroke with the geometry needed to inspect cap-pool choices."""
    pts = np.asarray(s.get("points", []), dtype=np.float64).reshape(-1, 2)
    p0 = pts[0] if len(pts) > 0 else np.asarray([0.0, 0.0], dtype=np.float64)
    p1 = pts[-1] if len(pts) > 0 else np.asarray([0.0, 0.0], dtype=np.float64)
    center = np.asarray(s.get("center", (0.0, 0.0)), dtype=np.float64)
    direction = np.asarray(s.get("direction", (0.0, 0.0)), dtype=np.float64)
    return {
        "index": int(s.get("index", -1)),
        "arc": float(s.get("arc", 0.0)),
        "chord": float(s.get("chord", 0.0)),
        "straightness": float(s.get("straightness", 0.0)),
        "p90_pca_line_error": float(s.get("p90_pca_line_error", 0.0)),
        "pca_rms_error": float(s.get("pca_rms_error", 0.0)),
        "p90_chord_deviation": float(s.get("p90_chord_deviation", 0.0)),
        "chord_deviation_ratio": float(s.get("chord_deviation_ratio", 0.0)),
        "center": [float(center[0]), float(center[1])],
        "direction": [float(direction[0]), float(direction[1])],
        "p0": [float(p0[0]), float(p0[1])],
        "p1": [float(p1[0]), float(p1[1])],
        "points": [[float(x), float(y)] for x, y in pts.tolist()],
    }


def draw_cap_pool_image(shape, entry, infos, cap_pool_infos):
    """Visualize strokes available to cap-loop search for one side entry."""
    h, w = shape[:2]
    out = np.full((h, w, 3), 255, dtype=np.uint8)
    side_ids = {int(s.get("index", -1)) for s in entry.get("strokes", [])}
    cap_pool_ids = {int(s.get("index", -1)) for s in cap_pool_infos}
    available_ids = cap_pool_ids - side_ids

    for s in infos:
        sid = int(s.get("index", -1))
        pts = np.asarray(s["points"], dtype=np.float64).reshape(-1, 1, 2).astype(np.int32)
        if sid in available_ids:
            color = (0, 170, 0)
            thickness = 3
        elif sid in side_ids:
            color = (255, 0, 0)
            thickness = 4
        elif sid in cap_pool_ids:
            color = (160, 160, 160)
            thickness = 2
        else:
            color = (225, 225, 225)
            thickness = 1
        cv2.polylines(out, [pts], False, color, thickness, cv2.LINE_AA)

        if sid in available_ids or sid in side_ids:
            c = np.asarray(s["center"], dtype=np.float64)
            label = f"s{sid}"
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

    cluster_id = int(entry.get("cluster_id", -1))
    cv2.putText(
        out,
        f"C{cluster_id} cap stroke pool: green=available, blue=side excluded",
        (15, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        out,
        f"available={len(available_ids)} side_excluded={len(side_ids & cap_pool_ids)} cap_pool_total={len(cap_pool_ids)}",
        (15, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    return out


def write_cap_pool_debug_json(path, entry, rank, cap_pool_infos):
    side_ids = {int(s.get("index", -1)) for s in entry.get("strokes", [])}
    cap_pool_ids = {int(s.get("index", -1)) for s in cap_pool_infos}
    available = [s for s in cap_pool_infos if int(s.get("index", -1)) not in side_ids]
    excluded_side_pool = [s for s in cap_pool_infos if int(s.get("index", -1)) in side_ids]
    payload = {
        "rank": int(rank),
        "cluster_id": int(entry.get("cluster_id", -1)),
        "source": entry.get("source", "unknown"),
        "parent_cluster_id": entry.get("parent_cluster_id", None),
        "removal_depth": int(entry.get("removal_depth", 0)),
        "removed_stroke_indices": list(entry.get("removed_stroke_indices", [])),
        "side_indices": sorted(side_ids),
        "cap_pool_indices_before_side_removal": sorted(cap_pool_ids),
        "available_cap_pool_indices": [int(s.get("index", -1)) for s in available],
        "side_indices_excluded_from_cap_pool": [int(s.get("index", -1)) for s in excluded_side_pool],
        "available_cap_pool_count": int(len(available)),
        "cap_pool_count_before_side_removal": int(len(cap_pool_infos)),
        "available_cap_pool_strokes": [serialize_stroke_info_for_cap_pool(s) for s in available],
        "side_strokes_excluded_from_cap_pool": [serialize_stroke_info_for_cap_pool(s) for s in excluded_side_pool],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_safe_debug_value(payload), f, indent=2)


def component_endpoint_debug_records(strokes, comp, endpoint_tol):
    records = []
    comp = list(map(int, comp))
    for stroke_local_i in comp:
        s = strokes[stroke_local_i]
        for endpoint_i, p in enumerate(stroke_endpoint_points(s)):
            degree, matches = endpoint_connection_degree_in_component(
                strokes,
                comp,
                stroke_local_i,
                endpoint_i,
                endpoint_tol=endpoint_tol,
            )
            records.append({
                "stroke": int(s.get("index", -1)),
                "role": "start" if endpoint_i == 0 else "end",
                "point": [float(p[0]), float(p[1])],
                "degree": int(degree),
                "matches": [
                    {
                        "stroke": int(match_stroke),
                        "role": "start" if int(match_endpoint_i) == 0 else "end",
                    }
                    for match_stroke, match_endpoint_i in matches
                ],
            })
    return records


def write_component_normalization_debug_json(path, entry, rank, cap_pool_infos, endpoint_tol):
    side_ids = {int(s.get("index", -1)) for s in entry.get("strokes", [])}
    non_side_infos = [s for s in cap_pool_infos if int(s.get("index", -1)) not in side_ids]
    comps = connected_components_by_endpoint_proximity(non_side_infos, endpoint_tol=endpoint_tol)

    component_records = []
    for component_i, comp in enumerate(comps):
        comp = list(map(int, comp))
        normalized = normalize_cap_loop_component(
            non_side_infos,
            comp,
            endpoint_tol=endpoint_tol,
        )
        norm_comp = list(map(int, normalized.get("component_local_indices", [])))
        component_records.append({
            "component_index": int(component_i),
            "component": {
                "local_indices": comp,
                "stroke_indices": [int(non_side_infos[i].get("index", -1)) for i in comp],
                "closed": bool(is_closed_stroke_component_by_endpoint_proximity(
                    non_side_infos,
                    comp,
                    endpoint_tol=endpoint_tol,
                )),
                "endpoints": component_endpoint_debug_records(
                    non_side_infos,
                    comp,
                    endpoint_tol,
                ),
            },
            "normalized_component": {
                "local_indices": norm_comp,
                "stroke_indices": [int(non_side_infos[i].get("index", -1)) for i in norm_comp],
                "closed": bool(normalized.get("closed", False)),
                "removed_open_branch_strokes": list(normalized.get("removed_open_branch_strokes", [])),
                "removed_post_loop_self_strokes": list(normalized.get("removed_post_loop_self_strokes", [])),
                "trace": list(normalized.get("trace", [])),
                "endpoints": component_endpoint_debug_records(
                    non_side_infos,
                    norm_comp,
                    endpoint_tol,
                ) if norm_comp else [],
            },
        })

    payload = {
        "rank": int(rank),
        "cluster_id": int(entry.get("cluster_id", -1)),
        "source": entry.get("source", "unknown"),
        "parent_cluster_id": entry.get("parent_cluster_id", None),
        "removal_depth": int(entry.get("removal_depth", 0)),
        "removed_stroke_indices": list(entry.get("removed_stroke_indices", [])),
        "side_indices": sorted(side_ids),
        "endpoint_tol": float(endpoint_tol),
        "non_side_cap_pool_indices": [int(s.get("index", -1)) for s in non_side_infos],
        "component_count": int(len(component_records)),
        "components": component_records,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_safe_debug_value(payload), f, indent=2)


def draw_component_normalization_image(shape, entry, cap_pool_infos, endpoint_tol, component_index=None):
    """Draw raw and normalized cap components for one side-cluster trial."""
    h, w = shape[:2]
    out = np.full((h, w, 3), 255, dtype=np.uint8)
    side_ids = {int(s.get("index", -1)) for s in entry.get("strokes", [])}
    non_side_infos = [s for s in cap_pool_infos if int(s.get("index", -1)) not in side_ids]
    comps = connected_components_by_endpoint_proximity(non_side_infos, endpoint_tol=endpoint_tol)

    for s in entry.get("strokes", []):
        pts = s["points"].reshape(-1, 1, 2).astype(np.int32)
        cv2.polylines(out, [pts], False, (0, 0, 255), 4, cv2.LINE_AA)
        c = s["center"]
        cv2.putText(out, f"side s{int(s['index'])}", (int(c[0]), int(c[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 160), 1, cv2.LINE_AA)

    selected_component_indices = (
        set(range(len(comps)))
        if component_index is None
        else {int(component_index)}
    )

    for ci, comp in enumerate(comps):
        if ci not in selected_component_indices:
            continue
        comp = list(map(int, comp))
        normalized = normalize_cap_loop_component(non_side_infos, comp, endpoint_tol=endpoint_tol)
        norm_comp = list(map(int, normalized.get("component_local_indices", [])))
        norm_set = set(norm_comp)
        removed_branch = set(int(x) for x in normalized.get("removed_open_branch_strokes", []))
        removed_self = set(int(x) for x in normalized.get("removed_post_loop_self_strokes", []))
        normalized_closed = bool(normalized.get("closed", False))
        color = random_color(ci + 80)
        norm_color = (0, 170, 0) if normalized_closed else (0, 140, 255)

        all_points = []
        for li in comp:
            s = non_side_infos[li]
            pts = s["points"].reshape(-1, 1, 2).astype(np.int32)
            cv2.polylines(out, [pts], False, (210, 210, 210), 2, cv2.LINE_AA)
            all_points.extend([np.asarray(p, dtype=np.float64) for p in stroke_endpoint_points(s)])

        for li in comp:
            s = non_side_infos[li]
            pts = s["points"].reshape(-1, 1, 2).astype(np.int32)
            sid = int(s["index"])
            if li in norm_set:
                stroke_color = norm_color
                thickness = 4
                label_prefix = "norm"
            elif sid in removed_branch or sid in removed_self:
                stroke_color = (150, 150, 150)
                thickness = 2
                label_prefix = "removed"
            else:
                stroke_color = color
                thickness = 2
                label_prefix = "raw"
            cv2.polylines(out, [pts], False, stroke_color, thickness, cv2.LINE_AA)
            c = s["center"]
            cv2.putText(out, f"{label_prefix} s{sid}", (int(c[0]), int(c[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, stroke_color, 1, cv2.LINE_AA)

        for li in norm_comp:
            s = non_side_infos[li]
            for endpoint_i, p in enumerate(stroke_endpoint_points(s)):
                degree, _matches = endpoint_connection_degree_in_component(
                    non_side_infos,
                    norm_comp,
                    li,
                    endpoint_i,
                    endpoint_tol=endpoint_tol,
                )
                px, py = int(round(float(p[0]))), int(round(float(p[1])))
                endpoint_color = (0, 150, 0) if degree == 1 else (0, 0, 255)
                cv2.circle(out, (px, py), 5 if degree == 1 else 7, endpoint_color, 2, cv2.LINE_AA)
                cv2.putText(out, f"d{degree}", (px + 5, py - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, endpoint_color, 1, cv2.LINE_AA)

        if all_points:
            label_pos = np.mean(np.asarray(all_points), axis=0)
            status = "closed" if normalized_closed else "open"
            norm_strokes = [int(non_side_infos[i]["index"]) for i in norm_comp]
            cv2.putText(out, f"component {ci}: normalized {status}, strokes={norm_strokes}",
                        (int(label_pos[0]) + 8, int(label_pos[1]) - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, norm_color, 1, cv2.LINE_AA)

    title_component = "all" if component_index is None else str(int(component_index))
    cv2.putText(out, f"cluster {entry.get('cluster_id', -1)} normalized cap components: {title_component}",
                (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(out, "red=side removed; green=normalized closed; orange=normalized open; gray=removed/raw",
                (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def save_component_normalization_debug_pngs(out_dir, img_shape, entry, cap_pool_infos, endpoint_tol):
    """Write normalized cap component overview and per-component PNGs."""
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(
        os.path.join(out_dir, "components_normalized.png"),
        draw_component_normalization_image(img_shape, entry, cap_pool_infos, endpoint_tol),
    )

    side_ids = {int(s.get("index", -1)) for s in entry.get("strokes", [])}
    non_side_infos = [s for s in cap_pool_infos if int(s.get("index", -1)) not in side_ids]
    comps = connected_components_by_endpoint_proximity(non_side_infos, endpoint_tol=endpoint_tol)
    for ci, _comp in enumerate(comps):
        cv2.imwrite(
            os.path.join(out_dir, f"normalized_component_{ci:03d}.png"),
            draw_component_normalization_image(
                img_shape,
                entry,
                cap_pool_infos,
                endpoint_tol,
                component_index=ci,
            ),
        )


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
    iou_cap_mask = cap_mask if cap_mask_override is not None else cap_candidate.get("mask", None)
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
        normalized = normalize_cap_loop_component(non_side_infos, comp, endpoint_tol=endpoint_tol)
        normalized_closed = bool(normalized.get("closed", False))
        normalized_comp = normalized.get("component_local_indices", [])
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
            normalized_status = "closed" if normalized_closed else "open"
            cv2.putText(out, f"comp {ci}: {status}, prune={pruned_status}, norm={normalized_status}, n={len(normalized_comp)}/{len(comp)}",
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
    cap_pool_infos = build_cap_pool_infos(
        infos,
        min_arc=CAP_POOL_MIN_ARC,
        endpoint_tol=args.cap_loop_endpoint_tol,
        keep_short_if_connected_gt=CAP_POOL_SHORT_STROKE_KEEP_CONNECTED_GT,
    )
    summary_path = os.path.join(out_dir, "cap_endpoint_graph_summary.txt")

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("==== Cap Endpoint Graph Summary ====\n\n")
        f.write(f"cap_pool: strokes with arc >= {float(CAP_POOL_MIN_ARC):.1f}, plus short strokes whose two endpoints together touch > {int(CAP_POOL_SHORT_STROKE_KEEP_CONNECTED_GT)} other strokes within endpoint_tol\n")
        f.write(f"endpoint_tol: {float(args.cap_loop_endpoint_tol):.1f}\n")
        f.write("Original direction groups are listed. Selected/pass subgroups are also listed so 3D recovery can use their endpoint graph.\n\n")

        selected_cluster_id = model.get("selected_cluster_id", None)

        def should_emit_endpoint_graph(entry):
            if int(entry.get("removal_depth", 0)) == 0 and entry.get("parent_cluster_id", None) is None:
                return True
            details = entry.get("details", {})
            if selected_cluster_id is not None and int(entry.get("cluster_id", -1)) == int(selected_cluster_id):
                return True
            iou_output_thresh = float(getattr(args, "iou_rank_output_thresh", 0.0) or 0.0)
            if (
                bool(details.get("best_sweep_valid", False))
                and float(details.get("best_sweep_iou", 0.0)) >= iou_output_thresh
            ):
                return True
            return bool(details.get("selection_passed", False) or details.get("selected_by_cap_validation", False))

        for entry in cluster_debug:
            if not should_emit_endpoint_graph(entry):
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
                normalized = normalize_cap_loop_component(
                    non_side_infos,
                    comp,
                    endpoint_tol=args.cap_loop_endpoint_tol,
                )
                f.write(
                    f"  component {ci:03d}: closed={closed}, strokes={gids}, "
                    f"pruned_closed={pruned_closed}, pruned_strokes={pruned_gids}, "
                    f"removed_open_branch_strokes={removed_gids}, "
                    f"normalized_closed={bool(normalized.get('closed', False))}, "
                    f"normalized_strokes={normalized.get('component_stroke_indices', [])}, "
                    f"removed_post_loop_self_strokes={normalized.get('removed_post_loop_self_strokes', [])}\n"
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
    """Return original direction group ids that eventually produced a selectable result."""
    ids = set()
    if model is None:
        return ids
    for item in model.get("cap_search_trace", []):
        if not item.get("selection_passed", item.get("cap_found", False)):
            continue
        parent_id = item.get("parent_cluster_id", None)
        if parent_id is None:
            parent_id = item.get("cluster_id", None)
        if parent_id is not None:
            ids.add(int(parent_id))
    selected_cluster_id = model.get("selected_cluster_id", None)
    if selected_cluster_id is not None:
        for entry in model.get("cluster_debug", []):
            if int(entry.get("cluster_id", -1)) != int(selected_cluster_id):
                continue
            parent_id = entry.get("parent_cluster_id", None)
            if parent_id is None:
                parent_id = entry.get("cluster_id", None)
            if parent_id is not None:
                ids.add(int(parent_id))
            break
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

    For every stroke id in cap_candidate stroke_indices, draw the straight chord p0???????????????????????????????????????????????????????????????? from
    05a_stroke_directions.txt (same line thickness as the primary loop strokes).

    Also draw each summary match as a segment between the matched endpoints (05a positions);
    when endpoints are farther apart than gap_connector_px, use a thinner bridge stroke.

    No summary-file coordinate fallback ??missing 05a data yields None.
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
    IoU / sweep masks use a strict endpoint-ordered, gap-bridged, filled cap face.

    Sweep vector = longest side stroke by arc length, oriented near???????????????????????????????????????????????????????????????????????????????????????????vs source cap mask.
    """
    if cap_candidate is None:
        return None
    stroke_indices = cap_candidate.get("stroke_indices", [])
    wanted = {int(i) for i in stroke_indices}
    loop_strokes = collect_strokes_by_indices(infos, stroke_indices)
    if len(loop_strokes) != len(wanted):
        return None
    bt = max(2, int(cap_loop_thickness))

    source_cap, mask_source = cap_ordered_filled_sweep_source_mask(
        cap_candidate,
        infos,
        endpoint_tol=endpoint_tol,
        thickness=bt,
    )
    if source_cap is None or np.count_nonzero(source_cap > 0) == 0:
        return None

    geometry = longest_side_stroke_copy_geometry(
        entry,
        cap_candidate,
        cap_mask_override=source_cap,
        direction_angle_tol=copy_direction_angle_tol,
        sketch_mask=sketch_mask,
        iou_compare_percent=iou_compare_percent,
    )
    if geometry is None:
        return None
    vec = geometry["vector"]
    copied_cap = translate_mask(source_cap, vec)
    side_bin = rasterize_side_strokes(shape, entry.get("strokes", []), thickness=4)
    swept_cap = fill_binary_mask(sweep_mask_along_vector(source_cap, vec))
    occupied = fill_binary_mask(swept_cap)
    return {
        "source_cap": source_cap,
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


def compute_cap_sweep_similarity(
    shape,
    entry,
    cap_candidate,
    infos,
    endpoint_tol,
    cap_loop_thickness,
    sketch_mask=None,
    copy_direction_angle_tol=None,
    iou_compare_percent=None,
):
    """Compute cap-sweep occupancy IoU against the input enclosed mask."""
    result = {
        "valid": False,
        "iou": 0.0,
        "intersection": 0,
        "union": 0,
        "sweep_area": 0,
        "copy_side_stroke": None,
        "copy_reason": "",
        "mask_source": "",
        "geometry": None,
    }
    if cap_candidate is None or sketch_mask is None or np.count_nonzero(sketch_mask > 0) == 0:
        return result

    masks = build_bbox_masks_geometric(
        shape,
        entry,
        cap_candidate,
        infos,
        endpoint_tol,
        cap_loop_thickness,
        debug_dir=None,
        copy_direction_angle_tol=copy_direction_angle_tol,
        sketch_mask=sketch_mask,
        iou_compare_percent=iou_compare_percent,
    )
    if masks is None:
        result["mask_source"] = "strict_filled_cap_unavailable"
        return result

    occupied = masks.get("occupied", None)
    if occupied is None or np.count_nonzero(occupied > 0) == 0:
        return result

    iou, intersection, union = binary_mask_iou(occupied, sketch_mask)
    geometry = masks.get("geometry", None) or {}
    result.update({
        "valid": True,
        "iou": float(iou),
        "intersection": int(intersection),
        "union": int(union),
        "sweep_area": int(binary_mask_area(occupied)),
        "copy_side_stroke": (
            None if geometry.get("selected_stroke", None) is None else int(geometry.get("selected_stroke", -1))
        ),
        "copy_reason": str(geometry.get("copy_selection_reason", "")),
        "mask_source": str(masks.get("mask_source", "")),
        "geometry": geometry,
    })
    return result


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


def clean_debug_artifact_dir(path, remove_dirs=False):
    os.makedirs(path, exist_ok=True)
    for name in os.listdir(path):
        full_path = os.path.join(path, name)
        if os.path.isdir(full_path):
            if remove_dirs:
                shutil.rmtree(full_path)
            continue
        if name.lower().endswith(('.png', '.txt', '.json')):
            os.remove(full_path)


def prepare_cluster_progress_debug_dir(debug_dir):
    if debug_dir is None:
        return
    clean_debug_artifact_dir(os.path.join(debug_dir, 'clusters'), remove_dirs=True)


def cluster_debug_base_name(entry, rank):
    safe_score = sanitize_score_for_filename(float(entry.get('score', 0.0)))
    cluster_id = int(entry.get('cluster_id', -1))
    return f'rank_{rank:02d}_cluster_{cluster_id:02d}_score_{safe_score}'


def write_cluster_progress_status(debug_dir, entry, rank, phase, **extra):
    if debug_dir is None:
        return

    cluster_dir = os.path.join(debug_dir, 'clusters')
    os.makedirs(cluster_dir, exist_ok=True)
    base = cluster_debug_base_name(entry, rank)
    out_dir = os.path.join(cluster_dir, base)
    os.makedirs(out_dir, exist_ok=True)

    payload = {
        'updated_at': datetime.now().isoformat(timespec='seconds'),
        'phase': str(phase),
        'rank': int(rank),
        'base': base,
        'cluster_id': int(entry.get('cluster_id', -1)),
        'source': entry.get('source', 'unknown'),
        'parent_cluster_id': entry.get('parent_cluster_id', None),
        'removal_depth': int(entry.get('removal_depth', 0)),
        'removed_stroke_indices': list(entry.get('removed_stroke_indices', [])),
        'side_indices': [int(s['index']) for s in entry.get('strokes', [])],
        'n_strokes': int(len(entry.get('strokes', []))),
    }
    payload.update(json_safe_debug_value(extra))

    for path in (
        os.path.join(cluster_dir, 'current_status.json'),
        os.path.join(out_dir, 'status.json'),
    ):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2)


def write_cluster_cap_search_step(debug_dir, entry, rank, event, **extra):
    if debug_dir is None:
        return

    cluster_dir = os.path.join(debug_dir, 'clusters')
    os.makedirs(cluster_dir, exist_ok=True)
    base = cluster_debug_base_name(entry, rank)
    out_dir = os.path.join(cluster_dir, base)
    os.makedirs(out_dir, exist_ok=True)

    payload = {
        'updated_at': datetime.now().isoformat(timespec='seconds'),
        'event': str(event),
        'rank': int(rank),
        'base': base,
        'cluster_id': int(entry.get('cluster_id', -1)),
        'source': entry.get('source', 'unknown'),
        'parent_cluster_id': entry.get('parent_cluster_id', None),
        'removal_depth': int(entry.get('removal_depth', 0)),
        'removed_stroke_indices': list(entry.get('removed_stroke_indices', [])),
        'side_indices': [int(s['index']) for s in entry.get('strokes', [])],
        'n_strokes': int(len(entry.get('strokes', []))),
    }
    payload.update(json_safe_debug_value(extra))

    live_path = os.path.join(out_dir, 'cap_search_live.json')
    with open(live_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)

    steps_path = os.path.join(out_dir, 'cap_search_steps.jsonl')
    with open(steps_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(payload, ensure_ascii=True) + '\n')


def save_cluster_progress_preview_outputs(debug_dir, img_shape, entry, rank, infos, cap_pool_infos, endpoint_tol, stop_event=None):
    if debug_dir is None or cap_search_stop_requested(stop_event):
        return

    cluster_dir = os.path.join(debug_dir, 'clusters')
    os.makedirs(cluster_dir, exist_ok=True)
    out_dir = os.path.join(cluster_dir, cluster_debug_base_name(entry, rank))
    os.makedirs(out_dir, exist_ok=True)

    cv2.imwrite(os.path.join(out_dir, 'cluster.png'), draw_single_cluster_image(img_shape, entry, selected=False, thickness=4))
    cv2.imwrite(os.path.join(out_dir, 'cap_pool.png'), draw_cap_pool_image(img_shape, entry, infos, cap_pool_infos))
    write_cap_pool_debug_json(os.path.join(out_dir, 'cap_pool.json'), entry, rank, cap_pool_infos)
    write_component_normalization_debug_json(
        os.path.join(out_dir, 'components_normalized.json'),
        entry,
        rank,
        cap_pool_infos,
        endpoint_tol,
    )
    save_component_normalization_debug_pngs(
        out_dir,
        img_shape,
        entry,
        cap_pool_infos,
        endpoint_tol,
    )


def json_safe_debug_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): json_safe_debug_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe_debug_value(v) for v in value]
    return value


def serialize_cap_candidate_for_debug(cap_candidate):
    if cap_candidate is None:
        return None
    payload = {}
    for key, value in cap_candidate.items():
        if key in {'mask', 'enclosed_mask'}:
            continue
        payload[key] = json_safe_debug_value(value)
    mask = cap_candidate.get('mask', None)
    enclosed_mask = cap_candidate.get('enclosed_mask', None)
    payload['mask_area'] = 0 if mask is None else int(binary_mask_area(mask))
    payload['enclosed_mask_area'] = 0 if enclosed_mask is None else int(binary_mask_area(enclosed_mask))
    return payload


def summarize_cap_candidate_for_status(cap_candidate):
    if cap_candidate is None:
        return None
    return {
        'area': int(cap_candidate.get('area', 0)),
        'enclosed_area': int(cap_candidate.get('enclosed_area', 0)),
        'bbox_area': int(cap_candidate.get('bbox_area', 0)),
        'score': float(cap_candidate.get('score', 0.0)),
        'total_arc': float(cap_candidate.get('total_arc', 0.0)),
        'stroke_indices': [int(x) for x in cap_candidate.get('stroke_indices', [])],
        'topology_kind': cap_candidate.get('topology_kind', ''),
        'loop_detection': cap_candidate.get('loop_detection', ''),
        'mask_area': 0 if cap_candidate.get('mask', None) is None else int(binary_mask_area(cap_candidate['mask'])),
        'enclosed_mask_area': (
            0
            if cap_candidate.get('enclosed_mask', None) is None
            else int(binary_mask_area(cap_candidate['enclosed_mask']))
        ),
    }




def save_geometry_face_debug_outputs(cluster_out_dir, payload):
    if not payload:
        return

    real_mask = decode_debug_mask_png(payload.get("real_mask_png", None))
    support_mask = decode_debug_mask_png(payload.get("support_mask_png", None))
    final_mask = decode_debug_mask_png(payload.get("final_mask_png", None))
    if final_mask is None:
        return

    component_index = payload.get("component_index", None)
    origin = np.asarray(payload.get("origin", [0.0, 0.0]), dtype=np.float64).reshape(2)
    h, w = final_mask.shape[:2]
    debug_root = os.path.join(cluster_out_dir, "geometry_face_debug")
    os.makedirs(debug_root, exist_ok=True)
    component_name = "component_unknown" if component_index is None else f"component_{int(component_index):02d}"
    out_dir = os.path.join(debug_root, component_name)
    clean_debug_artifact_dir(out_dir, remove_dirs=True)
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(json_safe_debug_value(payload), f, indent=2)

    if real_mask is not None:
        cv2.imwrite(os.path.join(out_dir, "real_mask.png"), real_mask)
    if support_mask is not None:
        cv2.imwrite(os.path.join(out_dir, "support_mask.png"), support_mask)
    cv2.imwrite(os.path.join(out_dir, "final_mask.png"), final_mask)

    local_overlay = np.full((h, w, 3), 255, dtype=np.uint8)
    if final_mask is not None:
        colorize(local_overlay, final_mask, (220, 220, 220))
    if real_mask is not None:
        colorize(local_overlay, real_mask, (45, 45, 45))
    if support_mask is not None:
        colorize(local_overlay, support_mask, (0, 190, 255))

    support_overlay = local_overlay.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    for support_i, record in enumerate(payload.get("support_records", [])):
        roi = list(record.get("roi", []))
        if len(roi) == 4:
            x0, y0, rw, rh = [int(v) for v in roi]
            cv2.rectangle(support_overlay, (x0, y0), (x0 + max(0, rw - 1), y0 + max(0, rh - 1)), (0, 128, 255), 1, cv2.LINE_AA)
        p0 = record.get("start_point", None)
        p1 = record.get("end_point", None)
        if p0 is not None and p1 is not None:
            a = tuple(np.rint(np.asarray(p0, dtype=np.float64).reshape(2) - origin).astype(np.int32).tolist())
            b = tuple(np.rint(np.asarray(p1, dtype=np.float64).reshape(2) - origin).astype(np.int32).tolist())
            cv2.line(support_overlay, a, b, (0, 128, 255), 1, cv2.LINE_AA)
            cv2.circle(support_overlay, a, 2, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(support_overlay, b, 2, (255, 0, 0), -1, cv2.LINE_AA)
        text_y = 20 + 18 * support_i
        if text_y < h - 8:
            text = f"gap{support_i}: d={float(record.get('distance', 0.0)):.1f}"
            cv2.putText(support_overlay, text, (8, text_y), font, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(support_overlay, text, (8, text_y), font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(os.path.join(out_dir, "support_connectors_overlay.png"), support_overlay)

    face_overlay = local_overlay.copy()
    faces = list(payload.get("faces", []))
    for face_i, face in enumerate(faces):
        pts_abs = np.asarray(face.get("ordered_loop_points", []), dtype=np.float64).reshape(-1, 2)
        if len(pts_abs) < 3:
            continue
        pts = np.rint(pts_abs - origin[None, :]).astype(np.int32).reshape(-1, 1, 2)
        color = random_color(face_i + 700)
        fill = face_overlay.copy()
        cv2.fillPoly(fill, [pts], color)
        face_overlay = cv2.addWeighted(face_overlay, 1.0, fill, 0.30, 0.0)
        cv2.polylines(face_overlay, [pts], True, color, 2, cv2.LINE_AA)
        center = np.mean(pts[:, 0, :], axis=0)
        label = f"F{face_i} strokes={face.get('stroke_indices', [])}"
        cv2.putText(face_overlay, label, (int(center[0]), int(center[1])), font, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(face_overlay, label, (int(center[0]), int(center[1])), font, 0.45, color, 1, cv2.LINE_AA)
    if not faces:
        cv2.putText(face_overlay, "no geometry faces detected", (12, min(h - 12, 24)), font, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(face_overlay, "no geometry faces detected", (12, min(h - 12, 24)), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(os.path.join(out_dir, "face_regions_overlay.png"), face_overlay)

    cluster_png_path = os.path.join(cluster_out_dir, "cluster.png")
    cluster_img = cv2.imread(cluster_png_path)
    if cluster_img is not None:
        cluster_overlay = cluster_img.copy()
        for face_i, face in enumerate(faces):
            pts_abs = np.rint(np.asarray(face.get("ordered_loop_points", []), dtype=np.float64)).astype(np.int32).reshape(-1, 1, 2)
            if len(pts_abs) < 3:
                continue
            color = random_color(face_i + 700)
            fill = cluster_overlay.copy()
            cv2.fillPoly(fill, [pts_abs], color)
            cluster_overlay = cv2.addWeighted(cluster_overlay, 1.0, fill, 0.22, 0.0)
            cv2.polylines(cluster_overlay, [pts_abs], True, color, 2, cv2.LINE_AA)
        cv2.imwrite(os.path.join(out_dir, "face_regions_on_cluster.png"), cluster_overlay)



def restore_geometry_face_debug_outputs_from_steps(cluster_out_dir):
    steps_path = os.path.join(cluster_out_dir, "cap_search_steps.jsonl")
    if not os.path.isfile(steps_path):
        return

    debug_root = os.path.join(cluster_out_dir, "geometry_face_debug")
    clean_debug_artifact_dir(debug_root, remove_dirs=True)
    restored = False
    try:
        with open(steps_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except Exception:
                    continue
                if item.get("event") != "geometry_face_extraction_finished":
                    continue
                payload = item.get("geometry_face_debug", None)
                if not payload:
                    continue
                save_geometry_face_debug_outputs(cluster_out_dir, payload)
                restored = True
    except OSError:
        return

    if not restored and os.path.isdir(debug_root):
        try:
            os.rmdir(debug_root)
        except OSError:
            pass

def build_outer_face_pixel_debug_payload(area_faces, outer_face_i, outer_face_selection, component_index=None):
    if outer_face_i is None or len(area_faces) <= 1:
        return None

    payload_faces = []
    for i, face in enumerate(area_faces):
        pts = np.asarray(face.get("ordered_loop_points", []), dtype=np.float64).reshape(-1, 2)
        if len(pts) < 3:
            continue
        payload_faces.append({
            "face_index": int(i),
            "abs_area": float(face.get("abs_area", 0.0)),
            "stroke_indices": [int(x) for x in face.get("stroke_indices", [])],
            "ordered_loop_points": [[float(x), float(y)] for x, y in pts.tolist()],
        })
    if not payload_faces:
        return None

    return {
        "component_index": None if component_index is None else int(component_index),
        "dropped_outer_face_index": int(outer_face_i),
        "area_face_count": int(len(area_faces)),
        "faces": payload_faces,
        "outer_face_selection": json_safe_debug_value(outer_face_selection),
    }


def _outer_face_debug_text_chunks(text, max_chars=88):
    text = "" if text is None else str(text)
    if not text:
        return []
    return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]


def make_outer_face_pixel_debug_canvas(cluster_out_dir, faces, pad=24):
    cluster_png_path = os.path.join(cluster_out_dir, "cluster.png")
    base = cv2.imread(cluster_png_path)
    if base is not None:
        return base

    point_sets = []
    for face in faces:
        pts = np.asarray(face.get("ordered_loop_points", []), dtype=np.float64).reshape(-1, 2)
        if len(pts) >= 3:
            point_sets.append(pts)
    if not point_sets:
        return np.full((256, 256, 3), 255, dtype=np.uint8)

    all_pts = np.vstack(point_sets)
    max_x = int(math.ceil(float(np.max(all_pts[:, 0])))) + int(pad)
    max_y = int(math.ceil(float(np.max(all_pts[:, 1])))) + int(pad)
    width = max(64, int(max_x + 1))
    height = max(64, int(max_y + 1))
    return np.full((height, width, 3), 255, dtype=np.uint8)


def save_outer_face_pixel_debug_outputs(cluster_out_dir, payload):
    if not payload:
        return

    faces = list(payload.get("faces", []))
    if not faces:
        return

    outer_face_selection = payload.get("outer_face_selection", {}) or {}
    outer_face_i = outer_face_selection.get("selected_index", payload.get("dropped_outer_face_index", None))
    if outer_face_i is None:
        return
    outer_face_i = int(outer_face_i)
    if outer_face_i < 0 or outer_face_i >= len(faces):
        return

    component_index = payload.get("component_index", None)
    outer_root_dir = os.path.join(cluster_out_dir, "outer_face_pixel_debug")
    os.makedirs(outer_root_dir, exist_ok=True)
    component_name = "component_unknown" if component_index is None else f"component_{int(component_index):02d}"
    out_dir = os.path.join(outer_root_dir, component_name)
    clean_debug_artifact_dir(out_dir, remove_dirs=True)

    base = make_outer_face_pixel_debug_canvas(cluster_out_dir, faces)
    colors = [
        (36, 28, 237),
        (55, 180, 80),
        (255, 177, 32),
        (163, 73, 164),
        (0, 162, 232),
        (180, 120, 30),
    ]
    font = cv2.FONT_HERSHEY_SIMPLEX

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(json_safe_debug_value(payload), f, indent=2)

    overlay = base.copy()
    for i, face in enumerate(faces):
        pts = np.rint(np.asarray(face.get("ordered_loop_points", []), dtype=np.float64)).astype(np.int32).reshape(-1, 1, 2)
        if len(pts) < 3:
            continue
        color = colors[i % len(colors)]
        fill = overlay.copy()
        cv2.fillPoly(fill, [pts], color)
        overlay = cv2.addWeighted(overlay, 1.0, fill, 0.15, 0.0)
        cv2.polylines(overlay, [pts], True, color, 2, cv2.LINE_AA)
        cx = int(np.mean(pts[:, 0, 0]))
        cy = int(np.mean(pts[:, 0, 1]))
        label = f"F{i}"
        if i == outer_face_i:
            label += " outer"
        cv2.putText(overlay, label, (cx, cy), font, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(overlay, label, (cx, cy), font, 0.65, color, 1, cv2.LINE_AA)
    cv2.imwrite(os.path.join(out_dir, "area_faces_overlay.png"), overlay)

    for i, face in enumerate(faces):
        pts = np.rint(np.asarray(face.get("ordered_loop_points", []), dtype=np.float64)).astype(np.int32).reshape(-1, 1, 2)
        if len(pts) < 3:
            continue
        color = colors[i % len(colors)]
        single = base.copy()
        fill = single.copy()
        cv2.fillPoly(fill, [pts], color)
        single = cv2.addWeighted(single, 1.0, fill, 0.25, 0.0)
        cv2.polylines(single, [pts], True, color, 2, cv2.LINE_AA)
        lines = [
            f"F{i} area={float(face.get('abs_area', 0.0)):.1f}",
            "strokes=" + ",".join(map(str, face.get("stroke_indices", []))),
        ]
        y = 24
        for raw_line in lines:
            for line in _outer_face_debug_text_chunks(raw_line):
                cv2.putText(single, line, (15, y), font, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(single, line, (15, y), font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
                y += 24
        cv2.imwrite(os.path.join(out_dir, f"face_{i}_overlay.png"), single)

    masks, canvas = rasterize_planar_face_masks(faces)
    selection_faces = outer_face_selection.get("faces", [])
    contained_faces = []
    if 0 <= outer_face_i < len(selection_faces):
        contained_faces = list(selection_faces[outer_face_i].get("contained_faces", []))

    outer_pts = np.rint(np.asarray(faces[outer_face_i].get("ordered_loop_points", []), dtype=np.float64)).astype(np.int32).reshape(-1, 1, 2)
    outer_mask = None if outer_face_i >= len(masks) or masks[outer_face_i] is None else (np.asarray(masks[outer_face_i]["mask"], dtype=np.uint8) > 0)
    for item in contained_faces:
        inner_i = int(item.get("face_index", -1))
        if inner_i < 0 or inner_i >= len(faces):
            continue
        inner_pts = np.rint(np.asarray(faces[inner_i].get("ordered_loop_points", []), dtype=np.float64)).astype(np.int32).reshape(-1, 1, 2)

        overlay_img = base.copy()
        cv2.polylines(overlay_img, [outer_pts], True, (0, 0, 255), 3, cv2.LINE_AA)
        fill = overlay_img.copy()
        cv2.fillPoly(fill, [inner_pts], (0, 255, 255))
        overlay_img = cv2.addWeighted(overlay_img, 1.0, fill, 0.35, 0.0)
        cv2.polylines(overlay_img, [inner_pts], True, (0, 180, 180), 2, cv2.LINE_AA)
        lines = [
            f"F{outer_face_i} contains F{inner_i}",
            f"overlap={int(item.get('overlap_pixels', 0))} / inner={int(item.get('inner_pixels', 0))}",
            f"ratio={float(item.get('contained_ratio', 0.0)):.4f}",
        ]
        y = 24
        for line in lines:
            cv2.putText(overlay_img, line, (15, y), font, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(overlay_img, line, (15, y), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            y += 24
        cv2.imwrite(os.path.join(out_dir, f"containment_outer_{outer_face_i}_inner_{inner_i}.png"), overlay_img)

        inner_mask = None if inner_i >= len(masks) or masks[inner_i] is None else (np.asarray(masks[inner_i]["interior_mask"], dtype=np.uint8) > 0)
        if outer_mask is None or inner_mask is None:
            continue
        overlap_mask = outer_mask & inner_mask
        mask_vis = np.full((outer_mask.shape[0], outer_mask.shape[1], 3), 255, dtype=np.uint8)
        colorize(mask_vis, outer_mask.astype(np.uint8) * 255, (220, 220, 255))
        colorize(mask_vis, inner_mask.astype(np.uint8) * 255, (150, 235, 235))
        colorize(mask_vis, overlap_mask.astype(np.uint8) * 255, (0, 0, 255))
        cv2.imwrite(os.path.join(out_dir, f"containment_outer_{outer_face_i}_inner_{inner_i}_mask.png"), mask_vis)



def restore_outer_face_pixel_debug_outputs_from_steps(cluster_out_dir):
    steps_path = os.path.join(cluster_out_dir, "cap_search_steps.jsonl")
    if not os.path.isfile(steps_path):
        return

    outer_root_dir = os.path.join(cluster_out_dir, "outer_face_pixel_debug")
    clean_debug_artifact_dir(outer_root_dir, remove_dirs=True)
    restored = False
    try:
        with open(steps_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except Exception:
                    continue
                if item.get("event") != "planar_face_extraction_finished":
                    continue
                payload = item.get("outer_face_pixel_debug", None)
                if not payload:
                    continue
                save_outer_face_pixel_debug_outputs(cluster_out_dir, payload)
                restored = True
    except OSError:
        return

    if not restored and os.path.isdir(outer_root_dir):
        try:
            os.rmdir(outer_root_dir)
        except OSError:
            pass

def cluster_candidate_failure_reasons(result, sweep_gate_enabled=False, sweep_iou_stop_thresh=0.0):
    reasons = []
    cap_candidate = result.get('cap_candidate', None)
    if cap_candidate is None:
        reasons.append('no_cap')
        return reasons

    sweep_info = result.get('sweep_info') or {}
    if sweep_gate_enabled:
        if not bool(sweep_info.get('valid', False)):
            reasons.append('sweep_invalid')
        elif float(sweep_info.get('iou', 0.0)) < float(sweep_iou_stop_thresh):
            reasons.append(f'sweep_iou_below_{float(sweep_iou_stop_thresh):.4f}')

    connection_info = result.get('connection_info') or {}
    if not bool(connection_info.get('passed', True)):
        reasons.append('side_cap_disconnected')

    if not result.get('selection_passed', False) and not reasons:
        reasons.append('selection_rejected')
    return reasons


def save_single_cluster_candidate_debug_output(
    debug_dir,
    img_shape,
    entry,
    rank,
    infos,
    cap_pool_infos,
    best_cap,
    cap_evaluations,
    endpoint_tol,
    cap_loop_thickness,
    valid_input_enclosed_mask=None,
    sweep_gate_enabled=False,
    sweep_iou_stop_thresh=0.0,
    copy_direction_angle_tol=None,
    copy_iou_compare_percent=None,
    stop_event=None,
):
    if debug_dir is None or cap_search_stop_requested(stop_event):
        return

    cluster_dir = os.path.join(debug_dir, 'clusters')
    os.makedirs(cluster_dir, exist_ok=True)
    base = cluster_debug_base_name(entry, rank)
    out_dir = os.path.join(cluster_dir, base)
    clean_debug_artifact_dir(out_dir, remove_dirs=True)

    details = entry.get('details', {})
    sketch_mask = valid_input_enclosed_mask
    sketch_area = binary_mask_area(sketch_mask) if sketch_mask is not None else 0
    cap_pool_infos = infos if cap_pool_infos is None else cap_pool_infos

    if cap_search_stop_requested(stop_event):
        return
    cv2.imwrite(os.path.join(out_dir, 'cluster.png'), draw_single_cluster_image(img_shape, entry, selected=False, thickness=4))
    restore_geometry_face_debug_outputs_from_steps(out_dir)
    restore_outer_face_pixel_debug_outputs_from_steps(out_dir)
    if cap_search_stop_requested(stop_event):
        return
    cv2.imwrite(os.path.join(out_dir, 'best_cap.png'), draw_cluster_best_cap_image(img_shape, entry, cap_candidate=best_cap, selected=False))
    if cap_search_stop_requested(stop_event):
        return
    cv2.imwrite(os.path.join(out_dir, 'cap_pool.png'), draw_cap_pool_image(img_shape, entry, infos, cap_pool_infos))
    if cap_search_stop_requested(stop_event):
        return
    write_cap_pool_debug_json(os.path.join(out_dir, 'cap_pool.json'), entry, rank, cap_pool_infos)
    if cap_search_stop_requested(stop_event):
        return
    write_component_normalization_debug_json(
        os.path.join(out_dir, 'components_normalized.json'),
        entry,
        rank,
        cap_pool_infos,
        endpoint_tol,
    )
    if cap_search_stop_requested(stop_event):
        return
    save_component_normalization_debug_pngs(
        out_dir,
        img_shape,
        entry,
        cap_pool_infos,
        endpoint_tol,
    )

    best_eval_index = None
    for idx, result in enumerate(cap_evaluations):
        if result.get('cap_candidate') is best_cap and best_cap is not None:
            best_eval_index = idx
            break

    candidate_summaries = []
    for idx, result in enumerate(cap_evaluations):
        if cap_search_stop_requested(stop_event):
            return
        cap_candidate = result.get('cap_candidate')
        sweep_info = result.get('sweep_info') or {}
        connection_info = result.get('connection_info') or {}
        prefix = f'candidate_{idx:02d}'

        bbox_masks = None
        if cap_candidate is not None:
            bbox_masks = save_bbox_masks_geometric(
                out_dir,
                prefix,
                img_shape,
                entry,
                cap_candidate,
                infos,
                endpoint_tol=endpoint_tol,
                cap_loop_thickness=cap_loop_thickness,
                debug_dir=debug_dir,
                copy_direction_angle_tol=copy_direction_angle_tol,
                sketch_mask=sketch_mask,
                iou_compare_percent=copy_iou_compare_percent,
            )

        overlay_geometry = bbox_masks.get('geometry') if bbox_masks is not None else sweep_info.get('geometry', None)
        overlay_img = draw_cluster_side_and_best_cap_overlay(
            img_shape,
            entry,
            cap_candidate=cap_candidate,
            selected=False,
            copy_direction_angle_tol=copy_direction_angle_tol,
            sketch_mask=sketch_mask,
            copy_geometry_override=overlay_geometry,
            iou_compare_percent=copy_iou_compare_percent,
        )
        cv2.imwrite(os.path.join(out_dir, prefix + '_overlay.png'), overlay_img)
        cv2.imwrite(os.path.join(out_dir, prefix + '_cap.png'), draw_cluster_best_cap_image(img_shape, entry, cap_candidate=cap_candidate, selected=False))

        sweep_img = None
        if bbox_masks is None and cap_candidate is not None:
            bbox_vector = estimate_cap_to_far_side_vector(
                entry,
                cap_candidate,
                direction_angle_tol=copy_direction_angle_tol,
                sketch_mask=sketch_mask,
                iou_compare_percent=copy_iou_compare_percent,
            )
            if bbox_vector is not None:
                bbox_masks = save_bbox_masks_from_overlay(out_dir, prefix, overlay_img, bbox_vector)
                sweep_img = draw_bbox_from_bestcap_overlay(overlay_img, bbox_vector)
        elif bbox_masks is not None:
            sweep_img = draw_cap_sweep_visualization(bbox_masks)

        candidate_iou = 0.0
        candidate_intersection = 0
        candidate_union = 0
        candidate_sweep_area = 0
        if bbox_masks is not None and sweep_img is not None:
            cv2.imwrite(os.path.join(out_dir, prefix + '_cap_sweep.png'), sweep_img)
            if sketch_area > 0 and sketch_mask is not None:
                candidate_sweep_area = int(binary_mask_area(bbox_masks['occupied']))
                candidate_iou, candidate_intersection, candidate_union = binary_mask_iou(bbox_masks['occupied'], sketch_mask)

        payload = {
            'candidate_index': int(idx),
            'is_best_candidate': bool(idx == best_eval_index),
            'selection_passed': bool(result.get('selection_passed', False)),
            'failure_reasons': cluster_candidate_failure_reasons(
                result,
                sweep_gate_enabled=sweep_gate_enabled,
                sweep_iou_stop_thresh=sweep_iou_stop_thresh,
            ),
            'cap_candidate': serialize_cap_candidate_for_debug(cap_candidate),
            'sweep_info': json_safe_debug_value(sweep_info),
            'connection_info': json_safe_debug_value(connection_info),
            'saved_files': {
                'cap_png': prefix + '_cap.png',
                'overlay_png': prefix + '_overlay.png',
                'cap_sweep_png': None if sweep_img is None else prefix + '_cap_sweep.png',
                'source_cap_mask_png': None if bbox_masks is None else prefix + '_mask_source_cap.png',
                'copied_cap_mask_png': None if bbox_masks is None else prefix + '_mask_copied_cap.png',
                'side_strokes_mask_png': None if bbox_masks is None else prefix + '_mask_side_strokes.png',
                'swept_cap_mask_png': None if bbox_masks is None else prefix + '_mask_swept_cap.png',
                'occupied_mask_png': None if bbox_masks is None else prefix + '_mask_extrusion_occupied.png',
            },
            'debug_metrics': {
                'occupied_iou_vs_sketch': float(candidate_iou),
                'occupied_intersection': int(candidate_intersection),
                'occupied_union': int(candidate_union),
                'occupied_area': int(candidate_sweep_area),
            },
        }
        with open(os.path.join(out_dir, prefix + '.json'), 'w', encoding='utf-8') as f:
            json.dump(json_safe_debug_value(payload), f, indent=2)
        candidate_summaries.append(payload)

    summary = {
        'rank': int(rank),
        'cluster_id': int(entry.get('cluster_id', -1)),
        'score': float(entry.get('score', 0.0)),
        'source': entry.get('source', 'unknown'),
        'parent_cluster_id': entry.get('parent_cluster_id', None),
        'removal_depth': int(entry.get('removal_depth', 0)),
        'removed_stroke_indices': list(entry.get('removed_stroke_indices', [])),
        'side_indices': [int(i) for i in entry.get('indices', [])],
        'n': int(details.get('n', len(entry.get('strokes', [])))),
        'direction': json_safe_debug_value(entry.get('direction', None)),
        'details': json_safe_debug_value(details),
        'best_cap': serialize_cap_candidate_for_debug(best_cap),
        'best_candidate_index': None if best_eval_index is None else int(best_eval_index),
        'candidate_count': int(len(cap_evaluations)),
        'sweep_gate_enabled': bool(sweep_gate_enabled),
        'sweep_iou_stop_thresh': float(sweep_iou_stop_thresh),
        'sketch_area': int(sketch_area),
        'files': {
            'cluster_png': 'cluster.png',
            'best_cap_png': 'best_cap.png',
            'cap_pool_png': 'cap_pool.png',
            'cap_pool_json': 'cap_pool.json',
            'components_normalized_png': 'components_normalized.png',
            'components_normalized_json': 'components_normalized.json',
        },
        'candidates': candidate_summaries,
    }
    with open(os.path.join(out_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(json_safe_debug_value(summary), f, indent=2)


def save_iou_ranked_side_bestcap_overlays(
    debug_dir,
    records,
    sketch_area,
    iou_output_thresh=0.6,
    search_meta=None,
):
    """Save IoU-ranked side/cap overlays in the legacy format consumed by 3D recovery."""
    ranked_dir = os.path.join(debug_dir, "cluster_side_caps_iou_ranked")
    clean_ranked_output_dir(ranked_dir)

    try:
        threshold = float(iou_output_thresh or 0.0)
    except Exception:
        threshold = 0.0
    threshold = max(0.0, threshold)
    search_meta = search_meta or {}

    if not records:
        summary_path = os.path.join(ranked_dir, "iou_similarity_summary.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("==== Cap Sweep IoU Similarity Ranking ====\n\n")
            f.write("No completed side/cap record was available for IoU ranking output.\n")
            f.write("ranking_mode: none\n")
            f.write(f"iou_rank_output_thresh: {threshold:.6f}\n")
            f.write(f"cap_sweep_iou_stop_thresh: {float(search_meta.get('cap_sweep_iou_stop_thresh', 0.0)):.6f}\n")
            f.write(f"cap_search_time_limit_sec: {float(search_meta.get('cap_search_time_limit_sec', 0.0)):.3f}\n")
            f.write(f"cap_search_elapsed_sec: {float(search_meta.get('cap_search_elapsed_sec', 0.0)):.3f}\n")
            f.write(f"cap_search_time_limit_reached: {bool(search_meta.get('cap_search_time_limit_reached', False))}\n")
            f.write(f"cap_search_stop_reason: {search_meta.get('cap_search_stop_reason', '')}\n")
            f.write(f"sketch_mask_area: {int(sketch_area)}\n")
            f.write("all_iou_records: 0\n")
            f.write("ranked_overlay_records: 0\n")
        return

    def rank_key(record):
        return (
            float(record.get("iou", 0.0)),
            1 if record.get("selected", False) else 0,
            1 if record.get("selection_passed", False) else 0,
            1 if record.get("side_cap_passed", True) else 0,
            -int(record.get("search_rank", 0)),
        )

    threshold_records = [
        r
        for r in records
        if float(r.get("iou", 0.0)) >= threshold
    ]
    ranking_mode = "all_iou_above_threshold"
    sorted_records = sorted(threshold_records, key=rank_key, reverse=True)
    if not sorted_records:
        fallback_record = max(records, key=rank_key, default=None)
        sorted_records = [fallback_record] if fallback_record is not None else []
        ranking_mode = "fallback_best_iou_below_threshold" if sorted_records else "none"

    ranked_record_ids = {id(r) for r in sorted_records}
    below_threshold_records = [
        r
        for r in records
        if id(r) not in ranked_record_ids
    ]
    passed_records = [r for r in records if r.get("selection_passed", False)]
    summary_path = os.path.join(ranked_dir, "iou_similarity_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("==== Cap Sweep IoU Similarity Ranking ====\n\n")
        if ranking_mode == "all_iou_above_threshold":
            f.write("All completed side/cap records whose IoU reaches the output threshold are ranked and written here.\n")
        elif sorted_records:
            f.write(
                "No completed side/cap record reached the output threshold. The best available IoU candidate is written as iou_rank 00 so downstream 3D recovery can consume it unchanged.\n"
            )
        else:
            f.write("No side/cap record was available for IoU ranking output.\n")
        f.write(f"ranking_mode: {ranking_mode}\n")
        f.write(f"iou_rank_output_thresh: {threshold:.6f}\n")
        f.write(f"cap_sweep_iou_stop_thresh: {float(search_meta.get('cap_sweep_iou_stop_thresh', 0.0)):.6f}\n")
        f.write(f"cap_search_time_limit_sec: {float(search_meta.get('cap_search_time_limit_sec', 0.0)):.3f}\n")
        f.write(f"cap_search_elapsed_sec: {float(search_meta.get('cap_search_elapsed_sec', 0.0)):.3f}\n")
        f.write(f"cap_search_time_limit_reached: {bool(search_meta.get('cap_search_time_limit_reached', False))}\n")
        f.write(f"cap_search_stop_reason: {search_meta.get('cap_search_stop_reason', '')}\n")
        f.write(f"sketch_mask_area: {int(sketch_area)}\n")
        f.write(f"all_iou_records: {len(records)}\n")
        f.write(f"ranked_overlay_records: {len(sorted_records)}\n")
        f.write(f"ranked_selection_passed_records: {len(passed_records)}\n")
        f.write(f"ranked_fallback_records: {0 if ranking_mode == 'all_iou_above_threshold' else len(sorted_records)}\n")
        f.write(f"below_threshold_or_unwritten_records: {len(below_threshold_records)}\n")
        f.write("iou = intersection(mask_extrusion_occupied, sketch_enclosed_mask) / union(...)\n\n")

        if not sorted_records:
            f.write("No side/cap overlay records were written.\n")

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
                f"selected={record.get('selected', False)}, "
                f"selection_passed={record.get('selection_passed', False)}, "
                f"selected_by_iou_fallback={record.get('selected_by_iou_fallback', False)}, "
                f"side={record.get('side_indices', [])}, "
                f"side_cap_connected={record.get('side_cap_connected_count', 0)}/{record.get('side_cap_side_count', 0)}, "
                f"side_cap_range_checked={record.get('side_cap_range_checked_count', 0)}, "
                f"side_cap_range_method={record.get('side_cap_range_method', '')}, "
                f"side_cap_passed={record.get('side_cap_passed', True)}, "
                f"side_cap_disconnected={record.get('side_cap_disconnected_strokes', [])}, "
                f"side_cap_ignored={record.get('side_cap_ignored_disconnected_strokes', [])}, "
                f"source={record['base']}\n"
            )

        if below_threshold_records:
            f.write("\nBelow-threshold or otherwise unwritten candidates, sorted by IoU:\n")
            for rejected_rank, record in enumerate(sorted(below_threshold_records, key=rank_key, reverse=True)):
                f.write(
                    f"rejected_iou {rejected_rank:02d}: iou={record['iou']:.6f}, "
                    f"selected={record.get('selected', False)}, "
                    f"selection_passed={record.get('selection_passed', False)}, "
                    f"side={record.get('side_indices', [])}, "
                    f"side_cap_connected={record.get('side_cap_connected_count', 0)}/{record.get('side_cap_side_count', 0)}, "
                    f"side_cap_range_checked={record.get('side_cap_range_checked_count', 0)}, "
                    f"side_cap_range_method={record.get('side_cap_range_method', '')}, "
                    f"side_cap_passed={record.get('side_cap_passed', True)}, "
                    f"side_cap_disconnected={record.get('side_cap_disconnected_strokes', [])}, "
                    f"side_cap_ignored={record.get('side_cap_ignored_disconnected_strokes', [])}, "
                    f"copy_side_stroke={record.get('copy_side_stroke', -1)}, "
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
    clean_ranked_output_dir(out_dir)
    selected_cluster_id = model.get("selected_cluster_id", None)
    successful_parent_ids = successful_direction_parent_ids(model)
    cap_pool_infos = build_cap_pool_infos(
        infos,
        min_arc=CAP_POOL_MIN_ARC,
        endpoint_tol=args.cap_loop_endpoint_tol,
        keep_short_if_connected_gt=CAP_POOL_SHORT_STROKE_KEEP_CONNECTED_GT,
    )
    sketch_mask_path = os.path.join(debug_dir, "00c_input_enclosed_mask.png")
    sketch_mask = cv2.imread(sketch_mask_path, cv2.IMREAD_GRAYSCALE)
    sketch_area = binary_mask_area(sketch_mask) if sketch_mask is not None else 0
    iou_rank_records = []
    iou_output_thresh = float(getattr(args, "iou_rank_output_thresh", 0.0) or 0.0)

    summary_path = os.path.join(out_dir, "cluster_side_cap_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("==== Per-cluster Side/Cap Visualization Summary ====\n\n")
        f.write("Every direction group is tried as side strokes.\n")
        f.write("When a direction group has no valid cap, remove-k-stroke subgroups are tried level by level until success or one stroke remains.\n")
        f.write("Clusters with n<2 are recorded but do not compute cap.\n")
        f.write("Direction-group search paths that produced the selected result or IoU-ranked candidates are emitted here.\n")
        f.write("Each emitted cluster stores the best cap after evaluating sweep IoU and side-cap connection gates.\n\n")

        for rank, entry in enumerate(cluster_debug):
            details = entry.get("details", {})
            if details.get("skip_percluster_output", False):
                continue
            parent_id = entry_original_parent_id(entry)
            details_iou_rank_candidate = (
                bool(details.get("best_sweep_valid", False))
                and float(details.get("best_sweep_iou", 0.0)) >= iou_output_thresh
            )
            selected = entry.get("cluster_id") == selected_cluster_id
            if parent_id not in successful_parent_ids and not details_iou_rank_candidate and not selected:
                continue

            cluster_id = entry.get("cluster_id", -1)
            score_tag = sanitize_score_for_filename(float(entry.get("score", 0.0)))
            base = f"rank_{rank:02d}_cluster_{cluster_id:02d}_score_{score_tag}"

            side_img = draw_single_cluster_image(img_shape, entry, selected=selected, thickness=4)
            cv2.imwrite(os.path.join(out_dir, base + "_side.png"), side_img)

            cap_candidate = None
            skip_reason = ""
            local_selection_passed = bool(details.get("selection_passed", False))
            local_connection_info = None

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
                    min_bbox_area=args.min_cap_bbox_area,
                    min_total_arc=args.min_cap_total_arc,
                    thickness=args.cap_loop_thickness,
                    max_loop_subset_size=args.cap_loop_max_subset_size,
                )
                cap_candidate, _sweep_info, local_connection_info, local_selection_passed, _evaluated = choose_best_cap_candidate_for_selection(
                    entry,
                    candidates,
                    infos,
                    img_shape,
                    args.cap_loop_endpoint_tol,
                    args.cap_loop_thickness,
                    valid_input_enclosed_mask=sketch_mask if sketch_area > 0 else None,
                    sweep_gate_enabled=bool(sketch_area > 0 and float(args.cap_sweep_iou_stop_thresh or 0.0) > 0.0),
                    sweep_iou_stop_thresh=args.cap_sweep_iou_stop_thresh,
                    copy_direction_angle_tol=args.parallel_angle_thresh,
                    copy_iou_compare_percent=args.copy_side_iou_compare_percent,
                    side_cap_connect_tol=args.side_cap_connect_tol,
                )
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
            sweep_iou = 0.0
            sweep_intersection = 0
            sweep_union = 0
            sweep_area = 0
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
                    sweep_iou, sweep_intersection, sweep_union = binary_mask_iou(bbox_masks["occupied"], sketch_mask)
                    iou_rank_records.append({
                        "base": base,
                        "search_rank": int(rank),
                        "cluster_id": int(cluster_id),
                        "overlay_img": overlay_img.copy(),
                        "sweep_area": sweep_area,
                        "sketch_area": sketch_area,
                        "iou": sweep_iou,
                        "intersection": sweep_intersection,
                        "union": sweep_union,
                        "selected": bool(selected),
                        "selection_passed": bool(local_selection_passed),
                        "selected_by_iou_fallback": bool(details.get("selected_by_iou_fallback", False)),
                        "cap_validation_fallback_reason": str(
                            details.get("cap_validation_fallback_reason", "")
                        ),
                        "side_indices": list(entry.get("indices", [])),
                        "side_cap_connected_count": int(
                            (local_connection_info or {}).get("connected_count", details.get("side_cap_connected_count", 0))
                        ),
                        "side_cap_side_count": int(
                            (local_connection_info or {}).get("side_count", details.get("side_cap_side_count", 0))
                        ),
                        "side_cap_range_checked_count": int(
                            (local_connection_info or {}).get(
                                "range_checked_count",
                                details.get("side_cap_range_checked_count", 0),
                            )
                        ),
                        "side_cap_range_method": str(
                            (local_connection_info or {}).get(
                                "cap_range_method",
                                details.get("side_cap_range_method", ""),
                            )
                        ),
                        "side_cap_passed": bool(
                            (local_connection_info or {}).get("passed", details.get("side_cap_connect_passed", True))
                        ),
                        "side_cap_disconnected_strokes": list(
                            (local_connection_info or {}).get(
                                "disconnected_strokes",
                                details.get("side_cap_disconnected_strokes", []),
                            )
                        ),
                        "side_cap_ignored_disconnected_strokes": list(
                            (local_connection_info or {}).get(
                                "ignored_disconnected_strokes",
                                details.get("side_cap_ignored_disconnected_strokes", []),
                            )
                        ),
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
                    f"selection_passed={local_selection_passed} "
                    f"side={entry.get('indices', [])} best_cap_strokes={cap_candidate.get('stroke_indices', [])} "
                    f"best_cap_area={cap_candidate.get('area', 0)} best_cap_enclosed_area={cap_candidate.get('enclosed_area', 0)} "
                f"best_cap_bbox_area={cap_candidate.get('bbox_area', 0)} best_cap_bbox={cap_candidate.get('bbox', None)} "
                    f"best_sweep_iou={float(sweep_iou):.6f} best_sweep_intersection={int(sweep_intersection)} best_sweep_union={int(sweep_union)} best_sweep_area={int(sweep_area)} "
                    f"side_cap_connected={(local_connection_info or {}).get('connected_count', details.get('side_cap_connected_count', 0))}/{(local_connection_info or {}).get('side_count', details.get('side_cap_side_count', 0))} "
                    f"side_cap_range_checked={(local_connection_info or {}).get('range_checked_count', details.get('side_cap_range_checked_count', 0))} "
                    f"side_cap_range_method={(local_connection_info or {}).get('cap_range_method', details.get('side_cap_range_method', ''))} "
                    f"side_cap_passed={(local_connection_info or {}).get('passed', details.get('side_cap_connect_passed', True))} "
                    f"side_cap_disconnected={(local_connection_info or {}).get('disconnected_strokes', details.get('side_cap_disconnected_strokes', []))} "
                    f"side_cap_ignored={(local_connection_info or {}).get('ignored_disconnected_strokes', details.get('side_cap_ignored_disconnected_strokes', []))} "
                    f"best_cap_total_arc={cap_candidate.get('total_arc', 0.0):.1f} "
                    f"best_cap_score={cap_candidate.get('score', 0.0):.1f} "
                    f"best_cap_topology={cap_candidate.get('topology_kind', '')} "
                    f"loop_detection={cap_candidate.get('loop_detection', '')} "
                    f"removed_post_loop_self_strokes={cap_candidate.get('removed_post_loop_self_strokes', [])} "
                    f"{copy_side_txt} "
                    f"source={entry.get('source', 'unknown')} parent={entry.get('parent_cluster_id', None)} "
                    f"removed={entry.get('removed_stroke_indices', [])} depth={entry.get('removal_depth', 0)}\n"
                )
            else:
                f.write(
                    f"rank {rank:02d} cluster {cluster_id:02d} selected={selected} "
                    f"selection_passed={local_selection_passed} "
                    f"side={entry.get('indices', [])} cap=NONE reason={skip_reason} "
                    f"source={entry.get('source', 'unknown')} parent={entry.get('parent_cluster_id', None)} "
                    f"removed={entry.get('removed_stroke_indices', [])} depth={entry.get('removal_depth', 0)}\n"
                )

    save_iou_ranked_side_bestcap_overlays(
        debug_dir,
        iou_rank_records,
        sketch_area,
        iou_output_thresh=iou_output_thresh,
        search_meta={
            "cap_sweep_iou_stop_thresh": float(getattr(args, "cap_sweep_iou_stop_thresh", 0.0) or 0.0),
            "cap_search_time_limit_sec": float(model.get("cap_search_time_limit_sec", 0.0) or 0.0),
            "cap_search_elapsed_sec": float(model.get("cap_search_elapsed_sec", 0.0) or 0.0),
            "cap_search_time_limit_reached": bool(model.get("cap_search_time_limit_reached", False)),
            "cap_search_stop_reason": model.get("cap_search_stop_reason", ""),
        },
    )


def save_debug_report(path, args, raw_strokes, merged_strokes, infos, line_strokes, model, candidates):
    side_prefilter_strokes = filter_side_direction_group_strokes_for_args(infos, args)
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
        f.write(f"  side prefilter candidates: {len(side_prefilter_strokes)}\n")
        f.write(f"  selected side strokes: {len(model['inliers'])}\n")
        f.write(f"  cap candidates: {len(candidates)}\n")
        f.write("\nMerged stroke infos:\n")
        for s in infos:
            c = s["center"]
            dbg = stroke_direction_debug_values(s)
            f.write(f"  stroke {s['index']:03d}: arc={s['arc']:.1f}, chord={s['chord']:.1f}, straightness={s['straightness']:.3f}, p90_pca_line_error={s.get('p90_pca_line_error', 0.0):.2f}, pca_rms_error={s.get('pca_rms_error', 0.0):.2f}, chord_dev_ratio={s.get('chord_deviation_ratio', 0.0):.4f}, center=({c[0]:.1f},{c[1]:.1f}), pca_axis=({dbg['pca_axis'][0]:.4f},{dbg['pca_axis'][1]:.4f}), pca_angle={dbg['pca_angle']:.2f}\n")
        f.write("\nLine stroke candidates:\n")
        for s in line_strokes:
            c = s["center"]
            dbg = stroke_direction_debug_values(s)
            f.write(f"  stroke {s['index']:03d}: arc={s['arc']:.1f}, straightness={s['straightness']:.3f}, center=({c[0]:.1f},{c[1]:.1f}), pca_axis=({dbg['pca_axis'][0]:.4f},{dbg['pca_axis'][1]:.4f}), pca_angle={dbg['pca_angle']:.2f}\n")
        f.write("\nSide prefilter candidates used for PCA direction clustering:\n")
        for s in side_prefilter_strokes:
            dbg = stroke_direction_debug_values(s)
            f.write(f"  stroke {s['index']:03d}: arc={s['arc']:.1f}, chord={s['chord']:.1f}, straightness={s['straightness']:.3f}, p90_pca_line_error={s.get('p90_pca_line_error', 0.0):.2f}, pca_rms_error={s.get('pca_rms_error', 0.0):.2f}, chord_dev_ratio={s.get('chord_deviation_ratio', 0.0):.4f}, pca_angle={dbg['pca_angle']:.2f}\n")
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


def compute_input_enclosed_mask_debug_data(bw, infos, endpoint_tol=50.0):
    """Return the input enclosed mask and debug metadata used by sweep IoU checks."""
    enclosed_mask, closed_strokes, close_info = make_input_enclosed_region_mask(
        bw,
        stroke_infos=infos,
        endpoint_tol=endpoint_tol,
        return_debug=True,
    )
    return enclosed_mask, closed_strokes, close_info


def save_input_enclosed_mask_debug_outputs(debug_dir, enclosed_mask, closed_strokes, close_info):
    """Save precomputed input enclosed mask debug outputs."""
    if debug_dir is None:
        return
    ensure_dir(debug_dir)
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

    direction_group_strokes = filter_side_direction_group_strokes_for_args(infos, args)
    write_side_direction_prefilter_report(
        os.path.join(debug_dir, "05c_side_direction_prefilter.txt"),
        infos,
        args,
    )
    cv2.imwrite(
        os.path.join(debug_dir, "05c_side_prefilter_candidates.png"),
        draw_side_prefilter_candidates_image(img_shape, infos, direction_group_strokes, args=args, thickness=4),
    )
    write_direction_groups_debug_report(
        os.path.join(debug_dir, "05c_all_stroke_direction_groups.txt"),
        direction_group_strokes,
        angle_thresh=args.parallel_angle_thresh,
        min_stroke_length=args.min_stroke_length,
        min_straightness=args.side_straightness,
        args=args,
    )
    cv2.imwrite(
        os.path.join(debug_dir, "05c_all_stroke_direction_groups.png"),
        draw_direction_groups_image(
            img_shape,
            direction_group_strokes,
            angle_thresh=args.parallel_angle_thresh,
            min_stroke_length=args.min_stroke_length,
            min_straightness=args.side_straightness,
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
    direction_group_strokes = filter_side_direction_group_strokes_for_args(infos, args)
    write_side_direction_prefilter_report(
        os.path.join(debug_dir, "05c_side_direction_prefilter.txt"),
        infos,
        args,
    )
    cv2.imwrite(
        os.path.join(debug_dir, "05c_side_prefilter_candidates.png"),
        draw_side_prefilter_candidates_image(img.shape, infos, direction_group_strokes, args=args, thickness=4),
    )
    write_direction_groups_debug_report(
        os.path.join(debug_dir, "05c_all_stroke_direction_groups.txt"),
        direction_group_strokes,
        angle_thresh=args.parallel_angle_thresh,
        min_stroke_length=args.min_stroke_length,
        min_straightness=args.side_straightness,
        args=args,
    )
    cv2.imwrite(
        os.path.join(debug_dir, "05c_all_stroke_direction_groups.png"),
        draw_direction_groups_image(
            img.shape,
            direction_group_strokes,
            angle_thresh=args.parallel_angle_thresh,
            min_stroke_length=args.min_stroke_length,
            min_straightness=args.side_straightness,
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


def iou_rank00_output_exists(debug_dir):
    if not debug_dir:
        return False
    ranked_dir = os.path.join(debug_dir, "cluster_side_caps_iou_ranked")
    if not os.path.isdir(ranked_dir):
        return False

    try:
        for name in os.listdir(ranked_dir):
            lower_name = name.lower()
            if lower_name.startswith("iou_rank_00_") and lower_name.endswith("_side_bestcap_overlay.png"):
                return True
    except OSError:
        return False

    summary_path = os.path.join(ranked_dir, "iou_similarity_summary.txt")
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("iou_rank 00:"):
                    return True
    except OSError:
        return False
    return False



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
    parser.add_argument(
        "--side-straightness",
        type=float,
        default=0.85,
        help="Hard prefilter for side-stroke clustering. Only strokes with straightness >= this value enter PCA direction grouping for side selection.",
    )
    parser.add_argument(
        "--side-min-chord-px",
        type=float,
        default=25.0,
        help="Minimum endpoint chord length for a stroke to enter side PCA direction clustering.",
    )
    parser.add_argument(
        "--side-line-p90-error-px",
        type=float,
        default=4.0,
        help="Absolute p90 distance-to-PCA-line limit for side-stroke clustering. Used with --side-line-p90-error-ratio; <=0 disables the absolute part.",
    )
    parser.add_argument(
        "--side-line-p90-error-ratio",
        type=float,
        default=0.035,
        help="Relative p90 distance-to-PCA-line limit, multiplied by chord length. Used with --side-line-p90-error-px; <=0 disables the ratio part.",
    )
    parser.add_argument(
        "--side-line-rms-error-px",
        type=float,
        default=2.5,
        help="Absolute RMS distance-to-PCA-line limit for side-stroke clustering. Used with --side-line-rms-error-ratio; <=0 disables the absolute part.",
    )
    parser.add_argument(
        "--side-line-rms-error-ratio",
        type=float,
        default=0.025,
        help="Relative RMS distance-to-PCA-line limit, multiplied by chord length. Used with --side-line-rms-error-px; <=0 disables the ratio part.",
    )
    parser.add_argument(
        "--side-chord-dev-ratio-max",
        type=float,
        default=0.08,
        help="Maximum p90 endpoint-chord deviation divided by chord length for side-stroke clustering. <=0 disables this filter.",
    )
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

    # Cluster-based side selection controls.
    # Active score uses only count_weight and same_loop_penalty.
    # Other weight arguments are kept for CLI compatibility and debug experiments, but are ignored.
    parser.add_argument("--cluster-min-perp-spread", type=float, default=10.0)
    parser.add_argument("--same-loop-endpoint-tol", type=float, default=12.0)
    parser.add_argument("--cluster-count-weight", type=float, default=10000)
    parser.add_argument("--cluster-length-weight", type=float, default=0.7)
    parser.add_argument("--cluster-straightness-weight", type=float, default=10000)
    parser.add_argument("--cluster-spread-weight", type=float, default=1.5)
    parser.add_argument("--cluster-same-loop-penalty", type=float, default=10000)
    parser.add_argument("--cluster-low-spread-penalty", type=float, default=250.0)
    parser.add_argument("--cluster-length-similarity-weight", type=float, default=10000.0,
                        help="Legacy argument retained for compatibility; currently ignored by side-cluster scoring.")

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
    parser.add_argument("--min-cap-bbox-area", type=int, default=0,
                        help="Reject cap candidates whose cap bbox area is smaller than this value. The bbox is computed on loop pixels union enclosed cap pixels.")
    parser.add_argument("--min-cap-total-arc", type=float, default=0.0,
                        help="Reject cap candidates whose total stroke arc length is smaller than this value.")
    parser.add_argument("--cap-sweep-iou-stop-thresh", type=float, default=0.0,
                        help="Only stop cap search when a removal-depth round contains at least one cap whose swept extrusion occupancy IoU against the input enclosed mask is >= this threshold. 0 keeps the old cap-only stopping behavior.")
    parser.add_argument("--iou-rank-output-thresh", type=float, default=0.6,
                        help="Write every completed side/cap result with swept extrusion IoU >= this threshold to cluster_side_caps_iou_ranked. If none reach it, the best available IoU result is still written as rank 00.")
    parser.add_argument("--cap-search-time-limit-sec", type=float, default=0.0,
                        help="Soft wall-clock limit for cap subgroup search. 0 disables the limit; completed candidates are still ranked when the limit is reached.")
    parser.add_argument("--side-cap-connect-tol", type=float, default=20.0,
                        help="Require every final side stroke to have at least one endpoint within this pixel distance of the selected cap. <=0 disables this gate.")
    parser.add_argument("--cap-loop-endpoint-tol", type=float, default=12.0,
                        help="Endpoint tolerance for deciding whether remaining strokes form a closed cap loop.")
    parser.add_argument("--cap-loop-thickness", type=int, default=2,
                        help="Raster thickness used to draw loop-based cap candidate masks.")
    parser.add_argument("--cap-loop-max-subset-size", type=int, default=14,
                        help="Search guard for endpoint-port cycle enumeration. The implementation allows longer visible loops when a component is split into many short traced strokes.")
    parser.add_argument("--cap-subgroup-max-removals", type=int, default=-1,
                        help="Max remove-k subgroup depth for direction groups that have no valid cap. Use -1 to continue until one stroke remains.")
    parser.add_argument("--cap-round-workers", "--cap-round-threads", type=int, default=1,
                        help="Number of CPU workers used to evaluate cap-search clusters/subgroups. 1 keeps serial behavior.")
    parser.add_argument("--cap-worker-backend", choices=("thread", "process"), default="thread",
                        help="Parallel backend for --cap-round-workers > 1. process uses separate Python processes for true multi-core CPU execution.")

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
    input_enclosed_mask, input_closed_strokes, input_enclosed_info = compute_input_enclosed_mask_debug_data(
        bw,
        infos,
        endpoint_tol=args.cap_loop_endpoint_tol,
    )
    save_input_enclosed_mask_debug_outputs(
        args.debug_dir,
        input_enclosed_mask,
        input_closed_strokes,
        input_enclosed_info,
    )
    step_line_strokes = [
        s for s in infos
        if s["arc"] >= args.min_stroke_length and s["straightness"] >= args.straightness
    ]
    save_stroke_info_debug_outputs(args.debug_dir, args, img.shape, infos, step_line_strokes)
    model, line_strokes = choose_extrusion_model(args, infos, img.shape, skel)
    save_stroke_info_debug_outputs(args.debug_dir, args, img.shape, infos, line_strokes)

    if model is None:
        raise RuntimeError(
            "Could not estimate extrusion direction. Try --force-parallel, lower --side-straightness, lower --straightness, "
            "lower --min-stroke-length, increase --merge-gap, or provide --vp manually."
        )
    if len(model["inliers"]) == 0:
        raise RuntimeError("Extrusion direction model has no side stroke inliers.")

    # Cap-validated side cluster selection:
    # compute the largest-area cap candidate for every side cluster with n>=2;
    # select the first ranked selectable side cluster that has a legal cap.
    cap_pool_infos = build_cap_pool_infos(
        infos,
        min_arc=CAP_POOL_MIN_ARC,
        endpoint_tol=args.cap_loop_endpoint_tol,
        keep_short_if_connected_gt=CAP_POOL_SHORT_STROKE_KEEP_CONNECTED_GT,
    )
    model, candidates = validate_side_clusters_by_cap_candidates(
        model,
        infos,
        skel.shape,
        endpoint_tol=args.cap_loop_endpoint_tol,
        min_pixels=args.min_cap_pixels,
        min_enclosed_area=args.min_cap_enclosed_area,
        min_bbox_area=args.min_cap_bbox_area,
        min_total_arc=args.min_cap_total_arc,
        thickness=args.cap_loop_thickness,
        max_loop_subset_size=args.cap_loop_max_subset_size,
        max_subgroup_removals=args.cap_subgroup_max_removals,
        cap_pool_infos=cap_pool_infos,
        input_enclosed_mask=input_enclosed_mask,
        sweep_iou_stop_thresh=args.cap_sweep_iou_stop_thresh,
        copy_direction_angle_tol=args.parallel_angle_thresh,
        copy_iou_compare_percent=args.copy_side_iou_compare_percent,
        side_cap_connect_tol=args.side_cap_connect_tol,
        progress_debug_dir=args.debug_dir,
        cap_round_workers=args.cap_round_workers,
        cap_worker_backend=args.cap_worker_backend,
        cap_search_time_limit_sec=args.cap_search_time_limit_sec,
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
        if iou_rank00_output_exists(args.debug_dir):
            print(
                "WARNING: no cap sweep reached --cap-sweep-iou-stop-thresh; "
                "continuing with cluster_side_caps_iou_ranked/iou_rank 00 as the downstream 3D fallback."
            )
        elif float(args.cap_sweep_iou_stop_thresh or 0.0) > 0.0:
            timeout_text = ""
            if model.get("cap_search_time_limit_reached", False):
                timeout_text = (
                    f" Search stopped after the time limit "
                    f"({float(model.get('cap_search_elapsed_sec', 0.0)):.2f}s / "
                    f"{float(model.get('cap_search_time_limit_sec', 0.0)):.2f}s)."
                )
            raise RuntimeError(
                "No side cluster produced a cap sweep whose extrusion occupancy IoU passed --cap-sweep-iou-stop-thresh. "
                "No iou_rank 00 fallback was available for downstream 3D recovery."
                + timeout_text
                + " "
                "Debug outputs were saved; check 05d_cap_search_trace.txt and 05e_cap_search_rounds.txt for details."
            )
        else:
            raise RuntimeError(
                "No side cluster produced a legal closed-loop cap candidate. "
                "No iou_rank 00 fallback was available for downstream 3D recovery. "
                "Debug outputs were saved; check 05b_direction_cluster_scores.txt for cap_validation details."
            )

    draw_result(img, skel, model, side_mask, candidates, args.output)
    print_debug(raw_strokes, merged_strokes, infos, line_strokes, model, candidates, args.output, args.debug_dir)


if __name__ == "__main__":
    main()

