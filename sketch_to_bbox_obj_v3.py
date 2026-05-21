#!/usr/bin/env python3
import argparse
import json
import math
import os
from dataclasses import dataclass
from itertools import product
from typing import List, Tuple, Optional, Dict, Any

import cv2
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class Segment:
    p1: np.ndarray
    p2: np.ndarray
    length: float
    theta: float
    cluster: int = -1


@dataclass
class RootEdge:
    axis: int
    root: np.ndarray
    far: np.ndarray
    score: float
    sign: int

    @property
    def vec(self) -> np.ndarray:
        return self.far - self.root


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_image(path: str, img: np.ndarray, title: Optional[str] = None):
    ensure_dir(os.path.dirname(path))
    plt.figure(figsize=(7, 7))
    if img.ndim == 2:
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if title:
        plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches='tight')
    plt.close()


def angle_mod_pi(theta: float) -> float:
    return theta % math.pi


def ang_diff_mod_pi(a: float, b: float) -> float:
    d = abs(a - b) % math.pi
    return min(d, math.pi - d)


def segment_intersection(a1, a2, b1, b2) -> Optional[np.ndarray]:
    x1, y1 = a1
    x2, y2 = a2
    x3, y3 = b1
    x4, y4 = b2
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-8:
        return None
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
    return np.array([px, py], dtype=np.float64)


def point_to_segment_distance(p, a, b):
    ab = b - a
    if np.linalg.norm(ab) < 1e-8:
        return float(np.linalg.norm(p - a))
    t = np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0.0, 1.0)
    proj = a + t * ab
    return float(np.linalg.norm(p - proj))


def local_density(binary: np.ndarray, p: np.ndarray, radius=8) -> float:
    h, w = binary.shape
    x, y = int(round(p[0])), int(round(p[1]))
    x0, x1 = max(0, x - radius), min(w, x + radius + 1)
    y0, y1 = max(0, y - radius), min(h, y + radius + 1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return float(binary[y0:y1, x0:x1].mean() / 255.0)


def line_support(binary: np.ndarray, a: np.ndarray, b: np.ndarray, samples=80, radius=3) -> float:
    h, w = binary.shape
    vals = []
    for t in np.linspace(0.0, 1.0, samples):
        p = a * (1 - t) + b * t
        x = int(round(p[0]))
        y = int(round(p[1]))
        x0, x1 = max(0, x - radius), min(w, x + radius + 1)
        y0, y1 = max(0, y - radius), min(h, y + radius + 1)
        vals.append(binary[y0:y1, x0:x1].mean() / 255.0 if x1 > x0 and y1 > y0 else 0.0)
    return float(np.mean(vals))


# ---------- Step 1 ----------

def extract_foreground(rgb: np.ndarray, debug_dir: str) -> np.ndarray:
    border = np.concatenate([
        rgb[:8, :, :].reshape(-1, 3),
        rgb[-8:, :, :].reshape(-1, 3),
        rgb[:, :8, :].reshape(-1, 3),
        rgb[:, -8:, :].reshape(-1, 3),
    ], axis=0)
    bg = np.median(border, axis=0)
    diff = np.linalg.norm(rgb.astype(np.float32) - bg.astype(np.float32), axis=2)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    fg = ((gray < np.percentile(gray, 70) - 10) | (diff > 25)).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel)
    fg = cv2.dilate(fg, kernel, iterations=1)
    save_image(os.path.join(debug_dir, '01_foreground_mask.png'), fg, 'Foreground mask')
    return fg


# ---------- Step 2 ----------

def detect_segments(binary: np.ndarray, debug_dir: str, min_len=24) -> List[Segment]:
    lines = cv2.HoughLinesP(binary, 1, np.pi / 180.0, threshold=22, minLineLength=min_len, maxLineGap=12)
    segs: List[Segment] = []
    vis = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    if lines is not None:
        for l in lines[:, 0, :]:
            x1, y1, x2, y2 = map(float, l)
            p1 = np.array([x1, y1], dtype=np.float64)
            p2 = np.array([x2, y2], dtype=np.float64)
            length = float(np.linalg.norm(p2 - p1))
            if length < min_len:
                continue
            theta = angle_mod_pi(math.atan2(y2 - y1, x2 - x1))
            segs.append(Segment(p1, p2, length, theta))
            cv2.line(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
    save_image(os.path.join(debug_dir, '02_hough_segments.png'), vis, f'Hough segments: {len(segs)}')
    return segs


# ---------- Step 3 ----------

def weighted_kmeans_dir(segments: List[Segment], k=3, iters=30):
    X = np.array([[math.cos(2 * s.theta), math.sin(2 * s.theta)] for s in segments], dtype=np.float64)
    W = np.array([s.length for s in segments], dtype=np.float64)

    bins = 180
    hist = np.zeros(bins, dtype=np.float64)
    for s in segments:
        idx = int((s.theta / math.pi) * bins) % bins
        hist[idx] += s.length

    peak_ids = np.argsort(hist)[-k:]
    centers = []
    for pid in peak_ids:
        theta = (pid + 0.5) / bins * math.pi
        centers.append([math.cos(2 * theta), math.sin(2 * theta)])
    centers = np.array(centers, dtype=np.float64)

    for _ in range(iters):
        dists = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        labels = np.argmin(dists, axis=1)
        new_centers = []
        for j in range(k):
            mask = labels == j
            if not np.any(mask):
                new_centers.append(centers[j])
                continue
            c = np.average(X[mask], axis=0, weights=W[mask])
            n = np.linalg.norm(c)
            c = c / n if n > 1e-8 else centers[j]
            new_centers.append(c)
        new_centers = np.array(new_centers)
        if np.allclose(new_centers, centers, atol=1e-6):
            centers = new_centers
            break
        centers = new_centers

    thetas = np.array([0.5 * math.atan2(c[1], c[0]) % math.pi for c in centers])
    order = np.argsort(thetas)
    centers = centers[order]
    thetas = thetas[order]
    dists = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    labels = np.argmin(dists, axis=1)
    for s, lbl in zip(segments, labels):
        s.cluster = int(lbl)
    return thetas.tolist(), labels.tolist()


def visualize_clusters(rgb: np.ndarray, segments: List[Segment], thetas: List[float], debug_dir: str):
    colors = [(255, 140, 80), (80, 220, 80), (80, 100, 255)]
    vis = rgb.copy()
    for s in segments:
        c = colors[s.cluster % len(colors)]
        cv2.line(vis, tuple(np.round(s.p1).astype(int)), tuple(np.round(s.p2).astype(int)), c, 3)
    save_image(os.path.join(debug_dir, '03_segment_clusters.png'), vis, 'Direction clusters')

    plt.figure(figsize=(7, 3))
    bins = np.linspace(0, 180, 181)
    angles_deg = [math.degrees(s.theta) for s in segments]
    weights = [s.length for s in segments]
    plt.hist(angles_deg, bins=bins, weights=weights, color='gray')
    for t in thetas:
        plt.axvline(math.degrees(t), color='r', linestyle='--')
    plt.xlabel('Angle (deg, modulo 180)')
    plt.ylabel('Weighted count')
    plt.title('Direction histogram + cluster centers')
    plt.tight_layout()
    plt.savefig(os.path.join(debug_dir, '04_direction_histogram.png'), dpi=180)
    plt.close()


# ---------- Root candidates ----------

def endpoint_root_candidates(binary: np.ndarray, segments: List[Segment]) -> List[Tuple[float, np.ndarray, int]]:
    pts = []
    for s in segments:
        pts.extend([s.p1, s.p2])
    if not pts:
        return []
    pts = np.array(pts, dtype=np.float64)
    used = np.zeros(len(pts), dtype=bool)
    cands = []
    for i in range(len(pts)):
        if used[i]:
            continue
        cluster_pts = [pts[i]]
        used[i] = True
        for j in range(i + 1, len(pts)):
            if used[j]:
                continue
            if np.linalg.norm(pts[j] - pts[i]) <= 10:
                cluster_pts.append(pts[j])
                used[j] = True
        p = np.mean(cluster_pts, axis=0)
        density = local_density(binary, p, radius=8)
        cl_present = set()
        support = 0.0
        endpoint_hits = 0
        for s in segments:
            d1 = np.linalg.norm(s.p1 - p)
            d2 = np.linalg.norm(s.p2 - p)
            if min(d1, d2) < 12:
                endpoint_hits += 1
                cl_present.add(s.cluster)
                support += s.length
        score = 130.0 * endpoint_hits + 70.0 * len(cl_present) + 0.15 * support + 40.0 * density
        cands.append((float(score), p, len(cl_present)))
    return cands


def line_intersection_candidates(binary: np.ndarray, segments: List[Segment]) -> List[Tuple[float, np.ndarray, int]]:
    candidates = []
    n = len(segments)
    for i in range(n):
        for j in range(i + 1, n):
            if segments[i].cluster == segments[j].cluster:
                continue
            p = segment_intersection(segments[i].p1, segments[i].p2, segments[j].p1, segments[j].p2)
            if p is None:
                continue
            d1 = point_to_segment_distance(p, segments[i].p1, segments[i].p2)
            d2 = point_to_segment_distance(p, segments[j].p1, segments[j].p2)
            if d1 > 14 or d2 > 14:
                continue
            density = local_density(binary, p, radius=10)
            support = 0.0
            cl_present = set()
            endpoint_bonus = 0
            for s in segments:
                if point_to_segment_distance(p, s.p1, s.p2) < 10:
                    support += s.length
                    cl_present.add(s.cluster)
                if min(np.linalg.norm(s.p1 - p), np.linalg.norm(s.p2 - p)) < 10:
                    endpoint_bonus += 1
            score = density * 40 + 0.6 * support + 75 * len(cl_present) + 40 * endpoint_bonus
            candidates.append((float(score), p, len(cl_present)))
    return candidates


def root_candidates(binary: np.ndarray, segments: List[Segment], max_keep: int = 20) -> List[Tuple[float, np.ndarray, int]]:
    candidates = endpoint_root_candidates(binary, segments) + line_intersection_candidates(binary, segments)
    if not candidates:
        ys, xs = np.where(binary > 0)
        if len(xs) == 0:
            return [(0.0, np.array([binary.shape[1] / 2.0, binary.shape[0] / 2.0]), 0)]
        return [(0.0, np.array([xs.mean(), ys.mean()], dtype=np.float64), 0)]

    candidates.sort(key=lambda x: x[0], reverse=True)
    kept: List[Tuple[float, np.ndarray, int]] = []
    for score, p, k in candidates:
        if all(np.linalg.norm(p - q) > 10 for _, q, _ in kept):
            kept.append((score, p, k))
        if len(kept) >= max_keep:
            break
    return kept


def visualize_root_candidates(binary: np.ndarray, candidates: List[Tuple[float, np.ndarray, int]], root: np.ndarray, debug_dir: str):
    vis = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    for idx, (score, p, k) in enumerate(candidates):
        col = (0, 200, 255) if k >= 3 else (255, 120, 0)
        cv2.circle(vis, tuple(np.round(p).astype(int)), 4, col, -1)
        cv2.putText(vis, str(idx), tuple(np.round(p + np.array([5, -5])).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.circle(vis, tuple(np.round(root).astype(int)), 9, (255, 0, 255), -1)
    save_image(os.path.join(debug_dir, '05_root_candidates.png'), vis, 'Root corner candidates')


# ---------- Axis tracing ----------

def smooth1d(vals: np.ndarray, k=5) -> np.ndarray:
    if len(vals) == 0 or k <= 1:
        return vals
    ker = np.ones(k, dtype=np.float64) / k
    return np.convolve(vals, ker, mode='same')


def trace_axis_extent(binary: np.ndarray, root: np.ndarray, direction: np.ndarray,
                      max_len: float, step: float = 2.0, radius: int = 3,
                      thr: float = 0.09, min_run: int = 4) -> Tuple[np.ndarray, float]:
    h, w = binary.shape
    pts = []
    vals = []
    direction = direction / max(np.linalg.norm(direction), 1e-8)
    for t in np.arange(0.0, max_len, step):
        p = root + direction * t
        if p[0] < 0 or p[0] >= w or p[1] < 0 or p[1] >= h:
            break
        pts.append(p.copy())
        vals.append(local_density(binary, p, radius=radius))
    if len(pts) < 4:
        return root.copy(), 0.0

    vals = smooth1d(np.array(vals, dtype=np.float64), k=5)
    best_end = 0
    best_score = 0.0
    run_start = None
    run_sum = 0.0
    for i, v in enumerate(vals):
        if v >= thr:
            if run_start is None:
                run_start = i
                run_sum = 0.0
            run_sum += v
        else:
            if run_start is not None:
                run_len = i - run_start
                if run_start <= 6 and run_len >= min_run:
                    score = run_sum + 0.03 * i
                    if score > best_score:
                        best_score = score
                        best_end = i - 1
                run_start = None
                run_sum = 0.0
    if run_start is not None:
        run_len = len(vals) - run_start
        if run_start <= 6 and run_len >= min_run:
            score = run_sum + 0.03 * (len(vals) - 1)
            if score > best_score:
                best_score = score
                best_end = len(vals) - 1

    if best_score <= 0:
        idx = int(np.argmax(vals[3:])) + 3 if len(vals) > 3 else int(np.argmax(vals))
        if vals[idx] < thr * 0.6:
            return root.copy(), 0.0
        best_end = idx
        best_score = float(vals[idx])
    return pts[best_end], float(best_score)


def axis_projection_upper_bound(root: np.ndarray, segments: List[Segment], axis: int, direction: np.ndarray,
                                perp_thresh: float = 18.0, margin: float = 20.0) -> float:
    direction = direction / max(np.linalg.norm(direction), 1e-8)
    ts = []
    for s in segments:
        if s.cluster != axis:
            continue
        for p in (s.p1, s.p2):
            rel = p - root
            t = float(np.dot(rel, direction))
            perp = float(abs(np.cross(direction, rel)))
            if t > 0 and perp < perp_thresh:
                ts.append(t)
    if not ts:
        return 220.0
    return max(30.0, max(ts) + margin)


def trace_both_signs(binary: np.ndarray, root: np.ndarray, theta: float, axis: int, segments: List[Segment]) -> List[RootEdge]:
    d = np.array([math.cos(theta), math.sin(theta)], dtype=np.float64)
    results = []
    for sign in (+1, -1):
        max_len = axis_projection_upper_bound(root, segments, axis, d * sign)
        far, score = trace_axis_extent(binary, root, d * sign, max_len=max_len)
        length = np.linalg.norm(far - root)
        if length >= 15:
            score2 = score * 100.0 + 0.55 * length
            results.append(RootEdge(axis=axis, root=root.copy(), far=far, score=float(score2), sign=sign))
    return results


# ---------- Cuboid build / score ----------

def order_axes(thetas: List[float]) -> Tuple[int, int, int]:
    vertical_ref = math.pi / 2
    y_idx = int(np.argmin([ang_diff_mod_pi(t, vertical_ref) for t in thetas]))
    rest = [i for i in range(3) if i != y_idx]

    def slope_mag(i):
        c = abs(math.cos(thetas[i]))
        return abs(math.tan(thetas[i])) if c > 1e-6 else 1e9

    rest = sorted(rest, key=lambda i: slope_mag(i))
    x_idx, z_idx = rest[0], rest[1]
    return x_idx, y_idx, z_idx


def build_projected_cuboid(root: np.ndarray, vecs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    vx, vy, vz = vecs['X'], vecs['Y'], vecs['Z']
    return {
        '000': root,
        '100': root + vx,
        '010': root + vy,
        '001': root + vz,
        '110': root + vx + vy,
        '101': root + vx + vz,
        '011': root + vy + vz,
        '111': root + vx + vy + vz,
    }


def projected_cuboid_edges(P: dict):
    return [
        ('000', '100'), ('000', '010'), ('000', '001'),
        ('100', '110'), ('100', '101'),
        ('010', '110'), ('010', '011'),
        ('001', '101'), ('001', '011'),
        ('110', '111'), ('101', '111'), ('011', '111')
    ]


def score_projected_cuboid(binary: np.ndarray, P: dict) -> float:
    supports, lengths = [], []
    h, w = binary.shape
    inside_count = 0
    for a, b in projected_cuboid_edges(P):
        pa = P[a]
        pb = P[b]
        L = np.linalg.norm(pb - pa)
        lengths.append(L)
        if (0 <= pa[0] < w and 0 <= pa[1] < h and 0 <= pb[0] < w and 0 <= pb[1] < h):
            inside_count += 1
        if L < 12:
            supports.append(0.0)
        else:
            supports.append(line_support(binary, pa, pb, samples=max(25, int(L / 3)), radius=3))
    supports_sorted = sorted(supports, reverse=True)
    score = 110.0 * sum(supports_sorted[:8]) + 10.0 * sum(supports_sorted[8:])
    score += 0.25 * sum(sorted(lengths, reverse=True)[:3])
    score += 8.0 * inside_count
    if min(lengths[:3]) < 8:
        score -= 100.0
    return float(score)


def refine_lengths(binary: np.ndarray, root: np.ndarray, vecs: Dict[str, np.ndarray], iters: int = 3) -> Dict[str, np.ndarray]:
    vecs = {k: v.copy() for k, v in vecs.items()}
    init_lens = {k: np.linalg.norm(v) for k, v in vecs.items()}
    scales = [0.75, 0.85, 0.95, 1.0, 1.05, 1.15, 1.3, 1.45]
    for _ in range(iters):
        improved = False
        for k in ['X', 'Y', 'Z']:
            base = vecs[k]
            base_len = np.linalg.norm(base)
            if base_len < 1e-8:
                continue
            direction = base / base_len
            best_vec = base.copy()
            best_score = score_projected_cuboid(binary, build_projected_cuboid(root, vecs))
            lo = 0.65 * init_lens[k]
            hi = 1.6 * init_lens[k]
            for s in scales:
                L = np.clip(base_len * s, lo, hi)
                cand = direction * L
                trial = dict(vecs)
                trial[k] = cand
                sc = score_projected_cuboid(binary, build_projected_cuboid(root, trial))
                if sc > best_score:
                    best_score = sc
                    best_vec = cand
                    improved = True
            vecs[k] = best_vec
        if not improved:
            break
    return vecs


def visualize_root_detail(rgb: np.ndarray, root: np.ndarray, vecs: Dict[str, np.ndarray], P: Dict[str, np.ndarray],
                          idx: int, score: float, out_path: str):
    colors = {'X': (80, 100, 255), 'Y': (80, 220, 80), 'Z': (255, 140, 80)}
    vis = rgb.copy()
    for a, b in projected_cuboid_edges(P):
        cv2.line(vis, tuple(np.round(P[a]).astype(int)), tuple(np.round(P[b]).astype(int)), (255, 0, 255), 2)
    cv2.circle(vis, tuple(np.round(root).astype(int)), 8, (255, 255, 0), -1)
    for k in ['X', 'Y', 'Z']:
        far = root + vecs[k]
        cv2.line(vis, tuple(np.round(root).astype(int)), tuple(np.round(far).astype(int)), colors[k], 4)
        cv2.circle(vis, tuple(np.round(far).astype(int)), 6, colors[k], -1)
    save_image(out_path, vis, f'Root #{idx} | score={score:.2f}')


def choose_best_root_and_axes(binary: np.ndarray, rgb: np.ndarray, segments: List[Segment], thetas: List[float],
                              root_debug_dir: str) -> Dict[str, Any]:
    candidates = root_candidates(binary, segments, max_keep=20)
    axis_order = order_axes(thetas)
    cluster_to_axisname = {axis_order[0]: 'X', axis_order[1]: 'Y', axis_order[2]: 'Z'}
    per_root_results = []
    best_global = None

    for ridx, (cand_score, root, kcount) in enumerate(candidates):
        axis_options = []
        valid = True
        for axis, theta in enumerate(thetas):
            opts = trace_both_signs(binary, root, theta, axis, segments)
            if not opts:
                valid = False
                break
            axis_options.append(sorted(opts, key=lambda x: x.score, reverse=True)[:2])
        if not valid:
            per_root_results.append({
                'root_idx': ridx,
                'root_xy': [float(root[0]), float(root[1])],
                'candidate_score': float(cand_score),
                'cluster_count': int(kcount),
                'valid': False,
            })
            continue

        best_local = None
        for combo in product(*axis_options):
            vecs = {cluster_to_axisname[e.axis]: e.vec for e in combo}
            P = build_projected_cuboid(root, vecs)
            sc = score_projected_cuboid(binary, P) + sum(e.score for e in combo) + 0.2 * cand_score
            if best_local is None or sc > best_local['score_before_refine']:
                best_local = {
                    'score_before_refine': float(sc),
                    'root': root.copy(),
                    'combo': combo,
                    'vecs': vecs,
                    'P': P,
                    'candidate_score': float(cand_score),
                    'cluster_count': int(kcount),
                }

        if best_local is None:
            per_root_results.append({
                'root_idx': ridx,
                'root_xy': [float(root[0]), float(root[1])],
                'candidate_score': float(cand_score),
                'cluster_count': int(kcount),
                'valid': False,
            })
            continue

        best_local['vecs'] = refine_lengths(binary, best_local['root'], best_local['vecs'], iters=4)
        best_local['P'] = build_projected_cuboid(best_local['root'], best_local['vecs'])
        best_local['score'] = score_projected_cuboid(binary, best_local['P']) + 0.2 * cand_score
        best_local['axis_lengths_px'] = {k: float(np.linalg.norm(v)) for k, v in best_local['vecs'].items()}
        best_local['valid'] = True
        best_local['root_idx'] = ridx
        visualize_root_detail(
            rgb,
            best_local['root'],
            best_local['vecs'],
            best_local['P'],
            ridx,
            best_local['score'],
            os.path.join(root_debug_dir, f'root_{ridx:02d}.png')
        )

        per_root_results.append({
            'root_idx': ridx,
            'root_xy': [float(root[0]), float(root[1])],
            'candidate_score': float(cand_score),
            'cluster_count': int(kcount),
            'valid': True,
            'score_before_refine': float(best_local['score_before_refine']),
            'score': float(best_local['score']),
            'axis_lengths_px': best_local['axis_lengths_px'],
        })

        if best_global is None or best_local['score'] > best_global['score']:
            best_global = best_local

    if best_global is None:
        root = candidates[0][1]
        vecs = {'X': np.array([40.0, 0.0]), 'Y': np.array([0.0, 40.0]), 'Z': np.array([25.0, 25.0])}
        best_global = {
            'root': root.copy(),
            'vecs': vecs,
            'P': build_projected_cuboid(root, vecs),
            'score': 0.0,
            'axis_order': axis_order,
            'candidate_score': float(candidates[0][0]),
            'cluster_count': int(candidates[0][2]),
            'root_idx': 0,
        }

    best_global['candidates'] = candidates
    best_global['per_root_results'] = per_root_results
    best_global['axis_order'] = axis_order
    return best_global


def visualize_root_axes(rgb: np.ndarray, root: np.ndarray, vecs: Dict[str, np.ndarray], debug_dir: str):
    colors = {'X': (80, 100, 255), 'Y': (80, 220, 80), 'Z': (255, 140, 80)}
    vis = rgb.copy()
    cv2.circle(vis, tuple(np.round(root).astype(int)), 8, (255, 0, 255), -1)
    for k in ['X', 'Y', 'Z']:
        far = root + vecs[k]
        c = colors[k]
        cv2.line(vis, tuple(np.round(root).astype(int)), tuple(np.round(far).astype(int)), c, 4)
        cv2.circle(vis, tuple(np.round(far).astype(int)), 7, c, -1)
    save_image(os.path.join(debug_dir, '06_root_axes.png'), vis, 'Chosen root and 3 outgoing axes')


def draw_projected_cuboid(rgb: np.ndarray, P: dict, debug_dir: str):
    vis = rgb.copy()
    for a, b in projected_cuboid_edges(P):
        cv2.line(vis, tuple(np.round(P[a]).astype(int)), tuple(np.round(P[b]).astype(int)), (255, 0, 255), 2)
    for k, p in P.items():
        cv2.circle(vis, tuple(np.round(p).astype(int)), 4, (0, 255, 255), -1)
        cv2.putText(vis, k, tuple(np.round(p + np.array([4, -4])).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
    save_image(os.path.join(debug_dir, '07_projected_cuboid_overlay.png'), vis, 'Projected cuboid overlay')


# ---------- OBJ ----------

def write_obj(path: str, lx: float, ly: float, lz: float):
    verts = [
        (0, 0, 0), (lx, 0, 0), (0, ly, 0), (lx, ly, 0),
        (0, 0, lz), (lx, 0, lz), (0, ly, lz), (lx, ly, lz),
    ]
    faces = [
        (1, 2, 4, 3), (5, 6, 8, 7), (1, 2, 6, 5),
        (3, 4, 8, 7), (1, 3, 7, 5), (2, 4, 8, 6),
    ]
    with open(path, 'w', encoding='utf-8') as f:
        f.write('# Recovered 3D bounding box (canonical coordinates)\n')
        for v in verts:
            f.write(f'v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n')
        for quad in faces:
            f.write('f ' + ' '.join(map(str, quad)) + '\n')


def main():
    parser = argparse.ArgumentParser(description='Recover a simple 3D bbox OBJ from a box-like hand-drawn sketch PNG.')
    parser.add_argument('input_png', help='Input sketch PNG (RGB)')
    parser.add_argument('--out_dir', default='bbox_debug_out', help='Output directory')
    parser.add_argument('--min_len', type=int, default=24, help='Min line length for Hough segments')
    args = parser.parse_args()

    out_dir = args.out_dir
    debug_dir = os.path.join(out_dir, 'debug')
    roots_dir = os.path.join(debug_dir, 'roots')
    ensure_dir(debug_dir)
    ensure_dir(roots_dir)

    bgr = cv2.imread(args.input_png, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(args.input_png)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    save_image(os.path.join(debug_dir, '00_input.png'), rgb, 'Input sketch')

    binary = extract_foreground(rgb, debug_dir)
    segments = detect_segments(binary, debug_dir, min_len=args.min_len)
    if len(segments) < 6:
        raise RuntimeError('Too few line segments detected. Try lowering --min_len or providing a cleaner sketch.')

    thetas, _ = weighted_kmeans_dir(segments, k=3, iters=40)
    visualize_clusters(rgb, segments, thetas, debug_dir)

    best = choose_best_root_and_axes(binary, rgb, segments, thetas, roots_dir)
    root = best['root']
    visualize_root_candidates(binary, best['candidates'], root, debug_dir)
    visualize_root_axes(rgb, root, best['vecs'], debug_dir)
    draw_projected_cuboid(rgb, best['P'], debug_dir)

    lx = float(np.linalg.norm(best['vecs']['X']))
    ly = float(np.linalg.norm(best['vecs']['Y']))
    lz = float(np.linalg.norm(best['vecs']['Z']))
    obj_path = os.path.join(out_dir, 'recovered_bbox.obj')
    write_obj(obj_path, lx, ly, lz)

    summary = {
        'cluster_angles_deg': [round(math.degrees(t), 3) for t in thetas],
        'axis_order_cluster_ids': {'X': int(best['axis_order'][0]), 'Y': int(best['axis_order'][1]), 'Z': int(best['axis_order'][2])},
        'best_root_idx': int(best['root_idx']),
        'root_xy': [float(root[0]), float(root[1])],
        'projected_edge_lengths_px': {'X': lx, 'Y': ly, 'Z': lz},
        'score': float(best['score']),
        'obj_path': obj_path,
        'all_root_trials': best['per_root_results'],
        'debug_root_dir': roots_dir,
        'note': 'This version keeps up to 20 root candidates, tries a cuboid fit for each, saves every root visualization, and picks the best overlay score.'
    }
    with open(os.path.join(out_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f'Wrote OBJ to: {obj_path}')
    print(f'Debug outputs in: {debug_dir}')


if __name__ == '__main__':
    main()
