#!/usr/bin/env python3
import argparse
import json
import math
import os
from dataclasses import dataclass
from itertools import product
from typing import List, Tuple, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class Segment:
    p1: np.ndarray  # (2,)
    p2: np.ndarray  # (2,)
    length: float
    theta: float    # angle in [0, pi)
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


# ---------- Utility ----------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_image(path: str, img: np.ndarray, title: Optional[str] = None):
    ensure_dir(os.path.dirname(path))
    if img.ndim == 2:
        plt.figure(figsize=(7, 7))
        plt.imshow(img, cmap='gray')
    else:
        plt.figure(figsize=(7, 7))
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


# ---------- Step 1: Foreground extraction ----------

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


# ---------- Step 2: Segment extraction ----------

def detect_segments(binary: np.ndarray, debug_dir: str, min_len=24) -> List[Segment]:
    lines = cv2.HoughLinesP(binary, 1, np.pi / 180.0, threshold=22,
                            minLineLength=min_len, maxLineGap=12)
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


# ---------- Step 3: Direction clustering ----------

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

def root_candidates(binary: np.ndarray, segments: List[Segment]) -> List[Tuple[float, np.ndarray, int]]:
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
            for s in segments:
                if point_to_segment_distance(p, s.p1, s.p2) < 10:
                    support += s.length
                    cl_present.add(s.cluster)
            score = density * 50 + support + 70 * len(cl_present)
            candidates.append((float(score), p, len(cl_present)))

    if not candidates:
        ys, xs = np.where(binary > 0)
        if len(xs) == 0:
            return [(0.0, np.array([binary.shape[1] / 2.0, binary.shape[0] / 2.0]), 0)]
        return [(0.0, np.array([xs.mean(), ys.mean()], dtype=np.float64), 0)]

    # de-duplicate nearby candidates by non-max suppression
    candidates.sort(key=lambda x: x[0], reverse=True)
    kept: List[Tuple[float, np.ndarray, int]] = []
    for score, p, k in candidates:
        if all(np.linalg.norm(p - q) > 12 for _, q, _ in kept):
            kept.append((score, p, k))
        if len(kept) >= 25:
            break
    return kept


def visualize_root_candidates(binary: np.ndarray, candidates: List[Tuple[float, np.ndarray, int]], root: np.ndarray, debug_dir: str):
    vis = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    for score, p, k in candidates[:40]:
        col = (0, 200, 255) if k >= 3 else (255, 120, 0)
        cv2.circle(vis, tuple(np.round(p).astype(int)), 4, col, -1)
    cv2.circle(vis, tuple(np.round(root).astype(int)), 9, (255, 0, 255), -1)
    save_image(os.path.join(debug_dir, '05_root_candidates.png'), vis, 'Root corner candidates')


# ---------- Ray tracing on an axis ----------

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
        # fallback: farthest local max after first few samples
        idx = int(np.argmax(vals[3:])) + 3 if len(vals) > 3 else int(np.argmax(vals))
        if vals[idx] < thr * 0.6:
            return root.copy(), 0.0
        best_end = idx
        best_score = float(vals[idx])

    return pts[best_end], float(best_score)


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


# ---------- Build / score projected cuboid ----------

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


def build_projected_cuboid(root: np.ndarray, vecs: dict):
    vx, vy, vz = vecs['X'], vecs['Y'], vecs['Z']
    P = {
        '000': root,
        '100': root + vx,
        '010': root + vy,
        '001': root + vz,
        '110': root + vx + vy,
        '101': root + vx + vz,
        '011': root + vy + vz,
        '111': root + vx + vy + vz,
    }
    return P


def projected_cuboid_edges(P: dict):
    return [
        ('000', '100'), ('000', '010'), ('000', '001'),
        ('100', '110'), ('100', '101'),
        ('010', '110'), ('010', '011'),
        ('001', '101'), ('001', '011'),
        ('110', '111'), ('101', '111'), ('011', '111'),
    ]


def score_projected_cuboid(binary: np.ndarray, P: dict) -> float:
    supports = []
    edges = projected_cuboid_edges(P)
    lengths = []
    for a, b in edges:
        pa = P[a]
        pb = P[b]
        L = np.linalg.norm(pb - pa)
        lengths.append(L)
        if L < 12:
            supports.append(0.0)
        else:
            s = line_support(binary, pa, pb, samples=max(25, int(L / 3)), radius=3)
            supports.append(s)
    supports_sorted = sorted(supports, reverse=True)
    # tolerate missing hidden edges: score the best 8 edges more heavily
    score = 110.0 * sum(supports_sorted[:8]) + 10.0 * sum(supports_sorted[8:])
    score += 0.25 * sum(sorted(lengths, reverse=True)[:3])
    # discourage degenerate tiny cuboids
    if min(lengths[:3]) < 8:
        score -= 100.0
    return float(score)


def refine_lengths(binary: np.ndarray, root: np.ndarray, vecs: dict, iters: int = 3) -> dict:
    vecs = {k: v.copy() for k, v in vecs.items()}
    init_lens = {k: np.linalg.norm(v) for k, v in vecs.items()}
    scales = [0.8, 0.9, 1.0, 1.1, 1.2, 1.35]
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
            lo = 0.7 * init_lens[k]
            hi = 1.45 * init_lens[k]
            for s in scales:
                L = np.clip(base_len * s, lo, hi)
                cand = direction * L
                vecs_trial = dict(vecs)
                vecs_trial[k] = cand
                P = build_projected_cuboid(root, vecs_trial)
                sc = score_projected_cuboid(binary, P)
                if sc > best_score:
                    best_score = sc
                    best_vec = cand
                    improved = True
            vecs[k] = best_vec
        if not improved:
            break
    return vecs


def choose_best_root_and_axes(binary: np.ndarray, segments: List[Segment], thetas: List[float]):
    candidates = root_candidates(binary, segments)
    axis_order = order_axes(thetas)
    cluster_to_axisname = {axis_order[0]: 'X', axis_order[1]: 'Y', axis_order[2]: 'Z'}

    best = None
    for _, root, _ in candidates:
        axis_options = []
        valid = True
        for axis, theta in enumerate(thetas):
            opts = trace_both_signs(binary, root, theta, axis, segments)
            if not opts:
                valid = False
                break
            # keep two best per axis
            opts = sorted(opts, key=lambda x: x.score, reverse=True)[:2]
            axis_options.append(opts)
        if not valid:
            continue

        for combo in product(*axis_options):
            used_signs = [e.sign for e in combo]
            # avoid exact duplicate direction vectors caused by symmetric traces
            if len(combo) != 3:
                continue
            vecs = {}
            for edge in combo:
                vecs[cluster_to_axisname[edge.axis]] = edge.vec
            P = build_projected_cuboid(root, vecs)
            sc = score_projected_cuboid(binary, P)
            sc += sum(e.score for e in combo)
            if best is None or sc > best['score']:
                best = {
                    'score': float(sc),
                    'root': root.copy(),
                    'combo': combo,
                    'vecs': vecs,
                    'candidates': candidates,
                    'axis_order': axis_order,
                }

    if best is None:
        # fallback to strongest candidate only
        root = candidates[0][1]
        axis_order = order_axes(thetas)
        cluster_to_axisname = {axis_order[0]: 'X', axis_order[1]: 'Y', axis_order[2]: 'Z'}
        vecs = {}
        combo = []
        for axis, theta in enumerate(thetas):
            opts = trace_both_signs(binary, root, theta, axis, segments)
            if not opts:
                d = np.array([math.cos(theta), math.sin(theta)], dtype=np.float64)
                opts = [RootEdge(axis=axis, root=root.copy(), far=root + d * 40, score=0.0, sign=1)]
            best_edge = sorted(opts, key=lambda x: x.score, reverse=True)[0]
            combo.append(best_edge)
            vecs[cluster_to_axisname[axis]] = best_edge.vec
        best = {'score': 0.0, 'root': root.copy(), 'combo': combo, 'vecs': vecs, 'candidates': candidates, 'axis_order': axis_order}

    # coordinate-ascent refinement on axis lengths
    best['vecs'] = refine_lengths(binary, best['root'], best['vecs'], iters=4)
    best['P'] = build_projected_cuboid(best['root'], best['vecs'])
    best['score'] = score_projected_cuboid(binary, best['P'])
    return best


def visualize_root_axes(rgb: np.ndarray, root: np.ndarray, vecs: dict, debug_dir: str):
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


# ---------- Step 7: Write OBJ ----------

def write_obj(path: str, lx: float, ly: float, lz: float):
    verts = [
        (0, 0, 0),
        (lx, 0, 0),
        (0, ly, 0),
        (lx, ly, 0),
        (0, 0, lz),
        (lx, 0, lz),
        (0, ly, lz),
        (lx, ly, lz),
    ]
    faces = [
        (1, 2, 4, 3),
        (5, 6, 8, 7),
        (1, 2, 6, 5),
        (3, 4, 8, 7),
        (1, 3, 7, 5),
        (2, 4, 8, 6),
    ]
    with open(path, 'w', encoding='utf-8') as f:
        f.write('# Recovered 3D bounding box (canonical coordinates)\n')
        for v in verts:
            f.write(f'v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n')
        for quad in faces:
            f.write('f ' + ' '.join(map(str, quad)) + '\n')


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser(description='Recover a simple 3D bbox OBJ from a box-like hand-drawn sketch PNG.')
    parser.add_argument('input_png', help='Input sketch PNG (RGB)')
    parser.add_argument('--out_dir', default='bbox_debug_out', help='Output directory')
    parser.add_argument('--min_len', type=int, default=24, help='Min line length for Hough segments')
    args = parser.parse_args()

    out_dir = args.out_dir
    debug_dir = os.path.join(out_dir, 'debug')
    ensure_dir(debug_dir)

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

    best = choose_best_root_and_axes(binary, segments, thetas)
    root = best['root']
    visualize_root_candidates(binary, best['candidates'], root, debug_dir)
    visualize_root_axes(rgb, root, best['vecs'], debug_dir)
    P = best['P']
    draw_projected_cuboid(rgb, P, debug_dir)

    lx = float(np.linalg.norm(best['vecs']['X']))
    ly = float(np.linalg.norm(best['vecs']['Y']))
    lz = float(np.linalg.norm(best['vecs']['Z']))
    obj_path = os.path.join(out_dir, 'recovered_bbox.obj')
    write_obj(obj_path, lx, ly, lz)

    summary = {
        'cluster_angles_deg': [round(math.degrees(t), 3) for t in thetas],
        'axis_order_cluster_ids': {'X': int(best['axis_order'][0]), 'Y': int(best['axis_order'][1]), 'Z': int(best['axis_order'][2])},
        'root_xy': [float(root[0]), float(root[1])],
        'projected_edge_lengths_px': {'X': lx, 'Y': ly, 'Z': lz},
        'score': best['score'],
        'obj_path': obj_path,
        'note': 'This version traces axis extents in both directions, searches multiple root candidates, and refines lengths by projected-edge support.'
    }
    with open(os.path.join(out_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f'Wrote OBJ to: {obj_path}')
    print(f'Debug outputs in: {debug_dir}')


if __name__ == '__main__':
    main()
