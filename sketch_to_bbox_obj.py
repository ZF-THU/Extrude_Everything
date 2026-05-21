#!/usr/bin/env python3
import argparse
import json
import math
import os
from dataclasses import dataclass
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

    @property
    def direction(self):
        v = self.p2 - self.p1
        n = np.linalg.norm(v)
        if n < 1e-8:
            return np.array([1.0, 0.0])
        return v / n


@dataclass
class RootEdge:
    axis: int
    root: np.ndarray
    far: np.ndarray
    score: float


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
    theta = theta % math.pi
    return theta


def ang_diff_mod_pi(a: float, b: float) -> float:
    d = abs(a - b) % math.pi
    return min(d, math.pi - d)


def point_line_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    if np.linalg.norm(ab) < 1e-8:
        return float(np.linalg.norm(p - a))
    return float(abs(np.cross(ab, p - a)) / np.linalg.norm(ab))


def segment_intersection(a1, a2, b1, b2) -> Optional[np.ndarray]:
    # line-line intersection, not segment-only; caller can apply tolerance.
    x1, y1 = a1
    x2, y2 = a2
    x3, y3 = b1
    x4, y4 = b2
    denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    if abs(denom) < 1e-8:
        return None
    px = ((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4 - y3*x4)) / denom
    py = ((x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4)) / denom
    return np.array([px, py], dtype=np.float64)


def point_to_segment_distance(p, a, b):
    ab = b - a
    if np.linalg.norm(ab) < 1e-8:
        return float(np.linalg.norm(p - a))
    t = np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0.0, 1.0)
    proj = a + t * ab
    return float(np.linalg.norm(p - proj))


def line_support(binary: np.ndarray, a: np.ndarray, b: np.ndarray, samples=80, radius=2) -> float:
    h, w = binary.shape
    vals = []
    for t in np.linspace(0.0, 1.0, samples):
        p = a * (1 - t) + b * t
        x = int(round(p[0]))
        y = int(round(p[1]))
        x0, x1 = max(0, x-radius), min(w, x+radius+1)
        y0, y1 = max(0, y-radius), min(h, y+radius+1)
        vals.append(binary[y0:y1, x0:x1].mean() if x1 > x0 and y1 > y0 else 0.0)
    return float(np.mean(vals))


# ---------- Step 1: Foreground extraction ----------

def extract_foreground(rgb: np.ndarray, debug_dir: str) -> np.ndarray:
    h, w, _ = rgb.shape
    border = np.concatenate([
        rgb[:8, :, :].reshape(-1, 3),
        rgb[-8:, :, :].reshape(-1, 3),
        rgb[:, :8, :].reshape(-1, 3),
        rgb[:, -8:, :].reshape(-1, 3),
    ], axis=0)
    bg = np.median(border, axis=0)

    diff = np.linalg.norm(rgb.astype(np.float32) - bg.astype(np.float32), axis=2)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # foreground if darker than background or color-different enough
    fg = ((gray < np.percentile(gray, 70) - 10) | (diff > 25)).astype(np.uint8) * 255

    # clean up
    kernel = np.ones((3, 3), np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel)
    fg = cv2.dilate(fg, kernel, iterations=1)

    save_image(os.path.join(debug_dir, '01_foreground_mask.png'), fg, 'Foreground mask')
    return fg


# ---------- Step 2: Segment extraction ----------

def detect_segments(binary: np.ndarray, debug_dir: str, min_len=30) -> List[Segment]:
    lines = cv2.HoughLinesP(binary, 1, np.pi / 180.0, threshold=35,
                            minLineLength=min_len, maxLineGap=10)
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
    X = np.array([[math.cos(2*s.theta), math.sin(2*s.theta)] for s in segments], dtype=np.float64)
    W = np.array([s.length for s in segments], dtype=np.float64)

    # init from weighted histogram peaks
    bins = 180
    hist = np.zeros(bins, dtype=np.float64)
    for s in segments:
        idx = int((s.theta / math.pi) * bins) % bins
        hist[idx] += s.length
    hist_s = cv2.GaussianBlur(hist.reshape(1, -1), (1, 0), 3).flatten() if bins > 1 else hist
    peak_ids = np.argsort(hist)[-k:]
    centers = []
    for pid in peak_ids:
        theta = (pid + 0.5) / bins * math.pi
        centers.append([math.cos(2*theta), math.sin(2*theta)])
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

    # Reassign using ordered centers
    dists = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    labels = np.argmin(dists, axis=1)
    for s, lbl in zip(segments, labels):
        s.cluster = int(lbl)

    return thetas.tolist(), labels.tolist()


def visualize_clusters(rgb: np.ndarray, segments: List[Segment], thetas: List[float], debug_dir: str):
    colors = [(255, 80, 80), (80, 220, 80), (80, 160, 255)]
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


# ---------- Step 4: Candidate root corner ----------

def local_density(binary: np.ndarray, p: np.ndarray, radius=8) -> float:
    h, w = binary.shape
    x, y = int(round(p[0])), int(round(p[1]))
    x0, x1 = max(0, x-radius), min(w, x+radius+1)
    y0, y1 = max(0, y-radius), min(h, y+radius+1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return float(binary[y0:y1, x0:x1].mean() / 255.0)


def choose_root(binary: np.ndarray, segments: List[Segment], debug_dir: str) -> np.ndarray:
    candidates = []
    n = len(segments)
    for i in range(n):
        for j in range(i+1, n):
            if segments[i].cluster == segments[j].cluster:
                continue
            p = segment_intersection(segments[i].p1, segments[i].p2, segments[j].p1, segments[j].p2)
            if p is None:
                continue
            # keep if close to both segments
            d1 = point_to_segment_distance(p, segments[i].p1, segments[i].p2)
            d2 = point_to_segment_distance(p, segments[j].p1, segments[j].p2)
            if d1 > 12 or d2 > 12:
                continue
            density = local_density(binary, p, radius=10)
            # count support from 3 clusters
            support = 0
            cl_present = set()
            for s in segments:
                if point_to_segment_distance(p, s.p1, s.p2) < 10:
                    support += s.length
                    cl_present.add(s.cluster)
            score = density * 50 + support + 60 * len(cl_present)
            candidates.append((score, p, len(cl_present)))

    if not candidates:
        # fallback: choose center of foreground mass
        ys, xs = np.where(binary > 0)
        root = np.array([xs.mean(), ys.mean()], dtype=np.float64)
    else:
        candidates.sort(key=lambda x: x[0], reverse=True)
        root = candidates[0][1]

    vis = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    if candidates:
        for score, p, k in candidates[:40]:
            col = (0, 200, 255) if k >= 3 else (255, 120, 0)
            cv2.circle(vis, tuple(np.round(p).astype(int)), 4, col, -1)
    cv2.circle(vis, tuple(np.round(root).astype(int)), 9, (255, 0, 255), -1)
    save_image(os.path.join(debug_dir, '05_root_candidates.png'), vis, 'Root corner candidates')
    return root


# ---------- Step 5: Trace one edge per axis ----------

def nearest_endpoint_to_root(seg: Segment, root: np.ndarray):
    d1 = np.linalg.norm(seg.p1 - root)
    d2 = np.linalg.norm(seg.p2 - root)
    if d1 <= d2:
        return seg.p1, seg.p2, d1
    return seg.p2, seg.p1, d2


def choose_edge_for_axis(binary: np.ndarray, segments: List[Segment], root: np.ndarray, axis: int, theta: float) -> RootEdge:
    axis_dir = np.array([math.cos(theta), math.sin(theta)], dtype=np.float64)
    best = None
    for s in segments:
        if s.cluster != axis:
            continue
        near, far, root_dist = nearest_endpoint_to_root(s, root)
        if root_dist > max(18.0, 0.12 * s.length):
            continue
        v = far - near
        if np.dot(v, axis_dir) < 0:
            far, near = near, far
            v = far - near
        if np.linalg.norm(v) < 20:
            continue
        support = line_support(binary, near, far, samples=max(30, int(np.linalg.norm(v) // 3)), radius=3)
        score = support * 100 + np.linalg.norm(v) - 3 * root_dist
        if best is None or score > best.score:
            best = RootEdge(axis=axis, root=near, far=far, score=float(score))

    # fallback: ray along axis
    if best is None:
        h, w = binary.shape
        candidates = []
        for sign in (+1, -1):
            d = axis_dir * sign
            end = root.copy()
            for t in np.linspace(15, max(h, w) * 0.8, 80):
                p = root + d * t
                if p[0] < 0 or p[0] >= w or p[1] < 0 or p[1] >= h:
                    break
                score = local_density(binary, p, radius=3)
                candidates.append((score, p.copy()))
            if candidates:
                candidates.sort(key=lambda x: x[0], reverse=True)
                far = candidates[0][1]
                best = RootEdge(axis=axis, root=root.copy(), far=far, score=float(candidates[0][0]))
                break
    return best


def visualize_root_edges(rgb: np.ndarray, root: np.ndarray, root_edges: List[RootEdge], debug_dir: str):
    colors = [(255, 80, 80), (80, 220, 80), (80, 160, 255)]
    vis = rgb.copy()
    cv2.circle(vis, tuple(np.round(root).astype(int)), 8, (255, 0, 255), -1)
    for re in root_edges:
        c = colors[re.axis]
        cv2.line(vis, tuple(np.round(re.root).astype(int)), tuple(np.round(re.far).astype(int)), c, 4)
        cv2.circle(vis, tuple(np.round(re.far).astype(int)), 7, c, -1)
    save_image(os.path.join(debug_dir, '06_root_axes.png'), vis, 'Chosen root and 3 outgoing axes')


# ---------- Step 6: Build projected cuboid ----------

def order_axes(thetas: List[float]) -> Tuple[int, int, int]:
    # Map clusters to X, Y, Z. Y = closest to vertical. Remaining: flatter angle -> X, other -> Z.
    vertical_ref = math.pi / 2
    y_idx = int(np.argmin([ang_diff_mod_pi(t, vertical_ref) for t in thetas]))
    rest = [i for i in range(3) if i != y_idx]
    # smaller absolute slope => X, larger => Z
    def slope_mag(i):
        return abs(math.tan(thetas[i])) if abs(math.cos(thetas[i])) > 1e-6 else 1e9
    rest = sorted(rest, key=lambda i: slope_mag(i))
    x_idx, z_idx = rest[0], rest[1]
    return x_idx, y_idx, z_idx


def build_projected_cuboid(root: np.ndarray, root_edges: List[RootEdge], axis_order: Tuple[int, int, int]):
    edge_map = {re.axis: re for re in root_edges}
    x_idx, y_idx, z_idx = axis_order
    vx = edge_map[x_idx].far - root
    vy = edge_map[y_idx].far - root
    vz = edge_map[z_idx].far - root

    P = {}
    P['000'] = root
    P['100'] = root + vx
    P['010'] = root + vy
    P['001'] = root + vz
    P['110'] = root + vx + vy
    P['101'] = root + vx + vz
    P['011'] = root + vy + vz
    P['111'] = root + vx + vy + vz
    return P, vx, vy, vz


def draw_projected_cuboid(rgb: np.ndarray, P: dict, debug_dir: str):
    vis = rgb.copy()
    edges = [
        ('000', '100'), ('000', '010'), ('000', '001'),
        ('100', '110'), ('100', '101'),
        ('010', '110'), ('010', '011'),
        ('001', '101'), ('001', '011'),
        ('110', '111'), ('101', '111'), ('011', '111')
    ]
    for a, b in edges:
        cv2.line(vis, tuple(np.round(P[a]).astype(int)), tuple(np.round(P[b]).astype(int)), (255, 0, 255), 2)
    for k, p in P.items():
        cv2.circle(vis, tuple(np.round(p).astype(int)), 4, (0, 255, 255), -1)
        cv2.putText(vis, k, tuple(np.round(p + np.array([4, -4])).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
    save_image(os.path.join(debug_dir, '07_projected_cuboid_overlay.png'), vis, 'Projected cuboid overlay')


# ---------- Step 7: Write OBJ ----------

def write_obj(path: str, lx: float, ly: float, lz: float):
    # Canonical axis-aligned cuboid in 3D. Up to unknown camera/scale, this is the recovered 3D bbox.
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
    parser.add_argument('--min_len', type=int, default=30, help='Min line length for Hough segments')
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

    root = choose_root(binary, segments, debug_dir)
    root_edges = []
    for axis, theta in enumerate(thetas):
        re = choose_edge_for_axis(binary, segments, root, axis, theta)
        if re is None:
            raise RuntimeError(f'Failed to trace an outgoing edge for axis {axis}.')
        root_edges.append(re)
    visualize_root_edges(rgb, root, root_edges, debug_dir)

    axis_order = order_axes(thetas)
    P, vx, vy, vz = build_projected_cuboid(root, root_edges, axis_order)
    draw_projected_cuboid(rgb, P, debug_dir)

    lx = float(np.linalg.norm(vx))
    ly = float(np.linalg.norm(vy))
    lz = float(np.linalg.norm(vz))
    obj_path = os.path.join(out_dir, 'recovered_bbox.obj')
    write_obj(obj_path, lx, ly, lz)

    # Save a compact summary.
    summary = {
        'cluster_angles_deg': [round(math.degrees(t), 3) for t in thetas],
        'axis_order_cluster_ids': {'X': int(axis_order[0]), 'Y': int(axis_order[1]), 'Z': int(axis_order[2])},
        'root_xy': root.tolist(),
        'projected_edge_lengths_px': {'X': lx, 'Y': ly, 'Z': lz},
        'obj_path': obj_path,
        'note': 'This prototype assumes a single box-like sketch under axonometric/oblique projection. The OBJ is canonical 3D, up to unknown global scale and camera.'
    }
    with open(os.path.join(out_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f'Wrote OBJ to: {obj_path}')
    print(f'Debug outputs in: {debug_dir}')


if __name__ == '__main__':
    main()
