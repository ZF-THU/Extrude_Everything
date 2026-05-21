#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Line extraction + optional hand-drawn wobble (from Sketch_Own_Cur.py logic).
Requires OpenCV (cv2). Intended for use after Blender renders a PNG.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def thinning(bw_u8: np.ndarray) -> np.ndarray:
    bw01 = (bw_u8 > 0).astype(np.uint8) * 255
    try:
        if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
            return cv2.ximgproc.thinning(bw01, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    except Exception:
        pass

    img = bw01.copy()
    skel = np.zeros_like(img)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded
        if cv2.countNonZero(img) == 0:
            break
    return skel


def wobble_warp(binary_u8: np.ndarray, amp_px: float, smooth: int, seed: int) -> np.ndarray:
    if amp_px <= 0:
        return binary_u8

    h, w = binary_u8.shape[:2]
    rng = np.random.RandomState(seed)

    dx = rng.uniform(-1, 1, (h, w)).astype(np.float32)
    dy = rng.uniform(-1, 1, (h, w)).astype(np.float32)

    if smooth and smooth > 1:
        if smooth % 2 == 0:
            smooth += 1
        dx = cv2.GaussianBlur(dx, (smooth, smooth), 0)
        dy = cv2.GaussianBlur(dy, (smooth, smooth), 0)

    mx = max(1e-6, float(np.max(np.abs(dx))))
    my = max(1e-6, float(np.max(np.abs(dy))))
    dx = (dx / mx) * float(amp_px)
    dy = (dy / my) * float(amp_px)

    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = xx + dx
    map_y = yy + dy

    warped = cv2.remap(
        binary_u8,
        map_x,
        map_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    warped = (warped > 0).astype(np.uint8) * 255
    return warped


def ink_variation(line_u8: np.ndarray, rough: int) -> np.ndarray:
    if rough <= 0:
        return line_u8
    k = 2 * rough + 1
    blur = cv2.GaussianBlur(line_u8, (k, k), 0)
    return (blur > 80).astype(np.uint8) * 255


def convert_bgr_to_handdraw_sketch(
    bgr: np.ndarray,
    *,
    black_thr: int = 40,
    close_iter: int = 1,
    no_thinning: bool = False,
    stroke_width: int = 3,
    handdraw: bool = True,
    wobble_amp: float = 1.2,
    wobble_smooth: int = 61,
    wobble_seed: int = 1,
    ink_rough: int = 1,
    invert: bool = True,
    rel_hash: int = 0,
) -> np.ndarray:
    """
    rel_hash: mixed into wobble seed so different filenames differ when base_seed is fixed.
    """
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k_widen = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    m = np.max(bgr, axis=2)
    line = (m <= int(black_thr)).astype(np.uint8) * 255

    if close_iter > 0:
        line = cv2.morphologyEx(line, cv2.MORPH_CLOSE, k_close, iterations=int(close_iter))

    if not no_thinning:
        line = thinning(line)

    base_seed = int(wobble_seed)
    seed = (base_seed + (rel_hash & 0xFFFFFFFF)) & 0x7FFFFFFF
    if handdraw:
        line = wobble_warp(line, amp_px=float(wobble_amp), smooth=int(wobble_smooth), seed=seed)

    sw = max(1, int(stroke_width))
    if sw > 1:
        line = cv2.dilate(line, k_widen, iterations=sw - 1)

    if handdraw and ink_rough > 0:
        line = ink_variation(line, rough=int(ink_rough))

    return (255 - line) if invert else line


def convert_png_to_handdraw_sketch(
    input_path: Path,
    output_path: Path,
    **kwargs,
) -> None:
    bgr = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {input_path}")
    rel_hash = hash(str(input_path.resolve())) & 0xFFFFFFFF
    out = convert_bgr_to_handdraw_sketch(bgr, rel_hash=rel_hash, **kwargs)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), out)


def _parse_args():
    ap = argparse.ArgumentParser(description="Convert a render PNG to a hand-drawn sketch PNG.")
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--black-thr", "--black_thr", type=int, default=40)
    ap.add_argument("--close-iter", "--close_iter", type=int, default=1)
    ap.add_argument("--no-thinning", "--no_thinning", action="store_true")
    ap.add_argument("--stroke-width", "--stroke_width", type=int, default=3)
    ap.add_argument(
        "--handdraw",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply wobble + ink roughness (Sketch_Own_Cur --handdraw). Default: true.",
    )
    ap.add_argument("--wobble-amp", "--wobble_amp", type=float, default=1.2)
    ap.add_argument("--wobble-smooth", "--wobble_smooth", type=int, default=61)
    ap.add_argument("--wobble-seed", "--wobble_seed", type=int, default=1)
    ap.add_argument("--ink-rough", "--ink_rough", type=int, default=1)
    ap.add_argument(
        "--invert",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Black strokes on white background (--no-invert: white on black). Default: true.",
    )
    return ap.parse_args()


def main():
    args = _parse_args()
    convert_png_to_handdraw_sketch(
        args.input,
        args.output,
        black_thr=args.black_thr,
        close_iter=args.close_iter,
        no_thinning=args.no_thinning,
        stroke_width=args.stroke_width,
        handdraw=args.handdraw,
        wobble_amp=args.wobble_amp,
        wobble_smooth=args.wobble_smooth,
        wobble_seed=args.wobble_seed,
        ink_rough=args.ink_rough,
        invert=args.invert,
    )
    print(f"[sketch] wrote {args.output}")


if __name__ == "__main__":
    main()
