#!/usr/bin/env python3
"""
Signature presence detector (uses per-form rects).
Rect format: (x, y, w, h) as fractions of page size, 0..1.
"""

import argparse
import glob
import math
import os
import re
import cv2
import numpy as np
from skimage import measure

# ---- Your rects (x, y, w, h) as fractions ----
RECTS = {
    "1385":       (0.048, 0.710, 0.148, 0.082),
    "1301-2022":  (0.032, 0.921, 0.149, 0.034),
}

# ---- Core ops ----


def binarize(gray):
    if len(gray.shape) == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return bw  # white=255, black=0


def remove_lines(ink_mask, horiz_len=25, vert_len=25):
    # ink_mask: 0/255, black=0 means ink -> convert to white-on-black for morphology
    mask = (ink_mask == 0).astype(np.uint8) * 255
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, horiz_len), 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(1, vert_len)))
    h = cv2.morphologyEx(mask, cv2.MORPH_OPEN, hk)
    v = cv2.morphologyEx(mask, cv2.MORPH_OPEN, vk)
    lines = cv2.max(h, v)
    cleaned = cv2.subtract(mask, lines)
    return cleaned


def shape_score(roi_bw,
                min_area=50, max_area=6000,
                max_extent=0.65, min_solidity=0.10, max_solidity=0.95,
                min_circ_inv=1.5,
                horiz_len=25, vert_len=25):
    """
    Return best 0..1 score among components for "signature-likeness".
    Softer defaults to handle thin/small signatures.
    """
    fg = remove_lines(roi_bw, horiz_len, vert_len)
    fg = cv2.dilate(fg, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2, 2)), 1)

    labels = measure.label(fg > 0, connectivity=2)
    best = 0.0
    for p in measure.regionprops(labels):
        area = p.area
        if not (min_area <= area <= max_area):
            continue

        minr, minc, maxr, maxc = p.bbox
        bb_area = max(1, (maxr - minr) * (maxc - minc))
        extent = area / bb_area
        if extent >= max_extent:
            continue

        solidity = p.solidity if p.solidity is not None else 1.0
        if not (min_solidity <= solidity <= max_solidity):
            continue

        per = p.perimeter if p.perimeter is not None else 0.0
        circ_inv = (per * per) / (4.0 * math.pi *
                                  area) if area > 0 and per > 0 else 0.0
        if circ_inv < min_circ_inv:
            continue

        score = (1.0 - extent) * 0.5 \
            + (1.0 - abs(solidity - 0.55) / 0.55) * 0.3 \
            + min(circ_inv / 5.0, 1.0) * 0.2
        best = max(best, score)
    return best


# ---- Helpers ----
FORM_ID_RE = re.compile(r"^(?P<id>1301-2022|1385)\b")


def pick_rect_for_file(path, forced_id=None):
    if forced_id:
        if forced_id not in RECTS:
            raise KeyError(
                f"Unknown rect id '{forced_id}'. Known: {list(RECTS)}")
        return forced_id, RECTS[forced_id]
    base = os.path.basename(path)
    m = FORM_ID_RE.match(base)
    if not m:
        raise KeyError(f"Cannot infer form id from filename '{base}'. "
                       f"Use --rect-id to pick one of {list(RECTS)}")
    fid = m.group("id")
    return fid, RECTS[fid]


def xywh_to_tblr(x, y, w, h):
    """(x,y,w,h) -> (top,bottom,left,right) as fractions."""
    top = y
    bottom = y + h
    left = x
    right = x + w
    # clamp
    top = max(0.0, min(1.0, top))
    bottom = max(0.0, min(1.0, bottom))
    left = max(0.0, min(1.0, left))
    right = max(0.0, min(1.0, right))
    return top, bottom, left, right


def detect_signature(img_path, rect_xywh, thresh=0.40):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(img_path)
    h, w = img.shape
    bw = binarize(img)

    x, y, rw, rh = rect_xywh
    top, bottom, left, right = xywh_to_tblr(x, y, rw, rh)
    y1, y2 = int(top * h), int(bottom * h)
    x1, x2 = int(left * w), int(right * w)
    roi_bw = bw[y1:y2, x1:x2]

    s = shape_score(roi_bw)
    return (s >= thresh), s, (x1, y1, x2, y2)

# ---- CLI ----


def main():
    ap = argparse.ArgumentParser(
        description="Yes/No signature detector using per-form rectangles.")
    ap.add_argument("images", nargs="+", help="Image paths or globs")
    ap.add_argument("--rect-id", choices=list(RECTS.keys()),
                    help="Force a specific rect id (otherwise inferred from filename prefix)")
    ap.add_argument("--thresh", type=float, default=0.40,
                    help="Decision threshold on shape score (default: 0.40)")
    args = ap.parse_args()

    # expand globs
    files = []
    for p in args.images:
        g = glob.glob(p)
        files.extend(g if g else [p])

    for f in files:
        try:
            fid, rect = pick_rect_for_file(f, args.rect_id)
            ok, score, (x1, y1, x2, y2) = detect_signature(
                f, rect, args.thresh)
            print(f"{f} [{fid}] -> {'Signature detected' if ok else 'No signature'} (score={score:.2f}) "
                  f"ROI=({x1},{y1})-({x2},{y2})")
        except Exception as e:
            print(f"{f}: ERROR - {e}")


if __name__ == "__main__":
    main()
