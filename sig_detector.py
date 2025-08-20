import math
import os
from pathlib import Path
import tomllib
from typing import List, Tuple, Optional, Dict
import cv2
import numpy as np
from skimage import measure


# Read config
with open("config.toml", "rb") as f:
    config = tomllib.load(f)
DEBUG_OUTPUT_DIR = config["debug"]["output_dir"]


def load_rect_from_config(doc_type: str) -> Tuple[float, float, float, float]:
    """Load signature ROI rectangle for a document type from config.toml [rects]."""
    with open("config.toml", "rb") as f:
        cfg = tomllib.load(f)
    rects = cfg.get("rects", {})
    if doc_type in rects:
        return tuple(rects[doc_type])
    elif "default" in rects:
        return tuple(rects["default"])
    else:
        raise KeyError(
            f"No rect found for type '{doc_type}' and no default in config")


def load_signature_thresholds() -> Dict[str, float]:
    """
    Reads thresholds from config.toml -> [signature_thresholds].
    Defaults: maybe_lower=0.6, yes_lower=0.7.
    Ensures 0 <= maybe_lower <= yes_lower <= 1.
    """
    try:
        with open("config.toml", "rb") as f:
            cfg = tomllib.load(f)
        s = cfg.get("signature_thresholds", {})
    except FileNotFoundError:
        s = {}

    yes_lower = float(s.get("yes_lower", 0.7))
    maybe_lower = float(s.get("maybe_lower", 0.6))

    # clamp & order
    yes_lower = max(0.0, min(1.0, yes_lower))
    maybe_lower = max(0.0, min(yes_lower, maybe_lower))
    return {"maybe_lower": maybe_lower, "yes_lower": yes_lower}


def load_sign_labels() -> Dict[str, str]:
    """
    Reads label names from config.toml -> [sign_labels].
    Defaults: none='none', review='review', present='present'.
    """
    try:
        with open("config.toml", "rb") as f:
            cfg = tomllib.load(f)
        sl = cfg.get("sign_labels", {})
    except FileNotFoundError:
        sl = {}

    none_lbl = str(sl.get("none", "none"))
    review_lbl = str(sl.get("review", "review"))
    present_lbl = str(sl.get("present", "present"))

    return {"none": none_lbl, "review": review_lbl, "present": present_lbl}


def classify_score(score: float, thresholds: Dict[str, float], labels: Dict[str, str]) -> str:
    """
    Map score to the configured label:
      score >= yes_lower          -> labels["present"]
      maybe_lower <= score < yes  -> labels["review"]
      score < maybe_lower         -> labels["none"]
    """
    if score >= thresholds["yes_lower"]:
        return labels["present"]
    elif score >= thresholds["maybe_lower"]:
        return labels["review"]
    else:
        return labels["none"]


# ---- Core ops ----

def binarize(gray):
    """Converts a grayscale image to binary using Otsu's method."""
    if len(gray.shape) == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return bw  # white=255, black=0


def remove_lines(
    ink_mask,
    horiz_len=25,
    vert_len=25,
    debug_folder: Optional[str] = None,
    step_prefix: str = "",
):
    """Removes horizontal and vertical lines from a binary mask."""
    # ink_mask: 0/255, black=0 -> convert to white-on-black for morphology
    mask = (ink_mask == 0).astype(np.uint8) * 255
    if debug_folder:
        cv2.imwrite(os.path.join(
            debug_folder, f"{step_prefix}_0_inverted_mask.png"), mask)

    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, horiz_len), 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(1, vert_len)))
    h = cv2.morphologyEx(mask, cv2.MORPH_OPEN, hk)
    v = cv2.morphologyEx(mask, cv2.MORPH_OPEN, vk)
    lines = cv2.max(h, v)
    cleaned = cv2.subtract(mask, lines)

    if debug_folder:
        cv2.imwrite(os.path.join(
            debug_folder, f"{step_prefix}_1_horizontal_lines.png"), h)
        cv2.imwrite(os.path.join(
            debug_folder, f"{step_prefix}_2_vertical_lines.png"), v)
        cv2.imwrite(os.path.join(
            debug_folder, f"{step_prefix}_3_all_lines.png"), lines)
        cv2.imwrite(os.path.join(
            debug_folder, f"{step_prefix}_4_cleaned.png"), cleaned)

    return cleaned


def shape_score(
    roi_bw,
    min_area=100,
    max_area=6000,
    max_extent=0.65,
    min_solidity=0.10,
    max_solidity=0.95,
    min_circ_inv=1.5,
    horiz_len=25,
    vert_len=25,
    debug_folder: Optional[str] = None,
):
    """
    Return best 0..1 score among components for "signature-likeness".
    Softer defaults to handle thin/small signatures.
    """
    fg = remove_lines(roi_bw, horiz_len, vert_len,
                      debug_folder=DEBUG_OUTPUT_DIR, step_prefix="02")
    fg = cv2.dilate(fg, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2, 2)), 1)
    if debug_folder:
        cv2.imwrite(os.path.join(debug_folder, "03_dilated.png"), fg)

    labels = measure.label(fg > 0, connectivity=2)
    if debug_folder:
        labeled_img_vis = cv2.normalize(
            labels, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        labeled_img_vis = cv2.applyColorMap(labeled_img_vis, cv2.COLORMAP_JET)
        labeled_img_vis[labels == 0] = [0, 0, 0]
        cv2.imwrite(os.path.join(
            debug_folder, "04_labels.png"), labeled_img_vis)

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

        score = (
            (1.0 - extent) * 0.5
            + (1.0 - abs(solidity - 0.55) / 0.55) * 0.3
            + min(circ_inv / 5.0, 1.0) * 0.2
        )
        best = max(best, score)
    return best


def xywh_to_tblr(x, y, w, h):
    """(x,y,w,h) -> (top,bottom,left,right) as fractions."""
    top, left = y, x
    bottom, right = y + h, x + w
    return max(0.0, top), min(1.0, bottom), max(0.0, left), min(1.0, right)


def detect_signature(
    img_path: Path,
    rect_xywh: Tuple[float, float, float, float],
    thresh: float,  # not used for score calculation, but kept for compatibility/prints
    debug_folder: Optional[str] = None,
):
    """Detects signature likelihood score and crops ROI."""
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(img_path)

    h, w, _ = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bw = binarize(gray)
    if debug_folder:
        cv2.imwrite(os.path.join(debug_folder, "00_binarized_full.png"), bw)

    x, y, rw, rh = rect_xywh
    top, bottom, left, right = xywh_to_tblr(x, y, rw, rh)
    y1, y2 = int(top * h), int(bottom * h)
    x1, x2 = int(left * w), int(right * w)

    roi_bw = bw[y1:y2, x1:x2]
    roi_color = img[y1:y2, x1:x2]

    if debug_folder:
        cv2.imwrite(os.path.join(debug_folder, "01_roi_bw.png"), roi_bw)
        cv2.imwrite(os.path.join(debug_folder, "01_roi_color.png"), roi_color)

    s = shape_score(roi_bw, debug_folder=DEBUG_OUTPUT_DIR)
    return s, (x1, y1, x2, y2), roi_color


def sig_detector_main(files: List[Tuple[Path, str]], thresh: Optional[float] = None, debug=True):
    """
    Process images for signatures and RETURN rows for reporting.
    Returns rows with: file, doc_type, sign_level, score, bbox, signature_box_path
    """
    results = []
    if not files:
        print("No image files found.")
        return results

    thresholds = load_signature_thresholds()
    labels = load_sign_labels()
    # If caller passes no thresh, use yes_lower (minimal behavior parity)
    effective_thresh = thresholds["yes_lower"] if thresh is None else float(
        thresh)

    for f, doc_type in files:
        try:
            doc_basename_ext = os.path.basename(f)
            doc_basename = os.path.splitext(doc_basename_ext)[0]
            doc_folder = os.path.dirname(f) or "."

            debug_folder = DEBUG_OUTPUT_DIR
            if debug:
                debug_folder = os.path.join(doc_folder, "sig_debug")
                os.makedirs(debug_folder, exist_ok=True)
                print(
                    f"Debug mode ON. Saving pipeline images to: {debug_folder}")

            rect = load_rect_from_config(doc_type)

            score, (x1, y1, x2, y2), roi_img = detect_signature(
                f, rect, effective_thresh, debug_folder=debug_folder
            )

            sign_level = classify_score(float(score), thresholds, labels)

            sig_box_filename = f"{doc_basename}_signature_box.png"
            sig_box_path = os.path.join(doc_folder, sig_box_filename)
            cv2.imwrite(sig_box_path, roi_img)

            print(
                f"[{doc_basename_ext}] -> {sign_level} (score={score:.2f}) | Box: {sig_box_path}")
            if debug:
                with open(os.path.join(doc_folder, "signatures.log"), "a", encoding="utf-8") as log_file:
                    log_file.write(
                        f"{doc_basename_ext}: {sign_level} (score={score:.2f})\n")

            results.append({
                "file": doc_basename_ext,
                "doc_type": doc_type,
                # ONLY this label (no has_signature)
                "sign_level": sign_level,
                "score": round(float(score), 4),
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "signature_box_path": sig_box_path
            })

        except Exception as e:
            print(f"ERROR processing {f}: {e}")
            results.append({
                "file": os.path.basename(str(f)),
                "doc_type": doc_type,
                "sign_level": "error",
                "score": None,
                "bbox": None,
                "signature_box_path": None,
                "error": str(e)
            })

    return results
