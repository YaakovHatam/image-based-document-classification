import math
import os
from pathlib import Path
import tomllib
from typing import List, Tuple, Optional, Dict
import cv2
import numpy as np
from skimage import measure


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
        raise KeyError(f"No rect found for type '{doc_type}' and no default in config")


def load_signature_thresholds() -> Dict[str, float]:
    """
    Reads thresholds from config.toml -> [signature_thresholds].
    """
    try:
        with open("config.toml", "rb") as f:
            cfg = tomllib.load(f)
        s = cfg.get("signature_thresholds", {})
    except FileNotFoundError:
        s = {}

    yes_lower = float(s.get("yes_lower", 0.7))
    maybe_lower = float(s.get("maybe_lower", 0.6))

    yes_lower = max(0.0, min(1.0, yes_lower))
    maybe_lower = max(0.0, min(yes_lower, maybe_lower))
    return {"maybe_lower": maybe_lower, "yes_lower": yes_lower}


def load_sign_labels() -> Dict[str, str]:
    """
    Reads label names from config.toml -> [sign_labels].
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


def classify_score(
    score: float, thresholds: Dict[str, float], labels: Dict[str, str]
) -> str:
    """
    Map score to the configured label.
    """
    if score >= thresholds["yes_lower"]:
        return labels["present"]
    elif score >= thresholds["maybe_lower"]:
        return labels["review"]
    else:
        return labels["none"]


# ---- Core ops ----


def binarize(gray):
    """
    ## IMPROVEMENT: Using adaptive thresholding.
    This is more robust to varying lighting conditions than a single global threshold (like Otsu's).
    """
    if len(gray.shape) == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    # Apply a Gaussian blur to reduce noise before thresholding
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive thresholding calculates a threshold for smaller regions of the image.
    bw = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    return bw  # black=0, white=255 -> after inversion, ink is white


def remove_lines(
    ink_mask,
    horiz_len=25,
    vert_len=25,
    debug_folder: Optional[str] = None,
    step_prefix: str = "",
):
    """Removes horizontal and vertical lines from a binary mask."""
    mask = ink_mask.copy()
    if debug_folder:
        cv2.imwrite(os.path.join(debug_folder, f"{step_prefix}_0_input_mask.png"), mask)

    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, horiz_len), 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(1, vert_len)))
    h = cv2.morphologyEx(mask, cv2.MORPH_OPEN, hk)
    v = cv2.morphologyEx(mask, cv2.MORPH_OPEN, vk)
    lines = cv2.max(h, v)
    cleaned = cv2.subtract(mask, lines)

    if debug_folder:
        cv2.imwrite(
            os.path.join(debug_folder, f"{step_prefix}_1_horizontal_lines.png"), h
        )
        cv2.imwrite(
            os.path.join(debug_folder, f"{step_prefix}_2_vertical_lines.png"), v
        )
        cv2.imwrite(os.path.join(debug_folder, f"{step_prefix}_3_all_lines.png"), lines)
        cv2.imwrite(os.path.join(debug_folder, f"{step_prefix}_4_cleaned.png"), cleaned)

    return cleaned


def shape_score(
    roi_bw,
    min_area=100,
    max_area=15000,  ## IMPROVEMENT: Increased max area to catch larger signatures
    min_density=0.1,
    max_density=0.7,
    min_aspect_ratio=0.1,
    max_aspect_ratio=10.0,
    horiz_len=25,
    vert_len=25,
    debug_folder: Optional[str] = None,
):
    """
    ## IMPROVEMENT: This function now analyzes clusters of ink, not just individual blobs.
    A signature is often made of multiple disconnected strokes. This new logic groups them.
    """
    # The input roi_bw has ink as white (255) on a black (0) background.
    cleaned = remove_lines(
        roi_bw, horiz_len, vert_len, debug_folder=debug_folder, step_prefix="02"
    )

    ## IMPROVEMENT: Connect nearby components to form signature clusters.
    # Dilation helps to merge strokes that are close to each other.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.dilate(cleaned, kernel, iterations=2)
    if debug_folder:
        cv2.imwrite(os.path.join(debug_folder, "03_dilated.png"), dilated)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if debug_folder:
        ## FIX: Corrected the typo from COLOR_GRAY_BGR to COLOR_GRAY2BGR
        contour_img = cv2.cvtColor(roi_bw, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
        cv2.imwrite(os.path.join(debug_folder, "04_contours.png"), contour_img)

    best_score = 0.0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (min_area <= area <= max_area):
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        bb_area = w * h
        if bb_area == 0:
            continue

        # ## IMPROVEMENT: New scoring metrics based on cluster properties.
        # Density: How much of the bounding box is filled with ink?
        density = area / bb_area

        # Aspect Ratio: Is the signature wide or tall?
        aspect_ratio = w / h if h > 0 else 0

        # Check if the properties are within a reasonable range for a signature.
        if not (min_density <= density <= max_density):
            continue
        if not (min_aspect_ratio <= aspect_ratio <= max_aspect_ratio):
            continue

        # A simple scoring function that combines these properties.
        # This can be tuned for better performance.
        score = (density / max_density) * 0.6 + (
            1 - abs(aspect_ratio - 2.5) / 7.5
        ) * 0.4
        best_score = max(best_score, score)

    return best_score


def xywh_to_tblr(x, y, w, h):
    """(x,y,w,h) -> (top,bottom,left,right) as fractions."""
    top, left = y, x
    bottom, right = y + h, x + w
    return max(0.0, top), min(1.0, bottom), max(0.0, left), min(1.0, right)


def detect_signature(
    img_path: Path,
    rect_xywh: Tuple[float, float, float, float],
    debug_folder: Optional[str] = None,
):
    """Detects signature likelihood score and crops ROI."""
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(img_path)

    h, w, _ = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # The binarize function now returns an image where ink is white.
    bw = binarize(gray)
    if debug_folder:
        # For visualization, we can invert it back.
        cv2.imwrite(os.path.join(debug_folder, "00_binarized_full.png"), 255 - bw)

    x, y, rw, rh = rect_xywh
    top, bottom, left, right = xywh_to_tblr(x, y, rw, rh)
    y1, y2 = int(top * h), int(bottom * h)
    x1, x2 = int(left * w), int(right * w)

    roi_bw = bw[y1:y2, x1:x2]
    roi_color = img[y1:y2, x1:x2]

    if debug_folder:
        cv2.imwrite(os.path.join(debug_folder, "01_roi_bw.png"), roi_bw)
        cv2.imwrite(os.path.join(debug_folder, "01_roi_color.png"), roi_color)

    s = shape_score(roi_bw, debug_folder=debug_folder)
    return s, (x1, y1, x2, y2), roi_color


def sig_detector_main(files: List[Tuple[Path, str]], debug=True):
    """
    Process images for signatures and RETURN rows for reporting.
    """
    results = []
    if not files:
        print("No image files found.")
        return results

    thresholds = load_signature_thresholds()
    labels = load_sign_labels()

    for f, doc_type in files:
        try:
            doc_basename_ext = os.path.basename(f)
            doc_basename = os.path.splitext(doc_basename_ext)[0]
            doc_folder = os.path.dirname(f) or "."

            debug_folder = None
            if debug:
                debug_folder = os.path.join(doc_folder, "sig_debug")
                os.makedirs(debug_folder, exist_ok=True)
                print(f"Debug mode ON. Saving pipeline images to: {debug_folder}")

            rect = load_rect_from_config(doc_type)

            score, (x1, y1, x2, y2), roi_img = detect_signature(
                f, rect, debug_folder=debug_folder
            )

            sign_level = classify_score(float(score), thresholds, labels)

            sig_box_filename = f"{doc_basename}_signature_box.png"
            sig_box_path = os.path.join(doc_folder, sig_box_filename)
            cv2.imwrite(sig_box_path, roi_img)

            print(
                f"[{doc_basename_ext}] -> {sign_level} (score={score:.2f}) | Box: {sig_box_path}"
            )
            if debug:
                with open(
                    os.path.join(doc_folder, "signatures.log"), "a", encoding="utf-8"
                ) as log_file:
                    log_file.write(
                        f"{doc_basename_ext}: {sign_level} (score={score:.2f})\n"
                    )

            results.append(
                {
                    "file": doc_basename_ext,
                    "doc_type": doc_type,
                    "sign_level": sign_level,
                    "score": round(float(score), 4),
                    "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    "signature_box_path": sig_box_path,
                }
            )

        except Exception as e:
            print(f"ERROR processing {f}: {e}")
            results.append(
                {
                    "file": os.path.basename(str(f)),
                    "doc_type": doc_type,
                    "sign_level": "error",
                    "score": None,
                    "bbox": None,
                    "signature_box_path": None,
                    "error": str(e),
                }
            )

    return results
