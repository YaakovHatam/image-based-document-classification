DEBUG_STEP_COUNTER = 0
from pathlib import Path
import cv2 as cv
import os
import numpy as np
from PIL import Image
from typing import List

# --------------------
# CONFIG
# --------------------
import tomllib  # For Python 3.11+, use `import toml` for earlier versions

with open("config.toml", "rb") as f:
    config = tomllib.load(f)

IMG_WIDTH = config["general"]["img_width"]
IMG_HEIGHT = config["general"]["img_height"]
HEADER_RATIO = config["general"]["header_ratio"]
FOOTER_RATIO = config["general"]["footer_ratio"]

DEBUG_MODE = config["debug"]["mode"]
DEBUG_OUTPUT_DIR = config["debug"]["output_dir"]
DEBUG_STEP_COUNTER = 0

# ORB parameters
orb = cv.ORB_create(nfeatures=500)
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)


def save_debug_image(step_name: str, img, prefix: str = "debug"):
    """Save a debug image to the debug output directory with incremental step number."""
    global DEBUG_STEP_COUNTER
    if not DEBUG_MODE:
        return

    os.makedirs(DEBUG_OUTPUT_DIR, exist_ok=True)

    # Convert PIL to NumPy if needed
    if isinstance(img, Image.Image):
        img = np.array(img)  # RGB
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)

    DEBUG_STEP_COUNTER += 1
    filename = f"{DEBUG_STEP_COUNTER:03d}_{prefix}_{step_name}.png"
    cv.imwrite(os.path.join(DEBUG_OUTPUT_DIR, "debug", filename), img)


# --------------------
# IMAGE HELPERS
# --------------------
def preprocess(img: Image.Image):
    """Convert to OpenCV format, grayscale, resize, and binarize."""
    if isinstance(img, Image.Image):
        img = np.array(img)  # PIL → NumPy (RGB)
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)

    save_debug_image("Original", img)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    save_debug_image("Gray", gray)

    gray = cv.resize(gray, (IMG_WIDTH, IMG_HEIGHT))
    save_debug_image("Resized", gray)

    gray = cv.GaussianBlur(gray, (3, 3), 0)
    save_debug_image("Blurred", gray)

    bin_img = cv.adaptiveThreshold(
        gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 15, 10
    )
    save_debug_image("Binarized", bin_img)

    return bin_img


def get_header_footer(img):
    """Extract header and footer regions."""
    h = img.shape[0]
    header = img[0 : int(h * HEADER_RATIO), :]
    footer = img[int(h * (1 - FOOTER_RATIO)) :, :]

    save_debug_image("Header", header)
    save_debug_image("Footer", footer)

    return header, footer


# --- Inserted helpers ---
def geometric_inliers(kp_t, kp_p, good_matches, ransac_thresh=3.0):
    """
    Compute number of geometric inliers using RANSAC homography between template (kp_t) and page (kp_p).
    """
    if len(good_matches) < 4:
        return 0
    src = np.float32([kp_t[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst = np.float32([kp_p[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    H, mask = cv.findHomography(src, dst, cv.RANSAC, ransac_thresh)
    return int(mask.sum()) if mask is not None else 0


def knn_ratio_matches(des_t, des_p, ratio=0.75):
    """
    Return list of 'good' matches using Lowe's ratio test (template -> page).
    """
    if des_t is None or des_p is None:
        return []
    knn = bf.knnMatch(des_t, des_p, k=2)
    good = []
    for pair in knn:
        if len(pair) == 2:
            m, n = pair
            if m.distance < ratio * n.distance:
                good.append(m)
    return good


def extract_features(img):
    """Extract ORB features (keypoints + descriptors)."""
    kp, des = orb.detectAndCompute(img, None)
    return kp, des


# --------------------
# MATCHING/SCORING
# --------------------


def match_scores_pair(kp_p, des_p, kp_t, des_t):
    """
    Compute raw good-match count and geometric inliers between a page region and a template region.
    Returns (raw_good, inliers).
    """
    good = knn_ratio_matches(des_t, des_p, ratio=0.75)
    inliers = geometric_inliers(kp_t, kp_p, good, ransac_thresh=3.0)
    return len(good), inliers


# --------------------
# BUILD TEMPLATE DB
# --------------------


# --------------------
# RECOGNIZE PAGE
# --------------------
def recognize_page(img_path, template_db):
    img = preprocess(img_path)
    header, footer = get_header_footer(img)

    kp_h_p, des_h_p = extract_features(header)
    kp_f_p, des_f_p = extract_features(footer)
    page_kpts_total = len(kp_h_p or []) + len(kp_f_p or [])

    scores = {}  # template_name -> primary score (inliers total)
    details = {}  # template_name -> dict with raw/inliers/percentages

    for template_name, feats in template_db.items():
        # Header region
        raw_h, inl_h = match_scores_pair(
            kp_h_p, des_h_p, feats["header"]["kp"], feats["header"]["des"]
        )
        # Footer region
        raw_f, inl_f = match_scores_pair(
            kp_f_p, des_f_p, feats["footer"]["kp"], feats["footer"]["des"]
        )

        raw_total = raw_h + raw_f
        inl_total = inl_h + inl_f
        tmpl_kpts_total = feats["kpts_total"]

        # Normalized percentages
        pct_vs_template = (
            (inl_total / tmpl_kpts_total * 100.0) if tmpl_kpts_total > 0 else 0.0
        )
        pct_dice = 2.0 * inl_total / max(tmpl_kpts_total + page_kpts_total, 1) * 100.0

        scores[template_name] = inl_total
        details[template_name] = {
            "raw_total": int(raw_total),
            "inliers_total": int(inl_total),
            "tmpl_kpts_total": int(tmpl_kpts_total),
            "page_kpts_total": int(page_kpts_total),
            "pct_vs_template": float(pct_vs_template),
            "pct_dice": float(pct_dice),
            "raw_header": int(raw_h),
            "raw_footer": int(raw_f),
            "inliers_header": int(inl_h),
            "inliers_footer": int(inl_f),
        }

    # Determine best by inliers
    best_template = max(scores, key=scores.get)
    best_score = scores[best_template]

    # Second percentage: relative to best (for logging/reporting)
    best_inliers = max(v["inliers_total"] for v in details.values()) if details else 0
    for v in details.values():
        v["pct_of_best"] = (
            (v["inliers_total"] / best_inliers * 100.0) if best_inliers > 0 else 0.0
        )

    return best_template, best_score, scores, details


def recognize_page_with_orientation(img: Image.Image, template_db):
    img = preprocess(img)

    def score_against_templates(image):
        header, footer = get_header_footer(image)
        kp_h_p, des_h_p = extract_features(header)
        kp_f_p, des_f_p = extract_features(footer)
        page_kpts_total = len(kp_h_p or []) + len(kp_f_p or [])

        scores = {}
        details = {}
        for template_name, feats in template_db.items():
            raw_h, inl_h = match_scores_pair(
                kp_h_p, des_h_p, feats["header"]["kp"], feats["header"]["des"]
            )
            raw_f, inl_f = match_scores_pair(
                kp_f_p, des_f_p, feats["footer"]["kp"], feats["footer"]["des"]
            )

            raw_total = raw_h + raw_f
            inl_total = inl_h + inl_f
            tmpl_kpts_total = feats["kpts_total"]

            pct_vs_template = (
                (inl_total / tmpl_kpts_total * 100.0) if tmpl_kpts_total > 0 else 0.0
            )
            pct_dice = (
                2.0 * inl_total / max(tmpl_kpts_total + page_kpts_total, 1) * 100.0
            )

            scores[template_name] = inl_total
            details[template_name] = {
                "raw_total": int(raw_total),
                "inliers_total": int(inl_total),
                "tmpl_kpts_total": int(tmpl_kpts_total),
                "page_kpts_total": int(page_kpts_total),
                "pct_vs_template": float(pct_vs_template),
                "pct_dice": float(pct_dice),
            }

        best_inliers = (
            max(v["inliers_total"] for v in details.values()) if details else 0
        )
        for v in details.values():
            v["pct_of_best"] = (
                (v["inliers_total"] / best_inliers * 100.0) if best_inliers > 0 else 0.0
            )

        return scores, details

    # Normal orientation
    scores_normal, details_normal = score_against_templates(img)

    # Rotated orientation (180 degrees)
    rotated_img = cv.rotate(img, cv.ROTATE_180)
    scores_rotated, details_rotated = score_against_templates(rotated_img)

    best_template_normal = max(scores_normal, key=scores_normal.get)
    best_score_normal = scores_normal[best_template_normal]

    best_template_rotated = max(scores_rotated, key=scores_rotated.get)
    best_score_rotated = scores_rotated[best_template_rotated]

    if best_score_rotated > best_score_normal:
        orientation = "upside_down"
        best_template = best_template_rotated
        best_score = best_score_rotated
        scores = scores_rotated
        details = details_rotated
    else:
        orientation = "normal"
        best_template = best_template_normal
        best_score = best_score_normal
        scores = scores_normal
        details = details_normal

    return best_template, best_score, scores, orientation, details


# --------------------
# MAIN
# --------------------
def template_detection_main(
    templates, images: List[Image.Image], out_dir: Path, source_filename
):
    global DEBUG_OUTPUT_DIR
    global DEBUG_STEP_COUNTER

    os.makedirs(out_dir, exist_ok=True)
    DEBUG_OUTPUT_DIR = out_dir

    results_dict = {"source_filename": Path(source_filename).name, "pages": []}

    print("\n[INFO] Recognizing test pages...")
    for i, img in enumerate(images):
        best_template, best_score, all_scores, orientation, details = (
            recognize_page_with_orientation(img, templates)
        )

        sorted_templates = sorted(
            details.items(), key=lambda x: x[1]["inliers_total"], reverse=True
        )
        first_template, first_data = sorted_templates[0]
        # Second best template details (if exists)
        if len(sorted_templates) > 1:
            second_template, second_data = sorted_templates[1]
        else:
            second_template, second_data = None, {"pct_of_best": 0.0}

        page_path_file = f"{Path(source_filename).stem}_page{i+1}.png"
        page_result = {
            "file_page_number": i + 1,
            "form_type": first_template.split("_")[0],  # adjust to your naming
            "source_form_page": (
                int(first_template.split("_")[-1]) + 1
                if "_" in first_template
                else None
            ),
            "roate": 180 if orientation == "upside_down" else 0,
            "confidence_first_template": round(first_data["pct_of_best"] / 100, 2),
            "confidence_second_template": round(second_data["pct_of_best"] / 100, 2),
            "second_source_form_type": (
                second_template.split("_")[0] if second_template else None
            ),
            "page_path": os.path.join(out_dir, page_path_file),
        }
        # save image to out_dir
        img.save(out_dir / page_path_file)

        results_dict["pages"].append(page_result)

    DEBUG_STEP_COUNTER = 0
    return results_dict
