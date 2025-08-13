DEBUG_STEP_COUNTER = 0
import json
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

# Ensure config.toml exists and has the required keys
# For demonstration, creating a dummy config if it doesn't exist.
if not os.path.exists("config.toml"):
    default_config = """
[general]
img_width = 800
img_height = 1000
header_ratio = 0.2
footer_ratio = 0.2
pct_vs_template_treshold = 10.0

[debug]
mode = true
output_dir = "debug_output"
"""
    with open("config.toml", "w") as f:
        f.write(default_config)


with open("config.toml", "rb") as f:
    config = tomllib.load(f)

IMG_WIDTH = config["general"]["img_width"]
IMG_HEIGHT = config["general"]["img_height"]
HEADER_RATIO = config["general"]["header_ratio"]
FOOTER_RATIO = config["general"]["footer_ratio"]
PCT_VS_TEMPLATE_TRESHOLD = config["general"]["pct_vs_template_treshold"]


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

    os.makedirs(os.path.join(DEBUG_OUTPUT_DIR, "debug"), exist_ok=True)

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
    # bf.knnMatch can throw an error if descriptors are empty
    try:
        knn = bf.knnMatch(des_t, des_p, k=2)
    except cv.error:
        return []

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


def recognize_page_with_orientation(img: Image.Image, template_db):
    """
    Recognizes a page by checking both normal and 180-degree rotated orientations.
    Returns the best match, its score, orientation, and detailed scores.
    """
    img_normal = preprocess(img)

    def score_against_templates(image_to_test):
        header, footer = get_header_footer(image_to_test)

        kp_h_p, des_h_p = extract_features(header)
        kp_f_p, des_f_p = extract_features(footer)
        page_kpts_total = len(kp_h_p or []) + len(kp_f_p or [])

        scores = {}
        details = {}
        if not template_db:  # Handle empty template database
            return {}, {}

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

        if not details:  # Handle no templates matched
            return {}, {}

        best_inliers = max(v["inliers_total"] for v in details.values())
        for v in details.values():
            v["pct_of_best"] = (
                (v["inliers_total"] / best_inliers * 100.0) if best_inliers > 0 else 0.0
            )

        return scores, details

    # === Score Normal Orientation (0 degrees) ===
    print("[INFO] Checking normal orientation (0 degrees)...")
    scores_normal, details_normal = score_against_templates(img_normal)
    best_template_normal = (
        max(scores_normal, key=scores_normal.get) if scores_normal else None
    )
    best_score_normal = scores_normal.get(best_template_normal, 0)

    # === Score Rotated Orientation (180 degrees) ===
    print("[INFO] Checking rotated orientation (180 degrees)...")
    img_rotated = cv.rotate(img_normal, cv.ROTATE_180)
    save_debug_image("Input_Rotated_180_Degrees", img_rotated)
    scores_rotated, details_rotated = score_against_templates(img_rotated)
    best_template_rotated = (
        max(scores_rotated, key=scores_rotated.get) if scores_rotated else None
    )
    best_score_rotated = scores_rotated.get(best_template_rotated, 0)

    # === Compare and select the best orientation ===
    if best_score_normal >= best_score_rotated:
        print(f"[INFO] Best match is NORMAL orientation. Score: {best_score_normal}")
        if not best_template_normal:  # Handle case where no templates matched at all
            return "None", 0, 0, {}, {}
        return best_template_normal, best_score_normal, 0, scores_normal, details_normal
    else:
        print(f"[INFO] Best match is ROTATED orientation. Score: {best_score_rotated}")
        if not best_template_rotated:  # Handle case where no templates matched at all
            return "None", 0, 0, {}, {}
        return (
            best_template_rotated,
            best_score_rotated,
            180,
            scores_rotated,
            details_rotated,
        )


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

    print("\n[INFO] Recognizing test pages:", source_filename)
    for i, img in enumerate(images):
        DEBUG_STEP_COUNTER = 0  # Reset counter for each page
        (
            best_template,
            best_score,
            orientation,  # <-- Receive orientation
            all_scores,
            details,
        ) = recognize_page_with_orientation(img, templates)

        if not details:  # Check if any details were returned
            print(f"[WARNING] No match found for page {i+1}.")
            sorted_templates = []
            first_data = {"pct_vs_template": 0, "pct_dice": 0}
            first_template = "None"
        else:
            sorted_templates = sorted(
                details.items(), key=lambda x: x[1]["inliers_total"], reverse=True
            )
            first_template, first_data = sorted_templates[0]

        if DEBUG_MODE and sorted_templates:
            with open(out_dir / f"sorted_templates_page_{i+1}.json", "w") as f:
                json.dump(sorted_templates, f, indent=4)

        page_path_file = f"{Path(source_filename).stem}_page{i+1}.png"

        if (
            first_template == "None"
            or first_data["pct_vs_template"] < PCT_VS_TEMPLATE_TRESHOLD
        ):
            page_result = {
                "file_page_number": i + 1,
                "predicted_form_type": "None",
                "predicted_form_page": -1,
                "rotate": orientation,  # <-- Use detected orientation
                "pct_vs_template": 0,
                "pct_dice": 0,
                "page_path": os.path.join(out_dir, page_path_file),
            }
        else:
            page_result = {
                "file_page_number": i + 1,
                "predicted_form_type": first_template.split("_")[0],
                "predicted_form_page": (
                    int(first_template.split("_")[-1])
                    if "_" in first_template and first_template.split("_")[-1].isdigit()
                    else -1
                ),
                "rotate": orientation,  # <-- Use detected orientation
                "pct_vs_template": first_data["pct_vs_template"],
                "pct_dice": first_data["pct_dice"],
                "page_path": os.path.join(out_dir, page_path_file),
            }
        # save image to out_dir
        if orientation == 180:
            img = img.rotate(180, expand=True)
        img.save(out_dir / page_path_file)

        results_dict["pages"].append(page_result)

    return results_dict
