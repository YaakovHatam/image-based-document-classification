# up/template_detection.py

import json
from pathlib import Path
import cv2 as cv
import os
import numpy as np
from PIL import Image
from typing import List
import tomllib

DEBUG_STEP_COUNTER = 0

akaze = cv.AKAZE_create()
# --------------------
# CONFIG
# --------------------


with open("config.toml", "rb") as f:
    config = tomllib.load(f)

    IMG_WIDTH = config["general"]["img_width"]
    IMG_HEIGHT = config["general"]["img_height"]
    # Ratios are now loaded per-template, so we remove them from here
    PCT_VS_TEMPLATE_TRESHOLD = config["general"]["pct_vs_template_treshold"]

    DEBUG_MODE = config["debug"]["mode"]
    DEBUG_OUTPUT_DIR = config["debug"]["output_dir"]
    DEBUG_STEP_COUNTER = 0

# ORB parameters
orb = cv.ORB_create(nfeatures=500)
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)


def save_debug_image(step_name: str, img, prefix: str = "debug", img_idx=None):
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
    if img_idx is not None:
        filename = f"img{img_idx}_{DEBUG_STEP_COUNTER:03d}_{prefix}_{step_name}.png"

    cv.imwrite(os.path.join(DEBUG_OUTPUT_DIR, "debug", filename), img)


# --------------------
# IMAGE HELPERS
# --------------------
def compute_skew_and_rotate(img: np.ndarray) -> np.ndarray:
    """Detects and corrects small skew angles in an image."""
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.bitwise_not(gray)
    thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    if coords.size == 0:
        return img
    angle = cv.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv.warpAffine(
        img, M, (w, h), flags=cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE
    )
    print(f"[INFO] Detected skew angle: {angle:.2f} degrees. Correcting...")
    return rotated


def remove_color_noise(img: np.ndarray) -> np.ndarray:
    """Reduce color noise present in scanned documents."""
    if img.ndim == 3:
        return cv.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    return img


def resize_img(img):
    if isinstance(img, Image.Image):
        img = np.array(img)
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
    return cv.resize(img, (IMG_WIDTH, IMG_HEIGHT))


def preprocess(img: Image.Image, img_idx=None):
    """Convert to OpenCV format, deskew, denoise, grayscale and binarize."""
    if isinstance(img, Image.Image):
        img = np.array(img)
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
    save_debug_image("Original", img, img_idx)
    img = remove_color_noise(img)
    save_debug_image("Denoised", img, img_idx)
    img = compute_skew_and_rotate(img)
    save_debug_image("Deskewed", img, img_idx)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    save_debug_image("Gray", gray, img_idx)
    gray = cv.resize(gray, (IMG_WIDTH, IMG_HEIGHT))
    save_debug_image("Resized", gray, img_idx)
    gray = cv.GaussianBlur(gray, (3, 3), 0)
    save_debug_image("Blurred", gray, img_idx)
    bin_img = cv.adaptiveThreshold(
        gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 15, 10
    )
    kernel = np.ones((3, 3), np.uint8)
    bin_img = cv.morphologyEx(bin_img, cv.MORPH_OPEN, kernel, iterations=1)
    save_debug_image("Binarized", bin_img, img_idx)
    return bin_img


def get_header_footer(img, header_ratio, footer_ratio):
    """Extract header and footer regions using provided ratios."""
    h = img.shape[0]
    header = img[0 : int(h * header_ratio), :]
    footer = img[int(h * (1 - footer_ratio)) :, :]
    save_debug_image("Header", header)
    save_debug_image("Footer", footer)
    return header, footer


def geometric_inliers(kp_t, kp_p, good_matches, ransac_thresh=3.0):
    if len(good_matches) < 4:
        return 0
    src = np.float32([kp_t[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst = np.float32([kp_p[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    H, mask = cv.findHomography(src, dst, cv.RANSAC, ransac_thresh)
    return int(mask.sum()) if mask is not None else 0


def knn_ratio_matches(des_t, des_p, ratio=0.75):
    if des_t is None or des_p is None:
        return []
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
    kp, des = akaze.detectAndCompute(img, None)
    return kp, des


# --------------------
# MATCHING/SCORING
# --------------------


def match_scores_pair(kp_p, des_p, kp_t, des_t):
    good = knn_ratio_matches(des_t, des_p, ratio=0.75)
    inliers = geometric_inliers(kp_t, kp_p, good, ransac_thresh=3.0)
    return len(good), inliers


def recognize_page_with_orientation(img: Image.Image, template_db, img_idx=None):
    img_normal = preprocess(img, img_idx)

    def score_against_templates(image_to_test):
        scores = {}
        details = {}
        if not template_db:
            return {}, {}

        for template_name, feats in template_db.items():
            # Use the template-specific ratios to extract header/footer from the page
            header_ratio = feats["header_ratio"]
            footer_ratio = feats["footer_ratio"]
            header_p, footer_p = get_header_footer(
                image_to_test, header_ratio, footer_ratio
            )

            # Extract features from the page's header/footer slices
            kp_h_p, des_h_p = extract_features(header_p)
            kp_f_p, des_f_p = extract_features(footer_p)
            page_kpts_total = len(kp_h_p or []) + len(kp_f_p or [])

            # Match page regions against this specific template's stored features
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

            scores[template_name] = pct_dice
            details[template_name] = {
                "raw_total": int(raw_total),
                "inliers_total": int(inl_total),
                "tmpl_kpts_total": int(tmpl_kpts_total),
                "page_kpts_total": int(page_kpts_total),
                "pct_vs_template": float(pct_vs_template),
                "pct_dice": float(pct_dice),
            }

        if not details:
            return {}, {}

        best_inliers = max(v["inliers_total"] for v in details.values())
        for v in details.values():
            v["pct_of_best"] = (
                (v["inliers_total"] / best_inliers * 100.0) if best_inliers > 0 else 0.0
            )

        return scores, details

    orientations = {
        0: img_normal,
        90: cv.rotate(img_normal, cv.ROTATE_90_CLOCKWISE),
        180: cv.rotate(img_normal, cv.ROTATE_180),
        270: cv.rotate(img_normal, cv.ROTATE_90_COUNTERCLOCKWISE),
    }
    best_orientation, best_score, best_template = 0, -1, "None"
    best_scores_dict, best_details_dict = {}, {}

    for angle, img_rotated in orientations.items():
        print(f"[INFO] Checking orientation: {angle} degrees...")
        if DEBUG_MODE:
            save_debug_image(f"Input_Rotated_{angle}_Degrees", img_rotated, img_idx)

        scores, details = score_against_templates(img_rotated)
        if not scores:
            continue

        current_best_template = max(scores, key=scores.get)
        current_best_score = scores[current_best_template]

        if current_best_score > best_score:
            best_score = current_best_score
            best_orientation = angle
            best_template = current_best_template
            best_scores_dict = scores
            best_details_dict = details

    print(
        f"[INFO] Best match is {best_template} at {best_orientation} degrees with score {best_score}"
    )

    return (
        best_template,
        best_score,
        best_orientation,
        best_scores_dict,
        best_details_dict,
    )


# --------------------
# MAIN
# --------------------
def template_detection_main(
    templates, images: List[Image.Image], out_dir: Path, source_filename
):
    global DEBUG_OUTPUT_DIR, DEBUG_STEP_COUNTER
    os.makedirs(out_dir, exist_ok=True)
    DEBUG_OUTPUT_DIR = out_dir
    results_dict = {"source_filename": Path(source_filename).name, "pages": []}

    print("\n[INFO] Recognizing test pages:", source_filename)
    for i, img in enumerate(images):
        DEBUG_STEP_COUNTER = 0
        (
            best_template,
            best_score,
            orientation,
            all_scores,
            details,
        ) = recognize_page_with_orientation(img, templates, i)

        if not details:
            print(f"[WARNING] No match found for page {i+1}.")
            sorted_templates, first_data, first_template = (
                [],
                {"pct_vs_template": 0, "pct_dice": 0},
                "None",
            )
        else:
            sorted_templates = sorted(
                details.items(), key=lambda x: x[1]["inliers_total"], reverse=True
            )
            first_template, first_data = sorted_templates[0]

        if DEBUG_MODE and sorted_templates:
            with open(out_dir / f"sorted_templates_page_{i+1}.json", "w") as f:
                json.dump(sorted_templates, f, indent=4)

        page_path_file = f"{Path(source_filename).stem}_page{i+1}_resized.png"

        if (
            first_template == "None"
            or first_data["pct_vs_template"] < PCT_VS_TEMPLATE_TRESHOLD
        ):
            page_result = {
                "file_page_number": i + 1,
                "predicted_form_type": "None",
                "predicted_form_page": -1,
                "rotate": orientation,
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
                "rotate": orientation,
                "pct_vs_template": first_data["pct_vs_template"],
                "pct_dice": first_data["pct_dice"],
                "page_path": os.path.join(out_dir, page_path_file),
            }

        if orientation != 0:
            img = img.rotate(orientation, expand=True)

        img_to_save = resize_img(img)
        cv.imwrite(os.path.join(out_dir, page_path_file), img_to_save)
        results_dict["pages"].append(page_result)

    return results_dict
