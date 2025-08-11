import cv2 as cv
import os
import numpy as np
from glob import glob

# --------------------
# CONFIG
# --------------------
TEMPLATE_DIR = "templates"
TEST_DIR = "test_pages"
OUT_DIR = "test_pages"

IMG_WIDTH = 1024  # normalization width
IMG_HEIGHT = 1448  # normalization height (A4 ~ 1:1.41 ratio)

HEADER_RATIO = 0.15  # top 15% as header
FOOTER_RATIO = 0.15  # bottom 15% as footer

# ORB parameters
orb = cv.ORB_create(nfeatures=500)
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)


# --------------------
# IMAGE HELPERS
# --------------------
def preprocess(img_path):
    """Load, grayscale, resize, and binarize."""
    img = cv.imread(img_path)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.resize(gray, (IMG_WIDTH, IMG_HEIGHT))
    gray = cv.GaussianBlur(gray, (3, 3), 0)
    bin_img = cv.adaptiveThreshold(
        gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 15, 10
    )
    return bin_img


def get_header_footer(img):
    """Extract header and footer regions."""
    h = img.shape[0]
    header = img[0 : int(h * HEADER_RATIO), :]
    footer = img[int(h * (1 - FOOTER_RATIO)) :, :]
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
def build_template_db():
    template_db = {}
    for path in glob(os.path.join(TEMPLATE_DIR, "*")):
        name = os.path.splitext(os.path.basename(path))[0]
        img = preprocess(path)
        header, footer = get_header_footer(img)

        kp_h, des_h = extract_features(header)
        kp_f, des_f = extract_features(footer)

        template_db[name] = {
            "header": {"kp": kp_h, "des": des_h, "kpts": len(kp_h or [])},
            "footer": {"kp": kp_f, "des": des_f, "kpts": len(kp_f or [])},
            "kpts_total": (len(kp_h or []) + len(kp_f or [])),
        }
        print(f"[TEMPLATE] Loaded {name} (header_kpts={len(kp_h or [])}, footer_kpts={len(kp_f or [])})")
    return template_db


# --------------------
# RECOGNIZE PAGE
# --------------------
def recognize_page(img_path, template_db):
    img = preprocess(img_path)
    header, footer = get_header_footer(img)

    kp_h_p, des_h_p = extract_features(header)
    kp_f_p, des_f_p = extract_features(footer)
    page_kpts_total = len(kp_h_p or []) + len(kp_f_p or [])

    scores = {}            # template_name -> primary score (inliers total)
    details = {}           # template_name -> dict with raw/inliers/percentages

    for template_name, feats in template_db.items():
        # Header region
        raw_h, inl_h = match_scores_pair(kp_h_p, des_h_p, feats["header"]["kp"], feats["header"]["des"])
        # Footer region
        raw_f, inl_f = match_scores_pair(kp_f_p, des_f_p, feats["footer"]["kp"], feats["footer"]["des"])

        raw_total = raw_h + raw_f
        inl_total = inl_h + inl_f
        tmpl_kpts_total = feats["kpts_total"]

        # Normalized percentages
        pct_vs_template = (inl_total / tmpl_kpts_total * 100.0) if tmpl_kpts_total > 0 else 0.0
        pct_dice = (2.0 * inl_total / max(tmpl_kpts_total + page_kpts_total, 1) * 100.0)

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
        v["pct_of_best"] = (v["inliers_total"] / best_inliers * 100.0) if best_inliers > 0 else 0.0

    return best_template, best_score, scores, details


def recognize_page_with_orientation(img_path, template_db):
    img = preprocess(img_path)

    def score_against_templates(image):
        header, footer = get_header_footer(image)
        kp_h_p, des_h_p = extract_features(header)
        kp_f_p, des_f_p = extract_features(footer)
        page_kpts_total = len(kp_h_p or []) + len(kp_f_p or [])

        scores = {}
        details = {}
        for template_name, feats in template_db.items():
            raw_h, inl_h = match_scores_pair(kp_h_p, des_h_p, feats["header"]["kp"], feats["header"]["des"])
            raw_f, inl_f = match_scores_pair(kp_f_p, des_f_p, feats["footer"]["kp"], feats["footer"]["des"])

            raw_total = raw_h + raw_f
            inl_total = inl_h + inl_f
            tmpl_kpts_total = feats["kpts_total"]

            pct_vs_template = (inl_total / tmpl_kpts_total * 100.0) if tmpl_kpts_total > 0 else 0.0
            pct_dice = (2.0 * inl_total / max(tmpl_kpts_total + page_kpts_total, 1) * 100.0)

            scores[template_name] = inl_total
            details[template_name] = {
                "raw_total": int(raw_total),
                "inliers_total": int(inl_total),
                "tmpl_kpts_total": int(tmpl_kpts_total),
                "page_kpts_total": int(page_kpts_total),
                "pct_vs_template": float(pct_vs_template),
                "pct_dice": float(pct_dice),
            }

        best_inliers = max(v["inliers_total"] for v in details.values()) if details else 0
        for v in details.values():
            v["pct_of_best"] = (v["inliers_total"] / best_inliers * 100.0) if best_inliers > 0 else 0.0

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
def template_detection_main(test_dir, out_dir):
    results_file = os.path.join(out_dir, "template_detection_results.txt")

    print("[INFO] Building template database...")
    template_db = build_template_db()
    with open(results_file, "w", encoding="utf-8") as f:
        print("\n[INFO] Recognizing test pages...")
        for path in glob(os.path.join(test_dir, "*.png")):
            best_template, best_score, all_scores, orientation, details = (
                recognize_page_with_orientation(path, template_db)
            )

            d_best = details[best_template]
            print(f"{os.path.basename(path)} -> {best_template}  "
                  f"(inliers={d_best['inliers_total']}, raw={d_best['raw_total']}, "
                  f"pct_vs_tmpl={d_best['pct_vs_template']:.2f}%, "
                  f"dice={d_best['pct_dice']:.2f}%, "
                  f"rel_best={d_best['pct_of_best']:.2f}%)")
            print(f"Orientation: {orientation}")

            print("All templates (sorted by inliers):")
            for template, score in sorted(all_scores.items(), key=lambda x: x[1], reverse=True):
                dt = details[template]
                print(f"  {template:25s} | inliers={dt['inliers_total']:4d} | raw={dt['raw_total']:4d} "
                      f"| pct_vs_tmpl={dt['pct_vs_template']:6.2f}% | dice={dt['pct_dice']:6.2f}% "
                      f"| rel_best={dt['pct_of_best']:6.2f}%")

            # --- Write to results file ---
            f.write(f"Source: {os.path.basename(path)}\n")
            f.write(f"  Orientation: {orientation}\n")
            f.write(f"  Best match: {best_template}\n")
            f.write(f"    inliers={d_best['inliers_total']}, raw={d_best['raw_total']}, "
                    f"pct_vs_tmpl={d_best['pct_vs_template']:.2f}%, "
                    f"dice={d_best['pct_dice']:.2f}%, "
                    f"rel_best={d_best['pct_of_best']:.2f}%\n")
            f.write("  All templates (sorted by inliers):\n")
            for template, score in sorted(all_scores.items(), key=lambda x: x[1], reverse=True):
                dt = details[template]
                f.write(f"    {template}: inliers={dt['inliers_total']}, raw={dt['raw_total']}, "
                        f"pct_vs_tmpl={dt['pct_vs_template']:.2f}%, dice={dt['pct_dice']:.2f}%, "
                        f"rel_best={dt['pct_of_best']:.2f}%\n")
            f.write("\n" + "-" * 60 + "\n\n")

        print(f"\n[INFO] Results saved to: {results_file}")


if __name__ == "__main__":
    template_detection_main(TEST_DIR, OUT_DIR)
