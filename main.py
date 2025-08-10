import cv2 as cv
import os
import numpy as np
from glob import glob
from pdf2image import convert_from_path
import os


def convert_templates(pdf_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for pdf in os.listdir(pdf_dir):
        if pdf.lower().endswith(".pdf"):
            pages = convert_from_path(os.path.join(pdf_dir, pdf), dpi=300)
            out_path = os.path.join(out_dir, pdf.replace(".pdf", ".png"))
            pages[0].save(out_path, "PNG")


# --------------------
# CONFIG
# --------------------
TEMPLATE_DIR = "templates"
TEST_DIR = "test_pages"
IMG_WIDTH = 1024  # normalization width
IMG_HEIGHT = 1448  # normalization height (A4 ~ 1:1.41 ratio)

HEADER_RATIO = 0.15  # top 15% as header
FOOTER_RATIO = 0.15  # bottom 15% as footer

# ORB parameters
orb = cv.ORB_create(nfeatures=500)
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)


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


def extract_features(img):
    """Extract ORB features."""
    kp, des = orb.detectAndCompute(img, None)
    return kp, des


def match_score(des1, des2):
    """Return number of good matches."""
    if des1 is None or des2 is None:
        return 0
    matches = bf.match(des1, des2)
    good = [m for m in matches if m.distance < 40]  # 40 is good threshold for ORB
    return len(good)


# --------------------
# BUILD TEMPLATE DB
# --------------------
def build_template_db():
    template_db = {}
    for path in glob(os.path.join(TEMPLATE_DIR, "*")):
        name = os.path.splitext(os.path.basename(path))[0]
        img = preprocess(path)
        header, footer = get_header_footer(img)
        _, des_header = extract_features(header)
        _, des_footer = extract_features(footer)
        template_db[name] = {"header": des_header, "footer": des_footer}
        print(f"[TEMPLATE] Loaded {name}")
    return template_db


# --------------------
# RECOGNIZE PAGE
# --------------------
def recognize_page(img_path, template_db):
    img = preprocess(img_path)
    header, footer = get_header_footer(img)
    _, des_header = extract_features(header)
    _, des_footer = extract_features(footer)

    best_score = -1
    best_template = None

    for template_name, feats in template_db.items():
        score_header = match_score(des_header, feats["header"])
        score_footer = match_score(des_footer, feats["footer"])
        total_score = score_header + score_footer

        if total_score > best_score:
            best_score = total_score
            best_template = template_name

    return best_template, best_score


# --------------------
# MAIN
# --------------------
if __name__ == "__main__":
    
    print("[INFO] Building template database...")
    template_db = build_template_db()

    print("\n[INFO] Recognizing test pages...")
    for path in glob(os.path.join(TEST_DIR, "*")):
        template_name, score = recognize_page(path, template_db)
        print(f"{os.path.basename(path)} -> {template_name} (score={score})")
