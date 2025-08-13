# src/signature_detector/preprocess.py

from typing import Tuple
import cv2
import numpy as np

# --- (The other functions clahe_enhance, detect_dominant_hough_angle, rotate_image, deskew, adaptive_binarize remain the same) ---
def clahe_enhance(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)

def detect_dominant_hough_angle(gray: np.ndarray, canny_thresh1=50, canny_thresh2=150,
                                min_line_len=30, max_line_gap=10) -> float:
    edges = cv2.Canny(gray, canny_thresh1, canny_thresh2)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180.0, threshold=60,
                            minLineLength=min_line_len, maxLineGap=max_line_gap)
    if lines is None: return 0.0
    angles = [np.degrees(np.arctan2(y2 - y1, x2 - x1)) for l in lines for x1, y1, x2, y2 in l]
    if not angles: return 0.0
    angles = np.array(angles)
    vx = np.cos(np.radians(angles)).mean()
    vy = np.sin(np.radians(angles)).mean()
    mean_ang = np.degrees(np.arctan2(vy, vx))
    return mean_ang

def rotate_image(img: np.ndarray, angle: float) -> np.ndarray:
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def deskew(gray: np.ndarray) -> Tuple[np.ndarray, float]:
    angle = detect_dominant_hough_angle(gray)
    if abs(angle) > 1.0 and abs(angle) < 45.0:
        rotated = rotate_image(gray, angle)
        return rotated, angle
    return gray, 0.0

def adaptive_binarize(gray: np.ndarray, block_size=31, C=12) -> np.ndarray:
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, block_size, C)

# THIS IS THE NEW, SUPERIOR FUNCTION
def remove_long_lines(binary_mask: np.ndarray, min_length_ratio=0.3, thickness=5) -> np.ndarray:
    """
    Finds and removes long HORIZONTAL and VERTICAL lines using morphological operations,
    which is more reliable for form lines than a general Hough Transform.
    """
    mask = binary_mask.copy()
    h, w = mask.shape[:2]
    
    # --- Remove HORIZONTAL lines ---
    # Use a fraction of the image width as the minimum line length
    hor_min_len = int(w * min_length_ratio)
    # Define a structuring element (kernel) that is a long horizontal line
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (hor_min_len, 1))
    # Use morphological opening to find all long horizontal lines
    detected_hor_lines = cv2.morphologyEx(mask, cv2.MORPH_OPEN, hor_kernel, iterations=2)
    # Subtract the detected lines from the original mask
    mask = cv2.subtract(mask, detected_hor_lines)

    # --- Remove VERTICAL lines ---
    # Use a fraction of the image height as the minimum line length
    ver_min_len = int(h * min_length_ratio)
    # Define a structuring element that is a long vertical line
    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, ver_min_len))
    # Find all long vertical lines
    detected_ver_lines = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ver_kernel, iterations=2)
    # Subtract them from the result
    mask = cv2.subtract(mask, detected_ver_lines)
    
    return mask