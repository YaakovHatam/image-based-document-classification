# src/signature_detector/features.py

from typing import List, Dict, Tuple
import numpy as np
import cv2
from skimage.morphology import skeletonize, remove_small_objects
from scipy import ndimage as ndi

def find_components(binary_mask: np.ndarray, min_area_ratio=0.0008) -> List[Dict]:
    """
    Finds all connected components in a binary mask that are above a minimum size.
    
    Args:
        binary_mask (np.ndarray): The input black-and-white image.
        min_area_ratio (float): Minimum component area as a fraction of total image area.

    Returns:
        List[Dict]: A list of dictionaries, each describing a component.
    """
    h, w = binary_mask.shape[:2]
    min_area = max(20, int(min_area_ratio * h * w))
    
    nb_components, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    
    comps = []
    for i in range(1, nb_components):
        x, y, ww, hh, area = stats[i]
        if area < min_area:
            continue
        
        # Create a mask for the individual component
        comp_mask = (labels == i).astype('uint8') * 255
        comps.append({
            'label': i,
            'area': int(area),
            'bbox': (int(x), int(y), int(ww), int(hh)),
            'mask': comp_mask
        })
    return comps

def compute_skeleton_and_endpoints(comp_mask: np.ndarray) -> Tuple[np.ndarray, int, int]:
    """
    Computes the skeleton of a component and counts its endpoints.
    
    Returns:
        Tuple containing the skeleton image, its length (pixel count), and endpoint count.
    """
    bool_mask = comp_mask > 0
    if not bool_mask.any():
        return np.zeros_like(comp_mask, dtype=np.uint8), 0, 0

    # Clean up small noise before skeletonization
    cleaned = remove_small_objects(bool_mask, min_size=10)
    if not cleaned.any():
        return np.zeros_like(comp_mask, dtype=np.uint8), 0, 0
    
    sk = skeletonize(cleaned).astype(np.uint8)
    sk_len = int(sk.sum())
    
    # Find endpoints by convolving and finding pixels with value 1 in the skeleton
    # that have only one neighbor (convolution result will be 2).
    kernel = np.ones((3, 3), dtype=np.uint8)
    conv = cv2.filter2D(sk, -1, kernel)
    endpoints = int(np.sum((sk == 1) & (conv == 2)))
    
    return sk, sk_len, endpoints

def enrich_components(comps: List[Dict], image_shape: Tuple[int, int]) -> List[Dict]:
    """
    Calculates advanced features for each component (solidity, skeleton, etc.).
    
    Args:
        comps (List[Dict]): The list of basic components from find_components.
        image_shape (Tuple[int, int]): The shape of the original image.
        
    Returns:
        List[Dict]: The same list, but with new features added to each dictionary.
    """
    h, w = image_shape
    for c in comps:
        x, y, ww, hh = c['bbox']
        mask = c['mask']
        bbox_area = max(1, ww * hh)
        
        # Basic geometric features
        c['extent'] = float(c['area']) / bbox_area
        c['aspect'] = float(ww) / float(hh) if hh > 0 else 0.0
        
        # Solidity (ratio of contour area to its convex hull area)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = contours[0]
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            c['solidity'] = float(c['area']) / hull_area if hull_area > 0 else 0.0
        else:
            c['solidity'] = 0.0
            
        # Skeleton-based features
        sk, sk_len, endpoints = compute_skeleton_and_endpoints(mask)
        c['skeleton'] = sk
        c['skeleton_length'] = sk_len
        c['endpoints'] = endpoints
        c['skeleton_density'] = float(sk_len) / bbox_area

    return comps