# src/signature_detector/visualize.py

import cv2
import numpy as np
import os
from typing import List, Tuple, Dict

def overlay_components(rgb_img: np.ndarray, comps: List[Dict], merged_bbox: Tuple[int,int,int,int]=None) -> np.ndarray:
    """Draws bounding boxes and skeletons on an image for debugging."""
    out = rgb_img.copy()
    
    # Draw individual component boxes (green) and skeletons (blue)
    for c in comps:
        x, y, w, h = c['bbox']
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 1) # Green, thin box
        
        skeleton = c.get('skeleton')
        if skeleton is not None and skeleton.any():
            # Color the skeleton pixels blue for visibility
            out[skeleton > 0] = [255, 0, 0] # BGR for blue
            
    # Draw the main merged bounding box (red and thick)
    if merged_bbox and merged_bbox[2] > 0 and merged_bbox[3] > 0:
        x, y, w, h = merged_bbox
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 0, 255), 2) # Red, thick box

    return out

def save_debug_images(
    image_path: str,
    output_dir: str,
    rgb_img: np.ndarray,
    binary_mask: np.ndarray,
    comps: List[Dict],
    merged_bbox: Tuple[int,int,int,int]=None
):
    """Saves visualization images to the specified debug directory."""
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Save the final binary mask used for component analysis
    mask_path = os.path.join(output_dir, f"{base_name}_debug_mask.png")
    cv2.imwrite(mask_path, binary_mask)
    
    # Save the overlay image showing detected components
    overlay_img = overlay_components(rgb_img, comps, merged_bbox)
    overlay_path = os.path.join(output_dir, f"{base_name}_debug_overlay.png")
    cv2.imwrite(overlay_path, overlay_img)