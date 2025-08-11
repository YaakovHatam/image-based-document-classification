# src/signature_detector/detector.py

import cv2
import numpy as np
from typing import Dict, Any

from . import preprocess
from . import features
from . import decision
from . import visualize

def detect_signature(image_path: str, config: Dict[Any, Any]) -> Dict[str, Any]:
    """
    Main pipeline for signature detection.
    Orchestrates preprocessing, feature extraction, and decision making.
    """
    # 1. Load Image
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        return {"found": False, "error": f"Failed to load image at {image_path}"}
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Preprocessing
    enhanced_gray = preprocess.clahe_enhance(gray)
    deskewed_gray, angle = preprocess.deskew(enhanced_gray)
    binary_mask = preprocess.adaptive_binarize(deskewed_gray, **config['preprocess']['binarize'])
    cleaned_mask = preprocess.remove_long_lines(binary_mask, **config['preprocess']['line_removal'])
    
    # 3. Feature Extraction
    components = features.find_components(cleaned_mask, config['features']['min_area_ratio'])
    enriched_components = features.enrich_components(components, gray.shape)
    hough_frac = preprocess.detect_dominant_hough_angle(gray) / 100.0

    # 4. Decision
    result = decision.decide(enriched_components, gray.shape, hough_frac, config['decision'])

    # 5. Visualization (Optional) - THIS IS THE CORRECTED LOGIC
    # Create debug images if the config flag is set, regardless of the result.
    if config.get("debug", {}).get("save_images", False):
        # We need to know which components contributed to the score for visualization
        contributing_comps = result.get('contributing_components', [])
        merged_bbox = result.get('bbox', None)
        
        visualize.save_debug_images(
            image_path=image_path,
            output_dir=config['debug']['output_dir'],
            rgb_img=img,
            binary_mask=cleaned_mask, # Show the mask AFTER line removal
            comps=contributing_comps,
            merged_bbox=merged_bbox
        )
        
    # Clean up result for final JSON output
    if 'contributing_components' in result:
        for comp in result['contributing_components']:
            comp.pop('skeleton', None)
            comp.pop('mask', None)
            
    return result