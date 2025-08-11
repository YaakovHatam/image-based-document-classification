# src/signature_detector/decision.py

from typing import List, Dict, Tuple, Any

# --- (merge_bboxes function remains the same) ---
def merge_bboxes(boxes: List[Tuple[int, int, int, int]]) -> Tuple[int, int, int, int]:
    """Merges multiple bounding boxes into a single one that encloses all."""
    if not boxes:
        return (0, 0, 0, 0)
    xs = [b[0] for b in boxes]
    ys = [b[1] for b in boxes]
    x2s = [b[0] + b[2] for b in boxes]
    y2s = [b[1] + b[3] for b in boxes]
    x1, y1 = min(xs), min(ys)
    x2, y2 = max(x2s), max(y2s)
    return (x1, y1, x2 - x1, y2 - y1)


# THIS IS THE REVISED FUNCTION
def decide(comps: List[Dict], image_shape: Tuple[int, int], hough_frac: float, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyzes component features to decide if a signature is present with stricter rules.
    """
    total_score = 0.0
    contributing_components = []

    if not comps:
        return {'found': False, 'score': 0.0, 'reason': 'No significant components found after preprocessing'}

    for c in comps:
        component_score = 0.0
        
        # NEW RULE: Penalize components that are very long and thin (like lines)
        aspect_ratio = c.get('aspect', 1.0)
        if aspect_ratio > 40 or aspect_ratio < 0.025:
             # This component is likely a line fragment, skip it
            continue

        # Rule 1: Skeleton should be reasonably long and dense
        if (c.get('skeleton_length', 0) >= params['min_skeleton_length'] and
            c.get('skeleton_density', 0.0) >= params['min_skeleton_density']):
            component_score += 1.0

        # Rule 2: Handwritten signatures have multiple endpoints. Give a bigger bonus.
        endpoints = c.get('endpoints', 0)
        if endpoints >= params['min_endpoints']:
            # More endpoints are a strong indicator of a signature
            component_score += (endpoints * 0.4) 
        
        # Rule 3: Solidity should not be too high (a solid blob) or too low
        if 0.1 < c.get('solidity', 0.0) < 0.95:
            component_score += 0.6
        
        if component_score > 0:
            c['score'] = component_score
            total_score += component_score
            contributing_components.append(c)

    if hough_frac <= params['max_hough_fraction']:
        total_score += 0.6
    
    threshold = params['score_threshold']['precision'] if params.get('precision_mode', True) else params['score_threshold']['recall']
    
    found = total_score >= threshold

    if found:
        boxes = [c['bbox'] for c in contributing_components]
        merged_bbox = merge_bboxes(boxes)
        return {
            'found': True,
            'score': round(total_score, 2),
            'threshold': threshold,
            'bbox': merged_bbox,
            'components_found': len(contributing_components),
            'contributing_components': contributing_components
        }
    else:
        return {'found': False, 'score': round(total_score, 2), 'threshold': threshold, 'reason': f'Score is below threshold'}