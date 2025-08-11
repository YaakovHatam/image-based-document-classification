# config.py
# This file contains the central configuration for the signature detector.

CONFIG = {
    # ===================================================================
    #                      INPUT/OUTPUT CONFIGURATION
    # ===================================================================
    "io": {
        "input_files": [
            r"..\out\signature\1301_page1 - sign_signature.png",
            r"..\out\signature\1301_page1_signature.png",
            r"..\out\signature\1385_page_sign_signature.png",
            r"..\out\signature\1385_page4_signature.png",
        ],
    },

    # ===================================================================
    #                      DEBUGGING & VISUALIZATION
    # ===================================================================
    "debug": {
        "save_images": True,
        "output_dir": "./debug_output",
    },

    # ===================================================================
    #                      PIPELINE PARAMETERS - "Back to Basics"
    # ===================================================================

    "preprocess": {
        "binarize": {"block_size": 31, "C": 12},
        # Parameters for the new, more precise line removal function
        "line_removal": {"min_length_ratio": 0.3, "thickness": 5},
    },

    "features": {
        # A more reasonable area ratio to catch smaller signatures
        "min_area_ratio": 0.001,
    },

    "decision": {
        "precision_mode": True,
        # A reasonable starting threshold
        "score_threshold": {"precision": 2.5, "recall": 1.8},
        # Lowering requirements to a sensible baseline
        "min_skeleton_length": 30,
        "min_skeleton_density": 0.015,
        "min_endpoints": 3,
        "max_hough_fraction": 0.4,
    },
}