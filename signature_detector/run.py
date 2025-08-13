# run.py
import json
import os
from typing import List
from src.signature_detector.detector import detect_signature
from config import CONFIG  # Import the configuration directly

def get_image_paths_from_config(config: dict) -> List[str]:
    """
    Gets the list of image paths to process from the configuration.
    It prioritizes 'input_files' list, then 'input_directory'.
    """
    io_config = config.get("io", {})
    
    # Method 1: Process a specific list of files
    if "input_files" in io_config:
        print("Processing mode: Using 'input_files' list from config.")
        return io_config["input_files"]
        
    # Method 2: Process all images in a directory
    if "input_directory" in io_config:
        print("Processing mode: Using 'input_directory' from config.")
        dir_path = io_config["input_directory"]
        if not os.path.isdir(dir_path):
            print(f"Error: The specified directory does not exist: {dir_path}")
            return []
            
        supported_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
        image_paths = [
            os.path.join(dir_path, f)
            for f in os.listdir(dir_path)
            if os.path.splitext(f)[1].lower() in supported_extensions
        ]
        print(f"Found {len(image_paths)} images in the directory.")
        return image_paths
        
    return []

def main():
    """
    Main execution script. Reads config, gets a list of images,
    and runs the detection pipeline for each one.
    """
    config = CONFIG
    
    # 1. Get the list of images to process from the config file
    image_paths = get_image_paths_from_config(config)
    
    if not image_paths:
        print("\nFATAL: No images to process. Please check your 'config.py'.")
        print("You must specify either 'input_files' or 'input_directory' in the 'io' section.")
        return

    # 2. Ensure debug directory exists if enabled
    if config.get("debug", {}).get("save_images", False):
        output_dir = config.get("debug", {}).get("output_dir", "./debug_output")
        os.makedirs(output_dir, exist_ok=True)
        print(f"Debug mode is ON. Output will be saved to '{output_dir}'")
    else:
        print("Debug mode is OFF.")

    # 3. Loop through all images and process them
    print(f"\n--- STARTING BATCH PROCESSING OF {len(image_paths)} IMAGES ---")
    all_results = {}
    for image_path in image_paths:
        print(f"\nProcessing: {os.path.basename(image_path)}")
        
        if not os.path.exists(image_path):
            print(f"  -> SKIPPED: File not found at '{image_path}'")
            continue
            
        result = detect_signature(image_path, config)
        
        # Print the result for the current file
        print(json.dumps(result, indent=2, ensure_ascii=False, default=lambda x: str(x)))
        all_results[os.path.basename(image_path)] = result

    print("\n--- BATCH PROCESSING COMPLETE ---")
    # Optional: You can save all_results to a single JSON file if needed later.

if __name__ == "__main__":
    main()