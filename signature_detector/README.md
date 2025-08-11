# Signature Detector (Classical Computer Vision)

This project implements a signature detector using classical computer vision techniques from OpenCV and Scikit-Image. It is designed to run entirely On-Premise on a CPU, without any pre-trained models or external APIs.

The entire configuration is managed through the `config.py` file.

## Project Structure

-   `run.py`: The main script to execute the detection pipeline.
-   `config.py`: **The central configuration file.** Edit the `CONFIG` dictionary in this file to change settings and point to your input image.
-   `src/signature_detector/`: The core Python package containing the detection logic.
    -   `detector.py`: The main pipeline orchestrator.
    -   `preprocess.py`: Image cleaning, deskewing, and line removal.
    -   `features.py`: Connected component analysis and feature calculation (skeleton, solidity, etc.).
    -   `decision.py`: The rule-based engine that scores components and makes the final decision.
    -   `visualize.py`: Utilities for creating debug images.
-   `requirements.txt`: Required Python libraries.

## Quick Start

1.  **Install dependencies** into a Python 3.9+ virtual environment:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Configure the input images:**
    -   Open the `config.py` file.
    -   In the `io` section, choose **one** of the following methods:
        -   **Method A (Specific Files):** Populate the `input_files` list with the full paths to the images you want to analyze.
        -   **Method B (Entire Directory):** Comment out `input_files` and provide a path to a folder in the `input_directory` variable. The script will process all compatible images in that folder.

3.  **Run the detector:**
    ```bash
    python run.py
    ```
    The results for each image will be printed sequentially to the console. If debugging is enabled, the visual outputs will be saved in the debug directory.

## How It Works

The pipeline is entirely rule-based (heuristic) and follows these steps:
1.  **Load & Preprocess**: The image is loaded, converted to grayscale, and enhanced. Long straight lines (like table borders) are detected and removed.
2.  **Binarize**: An adaptive threshold is applied to create a clean black-and-white (binary) image of the remaining content.
3.  **Find Components**: The code finds all "blobs" or "connected components" in the binary image that are large enough to be considered.
4.  **Enrich Features**: For each component, a rich set of features is calculated (area, solidity, aspect ratio) including advanced ones like its **skeleton**, number of **endpoints**, and **density**.
5.  **Decide**: A scoring algorithm gives points to each component based on its features (e.g., "does it look like handwriting?"). If the total score exceeds a threshold defined in `config.py`, it's classified as a signature.