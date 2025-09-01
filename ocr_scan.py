import cv2
import pytesseract
from PIL import Image
from pdf2image import convert_from_path
import os
import numpy as np

# --- For Windows Users: If Tesseract or Poppler are not in your PATH ---
# You might need to provide the paths to their executables.
# 1. Tesseract executable path
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# 2. Poppler path (for pdf2image)
# poppler_path = r"C:\path\to\your\poppler\bin"


def process_image_for_ocr(image):
    """
    Takes a PIL Image, converts
     it to an OpenCV format, preprocesses it,
    and returns the processed image.
    """
    # Convert PIL image to OpenCV format (numpy array)
    # Note: pdf2image gives RGB, OpenCV uses BGR, so we convert color space
    open_cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Convert the image to gray scale
    gray_image = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to get a clean binary image
    _, binary_image = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY_INV)

    return binary_image


def extract_numbers_from_pdf(pdf_path):
    """
    Converts a PDF to images and extracts numbers from each page.
    """
    print(f"--- Processing PDF: {os.path.basename(pdf_path)} ---")

    try:
        # On Windows, you might need to pass the poppler_path argument:
        # images = convert_from_path(pdf_path, poppler_path=poppler_path)
        images = convert_from_path(pdf_path)
    except Exception as e:
        print(f"  [ERROR] Could not convert PDF to images. Check Poppler installation.")
        print(f"  Details: {e}")
        return

    # Configure Tesseract to only look for digits
    custom_config = r"-c tessedit_char_whitelist=0123456789,. --psm 6"

    for i, image in enumerate(images):
        page_num = i + 1

        # Preprocess the image to improve OCR accuracy
        processed_image = process_image_for_ocr(image)

        # Use pytesseract to extract numbers
        extracted_text = pytesseract.image_to_string(
            processed_image, config=custom_config
        )

        # Clean up the output: remove empty lines and whitespace
        extracted_numbers = "\n".join(filter(str.strip, extracted_text.splitlines()))

        if extracted_numbers:
            print(f"  Page {page_num}:")
            print(extracted_numbers)
        else:
            print(f"  Page {page_num}: No numbers found.")
    print("-" * 30 + "\n")


def main():
    """
    Main function to find and process all PDFs in the 'ocr' directory.
    """
    # Directory containing the PDF files
    pdf_directory = "tp"

    # Check if the directory exists
    if not os.path.isdir(pdf_directory):
        print(f"Error: Directory '{pdf_directory}' not found.")
        print("Please create it and place your PDF files inside.")
        return

    # Get a list of all PDF files in the directory
    pdf_files = [f for f in os.listdir(pdf_directory) if f.lower().endswith(".pdf")]

    if not pdf_files:
        print(f"No PDF files found in the '{pdf_directory}' directory.")
        return

    print(f"Found {len(pdf_files)} PDF(s) to process.\n")

    # Process each PDF file
    for pdf_file in pdf_files:
        file_path = os.path.join(pdf_directory, pdf_file)
        extract_numbers_from_pdf(file_path)


if __name__ == "__main__":
    main()
