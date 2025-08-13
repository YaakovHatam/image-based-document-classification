# -*- coding: utf-8 -*-
import os
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import sys


def create_output_directory(base_dir="out/scan"):
    """
    Ensures the output directory exists. If not, it creates it.
    """
    try:
        os.makedirs(base_dir, exist_ok=True)
        print(f"Output directory '{base_dir}' is ready.")
    except OSError as e:
        print(f"Error creating directory {base_dir}: {e}")
        sys.exit(1)  # Exit if we can't create the output folder


def ocr_pdf_file(pdf_path, output_dir):
    """
    Performs OCR on a single PDF file and saves the text output.

    Args:
        pdf_path (str): The full path to the PDF file.
        output_dir (str): The directory where the output .txt file will be saved.
    """
    # Get the base filename without the extension
    base_filename = os.path.basename(pdf_path)
    filename_without_ext = os.path.splitext(base_filename)[0]
    output_txt_path = os.path.join(output_dir, f"{filename_without_ext}.txt")

    print(f"\n--- Processing: {base_filename} ---")

    try:
        # Convert all pages of the PDF to a list of PIL images
        # For Windows, you might need to specify the poppler path:
        # poppler_path=r"C:\path\to\poppler-xx\bin"
        pages = convert_from_path(pdf_path)

        full_text = ""
        # Process each page individually
        for i, page_image in enumerate(pages):
            print(f"  - Reading page {i + 1}/{len(pages)}...")

            # Use Tesseract to do OCR on the image, specifying Hebrew language
            # The 'heb' language code is for Hebrew.
            try:
                text = pytesseract.image_to_string(page_image, lang="heb")
                full_text += f"\n\n--- Page {i + 1} ---\n\n{text}"
            except pytesseract.TesseractError as e:
                print(f"    Error during OCR on page {i+1}: {e}")
                full_text += (
                    f"\n\n--- Page {i + 1} (Error) ---\n\nCould not extract text.\n"
                )

        # Write the extracted text from all pages to the output file
        with open(output_txt_path, "w", encoding="utf-8") as f:
            f.write(full_text)

        print(f"Success! Text saved to: {output_txt_path}")

    except Exception as e:
        print(f"Could not process file {base_filename}. Reason: {e}")


def main():
    """
    Main function to run the OCR process on a folder.
    """
    # --- For Windows Users: Uncomment the line below if Tesseract is not in your system's PATH ---
    # This is the default installation path for Tesseract 5 on Windows.
    # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

    # Get the folder path from the user
    input_folder = input(
        "Please enter the path to the folder containing your PDF files: "
    )

    if not os.path.isdir(input_folder):
        print(f"Error: The path '{input_folder}' is not a valid directory.")
        return

    output_folder = "./out/scan"
    create_output_directory(output_folder)

    # Loop through all files in the specified folder
    for filename in os.listdir(input_folder):
        # Check if the file is a PDF
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(input_folder, filename)
            ocr_pdf_file(file_path, output_folder)

    print("\n--- All PDF files have been processed. ---")


if __name__ == "__main__":
    main()
