from pathlib import Path
from typing import List

import fitz  # PyMuPDF
from PIL import Image


def pdf_to_images(pdf_path: str) -> List[Image.Image]:
    """Convert a PDF into portrait-oriented PIL Images.

    Args:
        pdf_path: Path to the input PDF file.

    Returns:
        A list of PIL Image objects (one per page).
    """
    pdf_path = Path(pdf_path)
    doc = fitz.open(pdf_path)
    images: List[Image.Image] = []

    for page_number, page in enumerate(doc, start=1):
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        # Ensure portrait orientation
        if img.width > img.height:
            img = img.rotate(90, expand=True)

        images.append(img)

    doc.close()
    return images
