from __future__ import annotations

from pathlib import Path
from typing import List

import fitz  # PyMuPDF
from PIL import Image


def pdf_to_images(pdf_path: str, output_dir: str) -> List[str]:
    """Split a PDF into portrait-oriented page images.

    Args:
        pdf_path: Path to the input PDF file.
        output_dir: Directory where page images will be stored. It is
            created if it does not already exist.

    Returns:
        A list of file paths to the generated page images in order.
    """
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    image_paths: List[str] = []

    for page_number, page in enumerate(doc, start=1):
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        if img.width > img.height:
            img = img.rotate(90, expand=True)

        filename = output_dir / f"page{page_number}.png"
        img.save(filename)
        image_paths.append(str(filename))

    doc.close()
    return image_paths
