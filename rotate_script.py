from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List
import shutil
from PIL import Image  # pip install Pillow


def process_page_images_from_json(
    json_file_path: str | Path,
    output_folder: str | Path,
    include_map: Dict[str, List[int]],
) -> List[Path]:
    """
    Process PNG page images listed in a JSON file:
      - Load JSON file.
      - Filter pages by (form_type, source_form_page) according to include_map.
      - For each matched item, open the PNG at 'page_path' (full path to the file),
        rotate clockwise by 'roate' degrees if > 0, and save to output_folder.
      - When saving, filename will be '<form_type>_<original_filename>'.
      - If roate == 0, copy the file as-is to output_folder.

    Args:
        json_file_path: Path to the JSON file containing the "pages" list.
        output_folder: Folder where processed/copy images will be written.
        include_map: Mapping of form_type -> list of source_form_page numbers to include.

    Returns:
        List of Paths to the written images.
    """
    json_file_path = Path(json_file_path)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    if not json_file_path.is_file():
        raise FileNotFoundError(f"JSON file not found: {json_file_path}")

    # Load JSON
    with open(json_file_path, "r", encoding="utf-8") as f:
        file_info = json.load(f)

    written: List[Path] = []
    for entry in file_info.get("pages", []):
        form_type = entry.get("predicted_form_type")
        src_page = entry.get("predicted_form_page")

        # Only process if form_type is in include_map and source_form_page is included
        if form_type not in include_map or src_page not in set(include_map[form_type]):
            print(
                f"{form_type} not in {include_map} or {src_page} not in include_map[form_type]"
            )
            continue

        page_path = entry.get("page_path")
        if not page_path:
            continue

        src_img_path = Path(page_path)
        if not src_img_path.is_file():
            raise FileNotFoundError(f"Image not found: {src_img_path}")

        # Rotation degrees
        degrees = entry.get("rotate", 0)
        try:
            deg = int(degrees) % 360
        except (TypeError, ValueError):
            deg = 0

        # Output filename: <form_type>_<original_filename>
        dst_filename = f"{form_type}_{src_img_path.name}"
        dst_img_path = output_folder / dst_filename

        if deg == 0:
            shutil.copy2(src_img_path, dst_img_path)
        else:
            with Image.open(src_img_path) as im:
                rotated = im.rotate(-deg, expand=True)  # clockwise rotation
                rotated.save(dst_img_path)

        written.append(dst_img_path)

    return written


# ----------------------------
# Process ALL customers under 'out/'
# ----------------------------
# if __name__ == "__main__":
#     base_input = Path("out")              # contains subfolders like 'customer1', 'customer2', ...
#     base_output = Path("output_assets")   # outputs go under 'output_assets/<customer>/'
#     include_map = {"1301-2022": [1], "1385": [1]}

#     for customer_dir in base_input.iterdir():
#         if not customer_dir.is_dir():
#             continue

#         json_path = customer_dir / "results.json"  # adjust if your JSON has a different name
#         if not json_path.exists():
#             print(f"Skipping {customer_dir.name}: no results.json found")
#             continue

#         out_dir = base_output / customer_dir.name
#         written = process_page_images_from_json(
#             json_file_path=json_path,
#             output_folder=out_dir,
#             include_map=include_map,
#         )
#         print(f"[{customer_dir.name}] Saved {len(written)} files to {out_dir}")
#         for p in written:
#             print("  ", p)
