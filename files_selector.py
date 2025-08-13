import json
from pathlib import Path
from typing import Dict, List, Tuple


def process_page_images_from_json(
    json_file_path: str | Path,
    include_map: Dict[str, List[int]],
) -> List[Tuple[Path, str]]:
    files = []
    json_file_path = Path(json_file_path)

    if not json_file_path.is_file():
        raise FileNotFoundError(f"JSON file not found: {json_file_path}")

    # Load JSON
    with open(json_file_path, "r", encoding="utf-8") as f:
        file_info = json.load(f)

    for entry in file_info.get("pages", []):
        form_type = entry.get("predicted_form_type")
        form_page = entry.get("predicted_form_page")

        # Only process if form_type is in include_map and source_form_page is included
        if form_type not in include_map or form_page not in set(include_map[form_type]):
            print(
                f"{form_type} not in {include_map} or {form_page} not in include_map[form_type]"
            )
            continue

        page_path = entry.get("page_path")
        if not page_path:
            continue

        page_path = Path(page_path)
        if not page_path.is_file():
            raise FileNotFoundError(f"Image not found: {page_path}")

        files.append((page_path, form_type))

    return files
