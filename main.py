from glob import glob
import json
import os
from pathlib import Path
import shutil
from input_pipeline import pdf_to_images
from sig_detector import sig_detector_main
from signature_cutting import process_folder_by_type_prefix
from template_detection import template_detection_main
from templates_db import build_template_db
from files_selector import process_page_images_from_json

if __name__ == "__main__":
    if os.path.exists("./out"):
        shutil.rmtree("./out")

    templates = build_template_db()

    for path in glob(os.path.join("to_process", "*.pdf")):
        pdf_to_process = Path(path)
        images = pdf_to_images(pdf_to_process)
        out_dir = Path("./out") / pdf_to_process.stem
        results_dict = template_detection_main(
            templates, images, out_dir, pdf_to_process.name
        )
        results_file = out_dir / "results.json"
        with open(results_file, "w") as f:
            json.dump(results_dict, f, indent=4)

    config_path = "./config.toml"

    include_map = {
        "1385": [1],
        "1301-2022": [1],
        "1344-2020": [1],
        "1344-2021": [1],
        "1344-2022": [1],
        "1344-2023": [1],
    }
    base_input = Path("out")  # contains subfolders like 'customer1', 'customer2', ...

    for customer_dir in base_input.iterdir():
        if not customer_dir.is_dir():
            continue

        json_path = customer_dir / "results.json"

        if not json_path.exists():
            print(f"Skipping {customer_dir.name}: no results.json found")
            continue

        files = process_page_images_from_json(
            json_file_path=json_path,
            include_map=include_map,
        )
        sig_detector_main(
            files=files,
            thresh=0.40,
        )
        print("\n")
        print("=" * 20)
