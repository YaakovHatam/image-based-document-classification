from glob import glob
import json
import os
from pathlib import Path
from input_pipeline import pdf_to_images
from signature_cutting import process_folder_by_type_prefix
from template_detection import template_detection_main
from templates_db import build_template_db
from rotate_script import process_page_images_from_json

if __name__ == "__main__":

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

    include_map = {"1301-2022": [1], "1385": [1]}
    base_input = Path("out")              # contains subfolders like 'customer1', 'customer2', ...
    base_output = Path("output_assets")   # outputs go under 'output_assets/<customer>/'
    include_map = {"1301-2022": [1], "1385": [1]}

    for customer_dir in base_input.iterdir():
        if not customer_dir.is_dir():
            continue

        json_path = customer_dir / "results.json"  # adjust if your JSON has a different name
        if not json_path.exists():
            print(f"Skipping {customer_dir.name}: no results.json found")
            continue

        out_dir = base_output / customer_dir.name
        process_page_images_from_json(
            json_file_path=json_path,
            output_folder=out_dir,
            include_map=include_map,
        )

    config_path = "./config.toml"
    process_folder_by_type_prefix(
        input_dir="./output_assets",
        output_dir="./out/signature",
        config_path=config_path,
    )
