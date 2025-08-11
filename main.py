from glob import glob
import json
import os
from pathlib import Path
from input_pipeline import pdf_to_images
from template_detection import template_detection_main
from templates_db import build_template_db


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
