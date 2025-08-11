from glob import glob
import os

from input_pipeline import pdf_to_images
from template_detection import extract_features, get_header_footer, preprocess

TEMPLATE_DIR = "templates_pdf"


def build_template_db():

    template_db = {}
    for path in glob(os.path.join(TEMPLATE_DIR, "*.pdf")):
        name = os.path.splitext(os.path.basename(path))[0]
        images = pdf_to_images(path)

        for i, img in enumerate(images):
            img = preprocess(img)
            header, footer = get_header_footer(img)

            kp_h, des_h = extract_features(header)
            kp_f, des_f = extract_features(footer)
            template_db_name = f"{name}_page_{i}"
            template_db[template_db_name] = {
                "header": {"kp": kp_h, "des": des_h, "kpts": len(kp_h or [])},
                "footer": {"kp": kp_f, "des": des_f, "kpts": len(kp_f or [])},
                "kpts_total": (len(kp_h or []) + len(kp_f or [])),
            }
            print(
                f"[TEMPLATE] Loaded {template_db_name} (header_kpts={len(kp_h or [])}, footer_kpts={len(kp_f or [])})"
            )
    return template_db
