from glob import glob
import os
import tomllib

from input_pipeline import pdf_to_images
from template_detection import extract_features, get_header_footer, preprocess

TEMPLATE_DIR = "templates_pdf"


def build_template_db():
    """
    Builds a database of template features, loading custom header/footer
    ratios for each template from config.toml.
    """
    # Load configuration to get template-specific ratios
    with open("config.toml", "rb") as f:
        config = tomllib.load(f)

    template_configs = config.get("templates", {})
    default_hr = template_configs.get("default_header_ratio", 0.25)
    default_fr = template_configs.get("default_footer_ratio", 0.25)
    specific_settings = template_configs.get("settings", {})

    template_db = {}
    for path in glob(os.path.join(TEMPLATE_DIR, "*.pdf")):
        name = os.path.splitext(os.path.basename(path))[0]

        # Get specific ratios for this template, or use defaults
        template_settings = specific_settings.get(name, {})
        header_ratio = template_settings.get("header_ratio", default_hr)
        footer_ratio = template_settings.get("footer_ratio", default_fr)

        images = pdf_to_images(path)

        for i, img in enumerate(images):
            img = preprocess(img)
            # Use the specific ratios for this template
            header, footer = get_header_footer(img, header_ratio, footer_ratio)

            kp_h, des_h = extract_features(header)
            kp_f, des_f = extract_features(footer)
            template_db_name = f"{name}_page_{i+1}"

            template_db[template_db_name] = {
                "header": {"kp": kp_h, "des": des_h, "kpts": len(kp_h or [])},
                "footer": {"kp": kp_f, "des": des_f, "kpts": len(kp_f or [])},
                "kpts_total": (len(kp_h or []) + len(kp_f or [])),
                # Store the ratios with the template
                "header_ratio": header_ratio,
                "footer_ratio": footer_ratio,
            }
            print(
                f"[TEMPLATE] Loaded {template_db_name} (h_ratio={header_ratio:.2f}, f_ratio={footer_ratio:.2f}, header_kpts={len(kp_h or [])}, footer_kpts={len(kp_f or [])})"
            )
    return template_db
