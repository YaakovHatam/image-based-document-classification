from glob import glob
import json
import os
from pathlib import Path
import shutil
from datetime import datetime
from typing import Dict, List

import tomllib  # Python 3.11+

# your modules
from input_pipeline import pdf_to_images
from sig_detector import sig_detector_main, load_signature_thresholds, load_sign_labels
from signature_cutting import process_folder_by_type_prefix  # keep if you use it
from template_detection import template_detection_main
from templates_db import build_template_db
from files_selector import process_page_images_from_json


def _load_include_map_from_toml(toml_path: Path) -> Dict[str, List[int]]:
    if not toml_path.is_file():
        raise FileNotFoundError(
            f"Missing {toml_path}. Please add an [include_map] section.")

    with open(toml_path, "rb") as f:
        cfg = tomllib.load(f)

    if "include_map" not in cfg:
        raise KeyError(
            "config.toml must contain a top-level [include_map] table.")

    raw = cfg["include_map"]
    if not isinstance(raw, dict):
        raise TypeError(
            "[include_map] must be a table of form_type = [page_numbers].")

    include_map: Dict[str, List[int]] = {}
    for k, v in raw.items():
        if not isinstance(v, list):
            raise TypeError(
                f"[include_map] '{k}' must be a list of page numbers.")
        include_map[str(k)] = [int(x) for x in v]
    return include_map


def run_pipeline(to_process_dir: Path, out_root: Path, config_toml: Path = Path("config.toml")) -> Path:
    """
    End-to-end run on 'to_process_dir', write under 'out_root' (clean), config from 'config_toml'.
    Returns path to summary JSON: out_root / 'signature_summary_all.json'
    """
    # Clean output for a fresh run
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    templates = build_template_db()

    # 1) PDFs -> images + template detection
    for path in glob(str(to_process_dir / "*.pdf")):
        pdf_to_process = Path(path)
        images = pdf_to_images(pdf_to_process)
        out_dir = out_root / pdf_to_process.stem
        results_dict = template_detection_main(
            templates, images, out_dir, pdf_to_process.name)
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "results.json", "w", encoding="utf-8") as f:
            json.dump(results_dict, f, indent=4, ensure_ascii=False)

    # 2) include_map
    include_map = _load_include_map_from_toml(config_toml)

    all_rows: List[dict] = []

    # thresholds & labels (for summary)
    thresholds = load_signature_thresholds()
    labels = load_sign_labels()

    # 3) signature detection across all folders
    for pdf_folder in sorted(out_root.iterdir()):
        if not pdf_folder.is_dir():
            continue

        json_path = pdf_folder / "results.json"
        if not json_path.exists():
            print(f"Skipping {pdf_folder.name}: no results.json found")
            continue

        files = process_page_images_from_json(
            json_file_path=json_path, include_map=include_map)

        results = sig_detector_main(
            files=files, thresh=thresholds["yes_lower"], debug=True) or []

        for r in results:
            all_rows.append({
                "pdf_folder": pdf_folder.name,
                "file": r.get("file", ""),
                "doc_type": r.get("doc_type", ""),
                # "none" | "review" | "present" | "error"
                "sign_level": r.get("sign_level", ""),
                "score": r.get("score"),
                # <<--- NEW: bubble up any error message
                "error": r.get("error")
            })

    # 4) Write consolidated JSON(s)
    total = len(all_rows)

    # Tally labels (dynamic, respects custom names)
    by_sign_level: Dict[str, int] = {}
    for r in all_rows:
        lvl = r.get("sign_level") or "unknown"
        by_sign_level[lvl] = by_sign_level.get(lvl, 0) + 1

    present_label = labels["present"]
    present_count = by_sign_level.get(present_label, 0)
    detection_rate = round(present_count / total, 4) if total else 0.0

    summary = {
        "overall": {
            "total_files": total,
            "present_count": present_count,
            "detection_rate": detection_rate,
            "thresholds": thresholds,
            "labels": labels,
            "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        },
        "by_sign_level": by_sign_level,
        "files": all_rows
    }

    summary_path = out_root / "signature_summary_all.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # NEW: dedicated errors list (easy to scan)
    errors = [
        {
            "pdf_folder": r["pdf_folder"],
            "file": r["file"],
            "doc_type": r["doc_type"],
            "error": r.get("error")
        }
        for r in all_rows
        if r.get("sign_level") == "error" or r.get("error")
    ]
    (out_root / "signature_errors.json").write_text(
        json.dumps(errors, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print("\n========================")
    print(f"Present (\"{present_label}\") rate: {detection_rate:.2%}")
    print(f"Summary: {summary_path}")
    if errors:
        print(
            f"Errors: {len(errors)} (see {out_root/'signature_errors.json'})")
        for e in errors[:10]:
            print(f" - {e['pdf_folder']}/{e['file']}: {e.get('error')}")
        if len(errors) > 10:
            print(" - ...")
    print("========================\n")

    return summary_path


if __name__ == "__main__":
    run_pipeline(Path("to_process"), Path("out"), Path("config.toml"))
