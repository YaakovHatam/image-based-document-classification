import json
import shutil


from main import run_pipeline


def test_end_to_end_on_fixtures_creates_clean_out_and_matches_expected(
    tmp_path, fixtures_dir, expected_summary, project_root
):
    config_toml = project_root / "config.toml"

    # copy stable PDFs into tmp working dir
    to_process = tmp_path / "to_process"
    to_process.mkdir(parents=True, exist_ok=True)
    for pdf in fixtures_dir.glob("*.pdf"):
        shutil.copy2(pdf, to_process / pdf.name)

    out_dir = tmp_path / "out"

    # Run the pipeline
    summary_path = run_pipeline(
        to_process_dir=to_process, out_root=out_dir, config_toml=config_toml
    )

    # Assert outputs exist
    assert out_dir.is_dir()
    assert summary_path.is_file()

    actual = json.loads(summary_path.read_text(encoding="utf-8"))

    # -- structure sanity --
    assert "files" in actual and isinstance(actual["files"], list)
    assert "overall" in actual and "by_sign_level" in actual

    # -- errors visibility + assertions --
    actual_errors = [
        {
            "pdf_folder": r.get("pdf_folder"),
            "file": r.get("file"),
            "doc_type": r.get("doc_type"),
            "error": r.get("error"),
        }
        for r in actual["files"]
        if r.get("sign_level") == "error" or r.get("error")
    ]

    if "errors" in expected_summary:
        # compare exactly (order-insensitive)
        def key(e):
            return (e.get("pdf_folder", ""), e.get("file", ""))

        assert sorted(actual_errors, key=key) == sorted(
            expected_summary["errors"], key=key
        ), (
            f"Errors mismatch.\nExpected:\n{json.dumps(expected_summary['errors'], indent=2, ensure_ascii=False)}\n"
            f"Actual:\n{json.dumps(actual_errors, indent=2, ensure_ascii=False)}"
        )
    else:
        # by default, we expect NO errors
        assert not actual_errors, "Unexpected errors:\n" + json.dumps(
            actual_errors, indent=2, ensure_ascii=False
        )

    # -- per-file label comparison --
    def index_files(rows):
        rows_sorted = sorted(
            rows, key=lambda r: (r.get("pdf_folder", ""), r.get("file", ""))
        )
        return [
            (r.get("pdf_folder"), r.get("file"), r.get("doc_type"), r.get("sign_level"))
            for r in rows_sorted
        ]

    exp_rows = index_files(expected_summary["files"])
    act_rows = index_files(actual["files"])
    assert exp_rows == act_rows, (
        "Per-file results differ.\n"
        f"Expected:\n{json.dumps(exp_rows, indent=2, ensure_ascii=False)}\n"
        f"Actual:\n{json.dumps(act_rows, indent=2, ensure_ascii=False)}"
    )

    # -- aggregated counts --
    act_by_label = {}
    for _, _, _, lvl in act_rows:
        act_by_label[lvl] = act_by_label.get(lvl, 0) + 1

    if "by_sign_level" in expected_summary:
        exp_by = expected_summary["by_sign_level"]
    else:
        exp_by = {}
        for _, _, _, lvl in exp_rows:
            exp_by[lvl] = exp_by.get(lvl, 0) + 1
    assert (
        act_by_label == exp_by
    ), f"by_sign_level mismatch: expected {exp_by}, got {act_by_label}"

    # -- detection_rate math --
    labels = actual["overall"]["labels"]
    present_label = labels["present"]
    present_count = act_by_label.get(present_label, 0)
    total = len(act_rows)
    expected_rate = round(present_count / total, 4) if total else 0.0
    assert (
        abs(expected_rate - actual["overall"]["detection_rate"]) < 1e-9
    ), "detection_rate mismatch with counts"
