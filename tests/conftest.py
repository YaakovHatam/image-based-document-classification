# tests/conftest.py
import json
from pathlib import Path
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def fixtures_dir(project_root) -> Path:
    p = project_root / "tests" / "fixtures" / "to_process_static"
    if not p.exists() or not any(p.glob("*.pdf")):
        pytest.skip(f"Missing fixtures PDFs under {p}")
    return p


@pytest.fixture
def expected_summary(project_root) -> dict:
    p = project_root / "tests" / "expected_signature_summary.json"
    if not p.is_file():
        pytest.skip(f"Missing expected summary file: {p}")
    return json.loads(p.read_text(encoding="utf-8"))
