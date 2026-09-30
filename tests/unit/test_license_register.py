"""G-LIC-1: docs/LICENSES.md names every model, font and dataset (regenerate it with
`python scripts/license_report.py --register` when this fails)."""

from __future__ import annotations

from pathlib import Path

import yaml

from benchmarks.generators.fonts import FONTS
from manga_ar.models.registry import REGISTRY
from manga_ar.typeset.fonts import VENDORED
from tools import check_licenses

ROOT = Path(__file__).resolve().parents[2]


def test_register_lists_models_fonts_and_datasets() -> None:
    text = (ROOT / "docs" / "LICENSES.md").read_text("utf-8")
    names = check_licenses.register_names(text)
    curated = yaml.safe_load((ROOT / "tools" / "license_register.yaml").read_text("utf-8"))
    expected = [*REGISTRY, *VENDORED, *FONTS]
    expected += [e["name"] for section in curated.values() for e in section]
    missing = [n for n in expected if check_licenses.dist_name(n) not in names]
    assert missing == []


def test_register_name_parsing() -> None:
    table = "| Name | Licence |\n|---|---|\n| scikit_image | BSD |\n"
    assert check_licenses.register_names(table) == {"name", "scikit-image"}
