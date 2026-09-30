"""Locations of packaged resources (fonts, default config, demo data)."""

from __future__ import annotations

from importlib import resources
from pathlib import Path


def package_dir() -> Path:
    return Path(str(resources.files("manga_ar")))


def fonts_dir() -> Path:
    """Vendored OFL fonts (shipped inside the wheel)."""
    return package_dir() / "assets" / "fonts"


def data_dir() -> Path:
    return package_dir() / "data"
