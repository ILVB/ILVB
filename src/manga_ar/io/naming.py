"""Natural sorting, output naming and path helpers."""

from __future__ import annotations

import os
import re
from collections.abc import Iterable
from pathlib import Path, PurePosixPath

IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif"})
ARCHIVE_EXTENSIONS = frozenset({".cbz", ".zip"})
_NUM = re.compile(r"(\d+)")


def natural_key(name: str) -> tuple[tuple[int, int | str], ...]:
    """Sort key so that ``page2`` < ``page10``; case-insensitive, stable across OSes."""
    parts = _NUM.split(str(name).replace("\\", "/").casefold())
    key: list[tuple[int, int | str]] = []
    for part in parts:
        if not part:
            continue
        key.append((0, int(part)) if part.isdigit() else (1, part))
    return tuple(key)


def natural_sorted(names: Iterable[str]) -> list[str]:
    return sorted(names, key=natural_key)


def is_image_name(name: str) -> bool:
    return PurePosixPath(name.replace("\\", "/")).suffix.lower() in IMAGE_EXTENSIONS


def is_archive_name(name: str) -> bool:
    return Path(name).suffix.lower() in ARCHIVE_EXTENSIONS


def output_name(stem: str, fmt: str, suffix: str = "_ar") -> str:
    """``page01`` + ``png`` → ``page01_ar.png`` (``jpg`` stays ``jpg``)."""
    ext = {"jpeg": "jpg"}.get(fmt, fmt)
    return f"{stem}{suffix}.{ext}"


def sidecar_name(stem: str, suffix: str = "_ar") -> str:
    return f"{stem}{suffix}.mangaar.json"


def long_path(path: Path) -> Path:
    """On Windows, prefix long absolute paths with ``\\\\?\\`` so >260-char paths work."""
    if os.name != "nt":
        return path
    resolved = str(path.resolve())
    if len(resolved) >= 240 and not resolved.startswith("\\\\?\\"):
        return Path("\\\\?\\" + resolved)
    return path
