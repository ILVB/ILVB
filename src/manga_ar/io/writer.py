"""Atomic writes and image encoding (never leaves truncated outputs)."""

from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path

import numpy as np
import numpy.typing as npt
from PIL import Image

from manga_ar.errors import MangaArError
from manga_ar.io.naming import long_path

RgbArray = npt.NDArray[np.uint8]


def _umask() -> int:
    mask = os.umask(0)
    os.umask(mask)
    return mask


# mkstemp creates 0600 files; outputs should get the permissions a plain open() would.
_FILE_MODE = 0o666 & ~_umask()


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write ``data`` to a temp file in the same directory, fsync, then ``os.replace``."""
    target = long_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".tmp-", suffix=target.suffix, dir=target.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        if os.name != "nt":
            tmp.chmod(_FILE_MODE)
        tmp.replace(target)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def atomic_write_text(path: Path, text: str) -> None:
    # "replace": a stray lone surrogate (undecodable file name) must never lose a report
    atomic_write_bytes(path, text.encode("utf-8", "replace"))


def encode_image(
    rgb: RgbArray,
    fmt: str,
    *,
    jpeg_quality: int = 95,
    webp_quality: int = 95,
    icc_profile: bytes | None = None,
    grayscale: bool = False,
    png_compress_level: int = 6,
) -> bytes:
    """Encode an RGB uint8 array. ``grayscale`` stores a single channel when lossless."""
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise MangaArError(f"expected RGB uint8 HxWx3, got {rgb.dtype} {rgb.shape}")
    img = Image.fromarray(rgb, "RGB")
    if grayscale and _is_gray(rgb):
        img = img.convert("L")
    buf = io.BytesIO()
    params: dict[str, object] = {}
    if icc_profile and img.mode == "RGB":
        params["icc_profile"] = icc_profile
    fmt = fmt.lower()
    if fmt == "png":
        img.save(buf, "PNG", optimize=False, compress_level=png_compress_level, **params)
    elif fmt in {"jpg", "jpeg"}:
        img.save(buf, "JPEG", quality=jpeg_quality, optimize=True, subsampling=0, **params)
    elif fmt == "webp":
        img.save(buf, "WEBP", quality=webp_quality, method=4, **params)
    else:
        raise MangaArError(f"unsupported output format {fmt!r}")
    return buf.getvalue()


def _is_gray(rgb: RgbArray) -> bool:
    return bool(
        np.array_equal(rgb[..., 0], rgb[..., 1]) and np.array_equal(rgb[..., 1], rgb[..., 2])
    )


def write_image(path: Path, rgb: RgbArray, fmt: str | None = None, **kwargs: object) -> None:
    fmt = fmt or path.suffix.lstrip(".").lower() or "png"
    atomic_write_bytes(path, encode_image(rgb, fmt, **kwargs))  # type: ignore[arg-type]
