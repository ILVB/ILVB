"""Robust image loading into the canonical RGB uint8 HxWx3 format.

Handles EXIF orientation, palette/alpha/CMYK/16-bit/grayscale modes, animated images,
truncated files (policy-driven), decompression bombs and non-ASCII paths. Files are read
as bytes (never ``cv2.imread`` on a path), so Windows non-ASCII paths are safe.
"""

from __future__ import annotations

import hashlib
import io
import threading
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
from PIL import Image, ImageFile, ImageOps, UnidentifiedImageError

from manga_ar.errors import ImageLoadError
from manga_ar.io.naming import display_name, long_path
from manga_ar.logging_setup import get_logger

log = get_logger(__name__)

RgbArray = npt.NDArray[np.uint8]
_GRAY_MODES = frozenset({"1", "L", "LA", "I", "I;16", "I;16B", "I;16L", "I;16N", "F", "La"})
_TRUNCATED_LOCK = threading.Lock()
_EXT_FORMATS = {
    ".png": "PNG",
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".webp": "WEBP",
    ".bmp": "BMP",
    ".tif": "TIFF",
    ".tiff": "TIFF",
    ".gif": "GIF",
}


@dataclass
class LoadLimits:
    max_pixels: int = 178_956_970
    min_side: int = 16
    max_side: int = 65_500
    truncated: Literal["strict", "lenient"] = "lenient"


@dataclass
class LoadedImage:
    """A decoded page plus provenance needed for faithful export."""

    rgb: RgbArray
    name: str
    sha256: str
    format: str
    original_mode: str
    was_grayscale: bool
    icc_profile: bytes | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def height(self) -> int:
        return int(self.rgb.shape[0])

    @property
    def width(self) -> int:
        return int(self.rgb.shape[1])


def read_bytes(path: Path) -> bytes:
    try:
        return long_path(path).read_bytes()
    except OSError as exc:
        raise ImageLoadError(f"cannot read {path}: {exc}") from exc


def load_image_file(path: Path, limits: LoadLimits | None = None) -> LoadedImage:
    """Load an image from disk (bytes first, so any path encoding works)."""
    return load_image_bytes(read_bytes(path), display_name(path.name), limits)


def load_image_bytes(data: bytes, name: str, limits: LoadLimits | None = None) -> LoadedImage:
    """Decode ``data`` into RGB uint8; raise :class:`ImageLoadError` on any failure."""
    limits = limits or LoadLimits()
    if not data:
        raise ImageLoadError(f"{name}: empty file (0 bytes)")
    sha = hashlib.sha256(data).hexdigest()
    try:
        return _load(data, name, limits, sha)
    except ImageLoadError:
        raise
    except MemoryError as exc:
        raise ImageLoadError(f"{name}: not enough memory to decode this image") from exc
    except Exception as exc:
        # TypeError/KeyError/IndexError/struct.error/EOFError… on crafted files (fuzzing).
        raise ImageLoadError(f"{name}: undecodable image ({type(exc).__name__}: {exc})") from exc


def _load(data: bytes, name: str, limits: LoadLimits, sha: str) -> LoadedImage:
    notes: list[str] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", Image.DecompressionBombWarning)
        try:
            img = Image.open(io.BytesIO(data))
        except Image.DecompressionBombError as exc:
            raise ImageLoadError(f"{name}: decompression bomb rejected ({exc})") from exc
        except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
            raise ImageLoadError(f"{name}: not a decodable image ({exc})") from exc
        fmt = str(img.format or "UNKNOWN")
        width, height = img.size
        _check_dimensions(name, width, height, limits)
        expected = _EXT_FORMATS.get(Path(name).suffix.lower())
        if expected and expected != fmt:
            notes.append(f"extension suggests {expected} but content is {fmt}")
        if getattr(img, "is_animated", False):
            notes.append(f"animated image ({getattr(img, 'n_frames', '?')} frames): using frame 1")
            img.seek(0)
        _decode(img, name, limits, notes)
        original_mode = img.mode
        icc = img.info.get("icc_profile") if original_mode in {"RGB", "RGBA", "P"} else None
        rgb = _to_rgb(_apply_exif(img, name, notes))
    for note in notes:
        log.warning("%s: %s", name, note)
    return LoadedImage(
        rgb=rgb,
        name=name,
        sha256=sha,
        format=fmt,
        original_mode=original_mode,
        was_grayscale=original_mode in _GRAY_MODES,
        icc_profile=icc if isinstance(icc, bytes) else None,
        warnings=notes,
    )


def _check_dimensions(name: str, width: int, height: int, limits: LoadLimits) -> None:
    if width <= 0 or height <= 0:
        raise ImageLoadError(f"{name}: zero-size image {width}x{height}")
    if min(width, height) < limits.min_side:
        raise ImageLoadError(f"{name}: image too small {width}x{height} (min {limits.min_side})")
    if max(width, height) > limits.max_side:
        raise ImageLoadError(f"{name}: side exceeds {limits.max_side}px ({width}x{height})")
    if width * height > limits.max_pixels:
        raise ImageLoadError(
            f"{name}: {width}x{height} exceeds the pixel limit {limits.max_pixels} "
            "(decompression-bomb guard)"
        )


def _decode(img: Image.Image, name: str, limits: LoadLimits, notes: list[str]) -> None:
    try:
        img.load()
        return
    except Exception as exc:
        if limits.truncated == "strict":
            raise ImageLoadError(f"{name}: corrupt or truncated image ({exc})") from exc
        first_error = exc
    # Lenient: retry with Pillow's truncated-image tolerance. The flag is global, so it is
    # toggled under a lock and restored immediately.
    with _TRUNCATED_LOCK:
        previous = ImageFile.LOAD_TRUNCATED_IMAGES
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        try:
            img.load()
        except Exception as exc:
            raise ImageLoadError(f"{name}: undecodable even in lenient mode ({exc})") from exc
        finally:
            ImageFile.LOAD_TRUNCATED_IMAGES = previous
    notes.append(f"corrupt/truncated data decoded best-effort ({first_error})")


def _apply_exif(img: Image.Image, name: str, notes: list[str]) -> Image.Image:
    try:
        out = ImageOps.exif_transpose(img)
    except (OSError, ValueError, SyntaxError, KeyError, TypeError) as exc:
        notes.append(f"EXIF orientation ignored ({exc})")
        return img
    return out if out is not None else img


def _to_rgb(img: Image.Image) -> RgbArray:
    mode = img.mode
    if mode in {"I", "I;16", "I;16B", "I;16L", "I;16N", "F"}:
        arr = np.asarray(img, dtype=np.float64)
        hi = float(arr.max()) if arr.size else 0.0
        # 16-bit data spans 0..65535 (÷257 maps it exactly onto 0..255); float images are
        # usually 0..1; anything else is already on an 8-bit scale.
        if hi > 255.0:
            arr = arr / 257.0
        elif mode == "F" and hi <= 1.0:
            arr = arr * 255.0
        gray8 = np.rint(np.clip(arr, 0, 255)).astype(np.uint8)
        return np.ascontiguousarray(np.repeat(gray8[..., None], 3, axis=2))
    if mode == "P":
        img = img.convert("RGBA" if "transparency" in img.info else "RGB")
    elif mode == "PA" or mode in {"LA", "La"} or mode == "RGBa":
        img = img.convert("RGBA")
    if img.mode == "RGBA":
        background = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(background, img).convert("RGB")
    elif img.mode != "RGB":
        img = img.convert("RGB")
    return np.ascontiguousarray(np.asarray(img, dtype=np.uint8))
