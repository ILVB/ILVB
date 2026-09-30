"""Test-facing entry point for the synthetic fixture generator.

The generator lives in ``manga_ar.synth`` because ``manga-arabic demo`` needs it at
runtime; this module re-exports it and adds corrupted-file builders used by the
loader/pipeline tests.
"""

from __future__ import annotations

import io
import struct
import zlib

import numpy as np
from PIL import Image

from manga_ar.synth import (  # noqa: F401 - re-exported for tests
    DEMO_TRANSLATIONS,
    TEXTS,
    GtRegion,
    SynthPage,
    adversarial_gutter_page,
    basic_page,
    demo_page,
    furigana_page,
    symbols_page,
    variety_page,
    webtoon_strip,
)


def png_bytes(arr: np.ndarray, mode: str | None = None, **params: object) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr, mode).save(buf, "PNG", **params)  # type: ignore[arg-type]
    return buf.getvalue()


def jpeg_bytes(img: Image.Image, **params: object) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "JPEG", **params)  # type: ignore[arg-type]
    return buf.getvalue()


def gradient_rgb(h: int = 64, w: int = 96) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    return np.stack([xx * 255 // w, yy * 255 // h, (xx + yy) % 256], -1).astype(np.uint8)


def corrupt_png_crc(data: bytes) -> bytes:
    """Flip one byte inside the first IDAT chunk's data so its CRC no longer matches."""
    idx = data.index(b"IDAT")
    mutable = bytearray(data)
    mutable[idx + 8] ^= 0xFF
    return bytes(mutable)


def tall_png_header(width: int, height: int) -> bytes:
    """A syntactically valid PNG header claiming huge dimensions (bomb probe)."""

    def chunk(tag: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + tag
            + payload
            + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    idat = zlib.compress(b"\x00" * 16)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")
