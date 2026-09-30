"""Corrupted/unsupported image matrix (E9) and mode handling."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from manga_ar.errors import ImageLoadError
from manga_ar.io.loader import LoadLimits, load_image_bytes, load_image_file
from tests.fixtures.synth import (
    corrupt_png_crc,
    gradient_rgb,
    jpeg_bytes,
    png_bytes,
    tall_png_header,
)

STRICT = LoadLimits(truncated="strict")
LENIENT = LoadLimits(truncated="lenient")


def _jpeg(arr: np.ndarray | None = None) -> bytes:
    return jpeg_bytes(Image.fromarray(gradient_rgb() if arr is None else arr), quality=95)


@pytest.mark.parametrize(
    ("data", "match"),
    [
        (b"", "empty file"),
        (b"\x00\x01garbage-not-an-image" * 10, "not a decodable image"),
        (b"\x89PNG\r\n\x1a\n" + b"\x00" * 20, "not a decodable image"),
        (b"\xff\xd8\xff\xe0" + b"\x00" * 4, "not a decodable image|corrupt|undecodable"),
    ],
)
def test_undecodable_inputs(data: bytes, match: str) -> None:
    with pytest.raises(ImageLoadError, match=match):
        load_image_bytes(data, "bad.png", LENIENT)


def test_truncated_jpeg_policy() -> None:
    data = _jpeg(gradient_rgb(256, 256))
    cut = data[: len(data) // 2]
    with pytest.raises(ImageLoadError, match=r"truncated|corrupt"):
        load_image_bytes(cut, "cut.jpg", STRICT)
    img = load_image_bytes(cut, "cut.jpg", LENIENT)
    assert img.rgb.shape == (256, 256, 3)
    assert any("best-effort" in w for w in img.warnings)


def test_bad_png_crc_policy() -> None:
    data = corrupt_png_crc(png_bytes(gradient_rgb()))
    with pytest.raises(ImageLoadError):
        load_image_bytes(data, "crc.png", STRICT)
    img = load_image_bytes(data, "crc.png", LENIENT)
    assert img.rgb.shape == (64, 96, 3) and img.warnings


def test_wrong_extension_is_warning_only() -> None:
    img = load_image_bytes(png_bytes(gradient_rgb()), "actually_png.jpg", STRICT)
    assert img.format == "PNG"
    assert any("extension" in w for w in img.warnings)


@pytest.mark.parametrize(("w", "h"), [(1, 1), (8, 500)])
def test_too_small_rejected(w: int, h: int) -> None:
    with pytest.raises(ImageLoadError, match="too small"):
        load_image_bytes(png_bytes(np.zeros((h, w, 3), np.uint8)), "tiny.png")


def test_absurd_dimensions_and_bomb() -> None:
    with pytest.raises(ImageLoadError, match=r"exceeds|bomb"):
        load_image_bytes(tall_png_header(70000, 20), "huge.png")
    with pytest.raises(ImageLoadError, match="pixel limit"):
        load_image_bytes(png_bytes(gradient_rgb(200, 200)), "b.png", LoadLimits(max_pixels=10_000))
    with pytest.raises(ImageLoadError, match=r"bomb|exceeds"):
        load_image_bytes(tall_png_header(60000, 60000), "bomb.png")


def test_animated_gif_first_frame() -> None:
    frames = [Image.new("RGB", (32, 32), c) for c in ((255, 0, 0), (0, 255, 0))]
    buf = io.BytesIO()
    frames[0].save(buf, "GIF", save_all=True, append_images=frames[1:], duration=50, loop=0)
    img = load_image_bytes(buf.getvalue(), "anim.gif")
    assert tuple(img.rgb[5, 5]) == (255, 0, 0)
    assert any("animated" in w for w in img.warnings)


def test_mode_conversions() -> None:
    # RGBA flattened onto white
    rgba = np.zeros((32, 32, 4), np.uint8)
    rgba[..., 0] = 255
    rgba[:16, :, 3] = 255  # top opaque red, bottom transparent
    img = load_image_bytes(png_bytes(rgba, "RGBA"), "a.png")
    assert tuple(img.rgb[2, 2]) == (255, 0, 0) and tuple(img.rgb[30, 2]) == (255, 255, 255)
    # palette with transparency
    pal = Image.new("P", (32, 32), 0)
    pal.putpalette([0, 0, 255] + [0, 0, 0] * 255)
    buf = io.BytesIO()
    pal.save(buf, "PNG", transparency=0)
    assert tuple(load_image_bytes(buf.getvalue(), "p.png").rgb[0, 0]) == (255, 255, 255)
    # LA
    la = Image.new("LA", (32, 32), (40, 255))
    buf = io.BytesIO()
    la.save(buf, "PNG")
    out = load_image_bytes(buf.getvalue(), "la.png")
    assert tuple(out.rgb[0, 0]) == (40, 40, 40) and out.was_grayscale
    # 16-bit grayscale
    g16 = (np.linspace(0, 65535, 32 * 32).reshape(32, 32)).astype(np.uint16)
    buf = io.BytesIO()
    Image.fromarray(g16).save(buf, "PNG")
    out = load_image_bytes(buf.getvalue(), "g16.png")
    assert out.rgb.dtype == np.uint8 and out.rgb.max() == 255 and out.rgb.min() == 0
    assert out.was_grayscale
    # CMYK JPEG
    cmyk = Image.new("CMYK", (32, 32), (0, 255, 255, 0))  # red-ish
    out = load_image_bytes(jpeg_bytes(cmyk, quality=95), "c.jpg")
    assert out.rgb.shape == (32, 32, 3) and out.rgb[..., 0].mean() > 200
    # 1-bit
    bw = Image.new("1", (32, 32), 1)
    buf = io.BytesIO()
    bw.save(buf, "PNG")
    assert load_image_bytes(buf.getvalue(), "bw.png").rgb.min() == 255


def test_exif_orientation_applied() -> None:
    arr = np.zeros((20, 40, 3), np.uint8)
    arr[:, :20] = 255  # left half white
    im = Image.fromarray(arr)
    exif = im.getexif()
    exif[0x0112] = 6  # rotate 90° CW on display
    data = jpeg_bytes(im, quality=95, exif=exif.tobytes())
    out = load_image_bytes(data, "rot.jpg")
    assert out.rgb.shape[:2] == (40, 20)  # rotated
    assert out.rgb[:15].mean() > 200  # former left half is now on top


def test_icc_profile_kept_and_channel_order() -> None:
    from PIL import ImageCms

    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    arr = np.zeros((32, 32, 3), np.uint8)
    arr[..., 0] = 200  # pure red: detects accidental BGR swaps
    out = load_image_bytes(png_bytes(arr, icc_profile=profile), "icc.png")
    assert out.icc_profile == profile
    assert tuple(out.rgb[0, 0]) == (200, 0, 0)


def test_non_ascii_and_long_paths(tmp_path: Path) -> None:
    deep = tmp_path / ("مجلد_عربي_" * 6) / ("漫画フォルダ_" * 6)
    deep.mkdir(parents=True)
    path = deep / "صفحة_ページ_01.png"
    path.write_bytes(png_bytes(gradient_rgb()))
    img = load_image_file(path)
    assert img.rgb.shape == (64, 96, 3) and img.name == "صفحة_ページ_01.png"
    with pytest.raises(ImageLoadError, match="cannot read"):
        load_image_file(deep / "missing.png")
