"""Pinned, checksum-verified fonts for the generator (all SIL OFL-1.1, never committed)."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
_GF = "https://raw.githubusercontent.com/google/fonts/main/ofl/"
_NOTO = "https://raw.githubusercontent.com/notofonts/noto-cjk/main/Sans/SubsetOTF/"


@dataclass(frozen=True)
class FontSpec:
    url: str
    sha256: str
    license: str = "OFL-1.1"

    @property
    def filename(self) -> str:
        return self.url.rsplit("/", 1)[-1]


FONTS: dict[str, FontSpec] = {
    "noto_jp": FontSpec(
        _NOTO + "JP/NotoSansJP-Regular.otf",
        "dff723ba59d57d136764a04b9b2d03205544f7cd785a711442d6d2d085ac5073",
    ),
    "noto_sc": FontSpec(
        _NOTO + "SC/NotoSansSC-Regular.otf",
        "faa6c9df652116dde789d351359f3d7e5d2285a2b2a1f04a2d7244df706d5ea9",
    ),
    "noto_kr": FontSpec(
        _NOTO + "KR/NotoSansKR-Regular.otf",
        "69975a0ac8472717870aefeab0a4d52739308d90856b9955313b2ad5e0148d68",
    ),
    "comic_neue": FontSpec(
        _GF + "comicneue/ComicNeue-Regular.ttf",
        "a0ee5a37c8b27c4db0700137d928598b1e23b0089e1546a8961909176b779360",
    ),
    "comic_neue_bold": FontSpec(
        _GF + "comicneue/ComicNeue-Bold.ttf",
        "3e7e5fccfd7e0788f317b43312151c1bd5cf058c9697a8d83eac3939050bd61e",
    ),
    "bangers": FontSpec(
        _GF + "bangers/Bangers-Regular.ttf",
        "4160a7311de9342674cce9160cde9fcbb30f48190397d86ff1b70b455af65824",
    ),
    "dela_gothic": FontSpec(
        _GF + "delagothicone/DelaGothicOne-Regular.ttf",
        "4ff87a0965f1b0505e5a2c58424bc6ad3cff27e56a82f21c2fc9d6b0e3857ee2",
    ),
    "zcool_kuaile": FontSpec(
        _GF + "zcoolkuaile/ZCOOLKuaiLe-Regular.ttf",
        "812a6fc1fe54b6d73a419245c32dfeba8aa33104d5be90d1cf6af082007cb71d",
    ),
    "black_han_sans": FontSpec(
        _GF + "blackhansans/BlackHanSans-Regular.ttf",
        "31960809284026681774a8e52dc19ebcad26cf69b0ad9d560f288296fbb52739",
    ),
}

# Plain lettering and display (stylized/SFX) faces per source language.
TEXT_FONT = {"ja": "noto_jp", "zh": "noto_sc", "ko": "noto_kr", "en": "comic_neue"}
DISPLAY_FONT = {"ja": "dela_gothic", "zh": "zcool_kuaile", "ko": "black_han_sans", "en": "bangers"}


class FontError(RuntimeError):
    pass


def cache_dir() -> Path:
    base = Path(os.environ.get("MANGAAR_CACHE_DIR", ROOT / ".cache"))
    return base / "bench_fonts"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@cache
def font_path(key: str) -> Path:
    """Local path of a verified font; downloads it once when missing."""
    spec = FONTS[key]
    target = cache_dir() / spec.filename
    if not target.is_file() or _sha256(target) != spec.sha256:
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            response = requests.get(spec.url, timeout=60)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise FontError(f"cannot download {spec.url}: {exc}") from exc
        tmp = target.with_suffix(".part")
        tmp.write_bytes(response.content)
        if _sha256(tmp) != spec.sha256:
            tmp.unlink()
            raise FontError(f"checksum mismatch for {spec.url}")
        tmp.replace(target)
    return target
