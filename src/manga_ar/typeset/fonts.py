"""Font registry with programmatic glyph-coverage validation (A4) and symbol fallback (A5).

Each font's cmap is read with fontTools. For the BASIC render path (reshaper + bidi) a
font must contain the initial/medial/final presentation forms and lam-alef ligatures of
the Arabic alphabet; isolated forms may be substituted by base letters (many OFL fonts
omit them, see DECISIONS SP-A). Fonts lacking contextual forms are RAQM-only.
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass, field
from pathlib import Path

from fontTools.ttLib import TTFont, TTLibError
from PIL import ImageFont

from manga_ar.errors import TypesetError
from manga_ar.logging_setup import get_logger
from manga_ar.resources import fonts_dir

log = get_logger(__name__)

# Arabic letters used by Arabic-language text (hamza … yeh).
ARABIC_LETTERS = tuple(range(0x0621, 0x063B)) + tuple(range(0x0641, 0x064B))
LAM_ALEF_LIGATURES = tuple(range(0xFEF5, 0xFEFD))
ARABIC_PUNCT = (0x060C, 0x061B, 0x061F, 0x066A, 0x066B, 0x066C)
WESTERN_DIGITS = tuple(range(0x30, 0x3A))
INDIC_DIGITS = tuple(range(0x0660, 0x066A))
BASIC_PUNCT = tuple(ord(c) for c in " !.():«»-…\"'")

# key -> (filename, family, licence, role)
VENDORED: dict[str, tuple[str, str, str, str]] = {
    "NotoNaskhArabic": ("NotoNaskhArabic-Variable.ttf", "Noto Naskh Arabic", "OFL-1.1", "arabic"),
    "NotoSansArabic": ("NotoSansArabic-Variable.ttf", "Noto Sans Arabic", "OFL-1.1", "arabic"),
    "Amiri": ("Amiri-Regular.ttf", "Amiri", "OFL-1.1", "arabic"),
    "Cairo": ("Cairo-Variable.ttf", "Cairo", "OFL-1.1", "arabic"),
    "Tajawal": ("Tajawal-Regular.ttf", "Tajawal", "OFL-1.1", "arabic"),
    "TajawalBold": ("Tajawal-Bold.ttf", "Tajawal", "OFL-1.1", "arabic"),
    "Almarai": ("Almarai-Regular.ttf", "Almarai", "OFL-1.1", "arabic"),
    "AlmaraiBold": ("Almarai-Bold.ttf", "Almarai", "OFL-1.1", "arabic"),
    "Changa": ("Changa-Variable.ttf", "Changa", "OFL-1.1", "arabic"),
    "Lalezar": ("Lalezar-Regular.ttf", "Lalezar", "OFL-1.1", "arabic"),
    "BalooBhaijaan2": ("BalooBhaijaan2-Variable.ttf", "Baloo Bhaijaan 2", "OFL-1.1", "arabic"),
    "NotoSansSymbols2": (
        "NotoSansSymbols2-Regular.ttf",
        "Noto Sans Symbols 2",
        "OFL-1.1",
        "symbol",
    ),
    "NotoSansSymbols": ("NotoSansSymbols-Variable.ttf", "Noto Sans Symbols", "OFL-1.1", "symbol"),
}

_SYSTEM_CANDIDATES: dict[str, list[str]] = {
    "DejaVuSans": [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/Library/Fonts/DejaVuSans.ttf",
    ],
    "SystemNotoNaskhArabic": [
        "/usr/share/fonts/truetype/noto/NotoNaskhArabic-Regular.ttf",
        "/usr/share/fonts/opentype/noto/NotoNaskhArabic-Regular.ttf",
    ],
    "Tahoma": ["C:/Windows/Fonts/tahoma.ttf"],
    "Arial": ["C:/Windows/Fonts/arial.ttf", "/Library/Fonts/Arial.ttf"],
    "GeezaPro": ["/System/Library/Fonts/GeezaPro.ttc", "/Library/Fonts/GeezaPro.ttc"],
}


@dataclass(frozen=True)
class Coverage:
    """Missing codepoints per requirement class (empty tuples mean covered)."""

    contextual: tuple[int, ...]
    isolated: tuple[int, ...]
    base_letters: tuple[int, ...]
    ligatures: tuple[int, ...]
    punctuation: tuple[int, ...]
    western_digits: tuple[int, ...]
    indic_digits: tuple[int, ...]

    @property
    def basic_ok(self) -> bool:
        """Usable on the BASIC path (isolated forms may fall back to base letters)."""
        return not self.contextual and not self.base_letters and not self.ligatures

    @property
    def needs_unshaped_isolated(self) -> bool:
        return bool(self.isolated)


@dataclass
class FontInfo:
    key: str
    path: Path
    family: str
    license: str
    role: str  # "arabic" | "symbol"
    source: str  # "vendored" | "system"
    cmap: frozenset[int] = field(repr=False)
    axes: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    coverage: Coverage | None = None
    index: int = 0

    def covers(self, codepoints: set[int] | frozenset[int]) -> bool:
        return codepoints <= self.cmap

    def missing(self, text: str) -> set[int]:
        return {ord(c) for c in text if ord(c) not in self.cmap and not c.isspace()}

    @property
    def basic_ok(self) -> bool:
        return self.coverage is not None and self.coverage.basic_ok

    @property
    def condensed_axes(self) -> dict[str, float] | None:
        """Axis values of the narrowest instance (``wdth`` minimum) if the font has one."""
        if "wdth" in self.axes:
            lo, default, _hi = self.axes["wdth"]
            if lo < default:
                return {"wdth": lo}
        return None


def _contextual_forms() -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Presentation-form codepoints the reshaper emits for ARABIC_LETTERS."""
    from arabic_reshaper.letters import LETTERS_ARABIC

    contextual: set[int] = set()
    isolated: set[int] = set()
    for letter, forms in LETTERS_ARABIC.items():
        if ord(letter) not in ARABIC_LETTERS:
            continue
        iso, ini, med, fin = forms
        if iso:
            isolated.add(ord(iso))
        for f in (ini, med, fin):
            if f:
                contextual.add(ord(f))
    # Alef maksura's initial/medial forms (FBE8/FBE9) never occur in Arabic-language text.
    contextual -= {0xFBE8, 0xFBE9}
    return tuple(sorted(contextual)), tuple(sorted(isolated))


@functools.lru_cache(maxsize=1)
def _required() -> tuple[tuple[int, ...], tuple[int, ...]]:
    return _contextual_forms()


def compute_coverage(cmap: frozenset[int]) -> Coverage:
    contextual, isolated = _required()

    def miss(cps: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(c for c in cps if c not in cmap)

    return Coverage(
        contextual=miss(contextual),
        isolated=miss(isolated),
        base_letters=miss(ARABIC_LETTERS),
        ligatures=miss(LAM_ALEF_LIGATURES),
        punctuation=miss(ARABIC_PUNCT[:3] + BASIC_PUNCT),
        western_digits=miss(WESTERN_DIGITS),
        indic_digits=miss(INDIC_DIGITS),
    )


def read_font(
    path: Path, key: str, family: str, license: str, role: str, source: str
) -> FontInfo | None:
    try:
        tt = TTFont(str(path), lazy=True, fontNumber=0)
        cmap = frozenset(tt.getBestCmap() or {})
        axes: dict[str, tuple[float, float, float]] = {}
        if "fvar" in tt:
            for axis in tt["fvar"].axes:
                axes[axis.axisTag] = (axis.minValue, axis.defaultValue, axis.maxValue)
        tt.close()
    except (TTLibError, OSError, KeyError, AssertionError) as exc:
        log.warning("font %s unreadable: %s", path, exc)
        return None
    info = FontInfo(key, path, family, license, role, source, cmap, axes)
    if role == "arabic":
        info.coverage = compute_coverage(cmap)
    return info


class FontRegistry:
    """Vendored + discovered system fonts with coverage metadata."""

    def __init__(self, directory: Path | None = None, include_system: bool = True) -> None:
        self.directory = directory or fonts_dir()
        self.fonts: dict[str, FontInfo] = {}
        for key, (filename, family, lic, role) in VENDORED.items():
            path = self.directory / filename
            if not path.is_file():
                log.warning("vendored font missing: %s", path)
                continue
            info = read_font(path, key, family, lic, role, "vendored")
            if info is not None:
                self.fonts[key] = info
        if include_system:
            for key, candidates in _SYSTEM_CANDIDATES.items():
                for cand in candidates:
                    p = Path(os.path.expandvars(cand))
                    if p.is_file():
                        info = read_font(
                            p, key, key, "system (see its own licence)", "arabic", "system"
                        )
                        if info is not None:
                            self.fonts[key] = info
                        break

    def get(self, key: str) -> FontInfo:
        try:
            return self.fonts[key]
        except KeyError as exc:
            raise TypesetError(f"unknown font {key!r}; available: {sorted(self.fonts)}") from exc

    def arabic_chain(self, primary: str, fallbacks: list[str], render_path: str) -> list[FontInfo]:
        """Ordered usable Arabic fonts: primary, configured fallbacks, then everything else."""
        others = sorted(k for k, f in self.fonts.items() if f.role == "arabic")
        order = [primary, *fallbacks, *others]
        chain: list[FontInfo] = []
        seen: set[str] = set()
        for key in order:
            if key in seen or key not in self.fonts:
                continue
            seen.add(key)
            info = self.fonts[key]
            if info.role != "arabic":
                continue
            if render_path == "basic" and not info.basic_ok:
                continue
            chain.append(info)
        if not chain:
            raise TypesetError("no Arabic font usable for the selected render path")
        return chain

    def symbol_chain(self, keys: list[str]) -> list[FontInfo]:
        chain = [self.fonts[k] for k in keys if k in self.fonts]
        chain += [f for k, f in self.fonts.items() if f.role == "symbol" and f not in chain]
        if "DejaVuSans" in self.fonts:
            chain.append(self.fonts["DejaVuSans"])
        return chain


@functools.lru_cache(maxsize=512)
def load_font(
    path: str, size: int, raqm: bool = False, axes: tuple[tuple[str, float], ...] = ()
) -> ImageFont.FreeTypeFont:
    """Cached FreeType font object keyed by (face, size, layout, variation)."""
    layout = ImageFont.Layout.RAQM if raqm else ImageFont.Layout.BASIC
    font = ImageFont.truetype(path, size, layout_engine=layout)
    if axes:
        try:
            axis_list = font.get_variation_axes()
            by_tag = dict(axes)
            values = []
            for axis in axis_list:
                name = axis.get("name", b"")
                tag = {"Width": "wdth", "Weight": "wght", "Slant": "slnt"}.get(
                    name.decode() if isinstance(name, bytes) else str(name), ""
                )
                values.append(float(by_tag.get(tag, axis.get("default") or 0)))
            font.set_variation_by_axes(values)
        except (OSError, ValueError) as exc:  # static font or FreeType without MM support
            log.debug("variation %s not applied to %s: %s", axes, path, exc)
    return font
