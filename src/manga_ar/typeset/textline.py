"""Shaped-line measurement and drawing with per-run font fallback (A4, A5, A6).

A visual line is split into runs: characters the region's Arabic font covers, and
symbol runs (♡ ♪ ☆ …) drawn with the symbol fonts. Every emitted codepoint is checked
against the cmap of the font that will draw it; uncovered characters fall back through
the registry and are never drawn as tofu.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass, field

from PIL import ImageDraw, ImageFont

from manga_ar.errors import TypesetError
from manga_ar.logging_setup import get_logger
from manga_ar.typeset.arabic_text import shape_line, strip_harakat
from manga_ar.typeset.fonts import FontInfo, FontRegistry, load_font

log = get_logger(__name__)
Run = tuple[str, FontInfo]


@dataclass
class FontChoice:
    """Fonts used for one region: the Arabic face plus fallbacks for other characters."""

    arabic: FontInfo
    fallbacks: list[FontInfo]  # symbol fonts first, then other registry faces
    axes: tuple[tuple[str, float], ...] = ()
    missing: set[str] = field(default_factory=set)

    @property
    def unshaped_isolated(self) -> bool:
        cov = self.arabic.coverage
        return bool(cov is not None and cov.needs_unshaped_isolated)

    def font(self, info: FontInfo, size: int) -> ImageFont.FreeTypeFont:
        axes = self.axes if info is self.arabic else ()
        return load_font(str(info.path), size, False, axes)

    def shape(self, logical: str, harakat: bool = True) -> str:
        return shape_line(logical, self.unshaped_isolated, harakat)

    def runs(self, visual: str) -> list[Run]:
        """Split a visual string into maximal runs drawable by a single font."""
        runs: list[Run] = []
        for ch in visual:
            face = self._face_for(ch)
            if face is None:
                continue
            if runs and runs[-1][1] is face:
                runs[-1] = (runs[-1][0] + ch, face)
            else:
                runs.append((ch, face))
        return runs

    def _face_for(self, ch: str) -> FontInfo | None:
        cp = ord(ch)
        if ch.isspace() or cp in self.arabic.cmap:
            return self.arabic
        for face in self.fallbacks:
            if cp in face.cmap:
                return face
        if ch not in self.missing:
            self.missing.add(ch)
            log.warning("no font covers U+%04X (%r); character dropped", cp, ch)
        return None

    def measure(self, visual: str, size: int, stroke: int = 0) -> int:
        return _measure(self, visual, size, stroke)

    def ink_extent(self, visual: str, size: int, stroke: int = 0) -> tuple[int, int]:
        """(top, bottom) of the ink relative to the drawing origin (baseline-aware)."""
        top, bottom = 0, 0
        first = True
        for text, face in self.runs(visual):
            _l, t, _r, b = self.font(face, size).getbbox(text, stroke_width=stroke)
            t, b = math.floor(t), math.ceil(b)
            top, bottom = (t, b) if first else (min(top, t), max(bottom, b))
            first = False
        return top, bottom

    def draw(
        self,
        draw: ImageDraw.ImageDraw,
        xy: tuple[float, float],
        visual: str,
        size: int,
        fill: tuple[int, int, int, int],
        stroke: int = 0,
        stroke_fill: tuple[int, int, int, int] | None = None,
    ) -> None:
        x, y = xy
        for text, face in self.runs(visual):
            font = self.font(face, size)
            draw.text(
                (x, y), text, font=font, fill=fill, stroke_width=stroke, stroke_fill=stroke_fill
            )
            x += font.getlength(text)


@functools.lru_cache(maxsize=8192)
def _measure_cached(key: tuple[object, ...], visual: str, size: int, stroke: int) -> int:
    choice = _CHOICES[key]
    width = 0.0
    for text, face in choice.runs(visual):
        width += choice.font(face, size).getlength(text)
    return round(width) + 2 * stroke


_CHOICES: dict[tuple[object, ...], FontChoice] = {}


def _measure(choice: FontChoice, visual: str, size: int, stroke: int) -> int:
    key = (choice.arabic.key, choice.axes, tuple(f.key for f in choice.fallbacks))
    _CHOICES.setdefault(key, choice)
    return _measure_cached(key, visual, size, stroke)


def choose_fonts(
    registry: FontRegistry,
    primary: str,
    fallbacks: list[str],
    symbols: list[str],
    text: str,
    harakat_stripped: bool = True,
) -> FontChoice:
    """First font in the chain whose cmap covers every codepoint the reshaper emits for
    ``text`` (A4 runtime check). Symbols are excluded: the symbol chain draws them."""
    chain = registry.arabic_chain(primary, fallbacks, "basic")
    symbol_chain = registry.symbol_chain(symbols)
    symbol_cps = set().union(*(f.cmap for f in symbol_chain)) if symbol_chain else set()
    body = strip_harakat(text) if harakat_stripped else text
    for face in chain:
        choice = FontChoice(face, [])
        emitted = {ord(c) for c in choice.shape(body) if not c.isspace()}
        needed = {cp for cp in emitted if not (cp in symbol_cps and cp not in face.cmap)}
        if needed <= face.cmap:
            others = [f for f in chain if f is not face]
            return FontChoice(face, symbol_chain + others)
    raise TypesetError(f"no Arabic font covers the text {text!r}")


class RaqmChoice:
    """Render path 2 (A2): raw logical text through libraqm with direction="rtl".

    NEVER combined with the reshaper or python-bidi (double reordering would reverse the
    text). Used only when every character is covered by the single Arabic face.
    """

    def __init__(self, arabic: FontInfo) -> None:
        self.arabic = arabic
        self.fallbacks: list[FontInfo] = []
        self.axes: tuple[tuple[str, float], ...] = ()
        self.missing: set[str] = set()

    def font(self, info: FontInfo, size: int) -> ImageFont.FreeTypeFont:
        return load_font(str(info.path), size, True, self.axes)

    def shape(self, logical: str, harakat: bool = True) -> str:
        return strip_harakat(logical) if harakat else logical

    def covers(self, text: str) -> bool:
        return all(c.isspace() or ord(c) in self.arabic.cmap for c in text)

    def measure(self, visual: str, size: int, stroke: int = 0) -> int:
        f = self.font(self.arabic, size)
        return round(f.getlength(visual, direction="rtl", language="ar")) + 2 * stroke

    def ink_extent(self, visual: str, size: int, stroke: int = 0) -> tuple[int, int]:
        f = self.font(self.arabic, size)
        _l, t, _r, b = f.getbbox(visual, stroke_width=stroke, direction="rtl", language="ar")
        return math.floor(t), math.ceil(b)

    def draw(
        self,
        draw: ImageDraw.ImageDraw,
        xy: tuple[float, float],
        visual: str,
        size: int,
        fill: tuple[int, int, int, int],
        stroke: int = 0,
        stroke_fill: tuple[int, int, int, int] | None = None,
    ) -> None:
        draw.text(
            xy,
            visual,
            font=self.font(self.arabic, size),
            fill=fill,
            stroke_width=stroke,
            stroke_fill=stroke_fill,
            direction="rtl",
            language="ar",
        )
