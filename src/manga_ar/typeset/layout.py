"""Shape-aware layout, organic fit and overflow ladder (A6–A9, A11, A14).

Strategy B ("shape"): each line's available width is the bubble interior's extent over
that line's vertical band, so text follows the bubble outline (round bubbles get a
pyramid/elliptical profile). Strategy A ("rect"): the largest inscribed rectangle, used
when the mask is unreliable (LEAK_FALLBACK) or requested. The largest font size whose
balanced wrap fits is chosen by integer binary search.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np
import numpy.typing as npt

from manga_ar.config import TypesetConfig
from manga_ar.detect.geometry import erode, largest_inscribed_rect
from manga_ar.errors import TypesetError
from manga_ar.logging_setup import get_logger
from manga_ar.schemas import BBox, Flag, Region, RegionType
from manga_ar.translate.normalize_ar import normalize_ar
from manga_ar.typeset.arabic_text import strip_harakat
from manga_ar.typeset.contrast import choose_colors
from manga_ar.typeset.fonts import FontRegistry
from manga_ar.typeset.textline import FontChoice, RaqmChoice, choose_fonts
from manga_ar.typeset.wrap import tokenize, wrap_fixed

log = get_logger(__name__)
BoolArray = npt.NDArray[np.bool_]
RgbArray = npt.NDArray[np.uint8]
Engine = FontChoice | RaqmChoice
# Reference string for line ink extents: tall alef/lam-alef, deep reh/yeh/meem descenders.
REF_TEXT = "أإلأ لي مرجع ـــ"
_REF_SIZE = 100


@dataclass
class Geometry:
    box: BBox  # page coordinates of ``mask``
    mask: BoolArray  # where ink may go
    strategy: str  # "shape" | "rect"

    @property
    def area(self) -> int:
        return int(self.mask.sum())


@dataclass
class PlacedLine:
    logical: str
    visual: str
    x: float  # draw origin, page coordinates
    y: float
    width: int


@dataclass
class Placement:
    size: int
    line_height: int
    lines: list[PlacedLine]
    engine: Engine
    strategy: str
    text_rgb: tuple[int, int, int]
    outline_rgb: tuple[int, int, int] | None
    outline_px: int
    ladder: list[str] = field(default_factory=list)
    overflow: bool = False
    ink: BBox | None = None
    geometry: Geometry | None = None


class _Rows:
    """Longest contiguous run per mask row, for fast band-span queries."""

    def __init__(self, mask: BoolArray) -> None:
        h = mask.shape[0]
        self.starts = np.full(h, -1, dtype=np.int64)
        self.ends = np.full(h, -1, dtype=np.int64)
        for y in range(h):
            row = mask[y]
            if not row.any():
                continue
            padded = np.concatenate(([False], row, [False]))
            edges = np.flatnonzero(padded[1:] != padded[:-1])
            runs = edges.reshape(-1, 2)
            k = int(np.argmax(runs[:, 1] - runs[:, 0]))
            self.starts[y], self.ends[y] = runs[k]
        valid = self.starts >= 0
        widths = np.where(valid, self.ends - self.starts, 0).astype(np.float64)
        self.top = int(np.argmax(valid)) if valid.any() else 0
        self.bottom = int(h - np.argmax(valid[::-1])) if valid.any() else 0
        total = widths.sum()
        ys = np.arange(h, dtype=np.float64)
        self.centroid = float((ys * widths).sum() / total) if total else h / 2.0

    def span(self, y0: float, y1: float) -> tuple[int, int]:
        a, b = max(0, math.floor(y0)), min(len(self.starts), math.ceil(y1))
        if b <= a or (self.starts[a:b] < 0).any():
            return (0, 0)
        return int(self.starts[a:b].max()), int(self.ends[a:b].min())


class Typesetter:
    """Pure layout + rendering of Arabic text into regions (A14)."""

    def __init__(
        self, cfg: TypesetConfig, registry: FontRegistry | None = None,
        source_size_cap: bool = False,
    ) -> None:  # fmt: skip
        self.cfg = cfg
        self.registry = registry or FontRegistry()
        self.source_size_cap = source_size_cap  # resolved by AppConfig.upgrade (v2 profile)
        from PIL import features

        self.raqm = bool(features.check("raqm"))
        self._metric_cache: dict[tuple[object, ...], tuple[list[float], float]] = {}

    # -------------------------------------------------------------- geometry
    def geometry(
        self,
        region: Region,
        page_shape: tuple[int, int],
        padding_floor: bool = False,
        strategy: str | None = None,
    ) -> Geometry:
        h, w = page_shape
        strategy = strategy or self.cfg.strategy
        use_bubble = (
            region.bubble_mask is not None
            and region.type in {RegionType.BUBBLE, RegionType.NARRATION}
            and Flag.LEAK_FALLBACK not in region.flags
        )
        if use_bubble:
            bm = region.bubble_mask
            assert bm is not None
            ref = region.safe_box or bm.bbox
            pad = (
                self.cfg.padding_min_px
                if padding_floor
                else max(
                    self.cfg.padding_min_px,
                    round(self.cfg.padding_frac * min(ref.width, ref.height)),
                )
            )
            mask = erode(bm.data, pad)
            if not mask.any():
                mask = bm.data.copy()
            return Geometry(bm.bbox, np.asarray(mask, dtype=np.bool_), strategy)
        # Free text / SFX / leaked bubble: the area the original text occupied.
        box = region.bbox
        if region.inpaint_mask is not None:
            box = box.union(region.inpaint_mask.bbox)
        if Flag.LEAK_FALLBACK in region.flags and region.safe_box is not None:
            box = box.union(region.safe_box)
        box = box.clip(w, h)
        return Geometry(box, np.ones((box.height, box.width), dtype=np.bool_), "rect")

    def extended_geometry(self, region: Region, clean: RgbArray, geom: Geometry) -> Geometry:
        """Ladder step 4: grow into adjacent *uniform* background, never across outlines.

        Bubbles with a trusted mask may use their whole interior (no padding) but never
        leave it. Free text / leak fallbacks flood-fill the cleaned page from the current
        area with a small tolerance (stops at outlines and artwork)."""
        h, w = clean.shape[:2]
        if (
            region.bubble_mask is not None
            and region.type in {RegionType.BUBBLE, RegionType.NARRATION}
            and Flag.LEAK_FALLBACK not in region.flags
        ):
            bm = region.bubble_mask
            return Geometry(bm.bbox, bm.data.copy(), geom.strategy)
        grow = max(geom.box.width, geom.box.height) // 2
        win = geom.box.expand(grow).clip(w, h)
        crop = np.ascontiguousarray(clean[win.y0 : win.y1, win.x0 : win.x1, ::-1])
        seed_mask = np.zeros((win.height + 2, win.width + 2), np.uint8)
        inner = np.zeros((win.height, win.width), bool)
        inner[
            geom.box.y0 - win.y0 : geom.box.y1 - win.y0, geom.box.x0 - win.x0 : geom.box.x1 - win.x0
        ] = geom.mask
        ys, xs = np.nonzero(inner)
        if ys.size == 0:
            return geom
        step = max(1, ys.size // 16)
        for y, x in zip(ys[::step], xs[::step], strict=False):
            if not seed_mask[y + 1, x + 1]:
                cv2.floodFill(
                    crop,
                    seed_mask,
                    (int(x), int(y)),
                    (0, 0, 0),
                    (6, 6, 6),
                    (6, 6, 6),
                    4 | cv2.FLOODFILL_MASK_ONLY | (255 << 8),
                )
        region_mask = (seed_mask[1:-1, 1:-1] > 0) | inner
        region_mask = erode(region_mask, self.cfg.padding_min_px)
        region_mask |= inner
        return Geometry(win, np.asarray(region_mask, dtype=np.bool_), "shape")

    # -------------------------------------------------------------- engines
    def engine_for(self, text: str, font_key: str | None, condensed: bool = False) -> Engine:
        primary = font_key or self.cfg.font
        if self.cfg.render_path == "raqm" and self.raqm:
            face = self.registry.get(primary) if primary in self.registry.fonts else None
            if face is not None and face.role == "arabic":
                eng = RaqmChoice(face)
                if eng.covers(strip_harakat(text)):
                    return eng
                log.info("RAQM path: %s lacks glyphs for this text; using BASIC path", primary)
        choice = choose_fonts(
            self.registry,
            primary,
            self.cfg.fallback_fonts,
            self.cfg.symbol_fonts,
            text,
            self.cfg.strip_harakat,
        )
        if condensed:
            axes = choice.arabic.condensed_axes
            if axes:
                choice.axes = tuple(sorted(axes.items()))
        return choice

    # ------------------------------------------------------------------ fit
    def _size_bounds(
        self, geom: Geometry, page_h: int, ceiling: int | None = None
    ) -> tuple[int, int]:
        lo = max(self.cfg.min_size_px, round(self.cfg.min_size_page_frac * page_h))
        rows = _Rows(geom.mask)
        usable = max(1, rows.bottom - rows.top)
        hi = min(
            round(self.cfg.max_size_safe_frac * usable), round(self.cfg.max_size_page_frac * page_h)
        )
        if ceiling is not None:
            hi = min(hi, ceiling)  # E-09: short strings never balloon past the original
        return lo, max(lo, hi)

    def source_ceiling(self, region: Region) -> int | None:
        """Size ceiling from the original lettering (v2, E-09): factor x the median line
        height (column width for vertical text); None without line geometry."""
        if not self.source_size_cap or not region.lines:
            return None
        extents = sorted(b.width if region.vertical else b.height for b in region.lines)
        return max(1, round(self.cfg.source_size_factor * extents[len(extents) // 2]))

    def _try(
        self,
        geom: Geometry,
        rows: _Rows,
        words: list[str],
        engine: Engine,
        size: int,
        spacing: float,
        outline: bool,
        rect: BBox | None,
    ) -> Placement | None:
        lh = max(size, round(size * spacing))
        stroke = max(1, round(self.cfg.outline_frac * size)) if outline else 0
        top_ref, bottom_ref = engine.ink_extent(engine.shape(REF_TEXT), size, stroke)
        ink_h = bottom_ref - top_ref
        if rect is not None:
            area_top, area_bottom = rect.y0, rect.y1
            cy = (rect.y0 + rect.y1) / 2.0
        else:
            area_top, area_bottom = rows.top, rows.bottom
            cy = rows.centroid
        available = area_bottom - area_top
        if ink_h > available:
            return None
        text_len = sum(len(wd) for wd in words)
        # Joining never crosses a space, so a line's width is the sum of its words' widths
        # plus spaces: measure words once at a reference size and scale (A13). The chosen
        # lines are re-measured exactly below.
        word_w, space_w = self._word_metrics(engine, tuple(words))
        scale = size / _REF_SIZE
        prefix = np.concatenate(([0.0], np.cumsum(word_w)))

        def measure(i: int, j: int) -> int:
            raw = float(prefix[j] - prefix[i] + (j - i - 1) * space_w) * scale
            return math.ceil(raw) + 1 + 2 * stroke

        candidates: list[tuple[float, list[tuple[int, int]], list[str], float]] = []
        max_lines = min(len(words), (available - ink_h) // lh + 1)
        for n in range(1, max_lines + 1):
            block = (n - 1) * lh + ink_h
            start = min(max(cy - block / 2.0, area_top), area_bottom - block)
            spans = []
            for k in range(n):
                y0 = start + k * lh
                if rect is not None:
                    spans.append((rect.x0, rect.x1))
                else:
                    spans.append(rows.span(y0, y0 + ink_h))
            widths = [right - left for left, right in spans]
            if min(widths) <= 0:
                continue
            wrap = wrap_fixed(words, widths, measure)
            if wrap is None:
                continue
            cost = wrap.cost + 0.03 * n + (0.25 if n == 1 and text_len > 14 else 0.0)
            candidates.append((cost, spans, wrap.lines, start - top_ref))
        # Verify candidates in cost order with exact widths: the scaled word-sum estimate can
        # be a pixel optimistic, and one failing wrap must not make the size infeasible.
        chosen = None
        for _cost, spans, lines, origin in sorted(candidates, key=lambda c: c[0]):
            exact = [
                engine.measure(engine.shape(ln, self.cfg.strip_harakat), size, stroke)
                for ln in lines
            ]
            if all(w <= right - left for w, (left, right) in zip(exact, spans, strict=True)):
                chosen = (spans, lines, exact, origin)
                break
        if chosen is None:
            return None
        spans, lines, widths, origin = chosen
        placed = []
        for k, (line, width, (left, right)) in enumerate(zip(lines, widths, spans, strict=True)):
            x: float
            if self.cfg.alignment == "right":
                x = right - width
            else:
                x = left + (right - left - width) / 2.0
            visual = engine.shape(line, self.cfg.strip_harakat)
            placed.append(
                PlacedLine(
                    line, visual, geom.box.x0 + x + stroke, geom.box.y0 + origin + k * lh, width
                )
            )
        return Placement(size, lh, placed, engine, geom.strategy, (0, 0, 0), None, stroke)

    def _word_metrics(self, engine: Engine, words: tuple[str, ...]) -> tuple[list[float], float]:
        # Stable identity (never id(): ids of freed engines are reused).
        key = (
            type(engine).__name__,
            engine.arabic.key,
            getattr(engine, "axes", ()),
            self.cfg.strip_harakat,
            words,
        )
        hit = self._metric_cache.get(key)
        if hit is not None:
            return hit
        widths = [
            float(engine.measure(engine.shape(wd, self.cfg.strip_harakat), _REF_SIZE))
            for wd in words
        ]
        pair = float(engine.measure(engine.shape("ا ا"), _REF_SIZE))
        single = float(engine.measure(engine.shape("اا"), _REF_SIZE))
        result = (widths, max(0.0, pair - single))
        self._metric_cache[key] = result
        return result

    def _search(
        self,
        geom: Geometry,
        words: list[str],
        engine: Engine,
        page_h: int,
        spacing: float,
        outline: bool,
        fixed_size: int | None = None,
        ceiling: int | None = None,
    ) -> Placement | None:
        rows = _Rows(geom.mask)
        rect = None
        if geom.strategy == "rect":
            rect = largest_inscribed_rect(geom.mask)
            if rect is None:
                return None
        if fixed_size is not None:
            return self._try(geom, rows, words, engine, fixed_size, spacing, outline, rect)
        lo, hi = self._size_bounds(geom, page_h, ceiling)
        top = hi
        best = None
        while lo <= hi:
            mid = (lo + hi) // 2
            cand = self._try(geom, rows, words, engine, mid, spacing, outline, rect)
            if cand is not None:
                best, lo = cand, mid + 1
            else:
                hi = mid - 1
        # Feasibility is almost but not strictly monotonic (discrete line counts); probe a
        # few sizes above the result so an isolated infeasible size cannot cap the fit.
        start = best.size + 1 if best is not None else lo
        for size in range(start, min(top, start + 3) + 1):
            cand = self._try(geom, rows, words, engine, size, spacing, outline, rect)
            if cand is not None:
                best = cand
        return best

    # ------------------------------------------------------------ public API
    def layout(
        self, region: Region, text: str, page_shape: tuple[int, int], clean: RgbArray | None = None
    ) -> Placement:
        """Place ``text`` in ``region``; walks the overflow ladder when needed (A9)."""
        h, w = page_shape
        text = normalize_ar(text, self.cfg.digits)
        if self.cfg.strip_harakat:
            text = strip_harakat(text)
        words = tokenize(text)
        if not words:
            raise TypesetError(f"{region.id}: nothing to typeset")
        outline_forced = region.type == RegionType.FREE_TEXT
        background = self._background(region, clean)
        _, _, px_probe = choose_colors(
            background, self.cfg.contrast_target, 20, self.cfg.outline_frac, outline_forced
        )
        outline = outline_forced or px_probe > 0
        fixed = region.override.size_px
        engine = self.engine_for(text, region.override.font)
        geom = self.geometry(region, page_shape)
        ladder: list[str] = []
        ceiling = self.source_ceiling(region)
        placement = self._search(
            geom, words, engine, h, self.cfg.line_spacing, outline, fixed, ceiling
        )
        if placement is None and fixed is None:
            steps: list[tuple[str, dict[str, float | bool]]] = [
                ("line-spacing-floor", {"spacing": self.cfg.line_spacing_floor}),
                ("condensed", {"condensed": True}),
                ("padding-floor", {"padding_floor": True}),
                ("extend-uniform", {"extend": True}),
            ]
            spacing, condensed, padding_floor, extend = self.cfg.line_spacing, False, False, False
            for name, change in steps:
                if name == "condensed" and (
                    not isinstance(engine, FontChoice) or not engine.arabic.condensed_axes
                ):
                    continue
                if name == "extend-uniform" and clean is None:
                    continue
                spacing = float(change.get("spacing", spacing))
                condensed = bool(change.get("condensed", condensed))
                padding_floor = bool(change.get("padding_floor", padding_floor))
                extend = bool(change.get("extend", extend))
                ladder.append(name)
                engine = self.engine_for(text, region.override.font, condensed)
                geom = self.geometry(region, page_shape, padding_floor)
                if extend and clean is not None:
                    geom = self.extended_geometry(region, clean, geom)
                placement = self._search(geom, words, engine, h, spacing, outline, None, ceiling)
                if placement is not None:
                    break
        if placement is None:
            ladder.append("hard-floor")
            placement = self._forced(region, geom, words, engine, page_shape, outline, fixed)
        text_rgb, outline_rgb, outline_px = choose_colors(
            background,
            self.cfg.contrast_target,
            placement.size,
            self.cfg.outline_frac,
            outline_forced,
        )
        placement.text_rgb = text_rgb
        placement.outline_rgb = outline_rgb
        placement.outline_px = outline_px if outline else 0
        placement.ladder = ladder
        placement.geometry = geom
        placement.ink = self._ink_box(placement)
        self._keep_inside(placement, w, h)
        return placement

    def _forced(
        self,
        region: Region,
        geom: Geometry,
        words: list[str],
        engine: Engine,
        page_shape: tuple[int, int],
        outline: bool,
        fixed: int | None,
    ) -> Placement:
        """Hard floor: smallest allowed size, widest feasible lines; flagged OVERFLOW_RISK."""
        h = page_shape[0]
        size = fixed or max(self.cfg.min_size_px, round(self.cfg.min_size_page_frac * h))
        stroke = max(1, round(self.cfg.outline_frac * size)) if outline else 0
        lh = max(size, round(size * self.cfg.line_spacing_floor))
        span = max(geom.box.width, 1)

        def measure(line: str) -> int:
            return engine.measure(engine.shape(line, self.cfg.strip_harakat), size, stroke)

        lines: list[str] = []
        current: list[str] = []
        for word in words:
            trial = " ".join([*current, word])
            if current and measure(trial) > span:
                lines.append(" ".join(current))
                current = [word]
            else:
                current.append(word)
        if current:
            lines.append(" ".join(current))
        top_ref, bottom_ref = engine.ink_extent(engine.shape(REF_TEXT), size, stroke)
        block = (len(lines) - 1) * lh + (bottom_ref - top_ref)
        cx = geom.box.x0 + geom.box.width / 2.0
        cy = geom.box.y0 + geom.box.height / 2.0
        origin = cy - block / 2.0 - top_ref
        placed = []
        for k, line in enumerate(lines):
            width = measure(line)
            placed.append(
                PlacedLine(
                    line,
                    engine.shape(line, self.cfg.strip_harakat),
                    cx - width / 2.0 + stroke,
                    origin + k * lh,
                    width,
                )
            )
        region.flag(Flag.OVERFLOW_RISK)
        return Placement(
            size, lh, placed, engine, geom.strategy, (0, 0, 0), None, stroke, overflow=True
        )

    def _background(self, region: Region, clean: RgbArray | None) -> tuple[int, int, int]:
        if region.type in {RegionType.BUBBLE, RegionType.NARRATION} and region.fill_color:
            return region.fill_color
        if clean is not None:
            h, w = clean.shape[:2]
            box = region.bbox.clip(w, h)
            if box.area:
                med = np.median(clean[box.y0 : box.y1, box.x0 : box.x1].reshape(-1, 3), axis=0)
                return (int(med[0]), int(med[1]), int(med[2]))
        return region.fill_color or (255, 255, 255)

    @staticmethod
    def _ink_box(p: Placement) -> BBox | None:
        box: BBox | None = None
        for line in p.lines:
            t, b = p.engine.ink_extent(line.visual, p.size, p.outline_px)
            lb = BBox(
                math.floor(line.x - p.outline_px),
                math.floor(line.y + t),
                math.ceil(line.x - p.outline_px + line.width),
                math.ceil(line.y + b),
            )
            box = lb if box is None else box.union(lb)
        return box

    def _keep_inside(self, p: Placement, w: int, h: int) -> None:
        """Never draw outside the image: shift the whole block back inside if needed."""
        if p.ink is None:
            return
        dx = -p.ink.x0 if p.ink.x0 < 0 else (w - p.ink.x1 if p.ink.x1 > w else 0)
        dy = -p.ink.y0 if p.ink.y0 < 0 else (h - p.ink.y1 if p.ink.y1 > h else 0)
        if dx or dy:
            for line in p.lines:
                line.x += dx
                line.y += dy
            p.ink = p.ink.translate(int(dx), int(dy))
