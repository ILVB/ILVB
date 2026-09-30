"""Classical text detector (no model download; always available).

Pipeline per polarity (dark-on-light, light-on-dark):
1. Adaptive threshold → ink mask.
2. Connected components → glyph candidates filtered by size, fill ratio and line-likeness
   (rejects screentone dots, hatching, panel borders and bubble outlines).
3. Candidates are grouped by their enclosing light region ("container": a bubble
   interior, a caption box, a panel) and, inside a container, by proximity.
4. Each group gets an orientation vote, column/row segmentation, furigana split and an
   exact pixel text mask (the glyph components themselves).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import numpy.typing as npt

from manga_ar.config import DetectConfig
from manga_ar.detect.base import TextBlock
from manga_ar.detect.geometry import disk
from manga_ar.schemas import BBox, CropMask

U8 = npt.NDArray[np.uint8]
BoolArray = npt.NDArray[np.bool_]


@dataclass(eq=False)  # identity semantics: groups hold numpy masks
class _Group:
    mask: BoolArray  # full-page glyph pixels of this group
    bbox: BBox
    size: float  # typical glyph extent
    n_glyphs: int
    polarity: str
    container_area: int
    background_std: float = 0.0  # luminance std-dev around the glyphs


def _box_gap(a: BBox, b: BBox) -> float:
    """Euclidean gap between two boxes (0 when they touch or overlap)."""
    dx = max(0, max(a.x0, b.x0) - min(a.x1, b.x1))
    dy = max(0, max(a.y0, b.y0) - min(a.y1, b.y1))
    return float(np.hypot(dx, dy))


class ClassicalDetector:
    """Threshold + connected-component text detector."""

    name = "classical"

    def __init__(self, cfg: DetectConfig) -> None:
        self.cfg = cfg

    # ------------------------------------------------------------------ API
    def detect(self, rgb: U8) -> list[TextBlock]:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        groups = self._groups(gray, "dark") + self._groups(gray, "light")
        groups = self._resolve_polarity_overlaps(groups, gray)
        # Text drawn straight onto hatching fuses with the lines into one huge component;
        # a pass on morphologically opened ink (1-px lines removed) recovers it.
        for extra in self._groups(gray, "dark", opened=True):
            if not any(
                extra.bbox.iou(g.bbox) > 0.3
                or extra.bbox.overlap_ratio(g.bbox) > 0.5
                or g.bbox.overlap_ratio(extra.bbox) > 0.5
                for g in groups
            ):
                groups.append(extra)
        blocks = [self._to_block(g) for g in groups]
        blocks = [b for b in blocks if b.bbox.area >= self.cfg.min_region_area]
        blocks = self._drop_nested(blocks)
        self._mark_sfx(blocks, min(gray.shape))
        blocks.sort(key=lambda b: (b.bbox.y0, b.bbox.x0))
        return blocks

    # ------------------------------------------------------------ binarise
    @staticmethod
    def _ink(gray: U8, polarity: str) -> U8:
        src = gray if polarity == "dark" else 255 - gray
        block = 31
        ink = cv2.adaptiveThreshold(
            src, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, block, 15
        )
        # Ignore faint local variations: ink must also be reasonably dark in absolute terms.
        ink[src > 200] = 0
        return np.asarray(ink, dtype=np.uint8)

    # -------------------------------------------------------------- groups
    def _groups(self, gray: U8, polarity: str, opened: bool = False) -> list[_Group]:
        h, w = gray.shape
        ink = self._ink(gray, polarity)
        if opened:
            ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
        max_side = self.cfg.max_glyph_frac * min(h, w)
        min_side = self.cfg.min_glyph_px
        strong = np.zeros(n, dtype=bool)
        weak = np.zeros(n, dtype=bool)
        for i in range(1, n):
            x, y, bw, bh, area = (int(v) for v in stats[i])
            big, small = max(bw, bh), min(bw, bh)
            if big > max_side or big < 2:
                continue
            fill = area / float(bw * bh)
            if big >= min_side and small >= 2 and fill >= 0.08 and big / small <= 15:
                if not self._line_like(labels, i, x, y, bw, bh) and self._contrasting_surround(
                    gray, labels, i, x, y, bw, bh, polarity
                ):
                    strong[i] = True
                elif big <= 3 * min_side:
                    # short thin strokes (dakuten, hatch fragments) never seed text but may
                    # join a text group they touch, so they get erased with it
                    weak[i] = True
                continue
            if big < min_side and area >= 2:
                weak[i] = True
        if not strong.any():
            return []
        strong_mask = strong[labels]
        weak_mask = weak[labels]
        # Containers: 4-connected light regions (thin outlines separate them).
        light = (ink == 0).astype(np.uint8)
        _, clabels, cstats, _ = cv2.connectedComponentsWithStats(light, connectivity=4)
        sizes = np.array([max(stats[i, 2], stats[i, 3]) for i in range(n)], dtype=np.float64)
        comp_container = np.zeros(n, dtype=np.int64)
        for i in (int(v) for v in np.flatnonzero(strong)):
            x, y, bw, bh = (int(v) for v in stats[i, :4])
            x0, y0 = max(0, x - 2), max(0, y - 2)
            x1, y1 = min(w, x + bw + 2), min(h, y + bh + 2)
            window = clabels[y0:y1, x0:x1]
            ring = np.concatenate([window[0], window[-1], window[:, 0], window[:, -1]])
            ring = ring[ring > 0]
            comp_container[i] = int(np.bincount(ring).argmax()) if ring.size else 0
        groups: list[_Group] = []
        for cid in np.unique(comp_container[strong]):
            members = np.flatnonzero(strong & (comp_container == cid))
            size = float(np.percentile(sizes[members], 75))
            radius = max(2, round(0.55 * size))
            # All work happens in a window around the members: every dilation below has a
            # radius <= ``radius`` (+3 for the container), so a margin of radius + 4 gives
            # results identical to full-page processing at a fraction of the cost.
            mx0 = int(stats[members, 0].min())
            my0 = int(stats[members, 1].min())
            mx1 = int((stats[members, 0] + stats[members, 2]).max())
            my1 = int((stats[members, 1] + stats[members, 3]).max())
            win = BBox(mx0, my0, mx1, my1).expand(radius + 4).clip(w, h)
            sl = (slice(win.y0, win.y1), slice(win.x0, win.x1))
            labels_w = labels[sl]
            cmask = np.isin(labels_w, members)
            joined = cv2.dilate(cmask.astype(np.uint8), disk(radius))
            if cid > 0:  # never bridge across the container's own boundary
                container = clabels[sl] == cid
                joined &= cv2.dilate(container.astype(np.uint8), disk(3))
            ng, glabels = cv2.connectedComponents(joined, connectivity=8)
            for g in range(1, ng):
                region = glabels == g
                gmask_w = cmask & region
                # attach weak pieces (dakuten, punctuation, dots) close to the glyphs
                near = cv2.dilate(gmask_w.astype(np.uint8), disk(max(2, int(0.3 * size)))) > 0
                gmask_w = gmask_w | (weak_mask[sl] & near)
                local = BBox.from_mask(gmask_w)
                if local is None:
                    continue
                box = local.translate(win.x0, win.y0)
                gmask = np.zeros((h, w), dtype=bool)
                gmask[sl] = gmask_w
                count = len(np.unique(labels_w[gmask_w & strong_mask[sl]]))
                carea = int(cstats[cid, cv2.CC_STAT_AREA]) if cid > 0 else h * w
                group = _Group(gmask, box, size, count, polarity, carea)
                if count <= 2:
                    group.background_std = self._background_std(gray, group)
                groups.append(group)
        kept = [g for g in groups if self._plausible(g)]
        # A glyph or two on textured art is rejected alone, but when it continues an
        # accepted neighbouring group of similar glyph size it is part of that text.
        for cand in groups:
            if cand in kept or cand.background_std <= 40.0:
                continue
            for host in kept:
                ratio = cand.size / max(host.size, 1.0)
                gap = _box_gap(cand.bbox, host.bbox)
                if 0.5 <= ratio <= 2.0 and gap <= 1.5 * host.size:
                    host.mask = host.mask | cand.mask
                    host.bbox = host.bbox.union(cand.bbox)
                    host.n_glyphs += cand.n_glyphs
                    break
        return kept

    @staticmethod
    def _background_std(gray: U8, g: _Group) -> float:
        """Luminance std-dev of the non-glyph pixels around a group (texture probe)."""
        h, w = gray.shape
        win = g.bbox.expand(max(4, int(g.size))).clip(w, h)
        sub = gray[win.y0 : win.y1, win.x0 : win.x1]
        glyphs = cv2.dilate(g.mask[win.y0 : win.y1, win.x0 : win.x1].astype(np.uint8), disk(2))
        vals = sub[glyphs == 0]
        return float(vals.std()) if vals.size else 0.0

    @staticmethod
    def _line_like(labels: npt.NDArray[np.int32], i: int, x: int, y: int, bw: int, bh: int) -> bool:
        """Thin straight strokes (hatching, speed lines) are not glyphs."""
        if max(bw, bh) < 6:
            return False
        pts = np.column_stack(np.nonzero(labels[y : y + bh, x : x + bw] == i)).astype(np.float32)
        if len(pts) < 5:
            return True
        (_, _), (rw, rh), angle = cv2.minAreaRect(pts)
        # The rect spans pixel centres, so a k-px stroke measures k-1. Thin straight
        # strokes that are axis-aligned can be glyphs (一, ー, 丨); diagonal ones are
        # hatching/speed lines.
        thin = min(rw, rh) + 1.0 <= 3.0 and max(rw, rh) >= 5
        tilt = abs(angle) % 90.0
        return bool(thin and 10.0 <= tilt <= 80.0)

    @staticmethod
    def _contrasting_surround(
        gray: U8,
        labels: npt.NDArray[np.int32],
        i: int,
        x: int,
        y: int,
        bw: int,
        bh: int,
        polarity: str,
    ) -> bool:
        """Glyph strokes contrast with what surrounds them: dark ink needs a light ring,
        light ink a dark ring. Rejects fragments of dark fills (e.g. a black bubble's
        interior next to white glyphs) that adaptive thresholding marks as ink."""
        h, w = gray.shape
        x0, y0 = max(0, x - 3), max(0, y - 3)
        x1, y1 = min(w, x + bw + 3), min(h, y + bh + 3)
        comp = (labels[y0:y1, x0:x1] == i).astype(np.uint8)
        ring = (cv2.dilate(comp, disk(2)) > 0) & (comp == 0)
        vals = gray[y0:y1, x0:x1][ring]
        if vals.size == 0:
            return False
        level = float(np.median(vals))
        return level > 120 if polarity == "dark" else level < 135

    def _plausible(self, g: _Group) -> bool:
        # light "glyphs" must sit on a dark area much larger than themselves (white text in
        # a black bubble), not be counters of dark glyphs.
        if g.polarity == "light" and (g.container_area < 4 * int(g.mask.sum()) or g.n_glyphs < 2):
            return False
        if g.n_glyphs <= 2 and g.background_std > 40.0:
            return False  # a couple of blobs on screentone/hatching: texture, not text
        if g.n_glyphs == 1:
            return g.size >= 2.0 * self.cfg.min_glyph_px
        return True

    @staticmethod
    def _ring_stats(gray: U8, g: _Group) -> tuple[float, float]:
        """Median and std-dev of luminance in a thin ring hugging the group's glyphs.

        Real text sits on a uniform, contrasting background (low std). Spurious groups of
        the opposite polarity (glyph counters, halos, background fragments between light
        glyphs) have rings mixing both colours (high std).
        """
        r = max(2, round(0.25 * g.size))
        h, w = gray.shape
        win = g.bbox.expand(r + 2).clip(w, h)
        m = g.mask[win.y0 : win.y1, win.x0 : win.x1].astype(np.uint8)
        ring = (cv2.dilate(m, disk(r)) > 0) & ~(cv2.dilate(m, disk(1)) > 0)
        vals = gray[win.y0 : win.y1, win.x0 : win.x1][ring]
        if vals.size == 0:
            return 255.0, 255.0
        return float(np.median(vals)), float(vals.std())

    def _resolve_polarity_overlaps(self, groups: list[_Group], gray: U8) -> list[_Group]:
        """Arbitrate between overlapping dark- and light-polarity groups (ring test)."""
        dark = [g for g in groups if g.polarity == "dark"]
        light = [g for g in groups if g.polarity == "light"]
        drop: set[int] = set()
        stats = {id(g): self._ring_stats(gray, g) for g in groups}
        for lg in light:
            l_med, l_std = stats[id(lg)]
            light_ok = l_med < 110 and l_std < 45
            rivals = [
                dg
                for dg in dark
                if lg.bbox.overlap_ratio(dg.bbox) > 0.4 or dg.bbox.overlap_ratio(lg.bbox) > 0.4
            ]
            # Size-comparability is symmetric: a light group much smaller than a dark rival
            # is a counter of dark glyphs; dark rivals much smaller than the light group are
            # counters of light glyphs and do not compete.
            comparable = all(lg.bbox.area >= 0.3 * dg.bbox.area for dg in rivals)
            big = [dg for dg in rivals if dg.bbox.area >= 0.3 * lg.bbox.area]
            if light_ok and comparable and all(l_std < stats[id(dg)][1] for dg in big):
                drop.update(id(dg) for dg in rivals)
            else:
                drop.add(id(lg))
        return [g for g in groups if id(g) not in drop]

    # ------------------------------------------------------------- blocks
    def make_block(
        self, glyph_mask: BoolArray, detector: str, polarity: str = "dark", score: float = 0.9,
        vertical: bool | None = None, lines: list[BBox] | None = None,
    ) -> TextBlock | None:  # fmt: skip
        """Build a TextBlock (orientation, lines, furigana) from a full-page glyph mask.

        Used by the ML detector adapters, which supply their own text pixels and, when they
        know them, the orientation (``vertical``; None = decide from the glyphs) and the line
        boxes (page coordinates; None = segment the glyph mask).
        """
        box = BBox.from_mask(glyph_mask)
        if box is None:
            return None
        crop = glyph_mask[box.y0 : box.y1, box.x0 : box.x1]
        n, _, stats, _ = cv2.connectedComponentsWithStats(crop.astype(np.uint8), 8)
        sizes = np.maximum(stats[1:, 2], stats[1:, 3]) if n > 1 else np.array([box.height])
        size = float(np.percentile(sizes, 75))
        block = self._to_block(
            _Group(glyph_mask, box, size, max(1, n - 1), polarity, 0), vertical, lines
        )
        block.detector = detector
        block.score = score
        return block

    def ink_mask(self, gray: U8, polarity: str) -> BoolArray:
        """Polarity-aware ink mask of a page (public for adapters)."""
        return self._ink(gray, polarity) > 0

    def _to_block(
        self, g: _Group, vertical: bool | None = None, given: list[BBox] | None = None
    ) -> TextBlock:
        crop = g.mask[g.bbox.y0 : g.bbox.y1, g.bbox.x0 : g.bbox.x1]
        g.size = max(g.size, self._projection_size(crop))
        if vertical is None:
            vertical = self._orientation(crop, g.size)
        if given:
            order = (lambda b: -b.x1) if vertical else (lambda b: b.y0)  # reading order
            lines = [b.translate(-g.bbox.x0, -g.bbox.y0) for b in sorted(given, key=order)]
            furigana: list[BBox] = []
        else:
            lines, furigana = self._segment_lines(crop, g.size, vertical)
        if lines:
            # line thickness (column width / row height) is a better glyph-size estimate
            # than component sizes, which under-count multi-part Hangul/Han glyphs.
            thickness = [b.width if vertical else b.height for b in lines]
            g.size = max(g.size, float(np.median(thickness)))
        lines = [b.translate(g.bbox.x0, g.bbox.y0) for b in lines]
        furigana = [b.translate(g.bbox.x0, g.bbox.y0) for b in furigana]
        text_mask = CropMask(g.bbox, crop.copy())
        score = min(1.0, 0.5 + 0.1 * g.n_glyphs)
        return TextBlock(
            bbox=g.bbox,
            lines=lines or [g.bbox],
            vertical=vertical,
            score=score,
            text_mask=text_mask,
            detector=self.name,
            glyph_size=g.size,
            polarity=g.polarity,
            furigana=furigana,
        )

    def _projection_size(self, crop: BoolArray) -> float:
        """Glyph size from projections: the smaller of the median row-run height and
        column-run width is the line thickness for either orientation."""
        rows = self._runs(crop.sum(axis=1).astype(np.int64), 2)
        cols = self._runs(crop.sum(axis=0).astype(np.int64), 2)
        if not rows or not cols:
            return 0.0
        return float(min(np.median([b - a for a, b in rows]), np.median([b - a for a, b in cols])))

    @staticmethod
    def glyph_boxes(crop: BoolArray, size: float) -> list[BBox]:
        """Assemble glyph-level boxes from connected pieces (radicals, jamo, dakuten).

        Pieces are merged while their union still fits one roughly square glyph cell (≤ 1.15 ×
        the glyph size, aspect ≤ 1.5); the size estimate is refined once from merged glyphs.
        """
        _, _, stats, _ = cv2.connectedComponentsWithStats(crop.astype(np.uint8), 8)
        boxes = [BBox(int(x), int(y), int(x + w), int(y + h)) for x, y, w, h, _ in stats[1:]]
        if not boxes:
            return []
        s = max(size, 1.0)
        for _ in range(2):
            merged = True
            while merged:
                merged = False
                boxes.sort(key=lambda b: (b.y0, b.x0))
                for i in range(len(boxes)):
                    for j in range(i + 1, len(boxes)):
                        u = boxes[i].union(boxes[j])
                        square = max(u.width, u.height) <= 1.5 * max(1, min(u.width, u.height))
                        if u.width <= 1.15 * s and u.height <= 1.15 * s and square:
                            boxes[i] = u
                            del boxes[j]
                            merged = True
                            break
                    if merged:
                        break
            full = [max(b.width, b.height) for b in boxes if max(b.width, b.height) >= 0.6 * s]
            if full:
                s = max(s, float(np.median(full)))
        return boxes

    def _orientation(self, crop: BoolArray, size: float) -> bool:
        """Vertical vs horizontal by nearest-neighbour direction between whole glyphs.

        Within a line, consecutive glyphs are closer (≈1.0–1.15 em) than adjacent lines or
        columns (≈1.3–1.6 em), so each glyph's nearest neighbour lies along the reading
        direction.
        """
        h, w = crop.shape
        glyphs = self.glyph_boxes(crop, size)
        if len(glyphs) <= 1:
            return bool(h > 1.3 * w)
        s = float(np.median([max(b.width, b.height) for b in glyphs]))
        pts = np.array([b.center for b in glyphs], dtype=np.float64)
        v_votes = h_votes = 0
        for k in range(len(pts)):
            d = pts - pts[k]
            dist = np.hypot(d[:, 0], d[:, 1])
            dist[k] = np.inf
            j = int(np.argmin(dist))
            if not np.isfinite(dist[j]) or dist[j] > 2.2 * s:
                continue
            if abs(d[j, 1]) > abs(d[j, 0]):
                v_votes += 1
            else:
                h_votes += 1
        if v_votes == h_votes:
            return bool(h > 1.3 * w)
        return v_votes > h_votes

    @staticmethod
    def _runs(profile: npt.NDArray[np.int64], min_gap: int) -> list[tuple[int, int]]:
        filled = profile > 0
        runs: list[tuple[int, int]] = []
        start = None
        gap = 0
        for i, on in enumerate(filled):
            if on:
                if start is None:
                    start = i
                gap = 0
            elif start is not None:
                gap += 1
                if gap >= min_gap:
                    runs.append((start, i - gap + 1))
                    start = None
                    gap = 0
        if start is not None:
            end = len(filled)
            while end > start and not filled[end - 1]:
                end -= 1
            runs.append((start, end))
        return runs

    def _segment_lines(
        self, crop: BoolArray, size: float, vertical: bool
    ) -> tuple[list[BBox], list[BBox]]:
        min_gap = max(2, round(0.15 * size))
        axis = 0 if vertical else 1  # vertical: split columns along x
        profile = crop.sum(axis=axis).astype(np.int64)
        runs = self._runs(profile, min_gap)
        boxes: list[BBox] = []
        for a, b in runs:
            if vertical:
                sub = crop[:, a:b]
                rows = np.flatnonzero(sub.any(axis=1))
                if rows.size:
                    boxes.append(BBox(a, int(rows[0]), b, int(rows[-1]) + 1))
            else:
                sub = crop[a:b, :]
                cols = np.flatnonzero(sub.any(axis=0))
                if cols.size:
                    boxes.append(BBox(int(cols[0]), a, int(cols[-1]) + 1, b))
        furigana: list[BBox] = []
        if vertical and self.cfg.suppress_furigana:
            boxes, furigana = self._split_furigana(crop, boxes, size)
        if vertical and len(boxes) >= 2 and self.cfg.suppress_furigana:
            widths = np.array([bx.width for bx in boxes], dtype=np.float64)
            main = float(np.median(widths[widths >= widths.max() * 0.7]))
            keep = []
            for bx in boxes:
                # Furigana columns are narrow *and* made of small glyphs; a column holding
                # one narrow full-size glyph (く, し, 1) is ordinary text.
                col = crop[bx.y0 : bx.y1, bx.x0 : bx.x1]
                glyphs = self.glyph_boxes(col, 0.5 * main)
                tall = max((gb.height for gb in glyphs), default=0)
                if bx.width < 0.6 * main and tall < 0.65 * main and len(glyphs) >= 2:
                    furigana.append(bx)
                else:
                    keep.append(bx)
            boxes = keep or boxes
        if vertical:
            boxes.sort(key=lambda bx: -bx.x1)  # columns read right → left
        else:
            boxes.sort(key=lambda bx: bx.y0)
        return boxes, furigana

    @staticmethod
    def _drop_nested(blocks: list[TextBlock]) -> list[TextBlock]:
        """Drop small blocks lying inside a block of much larger glyphs (e.g. artwork seen
        through the counter of a big SFX glyph)."""
        keep = []
        for b in blocks:
            nested = any(
                o is not b
                and o.bbox.overlap_ratio(b.bbox) < 0.9
                and b.bbox.overlap_ratio(o.bbox) >= 0.9
                and o.glyph_size >= 2.0 * b.glyph_size
                for o in blocks
            )
            if not nested:
                keep.append(b)
        return keep

    def _split_furigana(
        self, crop: BoolArray, boxes: list[BBox], size: float
    ) -> tuple[list[BBox], list[BBox]]:
        """Split a narrow strip of small glyphs off the right edge of a column.

        Furigana sit only 1–3 px from their column, closer than the column-split gap, so
        they are separated here with a fine gap on each column individually.
        """
        kept: list[BBox] = []
        found: list[BBox] = []
        for bx in boxes:
            sub = crop[bx.y0 : bx.y1, bx.x0 : bx.x1]
            runs = self._runs(sub.sum(axis=0).astype(np.int64), 2)
            if len(runs) >= 2:
                a, b = runs[-1]
                main_w = runs[-2][1] - runs[0][0]  # width of the column proper
                ref = max(main_w, 0.8 * size)
                strip = sub[:, a:b]
                glyphs = self.glyph_boxes(strip, 0.4 * ref)
                tall = max((gb.height for gb in glyphs), default=0)
                if (b - a) < 0.6 * ref and tall < 0.65 * ref and len(glyphs) >= 2:
                    rows = np.flatnonzero(strip.any(axis=1))
                    found.append(
                        BBox(bx.x0 + a, bx.y0 + int(rows[0]), bx.x0 + b, bx.y0 + int(rows[-1]) + 1)
                    )
                    main = sub[:, : runs[-2][1]]
                    rows = np.flatnonzero(main.any(axis=1))
                    kept.append(
                        BBox(
                            bx.x0 + runs[0][0],
                            bx.y0 + int(rows[0]),
                            bx.x0 + runs[-2][1],
                            bx.y0 + int(rows[-1]) + 1,
                        )
                    )
                    continue
            kept.append(bx)
        return kept, found

    def _mark_sfx(self, blocks: list[TextBlock], page_min_side: int) -> None:
        if not blocks:
            return
        sizes = np.array([b.glyph_size for b in blocks], dtype=np.float64)
        median = float(np.median(sizes))
        for b in blocks:
            if len(blocks) >= 2:
                sfx = b.glyph_size >= 0.05 * page_min_side and b.glyph_size >= 2.2 * median
            else:  # nothing to compare with: only very large lettering counts as SFX
                sfx = b.glyph_size >= 0.1 * page_min_side
            b.is_sfx = bool(sfx)
