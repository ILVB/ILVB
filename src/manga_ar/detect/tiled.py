"""Detection over long strips: per-tile detection, seam merge, OOM retry (S1, E10)."""

from __future__ import annotations

import cv2

from manga_ar.config import TilingConfig
from manga_ar.detect.base import RgbArray, TextBlock, TextDetector
from manga_ar.io.tiling import Tile, merge_tiled, needs_tiling, plan_tiles
from manga_ar.logging_setup import get_logger

log = get_logger(__name__)


def detect_page(rgb: RgbArray, detector: TextDetector, tiling: TilingConfig) -> list[TextBlock]:
    """Run ``detector`` on the page, tiling tall strips; retries with smaller tiles on OOM."""
    h, w = rgb.shape[:2]
    if not tiling.enabled or not needs_tiling(h, w, tiling.aspect_trigger, tiling.height_trigger):
        return detector.detect(rgb)
    tile_height = tiling.tile_height
    while True:
        try:
            return _detect_tiled(rgb, detector, tile_height, tiling.overlap)
        except MemoryError:
            if tile_height <= 512:
                raise
            tile_height //= 2
            log.warning("out of memory during detection; retrying with %dpx tiles", tile_height)


def _detect_tiled(
    rgb: RgbArray, detector: TextDetector, tile_height: int, overlap: int
) -> list[TextBlock]:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    tiles = plan_tiles(gray, tile_height, min(overlap, tile_height // 2 - 1))
    items: list[tuple[Tile, TextBlock]] = []
    for tile in tiles:
        for block in detector.detect(rgb[tile.y0 : tile.y1]):
            items.append((tile, block.translate(0, tile.y0)))
    merged = merge_tiled(items, box_of=lambda b: b.bbox, page_height=rgb.shape[0])
    out = []
    for block, complete in merged:
        if not complete:
            log.info("text block at %s is cut by every tile seam", block.bbox)
        out.append(block)
    return out
