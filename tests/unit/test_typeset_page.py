"""Page-level typesetting: composition, untranslated restore, failure isolation."""

from __future__ import annotations

import numpy as np
import pytest

from manga_ar.config import load_config
from manga_ar.detect.geometry import ellipse_mask
from manga_ar.errors import TypesetError
from manga_ar.schemas import (
    BBox,
    CropMask,
    Flag,
    PageDocument,
    Region,
    RegionOverride,
    RegionType,
    TranslationResult,
)
from manga_ar.typeset.fonts import FontRegistry
from manga_ar.typeset.layout import Placement, Typesetter
from manga_ar.typeset.page import restore_untranslated, typeset_page

CFG = load_config(environ={})
REG = FontRegistry()
SHAPE = (300, 520)


def _bubble(rid: str, x0: int, order: int, text: str | None) -> Region:
    box = BBox(x0, 40, x0 + 200, 240)
    region = Region(
        id=rid,
        type=RegionType.BUBBLE,
        bbox=box,
        fill_color=(255, 255, 255),
        reading_order=order,
        bubble_mask=CropMask(box, ellipse_mask((200, 200), BBox(0, 0, 200, 200))),
    )
    ink = BBox(x0 + 80, 110, x0 + 120, 170)
    region.inpaint_mask = CropMask(ink, np.ones((ink.height, ink.width), bool))
    if text is not None:
        region.translation = TranslationResult(provider="tm", text=text)
    return region


def _page() -> tuple[PageDocument, np.ndarray, np.ndarray]:
    original = np.full((*SHAPE, 3), 255, np.uint8)
    original[110:170, 80:120] = 0  # "source text" of region a
    original[110:170, 340:380] = 0  # "source text" of region b
    clean = np.full((*SHAPE, 3), 255, np.uint8)  # everything inpainted
    doc = PageDocument(source="p.png", width=SHAPE[1], height=SHAPE[0])
    doc.regions = [_bubble("a", 0, 0, "مرحبا بالعالم"), _bubble("b", 260, 1, None)]
    return doc, original, clean


def test_translated_typeset_and_untranslated_restored() -> None:
    doc, original, clean = _page()
    out = typeset_page(doc, original, clean, Typesetter(CFG.typeset, REG))
    a, b = doc.regions
    assert a.layout is not None and a.layout.lines and b.layout is None
    assert set(out.placements) == {"a"} and not out.failed
    assert (out.image[110:170, 340:380] == 0).all()  # original pixels back for b
    assert out.layer[..., 3][40:240, 0:200].any() and not out.layer[..., 3][:, 260:].any()


def test_erase_untranslated_keeps_clean_pixels() -> None:
    doc, original, clean = _page()
    out = typeset_page(doc, original, clean, Typesetter(CFG.typeset, REG), erase_untranslated=True)
    assert (out.image[110:170, 340:380] == 255).all()


def test_skip_override_and_edited_text() -> None:
    doc, original, clean = _page()
    doc.regions[0].override = RegionOverride(skip=True)
    doc.regions[1].override = RegionOverride(text="نص معدل")  # GUI edit of an untranslated one
    out = typeset_page(doc, original, clean, Typesetter(CFG.typeset, REG))
    assert doc.regions[0].layout is None and (out.image[110:170, 80:120] == 0).all()
    assert doc.regions[1].layout is not None and set(out.placements) == {"b"}


class _Failing(Typesetter):
    def layout(
        self,
        region: Region,
        text: str,
        page_shape: tuple[int, int],
        clean: np.ndarray | None = None,
    ) -> Placement:
        if region.id == "a":
            raise TypesetError("boom")
        return super().layout(region, text, page_shape, clean)


def test_region_failure_is_isolated_and_flags_reset() -> None:
    doc, original, clean = _page()
    doc.regions[1].translation = TranslationResult(provider="tm", text="لا تذهب")
    doc.regions[1].flag(Flag.OVERFLOW_RISK)  # stale from an earlier run
    out = typeset_page(doc, original, clean, _Failing(CFG.typeset, REG), shadow=True)
    a, b = doc.regions
    assert Flag.TYPESET_FAILED in a.flags and out.failed == ["a"]
    assert (out.image[110:170, 80:120] == 0).all()  # failed region keeps its original
    assert b.layout is not None and Flag.OVERFLOW_RISK not in b.flags
    # a successful re-run clears the failure flag
    typeset_page(doc, original, clean, Typesetter(CFG.typeset, REG))
    assert Flag.TYPESET_FAILED not in a.flags and a.layout is not None


def test_shape_mismatch_and_restore_helper() -> None:
    doc, original, clean = _page()
    with pytest.raises(ValueError, match="differ"):
        typeset_page(doc, original, clean[:-1], Typesetter(CFG.typeset, REG))
    base = restore_untranslated(original, clean, doc.regions, {"a"})
    assert (base[110:170, 80:120] == 255).all() and (base[110:170, 340:380] == 0).all()
