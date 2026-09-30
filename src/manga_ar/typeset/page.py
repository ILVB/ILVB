"""Page-level typesetting: lay out every translated region and compose the final page.

The cleaned page (``clean``) has *every* detected text removed, so a later edit of an
untranslated region (GUI / ``rerender``) can be typeset without re-inpainting. At
composition time regions that end up without Arabic text get their original pixels back
inside their inpaint mask, unless ``erase_untranslated`` is set (A15, P6 restore rule).
Failures are isolated per region: the region is flagged ``TYPESET_FAILED``, keeps its
original pixels and the page carries on.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from manga_ar.logging_setup import get_logger
from manga_ar.schemas import Flag, PageDocument, Region
from manga_ar.typeset.layout import Placement, Typesetter
from manga_ar.typeset.render import composite, render_layer, to_layout_result

log = get_logger(__name__)
RgbArray = npt.NDArray[np.uint8]
RgbaArray = npt.NDArray[np.uint8]


@dataclass
class PageRender:
    image: RgbArray  # final composite
    layer: RgbaArray  # transparent text layer (debug artifact)
    placements: dict[str, Placement] = field(default_factory=dict)
    failed: list[str] = field(default_factory=list)


def restore_untranslated(
    original: RgbArray, clean: RgbArray, regions: list[Region], typeset_ids: set[str]
) -> RgbArray:
    """``clean`` with original pixels restored in the inpaint mask of every region that
    received no Arabic text (untranslated, skipped or failed)."""
    base = clean.copy()
    h, w = base.shape[:2]
    for region in regions:
        if region.id in typeset_ids or region.inpaint_mask is None:
            continue
        mask = region.inpaint_mask.to_full(h, w)
        base[mask] = original[mask]
    return base


def typeset_page(
    doc: PageDocument,
    original: RgbArray,
    clean: RgbArray,
    typesetter: Typesetter,
    *,
    erase_untranslated: bool = False,
    shadow: bool = False,
) -> PageRender:
    """Lay out all regions of ``doc`` (updating ``region.layout`` and flags) and compose."""
    if original.shape != clean.shape:
        raise ValueError(f"original {original.shape} and clean {clean.shape} differ in shape")
    h, w = clean.shape[:2]
    placements: dict[str, Placement] = {}
    failed: list[str] = []
    for region in sorted(doc.regions, key=lambda r: r.reading_order):
        region.layout = None
        region.flags.difference_update({Flag.OVERFLOW_RISK, Flag.TYPESET_FAILED})
        text = region.arabic_text
        skipped = region.override.skip or Flag.SKIPPED in region.flags
        if skipped or text is None or not text.strip():
            continue
        try:
            placement = typesetter.layout(region, text, (h, w), clean)
            region.layout = to_layout_result(placement)
        except Exception as exc:  # noqa: BLE001 - one bad region must not sink the page
            log.warning(
                "%s: typesetting failed (%s: %s); keeping original pixels",
                region.id,
                type(exc).__name__,
                exc,
            )
            region.flag(Flag.TYPESET_FAILED)
            failed.append(region.id)
            continue
        placements[region.id] = placement
    base = (
        clean
        if erase_untranslated
        else restore_untranslated(original, clean, doc.regions, set(placements))
    )
    layer = render_layer((h, w), list(placements.values()), shadow=shadow)
    return PageRender(composite(base, layer), layer, placements, failed)
