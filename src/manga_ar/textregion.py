"""Typed ``TextRegion`` record (v0.2.0, step 1.0.1).

The contract every v2 stage consumes: id, page, polygon, mask reference, text,
confidence, script/lang, type, reading order, optional speaker and bubble polygon.
Legacy ``schemas.Region`` objects convert through ``manga_ar.textregion_legacy``.
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from manga_ar.ocr.langid import script_counts
from manga_ar.schemas import BBox, CropMask, rle_decode

TextType = Literal["dialogue", "thought", "narration", "sfx", "sign", "credit"]
Point = tuple[float, float]


class MaskRef(BaseModel):
    """Pixel mask stored inline: row-major runs (starting with ``False``) within ``bbox``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    bbox: tuple[int, int, int, int]
    rle: tuple[int, ...]

    @model_validator(mode="after")
    def _runs_cover_the_box(self) -> MaskRef:
        x0, y0, x1, y1 = self.bbox
        if x1 <= x0 or y1 <= y0:
            raise ValueError(f"empty mask bbox {self.bbox}")
        if sum(self.rle) != (x1 - x0) * (y1 - y0) or any(r < 0 for r in self.rle):
            raise ValueError(f"mask runs do not cover bbox {self.bbox}")
        return self

    def to_crop(self) -> CropMask:
        box = BBox.from_list(list(self.bbox))
        return CropMask(box, rle_decode(list(self.rle), (box.height, box.width)))

    @classmethod
    def from_crop(cls, mask: CropMask) -> MaskRef:
        data = mask.to_json()
        return cls(bbox=tuple(data["bbox"]), rle=tuple(data["rle"]))


class TextRegion(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    page: str
    polygon: tuple[Point, ...] = Field(min_length=3)
    mask: MaskRef | None = None
    text: str = ""
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    script: str | None = None  # ISO 15924 (Jpan, Hani, Hang, Latn, …)
    lang: str | None = None
    type: TextType = "dialogue"
    reading_order: int = Field(default=0, ge=0)
    speaker: str | None = None
    bubble_polygon: tuple[Point, ...] | None = None
    vertical: bool = False

    @field_validator("bubble_polygon")
    @classmethod
    def _bubble_is_a_polygon(cls, value: tuple[Point, ...] | None) -> tuple[Point, ...] | None:
        if value is not None and len(value) < 3:
            raise ValueError("bubble_polygon needs at least 3 points")
        return value

    @property
    def bbox(self) -> BBox:
        xs = [p[0] for p in self.polygon]
        ys = [p[1] for p in self.polygon]
        return BBox(
            math.floor(min(xs)), math.floor(min(ys)), math.ceil(max(xs)), math.ceil(max(ys))
        )


def script_tag(text: str) -> str | None:
    """Dominant ISO 15924 script of ``text`` (Japanese mixes kana and han: ``Jpan``)."""
    counts = script_counts(text)
    if counts["hangul"] and counts["hangul"] >= counts["kana"] + counts["han"]:
        return "Hang"
    if counts["kana"]:
        return "Jpan"
    if counts["han"]:
        return "Hani"
    if counts["latin"]:
        return "Latn"
    return None
