"""Benchmark data contracts shared by generators, adapters, metrics and the gates.

Ground truth (``GtPage``) and system output (``PageResult``) are version-independent: the
same schema describes v0.1.0 and v0.2.0 results, so one harness compares them.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

RegionType = Literal["dialogue", "thought", "narration", "sfx", "sign", "credit"]
Split = Literal["dev", "val", "test"]
Point = tuple[float, float]
Rle = dict[str, Any]  # benchmarks.rle format


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GtRegion(_Strict):
    region_id: str
    category: str
    type: RegionType
    lang: str
    text: str
    vertical: bool = False
    text_polygon: list[Point]
    text_mask: Rle
    bubble_mask: Rle | None = None  # interior of the bubble/caption, if any
    safe_mask: Rle  # where typeset ink may go (evaluation ground truth)
    reading_order: int
    speaker: str | None = None
    source_id: str = ""  # meaning id in the text bank (links parallel sources/references)
    font: str = ""
    font_size_px: float = 0.0  # size of the original lettering
    background: Literal["flat", "textured"] = "flat"
    references_ar: list[str] = Field(default_factory=list)
    reference_kind: Literal["none", "silver", "gold"] = "none"


class GtPage(_Strict):
    page_id: str
    series_id: str
    split: Split
    categories: list[str]
    lang: str
    reading_direction: Literal["rtl", "ltr"]
    width: int
    height: int
    image: str  # relative path of the page image
    clean: str  # relative path of the text-free ground-truth background
    seed: int
    generator: str
    regions: list[GtRegion]


class PredRegion(_Strict):
    region_id: str
    polygon: list[Point]
    text: str = ""
    confidence: float | None = None
    type: str = "dialogue"
    lang: str | None = None
    vertical: bool = False
    reading_order: int = 0
    text_mask: Rle | None = None
    flags: list[str] = Field(default_factory=list)


class TypesetReport(_Strict):
    region_id: str
    typeset: bool
    font: str = ""
    size_px: float = 0.0
    lines: list[str] = Field(default_factory=list)
    line_boxes: list[list[float]] = Field(default_factory=list)  # per-line ink [x0, y0, x1, y1]
    ink_mask: Rle | None = None
    overflow: bool = False
    needs_review: bool = False
    hyphen_breaks: list[str] = Field(default_factory=list)  # "word|index" of hyphenated breaks
    flags: list[str] = Field(default_factory=list)


class StageStats(_Strict):
    seconds: float
    peak_rss_mb: float


class PageResult(_Strict):
    page_id: str
    version: str  # "baseline" | "candidate"
    mode: Literal["detect_ocr", "erase", "typeset_gt", "translate_gt"]
    regions: list[PredRegion] = Field(default_factory=list)
    translations: dict[str, str] = Field(default_factory=dict)
    typeset: list[TypesetReport] = Field(default_factory=list)
    erased_image: str | None = None
    final_image: str | None = None
    stages: dict[str, StageStats] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)
    code_origin: str = ""  # path of the manga_ar package that produced this result
