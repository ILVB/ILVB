"""Typed data objects exchanged between pipeline stages, plus the sidecar JSON format.

Masks are stored cropped to their own bounding box (:class:`CropMask`) and serialised
as row-major run-length encodings, never as raw arrays.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from manga_ar import __version__
from manga_ar.errors import MangaArError

SCHEMA_VERSION = 1

BoolArray = npt.NDArray[np.bool_]
Rgb = tuple[int, int, int]


class RegionType(str, Enum):
    BUBBLE = "bubble"
    NARRATION = "narration"
    FREE_TEXT = "free_text"
    SFX = "sfx"


class Flag(str, Enum):
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    OCR_SUSPECT = "OCR_SUSPECT"
    OCR_FAILED = "OCR_FAILED"
    MERGED = "MERGED"
    LEAK_FALLBACK = "LEAK_FALLBACK"
    RESIDUAL_TEXT = "RESIDUAL_TEXT"
    OVERFLOW_RISK = "OVERFLOW_RISK"
    UNTRANSLATED = "UNTRANSLATED"
    SKIPPED = "SKIPPED"
    PASS_THROUGH = "PASS_THROUGH"
    INPAINT_FALLBACK = "INPAINT_FALLBACK"
    GLOSSARY_DEGRADED = "GLOSSARY_DEGRADED"
    TYPESET_FAILED = "TYPESET_FAILED"
    SEAM_MERGED = "SEAM_MERGED"
    FROM_SFX = "FROM_SFX"  # an SFX region processed as free text (``detect.sfx: translate``)


@dataclass(frozen=True, order=True)
class BBox:
    """Axis-aligned box with exclusive ``x1``/``y1`` (pixel-grid convention)."""

    x0: int
    y0: int
    x1: int
    y1: int

    def __post_init__(self) -> None:
        if self.x1 < self.x0 or self.y1 < self.y0:
            raise ValueError(f"invalid bbox {self}")

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x0 + self.x1) / 2.0, (self.y0 + self.y1) / 2.0)

    def intersection(self, other: BBox) -> BBox | None:
        x0, y0 = max(self.x0, other.x0), max(self.y0, other.y0)
        x1, y1 = min(self.x1, other.x1), min(self.y1, other.y1)
        if x1 <= x0 or y1 <= y0:
            return None
        return BBox(x0, y0, x1, y1)

    def union(self, other: BBox) -> BBox:
        return BBox(
            min(self.x0, other.x0),
            min(self.y0, other.y0),
            max(self.x1, other.x1),
            max(self.y1, other.y1),
        )

    def iou(self, other: BBox) -> float:
        inter = self.intersection(other)
        if inter is None:
            return 0.0
        union = self.area + other.area - inter.area
        return inter.area / union if union else 0.0

    def overlap_ratio(self, other: BBox) -> float:
        """Intersection area divided by this box's area."""
        inter = self.intersection(other)
        return inter.area / self.area if inter is not None and self.area else 0.0

    def contains(self, other: BBox) -> bool:
        return (
            self.x0 <= other.x0
            and self.y0 <= other.y0
            and self.x1 >= other.x1
            and self.y1 >= other.y1
        )

    def expand(self, px: int) -> BBox:
        return BBox(self.x0 - px, self.y0 - px, self.x1 + px, self.y1 + px)

    def clip(self, width: int, height: int) -> BBox:
        x0 = min(max(self.x0, 0), width)
        y0 = min(max(self.y0, 0), height)
        return BBox(x0, y0, min(max(self.x1, x0), width), min(max(self.y1, y0), height))

    def translate(self, dx: int, dy: int) -> BBox:
        return BBox(self.x0 + dx, self.y0 + dy, self.x1 + dx, self.y1 + dy)

    def to_list(self) -> list[int]:
        return [self.x0, self.y0, self.x1, self.y1]

    @classmethod
    def from_list(cls, values: list[int] | tuple[int, ...]) -> BBox:
        x0, y0, x1, y1 = (int(v) for v in values)
        return cls(x0, y0, x1, y1)

    @classmethod
    def from_mask(cls, mask: BoolArray) -> BBox | None:
        rows = np.flatnonzero(mask.any(axis=1))
        if rows.size == 0:
            return None
        cols = np.flatnonzero(mask.any(axis=0))
        return cls(int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1)


def rle_encode(mask: BoolArray) -> list[int]:
    """Row-major run lengths, starting with a (possibly empty) run of ``False``."""
    flat = np.ascontiguousarray(mask, dtype=np.bool_).ravel()
    if flat.size == 0:
        return []
    change = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    bounds = np.concatenate(([0], change, [flat.size]))
    runs = np.diff(bounds).tolist()
    if flat[0]:
        runs.insert(0, 0)
    return [int(r) for r in runs]


def rle_decode(runs: list[int], shape: tuple[int, int]) -> BoolArray:
    total = shape[0] * shape[1]
    if sum(runs) != total:
        raise MangaArError(f"RLE length {sum(runs)} does not match shape {shape}")
    values = np.zeros(len(runs), dtype=np.bool_)
    values[1::2] = True
    return np.repeat(values, runs).reshape(shape)


@dataclass
class CropMask:
    """Boolean mask stored only within ``bbox`` (page coordinates)."""

    bbox: BBox
    data: BoolArray

    def __post_init__(self) -> None:
        if self.data.shape != (self.bbox.height, self.bbox.width):
            raise ValueError(f"mask shape {self.data.shape} does not match bbox {self.bbox}")
        self.data = self.data.astype(np.bool_, copy=False)

    @classmethod
    def from_full(cls, mask: BoolArray, offset: tuple[int, int] = (0, 0)) -> CropMask | None:
        """Crop a full-size mask to its tight bbox; ``offset`` shifts to page coordinates."""
        bbox = BBox.from_mask(mask)
        if bbox is None:
            return None
        data = mask[bbox.y0 : bbox.y1, bbox.x0 : bbox.x1].copy()
        return cls(bbox.translate(offset[0], offset[1]), data)

    @property
    def area(self) -> int:
        return int(self.data.sum())

    def to_full(self, height: int, width: int) -> BoolArray:
        out = np.zeros((height, width), dtype=np.bool_)
        clipped = self.bbox.clip(width, height)
        if clipped.area == 0:
            return out
        sy, sx = clipped.y0 - self.bbox.y0, clipped.x0 - self.bbox.x0
        out[clipped.y0 : clipped.y1, clipped.x0 : clipped.x1] = self.data[
            sy : sy + clipped.height, sx : sx + clipped.width
        ]
        return out

    def window(self, box: BBox) -> BoolArray:
        """The mask restricted to ``box`` (page coordinates), shape ``(box.h, box.w)``."""
        out = np.zeros((box.height, box.width), dtype=np.bool_)
        inter = self.bbox.intersection(box)
        if inter is None:
            return out
        out[inter.y0 - box.y0 : inter.y1 - box.y0, inter.x0 - box.x0 : inter.x1 - box.x0] = (
            self.data[
                inter.y0 - self.bbox.y0 : inter.y1 - self.bbox.y0,
                inter.x0 - self.bbox.x0 : inter.x1 - self.bbox.x0,
            ]
        )
        return out

    def translate(self, dx: int, dy: int) -> CropMask:
        return CropMask(self.bbox.translate(dx, dy), self.data.copy())

    def to_json(self) -> dict[str, Any]:
        return {"bbox": self.bbox.to_list(), "rle": rle_encode(self.data)}

    @classmethod
    def from_json(cls, obj: dict[str, Any]) -> CropMask:
        bbox = BBox.from_list(obj["bbox"])
        return cls(bbox, rle_decode(list(obj["rle"]), (bbox.height, bbox.width)))


@dataclass
class OcrCandidate:
    engine: str
    text: str
    confidence: float | None
    score: float


@dataclass
class OcrResult:
    engine: str
    text: str
    confidence: float | None = None
    lang: str | None = None
    vertical: bool = False
    raw_text: str = ""
    decorations: list[str] = field(default_factory=list)
    alternatives: list[OcrCandidate] = field(default_factory=list)


@dataclass
class TranslationAttempt:
    provider: str
    ok: bool
    error: str | None = None
    elapsed_s: float = 0.0


@dataclass
class TranslationResult:
    provider: str
    text: str
    from_cache: bool = False
    attempts: list[TranslationAttempt] = field(default_factory=list)


@dataclass
class LayoutResult:
    font: str
    size_px: int
    lines: list[str]
    strategy: str
    box: BBox
    line_height: int
    fill: Rgb
    outline: Rgb | None = None
    outline_px: int = 0
    ladder: list[str] = field(default_factory=list)


@dataclass
class RegionOverride:
    """User edits from the GUI / sidecar. ``None`` means "no override"."""

    text: str | None = None
    font: str | None = None
    size_px: int | None = None
    source_lang: str | None = None
    skip: bool = False


@dataclass
class Region:
    id: str
    type: RegionType
    bbox: BBox
    lines: list[BBox] = field(default_factory=list)
    polygon: list[tuple[int, int]] = field(default_factory=list)
    bubble_mask: CropMask | None = None
    text_mask: CropMask | None = None
    inpaint_mask: CropMask | None = None
    safe_box: BBox | None = None
    reading_order: int = 0
    source_lang: str | None = None
    vertical: bool = False
    score: float = 1.0
    fill_color: Rgb | None = None
    inpaint_method: str | None = None
    ocr: OcrResult | None = None
    translation: TranslationResult | None = None
    layout: LayoutResult | None = None
    flags: set[Flag] = field(default_factory=set)
    override: RegionOverride = field(default_factory=RegionOverride)

    def flag(self, *flags: Flag) -> None:
        self.flags.update(flags)

    @property
    def source_text(self) -> str:
        return self.ocr.text if self.ocr is not None else ""

    @property
    def arabic_text(self) -> str | None:
        """The text to typeset: user override, else the accepted translation."""
        if self.override.text is not None:
            return self.override.text
        if self.translation is not None and Flag.UNTRANSLATED not in self.flags:
            return self.translation.text
        return None

    @property
    def is_translated(self) -> bool:
        return bool(self.arabic_text) and not self.override.skip


@dataclass
class PageDocument:
    source: str
    width: int
    height: int
    regions: list[Region] = field(default_factory=list)
    source_sha256: str = ""
    lang: str | None = None
    reading_order_mode: str = "manga_rtl"
    config_hash: str = ""
    status: str = "pending"
    warnings: list[str] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    clean_image: str | None = None  # inpainted page (all text removed), relative to sidecar
    source_image: str | None = None  # decoded original page, relative to the sidecar
    output: str | None = None  # rendered page, relative to the sidecar (or the CBZ)
    output_member: str | None = None  # member name when ``output`` is a CBZ
    settings: dict[str, Any] = field(default_factory=dict)  # config used (for rerender)
    was_grayscale: bool = False
    schema_version: int = SCHEMA_VERSION
    created_with: str = __version__

    def region(self, region_id: str) -> Region:
        for r in self.regions:
            if r.id == region_id:
                return r
        raise KeyError(region_id)

    # ------------------------------------------------------------------ JSON
    def to_json(self) -> dict[str, Any]:
        return _to_jsonable(self)  # type: ignore[no-any-return]

    def dumps(self) -> str:
        return json.dumps(self.to_json(), ensure_ascii=False, indent=1, sort_keys=False)

    @classmethod
    def from_json(cls, obj: dict[str, Any]) -> PageDocument:
        version = obj.get("schema_version")
        if version != SCHEMA_VERSION:
            raise MangaArError(
                f"unsupported sidecar schema_version {version!r} (expected {SCHEMA_VERSION})"
            )
        regions = [_region_from_json(r) for r in obj.get("regions", [])]
        kwargs = {k: v for k, v in obj.items() if k != "regions"}
        return cls(regions=regions, **kwargs)

    def save(self, path: Path) -> None:
        from manga_ar.io.writer import atomic_write_text

        atomic_write_text(path, self.dumps())

    @classmethod
    def load(cls, path: Path) -> PageDocument:
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise MangaArError(f"cannot read sidecar {path}: {exc}") from exc
        return cls.from_json(obj)


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, CropMask):
        return value.to_json()
    if isinstance(value, BBox):
        return value.to_list()
    if isinstance(value, Enum):
        return value.value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: _to_jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, set):
        return sorted(_to_jsonable(v) for v in value)
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, np.generic):
        return value.item()
    return value


def _opt(obj: dict[str, Any], key: str, fn: Any) -> Any:
    value = obj.get(key)
    return None if value is None else fn(value)


def _rgb(value: list[int]) -> Rgb:
    r, g, b = (int(v) for v in value)
    return (r, g, b)


def _region_from_json(obj: dict[str, Any]) -> Region:
    ocr = obj.get("ocr")
    tr = obj.get("translation")
    lay = obj.get("layout")
    return Region(
        id=obj["id"],
        type=RegionType(obj["type"]),
        bbox=BBox.from_list(obj["bbox"]),
        lines=[BBox.from_list(b) for b in obj.get("lines", [])],
        polygon=[(int(p[0]), int(p[1])) for p in obj.get("polygon", [])],
        bubble_mask=_opt(obj, "bubble_mask", CropMask.from_json),
        text_mask=_opt(obj, "text_mask", CropMask.from_json),
        inpaint_mask=_opt(obj, "inpaint_mask", CropMask.from_json),
        safe_box=_opt(obj, "safe_box", BBox.from_list),
        reading_order=int(obj.get("reading_order", 0)),
        source_lang=obj.get("source_lang"),
        vertical=bool(obj.get("vertical", False)),
        score=float(obj.get("score", 1.0)),
        fill_color=_opt(obj, "fill_color", _rgb),
        inpaint_method=obj.get("inpaint_method"),
        ocr=None
        if ocr is None
        else OcrResult(
            **{k: v for k, v in ocr.items() if k != "alternatives"},
            alternatives=[OcrCandidate(**a) for a in ocr.get("alternatives", [])],
        ),
        translation=None
        if tr is None
        else TranslationResult(
            **{k: v for k, v in tr.items() if k != "attempts"},
            attempts=[TranslationAttempt(**a) for a in tr.get("attempts", [])],
        ),
        layout=None
        if lay is None
        else LayoutResult(
            **{k: v for k, v in lay.items() if k not in {"box", "fill", "outline"}},
            box=BBox.from_list(lay["box"]),
            fill=_rgb(lay["fill"]),
            outline=_opt(lay, "outline", _rgb),
        ),
        flags={Flag(f) for f in obj.get("flags", [])},
        override=RegionOverride(**obj.get("override", {})),
    )
