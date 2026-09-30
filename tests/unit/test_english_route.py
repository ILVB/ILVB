"""1.0.5: English/Latin OCR route (config, line joining, pass-through, script checks)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from manga_ar.config import load_config
from manga_ar.ocr.base import LineResult
from manga_ar.ocr.langid import script_consistency
from manga_ar.ocr.router import OcrRouter
from manga_ar.pipeline import Pipeline
from manga_ar.schemas import BBox, Flag, OcrResult, PageDocument, Region, RegionType
from tests.e2e.fakes import FakeOcr, fake_stages, page


class LatinEngine:
    name = "rapid"
    languages = frozenset({"en"})
    reads_blocks = False

    def __init__(self) -> None:
        self.answers = ["Wait", "for me"]

    def available(self) -> bool:
        return True

    def recognize_line(self, line: np.ndarray, lang: str) -> LineResult:
        return LineResult(self.answers.pop(0), 0.95)  # one answer per line, in order

    def recognize_block(self, block: np.ndarray, lang: str, vertical: bool) -> LineResult:
        return self.recognize_line(block, lang)


def test_english_is_a_configurable_source() -> None:
    cfg = load_config(overrides={"input.source_lang": "en"}, environ={})
    assert cfg.input.source_lang == "en" and cfg.ocr.engines["en"] == ["rapid", "easyocr"]
    assert script_consistency("Wait for me!", "en") == 1.0
    assert script_consistency("待って", "en") == 0.0


def test_english_lines_are_joined_with_spaces() -> None:
    cfg = load_config(overrides={"ocr.engines.en": ["rapid"]}, environ={})
    img = np.full((200, 300, 3), 255, np.uint8)
    region = Region(id="r", type=RegionType.BUBBLE, bbox=BBox(40, 40, 260, 140),
                    lines=[BBox(40, 40, 260, 70), BBox(40, 90, 260, 140)])  # fmt: skip
    res = OcrRouter({"rapid": LatinEngine()}, cfg.ocr).recognize(img, region, "en")  # type: ignore[dict-item]
    assert res.text == "Wait for me"


class LatinOcr(FakeOcr):
    def recognize(self, page: np.ndarray, region: Region, lang: str) -> OcrResult:
        return OcrResult(engine="fake", text="OK!", lang=lang)


def test_latin_is_content_on_english_pages_but_passed_through_elsewhere(tmp_path: Path) -> None:
    Image.fromarray(page()).save(tmp_path / "in.png")
    for lang, passed in (("en", False), ("zh", True)):
        cfg = load_config(overrides={"input.source_lang": lang}, environ={})
        out = tmp_path / lang
        Pipeline(cfg, fake_stages(cfg, ocr=LatinOcr())).run([tmp_path / "in.png"], out)
        doc = PageDocument.load(out / "in_ar.mangaar.json")
        assert all((Flag.PASS_THROUGH in r.flags) == passed for r in doc.regions), lang
