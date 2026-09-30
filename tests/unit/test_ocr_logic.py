"""OCR router behaviour with fake engines (no models)."""

from __future__ import annotations

import numpy as np
import pytest

from manga_ar.config import load_config
from manga_ar.errors import ModelUnavailableError, OcrError
from manga_ar.ocr.base import LineResult
from manga_ar.ocr.router import OcrRouter
from manga_ar.schemas import BBox, Flag, Region, RegionOverride, RegionType

CFG = load_config(environ={})


class FakeEngine:
    def __init__(
        self,
        name: str,
        langs: set[str],
        outputs: list[object],
        reads_blocks: bool = False,
        available: bool = True,
    ) -> None:
        self.name = name
        self.languages = frozenset(langs)
        self.reads_blocks = reads_blocks
        self._outputs = list(outputs)
        self._available = available
        self.calls = 0

    def available(self) -> bool:
        return self._available

    def _next(self) -> LineResult:
        self.calls += 1
        out = self._outputs.pop(0) if len(self._outputs) > 1 else self._outputs[0]
        if isinstance(out, Exception):
            raise out
        assert isinstance(out, LineResult)
        return out

    def recognize_line(self, line: np.ndarray, lang: str) -> LineResult:
        return self._next()

    def recognize_block(self, block: np.ndarray, lang: str, vertical: bool) -> LineResult:
        return self._next()


def _page_and_region(vertical: bool = False) -> tuple[np.ndarray, Region]:
    page = np.full((200, 300, 3), 255, np.uint8)
    page[80:110, 60:240] = 0
    region = Region(
        id="p-r0",
        type=RegionType.BUBBLE,
        bbox=BBox(60, 80, 240, 110),
        lines=[BBox(60, 80, 240, 110)],
        vertical=vertical,
    )
    return page, region


REAL = ["manga_ocr", "easyocr", "rapid", "paddle"]


def _router(**engines: FakeEngine) -> OcrRouter:
    """Register fakes under real engine names (the config validates engine names)."""
    named = {}
    for real, eng in zip(REAL, engines.values(), strict=False):
        eng.name = real
        named[real] = eng
    order = list(named)
    cfg = CFG.replace(**{"ocr.engines": {"ja": order, "ko": order, "zh": order}})
    return OcrRouter(named, cfg.ocr)


def test_first_good_engine_wins() -> None:
    a = FakeEngine("a", {"ja"}, [LineResult("今日はいい天気", 0.95)])
    b = FakeEngine("b", {"ja"}, [LineResult("別の結果", 0.99)])
    page, region = _page_and_region()
    res = _router(a=a, b=b).recognize(page, region, "ja")
    assert res.engine == a.name and res.text == "今日はいい天気" and b.calls == 0
    assert not region.flags


def test_fallback_on_unavailable_and_errors() -> None:
    broken = FakeEngine("broken", {"ja"}, [ModelUnavailableError("no weights")])
    flaky = FakeEngine("flaky", {"ja"}, [OcrError("boom")])
    good = FakeEngine("good", {"ja"}, [LineResult("ありがとう", 0.9)])
    router = _router(broken=broken, flaky=flaky, good=good)
    page, region = _page_and_region()
    assert router.recognize(page, region, "ja").engine == good.name
    # a model that failed to load is not retried for the rest of the run
    router.recognize(page, region, "ja")
    assert broken.calls == 1


def test_suspicious_result_retries_next_engine() -> None:
    loop = FakeEngine("loop", {"ja"}, [LineResult("のののののののの", 0.99)])
    good = FakeEngine("good", {"ja"}, [LineResult("待って", 0.8)])
    page, region = _page_and_region()
    res = _router(loop=loop, good=good).recognize(page, region, "ja")
    assert res.engine == good.name and res.alternatives[0].engine == loop.name
    assert Flag.OCR_SUSPECT not in region.flags


@pytest.mark.parametrize(
    ("text", "conf", "reason_flag"),
    [
        ("", 0.9, Flag.OCR_SUSPECT),
        ("Hello world", 0.9, Flag.OCR_SUSPECT),  # wrong script for Japanese
        ("ああああああああああ", 0.9, Flag.OCR_SUSPECT),  # repetition loop
        ("今日は", 0.1, Flag.LOW_CONFIDENCE),
    ],
)
def test_garbage_is_flagged(text: str, conf: float, reason_flag: Flag) -> None:
    eng = FakeEngine("only", {"ja"}, [LineResult(text, conf)])
    page, region = _page_and_region()
    _router(only=eng).recognize(page, region, "ja")
    assert reason_flag in region.flags


def test_all_engines_fail_flags_ocr_failed() -> None:
    eng = FakeEngine("x", {"ja"}, [OcrError("dead")])
    page, region = _page_and_region()
    res = _router(x=eng).recognize(page, region, "ja")
    assert res.text == "" and Flag.OCR_FAILED in region.flags
    none = _router(y=FakeEngine("y", {"ko"}, [LineResult("x", 1.0)]))  # no ja engine
    region2 = _page_and_region()[1]
    none.recognize(page, region2, "ja")
    assert Flag.OCR_FAILED in region2.flags


def test_verify_multi_engine_picks_best_score() -> None:
    a = FakeEngine("a", {"zh"}, [LineResult("今天天气", 0.55)])
    b = FakeEngine("b", {"zh"}, [LineResult("今天天气很好", 0.98)])
    a.name, b.name = "rapid", "easyocr"
    cfg = CFG.replace(
        **{"ocr.engines": {"zh": ["rapid", "easyocr"]}, "ocr.verify_multi_engine": True}
    )
    page, region = _page_and_region()
    res = OcrRouter({"rapid": a, "easyocr": b}, cfg.ocr).recognize(page, region, "zh")
    assert res.engine == "easyocr" and a.calls == 1 and b.calls == 1


def test_recognize_all_decorations_passthrough_and_overrides() -> None:
    eng = FakeEngine(
        "e",
        {"ja", "ko"},
        [LineResult("ありがとう♡", 0.9), LineResult("OK!", 0.9), LineResult("고마워", 0.9)],
    )
    router = _router(e=eng)
    page, r1 = _page_and_region()
    r2 = Region(id="p-r1", type=RegionType.BUBBLE, bbox=BBox(10, 10, 50, 40))
    r3 = Region(
        id="p-r2",
        type=RegionType.BUBBLE,
        bbox=BBox(10, 150, 90, 190),
        override=RegionOverride(source_lang="ko"),
    )
    sfx = Region(id="p-r3", type=RegionType.SFX, bbox=BBox(0, 0, 5, 5))
    router.recognize_all(page, [r1, r2, r3, sfx], "ja")
    assert r1.ocr is not None and r1.ocr.text == "ありがとう" and r1.ocr.decorations == ["♡"]
    assert Flag.PASS_THROUGH in r2.flags
    assert r3.source_lang == "ko" and r3.ocr is not None and r3.ocr.text == "고마워"
    assert sfx.ocr is None


def test_vertical_region_is_reflowed_for_line_engines() -> None:
    seen: list[tuple[int, int]] = []

    class Spy(FakeEngine):
        def recognize_line(self, line: np.ndarray, lang: str) -> LineResult:
            seen.append(line.shape[:2])
            return LineResult("行こう", 0.9)

    page = np.full((300, 200, 3), 255, np.uint8)
    for k in range(3):  # three glyph-like blobs stacked in one column
        page[40 + 60 * k : 80 + 60 * k, 80:120] = 0
    region = Region(
        id="v",
        type=RegionType.BUBBLE,
        bbox=BBox(80, 40, 120, 200),
        lines=[BBox(80, 40, 120, 200)],
        vertical=True,
    )
    _router(s=Spy("s", {"ja"}, [LineResult("x", 1.0)])).recognize(page, region, "ja")
    h, w = seen[0]
    assert w > h  # the column became a horizontal strip


def test_language_detection_with_fakes() -> None:
    ja = FakeEngine("ja_e", {"ja"}, [LineResult("ありがとう", 0.9)])
    ko = FakeEngine("ko_e", {"ko"}, [LineResult("???", 0.2)])
    zh = FakeEngine("zh_e", {"zh"}, [LineResult("谢谢", 0.5)])
    ja.name, ko.name, zh.name = "manga_ocr", "easyocr", "rapid"
    cfg = CFG.replace(**{"ocr.engines": {"ja": ["manga_ocr"], "ko": ["easyocr"], "zh": ["rapid"]}})
    router = OcrRouter({"manga_ocr": ja, "easyocr": ko, "rapid": zh}, cfg.ocr)
    page, region = _page_and_region()
    lang, scores = router.detect_language(page, [region])
    assert lang == "ja" and scores["ja"] > scores["zh"] > scores["ko"]
    assert router.detect_language(page, []) == (None, {})
