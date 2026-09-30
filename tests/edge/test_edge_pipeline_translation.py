"""Edge cases E-01, E-05, E-07, E-12, E-14 on behaviour v0.1.0 already has.

These pin the current behaviour before Phase 1 changes it; the owner steps in
docs/EDGE_CASE_MATRIX.md extend them (policies, LLM output checks, series memory).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image

from manga_ar.config import load_config
from manga_ar.inpaint.strategy import RegionInpainter
from manga_ar.pipeline import Pipeline
from manga_ar.schemas import Flag, OcrResult, PageDocument, Region, RegionType
from tests.e2e.fakes import FakeOcr, FakeTranslator, fake_stages, page
from tests.unit.test_translate_service import FakeProvider, _region, _service

CFG = load_config(environ={})


class RecordingTranslator(FakeTranslator):
    def __init__(self) -> None:
        super().__init__()
        self.seen: list[str] = []

    def translate_regions(self, regions: Sequence[Region], src: str) -> None:
        self.seen += [r.id for r in regions]
        super().translate_regions(regions, src)


def _run(tmp: Path, img: np.ndarray, **stages: object) -> tuple[PageDocument, np.ndarray, int]:
    Image.fromarray(img).save(tmp / "in.png")
    report = Pipeline(CFG, fake_stages(CFG, **stages)).run([tmp / "in.png"], tmp / "out")
    doc = PageDocument.load(tmp / "out" / "in_ar.mangaar.json")
    final = np.asarray(Image.open(tmp / "out" / "in_ar.png").convert("RGB"))
    return doc, final, report.pages[0].regions


def test_e01_sfx_is_kept_original_not_translated_and_counted(tmp_path: Path) -> None:
    img = page()
    img[40:260, 370:390] = 0  # tall stylised block: SFX for the fake detector
    translator = RecordingTranslator()
    doc, final, counted = _run(tmp_path, img, translator=translator)
    sfx = [r for r in doc.regions if r.type == RegionType.SFX]
    assert len(sfx) == 1 and sfx[0].translation is None
    assert sfx[0].id not in translator.seen  # never enters the translator's context
    assert np.array_equal(final[40:260, 370:390], img[40:260, 370:390])  # not erased
    assert counted == len(doc.regions) == 3  # counted in the report
    assert not RegionInpainter.eligible(sfx[0])


def test_e05_punctuation_only_bubble_keeps_original_glyphs(tmp_path: Path) -> None:
    class PunctOcr(FakeOcr):
        def recognize(self, page: np.ndarray, region: Region, lang: str) -> OcrResult:
            if region.bbox.width == 40:
                return OcrResult(engine="fake", text="……?!", lang=lang)
            return super().recognize(page, region, lang)

    img = page()
    doc, final, _ = _run(tmp_path, img, ocr=PunctOcr())
    punct = next(r for r in doc.regions if r.bbox.width == 40)
    assert Flag.PASS_THROUGH in punct.flags and punct.translation is None
    assert np.array_equal(final[135:165, 70:110], img[135:165, 70:110])  # glyphs untouched
    other = next(r for r in doc.regions if r.bbox.width == 60)
    assert other.translation is not None and other.translation.text == "إلى اللقاء"


def test_e05_low_confidence_and_garbage_are_never_sent_for_translation() -> None:
    prov = FakeProvider("google")
    suspect = _region(0, "ののののののの")
    suspect.flag(Flag.OCR_SUSPECT)
    empty = _region(1, "")
    punct = _region(2, "…!?")
    punct.flag(Flag.PASS_THROUGH)
    _service(prov).translate_regions([suspect, empty, punct], "ja")
    assert prov.requests == []
    assert Flag.UNTRANSLATED in suspect.flags and Flag.UNTRANSLATED in empty.flags
    assert punct.translation is None and not RegionInpainter.eligible(punct)


def test_e07_sentence_spanning_bubbles_translated_in_context_output_per_bubble() -> None:
    prov = FakeProvider("google")
    first, second = _region(0, "待って"), _region(1, "行こう")  # one utterance, two bubbles
    _service(prov).translate_regions([second, first], "ja")
    assert len(prov.requests) == 1  # one page-level request carries both parts in order
    assert prov.requests[0].index("待って") < prov.requests[0].index("行こう")
    assert first.translation is not None and first.translation.text == "انتظر"
    assert second.translation is not None and second.translation.text == "لنذهب"


class _Padding(FakeProvider):
    def translate(self, text: str, src: str, tgt: str) -> str:
        self.requests.append(text)
        return "شكرا " * 60  # hallucinated additions: far longer than the source


def test_e12_wrong_script_and_hallucinated_length_are_never_accepted() -> None:
    regions = [_region(0, "ありがとう")]
    english = FakeProvider("google", mangle="english")
    padded = _Padding("mymemory")
    good = FakeProvider("libretranslate")
    _service(english, padded, good, **{"translate.batch": False}).translate_regions(regions, "ja")
    tr = regions[0].translation
    assert tr is not None and tr.provider == "libretranslate" and tr.text == "شكرًا"
    rejected = {a.provider: a.error for a in tr.attempts if not a.ok}
    assert set(rejected) == {"google", "mymemory"}
    only_bad = [_region(0, "ありがとう")]
    _service(FakeProvider("google", mangle="english")).translate_regions(only_bad, "ja")
    assert Flag.UNTRANSLATED in only_bad[0].flags and only_bad[0].arabic_text is None


def test_e14_glossary_locks_a_name_through_translation() -> None:
    from manga_ar.translate.glossary import Glossary

    gl = Glossary({"高橋さん": "تاكاهاشي-سان"})
    prov = FakeProvider("google", table={"ありがとう": "شكرًا", "高橋さん": "السيد طاكاهاشي"})
    regions = [_region(0, "高橋さん、ありがとう"), _region(1, "ありがとう、高橋さん")]
    _service(prov, glossary=gl).translate_regions(regions, "ja")
    for r in regions:
        assert r.translation is not None and "تاكاهاشي-سان" in r.translation.text
        assert "طاكاهاشي" not in r.translation.text  # the provider's variant never wins
