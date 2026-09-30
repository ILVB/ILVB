"""Translation service: failover, cache, batching integrity, glossary, validation, offline."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from manga_ar.config import load_config
from manga_ar.errors import ProviderError, RateLimitError
from manga_ar.schemas import BBox, Flag, OcrResult, Region, RegionOverride, RegionType
from manga_ar.translate.batching import chunk, decode, encode
from manga_ar.translate.cache import TranslationCache
from manga_ar.translate.glossary import Glossary
from manga_ar.translate.service import TranslationService, build_translation_service
from manga_ar.translate.tm import TranslationMemory, load_pairs
from tests.conftest import FakeClock

CFG = load_config(environ={})
DICT = {
    "ありがとう": "شكرًا",
    "待って": "انتظر",
    "行こう": "لنذهب",
    "大丈夫": "أنا بخير",
    "高橋さん": "السيد تاكاهاشي",
}


class FakeProvider:
    """Translates by dictionary lookup; supports index-tagged batches; scripted failures."""

    def __init__(
        self,
        name: str,
        network: bool = True,
        table: dict[str, str] | None = None,
        failures: list[Exception] | None = None,
        mangle: str | None = None,
        hang: threading.Event | None = None,
    ) -> None:
        self.name = name
        self.network = network
        self.table = DICT if table is None else table
        self.failures = list(failures or [])
        self.mangle = mangle
        self.hang = hang
        self.requests: list[str] = []

    def available(self) -> bool:
        return True

    def supports(self, src: str, tgt: str) -> bool:
        return tgt == "ar"

    def _one(self, text: str) -> str:
        out = text
        for k, v in sorted(self.table.items(), key=lambda kv: -len(kv[0])):
            out = out.replace(k, v)
        return out

    def translate(self, text: str, src: str, tgt: str) -> str:
        self.requests.append(text)
        if self.hang is not None:
            self.hang.wait(30)
        if self.failures:
            raise self.failures.pop(0)
        if self.mangle == "english":
            return "Thank you"
        if self.mangle == "cjk":
            return text
        if self.mangle == "drop-marker" and text.startswith("[1]"):
            return text.replace("[2]", "")
        if self.mangle == "drop-token":
            import re

            return re.sub(r"ZQX\d+X", "", self._one(text))
        segments = decode(text, text.count("[")) if text.startswith("[1]") else None
        if segments is not None:
            return encode([self._one(s) for s in segments])
        return self._one(text)


def _region(i: int, text: str, **kw: object) -> Region:
    return Region(
        id=f"r{i}",
        type=RegionType.BUBBLE,
        bbox=BBox(0, 0, 10, 10),
        reading_order=i,
        ocr=OcrResult(
            engine="fake",
            text=text,
            lang="ja",
            decorations=kw.pop(  # type: ignore[arg-type]
                "decorations", []
            ),
        ),
        **kw,
    )  # type: ignore[arg-type]


def _service(
    *providers: FakeProvider,
    cache: TranslationCache | None = None,
    glossary: Glossary | None = None,
    **cfg_over: object,
) -> TranslationService:
    cfg = CFG.replace(**cfg_over) if cfg_over else CFG
    return TranslationService(list(providers), cfg, cache, glossary, FakeClock())


def test_failover_order_and_batch_request() -> None:
    first = FakeProvider("google", failures=[ProviderError("down", retryable=False)])
    second = FakeProvider("mymemory")
    regions = [_region(0, "ありがとう"), _region(1, "待って"), _region(2, "行こう")]
    _service(first, second).translate_regions(regions, "ja")
    assert [r.translation.provider for r in regions] == ["mymemory"] * 3  # type: ignore[union-attr]
    assert [r.translation.text for r in regions] == ["شكرًا", "انتظر", "لنذهب"]  # type: ignore[union-attr]
    assert len(second.requests) == 1 and second.requests[0].startswith("[1]")  # one page batch
    attempts = regions[0].translation.attempts  # type: ignore[union-attr]
    assert attempts[0].provider == "google" and not attempts[0].ok and attempts[-1].ok


def test_tm_first_then_network_for_the_rest() -> None:
    tm = TranslationMemory({"ありがとう": "شكرًا جزيلًا"})
    net = FakeProvider("google")
    regions = [_region(0, "ありがとう"), _region(1, "待って")]
    TranslationService([tm, net], CFG, None, None, FakeClock()).translate_regions(regions, "ja")
    assert regions[0].translation.provider == "tm"  # type: ignore[union-attr]
    assert regions[1].translation.provider == "google"  # type: ignore[union-attr]
    assert net.requests == ["待って"]


def test_batch_count_mismatch_falls_back_to_per_region() -> None:
    prov = FakeProvider("google", mangle="drop-marker")
    regions = [_region(0, "ありがとう"), _region(1, "待って")]
    _service(prov).translate_regions(regions, "ja")
    assert all(r.translation and r.translation.provider == "google" for r in regions)
    assert prov.requests[0].startswith("[1]") and prov.requests[1:] == ["ありがとう", "待って"]


@pytest.mark.parametrize("mangle", ["english", "cjk"])
def test_wrong_script_rejected_then_next_provider(mangle: str) -> None:
    bad = FakeProvider("google", mangle=mangle)
    good = FakeProvider("mymemory")
    regions = [_region(0, "ありがとう")]
    _service(bad, good).translate_regions(regions, "ja")
    tr = regions[0].translation
    assert tr is not None and tr.provider == "mymemory"
    assert any(
        "arabic-ratio" in (a.error or "") or "identical" in (a.error or "") for a in tr.attempts
    )


def test_all_providers_fail_flags_untranslated_and_page_continues() -> None:
    regions = [_region(0, "ありがとう")]
    _service(FakeProvider("google", mangle="english")).translate_regions(regions, "ja")
    assert Flag.UNTRANSLATED in regions[0].flags and regions[0].arabic_text is None


def test_rate_limit_then_success_with_retry_after() -> None:
    prov = FakeProvider("google", failures=[RateLimitError("429", retry_after=5.0)])
    svc = _service(prov)
    regions = [_region(0, "待って")]
    svc.translate_regions(regions, "ja")
    assert regions[0].translation.provider == "google"  # type: ignore[union-attr]
    assert any(s >= 5.0 for s in svc.clock.sleeps)  # type: ignore[attr-defined]


def test_breaker_shifts_traffic_to_next_provider() -> None:
    flaky = FakeProvider("google", failures=[ProviderError("503")] * 20)
    backup = FakeProvider("mymemory")
    svc = _service(flaky, backup, **{"translate.batch": False})
    regions = [_region(i, t) for i, t in enumerate(["ありがとう", "待って", "行こう"])]
    svc.translate_regions(regions, "ja")
    assert all(r.translation.provider == "mymemory" for r in regions)  # type: ignore[union-attr]
    assert len(flaky.requests) == 5  # breaker opened; later regions skip google entirely


def test_hung_provider_is_time_bounded() -> None:
    release = threading.Event()
    hung = FakeProvider("google", hang=release)
    backup = FakeProvider("mymemory")
    svc = _service(
        hung,
        backup,
        **{
            "translate.total_timeout": 0.05,
            "translate.connect_timeout": 0.05,
            "translate.max_attempts": 2,
        },
    )
    regions = [_region(0, "待って")]
    try:
        svc.translate_regions(regions, "ja")
    finally:
        release.set()
    assert regions[0].translation.provider == "mymemory"  # type: ignore[union-attr]


def test_every_network_provider_goes_through_a_deadline() -> None:
    providers = [
        FakeProvider("google"),
        FakeProvider("mymemory"),
        FakeProvider("local", network=False),
    ]
    svc = _service(*providers)
    assert set(svc.callers) == {"google", "mymemory"}
    assert all(c.deadline == CFG.translate.total_timeout for c in svc.callers.values())


def test_cache_hit_miss_and_persistence(tmp_path: Path) -> None:
    path = tmp_path / "t.sqlite3"
    p1 = FakeProvider("google")
    _service(p1, cache=TranslationCache(path)).translate_regions([_region(0, "待って")], "ja")
    assert len(p1.requests) == 1
    p2 = FakeProvider("google")
    regions = [_region(0, "待って")]
    _service(p2, cache=TranslationCache(path)).translate_regions(regions, "ja")
    assert p2.requests == [] and regions[0].translation.from_cache  # type: ignore[union-attr]
    miss = [_region(0, "行こう")]
    _service(p2, cache=TranslationCache(path)).translate_regions(miss, "ja")
    assert p2.requests == ["行こう"]


def test_dedupe_same_text_translated_once() -> None:
    prov = FakeProvider("google", **{})
    regions = [_region(0, "待って"), _region(1, "待って")]
    _service(prov, **{"translate.batch": False}).translate_regions(regions, "ja")
    assert prov.requests == ["待って"] and regions[1].translation.text == "انتظر"  # type: ignore[union-attr]


def test_glossary_placeholders_survive_and_degrade() -> None:
    gl = Glossary({"高橋さん": "تاكاهاشي-سان"})
    ok = FakeProvider("google", table={**DICT, "高橋さん": "X"})
    regions = [_region(0, "高橋さん、ありがとう")]
    _service(ok, glossary=gl).translate_regions(regions, "ja")
    assert "تاكاهاشي-سان" in regions[0].translation.text  # type: ignore[union-attr]
    assert Flag.GLOSSARY_DEGRADED not in regions[0].flags
    lossy = FakeProvider("google", mangle="drop-token")
    regions = [_region(0, "高橋さん、ありがとう")]
    _service(lossy, glossary=gl).translate_regions(regions, "ja")
    assert Flag.GLOSSARY_DEGRADED in regions[0].flags
    assert regions[0].translation.text  # type: ignore[union-attr]


def test_chunking_respects_provider_limit() -> None:
    long_texts = [f"{'待って' * 400}{i}" for i in range(6)]  # ~1200 chars each, distinct
    for idxs in chunk(long_texts, 4500):
        assert len(encode([long_texts[i] for i in idxs])) <= 4500
    prov = FakeProvider("google")
    regions = [_region(i, t) for i, t in enumerate(long_texts)]
    _service(prov, **{"translate.max_length_ratio": 50.0}).translate_regions(regions, "ja")
    assert len(prov.requests) >= 2 and all(len(r) <= 4500 for r in prov.requests)


def test_offline_skips_network_and_suspect_and_skips() -> None:
    net = FakeProvider("google")
    local = FakeProvider("local", network=False)
    svc = _service(net, local, **{"runtime.offline": True})
    suspect = _region(1, "ののののの")
    suspect.flag(Flag.OCR_SUSPECT)
    skipped = _region(2, "待って", override=RegionOverride(skip=True))
    manual = _region(3, "待って", override=RegionOverride(text="نص المستخدم"))
    passthrough = _region(4, "OK")
    passthrough.flag(Flag.PASS_THROUGH)
    regions = [_region(0, "ありがとう"), suspect, skipped, manual, passthrough]
    svc.translate_regions(regions, "ja")
    assert net.requests == [] and regions[0].translation.provider == "local"  # type: ignore[union-attr]
    assert Flag.UNTRANSLATED in suspect.flags and suspect.translation is None
    assert skipped.translation is None and manual.arabic_text == "نص المستخدم"
    assert passthrough.translation is None


def test_decorations_reappended_and_normalised() -> None:
    prov = FakeProvider("google", table={"ありがとう": "شكرا?"})
    regions = [_region(0, "ありがとう", decorations=["♡"])]
    _service(prov).translate_regions(regions, "ja")
    assert regions[0].translation.text == "شكرا؟ ♡"  # type: ignore[union-attr]


def test_translate_text_and_builder(tmp_path: Path) -> None:
    tm_file = tmp_path / "tm.json"
    tm_file.write_text('{"待って": "انتظر!"}', encoding="utf-8")
    cfg = CFG.replace(
        **{"translate.tm_file": str(tm_file), "runtime.offline": True, "translate.cache": False}
    )
    from manga_ar.models.manager import ModelManager

    svc = build_translation_service(cfg, ModelManager(tmp_path, offline=True), tmp_path)
    res = svc.translate_text("待って", "ja")
    assert res is not None and res.provider == "tm" and res.text == "انتظر!"
    assert svc.translate_text("行こう", "ja") is None  # offline, nothing else available
    assert not svc.has_offline_provider("ja")


def test_batching_decode_variants() -> None:
    assert decode("[1] أ [2] ب", 2) == ["أ", "ب"]
    assert decode("［١］ أ 【٢】 ب", 2) == ["أ", "ب"]  # full-width brackets, Arabic digits
    assert decode("[1] أ [3] ب", 2) is None
    assert decode("مقدمة [1] أ [2] ب", 2) is None
    assert decode("[1] أ [2]", 2) is None


def test_tm_file_formats(tmp_path: Path) -> None:
    for name, content in (
        ("a.json", '{"待って": "انتظر"}'),
        ("a.yaml", "待って: انتظر\n"),
        ("a.csv", "# comment,x\n待って,انتظر\n"),
    ):
        f = tmp_path / name
        f.write_text(content, encoding="utf-8")
        assert load_pairs(f) == {"待って": "انتظر"}
    bad = tmp_path / "a.txt"
    bad.write_text("x", encoding="utf-8")
    from manga_ar.errors import ConfigError

    with pytest.raises(ConfigError):
        load_pairs(bad)
    assert TranslationMemory({"待って ": "x"}).lookup("待って") == "x"
